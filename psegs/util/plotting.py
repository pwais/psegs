# Copyright 2023 Maintainers of PSegs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np


PLOTLY_INIT_HTML = """
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src='https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js'></script>
    <script>requirejs.config({
        paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});
        if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>
    """


def color_to_opencv(color):
  r, g, b = np.clip(color, 0, 255).astype(int).tolist()
  return r, g, b


def contrasting_color(color):
  r, g, b = (np.array(color) / 255.).tolist()
  
  import colorsys
  h, s, v = colorsys.rgb_to_hsv(r, g, b)
  
  # Pick contrasting hue and lightness
  h = abs(1. - h)
  v = abs(1. - v)
  
  rgb = 255 * np.array(colorsys.hsv_to_rgb(h, s, v))  
  return tuple(rgb.astype(int).tolist())


def draw_bbox_in_image(np_image, bbox, color=None, label_txt='', thickness=2):
  """Draw a bounding box in `np_image`.

  Args:
    np_image (numpy.ndarray): Draw in this image.
    bbox: A (x1, y1, x2, y2) tuple or a bbox instance.
    color (tuple): An (r, g, b) tuple specifying the border color; by
        default use a category-determined color.
    label_txt (str): Override for the label text drawn for this box.  Prefer
        `bbox.category_name`, then this category string.  Omit label if 
        either is empty.
    thickness (int): thickness of the line in pixels.
        use the `category` attribute; omit label text if either is empty
  """
  import cv2
  from psegs.datum.bbox2d import BBox2D
  
  if not isinstance(bbox, BBox2D):
    bbox = BBox2D.from_x1_y1_x2_y2(*bbox)

  label_txt = label_txt or bbox.category_name
  if not color:
    from oarphpy.plotting import hash_to_rbg
    color = hash_to_rbg(label_txt)

  x1, y1, x2, y2 = bbox.get_x1_y1_x2_y2()

  ### Draw Box
  cv2.rectangle(
    np_image,
    (x1, y1),
    (x2, y2),
    color_to_opencv(color),
    thickness=thickness)

  ### Draw Text
  FONT_SCALE = 0.8
  FONT = cv2.FONT_HERSHEY_PLAIN
  PADDING = 2 # In pixels

  ret = cv2.getTextSize(label_txt, FONT, fontScale=FONT_SCALE, thickness=1)
  ((text_width, text_height), _) = ret

  # Draw the label above the bbox by default ...
  tx1, ty1 = bbox.x, bbox.y - PADDING

  # ... unless the text would draw off the edge of the image ...
  if ty1 - text_height - PADDING <= 0:
    ty1 += bbox.height + text_height + 2 * PADDING
  ty2 = ty1 - text_height - PADDING

  # ... and also shift left if necessary.
  if tx1 + text_width > np_image.shape[1]:
    tx1 -= (tx1 + text_width + PADDING - np_image.shape[1])
  tx2 = tx1 + text_width
  
  cv2.rectangle(
    np_image,
    (tx1, ty1 + PADDING),
    (tx2, ty2 - PADDING),
    color_to_opencv(color),
    cv2.FILLED)

  text_color = contrasting_color(color)
  cv2.putText(
    np_image,
    label_txt,
    (tx1, ty1),
    FONT,
    FONT_SCALE,
    color_to_opencv(text_color),
    1) # thickness


def draw_cuboid_xy_in_image(img, pts, base_color_rgb, alpha=0.3, thickness=2):
  """Draw a cuboid in the given image.  Color the front face lighter than
  the rest of the cuboid edges to indicate orientation.

  Args:
    img (np.array): Draw in this image.
    pts (np.array): An array of 8 by 2 representing pixel locatons (x, y)
      of the corners of the cuboid.  The first four coordinates are the front
      face and the last four are the rear face.  The front and back faces
      can wind in either CW or CCW order (but both must wind in the same order)
    base_color_rgb (tuple): An (r, g, b) sequence specifying the color of
      the cuboid, with components in [0, 255]
    alpha (float): Blend cuboid color into the image using weight [0, 1].
    thickness (int): line thickness of cuboid edges.
  """

  # Drawing is expensive! Skip if completely offscreen.
  h, w = img.shape[:2]
  idx = np.where(
    (pts[:, 0] >= 0) & (pts[:, 0] <= w) &
    (pts[:, 1] >= 0) & (pts[:, 1] <= h))
  if not idx[0].any():
    return

  base_color = np.array(base_color_rgb)
  front_color = color_to_opencv(base_color + 0.6 * 255)
  back_color = color_to_opencv(base_color - 0.6 * 255)
  center_color = color_to_opencv(base_color)

  import cv2
  # OpenCV can't draw transparent colors, so we use the 'overlay image' trick
  overlay = img.copy()

  def to_opencv_px(arr):
    return np.rint(arr).astype(int)

  front = to_opencv_px(pts[:4])
  assert front.shape == (4, 2), "OpenCV requires nx2, have %s" % (front,)
  cv2.polylines(
    overlay,
    [front],
    True, # is_closed
    front_color,
    thickness)

  back = to_opencv_px(pts[4:])
  assert back.shape == (4, 2), "OpenCV requires nx2, have %s" % (back,)
  cv2.polylines(
    overlay,
    [back],
    True, # is_closed
    back_color,
    thickness)
  
  for start, end in zip(front.tolist(), back.tolist()):
    cv2.line(overlay, tuple(start), tuple(end), center_color, thickness)

  # Add transparent fill
  CUBOID_FILL_ALPHA = max(0, alpha - 0.1)
  from scipy.spatial import ConvexHull
  hull = ConvexHull(pts)
  corners_uv = to_opencv_px(
    np.array([
      (pts[v, 0], pts[v, 1]) for v in hull.vertices]))
  coverlay = overlay.copy()
  cv2.fillPoly(coverlay, [corners_uv], center_color)
  overlay[:] = cv2.addWeighted(
    coverlay, CUBOID_FILL_ALPHA, overlay, 1 - CUBOID_FILL_ALPHA, 0)

  # Now blend with input image!
  img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def rgb_for_distance(d_meters, period_meters=10.):
  """Given a distance `d_meters` or an array of distances, return an
  `np.array([r, g, b])` color array for the given distance (or a 2D array
  of colors if the input is an array)).  We choose a distinct hue every
  `period_meters` and interpolate between hues for `d_meters`.
  """
  import colorsys
  from oarphpy.plotting import hash_to_rbg

  if not isinstance(d_meters, np.ndarray):
    d_meters = np.array([d_meters])
  
  SEED = 1337 # Colors for the first 10 buckets verified to be very distinct
  base_rgb = hash_to_rbg(SEED)
  base_h, base_s, base_v = colorsys.rgb_to_hsv(*base_rgb)
  
  # colorsys takes Hues in [0, 1] and the colors spaced ~0.5 apart are
  # complimentary.  We pick a value != 0.5 to create a coloring that is
  # out-of-phase with the HSV color wheel (ensures distinct colors across
  # depths)
  COLOR_STEP = 0.5 + 1. / 12
  max_bucket = int(np.ceil(d_meters.max() / period_meters))
  bucket_to_hsv = [
    (base_h + (bucket * COLOR_STEP % 1.0), base_s, base_v)
    for bucket in range(max_bucket + 2)
  ]
  bucket_to_rgb = [colorsys.hsv_to_rgb(*hsv) for hsv in bucket_to_hsv]
  bucket_to_color = np.array(bucket_to_rgb)

  # Use numpy's indexing for fast "table lookup" of bucket ids (bids) in
  # the "table" bucket_to_color
  bucket_below = np.floor(d_meters / period_meters)
  bucket_above = bucket_below + 1

  color_below = bucket_to_color[bucket_below.astype(int)]
  color_above = bucket_to_color[bucket_above.astype(int)]

  # For each distance, interpolate to *nearest* color based on L1 distance
  d_relative = d_meters / period_meters
  l1_dist_below = np.abs(d_relative - bucket_below)
  l1_dist_above = np.abs(d_relative - bucket_above)

  colors = (
    (1. - l1_dist_below) * color_below.T + 
    (1. - l1_dist_above) * color_above.T)

  colors = colors.T
  if len(d_meters) == 1:
    return colors[0]
  else:
    return colors


def draw_xy_depth_in_image(
      img,
      pts,
      marker_radius=-1,
      alpha=.4,
      period_meters=10.,
      user_colors=None):
  """Draw a point cloud `pts` in `img`; *modifies* `img` in-place (so you can 
  compose this draw call with others). Point color interpolates between
  standard colors for each `period_meters` tick.  Optionally override this
  behavior using `user_colors`.

  Args:
    img (np.array): Draw in this image.
    pts (np.array): An array of N by 3 points in form
      (pixel x, pixel y, depth meters).
    marker_radius (int): Draw a marker with this size (or a non-positive
      number to auto-choose based upon number of points).
    alpha (float): Blend point color using weight [0, 1].
    period_meters (float): Choose a distinct hue every `period_meters` and
      interpolate between hues.
    user_colors (np.array): Instead of coloring by distance, use this array
      of nx3 colors.
  """

  # OpenCV can't draw transparent colors, so we use the 'overlay image' trick:
  # First draw dots an an overlay...
  overlay = img.copy()
  h, w = overlay.shape[:2]

  if user_colors is not None:
    # Add color columns
    pts = np.hstack([pts, user_colors])
  else:
    pts = pts.copy()
  
  # Map points to pixels and filter off-screen points
  pts_xy = np.rint(pts[:, :2])
  pts[:, :2] = pts_xy[:, :2]
  pts = pts[np.where(
    (pts[:, 0] >= 0) & (pts[:, 0] < w) &
    (pts[:, 1] >= 0) & (pts[:, 1] < h) &
    (pts[:, 2] >= 0))]
  if not pts.any():
    return

  # Sort by distance descending; let nearer points draw over farther points
  pts = pts[-pts[:, 2].argsort()]
  
  if user_colors is not None:
    colors = pts[:, 3:]
  else:
    colors = rgb_for_distance(pts[:, 2], period_meters=period_meters)
  colors = np.clip(colors, 0, 255).astype(int)
  
  # Draw the markers! NB: numpy assignment is very fast, even for 1M+ pts
  yy = pts[:, 1].astype(np.int)
  xx = pts[:, 0].astype(np.int)
  overlay[yy, xx] = colors

  if marker_radius < 0:
    # Draw larger markers for fewer points (or user_colors) to make points
    # more conspicuous
    if user_colors is not None:
      marker_radius = 3
    elif pts.shape[0] <= 1e5:
      marker_radius = 2

  if marker_radius >= 1:
    # Draw a crosshairs marker
    for r in range(-marker_radius, marker_radius + 1):
      overlay[(yy + r) % h, xx] = colors
      overlay[yy, (xx + r) % w] = colors
        # NB: toroidal boundary conditions plot hack for speed ...

  # Now blend!
  import cv2
  img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw_depth_in_image(
      img,
      depth_channel,
      alpha=.4,
      period_meters=10.):
  """Draw a `depth_channel` in `img`; *modifies* `img` in-place (so you can 
  compose this draw call with others). Point color interpolates between
  standard colors for each `period_meters` tick.  Optionally override this
  behavior using `user_colors`.

  Args:
    img (np.array): Draw in this image.
    depth_channel (np.array): A depth channel of shape (h, w, 1) [or
      just (h, w)] that matches the size of `img`.  Each value is a depth value
      in meters.  Invalid values are ignored (drawn with 0 alpha) in the output.
    alpha (float): Blend point color using weight [0, 1].
    period_meters (float): Choose a distinct hue every `period_meters` and
      interpolate between hues.
  """

  # OpenCV can't draw transparent colors, so we use the 'overlay image' trick:
  # First draw dots an an overlay...
  overlay = img.copy()
  h, w = overlay.shape[:2]

  depth_channel = depth_channel.squeeze().copy()
  assert depth_channel.shape[:2] == (h, w)

  valid = np.where(
            (depth_channel > 0) & np.isfinite(depth_channel))
  if not valid[0].any():
    return
  
  color_d = np.zeros_like(depth_channel)
  color_d[valid] = depth_channel[valid]
  color_d = np.reshape(color_d, [-1])
  colors = rgb_for_distance(color_d, period_meters=period_meters)
  colors = np.clip(colors, 0, 255).astype(int)
  colors = np.reshape(colors, [h, w, 3])

  # Retain original color for invalid points
  overlay[valid] = colors[valid]

  # Now blend!
  import cv2
  img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def get_ortho_debug_image(
      uvd,
      min_u=0.,  min_v=0.,
      max_u=10., max_v=10.,
      pixels_per_meter=100,
      marker_radius=-1,
      period_meters=10.,
      user_colors=None):
  """Create and return an orthographic debug image for the given cloud of
  `(u, v, d)` points (in meters) rasterized at `pixels_per_meters`.
  Useful for visualizing a raw point cloud (or a half-space of one) as a
  2D image. The parameters (min_u, minv) and (max_u, max_v) define the
  bounding box of points to plot; provide `None` values to auto-fit to
  `uvd` extents.

  Args:
    uvd (np.array): An nx3 array of points (in units of meters) where
      the first axis (u) is the +u (left-to-right) axis of the debug image,
      the second axis (v) is the +v (bottom-to-top) axis of the dbeug image,
      and the third axis (d) is the depth of the point in the half-space
      (and determines color).
    min_u (float): The left image boundary (in meters).
    min_v (float): The bottom image boundary (in meters).
    max_u (float): The right image boundary (in meters).
    max_v (float): The top image boundary (in meters).
    pixels_per_meter (int): Rasterize points at this resolution.
    recenter_points (bool): Re-center the given points to be the center
      of the debug image.
    marker_radius (int): Draw a marker with this size (in pixels).
    period_meters (float) : Choose a distinct hue every `period_meters` and
      interpolate between hues.
    user_colors (np.array): (Optional) instead of coloring by distance, use
      this array of nx3 colors.
  Returns:
    np.array: A HWC RGB debug image.
  """

  if not uvd.any():
    if (min_u, min_v, max_u, max_v) != (None, None, None, None):
      w = int(pixels_per_meter * (max_u - min_u) + 1)
      h = int(pixels_per_meter * (max_v - min_v) + 1)
      return np.zeros((h, w, 3), dtype=np.uint8)
    else:
      return np.zeros((0, 0, 3), dtype=np.uint8)
  
  if min_u is None:
    min_u = uvd[:, 0].min()
  if max_u is None:
    max_u = uvd[:, 0].max()
  if min_v is None:
    min_v = uvd[:, 1].min()
  if max_v is None:
    max_v = uvd[:, 1].max()

  assert min_u <= max_u
  assert min_v <= max_v

  # Move points to the viewport frame
  uvd = uvd - np.array([min_u, min_v, 0])
  
  # (u, v) meters -> pixels
  uvd[:, (0, 1)] *= pixels_per_meter
  
  w = int(pixels_per_meter * (max_u - min_u) + 1)
  h = int(pixels_per_meter * (max_v - min_v) + 1)
  img = np.zeros((h, w, 3), dtype=np.uint8)
  
  draw_xy_depth_in_image(
    img,
    uvd,
    marker_radius=marker_radius,
    period_meters=period_meters,
    alpha=1.0,
    user_colors=user_colors)
  
  # image vertical axis is flipped
  img = np.flipud(img)

  return img


def images_to_html_video(images=[], fps=4, play_on_hover=True):
  """Given a sequence of HWC numpy images, create and return an HTML video
  
  Args:
    images (List of np.ndarray): A sequence of HWC images (prefer RGB)
    fps (float): Render video to this frames per second
    play_on_hover (bool): Make the video autoplay on hover
  
  Returns:
    str: An HTML string with the included video
  """
  
  import base64
  import tempfile
  import urllib.parse
  
  import imageio

  if not images:
    return "<i>(No images for video)</i>"

  h, w = images[0].shape[:2]

  # We tried BytesIO but imageio seems to struggle with that and ffmpeg
  buffer = tempfile.NamedTemporaryFile(suffix=".psegs_html_video.mp4")
  imageio.mimsave(
          buffer.name,
          images,
          fps=fps)

  video_bytes = open(buffer.name, 'rb').read()
  encoded = base64.b64encode(video_bytes)
  html_data = urllib.parse.quote(encoded.decode('ascii'))

  attrs = ''
  if play_on_hover:
    attrs = """
        onmouseover="this.play()" onmouseout="this.pause();this.currentTime=0;"
      """.strip()
  
  html = """
    <video width="{w}", height="{h}" controls {attrs}>
    <source type="video/mp4" src="data:video/mp4;base64,{html_data}" />
    </video>
  """.format(
        h=h, w=w,
        attrs=attrs,
        html_data=html_data)
  return html


def sample_to_html(
        spark,
        sample,
        include_videos=True,
        videos_n_frames=50,
        video_fps=4,
        rgbd_depth_to_clouds=True,
        include_cloud_viz=True,
        clouds_n_clouds=50,
        clouds_n_pts_per_plot=50000,
        cloud_include_cam_poses=True):
  
  from psegs import datum
  from psegs import table
  from psegs import util

  # Ensure we have a Spark Dataframe
  if isinstance(sample, datum.Sample):
    sdt = table.StampedDatumTable.from_sample(sample)
    sd_df = sdt.to_spark_df(spark=spark)
  elif isinstance(sample, table.StampedDatumTable):
    sd_df = sample.to_spark_df(spark=spark)
  elif hasattr(sample, '_jrdd'):
    sdt = table.StampedDatumTable.from_datum_rdd(sample, spark=spark)
    sd_df = sdt.to_spark_df(spark=spark)
  elif hasattr(sample, 'rdd'):
    # Probably is already a Spark Dataframe
    sd_df = sample
  else:
    raise ValueError("Don't know what to do with %s" % (sample,))

  
  sd_df = sd_df.repartition('uri.timestamp')
  sd_df = sd_df.persist()
  sd_df.createOrReplaceTempView('sd_df')
  reports = []


  util.log.info("Rendering summaries for %s datums ..." % sd_df.count())

  def _get_table_html(sql):
    res = spark.sql(sql)
    pdf = res.toPandas()
    util.log.info('\n%s\n' % str(pdf))
    pdf.style.set_precision(2)
    pdf.style.hide_index()
    css = """
        <style type="text/css" >
          table {
            border: none;
            border-collapse: collapse;
            border-spacing: 0;
            color: black;
            font-size: 14px;
            font-family: "Monaco", monospace;
            table-layout: fixed;
          }
          thead {
            border-bottom: 1px solid black;
            vertical-align: bottom;
          }
          tr, th, td {
            text-align: right;
            vertical-align: middle;
            padding: 0.5em 0.5em;
            line-height: normal;
            white-space: normal;
            max-width: none;
            border: none;
          }
          th {
            font-weight: bold;
          }
          tbody tr:nth-child(odd) {
            background: #f5f5f5;
          }
          tbody tr:hover {
            background: rgba(66, 165, 245, 0.2);
          }
        </style>
    """
    return css + pdf.style.render() # Pandas succint "to HTML" with nice style

  ## Summary ###############################################################
  html = _get_table_html("""
          SELECT 
            segment_id,
            dataset,
            split,
            n AS total_datums,
            1e-9 * duration_ns AS duration_sec,
            FROM_UNIXTIME(start * 1e-9) AS start,
            FROM_UNIXTIME(end * 1e-9) AS end
          FROM 
              (
                  SELECT
                      FIRST(uri.dataset) AS dataset,
                      FIRST(uri.split) AS split,
                      FIRST(uri.segment_id) AS segment_id,
                      MIN(uri.timestamp) AS start,
                      MAX(uri.timestamp) AS end,
                      MAX(uri.timestamp) - MIN(uri.timestamp) AS duration_ns,
                      COUNT(*) AS n
                  FROM sd_df
                  GROUP BY (uri.dataset, uri.split, uri.segment_id)
              )
      """)
  reports.append({'section': 'Sample', 'html': html})


  ## CameraImages ##########################################################
  html = _get_table_html("""
          SELECT 
            topic,
            n,
            (n / (1e-9 * duration_ns)) AS Hz,
            1e-9 * duration_ns AS duration_sec,
            width,
            height,
            channel_names,
            uncompressed_MBytes,
            uncompressed_MBytes / (1e-9 * duration_ns) AS uncompressed_MBps,
            FROM_UNIXTIME(start * 1e-9) AS start,
            FROM_UNIXTIME(end * 1e-9) AS end

          FROM 
              (
                  SELECT
                      FIRST(uri.topic) AS topic,
                      MIN(uri.timestamp) AS start,
                      MAX(uri.timestamp) AS end,
                      MAX(uri.timestamp) - MIN(uri.timestamp) AS duration_ns,
                      FIRST(camera_image.width) AS width,
                      FIRST(camera_image.height) AS height,
                      FIRST(camera_image.channel_names) AS channel_names,
                      1e-6 * FIRST(camera_image.width * camera_image.height * 3) AS uncompressed_MBytes,
                      COUNT(*) AS n
                  FROM sd_df
                  WHERE camera_image IS NOT NULL
                  GROUP BY uri.topic
              )
          ORDER BY topic
      """)
  reports.append({'section': 'CameraImages', 'html': html})


  ## PointClouds ###########################################################
  html = _get_table_html("""
          SELECT 
            topic,
            n,
            (n / (1e-9 * duration_ns)) AS Hz,
            1e-9 * duration_ns AS duration_sec,
            cloud_colnames,
            FROM_UNIXTIME(start * 1e-9) AS start,
            FROM_UNIXTIME(end * 1e-9) AS end

          FROM 
              (
                  SELECT
                      FIRST(uri.topic) AS topic,
                      MIN(uri.timestamp) AS start,
                      MAX(uri.timestamp) AS end,
                      MAX(uri.timestamp) - MIN(uri.timestamp) AS duration_ns,
                      FIRST(point_cloud.cloud_colnames) AS cloud_colnames,
                      COUNT(*) AS n
                  FROM sd_df
                  WHERE point_cloud IS NOT NULL
                  GROUP BY uri.topic
              )
          ORDER BY topic
      """)
  reports.append({'section': 'PointClouds', 'html': html})
  

  ## Transforms ############################################################
  html = _get_table_html("""
          SELECT 
            topic,
            n,
            (n / (1e-9 * duration_ns)) AS Hz,
            1e-9 * duration_ns AS duration_sec,
            xform,
            FROM_UNIXTIME(start * 1e-9) AS start,
            FROM_UNIXTIME(end * 1e-9) AS end

          FROM 
              (
                  SELECT
                      FIRST(uri.topic) AS topic,
                      MIN(uri.timestamp) AS start,
                      MAX(uri.timestamp) AS end,
                      MAX(uri.timestamp) - MIN(uri.timestamp) AS duration_ns,
                      FIRST(CONCAT(transform.src_frame, '->', transform.dest_frame)) AS xform,
                      COUNT(*) AS n
                  FROM sd_df
                  WHERE transform IS NOT NULL
                  GROUP BY uri.topic
              )
          ORDER BY topic
      """)
  reports.append({'section': 'Transforms', 'html': html})
  

  ## Cuboids ###############################################################
  html = _get_table_html("""
          SELECT 
            topic,
            n,
            (n / (1e-9 * duration_ns)) AS Hz,
            1e-9 * duration_ns AS duration_sec,
            FROM_UNIXTIME(start * 1e-9) AS start,
            FROM_UNIXTIME(end * 1e-9) AS end

          FROM 
              (
                  SELECT
                      FIRST(uri.topic) AS topic,
                      MIN(uri.timestamp) AS start,
                      MAX(uri.timestamp) AS end,
                      MAX(uri.timestamp) - MIN(uri.timestamp) AS duration_ns,
                      COUNT(*) AS n
                  FROM sd_df
                  WHERE SIZE(cuboids) > 0
                  GROUP BY uri.topic
              )
          ORDER BY topic
      """)
  reports.append({'section': 'Cuboids', 'html': html})

  # Find depth topics for later
  rows = spark.sql("""
              SELECT
                FIRST(uri.topic) AS topic,
                FIRST(camera_image.channel_names) AS channel_names
              FROM sd_df
              WHERE camera_image IS NOT NULL
              GROUP BY uri.topic
            """).collect()
  depth_camera_topics = set([
    r.topic
    for r in rows
    if 'depth' in r.channel_names
  ])

  ## Videos ################################################################
  if include_videos:
    topic_htmls = []

    camera_topics = spark.sql(
      "SELECT DISTINCT uri.topic FROM sd_df WHERE camera_image IS NOT NULL")
    camera_topics = sorted(r.topic for r in camera_topics.collect())
    for topic in camera_topics:
      util.log.info("... rendering video for %s ..." % topic)
      sample_sd_df = spark.sql("""
                        SELECT *
                        FROM sd_df
                        WHERE uri.topic == '{topic}'
                        ORDER BY RAND(1337)
                        LIMIT {videos_n_frames}
                    """.format(topic=topic, videos_n_frames=videos_n_frames))
      sample_sd_df = sample_sd_df.repartition('uri.timestamp')

      ##################################################
      ## We want to adapt period_meters for depth topics
      period_meters = 10.
      if topic in depth_camera_topics:
        def _get_depth_90th(row):
          ci = table.StampedDatumTable.sd_from_row(row.camera_image)
          depth = ci.get_depth()
          if depth is None:
              return 0
          else:
              return np.percentile(depth[depth > 0], 0.9)
        depth_top_90th = sample_sd_df.rdd.map(_get_depth_90th).max()
        if depth_top_90th <= 0.1:
            period_meters = 0.005
        elif depth_top_90th <= 1.0:
            period_meters = 0.05
        elif depth_top_90th <= 10.0:
            period_meters = 0.5
        else:
            period_meters = 10.
      
      ##################################################
      ## Now render video
      def _to_t_debug_image(row):
        import cv2
        ci = table.StampedDatumTable.sd_from_row(row.camera_image)
        image = ci.get_debug_image(period_meters=period_meters)
        aspect = float(ci.width) / float(ci.height)
        target_height = 400
        target_width = int(aspect * target_height)
        
        # Pad the width a little to make ffmpeg most efficient
        # (and complain less)
        if target_width % 16 != 0:
            target_width += 16 - (target_width % 16)
        image = cv2.resize(image, (target_width, target_height))

        return row.uri.timestamp, image

      if sample_sd_df.rdd.isEmpty():
        html = "<i>No images to viz</i>"
      else:
        iter_t_image = sample_sd_df.rdd.map(_to_t_debug_image).collect()
        images = [
          image
          for t, image in sorted(iter_t_image, key=lambda ti: ti[0])
        ]

        # Global re-scale for depth debug coloring
        if topic in depth_camera_topics:
          im_min = min(i.min() for i in images)
          im_max = min(i.max() for i in images)
          images = [
            (255 * 
              (i.astype('float') - im_min) / (im_max - im_min)).astype('uint8')
            for i in images
          ]
        html = images_to_html_video(images, fps=video_fps)
      if topic in depth_camera_topics:
        footer = """
          <br/><i>Depth coloring with {period_meters}-meter hue period</i><br/>
          """.format(period_meters=period_meters)
        html = html + footer
      topic_htmls.append((topic, html))
    
    def _to_section_html(info):
      topic, vhtml = info
      html = """
        <h3>{topic}</h3><br/><br/>
        {vhtml}
        <br/><br/>
      """.format(topic=topic, vhtml=vhtml)
      return html

    section_html = ''.join(_to_section_html(i) for i in topic_htmls)
    reports.append({'section': 'Videos', 'html': section_html})

  
  ## Clouds ################################################################
  need_plotly_init = False
  if include_cloud_viz:
    import pandas as pd
    import plotly.graph_objects as go
    
    need_plotly_init = True

    topic_htmls = []

    pc_topics = spark.sql(
      "SELECT DISTINCT uri.topic FROM sd_df WHERE point_cloud IS NOT NULL")
    pc_topics = sorted(r.topic for r in pc_topics.collect())
    if rgbd_depth_to_clouds:
      pc_topics += sorted(depth_camera_topics)
    
    for topic in pc_topics:
      util.log.info("... rendering point cloud viz for %s ..." % topic)

      ##################################################
      ## Get the clouds to viz
      orig_sample_sd_df = spark.sql("""
                                SELECT *
                                FROM sd_df
                                WHERE uri.topic == '{topic}'
                                ORDER BY RAND(1337)
                                LIMIT {clouds_n_clouds}
                            """.format(
                              topic=topic, clouds_n_clouds=clouds_n_clouds))
      orig_sample_sd_df = orig_sample_sd_df.repartition('uri.timestamp')

      if topic in depth_camera_topics:
        def _make_pc(row):
          sd = table.StampedDatumTable.sd_from_row(row)
          pc = sd.camera_image.depth_image_to_point_cloud()
          sd.camera_image = None
          sd.point_cloud = pc
          return sd
        sd_rdd = orig_sample_sd_df.rdd.map(_make_pc)
        sdt = table.StampedDatumTable.from_datum_rdd(sd_rdd, spark=spark)
        sample_sd_df = sdt.to_spark_df(spark=spark)
        sample_sd_df = sample_sd_df.persist()
      else:
        sample_sd_df = orig_sample_sd_df
      
      def _get_t_cloud_world(row):
        pc = table.StampedDatumTable.sd_from_row(row.point_cloud)
        cloud = pc.get_cloud()
        cloud = cloud[:, :3]
        T_world_from_ego = pc.ego_pose['ego', 'world']
        cloud_world = T_world_from_ego.apply(cloud).T
        return row.uri.timestamp, cloud_world
      
      ##################################################
      ## Package clouds for plotly
      total_n_world = 0
      cloud_t_worlds = sample_sd_df.rdd.map(_get_t_cloud_world).collect()
      cloud_worlds = [
        c for t, c in sorted(cloud_t_worlds, key=lambda tc: tc[0])
      ]
      cloud_df = None
      pts_per_cloud = int(clouds_n_pts_per_plot / len(cloud_worlds))
      for i, cloud in enumerate(cloud_worlds):
        total_n_world  += cloud.shape[0]
        color = int(255 * (float(i) / len(cloud_worlds)))
        if cloud.shape[0] > pts_per_cloud:
          idx = np.random.choice(
                  np.arange(cloud.shape[0]), pts_per_cloud)
          cloud = cloud[idx, :]
        cur_df = pd.DataFrame(cloud, columns=['x', 'y', 'z'])
        cur_df['color'] = 'rgb(%s, %s, %s)' % (color, color, color)
        if cloud_df is None:
          cloud_df = cur_df
        else:
          cloud_df = pd.concat([cloud_df, cur_df])
      
      ##################################################
      ## Create plots
      plots = []
      scatter = go.Scatter3d(
                x=cloud_df['x'], y=cloud_df['y'], z=cloud_df['z'],
                mode='markers',
                marker=dict(size=2, color=cloud_df['color'], opacity=0.5),)
      plots.append(scatter)

      ##################################################
      ## Add pose plots if needed
      if cloud_include_cam_poses:
        util.log.info("... adding camera poses for %s ..." % topic)
        if topic in depth_camera_topics:
          ci_sd_df = orig_sample_sd_df
        else:
          ci_sd_df = spark.sql("""
                          SELECT *
                          FROM sd_df
                          WHERE camera_image IS NOT NULL
                          ORDER BY RAND(1337)
                          LIMIT {limit}
                      """.format(limit=clouds_n_clouds))
        def _get_t_ci(row):
          ci = table.StampedDatumTable.sd_from_row(row.camera_image)
          return row.uri.timestamp, ci
        t_cis = ci_sd_df.rdd.map(_get_t_ci).collect()
        cis = [ci for t, ci in sorted(t_cis, key=lambda tc: tc[0])]

        plots += [ci.to_plotly_world_frame_3d() for ci in cis]

      ##################################################
      ## Render to HTML
      fig = go.Figure(data=plots)
      fig.update_layout(
        width=1000, height=700,
        scene_aspectmode='data')
      footer = """
            <i>Showing {sample} of {total} points from {n} clouds</i>
            """.format(
                  sample=len(cloud_df),
                  total=total_n_world,
                  n=clouds_n_clouds)
      html = (
        fig.to_html(include_plotlyjs=True, full_html=False) + '<br/><br/>' + 
        footer)     
      topic_htmls.append((topic, html))
    
    def _to_section_html(info):
      topic, pchtml = info
      html = """
        <h3>{topic}</h3><br/><br/>
        {pchtml}
        <br/><br/>
      """.format(topic=topic, pchtml=pchtml)
      return html

    section_html = ''.join(_to_section_html(i) for i in topic_htmls)
    reports.append({'section': 'Point Clouds', 'html': section_html})

      
  ## Generate full report!
  util.log.info(
    "... have reports %s ..." % ([i['section'] for i in reports],))

  def report_info_to_html(info):
    section = info['section']
    content = info['html']

    html = """
      <hr/>
      <h2>{section}</h2><br/><br/>

      {content}

      <br/><br/>
    """.format(section=section, content=content)
    return html

  full_html = "".join(report_info_to_html(info) for info in reports)
  if need_plotly_init:
    full_html = PLOTLY_INIT_HTML + '<br />' + full_html

  util.log.info(
    "... completed report is %.2f MBytes ..." % (1e-6 * len(full_html)))
  return full_html


if __name__ == '__main__':
  import sys
  sys.path.append('/opt/psegs')

  import os
  lidar_root = '/outer_root/media/970-evo-plus-raid0/lidarphone_lidar_scans/'
  # scan_dirs = [
  #   os.path.join(lidar_root, f)
  #   for f in os.listdir(lidar_root)
  #   if os.path.isdir(os.path.join(lidar_root, f))
  # ]
  # scan_dirs = sorted(scan_dirs)

  from psegs.datasets import ios_lidar
  from psegs.spark import Spark

  class F(ios_lidar.Fixtures):
    @classmethod
    def threeDScannerApp_data_root(cls):
      return lidar_root

  T = ios_lidar.IOSLidarSDTable
  T.FIXTURES = F

  seg_uris = T.get_all_segment_uris()
  import pprint
  # pprint.pprint(seg_uris)
  with Spark.sess() as spark:
    for suri in seg_uris:

      if suri.segment_id == 'Untitled Scan':
        print('fixme', suri)
        continue
      if suri.segment_id in ('amiot-crow-bar', 'headlands-downhill-2'):
        print("fixme", 'amiot-crow-bar')
        continue


      pprint.pprint('working on')
      pprint.pprint(suri)
      sd_df = T.as_df(spark, force_compute=True, only_segments=[suri])
      if not sd_df:
        print('no df!', suri)
        continue
    
      outpath = os.path.join(lidar_root, suri.segment_id + '.html')

      # import plotly.graph_objects as go
      # import pandas as pd
      # cloud_df = pd.DataFrame(np.ones((100, 3)), columns=['x', 'y', 'z'])
      # color = 128
      # cloud_df['color'] = 'rgb(%s, %s, %s)' % (color, color, color)
      
      # plots = []
      # scatter = go.Scatter3d(
      #           x=cloud_df['x'], y=cloud_df['y'], z=cloud_df['z'],
      #           mode='markers',
      #           marker=dict(size=2, color=cloud_df['color'], opacity=0.5),)
      # plots.append(scatter)

      # fig = go.Figure(data=plots)
      # fig.update_layout(
      #   width=1000, height=700,
      #   scene_aspectmode='data')
      # footer = """
      #       <i>asdgasgs</i>
      #       """
      # html = (
      #   fig.to_html(include_plotlyjs=True, full_html=False) + '<br/><br/>' + 
      #   footer)     



      html = sample_to_html(spark, sd_df)
      with open(outpath, 'w') as f:
        f.write(html)

      print('saved', outpath)
      print(suri)
