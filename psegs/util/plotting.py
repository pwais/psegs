# Copyright 2020 Maintainers of PSegs
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

  base_color = np.array(base_color_rgb)
  front_color = color_to_opencv(base_color + 0.3 * 255)
  back_color = color_to_opencv(base_color - 0.3 * 255)
  center_color = color_to_opencv(base_color)

  import cv2
  # OpenCV can't draw transparent colors, so we use the 'overlay image' trick
  overlay = img.copy()

  def to_opencv_px(arr):
    return np.rint(arr).astype(int)

  front = to_opencv_px(pts[:4])
  cv2.polylines(
    overlay,
    [front],
    True, # is_closed
    front_color,
    thickness)

  back = to_opencv_px(pts[4:])
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
  from oarphpy.plotting import hash_to_rbg

  if not isinstance(d_meters, np.ndarray):
    d_meters = np.array([d_meters])
  
  SEED = 1337 # Colors for the first 10 buckets verified to be very distinct
  max_bucket = int(np.ceil(d_meters.max() / period_meters))
  bucket_to_color = np.array(
    [hash_to_rbg(bucket + SEED) for bucket in range(max_bucket + 2)])

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


def draw_xy_depth_in_image(img, pts, marker_radius=-1, alpha=.4):
  """Draw a point cloud `pts` in `img`; *modifies* `img` in-place (so you can 
  compose this draw call with others). Point color interpolates between
  standard colors for each 10-meter tick.

  Args:
    img (np.array): Draw in this image.
    pts (np.array): An array of N by 3 points in form
      (pixel x, pixel y, depth meters).
    marker_radius (int): Draw a marker with this size (or a non-positive
      number to auto-choose based upon number of points).
    alpha (float): Blend point color using weight [0, 1].
  """

  # OpenCV can't draw transparent colors, so we use the 'overlay image' trick:
  # First draw dots an an overlay...
  overlay = img.copy()
  h, w = overlay.shape[:2]

  pts = pts.copy()
  
  # Map points to pixels and filter off-screen points
  pts_xy = np.rint(pts[:, :2])
  pts[:, :2] = pts_xy[:, :2]
  pts = pts[np.where(
    (pts[:, 0] >= 0) & (pts[:, 0] < w) &
    (pts[:, 1] >= 0) & (pts[:, 1] < h))]

  # Sort by distance descending; let nearer points draw over farther points
  pts = pts[-pts[:, -1].argsort()]
  if not pts.any():
    return

  colors = rgb_for_distance(pts[:, 2])
  colors = np.clip(colors, 0, 255).astype(int)
  
  # Draw the markers! NB: numpy assignment is very fast, even for 1M+ pts
  yy = pts[:, 1].astype(np.int)
  xx = pts[:, 0].astype(np.int)
  overlay[yy, xx] = colors

  if marker_radius < 0:
    # Draw larger markers for fewer points to make them conspicuous
    if pts.shape[0] >= 1e5:
      marker_radius = 2

  if marker_radius >= 1:
    # Draw a crosshairs marker
    for r in range(-marker_radius, marker_radius + 1):
      overlay[(yy + r) % h, xx] = colors
      overlay[yy, (xx + r) % w] = colors

  # Now blend!
  import cv2
  img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

# def get_2d
