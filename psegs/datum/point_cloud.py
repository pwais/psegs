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

import typing

import attr
import numpy as np

from psegs.datum.transform import Transform
from psegs.util import misc
from psegs.util import plotting as pspl


@attr.s(slots=True, eq=False, weakref_slot=False)
class PointCloud(object):
  """A cloud of `n` points `(x, y, z)` typically in the ego frame
  (versus the sensor frame).
  """

  sensor_name = attr.ib(type=str, default='')
  """str: Name of the point sensor, e.g. lidar_top"""

  timestamp = attr.ib(type=int, default=0)
  """int: Timestamp associated with this cloud; typically a Unix stamp in
  nanoseconds."""

  cloud = attr.ib(type=np.ndarray, default=None)
  """numpy.ndarray: Lidar points as an n-by-3 matrix of `(x, y, z)` points.
  Nominally, these points are in **ego** frame, not point sensor frame."""

  ego_to_sensor = attr.ib(type=Transform, default=Transform())
  """Transform: From ego / robot frame to the sensor frame (typically a static
  transform)."""

  ego_pose = attr.ib(type=Transform, default=Transform())
  """Transform: From world to ego / robot frame at the cuboid's `timestamp`"""

  extra = attr.ib(default={}, type=typing.Dict[str, str])
  """Dict[str, str]: A map for adhoc extra context"""

  def __eq__(self, other):
    return misc.attrs_eq(self, other)

  # @
  # def _get_2d_debug_image(
      


  def get_bev_debug_image(
        self, 
        cuboids=None,
        x_bounds_meters=(-50, 50),
        y_bounds_meters=(-50, 50),
        pixels_per_meter=200):
    """Create and return a BEV (Bird's-Eye-View) perspective debug image
    for this point cloud (i.e. flatten the z-axis).

    Args:
      cuboids (List[:class:`~psegs.datum.cuboid.Cuboid`]): Draw these 
        cuboids in the given debug image.
      x_bounds_meters (Tuple[int, int]): Filter points to to this min/max
        x-value in point cloud frame.
      y_bounds_meters (Tuple[int, int]): Filter points to to this min/max
        y-value in point cloud frame.
      pixels_per_meter (int): Rasterize debug image at this resolution.

    Returns:
      np.array: A HWC RGB debug image.
    """






    cuboids = cuboids or []

    ## Draw Cloud
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(dpi=150)
    fig.set_facecolor((0, 0, 0))
    canvas = FigureCanvas(fig)
    
    ax = fig.gca()

    xyz = self.cloud
    if colored_cloud:
      from psegs.util.plotting import rgb_for_distance
      # colors = [~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #   rgb_for_distance(np.linalg.norm(pt)) / 255
      #   for pt in self.cloud
      # ]
      colors = rgb_for_distance(np.linalg.norm(self.cloud, axis=1)) / 255
      ax.scatter(xyz[:, 0], xyz[:, 1], s=.1, c=colors)
    else:
      ax.scatter(xyz[:, 0], xyz[:, 1], s=.1)

    ## Draw Cuboids
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    for c in cuboids:
      box_xyz = c.get_box3d()
      box_xyz_2d = box_xyz[:, :2]

      from scipy.spatial import ConvexHull
      hull = ConvexHull(box_xyz_2d)
      corners = [(box_xyz_2d[v, 0], box_xyz_2d[v, 1]) for v in hull.vertices]
      polygon = Polygon(corners, closed=True)

      from oarphpy.plotting import hash_to_rbg
      color = np.array(hash_to_rbg(c.category_name)) / 255

      ax.add_collection(
        PatchCollection([polygon], facecolor=color, edgecolor=color, alpha=0.5))

    ax.axis('off')
    fig.tight_layout()

    # Render!
    canvas.draw()
    img_str, (width, height) = canvas.print_to_buffer()

    img = np.frombuffer(img_str, np.uint8).reshape((height, width, 4))
    return img[:, :, :3] # Return RGB for easy interop


  # @staticmethod
  # def get_halfspace_debug_image(
  #       cloud,                  # (x, y, z) in meters
  #       flatten_axis=0,         # +x
  #       u_axis=1,               # +y -> +u axis of image
  #       v_axis=2,               # +z -> +v axis of image
  #       u_bounds=(-10, 10),     # In meters
  #       v_bounds=(-10, 10),     # In meters
  #       pixels_per_meter=100):
  # """Create and return a half-space-flattened debug image for the given
  # `cloud` of (x, y, z) points.  For example, an RV (Range-Value-perspective)
  # image flattens the cloud's +x axis (forwards), and a BEV (Bird's-Eye-View)
  # image flattens the cloud's +z axis (up).

  # Args:
  #   cloud (np.array): An nx3 array of points (in units of meters)
  #     draw this cloud.
  #   flatten_axis (int): Flatten this `cloud` axis and use it as the image
  #     plane. Use a positive number to plot points in the positive half space
  #     and a negative number to plot in the negative half space.
  #   u_axis (int): Use this `cloud` axis as the +u (left-to-right) axis
  #     of the debug image.  Negative value flips the `cloud` axis.
  #   v_axis (int): Use this `cloud` axis as the +v (top-to-bottom) axis
  #     of the debug image.  Negative value flips the `cloud` axis.
  #   u_bounds_meters (Tuple[int, int]): Filter points to this min/max
  #     u_axis-value (in meters).
  #   v_bounds_meters (Tuple[int, int]): Filter points to this min/max
  #     v_axis-value (in meters).
  #   pixels_per_meter (int): Rasterize points at this resolution.
  # Returns:
  #   np.array: A HWC RGB debug image.
  # """

  # # Map cloud to (u, v, d) space
  # uvd = np.zeros_like(cloud)
  # uvd[:, 0] = cloud[:, abs(u_axis)] * np.sign(u_axis)
  # uvd[:, 1] = cloud[:, abs(v_axis)] * np.sign(v_axis)
  # uvd[:, 2] = cloud[:, abs(flatten_axis)] * np.sign(flatten_axis)

  # # Filter out-of-view points
  # uvd = uvd[np.where(
  #             (uvd[:, 0] >= u_bounds[0]) &
  #             (uvd[:, 0] <= u_bounds[1]) &
  #             (uvd[:, 1] >= v_bounds[0]) &
  #             (uvd[:, 1] <= v_bounds[1]) &
  #             (uvd[:, 2] >= 0))]
  
  # # Move points to the center of the debug image
  # uvd[: (0, 1)] -= np.min(uvd[: (0, 1)], axis=0)

  # # if ()

  # # uvd[:, ]
  # # if u_axis < 0:





  def get_front_rv_debug_image(
          self,
          cuboids=None,
          z_bounds_meters=(-3, 3),
          y_bounds_meters=(-20, 20),
          pixels_per_meter=50):
    """Create and return an RV (Range-Value) perspective debug image
    for this point cloud (in the +x direction).

    Args:
      cuboids (List[:class:`~psegs.datum.cuboid.Cuboid`]): Draw these 
        cuboids in the given debug image.
      z_bounds_meters (Tuple[int, int]): Filter points to to this min/max
        z-value in point cloud frame.
      y_bounds_meters (Tuple[int, int]): Filter points to to this min/max
        y-value in point cloud frame.
      pixels_per_meter (int): Rasterize debug image at this resolution.

    Returns:
      np.array: A HWC RGB debug image.
    """
    
    import cv2

    # Build the image to return
    w = sum(abs(v) for v in y_bounds_meters) * pixels_per_meter
    h = sum(abs(v) for v in z_bounds_meters) * pixels_per_meter
    img = np.zeros((h, w, 3)).astype(np.uint8)

    def yz_to_uv(yz):
      # cloud +y = img -x axis
      u = -yz[:, 0] * pixels_per_meter + w / 2.
      # cloud +z = img -y axis (down)
      v = -yz[:, 1] * pixels_per_meter + h / 2.
      return np.column_stack([u, v])

    ## Draw Cloud
    # Filter behind ego; keep only +x points
    cloud = self.cloud[:, :3]
    cloud = cloud[np.where(cloud[:, 0] >= 0)]

    # Convert to pixel (u, v, d)
    pts_d = cloud[:, 0]
    pts_uv = yz_to_uv(cloud[:, (1, 2)])
    pts = np.column_stack([pts_uv, pts_d])
    
    pspl.draw_xy_depth_in_image(img, pts, alpha=1.0)

    ## Draw Cuboids
    for c in cuboids or []:
      # TODO frame check ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      box_xyz = c.get_box3d()
      box_xyz_2d = box_xyz[:, (1, 2)]

      # TODO Filter behind +x !!!!!!!!!!!!!!!!!!!!

      from oarphpy.plotting import hash_to_rbg
      color = pspl.color_to_opencv(
        np.array(hash_to_rbg(c.category_name)))

      pspl.draw_cuboid_xy_in_image(
        img,
        yz_to_uv(box_xyz_2d),
        np.array(hash_to_rbg(c.category_name)),
        alpha=0.8)


      # from scipy.spatial import ConvexHull
      # hull = ConvexHull(box_xyz[:, 1:])
      # corners_yz = np.array([
      #   (box_xyz[v, 1], box_xyz[v, 2]) for v in hull.vertices])
      
      # from oarphpy.plotting import hash_to_rbg
      # color = pspl.color_to_opencv(
      #   np.array(hash_to_rbg(c.category_name)))

      # pts_uv = yz_to_uv(corners_yz)
      # pts_uv = np.rint(pts_uv).astype(np.int)

      # # Draw transparent fill
      # CUBOID_FILL_ALPHA = 0.6
      # coverlay = img.copy()
      # cv2.fillPoly(img, [pts_uv], color)
      # img[:] = cv2.addWeighted(
      #   coverlay, CUBOID_FILL_ALPHA, img, 1 - CUBOID_FILL_ALPHA, 0)
      
      # # Draw outline
      # cv2.polylines(
      #   img,
      #   [pts_uv],
      #   True, # is_closed
      #   color,
      #   1) #thickness

    return img


  def to_html(self, cuboids=None, bev_debug=False, rv_debug=False):
    cuboids = cuboids or []
    from psegs.datum.datumutils import to_preformatted
    import tabulate
    table = [
      [attr, to_preformatted(getattr(self, attr))]
      for attr in (
        'sensor_name',
        'timestamp',
        'ego_to_sensor')
    ]

    # TODO: BEV / RV cloud
    table.extend([
      ['Cloud', ''],
      ['Num Points', len(self.cloud)]
    ])

    html = tabulate.tabulate(table, tablefmt='html')

    ### Plotly 3d plot of cloud and cubes  TODO extract to au plotting ~~~~~~~~~~~~~~~~~~

    import plotly
    import plotly.graph_objects as go
    import pandas as pd

    cloud_df = pd.DataFrame(self.cloud, columns=['x', 'y', 'z'])

    from psegs.util.plotting import rgb_for_distance
    cloud_df['color'] = [
      rgb_for_distance(np.linalg.norm(pt))
      for pt in cloud_df[['x', 'y', 'z']].values
    ]

    # df_tmp = df_tmp[df_tmp['norm'] < 500]
    scatter = go.Scatter3d(
                name=self.sensor_name,
                x=cloud_df['x'], y=cloud_df['y'], z=cloud_df['z'],
                mode='markers',
                marker=dict(size=1, color=cloud_df['color'], opacity=0.8),)
    print('plotted %s' % len(self.cloud))

    lines = []
    colors = []
    for cuboid in cuboids:
      cbox = cuboid.get_box3d()
      front = [cbox[i,:] for i in (0, 1, 2, 3)]
      back = [cbox[i,:] for i in (4, 5, 6, 7)]
      
      from oarphpy.plotting import hash_to_rbg
      base_color_rgb = hash_to_rbg(cuboid.category_name)
      base_color = np.array(base_color_rgb)
      front_color = base_color + 0.3 * 255
      back_color = base_color - 0.3 * 255
      center_color = base_color
      
      def to_css_color(rgb):
        r, g, b = np.clip(rgb, 0, 255).astype(int).tolist()
        return 'rgb(%s,%s,%s)' % (r, g, b)

      def make_line(pts):
        return [None] + [list(p) for p in (pts + [pts[0]])] + [None]
      l = make_line(front)
      lines.append(l)
      def add_color(c, n):
        colors.extend(['rgb(0,0,0)'] + (n-2) * [to_css_color(c)] + ['rgb(0,0,0)'])
      add_color(front_color, len(l))
      # colors.extend(['rgb(0,0,0)'] + [to_css_color(front_color)] + ['rgba(0,0,0)'])
      # colors.append(to_css_color(front_color))
      # colors.append(to_css_color(front_color))
      
      for start, end in zip(front, back):
        l = make_line([start, end])
        lines.append(l)
        add_color(center_color, len(l))
        # lines.append(make_line([start, end]))
        # colors.extend(['rgb(0,0,0)'] + [to_css_color(center_color)] + ['rgb(0,0,0)'])
        # colors.append(to_css_color(center_color))
        # colors.append(to_css_color(center_color))

      l = make_line(back)
      lines.append(l)
      add_color(back_color, len(l))
      # lines.append(make_line(back))
      # colors.extend(['rgb(0,0,0)'] + [to_css_color(back_color)] + ['rgb(0,0,0)'])
      # colors.append(to_css_color(back_color))
      # colors.append(to_css_color(back_color))
        
    def to_line_vals(idx, lines):
      import itertools
      ipts = itertools.chain.from_iterable(lines)
      return [(pt[idx] if pt is not None else pt) for pt in ipts]
    lines_plot = go.Scatter3d(
                    name='labels|cuboids',
                    x=to_line_vals(0, lines),
                    y=to_line_vals(1, lines),
                    z=to_line_vals(2, lines),
                    mode='lines',
                    line=dict(width=3, color=colors))
        
    fig = go.Figure(data=[scatter, lines_plot])
    fig.update_layout(
      title=self.sensor_name,
      width=1000, height=700,
      scene_aspectmode='data')
      # scene_camera=dict(
      #   up=dict(x=0, y=0, z=1),
      #   eye=dict(x=0, y=0, z=0),
      #   center=dict(x=1, y=0, z=0),
      # ))
    plot_str = plotly.offline.plot(fig, output_type='div')

    html += '<br/><br/>' + plot_str
    
    return html
