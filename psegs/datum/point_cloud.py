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

import typing

import attr
import numpy as np

from oarphpy.spark import CloudpickeledCallable

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
  """numpy.ndarray: Lidar points as an n-by-d matrix (typically of 
  `(x, y, z)` points). Nominally, these points are in **ego** frame????
  not point sensor frame. need to check this because looks like we put in sensor frame?"""#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # TODO rename to cloud_array once we can dump SD parquet ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  cloud_factory = attr.ib(
    type=CloudpickeledCallable,
    converter=CloudpickeledCallable,
    default=None)
  """CloudpickeledCallable: A serializable factory function that emits an HWC
    numpy array image"""

  cloud_colnames = attr.ib(default=['x', 'y', 'z'])
  """List[str]: Semantic names for the columns (or dimensions / attributes)
  of the cloud.  Typically clouds have just 3-D (x, y, z) points, but some
  clouds have reflectance, RGB, labels, and/or other data."""

  # then start using get_cloud() in call sites.  could rename cloud to cloud_array and then just dump
  # the nuscenes SDTable that we built...
  # for SDTable and impls, lets:
  #  * default to callable use in the code for big assets
  #  * give SDTable a base class flag if the class to_row() should expand the data or not ! 

  ego_to_sensor = attr.ib(type=Transform, default=Transform())
  """Transform: From ego / robot frame to the sensor frame (typically a static
  transform)."""

  ego_pose = attr.ib(type=Transform, default=Transform())
  """Transform: From world to ego / robot frame at the cuboid's `timestamp`"""

  extra = attr.ib(default={}, type=typing.Dict[str, str])
  """Dict[str, str]: A map for adhoc extra context"""

  def __eq__(self, other):
    return misc.attrs_eq(self, other)

  @classmethod
  def create_world_frame_cloud(cls, sensor_name='', **kwargs):
    sensor_name = sensor_name or 'world_frame_cloud'
    ego_to_sensor = Transform(src_frame=sensor_name, dest_frame='ego')
    ego_pose = Transform(src_frame='ego', dest_frame='world')
    return cls(
            sensor_name=sensor_name,
            ego_to_sensor=ego_to_sensor,
            ego_pose=ego_pose,
            **kwargs)

  def get_cloud(self):
    if self.cloud is not None:
      return self.cloud
    elif self.cloud_factory != CloudpickeledCallable.empty():
      return self.cloud_factory()
    else:
      raise ValueError("No cloud data!")

  def get_col_idx(self, colname):
    for i in range(len(self.cloud_colnames)):
      if self.cloud_colnames[i] == colname:
        return i
    raise ValueError(
      "Colname %s not found in %s" % (colname, self.cloud_colnames))

  def get_xyz_axes(self):
    return [
      self.get_col_idx('x'),
      self.get_col_idx('y'),
      self.get_col_idx('z'),
    ]

  def get_xyz_cloud(self):
    cloud = self.get_cloud()
    xyz = cloud[:, self.get_xyz_axes()]
    return xyz

  # @
  # def _get_2d_debug_image(
      


  # def get_bev_debug_image(
  #       self, 
  #       cuboids=None,
  #       x_bounds_meters=(-50, 50),
  #       y_bounds_meters=(-50, 50),
  #       pixels_per_meter=200):
  #   """Create and return a BEV (Bird's-Eye-View) perspective debug image
  #   for this point cloud (i.e. flatten the z-axis).

  #   Args:
  #     cuboids (List[:class:`~psegs.datum.cuboid.Cuboid`]): Draw these 
  #       cuboids in the given debug image.
  #     x_bounds_meters (Tuple[int, int]): Filter points to to this min/max
  #       x-value in point cloud frame.
  #     y_bounds_meters (Tuple[int, int]): Filter points to to this min/max
  #       y-value in point cloud frame.
  #     pixels_per_meter (int): Rasterize debug image at this resolution.

  #   Returns:
  #     np.array: A HWC RGB debug image.
  #   """






  #   cuboids = cuboids or []

  #   ## Draw Cloud
  #   import matplotlib
  #   from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
  #   from matplotlib.figure import Figure

  #   fig = Figure(dpi=150)
  #   fig.set_facecolor((0, 0, 0))
  #   canvas = FigureCanvas(fig)
    
  #   ax = fig.gca()

  #   xyz = self.cloud
  #   if colored_cloud:
  #     from psegs.util.plotting import rgb_for_distance
  #     # colors = [~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #     #   rgb_for_distance(np.linalg.norm(pt)) / 255
  #     #   for pt in self.cloud
  #     # ]
  #     colors = rgb_for_distance(np.linalg.norm(self.cloud, axis=1)) / 255
  #     ax.scatter(xyz[:, 0], xyz[:, 1], s=.1, c=colors)
  #   else:
  #     ax.scatter(xyz[:, 0], xyz[:, 1], s=.1)

  #   ## Draw Cuboids
  #   from matplotlib.patches import Polygon
  #   from matplotlib.collections import PatchCollection

  #   for c in cuboids:
  #     box_xyz = c.get_box3d()
  #     box_xyz_2d = box_xyz[:, :2]

  #     from scipy.spatial import ConvexHull
  #     hull = ConvexHull(box_xyz_2d)
  #     corners = [(box_xyz_2d[v, 0], box_xyz_2d[v, 1]) for v in hull.vertices]
  #     polygon = Polygon(corners, closed=True)

  #     from oarphpy.plotting import hash_to_rbg
  #     color = np.array(hash_to_rbg(c.category_name)) / 255

  #     ax.add_collection(
  #       PatchCollection([polygon], facecolor=color, edgecolor=color, alpha=0.5))

  #   ax.axis('off')
  #   fig.tight_layout()

  #   # Render!
  #   canvas.draw()
  #   img_str, (width, height) = canvas.print_to_buffer()

  #   img = np.frombuffer(img_str, np.uint8).reshape((height, width, 4))
  #   return img[:, :, :3] # Return RGB for easy interop

  @staticmethod
  def paint_ego_cloud(cloud, camera_images=None):
    """ TODO comment """
    xyzrgb = np.ones((cloud.shape[0], 3 + 3)) * 128.
    xyzrgb[:, :3] = cloud[:, :3]

    camera_images = camera_images or []
    #alpha = 1. / len(camera_images) if camera_images else 1.
    for i, ci in enumerate(camera_images):
      uvd = ci.project_ego_to_image(xyzrgb[:, :3], omit_offscreen=False)
      
      img = ci.image
      h, w = ci.image.shape[:2]
      uvd[:, :2] = np.rint(uvd[:, :2])
      to_paint = np.where(
                (uvd[:, 0] >= 0) & 
                (uvd[:, 0] < w) &
                (uvd[:, 1] >= 0) & 
                (uvd[:, 1] < h) &
                (uvd[:, 2] >= 0.01))
      px_xy = uvd[to_paint].astype(np.int)
      painted = img[px_xy[:, 1], px_xy[:, 0], :]
      # if i == 0:
      xyzrgb[to_paint[0], 3:] = painted
      # else:
      #   xyzrgb[to_paint[0], 3:] = (
      #     alpha * xyzrgb[to_paint[0], 3:] + alpha * painted)

    return xyzrgb

  @staticmethod
  def get_ortho_debug_image(
        cloud,
        user_colors=None,
        cuboids=None,
        camera_images=None,
        ego_to_sensor=None,
        flatten_axis='+x',
        u_axis='+y',
        v_axis='+z',
        u_bounds=(-10, 10),
        v_bounds=(-10, 10),
        depth_values=None,
        filter_behind=True,
        pixels_per_meter=200):
    """Create and return a half-space-flattened debug image for the given
    `cloud` of (x, y, z) points.  For example, an RV (Range-Value-perspective)
    image flattens the cloud's +x axis (forwards), and a BEV (Bird's-Eye-View
    perspective) image flattens the cloud's +z axis (up).

    Args:
      cloud (np.array): An nx3 array of points (in units of meters)
        draw this cloud.
      user_colors (np.array): Optionally color each point using this nx3 array of
        RGB colors (with color values in [0, 255]).  By default, color
        points based on distance from the origin.
      cuboids (List[:class:`~psegs.datum.cuboid.Cuboid`]): Optionally draw
        these cuboids in the given debug image; cuboids must either (a) be in
        the ego frame (PSegs standard) and `ego_to_sensor` given, or (b) the
        caller of this method must first transform `cuboids` to the point
        sensor frame.
      camera_images (List[:class:`~psegs.datum.camera_image.CameraImage`]): 
        Optionally paint the cloud points using pixels from these camera
        images.  By default, color points based upon distance from the sensor
        origin.
      ego_to_sensor (:class:`~psegs.datum.transform.Transform`): Optional
        transform for projecting ego points (`cuboids` corners and
        `camera_image` rays) to the sensor frame.
      flatten_axis (str): Flatten this `cloud` axis and use it as the image
        plane. Use a positive sign and `filter_behind=True` to plot points in
        the positive half-space.
      u_axis (str): Use this `cloud` axis as the +u (left-to-right) axis
        of the debug image.  Negative sign flips the `cloud` axis.
      v_axis (str): Use this `cloud` axis as the +v (bottom-to-top) axis
        of the debug image.  Negative sign flips the `cloud` axis.
      u_bounds_meters (Tuple[int, int]): Restrict view to this min/max
        u_axis-value (in meters).  Use None to auto-fit.
      v_bounds_meters (Tuple[int, int]): Restrict view to this min/max
        v_axis-value (in meters).  Use None to auto-fit.
      depth_values (np.array): Optional nx1 array of depth-in-meters values to
        use for plot colors (in place of the raw `flatten_axis` values).
      filter_behind (bool): Restrict view to only positive points
        along the flattened dimension.
      pixels_per_meter (int): Rasterize points at this resolution.
    
    Returns:
      np.array: A HWC RGB debug image.
    """

    def pts_to_uvd(pts):
      # Return a copy of `pts` changing axis ordering to reflect the desired
      # `u`, `v`, and `d` axes (new x y and z).
      AXIS_NAME_TO_IDX = {'x': 0, 'y': 1, 'z': 2}
      AXES = (u_axis, v_axis, flatten_axis)

      uvd = np.zeros((pts.shape[0], 3))
      uid, vid, did = tuple(AXIS_NAME_TO_IDX[a[-1]] for a in AXES)
      us, vs, ds = tuple(-1. if a[0] == '-' else 1. for a in AXES)

      uvd[:, 0] = pts[:, uid] * us
      uvd[:, 1] = pts[:, vid] * vs
      uvd[:, 2] = pts[:, did] * ds

      return uvd

    # Map cloud to (u, v, d) space
    uvd = pts_to_uvd(cloud)

    unfiltered = None
    if filter_behind:
      unfiltered = uvd[:, 2] >= 0
      uvd = uvd[unfiltered]

    # Decide bounds
    if u_bounds is None:
      u_bounds = (uvd[:, 0].min(), uvd[:, 0].max())
    if v_bounds is None:
      v_bounds = (uvd[:, 1].min(), uvd[:, 1].max())
    if depth_values is not None:
      uvd[:, 2] = depth_values

    # Maybe paint the cloud
    if camera_images and (user_colors is None):
      cloud = cloud[:, :3] # Ignore any non-position columns
      to_paint = (
        ego_to_sensor.get_inverse().apply(cloud).T
        if ego_to_sensor
        else cloud)
      if unfiltered is not None:
        to_paint = to_paint[unfiltered]
      xyzrgb = PointCloud.paint_ego_cloud(to_paint, camera_images=camera_images)
      user_colors = xyzrgb[:, 3:]

    # Draw!
    img = pspl.get_ortho_debug_image(
            uvd,
            min_u=u_bounds[0],
            max_u=u_bounds[1],
            min_v=v_bounds[0],
            max_v=v_bounds[1],
            pixels_per_meter=pixels_per_meter,
            period_meters=10.,
            user_colors=user_colors)
  
    for c in cuboids or []:
      box_xyz = c.get_box3d()
      if ego_to_sensor is not None:
        box_xyz = ego_to_sensor.apply(box_xyz).T
      box_uvd = pts_to_uvd(box_xyz)

      if filter_behind:
        has_in_front = np.any(box_uvd[:, 2] >= 0)
        if not has_in_front:
          continue

      box_uv = box_uvd[:, (0, 1)] - np.array([u_bounds[0], v_bounds[1]])
      box_uv *= pixels_per_meter
      box_uv[:, 1] *= -1 # Debug image y-axis is flipped
      box_uv = np.rint(box_uv).astype(np.int)

      from oarphpy.plotting import hash_to_rbg
      # color = pspl.color_to_opencv(
      #   np.array(hash_to_rbg(c.category_name)))

      pspl.draw_cuboid_xy_in_image(
        img,
        box_uv,
        np.array(hash_to_rbg(c.category_name)),
        alpha=0.3)

    return img


  def get_front_rv_debug_image(
          self,
          cuboids=None,
          camera_images=None,
          z_bounds_meters=(-3, 3),
          y_bounds_meters=(-20, 20),
          pixels_per_meter=200):
    """Create and return an RV (Range-Value) perspective debug image
    for this point cloud (in the +x direction).

    Args:
      cuboids (List[:class:`~psegs.datum.cuboid.Cuboid`]): (Optional) draw
        these cuboids in the given debug image.
      camera_images (List[:class:`~psegs.datum.camera_image.CameraImage`]): 
        (Optional) paint the cloud points using pixels from these camera
        images.
      z_bounds_meters (Tuple[int, int]): Filter points to to this min/max
        z-value in point cloud frame.  Use `None` to auto-size.
      y_bounds_meters (Tuple[int, int]): Filter points to to this min/max
        y-value in point cloud frame.  Use `None` to auto-size.
      pixels_per_meter (int): Rasterize debug image at this resolution.

    Returns:
      np.array: A HWC RGB debug image.
    """
    cloud = self.get_xyz_cloud()
    return PointCloud.get_ortho_debug_image(
              cloud,
              cuboids=cuboids,
              camera_images=camera_images,
              ego_to_sensor=self.ego_to_sensor,
              flatten_axis='+x',
              u_axis='-y',
              v_axis='+z',
              u_bounds=y_bounds_meters,
              v_bounds=z_bounds_meters,
              filter_behind=True,
              pixels_per_meter=pixels_per_meter)
  

  def get_bev_debug_image(
          self,
          cuboids=None,
          camera_images=None,
          x_bounds_meters=(-80, 80),
          y_bounds_meters=(-80, 80),
          pixels_per_meter=20):
    """Create and return a BEV (Birds Eye View) perspective debug image
    for this point cloud.

    Args:
      cuboids (List[:class:`~psegs.datum.cuboid.Cuboid`]): Draw these 
        cuboids in the given debug image.
      camera_images (List[:class:`~psegs.datum.camera_image.CameraImage`]): 
        (Optional) paint the cloud points using pixels from these camera
        images.
      x_bounds_meters (Tuple[int, int]): Filter points to to this min/max
        x-value in point cloud frame.  Use `None` to auto-size.
      y_bounds_meters (Tuple[int, int]): Filter points to to this min/max
        y-value in point cloud frame.  Use `None` to auto-size.
      pixels_per_meter (int): Rasterize debug image at this resolution.

    Returns:
      np.array: A HWC RGB debug image.
    """
    cloud = self.get_xyz_cloud()
    depth_values = np.linalg.norm(cloud[:, (0, 1)], axis=-1)
    return PointCloud.get_ortho_debug_image(
              cloud,
              cuboids=cuboids,
              camera_images=camera_images,
              ego_to_sensor=self.ego_to_sensor,
              flatten_axis='-z',
              u_axis='+x',
              v_axis='+y',
              u_bounds=x_bounds_meters,
              v_bounds=y_bounds_meters,
              depth_values=depth_values,
              filter_behind=False,
              pixels_per_meter=pixels_per_meter)


    # import cv2

    # # Build the image to return
    # w = sum(abs(v) for v in y_bounds_meters) * pixels_per_meter
    # h = sum(abs(v) for v in z_bounds_meters) * pixels_per_meter
    # img = np.zeros((h, w, 3)).astype(np.uint8)

    # def yz_to_uv(yz):
    #   # cloud +y = img -x axis
    #   u = -yz[:, 0] * pixels_per_meter + w / 2.
    #   # cloud +z = img -y axis (down)
    #   v = -yz[:, 1] * pixels_per_meter + h / 2.
    #   return np.column_stack([u, v])

    # ## Draw Cloud
    # # Filter behind ego; keep only +x points
    # cloud = self.cloud[:, :3]
    # cloud = cloud[np.where(cloud[:, 0] >= 0)]

    # # Convert to pixel (u, v, d)
    # pts_d = cloud[:, 0]
    # pts_uv = yz_to_uv(cloud[:, (1, 2)])
    # pts = np.column_stack([pts_uv, pts_d])
    
    # pspl.draw_xy_depth_in_image(img, pts, alpha=1.0)

    # ## Draw Cuboids
    # for c in cuboids or []:
    #   # TODO frame check ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   box_xyz = c.get_box3d()
    #   box_xyz_2d = box_xyz[:, (1, 2)]

    #   # TODO Filter behind +x !!!!!!!!!!!!!!!!!!!!

    #   from oarphpy.plotting import hash_to_rbg
    #   color = pspl.color_to_opencv(
    #     np.array(hash_to_rbg(c.category_name)))

    #   pspl.draw_cuboid_xy_in_image(
    #     img,
    #     yz_to_uv(box_xyz_2d),
    #     np.array(hash_to_rbg(c.category_name)),
    #     alpha=0.8)


    #   # from scipy.spatial import ConvexHull
    #   # hull = ConvexHull(box_xyz[:, 1:])
    #   # corners_yz = np.array([
    #   #   (box_xyz[v, 1], box_xyz[v, 2]) for v in hull.vertices])
      
    #   # from oarphpy.plotting import hash_to_rbg
    #   # color = pspl.color_to_opencv(
    #   #   np.array(hash_to_rbg(c.category_name)))

    #   # pts_uv = yz_to_uv(corners_yz)
    #   # pts_uv = np.rint(pts_uv).astype(np.int)

    #   # # Draw transparent fill
    #   # CUBOID_FILL_ALPHA = 0.6
    #   # coverlay = img.copy()
    #   # cv2.fillPoly(img, [pts_uv], color)
    #   # img[:] = cv2.addWeighted(
    #   #   coverlay, CUBOID_FILL_ALPHA, img, 1 - CUBOID_FILL_ALPHA, 0)
      
    #   # # Draw outline
    #   # cv2.polylines(
    #   #   img,
    #   #   [pts_uv],
    #   #   True, # is_closed
    #   #   color,
    #   #   1) #thickness

    # return img

  def to_trimeshes_world_frame(
          self,
          period_meters=1.,
          max_num_points=100_000,
          colors=None):

    import trimesh
    from psegs.util.plotting import rgb_for_distance

    T_ego_from_sensor = self.ego_to_sensor[self.sensor_name, 'ego']
    T_world_from_ego = self.ego_pose['ego', 'world']
    p2w = T_world_from_ego @ T_ego_from_sensor
    w2p = p2w.get_inverse()

    xyz = self.get_xyz_cloud()
    xyz = w2p.apply(xyz).T
    if max_num_points >= 0 and xyz.shape[0] > max_num_points:
      idx = np.random.choice(
              np.arange(xyz.shape[0]), max_num_points)
      xyz = xyz[idx, :]
    
    if colors is None:
      colors = rgb_for_distance(
                  np.linalg.norm(xyz, axis=1),
                  period_meters=period_meters)
      colors = np.clip(colors, 0, 255).astype('uint8')
    pc_tmesh = trimesh.points.PointCloud(
                vertices=xyz if len(xyz) else np.array([[0., 0., 0.]]),  # TODO fixme trimesh wont GLTF empty array?????
                colors=colors if len(colors) else None)
    
    return [pc_tmesh]
      


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
    cloud = self.get_cloud()
    table.extend([
      ['Cloud', ''],
      ['Num Points', cloud.shape[0]]
    ])

    html = tabulate.tabulate(table, tablefmt='html')

    ### Plotly 3d plot of cloud and cubes  TODO extract to au plotting ~~~~~~~~~~~~~~~~~~

    import plotly
    import plotly.graph_objects as go
    import pandas as pd

    cloud = self.get_cloud()
    cloud_df = pd.DataFrame(cloud, columns=['x', 'y', 'z'])

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
    # print('plotted %s' % len(cloud))

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
