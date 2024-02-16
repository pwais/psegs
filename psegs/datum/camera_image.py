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






"""

TODOs

 * try center crop of wide lenses

video{start,end}datum with (start, end) [will work with time range segjoin?]
and ffmpeg explode










"""

import copy
import math
import typing

import attr
import numpy as np

from oarphpy.spark import CloudpickeledCallable

from psegs.datum.transform import Transform
from psegs.util import misc
from psegs.util import plotting as pspl


def l2_normalized(v):
  if len(v.shape) > 1:
    # Normalize row-wise
    return v / np.linalg.norm(v, axis=1)[:, np.newaxis]
  else:
    return v / np.linalg.norm(v)


def theta_signed(axis, v):
  return np.arctan2(np.cross(axis, v), np.dot(axis, v.T))


def depth_to_uvcs(hwc_arr):
  h, w, c = hwc_arr.shape[:3]
  px_y = np.tile(np.arange(h)[:, np.newaxis], [1, w])
  px_x = np.tile(np.arange(w)[np.newaxis, :], [h, 1])
  pyx = np.concatenate([px_y[:,:,np.newaxis], px_x[:, :, np.newaxis]], axis=-1)
  pyx = pyx.astype(np.float32)
  
  chans = [hwc_arr[:, :, i] for i in range(c)]
  vuc = np.dstack([pyx] + chans).reshape([-1, 2 + c])
  axes = list(range(2 + c))
  axes[0] = 1
  axes[1] = 0
  uvc = vuc[:, axes]
  return uvc


def uvdcs_to_xyzcs(uvdcs, fx, cx, fy, cy, normed_rays=False):
  rays = np.zeros((uvdcs.shape[0], 3))
  rays[:, 0] = (uvdcs[:, 0] - cx) / fx
  rays[:, 1] = (uvdcs[:, 1] - cy) / fy
  rays[:, 2] = 1.
  if normed_rays:
    rays /= np.linalg.norm(rays, axis=-1)[:, np.newaxis]
  xyz = uvdcs[:, 2][:, np.newaxis] * rays

  if uvdcs.shape[1] > 3:
    cs = uvdcs[:, 3:]
    return np.concatenate([xyz, cs], axis=1)
  else:
    return xyz


@attr.s(slots=True, eq=False, weakref_slot=False)
class CameraImage(object):
  """An image from a camera; typically the camera is calibrated.  The image
  could also be a depth image."""

  sensor_name = attr.ib(type=str, default='')
  """str: Name of the camera, e.g. camera_front"""

  image_jpeg = attr.ib(type=bytearray, default=bytearray())
  """bytearray: Buffer of image JPEG data"""

  image_png = attr.ib(type=bytearray, default=bytearray())
  """bytearray: Buffer of image PNG data"""

  image_factory = attr.ib(
      type=CloudpickeledCallable,
      converter=CloudpickeledCallable,
      default=None)
  """CloudpickeledCallable: A serializable factory function that emits an HWC
  numpy array image"""

  width = attr.ib(type=int, default=0, validator=None)
  """int: Width of image in pixels"""

  height = attr.ib(type=int, default=0, validator=None)
  """int: Height of image in pixels"""

  timestamp = attr.ib(type=int, default=0)
  """int: Timestamp associated with this image; typically a Unix stamp in
  nanoseconds."""

  ego_pose = attr.ib(type=Transform, default=Transform())
  """Transform: From world to ego / robot frame at the image's `timestamp`"""

  ego_to_sensor = attr.ib(type=Transform, default=Transform())
  """Transform: From ego / robot frame to the camera frame (typically a static
  transform)."""

  K = attr.ib(type=np.ndarray, default=np.eye(3, 3, dtype='float64'))
  """numpy.ndarray: The 3x3 intrinsic calibration camera matrix"""

  distortion_model = attr.ib(type=str, default="")
  """str: Optional distortion model, e.g. OPENCV"""

  distortion_kv = attr.ib(default={}, type=typing.Dict[str, float])
  """Dict[str, float]: A map of distortion parameter name -> distortion paramte
  value.  E.g. for OPENCV there might be entries for k1, k2, p1, p2."""

  channel_names = attr.ib(default=['r', 'g', 'b'])
  """List[str]: Semantic names for the channels (or dimensions / attributes)
  of the image. By default, the `image` member uses `imageio` to read an
  3-channel RGB image as a HWC array.  (Some PNGs could use an alpha channel
  to produce an RGBA image).  In the case of depth images, one of the channels
  (usually the first) decodes as depth in meters."""

  extra = attr.ib(default={}, type=typing.Dict[str, str])
  """Dict[str, str]: A map for adhoc extra context"""


  def __eq__(self, other):
    return misc.attrs_eq(self, other)

  def get_world_to_sensor(self):
    return (
      # FIXME this is inverse of what says it is? check with colmap data etc
      self.ego_to_sensor[self.sensor_name, 'ego'] @ 
      self.ego_pose['ego', 'world']
    )

  @classmethod
  def create_world_frame_ci(cls, sensor_name='', **kwargs):
    sensor_name = sensor_name or 'world_frame_camera_image'
    ego_to_sensor = Transform(src_frame=sensor_name, dest_frame='ego')
    ego_pose = Transform(src_frame='ego', dest_frame='world')
    return cls(
            sensor_name=sensor_name,
            ego_to_sensor=ego_to_sensor,
            ego_pose=ego_pose,
            **kwargs)

  @property
  def image(self):
    """Decode and return the image.

    Returns
      numpy.ndarray: An HWC image with values in [0, 255]
    """
    buf = self.image_buffer
    if not buf:
      if self.image_factory != CloudpickeledCallable.empty():
        return self.image_factory()
      else:
        raise ValueError("No image data!")
      
    from io import BytesIO
    import imageio
    return imageio.imread(BytesIO(buf))
  
  @property
  def image_buffer(self):
    """Return the byte buffer storing the wrapped image (if any).

    Returns
      bytearray: Raw image bytes; might be JPEG, PNG, etc.
    """
    return (self.image_jpeg or self.image_png)

  def get_fov(self):
    """Return the horizontal and verticle Fields of View in radians:
    (FoV_h, FoV_v)"""
    f_x = self.K[0, 0]
    f_y = self.K[1, 1]
    fov_h = 2. * math.atan(.5 * self.width / f_x)
    fov_v = 2. * math.atan(.5 * self.height / f_y)
    return fov_h, fov_v

  def get_chan(self, channel_name):
    for i, c in enumerate(self.channel_names):
      if c == channel_name:
        full_img = self.image
        depth = full_img[:, :, i]
        return depth
    return None

  def has_depth(self):
    return 'depth' in self.channel_names

  def get_depth(self):
    return self.get_chan('depth')

  def get_P(self, from_world=True):
    if from_world:
      xform = self.ego_pose['world', 'ego']
      RT_w2e = xform.get_transformation_matrix(homogeneous=True)
    else:
      RT_w2e = np.eye(4)
    
    xform = self.ego_to_sensor['ego', self.sensor_name]
    RT_e2c = xform.get_transformation_matrix(homogeneous=True)
    K_h = np.eye(4)
    K_h[:3, :3] = self.K
    P_h = K_h @ RT_e2c @ RT_w2e
    P = P_h[:3, :4]
    return P

  def has_rgb(self):
    missing = set(['r', 'g', 'b']) - set(self.channel_names)
    return not missing

  def get_opencv_distcoeffs(self):
    # fmt: off
    # OpenCV wants at least four numbers, perhaps more, in a specific order
    KEYS = (
      # Base model
      'k1', 'k2', 'p1', 'p2',
      # Full model
      'k3', 'k4', 'k5', 'k6', 
      # TODO support othermodels
    )
    # fmt: on

    dist_coeff_raw = [self.distortion_kv.get(k) for k in KEYS]
    dist_coeff_raw = [v for v in dist_coeff_raw if v is not None]
    if dist_coeff_raw:
      return np.array(dist_coeff_raw)
    else:
      return None

  def to_cv_undistorted_ci(self, alpha=0.):
    """Uses cache-friendly image_factory"""
    import cv2

    assert self.has_rgb()
    assert not self.has_depth(), 'TODO e.g. cv2.projectPoints for depth images'

    dist = self.get_opencv_distcoeffs()

    cur_wh = (self.width, self.height)
    newK, roi = cv2.getOptimalNewCameraMatrix(self.K, dist, cur_wh, alpha)
    rx, ry, rw, rh = roi

    # print()
    # print("self.K, dist, cur_wh, alpha", (self.K.tolist(), dist, cur_wh, alpha))
    # print("newK, roi", (newK.tolist(), roi))
    # print()


    def _get_undistorted():
      import cv2
      # NB: cv2.remap() might be faster for latency-critical use cases
      undistorted = cv2.undistort(self.image, self.K, dist, None, newK)
      # TODO support cv2.fisheye.undistortImage
      
      undistorted = undistorted[ry : ry + rh, rx : rx + rw]
      return undistorted

    undistorted_ci = copy.deepcopy(self)
    undistorted_ci.image_factory = CloudpickeledCallable(lambda: _get_undistorted())
    undistorted_ci.K = newK
    undistorted_ci.distortion_kv = {}
    undistorted_ci.distortion_model = ''
    undistorted_ci.width = rw
    undistorted_ci.height = rh
    return undistorted_ci

  def to_resized_ci(
        self,
        target_h=None,
        scale=1.0,
        interpolate='',
        resize_dtype='float32',
        final_dtype='uint8'):
    """Uses cache-friendly image_factory"""

    if target_h is not None:
      scale = float(target_h) / self.height      
    else:
      assert scale is not None
    tw = int(self.width * scale)
    th = int(self.height * scale)

    if not interpolate:
      if scale <= 1:
        interpolate = 'INTER_AREA'
      else:
        interpolate = 'INTER_CUBIC'

    def _get_resized():
      import cv2

      assert hasattr(cv2, interpolate), \
        (interpolate, [x for x in dir(cv2) if 'INTER_' in x])
      cv2_interp = getattr(cv2, interpolate)

      image = self.image
      if resize_dtype:
        image = image.astype(resize_dtype)
      resized = cv2.resize(image, (tw, th), interpolation=cv2_interp)
      if final_dtype:
        resized = resized.astype(final_dtype)
      return resized

    newK = self.K.copy()
    newK[:2, :] *= scale
    # K [[3168.395196984074, 0.0, 2048.995029305805], [0.0, 3248.9459634700142, 1030.390052631334], [0.0, 0.0, 1.0]] (2159, 3839) 
    # newK [[798.0823536034384, 0.0, 516.1183103252705], [0.0, 818.6320537877201, 259.625840033092], [0.0, 0.0, 1.0]] (544, 967) (0.2518885126334983, 0.25196850393700787)
    # scale_x = float(tw) / self.width
    # newK[0, 0] *= scale_x
    # newK[0, 2] *= scale_x

    # scale_y = float(th) / self.height
    # newK[1, 1] *= scale_y
    # newK[1, 2] *= scale_y

    # print('K', self.K.tolist(), (self.height, self.width), 'newK', newK.tolist(), (th, tw), scale)

    resized_ci = copy.deepcopy(self)
    resized_ci.image_factory = CloudpickeledCallable(lambda: _get_resized())
    resized_ci.K = newK
    resized_ci.width = tw
    resized_ci.height = th
    return resized_ci

  def depth_image_to_point_cloud(self):
    """Create and return a datum.PointCloud instance if this image is
    a depth image (and None otherwise)"""
    
    if not any(c == 'depth' for c in self.channel_names):
      return None

    # Re-order so that depth is always the first channel / column
    cs_names = list(self.channel_names)
    axes = list(range(len(self.channel_names)))
    for i, c in enumerate(self.channel_names):
      if c == 'depth' and i != 0:
        axes[0] = i
        axes[i] = 0
        cs_names[0] = 'depth'
        cs_names[i] = self.channel_names[0]
        break
    cloud_colnames = ['x', 'y', 'z'] + cs_names[1:]

    fx = self.K[0, 0]
    cx = self.K[0, 2]
    fy = self.K[1, 1]
    cy = self.K[1, 2]
    if self.image_factory != CloudpickeledCallable.empty():

      def _to_cloud(image_factory, axes, intrinsics):
        fx, cx, fy, cy = intrinsics
        depth_image = image_factory()
        depth_image = depth_image[:, :, axes]
        uvdcs = depth_to_uvcs(depth_image)
        
        # Ignore depth 0
        idx = np.where(uvdcs[:, 2] > 0)
        uvdcs = uvdcs[idx[0]]
        
        xyzcs = uvdcs_to_xyzcs(uvdcs, fx, cx, fy, cy)
        return xyzcs
      
      depth_factory = self.image_factory
      cloud_factory = lambda: _to_cloud(depth_factory, axes, (fx, cx, fy, cy))
      cloud = None
    else:
      depth_image = np.copy(self.image)
      depth_image = depth_image[:, :, axes]
      uvdcs = depth_to_uvcs(depth_image)

      # Ignore depth 0
      idx = np.where(uvdcs[:, 2] > 0)
      uvdcs = uvdcs[idx[0]]

      cloud = uvdcs_to_xyzcs(uvdcs, fx, cx, fy, cy)
      cloud_factory = None

    from psegs.datum.point_cloud import PointCloud
    pc = PointCloud(
          sensor_name=self.sensor_name + '|point_cloud',
          timestamp=self.timestamp,
          cloud_colnames=cloud_colnames,
          cloud_factory=cloud_factory,
          cloud=cloud,
          ego_pose=copy.deepcopy(self.ego_pose),
          ego_to_sensor=copy.deepcopy(self.ego_to_sensor),
          extra=copy.deepcopy(self.extra))
    return pc


  def get_debug_image(self, clouds=None, cuboids=None, period_meters=10.):
    """Create and return a debug image showing the given content projected
    onto this `CameraImage`.

    Args:
      clouds (List[:class:`~psegs.datum.point_cloud.PointCloud`]): Draw these 
        PointClouds in the given debug image.
      cuboids (List[:class:`~psegs.datum.cuboid.Cuboid`]): Draw these 
        cuboids in the given debug image.
      period_meters (float): Choose a distinct hue every `period_meters` and
        interpolate between hues.

    Returns:
      np.array: A HWC RGB debug image.
    """

    if any(c == 'depth' for c in self.channel_names):
      depth = self.get_depth()
      assert depth is not None, (self.channel_names, self.image.shape)
      debug_img = np.zeros((self.height, self.width, 3))
      uvd = depth_to_uvcs(depth.reshape([self.height, self.width, 1]))

      # Ignore depth 0
      idx = np.where(uvd[:, 2] > 0)
      uvd = uvd[idx[0]]

      from psegs.util.plotting import draw_xy_depth_in_image
      draw_xy_depth_in_image(
        debug_img, uvd, alpha=0.25, period_meters=period_meters)
    else:
      debug_img = np.copy(self.image)
    
    for pc in clouds or []:
      cloud_raw = pc.get_cloud()
      xyz = pc.ego_to_sensor.get_inverse().apply(cloud_raw[:, :3]).T # err why inv~~~~
      uvd = self.project_ego_to_image(xyz, omit_offscreen=True)
      pspl.draw_xy_depth_in_image(
        debug_img, uvd, marker_radius=4, alpha=0.9, period_meters=period_meters)
    
    for c in cuboids or []:
      # box_xyz = self.ego_to_sensor.apply(c.get_box3d()).T
      box_uvd = self.project_ego_to_image(c.get_box3d(), omit_offscreen=False)
      if (box_uvd[:, 2] <= 1e-6).all():
        continue
      
      from oarphpy.plotting import hash_to_rbg
      color = pspl.color_to_opencv(
        np.array(hash_to_rbg(c.category_name)))

      pspl.draw_cuboid_xy_in_image(
        debug_img,
        box_uvd[:, :2],
        np.array(hash_to_rbg(c.category_name)),
        alpha=0.3)
    
    return debug_img

  def project_ego_to_image(self, pts, omit_offscreen=True):
    """Project the given points into the image plane.

    Args:
      pts (numpy.ndarray): An n-by-3 array of points `(x, y, z)` in the **ego
        frame**.
      omit_offscreen (bool): Omit any point projected outside the image.
    
    Returns:
      numpy.ndarray: An n-by-3 array of points `(x, y, d)` in the image plane
        where `(x, y)` is a pixel location and `d` is depth in meters from
        the focal plane.
    """
    pts_in_cam = self.ego_to_sensor.apply(pts).T

    if omit_offscreen:
      fov_h, fov_v = self.get_fov()
      half_fov_h, half_fov_v = .5 * fov_h, .5 * fov_v

      Z_HAT = np.array([0, 1]) # Principal axis in X-Z and Y-Z planes
      pts_xz = pts_in_cam[:, (0, 2)]
      theta_h = theta_signed(l2_normalized(pts_xz), Z_HAT)
      pts_yz = pts_in_cam[:, (1, 2)]
      theta_v = theta_signed(l2_normalized(pts_yz), Z_HAT)

      PADDING_RADIANS = math.pi / 8
      idx_ = np.where(
              np.logical_and.reduce((
                # Filter off-the-edge points
                np.abs(theta_h) <= half_fov_h + PADDING_RADIANS,
                np.abs(theta_v) <= half_fov_v + PADDING_RADIANS)))
                # # Filter behind-screen points
                # uv[2, :] > 0)))
      idx_ = idx_[0]
      pts_in_cam = pts_in_cam[idx_, :]

    uvd = self.K.dot(pts_in_cam.T)
    uvd[0:2, :] /= uvd[2, :]
    uvd = uvd.T

    return uvd

  def _has_edge_in_fov(self, cuboid):
    
    f_x = self.K[0, 0]
    f_y = self.K[1, 1]
    fov_h = 2. * math.atan(.5 * self.width / f_x)
    fov_v = 2. * math.atan(.5 * self.height / f_y)

    def intervals_overlap(i1, i2):
      (s1, e1), (s2, e2) = (i1, i2)
      return max(s1, s2) <= min(e1, e2)

    # Check in x-y (horizontal) plane
    cuboid_pts_h_hat = l2_normalized(cuboid.box3d[:, :2])
    camera_pov_h_hat = l2_normalized(self.principal_axis_in_ego[:2])
    theta_h = theta_signed(camera_pov_h_hat, cuboid_pts_h_hat)
    is_in_fov_h = intervals_overlap(
                    (-.5 * fov_h, .5 * fov_h),
                    (theta_h.min(), theta_h.max()))

    # Check in x-z (vertical) plane
    XZ = np.array([0, 2])
    cuboid_pts_v_hat = l2_normalized(cuboid.box3d[:, XZ])
    camera_pov_v_hat = l2_normalized(self.principal_axis_in_ego[XZ])
    theta_v = theta_signed(camera_pov_v_hat, cuboid_pts_v_hat)
    is_in_fov_v = intervals_overlap(
                    (-.5 * fov_v, .5 * fov_v),
                    (theta_v.min(), theta_v.max()))

    # if cuboid.track_id == 'df33e853-f5d1-4e49-b0c7-b5523cfe75cd':
    #   print('offscreen', is_in_fov_h, is_in_fov_v)
    #   print(cuboid.box3d)
    #   import pdb; pdb.set_trace()
    # elif cuboid.track_id == '79f92a80-93dc-442b-8cce-1c8da11fbe3b':
    #   print('ON', is_in_fov_h, is_in_fov_v)
    #   print(cuboid.box3d)
    #   import pdb; pdb.set_trace()
    # return True
    # if cuboid.track_id == 'nuscenes_instance_token:e91afa15647c4c4994f19aeb302c7179':
    #   import pdb; pdb.set_trace()
    return is_in_fov_h and is_in_fov_v

  def project_cuboid_to_bbox(self, cuboid):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    bbox = BBox(
            im_width=self.width,
            im_height=self.height,
            category_name=cuboid.category_name,
            au_category=cuboid.au_category,
            cuboid=cuboid)
    
    ## Fill Points
    centroid = np.mean(cuboid.box3d, axis=0)
    pts_in_cam = self.cam_from_ego.apply(cuboid.box3d).T
    bbox.cuboid_in_cam = pts_in_cam
    centroid_in_cam = self.cam_from_ego.apply(centroid[np.newaxis, :]).T

    # nope nope fixme ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Since the cuboid could be behind or alongside the camera, not all
    # of the cuboid faces may be visible.  If the object is very large,
    # perhaps only a single edge is visible.  To find the image-space 
    # 2-D axis-aligned bounding box that bounds all cuboid points, we find
    # the horizonal and vertical angles relative to the camera principal
    # axis (Z in the camera frame) that fits all cuboid points.  Then
    # if the object is partially out of view (or even behind the camera),
    # it is easy to clip the bounding box to the camera field of view.

    def l2_normalized(v):
      if len(v.shape) > 1:
        # Normalize row-wise
        return v / np.linalg.norm(v, axis=1)[:, np.newaxis]
      else:
        return v / np.linalg.norm(v)

    def to_0_2pi(thetas):
      return (thetas + 2 * math.pi) % 2 * math.pi

    def theta_signed(cam_h, cuboid_h):
      thetas = np.arctan2(np.cross(cam_h, cuboid_h), np.dot(cam_h, cuboid_h.T))
      return thetas
      # return to_0_2pi(thetas)

    Z_HAT = np.array([0, 1]) # Principal axis in X-Z and Y-Z planes
    pts_xz = pts_in_cam[:, (0, 2)]
    theta_h = theta_signed(l2_normalized(pts_xz), Z_HAT)
    pts_yz = pts_in_cam[:, (1, 2)]
    theta_v = theta_signed(l2_normalized(pts_yz), Z_HAT)

    # center_h = theta_signed(Z_HAT, l2_normalized(centroid[(0, 2)]))
    # center_v = theta_signed(Z_HAT, l2_normalized(centroid[(1, 2)]))

    f_x = self.K[0, 0]
    f_y = self.K[1, 1]
    c_x = self.K[0, 2]
    c_y = self.K[1, 2]
    fov_h, fov_v = self.get_fov()

    t_h_min, t_h_max = theta_h.min(), theta_h.max()
    t_v_min, t_v_max = theta_v.min(), theta_v.max()

    def to_pixel(theta, fov, length):
      half_fov = .5 * fov
      # p = np.clip(theta, -half_fov, half_fov) / half_fov
      p = theta / half_fov
      p = (p + 1) / 2
      return length * p

    x1 = to_pixel(t_h_min, fov_h, self.width)
    x2 = to_pixel(t_h_max, fov_h, self.width)
    y1 = to_pixel(t_v_min, fov_v, self.height)
    y2 = to_pixel(t_v_max, fov_v, self.height)

    focal_pixel_h = (.5 * self.width) / math.tan(fov_h * .5)
    focal_pixel_v = (.5 * self.height) / math.tan(fov_v * .5)

    uvd = self.K.dot(pts_in_cam.T)
    uvd[0:2, :] /= uvd[2, :]
    uvd = uvd.T

    centroid_uvd = self.K.dot(centroid_in_cam.T)
    centroid_uvd[0:2, :] /= centroid_uvd[2, :]
    centroid_uvd = centroid_uvd.T[0, :]

    # # import pdb; pdb.set_trace()
    # uvt_good = np.stack([
    #   np.sin(theta_h) * np.linalg.norm(pts_xz, axis=1) * focal_pixel_h,
    #   np.sin(theta_v) * np.linalg.norm(pts_yz, axis=1) * focal_pixel_v,
    #   uvd[:,2],
    # ]).T

    # def to_point(theta, dist, fov, focal_l, pts):
    #   # disp = (theta > 0) * dist
    #   p_prime = 2. * np.tan(np.abs(theta) * .5) * focal_l * pts[:,0]
    #   return p_prime / np.abs(pts[:,1]) + .5 * dist

    # uvt = np.stack([
    #   to_point(theta_h, self.width, fov_h, f_x, pts_xz),
    #   to_point(theta_v, self.height, fov_v, f_y, pts_yz),
    #   # np.sin(theta_h - fov_h * .5) * f_x + .5 * self.width+ self.width, #np.linalg.norm(pts_xz, axis=1) * focal_pixel_h,
    #   # np.sin(theta_v - fov_v * .5) * f_y + , #np.linalg.norm(pts_yz, axis=1) * focal_pixel_v,
    #   uvd[:,2],
    # ]).T
    
    
    
    # FIXME docs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Suppose the camera is side-by-side with a long pole, where part of the
    # pole extends in front of and part behind the camera. We need to project
    # points that are behind the camera to a place that makes sense
    # geometrically. As points in front of the camera get closer, they get 
    # projected to the infinite horizon beyond the left and right edges
    # of the image.
    # We choose to project these points as follows:
    #   First, pretend these points are actually in front of the camera, and
    #   compute the angle they make between their projection onto the principal
    #   plane and the focal center.
    #   Second, define the ray in the principal plane that has this angle. Now
    #   follow that ray off-screen to "pseudo-infinity" (based upon focal 
    #   length of the camera) to plot the final projected point.
    pts_xy = pts_in_cam[:, :2]
    theta_xy = np.arctan2(pts_xy[:, 1], pts_xy[:, 0])
    PSEUDO_INF = 1 / 0.001
    uvt = np.stack([
      np.cos(theta_xy) * f_x * PSEUDO_INF,
      np.sin(theta_xy) * f_y * PSEUDO_INF,
      uvd[:,2],
    ]).T

    for r in range(8):
      # if abs(theta_h[r]) > fov_h * .5 or abs(theta_v[r]) > fov_v * .5:
      if uvd[r, 2] <= 0:
        uvd[r, :] = uvt[r, :]
    # uvd = uvt

    # if cuboid.track_id == 'nuscenes_instance_token:df8a0ce6d79446369952166553ede088':
    #   import pdb; pdb.set_trace()


    # print('')
    # uvd = self.project_ego_to_image(cuboid.box3d, omit_offscreen=False)

    bbox.cuboid_pts = uvd
    bbox.cuboid_center = centroid_uvd


    # print('uvd')
    # print(uvd)
    # print()
    # if cuboid.track_id == 'nuscenes_instance_token:e91afa15647c4c4994f19aeb302c7179':
    #   import pdb; pdb.set_trace()

    x1, x2 = np.min(uvd[:, 0]), np.max(uvd[:, 0])
    y1, y2 = np.min(uvd[:, 1]), np.max(uvd[:, 1])
    bbox.set_x1_y1_x2_y2(x1, y1, x2, y2)

    z = float(np.max(uvd[:, 2]))
    num_onscreen = bbox.get_num_onscreen_corners()
    bbox.has_offscreen = ((z <= 0) or (num_onscreen < 4))

    # While none of the points or cuboid points may be onscreen, if the object
    # is very close to the camera then a single edge of the cuboid or bbox
    # may intersect the screen.  TODO: proper frustum clipping for objects
    # that are beyond FoV and yet very slightly in front of the image plane.
    bbox.is_visible = (z > 0 and self._has_edge_in_fov(cuboid))
      # bbox.overlaps_with(common.BBox.of_size(self.width, self.height)))

    bbox.clamp_to_screen()

    ## Fill Pose
    bbox.cuboid_from_cam = \
      cuboid.obj_from_ego.translation - self.cam_from_ego.translation

    cuboid_from_cam_hat = \
      bbox.cuboid_from_cam / np.linalg.norm(bbox.cuboid_from_cam)
    
    cuboid_from_cam_hat = cuboid_from_cam_hat.reshape(3)

    from scipy.spatial.transform import Rotation as R
    X_HAT = np.array([1, 0, 0])
    obj_normal = cuboid.obj_from_ego.rotation.dot(X_HAT)
    cos_theta = cuboid_from_cam_hat.dot(obj_normal.reshape(3))
    rot_axis = np.cross(cuboid_from_cam_hat, obj_normal)
    obj_from_ray = R.from_rotvec(
          math.acos(cos_theta) * rot_axis / np.linalg.norm(rot_axis))
    bbox.ypr_camera_local = obj_from_ray.as_euler('zxy')

    return bbox

  NO_RV_SMOOTHING = -1  
  @staticmethod
  def get_cloud_rv_simple(
          im_size,
          uvd,
          ptvs=None,
          depth_soft_horizon_meters=50):
    """Return a greyscale Pointcloud Range-View image given a set of points.
    Points with a value (e.g. depth) of 0 have color black and points with
    a value of 255 have color white. Optionally smooth cloud sparsity using
    markers that are scaled inversely with depth according to
    `depth_soft_horizon_meters`.

    Args:
      img_size: tuple of image size (height, width) in pixels.
      uvd: Array of n-by-3 points containing (pixel x, pixel y, depth meters)
        values for all cloud points.
      ptvs: Array of n values; use these values for plot intensity.  By
        default, use depth from `uvd` with tanh smoothing. The given values are
        clipped to [0, 1].
      depth_soft_horizon_meters: If non-negative, smooth cloud sparsity using
        markers scaled by the depth values in `uvd`. We use tanh smoothing
        to create larger markers for points at most `depth_soft_horizon_meters`
        away from the camera; points farther than this threshhold will obtain
        less smoothing.

    Returns:
      An `img_size`-sized 1-channel image.
    """
    
    im_h, im_w = im_size
    uvd = np.copy(uvd)

    if ptvs is None:
      if depth_soft_horizon_meters > 0:
        ptvs = np.tanh(uvd[:, 2] / depth_soft_horizon_meters)
      else:
        ptvs = uvd[:, 2] / 128.
    ptvs = np.clip(ptvs, 0, 1)

    # First, re-order points by depth ascending so that farther points (which
    # may be smaller and brighter) are drawn over nearer points (which might
    # have larger markers via smoothing)
    order = (uvd[:,2]).argsort()
    uvd = uvd[order]
    ptvs = ptvs[order]
    
    # Decide on marker sizes and colors
    def marker_size(depth):
      if depth_soft_horizon_meters <= 0:
        return 0.05 * im_h * im_w
      else:
        def unitized_depth(z):
          return np.tanh(z / depth_soft_horizon_meters)
        v = 1 - unitized_depth(depth)
        pt_scale = 0.5 * im_w * im_h
        s = pt_scale * v
        return s
    marker_sizes = [marker_size(z) for z in uvd[:,2]]
    
    # Convert to RGB; if we don't, then Matplotlib uses weird greyscale
    # mapping.
    colors = [(v, v, v) for v in ptvs]

    # We use matplotlib for image rendering, as it's faster and more flexible
    # than direct numpy.
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(figsize=(im_w, im_h), dpi=1)
    fig.set_facecolor((0, 0, 0))
    canvas = FigureCanvas(fig)
    
    ax = fig.gca()
    ax.scatter(
        uvd[:,0], im_h - 1 - uvd[:,1],   # x, y
        s=marker_sizes, c=colors)
    
    # Crop plot so that it has 1-to-1 pixel correspondence with the camera
    # image.
    ax.axis('off')
    ax.set_xlim(0, im_w)
    ax.set_ylim(0, im_h)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)

    # Render!
    canvas.draw()
    img_str, (width, height) = canvas.print_to_buffer()

    rv_img = np.fromstring(img_str, np.uint8).reshape((im_h, im_w, 4))

    # Release memory
    fig.clear()
    canvas.get_renderer().clear()

    return rv_img[:, :, :1] # Return greyscale

  @staticmethod
  def get_cloud_rv_delaunay_smoothing(
          im_size,
          uvd,
          cloud,
          principal_axis):
    """Return an RGB Range-View image given a set of N points projected
    into the camera frame.  Use delaunay triangle smoothing: compute
    a delaunay triangulation of the 2D (x, y) projection of the points onto
    the image plane, and use these triangles to interpolate depth values
    for pixels that don't have point returns.

    The returned image is RGB, but can be interpreted as follows at HSV:
     * Hue: encodes depth; we hash depth to a hue which changes in 10-meter
        buckets, and measurments between these buckets are hue-interpolated.
        See `plotting.rgb_for_distance()`.
     * Saturation: (unused)
     * Value: encodes the normal of the triangle relative to the camera 
        perspective.  Normals orthogonal to the camera perspective have
        low brightness ("value")

    TODO: Create delaunay-based mesh from pointsensor perspective and render
    in camera view using a raytracer like pyrender.  Important when the 
    pointsensor has a very different vantage point from the camera and
    returns points that are behind the camera PoV-- in these cases, our
    delaunay smoothing will improperly mesh together foreground and
    background points.

    Points with a value (e.g. depth) of 0 have color black and points with
    a value of 255 have color white. Optionally smooth cloud sparsity using
    markers that are scaled inversely with depth according to
    `depth_soft_horizon_meters`.

    Args:
      img_size: tuple of image size (height, width) in pixels.
      uvd: Array of n-by-3 points containing (pixel x, pixel y, depth meters)
        values for all cloud points.
      cloud: Array of n-by-3 points containing (x, y, z) values
        of all points.  Used to determine triangle normals.
      principal_axis: 3-vector representing the camera's perspective.
        Used to determine triangle normals.  Must be in same frame as `cloud`
        (e.g. both in ego frame).

    Returns:
      An `img_size`-sized RGB image.
    """

    uvd = np.copy(uvd)

    # We use matplotlib for image rendering, as it's faster and more flexible
    # than direct numpy.
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.colors import ListedColormap
    from matplotlib.colors import Normalize
    
    img_h, img_w = im_size

    ###
    ### Create Hue / Depth-Colored image
    ###
    figd = Figure(figsize=(img_w, img_h), dpi=1)
    figd.set_facecolor((0, 0, 0))
    canvasd = FigureCanvas(figd)
    axd = figd.gca()

    MAX_DEPTH_METERS = 1000
    from au.plotting import rgb_for_distance
    zcolors = [
      (np.array(rgb_for_distance(z)) / 255).tolist()
      for z in np.arange(0, MAX_DEPTH_METERS, 0.1)
    ]
    cmap = ListedColormap(zcolors)
    cmap.set_under("black")
    cmap.set_over("white")
    norm = Normalize(vmin=0, vmax=MAX_DEPTH_METERS)

    import matplotlib.tri as mtri
    triang = mtri.Triangulation(uvd[:,0], img_h - 1 - uvd[:,1])

    tri_vertices = list(map(lambda index: uvd[index], triang.triangles))

    def maptocolor(tri):
      zval = np.mean(tri[:,2])
      return zval
    axd.tripcolor(
        triang,
        facecolors=np.array([maptocolor(tri) for tri in tri_vertices]),
        cmap=cmap, norm=norm)

    axd.axis('off')
    axd.set_xlim(0, img_w)
    axd.set_ylim(0, img_h)
    figd.tight_layout()
    figd.subplots_adjust(bottom=0, top=1, left=0, right=1)

    canvasd.draw()
    img_str_d, (width, height) = canvasd.print_to_buffer()

    depth_image = np.fromstring(img_str_d, np.uint8).reshape((height, width, 4))
    
    ###
    ### Create Normals Overlay
    ###

    def maptocolor_norm(tri):
      dspan = abs(tri[:,3].max() - tri[:,3].min())
      xspan = abs(tri[:,0].max() - tri[:,0].min())
      yspan = abs(tri[:,1].max() - tri[:,1].min())
      zspan = abs(tri[:,2].max() - tri[:,2].min())
      if any(v > 5 for v in (xspan, yspan, zspan)):
        return float('inf')
      tri = tri[:,:3]
      tri_norm = np.cross(tri[0] - tri[1], tri[0] - tri[2])
      tri_norm /= np.linalg.norm(tri_norm)
      return 1. - abs(principal_axis.dot(tri_norm))


    fign = Figure(figsize=(img_w, img_h), dpi=1)
    fign.set_facecolor((0, 0, 0))
    canvasn = FigureCanvas(fign)
    axn = fign.gca()
    zcolors_max = 1
    zcolors_norm = [
      (1-(np.array([z, z, z]) / zcolors_max)).tolist()
      for z in np.arange(0, zcolors_max, 0.01)
    ]
    cmap_norm = ListedColormap(zcolors_norm)
    cmap_norm.set_under("white")
    cmap_norm.set_over("black")
    norm_n = Normalize(vmin=0, vmax=zcolors_max + 1)

    xyzd = np.concatenate([cloud[:, 0:3], uvd[:, 2:]], axis=-1)
    fused_vertices = list(map(lambda index: xyzd[index], triang.triangles))
    axn.tripcolor(
        triang,
        facecolors=np.array([maptocolor_norm(tri) for tri in fused_vertices]),
        cmap=cmap_norm, norm=norm_n)

    axn.axis('off')
    axn.set_xlim(0, img_w)
    axn.set_ylim(0, img_h)
    fign.tight_layout()
    fign.subplots_adjust(bottom=0, top=1, left=0, right=1)

    canvasn.draw()       # draw the canvas, cache the renderer
    img_str_n, (width, height) = canvasn.print_to_buffer()

    normals_image = np.fromstring(img_str_n, np.uint8)
    normals_image = normals_image.reshape((height, width, 4))
    
    final_image = (
      depth_image.astype(float) * (normals_image.astype(float) / 255))
    
    # Release memory
    fign.clear()
    canvasn.get_renderer().clear()
    figd.clear()
    canvasd.get_renderer().clear()
    
    return final_image.astype(np.uint8)[:, :, :3]


  ALL_RV_IMG_TYPES = (
    'depth_delaunay_smoothed',
    'depth_smoothed',
    'depth',
    'height_smoothed',
    'height',
  )
  def get_cloud_rv_images(self, img_types):
    img_types = img_types or []
    if not img_types:
      return {}
    
    img_out = np.zeros((self.height, self.width, 3), dtype=np.uint8)
    clouds = [pc.cloud for pc in self.clouds]
    if not clouds:
      return img_out
    fused_cloud = np.concatenate(clouds)

    # Project points to cam
    pts_in_cam = self.cam_from_ego.apply(fused_cloud).T
    uvd = self.K.dot(pts_in_cam.T)
    uvd[0:2, :] /= uvd[2, :]
    uvd = uvd.T

    # Only keep onscreen points
    uvd = uvd.T
    indices = np.where(
              np.logical_and.reduce((
                # Filter offscreen points
                0 <= uvd[0, :], uvd[0, :] < self.width - 1.0,
                0 <= uvd[1, :], uvd[1, :] < self.height - 1.0,
                # Filter behind-screen points
                uvd[2, :] > 0)))
    indices = indices[0]
    uvd = uvd[:, indices].T

    # Compute height data and normalize
    ego_z = fused_cloud[indices, 2]
    ego_z = np.tanh((ego_z + 1) / 5)

    # Save unfiltered ego-frame points
    unfiltered_fused_cloud = fused_cloud[indices, :]

    im_size = (self.height, self.width)
    NO_SMOOTH = CameraImage.NO_RV_SMOOTHING

    rv_images = {}
    if 'depth' in img_types:
      rv_images['depth'] = CameraImage.get_cloud_rv_simple(
                              im_size,
                              uvd,
                              depth_soft_horizon_meters=NO_SMOOTH)
    if 'depth_smoothed' in img_types:
      rv_images['depth_smoothed'] = CameraImage.get_cloud_rv_simple(
                                        im_size, uvd)
    
    if 'height' in img_types:
      rv_images['height'] = CameraImage.get_cloud_rv_simple(
                              im_size,
                              uvd,
                              ptvs=ego_z,
                              depth_soft_horizon_meters=NO_SMOOTH)
    if 'height_smoothed' in img_types:
      rv_images['height_smoothed'] = CameraImage.get_cloud_rv_simple(
                                        im_size, uvd, ptvs=ego_z)

    if 'depth_delaunay_smoothed' in img_types:
      rv_images['depth_delaunay_smoothed'] = (
        CameraImage.get_cloud_rv_delaunay_smoothing(
          im_size,
          uvd,
          unfiltered_fused_cloud,
          self.principal_axis_in_ego))

    return rv_images
    




    

    # PT_RADIUS_PIXELS = 10
    # if channel == 'depth':
    #   print('start depth')
    #   pts_in_cam = self.cam_from_ego.apply(fused_cloud).T
    #   uvd = self.K.dot(pts_in_cam.T)
    #   uvd[0:2, :] /= uvd[2, :]

    #   uvd_out = None
    #   MAX_DEPTH_METERS = 80 # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   D_STEP = 10
    #   r = PT_RADIUS_PIXELS
    #   for dlo in range(0, MAX_DEPTH_METERS, D_STEP):
    #     dhi = dlo + D_STEP
    #     indices = np.where(
    #                 np.logical_and.reduce((
    #                   dlo <= uvd[2, :], uvd[2, :] < dhi)))
    #     indices = indices[0]
    #     uvd_bucket = uvd[:, indices]
    #     for rx in range(-r, r+1):
    #       for ry in range(-r, r+1):
    #         added = uvd_bucket + np.array([[rx], [ry], [0]])
    #         if uvd_out is None:
    #           uvd_out = added
    #         else:
    #           uvd_out = np.concatenate([uvd_out, added], axis=1)
      
    #   # Only keep onscreen points
    #   indices = np.where(
    #               np.logical_and.reduce((
    #                 # Filter offscreen points
    #                 0 <= uvd_out[0, :], uvd_out[0, :] < self.width - 1.0,
    #                 0 <= uvd_out[1, :], uvd_out[1, :] < self.height - 1.0,
    #                 # Filter behind-screen points
    #                 uvd_out[2, :] > 0)))
    #   indices = indices[0]
    #   uvd_out = uvd_out[:, indices].T

    #   # map depth -> pixel color
    #   uvd_out[2, :] = np.clip(255 * (uvd_out[2, :] / MAX_DEPTH_METERS), 0, 255)

    #   uvd_out = uvd_out.T
    #   np.sort(uvd_out, axis=-1)
    #   idx = np.floor(uvd_out.T[:2,:].T).astype(int)
    #   img_out[idx[:,1],idx[:0]] = uvd_out[:,2][:,np.newaxis]

    #   # uvd = self.project_ego_to_image(fused_cloud, omit_offscreen=True)

    #   # # TODO try to use a faster np array assign:
    #   # # idx = np.floor(uvd.T[:2,:].T).astype(int)
    #   # # img_out[idx[:,1],idx[:0]] = uvd[:,2][:,np.newaxis]
    #   # for pt in uvd.tolist():
    #   #   u, v, d = pt
    #   #   if 0 <= v < self.height and 0 <= u < self.width:
    #   #     MAX_DEPTH_METERS = 80 # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   #     d = np.clip(255 * (float(d) / MAX_DEPTH_METERS), 0, 255)
    #   #     radius = int(np.clip((1. - (d / 255)) * PT_RADIUS_PIXELS, 2, PT_RADIUS_PIXELS))
    #   #     for rr in range(-radius, radius+1):
    #   #       for rc in range(-radius, radius + 1):
    #   #         r = int(v + rr); c = int(u + rc)
    #   #         if 0 <= r < self.height and 0 <= c < self.width:
    #   #           img_out[r, c, :] = max(img_out[r, c, :], d)
    # elif channel == 'height':
    #   cloud_z = fused_cloud[:, 2]
    #   pts_in_cam = self.cam_from_ego.apply(fused_cloud).T
    #   uvd = self.K.dot(pts_in_cam.T)
    #   uvd[0:2, :] /= uvd[2, :]

    #   # Only keep onscreen points
    #   indices = np.where(
    #               np.logical_and.reduce((
    #                 # Filter offscreen points
    #                 0 <= uvd[0, :], uvd[0, :] < self.width - 1.0,
    #                 0 <= uvd[1, :], uvd[1, :] < self.height - 1.0,
    #                 # Filter behind-screen points
    #                 uvd[2, :] > 0)))
    #   indices = indices[0]

    #   uvh = uvd[:, indices].T
    #   uvh[:, 2] = cloud_z[indices]

    #   for pt in uvh.tolist():
    #     u, v, h = pt
    #     if 0 <= v < self.height and 0 <= u < self.width:
    #       MAX_HEIGHT_METERS = 10 # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #       h = np.clip(255 * (float(h) / MAX_HEIGHT_METERS), 0, 255)
    #       # u = int(u)
    #       # v = int(v)
    #       # img_out[v, u, :] = max(img_out[v, u, :], h)
    #       radius = int(np.clip((1. - (h / 255)) * PT_RADIUS_PIXELS, 2, PT_RADIUS_PIXELS))
    #         # makes less sense ....
    #       for rr in range(-radius, radius+1):
    #         for rc in range(-radius, radius + 1):
    #           r = int(v + rr); c = int(u + rc)
    #           if 0 <= r < self.height and 0 <= c < self.width:
    #             img_out[r, c, :] = max(img_out[r, c, :], h)
    # else:
    #   ValueError(channel) # TODO: BEV mebbe
    
    # return img_out

  def to_html(self):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    import tabulate
    from au import plotting as aupl
    table = [
      [attr, to_preformatted(getattr(self, attr))]
      for attr in (
        'camera_name',
        'timestamp',
        'cam_from_ego',
        'K',
        'principal_axis_in_ego')
    ]
    html = tabulate.tabulate(table, tablefmt='html')

    image = self.image
    if util.np_truthy(image):
      table = [
        ['<b>Image</b>'],
        [aupl.img_to_img_tag(image, display_viewport_hw=(1000, 1000))],
      ]
      html += tabulate.tabulate(table, tablefmt='html')

    if self.clouds:
      debug_img = np.copy(self.image)
      for pc in self.clouds:
        cloud = self.project_ego_to_image(pc.cloud, omit_offscreen=True)
        aupl.draw_xy_depth_in_image(debug_img, cloud, alpha=0.7)
      table = [
        ['<b>Image With Clouds</b>'],
        [aupl.img_to_img_tag(debug_img, display_viewport_hw=(1000, 1000))],
      ]
      html += tabulate.tabulate(table, tablefmt='html')

      # ## HACKS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      rv_images = self.get_cloud_rv_images(self.ALL_RV_IMG_TYPES)
      for img_type, image in rv_images.items():
        image = image.astype(np.uint8)
        table = [
          ['<b>RV Image: %s</b>' % img_type],
          [aupl.img_to_img_tag(image, display_viewport_hw=(1000, 1000))],
        ]
        html += tabulate.tabulate(table, tablefmt='html')
    
    if self.bboxes:
      debug_img = np.copy(self.image)
      for bbox in self.bboxes:
        bbox.draw_in_image(debug_img)
      table = [
        ['<b>Image With Boxes</b>'],
        [aupl.img_to_img_tag(debug_img, display_viewport_hw=(1000, 1000))],
      ]
      html += tabulate.tabulate(table, tablefmt='html')

      html += '<br /><b>Boxes</b><br />'
      table = [
        [aupl.img_to_img_tag(
            bbox.get_crop(image),
            image_viewport_hw=(300, 300)),
         bbox.to_html() + '<br /><hr />']
        for bbox in self.bboxes
      ]
      html += tabulate.tabulate(table, tablefmt='html')

    return html


  def to_plotly_world_frame_3d(self, frustum_size_meters=0.1):
    
    corners = np.array([
      [0, 0],
      [self.width, 0],
      [self.width, self.height],
      [0, self.height],
    ])

    f_x = self.K[0, 0]
    f_y = self.K[1, 1]
    c_x = self.K[0, 2]
    c_y = self.K[1, 2]

    rays_xy_cam = (corners - np.array([c_x, c_y])) / np.array([f_x, f_y])
    rays_xyz_cam = np.hstack([rays_xy_cam, np.ones((4, 1))])
    rays_xyz_cam *= frustum_size_meters

    cam_pts_cam = np.vstack([rays_xyz_cam, np.array([0, 0, 0])])
    cam_frame_pts_cam = np.array([
      [1., 0., 0.], # x hat
      [0., 1., 0.], # y hat
      [0., 0., 1.], # z hat
    ])
    cam_frame_pts_cam *= frustum_size_meters

    T_ego_from_sensor = self.ego_to_sensor[self.sensor_name, 'ego']
    cam_pts_ego = T_ego_from_sensor.apply(cam_pts_cam).T
    cam_frame_pts_ego = T_ego_from_sensor.apply(cam_frame_pts_cam).T

    T_world_from_ego = self.ego_pose['ego', 'world']
    cam_pts_world = T_world_from_ego.apply(cam_pts_ego).T
    cam_frame_pts_world = T_world_from_ego.apply(cam_frame_pts_ego).T

    frustum_corners = [
      cam_pts_world[0, :],
      cam_pts_world[1, :],
      cam_pts_world[2, :],
      cam_pts_world[3, :],
    ]
    cam_frame_x_hat = cam_frame_pts_world[0, :]
    cam_frame_y_hat = cam_frame_pts_world[1, :]
    cam_frame_z_hat = cam_frame_pts_world[2, :]
    cam_center = cam_pts_world[-1, :]

    lines = []
    colors = []

    def make_line(pts):
      return [None] + [list(p) for p in (pts + [pts[0]])] + [None]
    def to_css_color(rgb):
      r, g, b = np.clip(rgb, 0, 255).astype(int).tolist()
      return 'rgb(%s,%s,%s)' % (r, g, b)
    def add_color(c, n):
      colors.extend(['rgb(0,0,0)'] + (n-2) * [to_css_color(c)] + ['rgb(0,0,0)'])

    # lines from cam center to frustum corners
    for corner in frustum_corners:
      l = make_line([cam_center, corner])
      lines.append(l)
      add_color((255, 255, 0), len(l))
    
    # lines around square of frustum
    for i in range(4):
      start = frustum_corners[i]
      end = frustum_corners[(i + 1) % 4]

      l = make_line([start, end])
      lines.append(l)
      add_color((125, 125, 0), len(l))
    
    # R, G, B lines to show cam x-hat, y-hat, z-hat
    l = make_line([cam_center, cam_frame_x_hat])
    lines.append(l)
    add_color((255, 0, 0), len(l))
    l = make_line([cam_center, cam_frame_y_hat])
    lines.append(l)
    add_color((0, 255, 0), len(l))
    l = make_line([cam_center, cam_frame_z_hat])
    lines.append(l)
    add_color((0, 0, 255), len(l))

    import plotly
    import plotly.graph_objects as go
    def to_line_vals(idx, lines):
      import itertools
      ipts = itertools.chain.from_iterable(lines)
      return [(pt[idx] if pt is not None else pt) for pt in ipts]
    lines_plot = go.Scatter3d(
                    name=str(self.timestamp),
                    x=to_line_vals(0, lines),
                    y=to_line_vals(1, lines),
                    z=to_line_vals(2, lines),
                    mode='lines',
                    line=dict(width=3, color=colors))

    return lines_plot

  def to_trimeshes_world_frame(
         self,
         frustum_meters=.1,
         include_thumnail=True,
         thumb_height_pixels=128,
         thumb_height_meters=0.1,
         thumb_offset_meters=0.1,
         thumb_thickness_meters=0.001,
         thumb_alpha=0.75):
    
    import io
    import trimesh
    import shapely # Transitive requirement for frustum marker

    T_ego_from_sensor = self.ego_to_sensor[self.sensor_name, 'ego']
    T_world_from_ego = self.ego_pose['ego', 'world']
    
    # errrr wait 
    w2c = T_world_from_ego @ T_ego_from_sensor
    w2c = w2c.get_transformation_matrix(homogeneous=True)

    meshes = []

    # Create camera marker
    fov_h, fov_v = self.get_fov()
    fov_h_deg = fov_h * 180. / math.pi
    fov_v_deg = fov_v * 180. / math.pi
    cam = trimesh.creation.camera_marker(
                  trimesh.scene.Camera(
                      fov=(fov_h_deg, fov_v_deg)),
                  marker_height=frustum_meters) # Actually also frustum depth
    # cam[1].colors = [[.5, .5, .5, 1.]] * 5
      # Actually frustum color

    for m in cam:
      m.apply_transform(w2c)
      meshes.append(m)

    if include_thumnail:
      # Create the thumbnail image and texture
      from PIL import Image
      import imageio
      import cv2

      debug = self.image
      debug = debug.astype('uint8')
      
      h = thumb_height_pixels
      aspect = debug.shape[1] / debug.shape[0]
      w = int(aspect * h)

      debug = cv2.resize(debug, (w, h))
      
      buf = io.BytesIO()
      imageio.imwrite(buf, debug, format='jpg', quality=75)
      buf.seek(0)
      pil_img = Image.open(buf)
      # thumb_material = trimesh.visual.material.PBRMaterial(
      #               baseColorTexture=pil_img.copy(),
      #               baseColorFactor=[1.0, 1.0, 1.0, thumb_alpha],
      #               alphaMode="BLEND",
      #               doubleSided=True,
      #               alphaCutoff=0.01)
      thumb_material = trimesh.visual.material.SimpleMaterial(
                    image=pil_img.copy(),
                    # baseColorFactor=[1.0, 1.0, 1.0, thumb_alpha],
                    # alphaMode="BLEND",
                    doubleSided=True)
                    # ,
                    # alphaCutoff=0.01)
      
      # Create thumnail mesh
      thumb_h = thumb_height_meters
      thumb_w = aspect * thumb_h
      thumb_RT = np.eye(4, 4)
      thumb_RT[2, 3] -= thumb_offset_meters
      thumb_mesh = trimesh.creation.box(
                      extents=[thumb_w, thumb_h, thumb_thickness_meters],
                      transform=thumb_RT)
      thumb_mesh.visual.face_colors = (1., 1., 1.)

      thumb_mesh.visual.material = thumb_material
      thumb_mesh.visual.uv = np.zeros((len(thumb_mesh.vertices), 2))

      # -z face
      thumb_mesh.visual.uv[0] = [0, 1] # include a ud flip
      thumb_mesh.visual.uv[4] = [1, 1]
      thumb_mesh.visual.uv[2] = [0, 0]
      thumb_mesh.visual.uv[6] = [1, 0]

      # +z face
      thumb_mesh.visual.uv[1] = [0, 1] # include a ud flip
      thumb_mesh.visual.uv[5] = [1, 1]
      thumb_mesh.visual.uv[3] = [0, 0]
      thumb_mesh.visual.uv[7] = [1, 0]

      thumb_mesh.apply_transform(w2c)
      meshes.append(thumb_mesh)
  
    return meshes


# show_html(html)

# import numpy as np
# import plotly.graph_objects as go
# import skimage.io as sio

# x = np.linspace(-2,2, 128)
# x, z = np.meshgrid(x,x)
# y = np.sin(x**2*z)

# fig = go.Figure(go.Surface(x=x, y=y, z=z,
#                            colorscale='RdBu', 
#                            showscale=False))
# image = sio.imread ("https://raw.githubusercontent.com/empet/Discrete-Arnold-map/master/Images/cat-128.jpg") 
# print(image.shape)
# img = imag[:,:, 1] 
# Y = 0.5 * np.ones(y.shape)
# fig.add_surface(x=x, y=Y, z=z, 
#                 surfacecolor=np.flipud(img), 
#                 colorscale='matter_r', 
#                 showscale=False)
# fig.update_layout(width=600, height=600, 
#                   scene_camera_eye_z=0.6, 
#                   scene_aspectratio=dict(x=0.9, y=1, z=1));
# fig.show()


# @attr.s(slots=True, eq=False, weakref_slot=False)
# class CameraVideo(object):
#   """A video file event from a camera; the camera could be calibrated.  The video
#   might also have a depth channel (i.e. RGB-D video)."""

#   sensor_name = attr.ib(type=str, default='')
#   """str: Name of the camera, e.g. camera_front"""

#   video_bytes = attr.ib(type=bytearray, default=bytearray())
#   """bytearray: Buffer of video data (rare; use `imageio` to sniff for
#   video type)"""

#   video_uri = attr.ib(type=str, default='')

#   iter_image_factory = attr.ib(
#       type=CloudpickeledCallable,
#       converter=CloudpickeledCallable,
#       default=None)
#   """CloudpickeledCallable: A serializable factory function that emits a
#   stream of HWC numpy array images"""

#   width = attr.ib(type=int, default=0, validator=None)
#   """int: Width of images in pixels"""

#   height = attr.ib(type=int, default=0, validator=None)
#   """int: Height of images in pixels"""

#   start_timestamp = attr.ib(type=int, default=0)
#   """int: Timestamp associated with the start of this video; typically a Unix stamp in
#   nanoseconds."""

#   end_timestamp = attr.ib(type=int, default=0)
#   """int: Timestamp associated with the start of this video; typically a Unix stamp in
#   nanoseconds."""

#   K = attr.ib(type=np.ndarray, default=np.eye(3, 3))
#   """numpy.ndarray: The 3x3 intrinsic calibration camera matrix"""

#   distortion_model = attr.ib(type=str, default="")
#   """str: Optional distortion model, e.g. OPENCV"""

#   distortion_kv = attr.ib(default={}, type=typing.Dict[str, float])
#   """Dict[str, float]: A map of distortion parameter name -> distortion paramte
#   value.  E.g. for OPENCV there might be entries for k1, k2, p1, p2."""

#   channel_names = attr.ib(default=['r', 'g', 'b'])
#   """List[str]: Semantic names for the channels (or dimensions / attributes)
#   of the image. By default, the `image` member uses `imageio` to read an
#   3-channel RGB image as a HWC array.  (Some PNGs could use an alpha channel
#   to produce an RGBA image).  In the case of depth images, one of the channels
#   (usually the first) decodes as depth in meters."""

#   extra = attr.ib(default={}, type=typing.Dict[str, str])
#   """Dict[str, str]: A map for adhoc extra context"""





# @attr.s(slots=True, eq=False, weakref_slot=False)
# class CameraVideoEvent(object):
#   """A video file event from a camera; the camera could be calibrated.  The video
#   might also have a depth channel (i.e. RGB-D video).  A segment may have
#   *two* datums for video: one at the start and one at the end."""

#   is_start = attr.ib(type=bool, default=True)
#   """bool: This datum denotes the start of a video sequence; there might be
#   a separate datum at """

