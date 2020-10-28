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

import attr
import numpy as np

from psegs import util
from psegs import datum
from psegs.conf import C
from psegs.datasets.idsutil import IDatasetUtil
from psegs.util import misc


###############################################################################
### KITTI-360 Fixtures & Other Constants

class Fixtures(object):

  ROOT = C.EXT_DATA_ROOT / 'kitti-360'

  @classmethod
  def filepath(cls, rpath):
    return cls.ROOT / rpath


###############################################################################
### KITTI Parsing Utils

@attr.s(eq=False)
class Calibration(object):


  ### Camera Intrinsics (Rectified)

  # NB: We ignore the grey cameras (numbered 0 and 1) because the Benchmarks
  # do not contain images for them.

  P2 = attr.ib(type=np.ndarray, default=np.zeros((3, 4)))
  """3-by-4 Projective Matrix for Camera 2 (left color stereo)"""

  P3 = attr.ib(type=np.ndarray, default=np.zeros((3, 4)))
  """3-by-4 Projective Matrix for Camera 3 (right color stereo)"""


  ## Derived Attributes

  K2 = attr.ib(type=np.ndarray, default=np.zeros((3, 3)))
  """3-by-3 Camera Matrix for Camera 2 (left color stero).
  Derived from `P2`"""

  K3 = attr.ib(type=np.ndarray, default=np.zeros((3, 3)))
  """3-by-3 Camera Matrix for Camera 3 (right color stero).
  Derived from `P3`"""

  T2 = attr.ib(type=np.ndarray, default=np.zeros((1, 3)))
  """3-by-1 Translation vector from Camera 2 center from Lidar frame.
  We estimate this vector from `P2`.  See `velo_to_cam_2_rect` below."""

  T3 = attr.ib(type=np.ndarray, default=np.zeros((1, 3)))
  """3-by-1 Translation vector from Camera 3 center from Lidar frame.
  We estimate this vector from `P3`.  See `velo_to_cam_3_rect` below."""


  ### Raw Extrinsics

  R0_rect = attr.ib(type=datum.Transform, default=datum.Transform())
  """A rotation-only transform for projecting lidar points into the *rectified*
  camera frame.  Neglecting this transform will result in a skew between
  projected points and the center of rectified images.  Called `R0_rect` in
  Benchmark calibration data."""

  velo_to_cam_unrectified = attr.ib(
    type=datum.Transform, default=datum.Transform())
  """Raw transform from velodye to left color camera (camera 2) unrectified
  frame.  Called `Tr_velo_to_cam` in Benchmark calibration data."""


  sick_to_velo = attr.ib(type=datum.Transform, default=datum.Transform())
  """Raw transform from SICK laser frame to velodyne frame."""

  cam_left_raw_to_ego = attr.ib(type=datum.Transform, default=datum.Transform())
  cam_right_raw_to_ego = attr.ib(type=datum.Transform, default=datum.Transform())
  cam_left_fisheye_to_ego = attr.ib(type=datum.Transform, default=datum.Transform())
  cam_right_fisheye_to_ego = attr.ib(type=datum.Transform, default=datum.Transform())

  cam0_K = attr.ib(type=np.ndarray, default=np.zeros((3, 3)))
  cam1_K = attr.ib(type=np.ndarray, default=np.zeros((3, 3)))

  cam_left_raw_to_velo = attr.ib(
    type=datum.Transform, default=datum.Transform())

  RT_01 = attr.ib(
    type=datum.Transform, default=datum.Transform())

  ### Derived Extrinsics

  velo_to_cam_2_rect = attr.ib(type=datum.Transform, default=datum.Transform())
  """Transform from velodyne to left color camera rectified frame.  Use this
  transform with PSegs versus `velo_to_cam_unrectified`.
  
  In PSegs, we project points from lidar to camera using:
    pxpyd = K * [R|T] * xyz
  where uvd is pxpyd is a pixel (x, y, depth) value, K is the camera matrix,
  and [R|T] transforms from lidar to camera frame. However, KITTI only provides
  the projective matrix P and a transform [R|T] to the **left** camera frame.
  KITTI says to project points using:
    pxpyd = P * R0 * Tr_velo_to_cam * xyz
  We pick apart K and [R|T] from P for each camera for compatibility with
  PSegs.
  """

  velo_to_cam_3_rect = attr.ib(type=datum.Transform, default=datum.Transform())
  """Transform from velodyne to right color camera rectified frame.  Use this
  transform with PSegs versus `velo_to_cam_unrectified`."""


  def __eq__(self, other):
    return misc.attrs_eq(self, other)

  @classmethod
  def from_kitti_360_strs(
        cls,
        calib_cam_to_pose,
        calib_cam_to_velo,
        calib_sick_to_velo,
        perspective):
    """Create and return a `Calibration` instance from calibration data
    included in KITTI-360.  Each argument is a string with the contents
    of the file with the same name; FMI see 
    http://www.cvlibs.net/datasets/kitti-360/documentation.php
    """
    
    def str_to_arr(s, shape):
      from io import StringIO
      a = np.loadtxt(StringIO(s.strip()))
      return a.reshape(shape)
    
    def str_to_RT(s):
      return str_to_arr(s, shape=(3, 4))

    kwargs = {}

    ## Extrinsics

    kwargs['sick_to_velo'] = datum.Transform.from_transformation_matrix(
              str_to_RT(calib_sick_to_velo),
              src_frame='laser',
              dest_frame='lidar')

    cam_left_raw_to_velo = datum.Transform.from_transformation_matrix(
              str_to_RT(calib_cam_to_velo),
              src_frame='camera|left_raw',
              dest_frame='lidar')

    kwargs['cam_left_raw_to_velo'] = cam_left_raw_to_velo

    # Tr cam -> ego
    lines = [l.strip() for l in calib_cam_to_pose.split('\n')]
    cam_to_sRT = dict(l.split(':') for l in lines if l)
    kwargs['cam_left_raw_to_ego'] = datum.Transform.from_transformation_matrix(
              str_to_RT(cam_to_sRT['image_00']),
              src_frame='camera|left_raw',
              dest_frame='ego')
    kwargs['cam_right_raw_to_ego'] = datum.Transform.from_transformation_matrix(
              str_to_RT(cam_to_sRT['image_01']),
              src_frame='camera|right_raw',
              dest_frame='ego')
    kwargs['cam_left_fisheye_to_ego'] = datum.Transform.from_transformation_matrix(
              str_to_RT(cam_to_sRT['image_02']),
              src_frame='camera|left_fisheye',
              dest_frame='ego')
    kwargs['cam_right_fisheye_to_ego'] = datum.Transform.from_transformation_matrix(
              str_to_RT(cam_to_sRT['image_03']),
              src_frame='camera|left_fisheye',
              dest_frame='ego')
    

    ## Intrinsics

    # https://github.com/autonomousvision/kitti360Scripts/blob/081c08b34a14960611f459f23a0ad049542205c6/kitti360scripts/helpers/project.py#L76

    lines = [
      l.strip() for l in perspective.split('\n') if 'calib_time' not in l
    ]
    perspective_kv = dict(l.split(':') for l in lines if l)
    K_cam_left_rect = str_to_arr(perspective_kv['P_rect_00'], (3, 4))[:3, :3]
    K_cam_right_rect = str_to_arr(perspective_kv['P_rect_01'], (3, 4))[:3, :3]

    # we dont know what K vs P is yet :(

    kwargs['cam0_K'] = K_cam_left_rect
    kwargs['cam1_K'] = K_cam_right_rect

    # Transform looks a little off...
    kwargs['RT_01'] = datum.Transform(
              rotation=str_to_arr(perspective_kv['R_01'], (3, 3)),
              translation=str_to_arr(perspective_kv['T_01'], (3, 1)),
              src_frame='camera|left_raw',
              dest_frame='camera|right_raw')




    # # Parse raw data. Based upon pykitti.  We don't use pykitt directly due to
    # # its dependency issues and the way it confounds files objecs with parsing
    # # code and data structures.
    # # https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
    # lines = [l for l in calib_txt.split('\n') if l]
    # data = {}
    # for line in lines:
    #   # P0: 7.115377000000e+02 0.000000000000e+00 -> P0: np.array([...])
    #   # OR P0 7.115377000000e+02 0.000000000000e+00 -> P0: np.array([...])
    #   toks = [t for t in line.split(' ') if t]
    #   k = toks[0]
    #   if ':' in k:
    #     k = k.replace(':', '')
    #   # k, v = line.split(':', 1) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   # data[k] = np.array([float(vv) for vv in v.split()])
    #   data[k] = np.array([float(t) for t in toks[1:]])

    # kwargs = {}
    
    # # Load camera projective matrices
    # CAMERAS = ('P2', 'P3') # Ignore grey cameras!
    # for cam in CAMERAS:
    #   kwargs[cam] = np.reshape(data[cam], (3, 4))

    # ## Decide on keys
    # # Object and Tracking use different keys.  Default to Object, fall back
    # # to Tracking.
    # R0_rect_key = 'R0_rect'
    # if R0_rect_key not in data:
    #   R0_rect_key = 'R_rect' # Tracking

    # Tr_velo_to_cam_key = 'Tr_velo_to_cam'
    # if Tr_velo_to_cam_key not in data:
    #   Tr_velo_to_cam_key = 'Tr_velo_cam' # Tracking
    
    # Tr_imu_to_velo_key = 'Tr_imu_to_velo'
    # if Tr_imu_to_velo_key not in data:
    #   Tr_imu_to_velo_key = 'Tr_imu_velo'

    # ## Load extrinsics
    # kwargs['R0_rect'] = datum.Transform(
    #                       rotation=np.reshape(data[R0_rect_key], (3, 3)),
    #                       src_frame='camera|left_raw',
    #                       dest_frame='camera|left_sensor')

    # kwargs['velo_to_cam_unrectified'] = (
    #   datum.Transform.from_transformation_matrix(
    #     np.reshape(data[Tr_velo_to_cam_key], (3, 4)),
    #     src_frame='lidar', dest_frame='camera|left_grey_raw'))
    
    # kwargs['imu_to_velo'] = datum.Transform.from_transformation_matrix(
    #   np.reshape(data[Tr_imu_to_velo_key], (3, 4)),
    #   src_frame='oxts', dest_frame='lidar')
    
    return cls(**kwargs)


###############################################################################
### IDatasetUtil Impl

class DSUtil(IDatasetUtil):

  FIXTURES = Fixtures

  REQUIRED_DIRS = (
    'calibration',
    'data_2d_raw',
    'data_3d_raw',
    'data_3d_semantics',
    'data_3d_bboxes',
    'data_poses',
  )

  OPTIONAL_DIRS = (
    'data_2d_semantics',
  )

  @classmethod
  def emplace(cls):
    DIRS_REQUIRED = set(cls.FIXTURES.filepath(d) for d in cls.REQUIRED_DIRS)
    has_all_req = all(p.exists() for p in DIRS_REQUIRED)
    if not has_all_req:
      req = '\n        '.join('  * %s' % fname for fname in cls.all_zips())
      opt = '\n        '.join('  * %s' % fname for fname in cls.all_zips())
      cls.show_md("""
        Due to KITTI-360 license constraints, you need to manually accept the
        KITTI-360 license and download the files at
        [the KITTI-360 website](http://www.cvlibs.net/datasets/kitti-360/download.php).
        
        The KITTI-360 team provides download scripts that will help unzip
        files into place.  The total dataset is about 650GB unzipped
        (spinning disk OK).

        Required KITTI-360 data dirs:

        %s

        Optioanl KITTI-360 data dirs:

        %s
        """ % (req, opt))
      
      kitti_root = input(
        "Please enter the directory containing your KITTI zip archives; "
        "PSegs will create a (read-only) symlink to them: ")
      kitti_root = Path(kitti_root.strip())
      assert kitti_root.exists()
      assert kitti_root.is_dir()

      from oarphpy import util as oputil
      oputil.mkdir(str(cls.FIXTURES.ROOT.parent))

      cls.show_md("Symlink: \n%s <- %s" % (kitti_root, cls.FIXTURES.ROOT))
      os.symlink(kitti_root, cls.FIXTURES.ROOT)

      # Make symlink read-only
      import stat
      os.chmod(
        kitti_root,
        stat.S_IREAD|stat.S_IRGRP|stat.S_IROTH,
        follow_symlinks=False)

    cls.show_md("Validating KITTI archives ...")
    dirs_needed = set(cls.all_zips())
    dirs_have = set()
    for entry in cls.FIXTURES.ROOT.iterdir():
      if entry.name in cls.REQUIRED_DIRS:
        dirs_needed.remove(entry.name)
        dirs_have.add(entry.name)
    
    if dirs_needed:
      s_have = \
        '\n        '.join('  * %s' % fname for fname in dirs_have)
      s_needed = \
        '\n        '.join('  * %s' % fname for fname in dirs_needed)
      cls.show_md("""
        Missing some expected data dirs!

        Found:
        
        %s

        Missing:

        %s
      """ % (s_have, s_needed))
      return False
    
    cls.show_md("... all KITTI-360 data found!")
    return True
