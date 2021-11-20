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

import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import attr
import numpy as np
from oarphpy import util as oputil

from psegs import util
from psegs import datum
from psegs.conf import C
from psegs.datasets.idsutil import IDatasetUtil
from psegs.datum.transform import Transform
from psegs.table.sd_table import StampedDatumTableBase
from psegs.util import misc



###############################################################################
### KITTI Fixtures & Other Constants

class Fixtures(object):

  ROOT = C.EXT_DATA_ROOT / 'kitti_archives'

  OBJECT_BENCHMARK_FNAMES = (
    'data_object_label_2.zip',
    'data_object_image_2.zip',
    'data_object_image_3.zip',
    'data_object_prev_2.zip',
    'data_object_prev_3.zip',
    'data_object_velodyne.zip',
    'data_object_calib.zip',
  )

  TRACKING_BENCHMARK_FNAMES = (
    'data_tracking_label_2.zip',
    'data_tracking_image_2.zip',
    'data_tracking_image_3.zip',
    'data_tracking_velodyne.zip',
    'data_tracking_oxts.zip',
    'data_tracking_calib.zip',
  )

  @classmethod
  def zip_path(cls, zipname):
    return cls.ROOT / zipname


  ### Extension Data ##########################################################
  ### See https://github.com/pwais/psegs-kitti-ext

  EXT_DATA_ROOT = C.EXT_DATA_ROOT / 'psegs-kitti-ext'

  @classmethod
  def bench_to_raw_path(cls):
    return cls.EXT_DATA_ROOT / 'bench_to_raw_df'

  @classmethod
  def index_root(cls):
    """A r/w place to cache any temp / index data"""
    return C.PS_TEMP / 'kitti'


  ### Testing #################################################################

  TEST_FIXTURES_ROOT = Path('/tmp/psegs_kitti_test_fixtures')

  EXTERNAL_FIXTURES_ROOT = C.EXTERNAL_TEST_FIXTURES_ROOT / 'kitti'

  OBJ_TEST_FRAMES= ('002480', '002481', '002482')

  @classmethod
  def object_fixture_dir(cls):
    fixture_dir = cls.TEST_FIXTURES_ROOT / 'object'
    if util.missing_or_empty(fixture_dir):
      util.log.info(
        "Putting Object Benchmark test fixtures in %s" % fixture_dir)
      oputil.cleandir(fixture_dir)
      
      ## Extract all data for these frames
      util.unarchive_entries(
        cls.zip_path('data_object_image_2.zip'),
        ['training/image_2/%s.png' % f for f in cls.OBJ_TEST_FRAMES],
        fixture_dir)
      util.unarchive_entries(
        cls.zip_path('data_object_image_3.zip'),
        ['training/image_3/%s.png' % f for f in cls.OBJ_TEST_FRAMES],
        fixture_dir)
      util.unarchive_entries(
        cls.zip_path('data_object_velodyne.zip'),
        ['training/velodyne/%s.bin' % f for f in cls.OBJ_TEST_FRAMES],
        fixture_dir)
      util.unarchive_entries(
        cls.zip_path('data_object_calib.zip'),
        ['training/calib/%s.txt' % f for f in cls.OBJ_TEST_FRAMES],
        fixture_dir)
      util.unarchive_entries(
        cls.zip_path('data_object_label_2.zip'),
        ['training/label_2/%s.txt' % f for f in cls.OBJ_TEST_FRAMES],
        fixture_dir)
    
    return fixture_dir
  

  TRACKING_TEST_FRAMES = (
    '0009/000214',
    '0009/000215',
    '0015/000017',
    '0019/001055',
  )

  @classmethod
  def tracking_fixture_dir(cls):
    fixture_dir = cls.TEST_FIXTURES_ROOT / 'tracking'
    if util.missing_or_empty(fixture_dir):
      util.log.info(
        "Putting Tracking Benchmark test fixtures in %s" % fixture_dir)
      oputil.cleandir(fixture_dir)
      
      ## Extract all data for these frames
      util.unarchive_entries(
        cls.zip_path('data_tracking_image_2.zip'),
        ['training/image_02/%s.png' % f for f in cls.TRACKING_TEST_FRAMES],
        fixture_dir)
      util.unarchive_entries(
        cls.zip_path('data_tracking_image_3.zip'),
        ['training/image_03/%s.png' % f for f in cls.TRACKING_TEST_FRAMES],
        fixture_dir)
      util.unarchive_entries(
        cls.zip_path('data_tracking_velodyne.zip'),
        ['training/velodyne/%s.bin' % f for f in cls.TRACKING_TEST_FRAMES],
        fixture_dir)
      
      segs = [f.split('/')[0] for f in cls.TRACKING_TEST_FRAMES]
      util.unarchive_entries(
        cls.zip_path('data_tracking_calib.zip'),
        ['training/calib/%s.txt' % seg for seg in segs],
        fixture_dir)
      util.unarchive_entries(
        cls.zip_path('data_tracking_label_2.zip'),
        ['training/label_02/%s.txt' % seg for seg in segs],
        fixture_dir)
    
    return fixture_dir
  

  ### DSUtil Auto-download ####################################################

  @classmethod
  def maybe_emplace_psegs_kitti_ext(cls):
    if (cls.bench_to_raw_path().exists() and 
          cls.EXTERNAL_FIXTURES_ROOT.exists()):
      return
    
    from oarphpy import util as oputil
    util.log.info("Downloading latest PSegs KITTI Extension data ...")
    oputil.mkdir(str(cls.index_root()))
    psegs_kitti_ext_root = cls.index_root() / 'psegs_kitti_ext_tmp'
    if not psegs_kitti_ext_root.exists():
      oputil.run_cmd(
        "git clone https://github.com/pwais/psegs-kitti-ext %s" % \
          psegs_kitti_ext_root)

    util.log.info("... emplacing PSegs KITTI Extension data ...")
    def move(src, dest):
      oputil.mkdir(dest.parent)
      oputil.run_cmd("mv %s %s" % (src, dest))
    
    move(
      psegs_kitti_ext_root / 'assets' / 'bench_to_raw_df',
      cls.bench_to_raw_path())
    move(
      psegs_kitti_ext_root / 'ps_external_test_fixtures',
      cls.EXTERNAL_FIXTURES_ROOT)
    
    util.log.info("... emplace success!")
    util.log.info("(You can remove %s if needed)" % psegs_kitti_ext_root)



###############################################################################
### KITTI Parsing Utils


def load_transforms_from_oxts(oxts_str):
  """Parse Tracking Benchmark oxts files and return ego-to-world transforms.
  We ignore most of the oxts information.

  Based upon `pykitti <https://github.com/utiasSTARS/pykitti/blob/19d29b665ac4787a10306bbbbf8831181b38eb38/pykitti/utils.py#L107>`_

  See Also:
    * `KITTI OXTS Docs <https://github.com/pratikac/kitti/blob/eba7ba0f36917f72055060e9e59f344b72456cb9/readme.raw.txt#L105>`_

  Args:
    oxts_str (str): The string contents of a single Tracking Benchmark oxts
      file. These are in `data_tracking_oxts.zip`.
  
  Returns:
    Dict[int, :class:`~psegs.datum.transform.Transform`]: A map of frame
      number to the ego-to-world transform of the car at that frame.
  """
  from scipy.spatial.transform import Rotation as R

  lines = [l for l in oxts_str.split('\n') if l]
  
  EARTH_RADIUS_METERS = 6378137.
  scale = None

  frame_to_xform = {}
  for frame_num, line in enumerate(lines):
    toks = line.split(' ')
    lat = float(toks[0])
    lon = float(toks[1])
    alt = float(toks[2])
    roll = float(toks[3])
    pitch = float(toks[4])
    yaw = float(toks[5])

    if scale is None:
      scale = math.cos(lat * math.pi / 180)
    
    # Mercator projection
    tx = scale * lon * math.pi * EARTH_RADIUS_METERS / 180
    ty = scale * EARTH_RADIUS_METERS * (
            math.log(math.tan((90 + lat) * math.pi / 360)))
    tz = alt

    # TODO are these correct ? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    rot = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

    frame_to_xform[frame_num] = datum.Transform(
                                        rotation=rot,
                                        translation=[tx, ty, tz],
                                        src_frame='world',
                                        dest_frame='oxts')
  return frame_to_xform


def parse_tracking_label_cuboids(label_str):
  """Parse Tracking Benchmark labels for **an entire Tracking sequence**
  and mapping of frame number to lists of `Cuboid` and `BBox2d` instances.

  The label format for the Tracking Benchmark is identical to that for
  the Object Benchmark except that each line of a Tracking label string
  is prefixed with the following two values:
   * frame: An integer starting from 0 indicating the frame number;
      each frame has synchronized lidar and camera images.
   * track_id: An id distinct to the tracked object in the sequence

  See `parse_object_label_cuboids()` below.

  Args:
    label_str (str): The string contents of an Tracking Benchmark label file.

  Returns:
    Dict[int, List[:class:`~psegs.datum.cuboid.Cuboid`]]: A map of frame id
      to labels decoded as cuboids
    Dict[int, List[:class:`~psegs.datum.bbox2d.BBox3D`]]: A map of frame id
      to labels decoded as bboxes
  """

  lines = [l for l in label_str.split('\n') if l]
  frame_to_cuboids = defaultdict(list)
  frame_to_bboxes = defaultdict(list)
  for line in lines:
    toks = line.split(' ')
    frame_num = int(toks[0])
    track_id = str(toks[1])
    cuboids, bboxes = parse_object_label_cuboids(' '.join(toks[2:]))
    extra = {
      'kitti.track_id': str(track_id),
      'kitti.frame_num': str(frame_num),
    }
    for c in cuboids:
      c.track_id = track_id
      c.extra.update(**extra)
    for b in bboxes:
      b.extra.update(**extra)
    frame_to_cuboids[frame_num].extend(cuboids)
    frame_to_bboxes[frame_num].extend(bboxes)
  return frame_to_cuboids, frame_to_bboxes


def parse_object_label_cuboids(label_str):
  """Parse Object Benchmark labels and return a list of `Cuboid` and `BBox2d`
  instances.

  Notes:
    Due to KITTI label format and the unavailability of calibration in this
    helper, the `Cuboid` instance returned has `obj_from_ego` from the
    **camera** frame, and not the ego / lidar frame.  Furthermore,
    the `length_meters`, `width_meters`, and `height_meters` attributes are
    assigned for camera frame semantics.

  See also:
    * The `KITTI robot frame reference <http://www.cvlibs.net/datasets/kitti/setup.php>`_
    * `Label description <https://github.com/bostondiditeam/kitti/blob/71d51b8a66c9226369797d437315c3ca2b56f312/resources/devkit_object/readme.txt#L31>`_
    * `The KITTI Object Benchmark devkit <https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip>`_
    * `Google Lingvo parsing code <https://github.com/tensorflow/lingvo/blob/96eaa85c648c45585ca76493bba5991212bac38a/lingvo/tasks/car/tools/kitti_data.py#L44>`_
    * `SECOND (PointPillars) <https://github.com/traveller59/second.pytorch/blob/e42e4a0e17262ab7d180ee96a0a36427f2c20a44/second/data/kitti_dataset.py#L38>`_

  Args:
    label_str (str): The string contents of an Object Benchmark label file.

  Returns:
    List[:class:`~psegs.datum.cuboid.Cuboid`]: labels decoded as cuboids
    List[:class:`~psegs.datum.bbox2d.BBox3D`]: labels decoded as bboxes
  """
  from scipy.spatial.transform import Rotation as R

  lines = [l for l in label_str.split('\n') if l]

  cuboids = []
  bboxes = []
  for line in lines:
    toks = line.split(' ')
    assert len(toks) in (15, 16), "Invalid line %s" % line

    # The last column is score, which is optional (or lacking in label files)
    if len(toks) == 15:
      toks.append(-1.)

    # Context
    category_name = str(toks[0])
    truncated = float(toks[1])
    occluded = int(toks[2])
    alpha = float(toks[3])
      # The yaw from the camera principle ray to the obj; approximately
      # the yaw about the car's Z-axis.
    
    # BBox in camera frame
    left = float(toks[4])
    top = float(toks[5])
    right = float(toks[6])
    bottom = float(toks[7])

    # Cuboid in left color camera frame
    # height_meters = float(toks[8])~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # width_meters = float(toks[9])
    # length_meters = float(toks[10])
    # y_size = float(toks[8])
    # x_size = float(toks[9])
    # z_size = float(toks[10])
    kheight = float(toks[8])
    kwidth = float(toks[9])
    klength = float(toks[10])
    bottom_x = float(toks[11])
    bottom_y = float(toks[12])
    bottom_z = float(toks[13])
    rotation_y = float(toks[14])
      # The yaw of the object versus the camera's y-axis, which points down
      # (i.e. approximately antiparallel with the car's z-axis).
    score = float(toks[15])


    extra = {
      'kitti.truncated': str(truncated),
      'kitti.occluded': str(occluded),
      'kitti.score': str(score),
      'kitti.cam_relative_yaw': str(alpha),
    }

    bbox = datum.BBox2D(
                  x=left,
                  y=top,
                  width=right - left + 1,
                  height=bottom - top + 1,
                  category_name=category_name,
                  extra=extra)
    bbox.quantize()
    bboxes.append(bbox)
    
    # import math # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``

    # Rotation about the y-axis, which in KITTI camera frame is yaw, where
    # clockwise is to the right of the car.
    # https://github.com/xinshuoweng/AB3DMOT/blob/4009ba5855bda9a347d9f0a8bd72f351e3b00daf/kitti_utils.py#L313
    # # 
    # c = math.cos(rotation_y)
    # s = math.sin(rotation_y)
    # rot = np.array([[c,  0,  s],
    #                 [0,  1,  0],
    #                 [-s, 0,  c]])


    cuboids.append(
      datum.Cuboid(
        category_name=category_name,

        # See https://github.com/pratikac/kitti/blob/master/readme.tracking.txt#L84
        # length_meters=klength,
        # width_meters=kheight, # ~~~~ 
        # height_meters=kwidth, # ~~~~
        length_meters=klength,
        width_meters=kwidth, # ~~~~ 
        height_meters=kheight, # ~~~~
        # length_meters=z_size,
        # width_meters=y_size,
        # height_meters=x_size,
        obj_from_ego=datum.Transform(
          # rotation=rot,#R.from_euler('zxy', [rotation_y-math.pi, -math.pi/2, 0]).as_matrix(),
          # rotation=R.from_euler('yzx', [rotation_y, 0, 0]).as_matrix(),
          rotation=R.from_euler('zyx', [-rotation_y, 0, math.pi/2]).as_matrix(),
            # In addition to including the yaw label `rotation_y`, we apply
            # a pi/2 roll to account for the camera/lidar z-axis swap.
          # translation=[bottom_x, bottom_y - .5 * height_meters, bottom_z],
          # translation=[bottom_x, bottom_y - .5 * y_size, bottom_z],
          translation=[bottom_x, bottom_y - .5 * kheight, bottom_z],
          src_frame='camera|left',
          dest_frame='obj'),
        extra=extra))

  return cuboids, bboxes


@attr.s(eq=False)
class Calibration(object):
  """TODO more docs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
  
  Note that this class is designed to interface with the calibration data
  provided in the Benchmark datasets.  This calibration data is a *subset* of
  that available in the Raw Sync data (e.g. Benchmarks only have camera `P`
  Projective matrices, but Raw Sync data has explicit stereo baseline
  transforms).  Why?  While the Benchmark **training** data overlaps with
  the Raw Sync data, for the **test** split there is NO PUBLIC Raw Sync data
  available (not even calibration params), hence we stick to the data provided
  in the Benchmark releases.

  We don't use pykitti directly because:
    * It has odd dependencies, and not all are included in its setup.py
    * It's not compatible with the calibration data in the KITTI Benchmark zips
    * The pykitti code confounds file objects with other parsing / data
      structures
    * The pykitti code doesn't have much support for the Benchmarks; most
      support is for the Raw Sync data.
    
  See also:
    * `Google / Waymo's KITTI parsing code <https://github.com/tensorflow/lingvo/blob/96eaa85c648c45585ca76493bba5991212bac38a/lingvo/tasks/car/tools/kitti_data.py>`_
    * `kitti-object-eval-python <https://github.com/traveller59/kitti-object-eval-python/blob/9f385f8fd40c195a6370ae3682889d8d5dddf42b/kitti_common.py#L75>`_
    * `pykitti <https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/tracking.py#L125>`_

  """


  # pykitti 

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

  imu_to_velo = attr.ib(type=datum.Transform, default=datum.Transform())
  """Raw transform from IMU to velodyne frame.  Called `Tr_imu_to_velo` in
  Benchmark calibration data."""

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

  @staticmethod
  def derive_T_from_P(P):
    """KITTI provides only the camera Projective Matrix `P` for the Benchmarks;
    the KITTI authors compute `P` from the intrinsic Calibration Matrix `K`
    and other extrinsic calibration.  In this utility we extract `K` and a
    compatible transform [R|T] for projecing 3d points into the camera image.

    Problem: we want to extract K and [R|T] from the given P matrix.
    Reference: `Zisserman "Multiple View Gemoetric (2nd ed.) pg. 163
    <http://cvrs.whu.edu.cn/downloads/ebooks/Multiple%20View%20Geometry%20in%20Computer%20Vision%20(Second%20Edition).pdf>`_

    We can obtain K and R using an RQ decomposition on P.  For example:
      `K, RT = scipy.linalg.rq(P[:3, :3])`  
    However, we note that the P matrices given in the Benchmark data tend have
    the structure
    ::
            | a 0 b e |
            | 0 c d f |
            | 0 0 1 g |
    where the left-hand block mimics the structure of K.  Unsurprisingly, a
    RQ decomposition finds that R is the 3x3 identity and suggests:
    ::
        K = | a 0 b | 
            | 0 c d | 
            | 0 0 1 | 
    To use this decomposition, though, we'd have to deduce T from K.  If we 
    indeed assume that R = I, and further accept the raw values of P as
    intrinsics for K, we can solve for T as follows:
    ::
      P = K [R|T] = | fx  0 cx | | 1 0 0  Tx | 
                    |  0 fy cy | | 0 1 0  Ty |
                    |  0  0  1 | | 0 0 1  Tz |

                  = | fx  0 cx | | 1 0 0  (1/fx) * (P[0,3] - cx * P[2,3]) | 
                    |  0 fy cy | | 0 1 0  (1/fy) * (P[1,3] - cy * P[2,3]) |
                    |  0  0  1 | | 0 0 1  P[2,3]                          |

    From the above, we recover a T like:
    ::
      T_left_cam =     [ 0.05984926, -0.00035793,  0.0027459]^T
      T_right_cam =    [-0.47286266,  0.00239497,  0.0027299]^T
      Tr_velo_to_cam = [-0.00406977, -0.07631618, -0.2717806]^T

    So, `T_left_cam` is ~6cm long, and `T_right_cam` is ~47.3cm long;
    these figures tend to agree with the KITTI vehicle reference diagram:
    http://www.cvlibs.net/datasets/kitti/setup.php

    What's also notable is that the **raw** KITTI `Tr_velo_to_cam` transform
    (which has a translation norm of about 28cm) appears to be a transform to
    the left *grey* camera (camera 0) and not the left *color* camera,
    which is what we want.

    Qualitatively, the deduced `T` values appear to give good lidar-to-camera
    projections; see the test `test_kitti_object_lidar_camera_projection()`.

    Args:
      P (np.ndarray): A 3x4 projective matrix.
    
    Returns:
      T (np.ndarray): A derived 3x1 translation vector.
    """

    fx = P[0, 0]
    fy = P[1, 1]
    cx = P[0, 2]
    cy = P[1, 2]
    Tx = (1/fx) * (P[0,3] - cx * P[2,3])
    Ty = (1/fy) * (P[1,3] - cy * P[2,3])
    Tz = P[2, 3]
    
    return np.array([[Tx, Ty, Tz]]).T

  def __attrs_post_init__(self):
    # As noted above in `derive_T_from_P()`, we interpret raw intrinsics from
    # the provided P matrices and deduce T
    self.K2 = self.P2[:3, :3]
    self.K3 = self.P3[:3, :3]
    self.T2 = Calibration.derive_T_from_P(self.P2)
    self.T3 = Calibration.derive_T_from_P(self.P3)

    vel_to_cam_left_grey = self.R0_rect @ self.velo_to_cam_unrectified

    RT_left_color = datum.Transform(translation=self.T2)
    self.velo_to_cam_2_rect = RT_left_color @ vel_to_cam_left_grey

    # Bless this transform; explicitly set frame
    self.velo_to_cam_2_rect.src_frame = 'ego' # For KITTI, lidar is ego
    self.velo_to_cam_2_rect.dest_frame = 'camera|left'

    
    RT_right_color = datum.Transform(translation=self.T3)
    self.velo_to_cam_3_rect = RT_right_color @ vel_to_cam_left_grey

    # Bless this transform; explicitly set frame
    self.velo_to_cam_3_rect.src_frame = 'ego' # For KITTI, lidar is ego
    self.velo_to_cam_3_rect.dest_frame = 'camera|right'

  @classmethod
  def from_kitti_str(cls, calib_txt):
    """Create and return a `Calibration` instance from the given calibration
    file string `calib_txt`.  This string may originate from the calibration
    text files embedded in the `data_object_calib.zip` or
    `data_tracking_calib.zip` zips.  Note that these calibration files are 
    different than those provided with the KITTI Raw Sync data.
    """
    
    # Parse raw data. Based upon pykitti.  We don't use pykitt directly due to
    # its dependency issues and the way it confounds files objecs with parsing
    # code and data structures.
    # https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
    lines = [l for l in calib_txt.split('\n') if l]
    data = {}
    for line in lines:
      # P0: 7.115377000000e+02 0.000000000000e+00 -> P0: np.array([...])
      # OR P0 7.115377000000e+02 0.000000000000e+00 -> P0: np.array([...])
      toks = [t for t in line.split(' ') if t]
      k = toks[0]
      if ':' in k:
        k = k.replace(':', '')
      # k, v = line.split(':', 1) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      # data[k] = np.array([float(vv) for vv in v.split()])
      data[k] = np.array([float(t) for t in toks[1:]])

    kwargs = {}
    
    # Load camera projective matrices
    CAMERAS = ('P2', 'P3') # Ignore grey cameras!
    for cam in CAMERAS:
      kwargs[cam] = np.reshape(data[cam], (3, 4))

    ## Decide on keys
    # Object and Tracking use different keys.  Default to Object, fall back
    # to Tracking.
    R0_rect_key = 'R0_rect'
    if R0_rect_key not in data:
      R0_rect_key = 'R_rect' # Tracking

    Tr_velo_to_cam_key = 'Tr_velo_to_cam'
    if Tr_velo_to_cam_key not in data:
      Tr_velo_to_cam_key = 'Tr_velo_cam' # Tracking
    
    Tr_imu_to_velo_key = 'Tr_imu_to_velo'
    if Tr_imu_to_velo_key not in data:
      Tr_imu_to_velo_key = 'Tr_imu_velo'

    ## Load extrinsics
    kwargs['R0_rect'] = datum.Transform(
                          rotation=np.reshape(data[R0_rect_key], (3, 3)),
                          src_frame='camera|left_raw',
                          dest_frame='camera|left_sensor')

    kwargs['velo_to_cam_unrectified'] = (
      datum.Transform.from_transformation_matrix(
        np.reshape(data[Tr_velo_to_cam_key], (3, 4)),
        src_frame='lidar', dest_frame='camera|left_grey_raw'))
    
    kwargs['imu_to_velo'] = datum.Transform.from_transformation_matrix(
      np.reshape(data[Tr_imu_to_velo_key], (3, 4)),
      src_frame='oxts', dest_frame='lidar')
    
    return cls(**kwargs)


###############################################################################
### StampedDatum Table

def _is_image_or_scan_or_oxt(path):
  return not path.endswith('dataformat.txt') and (
      path.endswith('.png') or
      path.endswith('.bin') or
      ('oxt' in path and path.endswith('.txt')) or
      ('label' in path and path.endswith('.txt')))

def _rdd_of_all_archive_datafiles(spark, archive_paths):
  from oarphpy import spark as S
  
  rdds = []
  for path in archive_paths:
    archive_rdd = S.archive_rdd(spark, str(path))
    archive_rdd = archive_rdd.filter(
                      lambda fw: _is_image_or_scan_or_oxt(fw.name))
    archive_rdd = archive_rdd.map(
                    lambda fw:
                      (os.path.basename(fw.archive.archive_path), fw.name))
    rdds.append(archive_rdd)
  rdd = spark.sparkContext.union(rdds)
  return rdd


class BenchmarkToRawMapper(object):
  """This utility leverages artifacts from the 
  [PSegs-KITTI-Ext](https://github.com/pwais/psegs-kitti-ext) project to 
  look up contextual info from the KITTI Raw Data using Benchmark data.
  """

  ### Public API

  # Cache derived index files in the fixtures index_root directory. Saves users
  # a minute or two that it takes to sift through about 250k rows of the
  # bench_to_raw table.
  FIXTURES = Fixtures

  @classmethod
  def setup(cls, spark=None):
    util.log.info("Creating BenchmarkToRawMapper index ...")

    bench_to_raw_path = cls.FIXTURES.bench_to_raw_path()
    if os.path.exists(bench_to_raw_path):
      from psegs.spark import Spark
      with Spark.sess(spark) as spark:
        bench_to_raw_df = spark.read.parquet(str(bench_to_raw_path))
        bench_file_to_context = \
          cls._create_bench_file_to_context(bench_to_raw_df)
    else:
      bench_file_to_context = {}

    # Save index
    index_path = cls._bench_file_to_context_path()
    import pickle
    from oarphpy import util as oputil
    oputil.mkdir(str(index_path.parent))
    with open(index_path, 'wb') as f:
      pickle.dump(bench_file_to_context, f, protocol=pickle.HIGHEST_PROTOCOL)
      util.log.info(
        "Saved %s entries of BenchmarkToRawMapper index to %s ." % (
          len(bench_file_to_context), f.name))

  def __init__(self):
    assert os.path.exists(self._bench_file_to_context_path()), \
      "User needs to run setup() first"
    
    import pickle
    with open(self._bench_file_to_context_path(), 'rb') as f:
      self._bench_file_to_context = pickle.load(f)

  def get_extra(self, uri):
    key = self._bench_file_key(uri)
    # print(key in self._bench_file_to_context, key, list(self._bench_file_to_context.keys())[:10])
    extra = self._bench_file_to_context.get(key, {})
    extra = dict(
      (k, str(v)) for k, v in extra.items()
      if v)
    
    # Labels map to images, but don't claim the label is an image file
    if 'labels' in uri.topic:
      extra.pop('kitti.raw.sha-1', None)
      extra.pop('kitti.raw.filename', None)
    
    return extra

  def fill_timestamp(self, uri):
    """Fill the real timestamp of `uri` for the *train* split of the Object and
    Tracking Benchmarks; these timestamps are derived from the Raw Sync Data
    release of KITTI.  For the *text* split, we interpolate a plausible
    timestamp using the frame number and observation that training
    frames [are consistently sampled at 10Hz](https://github.com/pwais/psegs-kitti-ext/#sensor-sample-rates-are-consistently-10hz)
    """

    if uri.split == 'train':
      extra = self.get_extra(uri)
      t = int(extra.get('kitti.raw.timestamp', 0))
      if t > 0:
        uri.timestamp = t
        return

    ## Fallback for test split and/or absence of backing bench to raw data
    # For Tracking Benchmark, kitti.frame distinct per tracking segement and
    # indexes frames starting at 0.
    # For Object Benchmark, kitti.frame is just a split-global index starting
    # at 0.
    # We'll make synthetic timestamps start at unix time 1 in order to 
    # distinguish them from null (zero-value) timestamps, which can be 
    # recognized as erroneous.
    BASE_NS = int(1e9)
    frame = int(uri.extra['kitti.frame'])
    uri.timestamp = BASE_NS + int(frame * 1e8)


  ### Support

  @classmethod
  def _bench_file_to_context_path(cls):
    return cls.FIXTURES.index_root() / 'bench_file_to_context.pkl'

  @classmethod
  def _create_bench_file_to_context(cls, bench_to_raw_df):
    # We need to reconstruct the Tracking Benchmark segment <-> Raw Segment
    # mapping in order to deduce timstamps for oxts; oxts are single files in
    # benchmarks but multiple files in raw, so `bench_to_raw_pdf` does not
    # map benchmark oxts to raw oxts.  For simplicitly, we recover this
    # mapping using just the first velodyne file for each segment.
    df = bench_to_raw_df.filter(
      (bench_to_raw_df.benchmark == 'data_tracking_velodyne.zip') &
      (bench_to_raw_df.topic == 'velodyne_points') &
      (bench_to_raw_df.frame == 0))
    def to_segment_pair(row):
      vuri = kitti_archive_file_to_uri(row['benchmark'], row['b_filename'])
      return (row['segment'], vuri.segment_id)
    pair_rdd = df.rdd.map(to_segment_pair)
    raw_segment_to_bench_segment = dict(pair_rdd.collect())
  

    # Now collect context for each benchmark file
    df = bench_to_raw_df.filter(
      (bench_to_raw_df['b_filename'].isNotNull()) |
      (bench_to_raw_df['topic'] == 'oxts'))
    
    def to_index_entry(row):
      if row['topic'] == 'oxts':
        if row['segment'] not in raw_segment_to_bench_segment:
          # This OXTS is probably from the test split or a non-benchmark
          # segment
          return (None, None)
        key = cls._bench_file_key(
                    datum.URI(
                      topic='oxts',
                      segment_id=raw_segment_to_bench_segment[row['segment']],
                      extra={'kitti.frame': row['frame']}))
      else:
        key = cls._bench_file_key(datum.URI(extra={
          'kitti.archive': row['benchmark'],
          'kitti.archive.file': row['b_filename'],
        }))

      extra = {
        'kitti.raw.timestamp': row['nanostamp'],
        'kitti.raw.segment_category': row['segment_category'],
        'kitti.raw.segment': row['segment'],
        'kitti.raw.filename': row['r_filename'],
        'kitti.raw.sha-1': row['r_digest'],
      }
      return (key, extra)
    
    bench_file_to_context = dict(df.rdd.map(to_index_entry).collect())
    return bench_file_to_context

  @classmethod
  def _bench_file_key(cls, uri):
    if uri.topic == 'oxts' or 'oxts' in uri.extra.get('kitti.archive.file', ''):
      return (uri.segment_id, int(uri.extra.get('kitti.frame', 0)))
    elif 'labels' in uri.topic: 
      if 'object' in uri.extra['kitti.archive']:
        # Labels are in camera frame; map them to corresponding camera image
        archive = str(uri.extra['kitti.archive'])
        archive = archive.replace('label', 'image')
        cam_file = str(uri.extra['kitti.archive.file'])
        cam_file = cam_file.replace('label', 'image').replace('txt', 'png')
        return (archive, cam_file)
      elif 'tracking' in uri.extra['kitti.archive']:
        # Labels are in camera frame; map them to corresponding camera image
        archive = str(uri.extra['kitti.archive'])
        archive = archive.replace('label', 'image')
        cam_file = str(uri.extra['kitti.archive.file'])
        cam_file = cam_file.replace('label', 'image')
        cam_file = cam_file.replace('.txt', '/%s.png' % uri.extra['kitti.frame'])
        return (archive, cam_file)
      else:
        raise ValueError(uri)
    else:
      return (uri.extra['kitti.archive'], uri.extra['kitti.archive.file'])

  


def kitti_archive_file_to_uri(archive_name, entryname):
  if 'object' in archive_name:
    return kitti_object_file_to_uri(archive_name, entryname)
  elif 'tracking' in archive_name:
    return kitti_tracking_file_to_uri(archive_name, entryname)
  else:
    raise ValueError("Unsupported %s %s" % (archive_name, entryname))


def kitti_get_topic_for_filename(filename):
  if 'image_2' in filename or 'image_02' in filename or 'prev_2' in filename:
    return 'camera|left'
  elif 'image_3' in filename or 'image_03' in filename or 'prev_3' in filename:
    return 'camera|right'
  elif 'label_2' in filename or 'label_02' in filename:
    return 'labels|cuboids'
    # bboxes ? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif 'velodyne' in filename:
    return 'lidar'
  elif 'oxts' in filename:
    return 'ego_pose'
  else:
    raise ValueError(filename)

  # if archive_name in ('data_object_image_2.zip', 'data_object_image_3.zip'):
  #   if ktopic == 'image_2':
  #     uri.topic = 'camera|left'
  #   elif ktopic == 'image_3':
  #     uri.topic = 'camera|right'
  #   else:
  #     raise ValueError()
  #   uri.extra['kitti.frame'] = fname_prefix
  # elif archive_name in ('data_object_prev_2.zip', 'data_object_prev_3.zip'):
  #   if ktopic == 'prev_2':
  #     uri.topic = 'camera|left'
  #   elif ktopic == 'prev_3':
  #     uri.topic = 'camera|right'
  #   else:
  #     raise ValueError()
  #   prefix = fname.split('.')[0]
  #   frame, seqnum = prefix.split('_')
  #   uri.extra['kitti.frame'] = frame
  #   uri.extra['kitti.prev'] = seqnum
  # elif archive_name == 'data_object_label_2.zip':
  #   uri.topic = 'labels|cuboids'
  #   uri.extra['kitti.frame'] = fname_prefix
  #   # TODO bboxes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # elif archive_name == 'data_object_velodyne.zip':
  #   uri.topic = 'lidar'
  #   uri.extra['kitti.frame'] = fname_prefix
  # elif archive_name == 'data_object_calib.zip':
    

def kitti_object_file_to_uri(archive_name, entryname):
  """Create and return a URI for the given KITTI Object Benchmark file."""

  assert archive_name in Fixtures.OBJECT_BENCHMARK_FNAMES, archive_name

  split = 'test' if 'test' in entryname else 'train'
  uri = datum.URI(
            dataset='kitti-object',
            split=split,
            segment_id='kitti-object-benchmark-' + split,
              # NB: The Object Benchmark is just a bunch of scans rather than a
              # sequence of timed scans; we stuff all data in a fake segment
              # with this name.  We use the split to ensure that the training
              # and testing sets have distinct 'segments', since the data
              # is distinct.
            topic=kitti_get_topic_for_filename(entryname),
            extra={
              'kitti.archive': archive_name,
              'kitti.archive.file': entryname,
            })
  
  # Object Benchmark has filenames like
  # training/label_2/006415.txt
  # training/prev_2/007464_03.png
  # The 6-digit number is the frame number and links camera, lidar,
  # calibration etc.
  parts = Path(entryname).parts
  assert parts[0] in ('training', 'testing')
  assert len(parts) == 3
  ktopic = parts[1]
  fname = parts[2]
  fname_prefix = fname.split('.')[0]

  if archive_name in (
      'data_object_image_2.zip', 'data_object_image_3.zip',
      'data_object_label_2.zip', 'data_object_velodyne.zip'):

    # TODO bboxes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    uri.extra['kitti.frame'] = fname_prefix
  elif archive_name in ('data_object_prev_2.zip', 'data_object_prev_3.zip'):
    prefix = fname.split('.')[0]
    frame, seqnum = prefix.split('_')
    uri.extra['kitti.frame'] = frame
    uri.extra['kitti.prev'] = seqnum
  elif archive_name == 'data_object_calib.zip':
    raise ValueError("Can't address calibration!")
  else:
    raise ValueError("Dont know what to do with %s" % archive_name)

  return uri


def kitti_tracking_file_to_uri(archive_name, entryname):
  """Create and return a URI for the given KITTI Tracking Benchmark file."""

  assert archive_name in Fixtures.TRACKING_BENCHMARK_FNAMES, archive_name

  # Tracking Benchmark has filenames like
  # training/image_02/0009/000667.png
  # training/label_02/0002.txt
  # training/oxts/0002.txt
  # For sensor data:
  # The 6-digit number in the end file name is the frame number and links
  # camera, lidar, calibration etc.
  # The 4-digit number in the directory name is the segment_id
  # For context files:
  # The 4-digit number in the file name is the segment_id
  
  parts = Path(entryname).parts
  assert parts[0] in ('training', 'testing')
  assert len(parts) in (3, 4)
  ktopic = parts[1]
  if len(parts) == 4:
    ksegment_id = parts[2]
    frame = parts[3].split('.')[0]
  else:
    ksegment_id = parts[2].split('.')[0]
    frame = None
  
  # Train and Test segments have IDs with overlapping ranges of numbers; to
  # indicate that they are indeed however distinct, we prefix them with split.
  split = 'test' if 'test' in entryname else 'train'
  segment_id = 'kitti-tracking-' + split + '-' + ksegment_id

  uri = datum.URI(
          dataset='kitti-tracking',
          split='test' if 'test' in entryname else 'train',
          segment_id=segment_id,
          topic=kitti_get_topic_for_filename(entryname),
          extra={
            'kitti.archive': archive_name,
            'kitti.archive.file': entryname,
          })
  if frame:
    uri.extra['kitti.frame'] = frame

  return uri


class KITTISDTable(StampedDatumTableBase):
  
  FIXTURES = Fixtures

  INCLUDE_OBJ_PREV_FRAMES = True

  INCLUDE_OBJECT_BENCHMARK = True
  INCLUDE_TRACKING_BENCHMARK = True

  ## Subclass API

  @classmethod
  def _get_all_segment_uris(cls):
    import zipfile
    from oarphpy import util as oputil
    
    uris = set()
    for archive_path in cls._get_all_archive_paths():
      if os.path.exists(archive_path):
        for entryname in zipfile.ZipFile(archive_path).namelist():
          if _is_image_or_scan_or_oxt(entryname):
            uri = kitti_archive_file_to_uri(archive_path.name, entryname)
            uris.add(str(uri.to_segment_uri()))
    
    return sorted(datum.URI.from_str(uri) for uri in uris)

  @classmethod
  def _create_datum_rdds(cls, spark, existing_uri_df=None, only_segments=None):

    ## First build indices (saves several minutes per worker per chunk) ...
    class SDBenchmarkToRawMapper(BenchmarkToRawMapper):
      FIXTURES = cls.FIXTURES
    SDBenchmarkToRawMapper.setup(spark=spark)

    ## ... now build a set of tasks to do ...
    archive_paths = cls._get_all_archive_paths()
    task_rdd = _rdd_of_all_archive_datafiles(spark, archive_paths)
    task_rdd = task_rdd.cache()
    util.log.info("Discovered %s tasks ..." % task_rdd.count())
    
    ## ... convert to URIs and filter those tasks if necessary ...
    if existing_uri_df:
      # Since we keep track of the original archives and file names, we can
      # just filter on those.  We'll collect them in this process b/c the
      # maximal set of URIs is smaller than RAM.
      def to_task(row):
        return (row.extra.get('kitti.archive'),
                row.extra.get('kitti.archive.file'))
      skip_tasks = set(
        existing_uri_df.select('extra').rdd.map(to_task).collect())
      
      task_rdd = task_rdd.filter(lambda t: t not in skip_tasks)
      util.log.info(
        "Resume mode: have datums for %s datums; dropped %s tasks" % (
          existing_uri_df.count(), len(skip_tasks)))
    
    uri_rdd = task_rdd.map(lambda task: kitti_archive_file_to_uri(*task))
    if only_segments:
      util.log.info(
        "Filtering to only %s segments" % len(only_segments))
      uri_rdd = uri_rdd.filter(
        lambda uri: any(
          suri.soft_matches_segment(uri) for suri in only_segments))

    ## ... run tasks and create stamped datums.
    # from oarphpy.spark import cluster_cpu_count
    URIS_PER_CHUNK = os.cpu_count() * 64 # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ make class member so can configure to RAM
    uris = uri_rdd.collect()
    util.log.info("... creating datums for %s URIs." % len(uris))

    datum_rdds = []
    for chunk in oputil.ichunked(uris, URIS_PER_CHUNK):
      chunk_uri_rdd = spark.sparkContext.parallelize(chunk)
      datum_rdd = chunk_uri_rdd.flatMap(cls._iter_datums_from_uri)
      datum_rdds.append(datum_rdd)
      # if len(datum_rdds) >= 10:
      #   break # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return datum_rdds
  
  @classmethod
  def _get_all_archive_paths(cls):
    archives = []
    if cls.INCLUDE_OBJECT_BENCHMARK:
      archives += list(cls.FIXTURES.OBJECT_BENCHMARK_FNAMES)
      if not cls.INCLUDE_OBJ_PREV_FRAMES:
        archives = [arch for arch in archives if 'prev' not in arch]
    if cls.INCLUDE_TRACKING_BENCHMARK:
      archives += list(cls.FIXTURES.TRACKING_BENCHMARK_FNAMES)
    archives = [arch for arch in archives if 'calib' not in arch]
    paths = [cls.FIXTURES.zip_path(arch) for arch in archives]
    return paths


  ## Datum Construction Support

  @classmethod
  def _get_file_bytes(cls, uri=None, archive=None, entryname=None):
    """Read bytes for the file referred to by `uri`"""

    if uri is not None:
      archive = uri.extra['kitti.archive']
      entryname = uri.extra['kitti.archive.file']
    assert archive and entryname

    # Cache the Zipfiles for faster loading
    if not hasattr(cls, '_get_file_bytes_archives'):
      cls._get_file_bytes_archives = {}
    if archive not in cls._get_file_bytes_archives:
      import zipfile
      path = cls.FIXTURES.zip_path(archive)
      cls._get_file_bytes_archives[archive] = zipfile.ZipFile(path)
      
    
    try:
      return cls._get_file_bytes_archives[archive].read(entryname)
    except Exception as e:
        raise Exception((e, archive, uri))

  @classmethod
  def _get_segment_frame_to_pose(cls, segment_id):
    """Get the frame -> pose map for the given `segment_id`.  Cache these since
    multiple datum constructors will need to look up poses."""
    if not hasattr(cls, '_seg_to_poses'):
      cls._seg_to_poses = {}
    if segment_id not in cls._seg_to_poses:
      split, segnum = segment_id.split('-')[-2:]
      entryname = split + 'ing/oxts/' + segnum + '.txt'
      oxts_str = cls._get_file_bytes(
        archive='data_tracking_oxts.zip', entryname=entryname)
      oxts_str = oxts_str.decode()
      frame_to_xform = load_transforms_from_oxts(oxts_str)
      cls._seg_to_poses[segment_id] = frame_to_xform
    return cls._seg_to_poses[segment_id]

  @classmethod
  def _get_ego_pose(cls, uri):
    # Pose information for Object Benchmark not available
    if 'kitti-object-benchmark' in uri.segment_id:
      return datum.Transform(src_frame='world', dest_frame='ego')
    else:
      frame_to_xform = cls._get_segment_frame_to_pose(uri.segment_id)
      return frame_to_xform[int(uri.extra['kitti.frame'])]

  @classmethod
  def _get_calibration(cls, uri):
    """Get the `Calibration` instance for the given `uri`.  Cache these since
    multiple datum constructors will need to look up calibration."""

    if not hasattr(cls, '_obj_frame_to_calib'):
      cls._obj_frame_to_calib = {}
    if not hasattr(cls, '_tracking_seg_to_calib'):
      cls._tracking_seg_to_calib = {}
    
    if 'kitti-object-benchmark' in uri.segment_id:
      frame = uri.extra['kitti.frame']
      if frame not in cls._obj_frame_to_calib:
        entryname = uri.split + 'ing/calib/' + frame + '.txt'
        calib_str = cls._get_file_bytes(
          archive='data_object_calib.zip', entryname=entryname)
        calib_str = calib_str.decode()
        calib = Calibration.from_kitti_str(calib_str)
        cls._obj_frame_to_calib[frame] = calib
      return cls._obj_frame_to_calib[frame]
    
    else: # Tracking
      if uri.segment_id not in cls._tracking_seg_to_calib:
        split, segnum = uri.segment_id.split('-')[-2:]
        entryname = split + 'ing/calib/' + segnum + '.txt'
        calib_str = cls._get_file_bytes(
          archive='data_tracking_calib.zip', entryname=entryname)
        calib_str = calib_str.decode()
        calib = Calibration.from_kitti_str(calib_str)
        cls._tracking_seg_to_calib[uri.segment_id] = calib
      return cls._tracking_seg_to_calib[uri.segment_id]

  @classmethod
  def _project_cuboids_to_lidar_frame(cls, uri, cuboids):
    """Project the given `cuboids` from the camera frame to the lidar frame
    (using calibration for `uri`) and return a transformed copy.

    See also the tests:
     * `test_kitti_object_label_lidar_projection()`
     * `test_kitti_tracking_label_lidar_projection()`
    """
    import copy

    ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ## Note: KITTI Cuboids are in the *camera* frame and must be projected
    ## into the lidar frame for plotting. This test helps document and 
    ## ensure this assumption holds.
    ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    calib = cls._get_calibration(uri)
    lidar_to_cam = calib.R0_rect @ calib.velo_to_cam_unrectified
    cam_to_lidar = lidar_to_cam.get_inverse()

    cuboids = copy.deepcopy(cuboids)
    for c in cuboids:
      from psegs.datum.transform import Transform
      obj_from_ego_lidar = cam_to_lidar @ c.obj_from_ego
      c.obj_from_ego = obj_from_ego_lidar
      c.obj_from_ego.src_frame = 'ego' # In KITTI, lidar is the ego frame ~~~~~~~~~~
      c.obj_from_ego.dest_frame = 'obj'

    return cuboids

  @classmethod
  def _get_bench2raw_mapper(cls):
    if not hasattr(cls, '_bench2raw_mapper'):
      class SDBenchmarkToRawMapper(BenchmarkToRawMapper):
        FIXTURES = cls.FIXTURES
      cls._bench2raw_mapper = SDBenchmarkToRawMapper()
    return cls._bench2raw_mapper


  ## Datum Construction

  @classmethod
  def _iter_datums_from_uri(cls, uri):
    if uri.topic.startswith('camera'):
      yield cls._create_camera_image(uri)
    elif uri.topic.startswith('lidar'):
      yield cls._create_point_cloud(uri)
    elif uri.topic.startswith('labels'):
      for sd in cls._iter_labels(uri):
        yield sd
    elif uri.topic == 'ego_pose':
      for sd in cls._iter_ego_poses(uri):
        yield sd
    else:
      raise ValueError(uri)
  
  @classmethod
  def _create_camera_image(cls, uri):
    from psegs.util import misc

    image_png = cls._get_file_bytes(uri=uri)
    width, height = misc.get_png_wh(image_png)

    mapper = cls._get_bench2raw_mapper()
    mapper.fill_timestamp(uri)

    # timestamp = int(int(uri.extra['kitti.frame']) * 1e8)
    # # TODO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
    # uri.timestamp = timestamp

    ego_pose = cls._get_ego_pose(uri)

    calib = cls._get_calibration(uri)
    K = calib.K2
    ego_to_sensor = calib.velo_to_cam_2_rect
    if 'right' in uri.topic:
      K = calib.K3
      ego_to_sensor = calib.velo_to_cam_3_rect

    extra = mapper.get_extra(uri)

    ci = datum.CameraImage(
          sensor_name=uri.topic,
          image_png=bytearray(image_png),
          width=width,
          height=height,
          timestamp=uri.timestamp,
          ego_pose=ego_pose,
          K=K,
          ego_to_sensor=ego_to_sensor,
          extra=extra)
    return datum.StampedDatum(uri=uri, camera_image=ci)

  @classmethod
  def _create_point_cloud(cls, uri):
    lidar_bytes = cls._get_file_bytes(uri=uri)
    raw_lidar = np.frombuffer(lidar_bytes, dtype=np.float32).reshape((-1, 4))
    cloud = raw_lidar[:, :3]
    # unused: reflectance = raw_lidar[:, 3:]

    # timestamp = int(int(uri.extra['kitti.frame']) * 1e8)
    # # TODO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
    # uri.timestamp = timestamp
    mapper = cls._get_bench2raw_mapper()
    mapper.fill_timestamp(uri)

    # In KITTI, lidar is the ego frame
    ego_to_sensor = Transform(src_frame='ego', dest_frame='lidar')

    ego_pose = cls._get_ego_pose(uri)

    extra = mapper.get_extra(uri)

    pc = datum.PointCloud(
          sensor_name=uri.topic,
          timestamp=uri.timestamp,
          cloud=cloud,
          ego_to_sensor=ego_to_sensor,
          ego_pose=ego_pose,
          extra=extra)
    return datum.StampedDatum(uri=uri, point_cloud=pc)

  @classmethod
  def _iter_labels(cls, uri):
    # KITTI has no labels for test.
    # FMI see https://github.com/pwais/psegs-kitti-ext
    if uri.split == 'test':
      return
    
    if 'kitti-object-benchmark' in uri.segment_id:
      yield cls._get_object_labels(uri)
    else: # Tracking
      for sd in cls._iter_tracking_labels(uri):
        yield sd
  
  @classmethod
  def _get_object_labels(cls, uri):
    frame = uri.extra['kitti.frame']
    entryname = uri.split + 'ing/label_2/' + frame + '.txt'
    label_str = cls._get_file_bytes(
        archive='data_object_label_2.zip', entryname=entryname)
    label_str = label_str.decode()
    cuboids, bboxes = parse_object_label_cuboids(label_str)

    # FIXME bboxes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    cuboids = cls._project_cuboids_to_lidar_frame(uri, cuboids)

    # timestamp = int(int(frame) * 1e8)
    # # TODO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
    # uri.timestamp = timestamp
    mapper = cls._get_bench2raw_mapper()
    mapper.fill_timestamp(uri)

    for c in cuboids:
      c.timestamp = uri.timestamp
      c.ego_pose = cls._get_ego_pose(uri)
      c.extra = mapper.get_extra(uri)
    
    return datum.StampedDatum(uri=uri, cuboids=cuboids)
  
  @classmethod
  def _iter_tracking_labels(cls, uri):
    import copy
    
    split, segnum = uri.segment_id.split('-')[-2:]
    entryname = split + 'ing/label_02/' + segnum + '.txt'
    labels_str = cls._get_file_bytes(
      archive='data_tracking_label_2.zip', entryname=entryname)
    labels_str = labels_str.decode()

    f_to_cuboids, _ = parse_tracking_label_cuboids(labels_str)
      # NB: We ignore bboxes for the Tracking Benchmark
    
    mapper = cls._get_bench2raw_mapper()
    for frame, cuboids in f_to_cuboids.items():
      datum_uri = copy.deepcopy(uri)
      datum_uri.extra['kitti.frame'] = str(frame).zfill(6)

      cuboids = cls._project_cuboids_to_lidar_frame(uri, cuboids)

      # timestamp = int(int(frame) * 1e8)
      # # TODO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
      # datum_uri.timestamp = timestamp
      mapper.fill_timestamp(datum_uri)

      for c in cuboids:
        c.timestamp = datum_uri.timestamp
        c.ego_pose = cls._get_ego_pose(datum_uri)
        c.extra = mapper.get_extra(datum_uri)

      yield datum.StampedDatum(uri=datum_uri, cuboids=cuboids)

  @classmethod
  def _iter_ego_poses(cls, uri):
    import copy

    # Pose information for Object Benchmark not available
    if 'kitti-object-benchmark' in uri.segment_id:
      return
    
    mapper = cls._get_bench2raw_mapper()
    frame_to_xform = cls._get_segment_frame_to_pose(uri.segment_id)
    for frame, xform in frame_to_xform.items():
      datum_uri = copy.deepcopy(uri)
      # datum_uri.timestamp = int(int(frame) * 1e8) # FIXME ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      datum_uri.extra['kitti.frame'] = str(frame).zfill(6)
      mapper.fill_timestamp(datum_uri)
      yield datum.StampedDatum(uri=datum_uri, transform=xform)


###############################################################################
### IDatasetUtil Impl

class DSUtil(IDatasetUtil):

  FIXTURES = Fixtures

  @classmethod
  def all_zips(cls):
    import itertools
    all_zips = itertools.chain(
                  cls.FIXTURES.OBJECT_BENCHMARK_FNAMES,
                  cls.FIXTURES.TRACKING_BENCHMARK_FNAMES)
    return list(all_zips)

  @classmethod
  def emplace(cls):
    cls.FIXTURES.maybe_emplace_psegs_kitti_ext()

    if not cls.FIXTURES.ROOT.exists():
      zips = '\n        '.join('  * %s' % fname for fname in cls.all_zips())
      cls.show_md("""
        Due to KITTI license constraints, you need to manually accept the KITTI
        license to obtain the download URLs for the
        [Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) and
        [Object Benchmark](http://www.cvlibs.net/datasets/kitti/eval_object.php)
        zip files.  But once you have the URL, it's easy to write a short bash
        loop with `wget` to fetch them in parallel.

        You'll want to download all the following zip files (do not decompress
        them) to a single directory on a local disk (spinning disk OK):

        %s

        Once you've downloaded the archives, we'll need the path to where
        you put them.  Enter that below, or exit this program.

      """ % (zips,))
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
    zips_needed = set(cls.all_zips())
    zips_have = set()
    for entry in cls.FIXTURES.ROOT.iterdir():
      if entry.name in zips_needed:
        zips_needed.remove(entry.name)
        zips_have.add(entry.name)
    
    if zips_needed:
      s_have = \
        '\n        '.join('  * %s' % fname for fname in zips_have)
      s_needed = \
        '\n        '.join('  * %s' % fname for fname in zips_needed)
      cls.show_md("""
        Missing some expected archives!

        Found:
        
        %s

        Missing:

        %s
      """ % (s_have, s_needed))
      return False
    
    cls.show_md("... all KITTI archives found!")
    return True

  @classmethod
  def test(cls):
    from oarphpy import util as oputil
    oputil.run_cmd("cd %s && pytest -s -vvv -k test_kitti" % C.PS_ROOT)
    return True

  @classmethod
  def build_table(cls):
    KITTISDTable.build()
    return True
