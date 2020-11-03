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

import itertools
import os

import attr
import numpy as np

from psegs import util
from psegs import datum
from psegs.conf import C
from psegs.datasets.idsutil import IDatasetUtil
from psegs.table.sd_table import StampedDatumTableBase
from psegs.util import misc


###############################################################################
### KITTI-360 Fixtures & Other Constants

class Fixtures(object):

  ROOT = C.EXT_DATA_ROOT / 'kitti-360'

  CAMERAS = ('image_00', 'image_01', 'image_02', 'image_03')

  TRAIN_SEQUENCES = (
    '2013_05_28_drive_0000_sync',
    '2013_05_28_drive_0002_sync',
    '2013_05_28_drive_0003_sync',
    '2013_05_28_drive_0004_sync',
    '2013_05_28_drive_0005_sync',
    '2013_05_28_drive_0006_sync',
    '2013_05_28_drive_0007_sync',
    '2013_05_28_drive_0009_sync',
    '2013_05_28_drive_0010_sync',
  )

  TEST_SEQUENCES = tuple() # Data not released yet?

  @classmethod
  def filepath(cls, rpath):
    return cls.ROOT / rpath

  @classmethod
  def frame_id_to_fname(cls, frame_id):
    return str(frame_id).rjust(10, '0')


  @classmethod
  def camera_image_path(cls, sequence, camera_name, frame_id):
    if camera_name in ('image_00', 'image_01'):
      return (
        cls.ROOT / 'data_2d_raw' / 
          sequence / camera_name / 'data_rect'/ 
            (cls.frame_id_to_fname(frame_id) + ".png"))
    elif camera_name in ('image_02', 'image_03'):
      return (
        cls.ROOT / 'data_2d_raw' / 
          sequence / camera_name / 'data_rgb'/ 
            (cls.frame_id_to_fname(frame_id) + ".png"))
    else:
      raise ValueError("Unsupported camera %s" % camera_name)
  
  @classmethod
  def get_camera_frame_ids(cls, sequence, camera_name):
    from oarphpy import util as oputil
    paths = oputil.all_files_recursive(
      str(cls.ROOT / 'data_2d_raw' / sequence / camera_name),
      pattern='*.png')
    frame_ids = [
      int(os.path.split(path)[-1].split('.')[0])
      for path in paths
      if not oputil.is_stupid_mac_file(path)
    ]
    return frame_ids


  @classmethod
  def velodyne_cloud_path(cls, sequence, frame_id):
    return (
      cls.ROOT / 'data_3d_raw' / 
        sequence / 'velodyne_points' / 'data' / 
          (cls.frame_id_to_fname(frame_id) + ".bin"))

  @classmethod
  def velodyne_timestamps_path(cls, sequence):
    return (
      cls.ROOT / 'data_3d_raw' / 
        sequence / 'velodyne_points' / 'timestamps.txt')

  @classmethod
  def sick_cloud_path(cls, sequence, frame_id):
    return (
      cls.ROOT / 'data_3d_raw' / 
        sequence / 'sick_points' / 'data' / 
          (cls.frame_id_to_fname(frame_id) + ".bin"))

  @classmethod
  def sick_timestamps_path(cls, sequence):
    return (
      cls.ROOT / 'sick_points' / 
        sequence / 'velodyne_points' / 'timestamps.txt')

  @classmethod
  def get_raw_scan_frame_ids(cls, sequence, sensor):
    from oarphpy import util as oputil
    paths = oputil.all_files_recursive(
      str(cls.ROOT / 'data_3d_raw' / sequence / sensor),
      pattern='*.bin')
    frame_ids = [
      int(os.path.split(path)[-1].split('.')[0])
      for path in paths
      if not oputil.is_stupid_mac_file(path)
    ]
    return frame_ids
  


  @classmethod
  def cuboids_path(cls, sequence, split='train'):
    return (
      cls.ROOT / 'data_3d_bboxes' / split / (sequence + ".xml"))
  

  @classmethod
  def ego_poses_path(cls, sequence):
    return (
      cls.ROOT / 'data_poses' / sequence / 'poses.txt')

  @classmethod
  def cam0_poses_path(cls, sequence):
    return (
      cls.ROOT / 'data_poses' / sequence / 'cam0_to_world.txt')



###############################################################################
### KITTI Parsing Utils

def kitti_360_3d_bboxes_get_parsed_node(d):
  """Parse a node in a data_3d_bboxes XML file"""

  def to_ndarray(d):
    import numpy as np
    r = int(d['rows'])
    c = int(d['cols'])
    dtype = str(d['dt'])
    parse = float if dtype == 'f' else int
    data = [parse(t) for t in d['data'].split() if t]
    a = np.array(data)
    return a.reshape((r, c))

  def fill_cuboid(d):
    # Appears the cuboid bounds are encoded as a scaling transform in the
    # transform itself; in the raw XML, the vertices are +/- 0.5m for all
    # objects in the XML
    # FMI https://github.com/autonomousvision/kitti360Scripts/blob/081c08b34a14960611f459f23a0ad049542205c6/kitti360scripts/helpers/annotation.py#L125
    R = d['transform'][:3, :3]
    T = d['transform'][:3, 3]
    v = d['vertices']
    d['cuboid'] = np.matmul(R, v.T).T + T

  # ??? Not sure what this is about
  # FMI https://github.com/autonomousvision/kitti360Scripts/blob/081c08b34a14960611f459f23a0ad049542205c6/kitti360scripts/helpers/annotation.py#L154
  def to_class(label_value):
    K360_CLASSMAP = {
      'driveway': 'parking',
      'ground': 'terrain',
      'unknownGround': 'ground', 
      'railtrack': 'rail track'
    }
    if label_value in K360_CLASSMAP:
      return K360_CLASSMAP[label_value]
    else:
      return label_value

  out = {
    'index':            int(d['index']),
    'label':            str(d['label']),
    'k360_class_name':  to_class(str(d['label'])),
    'semanticId_orig':  int(d['semanticId_orig']),
    'semanticId':       int(d['semanticId']),
    'instanceId':       int(d['instanceId']),
    'category':         str(d['category']),
    
    # 'timestamp':        int(d['timestamp']),
    'is_static':        bool(d['timestamp'] == '-1'),
    'active_frame_id':  int(d['timestamp']),
      # `timestamp` is -1 if object is static, and `timestamp` is actually
      # a frame ID, not a unix time
    
    # 'dynamic':          int(d['dynamic']),
    #   # In the current release, dynamic is always 0 (?)
    
    'start_frame':      int(d['start_frame']),
    'end_frame':        int(d['end_frame']),
    
    'transform':        to_ndarray(d['transform']),
    'vertices':         to_ndarray(d['vertices']),
    'faces':            to_ndarray(d['faces']),
  }

  fill_cuboid(out)
  return out

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
### StampedDatumTable Impl

class KITTI360SDTable(StampedDatumTableBase):

  FIXTURES = Fixtures

  ## Dataset API

  @classmethod
  def get_uris_for_sequence(cls, sequence):
    if sequence in cls.FIXTURES.TRAIN_SEQUENCES:
      split = 'train'
    elif sequence in cls.FIXTURES.TEST_SEQUENCES:
      split = 'test'
    else:
      raise ValueError("Unknown sequence %s" % sequence)
    
    base_uri = datum.URI(
      dataset='kitti-360',
      split=split,
      segment_id=sequence)

    iter_uris = itertools.chain(
      cls._iter_ego_pose_uris(base_uri),
      cls._iter_camera_image_uris(base_uri),
      cls._iter_point_cloud_uris(base_uri),
      cls._iter_cuboid_uris(base_uri),
    )
    return list(iter_uris)

  @classmethod
  def create_stamped_datum(cls, uri):
    if uri.topic.startswith('camera'):
      return cls._create_camera_image(uri)
    elif uri.topic.startswith('lidar') or uri.topic.startswith('laser'):
      return cls._create_point_cloud(uri)
    elif uri.topic == 'ego_pose':
      return cls._create_ego_pose(uri)
    elif uri.topic == 'labels|cuboids':
      return cls._create_cuboids(uri)
    else:
      raise ValueError(uri)


  ## Subclass API




  ## Private API: Utils
  
  @classmethod
  def _frame_id_to_timestamp(cls, frame_id):
    # KITTI-360 has not yet released all timestamps; for now
    # pretend all data captured at 10Hz.
    TEN_MS_IN_NS = 10000000
    return int(frame_id * TEN_MS_IN_NS)

  @classmethod
  def _get_frame_id(cls, uri):
    return int(uri.extra['kitti-360.frame_id'])

  CAMERA_NAME_TO_TOPIC = {
    'image_00': 'camera|left_rect',
    'image_01': 'camera|right_rect',
    'image_02': 'camera|left_fisheye',
    'image_03': 'camera|right_fisheye',
  }


  ## Private API: Ego Pose

  @classmethod
  def _iter_ego_pose_uris(cls, base_uri):
    poses = np.loadtxt(cls.FIXTURES.ego_poses_path(base_uri.segment_id))
    frame_ids = poses[:,0]
    for frame_id in frame_ids:
      yield base_uri.replaced(
              topic='ego_pose',
              timestamp=cls._frame_id_to_timestamp(frame_id),
              extra={'kitti-360.frame_id': str(int(frame_id))})

  @classmethod
  def _create_ego_pose(cls, uri):
    transform = cls._get_ego_pose(uri.segment_id, cls._get_frame_id(uri))
    assert transform, "Programming Error: no pose available for %s" % uri
    return datum.StampedDatum(uri=uri, transform=transform)

  @classmethod
  def _get_ego_pose(cls, sequence, frame_id):
    if not hasattr(cls, '_ego_pose_cache'):
      cls._ego_pose_cache = {}
    if not sequence in cls._ego_pose_cache:
      poses = np.loadtxt(cls.FIXTURES.ego_poses_path(sequence))
      frame_ids = poses[:,0]
      frame_ids = [int(f) for f in frame_ids]
      poses_raw = np.reshape(poses[:, 1:],[-1, 3, 4])
      frame_to_RT = dict(zip(frame_ids, poses_raw))
      cls._ego_pose_cache[sequence] = frame_to_RT
    
    if frame_id not in cls._ego_pose_cache[sequence]:
      # Poses are incomplete :(
      # TODO see if we can interpolate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      return None
    
    RT_world_to_ego = cls._ego_pose_cache[sequence][frame_id]
    return datum.Transform.from_transformation_matrix(
                RT_world_to_ego,
                src_frame='world',
                dest_frame='ego')


  ## Private API: Camera Images

  @classmethod
  def _iter_camera_image_uris(cls, base_uri):
    for camera in cls.FIXTURES.CAMERAS:
      frame_ids = cls.FIXTURES.get_camera_frame_ids(base_uri.segment_id, camera)
      for frame_id in frame_ids:
        yield base_uri.replaced(
                topic=cls.CAMERA_NAME_TO_TOPIC[camera],
                timestamp=cls._frame_id_to_timestamp(frame_id),
                extra={
                  'kitti-360.frame_id': str(frame_id),
                  'kitti-360.camera': camera,
                })

  @classmethod
  def _create_camera_image(cls, uri):
    path = cls.FIXTURES.cam0_poses_path(uri.segment_id)

    # need calib!
    
    img_path = cls.FIXTURES.camera_image_path(
                  uri.segment_id,
                  uri.extra['kitti-360.camera'],
                  cls._get_frame_id(uri))

    return datum.StampedDatum(
            uri=uri)


  ## Private API: Point Clouds

  @classmethod
  def _iter_point_cloud_uris(cls, base_uri):
    frame_ids = cls.FIXTURES.get_raw_scan_frame_ids(
                    base_uri.segment_id, 'velodyne_points')
    timestamps_path = cls.FIXTURES.velodyne_timestamps_path(base_uri.segment_id)
    # TODO Use timestamps ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for frame_id in frame_ids:
      yield base_uri.replaced(
              topic='lidar',
              timestamp=cls._frame_id_to_timestamp(frame_id),
              extra={
                'kitti-360.frame_id': str(frame_id),
              })

    frame_ids = cls.FIXTURES.get_raw_scan_frame_ids(
                    base_uri.segment_id, 'sick_points')
    timestamps_path = cls.FIXTURES.sick_timestamps_path(base_uri.segment_id)
    # TODO Use timestamps ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for frame_id in frame_ids:
      yield base_uri.replaced(
              topic='laser|sick',
              timestamp=cls._frame_id_to_timestamp(frame_id),
              extra={
                'kitti-360.frame_id': str(frame_id),
              })

    # TODO: fused clouds ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      
  @classmethod
  def _create_point_cloud(cls, uri):

    # need calib!
    if uri.topic == 'lidar':
      vel_path = cls.FIXTURES.velodyne_cloud_path(
                      uri.segment_id, cls._get_frame_id(uri))
      cloud = np.fromfile(vel_path, dtype=np.float32)
      cloud = np.reshape(cloud, [-1, 4])
    elif uri.topic == 'laser|sick':
      sick_path = cls.FIXTURES.sick_cloud_path(
                      uri.segment_id, cls._get_frame_id(uri))
      cloud = np.fromfile(sick_path, dtype=np.float32)
      cloud = np.reshape(cloud, [-1, 2])
      cloud = np.concatenate([
                  np.zeros_like(cloud[:, 0:1]), # ? no x-dimension?
                  -cloud[:, 0:1],
                  cloud[:, 1:2]
                ],
                axis=1)
    else:
      raise ValueError(uri)

    return datum.StampedDatum(
            uri=uri)


  ## Private API: Cuboids
  
  @classmethod
  def _get_raw_cuboids_for_segment(cls, sequence):
    if not hasattr(cls, '_seq_to_raw_cuboids_cache'):
      cls._seq_to_raw_cuboids_cache = {}
    if not sequence in cls._seq_to_raw_cuboids_cache:
      import xmltodict
      path = cls.FIXTURES.cuboids_path(sequence)
      d = xmltodict.parse(open(path).read())
      objects = d['opencv_storage']
      obj_name_to_value = dict(
        (k, kitti_360_3d_bboxes_get_parsed_node(v))
        for (k, v) in objects.items())
      objs = [
        dict(v, obj_name=k)
        for (k, v) in obj_name_to_value.items()
      ]
      cls._seq_to_raw_cuboids_cache[sequence] = objs
    return cls._seq_to_raw_cuboids_cache[sequence]

  @classmethod
  def _iter_cuboid_uris(cls, base_uri):
    raw_cuboids = cls._get_raw_cuboids_for_segment(base_uri.segment_id)
    frame_ids = sorted(set(
      itertools.chain.from_iterable(
        range(c['start_frame'], c['end_frame'] + 1)
        for c in raw_cuboids)))
    for frame_id in frame_ids:
      yield base_uri.replaced(
              topic='labels|cuboids',
              timestamp=cls._frame_id_to_timestamp(frame_id),
              extra={
                'kitti-360.frame_id': str(frame_id),
              })
  
  @classmethod
  def _create_cuboids(cls, uri):
    frame_id = cls._get_frame_id(uri)
    raw_cuboids = cls._get_raw_cuboids_for_segment(uri.segment_id)

    def is_in_current_frame(obj):
      return (
        # Every object has a frame range
        (obj['start_frame'] <= frame_id <= obj['end_frame']) and
        # Static objects are active for the entire frame range; dynamic
        #  objects have different annotations for each frame.
        (obj['is_static'] or (obj['active_frame_id'] == frame_id)))
    
    raw_cuboids = [
      obj for obj in raw_cuboids
      if is_in_current_frame(obj)
    ]


    cuboids = []
    for obj in raw_cuboids:
      # Kitti-360 cuboids are in the IMU (world) frame:
      # x = forward, y = right, z = down
      # And vertices are in the (weird) order:
      # +x +y +z
      # +x +y -z
      # +x -y +z
      # +x -y -z | psegs convention differs for rear face:
      # -x +y -z |        -x +y +z
      # -x +y +z |        -x +y -z
      # -x -y -z |        -x -y +z
      # -x -y +z |        -x -y -z

      # We'll permute the faces to match psegs convention.
      front_world = obj['cuboid'][[0, 1, 2, 3], :]
      rear_world = obj['cuboid'][[5, 4, 7, 6], :]

      # Now get dimensions
      w = np.linalg.norm(front_world[0, :] - front_world[2, :])
      l = np.linalg.norm(front_world[0, :] - rear_world[0, :])
      h = np.linalg.norm(front_world[0, :] - front_world[1, :])

      # Now get pose (in world frame)
      T_world = np.mean(obj['cuboid'], axis=0)

      # KITTI-360 Transform confounds R and S; we need to separate them.
      # See also https://math.stackexchange.com/a/1463487
      obj_sR = obj['transform'][:3, :3]
      sx = np.linalg.norm(obj_sR[:, 0])
      sy = np.linalg.norm(obj_sR[:, 1])
      sz = np.linalg.norm(obj_sR[:, 2])
      R_world = obj_sR.copy()
      R_world[:, 0] *= 1. / sx
      R_world[:, 1] *= 1. / sy
      R_world[:, 2] *= 1. / sz

      T_world_to_obj = datum.Transform.from_transformation_matrix(
                          np.column_stack([R_world, T_world]),
                          src_frame='world',
                          dest_frame='obj')

      # Convert pose to ego frame (PSegs standard)
      T_world_to_ego = cls._get_ego_pose(uri.segment_id, frame_id)
      if not T_world_to_ego:
        # Labels are in world frame, so can't put them in ego frame
        continue

      T_obj_from_ego = (
        T_world_to_obj.get_inverse() @ T_world_to_ego).get_inverse()
      
      cuboids.append(datum.Cuboid(
          track_id=obj['instanceId'],
          category_name=obj['k360_class_name'],
          ps_category='TODO', # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
          timestamp=cls._frame_id_to_timestamp(frame_id),
          length_meters=l,
          width_meters=w,
          height_meters=h,
          obj_from_ego=T_obj_from_ego,
          ego_pose=T_world_to_ego,
          extra={
            'kitt-360.cuboid.index': str(obj['index']),
            'kitt-360.cuboid.label': str(obj['label']),
            'kitt-360.cuboid.category': str(obj['category']),
            'kitt-360.cuboid.start_frame': str(obj['start_frame']),
            'kitt-360.cuboid.end_frame': str(obj['end_frame']),
          }))

    return datum.StampedDatum(uri=uri, cuboids=cuboids)



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

        Optional KITTI-360 data dirs:

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
