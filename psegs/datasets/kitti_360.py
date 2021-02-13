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

from oarphpy import util as oputil

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

  FRONT_CAMERAS = ('image_00', 'image_01')
  FISHEYE_CAMERAS = ('image_02', 'image_03')

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
  def camera_timestamps_path(cls, sequence, camera_name):
    return (
      cls.ROOT / 'data_2d_raw' / sequence / camera_name / 'timestamps.txt')

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
      cls.ROOT / 'data_3d_raw' / 
        sequence / 'sick_points' / 'timestamps.txt')

  @classmethod
  def get_raw_scan_frame_ids(cls, sequence, sensor):
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
  def get_fused_scan_frame_ids(cls, sequence):
    paths = []
    paths += oputil.all_files_recursive(
      str(cls.ROOT / 'data_3d_semantics' / sequence / 'static'),
      pattern='*.ply')
    paths += oputil.all_files_recursive(
      str(cls.ROOT / 'data_3d_semantics' / sequence / 'dynamic'),
      pattern='*.ply')
    
    fnames = [
      os.path.split(path)[-1].split('.')[0]
      for path in paths
      if not oputil.is_stupid_mac_file(path)
    ]

    # 004631_004927.ply -> (4631, 4927)
    frame_intervals = [
      tuple(int(v) for v in fnames.split('.')[0].split('_'))
      for fname in fnames
    ]
    assert all(len(fi) == 2 for fi in frame_intervals)

    frame_ids = sorted(set(
      range(start, end + 1) for (start, end) in frame_intervals))
    return frame_ids

  @classmethod
  def get_fused_scan_frame_id_to_chan_to_path(cls, sequence):
    def build_frame_id_to_path(channel):
      paths = oputil.all_files_recursive(
                str(cls.ROOT / 'data_3d_semantics' / sequence / channel),
                pattern='*.ply')

      paths = [p for p in paths if not oputil.is_stupid_mac_file(p)]
      fnames = [
        os.path.split(path)[-1].split('.')[0]
        for path in paths
      ]

      # E.g. 004631_004927.ply -> (4631, 4927)
      frame_intervals = [
        tuple(int(v) for v in fname.split('.')[0].split('_'))
        for fname in fnames
      ]
      assert all(len(fi) == 2 for fi in frame_intervals)

      frame_id_to_path = {}
      for ((start, end), path) in zip(frame_intervals, paths):
        for frame_id in range(start, end + 1):
          frame_id_to_path[frame_id] = path
      return frame_id_to_path
    
    from collections import defaultdict
    frame_id_to_chan_to_path = defaultdict(dict)
    for frame_id, path in build_frame_id_to_path('static').items():
      frame_id_to_chan_to_path[frame_id]['static'] = path
    for frame_id, path in build_frame_id_to_path('dynamic').items():
      frame_id_to_chan_to_path[frame_id]['dynamic'] = path
    
    return frame_id_to_chan_to_path


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

def kitti_360_timestamps_to_nanostamps(txt):
  def line_to_nanostamp(line):
    # Timestamps are in the format:
    # YYYY-MM-DD HH:MM::SS.fffffffff (ISO 8601 format)
    # FMI https://github.com/autonomousvision/kitti360Scripts/blob/081c08b34a14960611f459f23a0ad049542205c6/kitti360scripts/devkits/accumuLaser/src/commons.cpp#L158
    # Numpy can parse these directly!
    t = np.datetime64(line.strip())
    return t.astype(np.uint64)
  
  lines = txt.split('\n')
  return [line_to_nanostamp(l) for l in lines if l]

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

  ### Camera Extrinsics (Ego is world / IMU)

  cam_left_rect_to_ego = attr.ib(type=datum.Transform, default=None)
  cam_right_rect_to_ego = attr.ib(type=datum.Transform, default=None)
  cam_left_fisheye_to_ego = attr.ib(type=datum.Transform, default=None)
  cam_right_fisheye_to_ego = attr.ib(type=datum.Transform, default=None)

  ### Camera Intrinsics (Rectified)

  cam0_K = attr.ib(type=np.ndarray, default=np.zeros((3, 3)))
  cam1_K = attr.ib(type=np.ndarray, default=np.zeros((3, 3)))

  ### Laser/Lidar Extrinsics

  cam_left_rect_to_velo = attr.ib(type=datum.Transform, default=None)

  sick_to_velo = attr.ib(type=datum.Transform, default=None)

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

    calib = cls()

    ## Extrinsics

    calib.sick_to_velo = datum.Transform.from_transformation_matrix(
              str_to_RT(calib_sick_to_velo),
              src_frame='laser|sick',
              dest_frame='lidar')

    # # It appears this one has an incorrect name-- in the KITTI-360 code, ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # the authors always use the inverse.
    # cam_left_raw_from_velo = datum.Transform.from_transformation_matrix(
    #   str_to_RT(calib_cam_to_velo))
    
    calib.cam_left_rect_to_velo = datum.Transform.from_transformation_matrix(
              str_to_RT(calib_cam_to_velo),
              src_frame='camera|left_rect',
              dest_frame='lidar')
    # calib.cam_left_rect_to_velo = cam_left_raw_from_velo.get_inverse()~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # calib.cam_left_rect_to_velo.src_frame = 'camera|left_rect'
    # calib.cam_left_rect_to_velo.dest_frame = 'lidar'

    # Tr cam -> ego
    lines = [l.strip() for l in calib_cam_to_pose.split('\n')]
    cam_to_sRT = dict(l.split(':') for l in lines if l)
    calib.cam_left_rect_to_ego = datum.Transform.from_transformation_matrix(
              str_to_RT(cam_to_sRT['image_00']),
              dest_frame='ego',
              src_frame='camera|left_rect')
    calib.cam_right_rect_to_ego = datum.Transform.from_transformation_matrix(
              str_to_RT(cam_to_sRT['image_01']),
              dest_frame='ego',
              src_frame='camera|right_rect')
    calib.cam_left_fisheye_to_ego = datum.Transform.from_transformation_matrix(
              str_to_RT(cam_to_sRT['image_02']),
              dest_frame='ego',
              src_frame='camera|left_fisheye')
    calib.cam_right_fisheye_to_ego = datum.Transform.from_transformation_matrix(
              str_to_RT(cam_to_sRT['image_03']),
              dest_frame='ego',
              src_frame='camera|left_fisheye')

    ## Intrinsics

    # https://github.com/autonomousvision/kitti360Scripts/blob/081c08b34a14960611f459f23a0ad049542205c6/kitti360scripts/helpers/project.py#L76

    lines = [
      l.strip() for l in perspective.split('\n') if 'calib_time' not in l
    ]
    perspective_kv = dict(l.split(':') for l in lines if l)
    calib.cam0_K = str_to_arr(perspective_kv['P_rect_00'], (3, 4))[:3, :3]
    calib.cam1_K = str_to_arr(perspective_kv['P_rect_01'], (3, 4))[:3, :3]
    
    return calib


###############################################################################
### StampedDatumTable Impl

class KITTI360SDTable(StampedDatumTableBase):

  FIXTURES = Fixtures


  INCLUDE_FISHEYES = False
  """bool: Should we emit label datums for the fisheye / side cameras?
  At the time of writing, the distortion parameters for the fisheyes are
  not available, so we can't make much use of the images / labels for
  these cameras.
  """

  INCLUDE_FUSED_CLOUDS = False
  """bool: Should we emit label datums for the kitti-360 fused lidar clouds?
  Note: these are the fused clouds in the `data_3d_semantics` portion of
  KITTI-360.
  """

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

  @classmethod
  def _get_all_segment_uris(cls):
    uris = [
      datum.URI(
        dataset='kitti-360',
        split='train',
        segment_id=seq)
      for seq in cls.FIXTURES.TRAIN_SEQUENCES
    ]
    return uris

  @classmethod
  def _create_datum_rdds(cls, spark, existing_uri_df=None, only_segments=None):
    
    ## First build a set of sequences to read ...

    # from psegs.spark import Spark
    # from oarphpy.spark import cluster_cpu_count
    
    util.log.info("Creating datums for KITTI-360 ...")

    seg_uris = cls.get_all_segment_uris()
    if only_segments:
      util.log.info(
        "Filtering to only %s segments" % len(only_segments))
      seg_uris = [
        uri for uri in seg_uris
        if any(
          suri.soft_matches_segment(uri) for suri in only_segments)
      ]
    
    ## ... now construct datum RDDS in chunks.
    URIS_PER_PARTITION = 256
      # Requires about 256 Megabytes of memory per chunk

    # from oarphpy import util as oputil
    datum_rdds = []
    for seg_uri in seg_uris:
      uris = cls.get_uris_for_sequence(seg_uri.segment_id)

      # Some datums are more expensive to create than others, so distribute
      # them evenly in the RDD
      uris = sorted(uris, key=lambda u: u.timestamp)
      uris = uris[:1500]

      seg_span_sec = 1e-9 * (uris[-1].timestamp - uris[0].timestamp)

      n_partitions = max(1, int(len(uris) / URIS_PER_PARTITION))

      util.log.info(
        "... seq %s has %s URIs spanning %2.f sec, creating %s slices ..." % (
          seg_uri.segment_id, len(uris), seg_span_sec, n_partitions))
      
      uri_rdd = spark.sparkContext.parallelize(uris, numSlices=n_partitions)

      # Are we trying to resume? Filter URIs if necessary.
      if existing_uri_df is not None:
        def to_datum_id(obj):
          return (
            obj.dataset,
            obj.split,
            obj.segment_id,
            obj.topic,
            obj.timestamp)
        key_uri_rdd = uri_rdd.map(lambda u: (to_datum_id(u), u))
        existing_keys = existing_uri_df.rdd.map(to_datum_id)
        uri_rdd = key_uri_rdd.subtractByKey(existing_keys).map(
                                        lambda kv: kv[0])

      datum_rdd = uri_rdd.map(cls.create_stamped_datum)
      
      # from pyspark import StorageLevel
      # datum_rdd = datum_rdd.persist(StorageLevel.DISK_ONLY) # hacks? ~~~~~~~~~~~~~~~~~~~
      datum_rdds.append(datum_rdd)
    
    util.log.info("... partitioned datums into %s RDDs." % len(datum_rdds))
    return datum_rdds


    #   URIS_PER_TASK



    # import itertools
    # iter_tasks = itertools.chain.from_iterable(
    #   ((seg_uri, p) for p in range(cls.PARTITIONS_PER_SEGMENT))
    #   for seg_uri in seg_uris)
    
    
    # for task_chunk in oputil.ichunked(iter_tasks, TASKS_PER_RDD):
    #   util.log.info([(str(u), p) for u, p in task_chunk])
    #   task_rdd = spark.sparkContext.parallelize(task_chunk)
    #   def iter_uris_for_task(task):
    #     seg_uri, partition = task
    #     uris = cls.get_uris_for_sequence(seg_uri.segment_id)
    #     for i, uri in enumerate(uris):
    #       if (i % cls.PARTITIONS_PER_SEGMENT) == partition:
    #         yield uri
      
    #   uri_rdd = task_rdd.flatMap(iter_uris_for_task)

      
      
    #   # Some datums are more expensive to materialize than others.  Force
    #   # a repartition to avoid stragglers.
    #   uri_rdd = uri_rdd.repartition(TASKS_PER_RDD)
    #   util.log.info(uri_rdd.count())

    #   datum_rdd = uri_rdd.map(cls.create_stamped_datum)
    #   datum_rdds.append(datum_rdd)
    # return datum_rdds




  ## Private API: Utils

  @classmethod
  def _get_frame_id(cls, uri):
    return int(uri.extra['kitti-360.frame_id'])

  CAMERA_NAME_TO_TOPIC = {
    'image_00': 'camera|left_rect',
    'image_01': 'camera|right_rect',
    'image_02': 'camera|left_fisheye',
    'image_03': 'camera|right_fisheye',
  }

  @classmethod
  def _get_calib(cls):
    if not hasattr(cls, '_calib'):
      def open_and_read(rpath):
        return open(cls.FIXTURES.filepath(rpath)).read()

      calib_cam_to_pose = open_and_read('calibration/calib_cam_to_pose.txt')
      calib_cam_to_velo = open_and_read('calibration/calib_cam_to_velo.txt')
      calib_sick_to_velo = open_and_read('calibration/calib_sick_to_velo.txt')
      perspective = open_and_read('calibration/perspective.txt')

      cls._calib = Calibration.from_kitti_360_strs(
                    calib_cam_to_pose,
                    calib_cam_to_velo,
                    calib_sick_to_velo,
                    perspective)
    return cls._calib

  @classmethod
  def _get_nanostamp(cls, sequence, channel, frame_id):
    if not hasattr(cls, '_seq_to_chan_to_ts'):
      cls._seq_to_chan_to_ts = {}
    
    if sequence not in cls._seq_to_chan_to_ts:
      def read_ts(chan):
        if chan in cls.CAMERA_NAME_TO_TOPIC.keys():
          path = cls.FIXTURES.camera_timestamps_path(sequence, chan)
        elif chan == 'velodyne':
          path = cls.FIXTURES.velodyne_timestamps_path(sequence)
        elif chan == 'sick':
          path = cls.FIXTURES.sick_timestamps_path(sequence)
        else:
          raise ValueError(chan)
        txt = open(path, 'r').read()
        return kitti_360_timestamps_to_nanostamps(txt)
      
      cls._seq_to_chan_to_ts[sequence] = dict(
        (chan, read_ts(chan))
        for chan in (
          ['velodyne', 'sick'] + list(cls.CAMERA_NAME_TO_TOPIC.keys())))
    
    return cls._seq_to_chan_to_ts[sequence][channel][frame_id]

  ## Private API: Ego Pose

  @classmethod
  def _iter_ego_pose_uris(cls, base_uri):
    poses = np.loadtxt(cls.FIXTURES.ego_poses_path(base_uri.segment_id))
    frame_ids = poses[:,0]
    frame_ids = [int(f) for f in frame_ids]
    for frame_id in frame_ids:
      # KITTI-360 poses are derived from their own lidar-heavy SLAM;
      # we'll use lidar timestamps for ego poses for convenience.
      timestamp = cls._get_nanostamp(
                      base_uri.segment_id, 'velodyne', frame_id)
      yield base_uri.replaced(
              topic='ego_pose',
              timestamp=timestamp,
              extra={'kitti-360.frame_id': str(frame_id)})

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
      # Poses are incomplete :(  There are even gaps for several seconds.
      return None
    
    RT_world_to_ego = cls._ego_pose_cache[sequence][frame_id]
    return datum.Transform.from_transformation_matrix(
                RT_world_to_ego,
                src_frame='world',
                dest_frame='ego')


  ## Private API: Camera Images

  @classmethod
  def _iter_camera_image_uris(cls, base_uri):
    cameras = list(cls.FIXTURES.FRONT_CAMERAS)
    if cls.INCLUDE_FISHEYES:
      cameras += list(cls.FIXTURES.FISHEYE_CAMERAS)
    for camera in cameras:
      frame_ids = cls.FIXTURES.get_camera_frame_ids(base_uri.segment_id, camera)
      for frame_id in frame_ids:
        yield base_uri.replaced(
                topic=cls.CAMERA_NAME_TO_TOPIC[camera],
                timestamp=
                  cls._get_nanostamp(base_uri.segment_id, camera, frame_id),
                extra={
                  'kitti-360.frame_id': str(frame_id),
                  'kitti-360.camera': camera,
                })

  @classmethod
  def _create_camera_image(cls, uri):
    frame_id = cls._get_frame_id(uri)
    calib = cls._get_calib()

    # TODO: use the camera0 pose? not sure why separate from IMU / ego pose
    # path = cls.FIXTURES.cam0_poses_path(uri.segment_id)
    
    img_path = cls.FIXTURES.camera_image_path(
                  uri.segment_id,
                  uri.extra['kitti-360.camera'],
                  cls._get_frame_id(uri))
    # image_png = open(img_path, 'rb').read()
    import imageio
    image_factory = lambda: imageio.imread(img_path)
    
    from psegs.util import misc
    width, height = misc.get_png_wh(open(img_path, 'rb').read(100))

    K = np.eye(3, 3)
    if uri.topic == 'camera|left_rect':
      K = calib.cam0_K
      ego_to_sensor = calib.cam_left_rect_to_ego.get_inverse()
    elif uri.topic == 'camera|right_rect':
      K = calib.cam1_K
      ego_to_sensor = calib.cam_right_rect_to_ego.get_inverse()
    elif uri.topic == 'camera|left_fisheye':
      K = calib.cam0_K # NB: no K for fisheyes -- this is just for debugging
      ego_to_sensor = calib.cam_left_fisheye_to_ego.get_inverse()
    elif uri.topic == 'camera|right_fisheye':
      K = calib.cam0_K # NB: no K for fisheyes -- this is just for debugging
      ego_to_sensor = calib.cam_right_fisheye_to_ego.get_inverse()
    else:
      raise ValueError(uri)

    ego_pose = cls._get_ego_pose(uri.segment_id, frame_id)
    ci = datum.CameraImage(
          sensor_name=uri.topic,
          # image_png=bytearray(image_png),
          image_factory=image_factory,
          width=width,
          height=height,
          timestamp=uri.timestamp,
          ego_pose=ego_pose or datum.Transform(),
          K=K,
          ego_to_sensor=ego_to_sensor,
          extra={'kitti-360.has-valid-ego-pose': str(bool(ego_pose))})
    return datum.StampedDatum(uri=uri, camera_image=ci)


  ## Private API: Point Clouds

  @classmethod
  def _get_fused_scan_idx(cls, sequence):
    if not hasattr(cls, '_fused_scan_idx'):
      cls._fused_scan_idx = {}
    if not sequence in cls._fused_scan_idx:
      frame_id_to_chan_to_path = (
        cls.FIXTURES.get_fused_scan_frame_id_to_chan_to_path(sequence))
      cls._fused_scan_idx[sequence] = frame_id_to_chan_to_path
    return cls._fused_scan_idx[sequence]

  @classmethod
  def _iter_point_cloud_uris(cls, base_uri):
    frame_ids = cls.FIXTURES.get_raw_scan_frame_ids(
                    base_uri.segment_id, 'velodyne_points')
    for frame_id in frame_ids:
      yield base_uri.replaced(
              topic='lidar',
              timestamp=
                cls._get_nanostamp(base_uri.segment_id, 'velodyne', frame_id),
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
              timestamp=
                cls._get_nanostamp(base_uri.segment_id, 'sick', frame_id),
              extra={
                'kitti-360.frame_id': str(frame_id),
              })

    if cls.INCLUDE_FUSED_CLOUDS:
      frame_id_to_chan_to_path = cls._get_fused_scan_idx(base_uri.segment_id)
      for frame_id, chan_to_path in frame_id_to_chan_to_path.items():
        # Fused clouds are in the world frame, so they are only useful for
        # frames where we have an ego pose
        has_pose = (
          cls._get_ego_pose(base_uri.segment_id, frame_id) is not None)
        if has_pose:
          for chan in chan_to_path.keys():
            yield base_uri.replaced(
                topic='lidar|fused_' + chan,
                timestamp=
                  cls._get_nanostamp(base_uri.segment_id, 'velodyne', frame_id),
                extra={
                  'kitti-360.frame_id': str(frame_id),
                })

  @classmethod
  def _create_point_cloud(cls, uri):
    frame_id = cls._get_frame_id(uri)
    calib = cls._get_calib()

    velo_to_ego = (
      calib.cam_left_rect_to_ego @ calib.cam_left_rect_to_velo.get_inverse())

    if uri.topic == 'lidar':
      vel_path = cls.FIXTURES.velodyne_cloud_path(
                      uri.segment_id, cls._get_frame_id(uri))
      def _get_vel_cloud(path):
        cloud = np.fromfile(path, dtype=np.float32)
        cloud = np.reshape(cloud, [-1, 4])
        return cloud

      cloud_factory = lambda: _get_vel_cloud(vel_path)

      ego_to_sensor = velo_to_ego.get_inverse()
    
    elif uri.topic == 'laser|sick':
      sick_path = cls.FIXTURES.sick_cloud_path(
                      uri.segment_id, cls._get_frame_id(uri))
      
      def _get_sick_cloud(path):
        cloud = np.fromfile(path, dtype=np.float32)
        cloud = np.reshape(cloud, [-1, 2])
        cloud = np.concatenate([
                    np.zeros_like(cloud[:, 0:1]), # ? no x-dimension?
                    -cloud[:, 0:1],
                    cloud[:, 1:2],
                  ],
                  axis=1)
        return cloud
      
      cloud_factory = lambda: _get_sick_cloud(sick_path)
      
      sick_from_ego = calib.sick_to_velo['laser|sick', 'lidar'] @ velo_to_ego
      ego_to_sensor = sick_from_ego.get_inverse()
    
    elif uri.topic in ('lidar|fused_static', 'lidar|fused_dynamic'):
      
      def _get_fused_cloud(uri, frame_id):
        T_world_to_ego = cls._get_ego_pose(uri.segment_id, frame_id)
        assert T_world_to_ego is not None, "Programming error: no pose %s" % uri
        # T_world_to_velo = (velo_to_ego.get_inverse() @ T_world_to_ego).get_inverse()
        T_world_to_velo = (T_world_to_ego @ velo_to_ego).get_inverse()
        
        chan = 'static' if 'static' in uri.topic else 'dynamic'
        frame_id_to_chan_to_path = cls._get_fused_scan_idx(uri.segment_id)
        path = frame_id_to_chan_to_path[frame_id][chan]

        import open3d
        pcd = open3d.io.read_point_cloud(str(cpath))
        cloud = np.asarray(pcd.points)
        cloud = T_world_to_velo.apply(cloud).T

        return cloud

      cloud_factory = lambda: _get_fused_cloud(uri, frame_id)

      ego_to_sensor = velo_to_ego.get_inverse()

    else:
      raise ValueError(uri)

    ego_pose = cls._get_ego_pose(uri.segment_id, frame_id)

    pc = datum.PointCloud(
          sensor_name=uri.topic,
          timestamp=uri.timestamp,
          # cloud=cloud,
          cloud_factory=cloud_factory,
          ego_to_sensor=ego_to_sensor,
          ego_pose=ego_pose or datum.Transform(),
          extra={'kitti-360.has-valid-ego-pose': str(bool(ego_pose))})
    return datum.StampedDatum(uri=uri, point_cloud=pc)


  ## Private API: Cuboids
  
  @classmethod
  def _get_raw_cuboids_for_segment(cls, sequence):
    if not hasattr(cls, '_seq_to_raw_cuboids_cache'):
      cls._seq_to_raw_cuboids_cache = {}
    if not sequence in cls._seq_to_raw_cuboids_cache:
      import xmltodict
      path = cls.FIXTURES.cuboids_path(sequence)
      d = xmltodict.parse(open(path).read())
        # NB: this parse() takes 4-6 sec and we can't make it any faster
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
      # KITTI-360 poses are derived from their own lidar-heavy SLAM;
      # we'll use lidar timestamps for cuboid labels because we believe
      # they are posed in this SLAM-based world frame.
      timestamp = cls._get_nanostamp(
                      base_uri.segment_id, 'velodyne', frame_id)
      yield base_uri.replaced(
              topic='labels|cuboids',
              timestamp=timestamp,
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
      T_obj_from_ego.src_frame = 'ego' # fixme ... our names here are broken but matrix is right? ~~~~~~~~~~~~~~~``
      T_obj_from_ego.dest_frame = 'obj'

      # Instance IDs are only distinct within each class:
      # https://github.com/autonomousvision/kitti360Scripts/issues/5#issuecomment-722217758
      # https://github.com/autonomousvision/kitti360Scripts/blob/feb142bd8d99df6cbde77ae46b17e912cb3a633b/kitti360scripts/helpers/annotation.py#L37
      track_id = 1000 * obj['semanticId'] + obj['instanceId']

      cuboids.append(datum.Cuboid(
          track_id=track_id,
          category_name=obj['k360_class_name'],
          ps_category='TODO', # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
          timestamp=uri.timestamp,
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
