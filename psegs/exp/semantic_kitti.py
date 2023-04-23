# Copyright 2021 Maintainers of PSegs
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

import copy
import os

import numpy as np

from psegs import datum
from psegs import util
from psegs.conf import C
from psegs.table.sd_table_factory import StampedDatumTableFactory



def parse_calibration(path):
  """Parse a calibration file and return a map to 4x4 Numpy matrices.
  Important keys returned:
  * Tr - the lidar to camera static transform
  * P2 - the left camera projective matrix P
  Based upon https://github.com/PRBonn/semantic-kitti-api/blob/9b5feda3b19ea560a298493b9a5ebebe0cbe2cc2/generate_sequential.py#L14
  """
  calib = {}

  with open(path) as f:
    for line in f:
      key, mat_str = line.strip().split(":")
      values = [float(v) for v in mat_str.strip().split()]
      mat = np.zeros((4, 4))
      mat[0, 0:4] = values[0:4]
      mat[1, 0:4] = values[4:8]
      mat[2, 0:4] = values[8:12]
      mat[3, 3] = 1.0
      calib[key] = mat
  return calib

def parse_poses(path):
  """Read a SemanticKITTI (per-scan) poses file and return a list of 4x4
  homogenous RT matrices that express world-to-left-camera transforms.  The
  index of this list is implicitly the scan ID.

  Based upon: https://github.com/PRBonn/semantic-kitti-api/blob/9b5feda3b19ea560a298493b9a5ebebe0cbe2cc2/generate_sequential.py#L42
  """
  poses = []
  with open(path) as f:
    for line in f:
      values = [float(v) for v in line.strip().split()]
      mat = np.zeros((4, 4))
      mat[0, 0:4] = values[0:4]
      mat[1, 0:4] = values[4:8]
      mat[2, 0:4] = values[8:12]
      mat[3, 3] = 1.0
      poses.append(mat)
  return poses



class Fixtures(object):

  # Please follow the instructions posted on the SemanticKITTI website to
  # obtain the data:
  # http://www.semantic-kitti.org/dataset.html#download
  # Additionally, if you wish to study optical flow, you'll want to expand
  # the KITTI zip file `data_odometry_color.zip`.
  # Extract the data as described to a directory and symlink that directory
  # path here:
  ROOT = C.EXT_DATA_ROOT / 'semantic_kitti'

  # Deduced from:
  # https://github.com/PRBonn/semantic-kitti-api/blob/c2d7712964a9541ed31900c925bf5971be2107c2/auxiliary/SSCDataset.py#L20
  SK_SPLIT_SEQUENCES = {
      "train": [
        "00", "01", "02", "03", 
        # "04", -- We ignore sequence 04 because it has no clouds with only
        #            static points
        "05", "06", "07", "09", "10"],
      "valid": ["08"],
      "test": [
        "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
  }

  SK_MOVING_LABELS = (
      252, # "moving-car"
      253, # "moving-bicyclist"
      254, # "moving-person"
      255, # "moving-motorcyclist"
      256, # "moving-on-rails"
      257, # "moving-bus"
      258, # "moving-truck"
      259, # "moving-other-vehicle"
  )

  @classmethod
  def get_scene_basepath(cls, seq):
    return os.path.join(cls.ROOT, 'dataset/sequences', seq)

  @classmethod
  def get_seq_to_nscans(cls):
    if not hasattr(cls, '_seq_to_nscans'):
      cls._seq_to_nscans = {}
      for seq in cls.SK_SPLIT_SEQUENCES['train']:
        scene_base = cls.get_scene_basepath(seq)
        last_vel = max(os.listdir(os.path.join(scene_base, 'velodyne')))
        n_scans = int(last_vel.replace('.bin', '')) + 1
        util.log.info("Found Sequence %s with %s scans" % (seq, n_scans))
        cls._seq_to_nscans[seq] = n_scans
      util.log.info("Found %s total scans" % sum(cls._seq_to_nscans.values()))
    return cls._seq_to_nscans

  @classmethod
  def get_calibration(cls, seq):
    scene_base = cls.get_scene_basepath(seq)
    return parse_calibration(os.path.join(scene_base, 'calib.txt'))

  @classmethod
  def get_poses(cls, seq):
    scene_base = cls.get_scene_basepath(seq)
    return parse_poses(os.path.join(scene_base, "poses.txt"))

  @classmethod
  def get_moving_mask_for_scan(cls, seq, scan_id):
    scene_base = cls.get_scene_basepath(seq)
    scan_name = str(scan_id).rjust(6, '0')
    labels_path = os.path.join(scene_base, 'labels', scan_name + '.label')
    labels = np.fromfile(labels_path, dtype=np.uint32)
    labels = labels.reshape((-1))
    sem_label = labels & 0xFFFF  # semantic label in lower half
    inst_label = labels >> 16    # instance id in upper half
      # NB: 22 / 252 is chase car in scene 08 !!!

    moving_mask = np.logical_or.reduce(
      tuple((sem_label == c) for c in cls.SK_MOVING_LABELS))
    return moving_mask

  @classmethod
  def read_scan_get_cloud(
          cls, seq, scan_id, remove_movers=True, filter_ego=True):
    scan_name = str(scan_id).rjust(6, '0')
    scene_base = cls.get_scene_basepath(seq)
    scan_path = os.path.join(scene_base, 'velodyne', scan_name + '.bin')

    # Read the raw lidar
    lidar_bytes = open(scan_path, 'rb').read()
    lidar = np.frombuffer(lidar_bytes, dtype=np.float32).reshape((-1, 4))
    cloud = np.ones(lidar.shape)  # need homogenous for transforms later
    cloud[:, 0:3] = lidar[:, 0:3]

    if remove_movers:
        # Clean out points for anything moving
        moving_mask = cls.get_moving_mask_for_scan(seq, scan_id)
        cloud = cloud[~moving_mask]#[:, :3]
    
    if filter_ego:
        pass # TODO ~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    return cloud


class SemanticKITTISDTable(StampedDatumTableFactory):
  
  FIXTURES = Fixtures

  ONLY_FRAMES_WITH_NO_MOVERS = True
  
  @classmethod
  def _get_all_segment_uris(cls):
    return [
      datum.URI(
        dataset='semantikitti', # fixme ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
        split='train',
        segment_id=str(seq))
      for seq in cls.FIXTURES.get_seq_to_nscans().keys()
    ]

  @classmethod
  def _create_datum_rdds(cls, spark, existing_uri_df=None, only_segments=None):
    assert existing_uri_df is None, "Resume feature not supported"
    
    seg_uris = cls.get_all_segment_uris()
    if only_segments:
      util.log.info("Filtering to only %s segments" % len(only_segments))
      seg_uris = [
        uri for uri in seg_uris
        if any(
          suri.soft_matches_segment_of(uri) for suri in only_segments)
      ]
    
    SK_SEQ_TO_NSCANS = cls.FIXTURES.get_seq_to_nscans()
    datum_rdds = []
    for seg_uri in seg_uris:
      seq = seg_uri.segment_id
      if cls.ONLY_FRAMES_WITH_NO_MOVERS:
        util.log.info(
          "Finding scans for sequence %s with no moving points ..." % seq) # FIXME we should keep these .... ?
        n_scans = SK_SEQ_TO_NSCANS[seq]
        slices = max(1, n_scans // 100)
        task_rdd = spark.sparkContext.parallelize(
          range(n_scans), numSlices=slices)
        scan_has_no_movers = (
          lambda scan_id: (
            not cls.FIXTURES.get_moving_mask_for_scan(seq, scan_id).any()))
        scans_no_movers = task_rdd.filter(scan_has_no_movers).collect()
        util.log.info(
          "... sequence %s has %s scans with no movers." % (
            seq, len(scans_no_movers)))
        scan_ids = scans_no_movers
      else:
        scan_ids = list(range(SK_SEQ_TO_NSCANS[seq]))
      
      # for testing scan_ids = scan_ids[:500]
      tasks = [(seg_uri, scan_id) for scan_id in scan_ids]
      
      # Emit camera_image RDD
      ctask_rdd = spark.sparkContext.parallelize(tasks)
      datum_rdd = ctask_rdd.map(lambda t: cls.create_camera_frame(*t))
      datum_rdds.append(datum_rdd)
      
      # Emit ego_pose RDD
      ptask_rdd = spark.sparkContext.parallelize(tasks)
      datum_rdd = ptask_rdd.map(lambda t: cls.create_ego_pose(*t))
      datum_rdds.append(datum_rdd)
      
      # Emit velodyne cloud RDD
      pctask_rdd = spark.sparkContext.parallelize(tasks)
      datum_rdd = pctask_rdd.map(lambda t: cls.create_point_cloud_in_world(*t))
      datum_rdds.append(datum_rdd)
  
    return datum_rdds
  
  @classmethod
  def _get_calib(cls, seq):
    if not hasattr(cls, '_calib'):
      cls._calib = {}
    if seq not in cls._calib:
      cls._calib[seq] = cls.FIXTURES.get_calibration(seq)
    return cls._calib[seq]
  
  @classmethod
  def _get_poses(cls, seq):
    if not hasattr(cls, '_poses'):
      cls._poses = {}
    if seq not in cls._poses:
      cls._poses[seq] = cls.FIXTURES.get_poses(seq)
    return cls._poses[seq]
  
  @classmethod
  def create_camera_frame(cls, base_uri, scan_id):
    seq = base_uri.segment_id
    calib = cls._get_calib(seq)
    
    uri = copy.deepcopy(base_uri)
    uri.topic = 'camera|left_rect'
    uri.timestamp = int(scan_id) # HACK!
    uri.extra['semantic_kitti.scan_id'] = str(scan_id)

    scene_base = cls.FIXTURES.get_scene_basepath(seq)
    scan_name = str(scan_id).rjust(6, '0')
    img_path = os.path.join(scene_base, 'image_2/', scan_name + '.png')
    assert os.path.exists(img_path), (
      "Did you remember to expand data_odometry_color.zip ? "
      "%s not found" % img_path)
    with open(img_path, 'rb') as f:
      width, height = util.get_png_wh(f.read(100))
                          # NB: Util only needs the first few bytes
    
    import imageio
    image_factory = lambda: imageio.imread(img_path)
    
    # HACK!!!  This is actually P !!!
    K = calib['P2']
    
    # hack! this is lidar to cam
    ego_to_sensor = datum.Transform.from_transformation_matrix(
        calib['Tr'], src_frame='lidar', dest_frame=uri.topic)
    
    sd_ego_pose = cls.create_ego_pose(base_uri, scan_id)
    ego_pose = sd_ego_pose.transform
    ci = datum.CameraImage(
        sensor_name=uri.topic,
        image_factory=image_factory,
        width=width,
        height=height,
        timestamp=uri.timestamp,
        ego_pose=ego_pose,
        K=K,
        ego_to_sensor=ego_to_sensor,
        extra={'semantic_kitti.scan_id': str(scan_id)})
    return datum.StampedDatum(uri=uri, camera_image=ci)
  
  @classmethod
  def create_ego_pose(cls, base_uri, scan_id):
    seq = base_uri.segment_id
    poses = cls._get_poses(seq)
    
    uri = copy.deepcopy(base_uri)
    uri.topic = 'ego_pose'
    uri.timestamp = int(scan_id) # HACK!
    uri.extra['semantic_kitti.scan_id'] = str(scan_id)
    
    # Move cloud into the world frame
    calib = cls._get_calib(seq)
    Tr = calib["Tr"]
    Tr_inv = np.linalg.inv(Tr)
    cam2_pose = poses[scan_id]
    pose = np.matmul(Tr_inv, np.matmul(cam2_pose, Tr))

    # # Hack! believe ego frame is lidar here?
    # poses = cls._get_poses(seq)
    # ego_pose = datum.Transform.from_transformation_matrix(
    #     poses[scan_id], src_frame='world', dest_frame='ego')

    # Hack! believe ego frame is lidar here?
    ego_pose = datum.Transform.from_transformation_matrix(
        pose, src_frame='world', dest_frame='ego')

    return datum.StampedDatum(uri=uri, transform=ego_pose)
  
  @classmethod
  def create_point_cloud_in_world(cls, base_uri, scan_id):
    
    uri = copy.deepcopy(base_uri)
    uri.topic = 'lidar|world' + (
      '_cleaned' if cls.ONLY_FRAMES_WITH_NO_MOVERS else '')
    uri.timestamp = int(scan_id) # HACK!
    uri.extra['semantic_kitti.scan_id'] = str(scan_id)
    
    sd_ego_pose = cls.create_ego_pose(base_uri, scan_id)
    ego_pose = sd_ego_pose.transform

    # # The cloud is in world coords so the ego pose is effectively null
    # ego_pose = datum.Transform()
    
    def _get_cloud(seq, sid):
      cloud = cls.FIXTURES.read_scan_get_cloud(
            seq,
            sid,
            remove_movers=cls.ONLY_FRAMES_WITH_NO_MOVERS)
      
      # # Move cloud into the world frame
      # calib = cls._get_calib(seq)
      # all_poses = cls._get_poses(seq)
      # Tr = calib["Tr"]
      # Tr_inv = np.linalg.inv(Tr)
      # cam2_pose = all_poses[sid]
      # pose = np.matmul(Tr_inv, np.matmul(cam2_pose, Tr))
      # cloud = np.matmul(pose, cloud.T).T
      
      return cloud

    pc = datum.PointCloud(
      sensor_name=uri.topic,
      timestamp=uri.timestamp,
      cloud_factory=lambda: _get_cloud(base_uri.segment_id, scan_id),
      ego_to_sensor=datum.Transform(), # Hack! ego frame is lidar frame
      ego_pose=ego_pose,
      extra={'semantic_kitti.scan_id': str(scan_id)})
    return datum.StampedDatum(uri=uri, point_cloud=pc)

