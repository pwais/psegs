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

from oarphpy import util as oputil

from psegs import datum
from psegs import util
from psegs.conf import C
from psegs.table.sd_table import StampedDatumTableBase

import numpy as np


### Utils & Core Fusion Algo Pieces

def get_point_idx_in_cuboid(cuboid, pc=None, cloud_ego=None):
  if cloud_ego is None:
    assert pc is not None
    cloud = pc.get_cloud()
    cloud_ego = pc.ego_to_sensor.get_inverse().apply(cloud[:, :3]).T
    
  cloud_obj = cuboid.obj_from_ego.get_inverse().apply(cloud_ego).T # TODO check with bev plots ...
#     print('cuboid.obj_from_ego', cuboid.obj_from_ego.translation)
#     print(cuboid.track_id, 'cuboid.obj_from_ego', cuboid.obj_from_ego.translation, 'cloud_obj', np.mean(cloud_obj, axis=0))
  
  # Filter to just object
  hl, hw, hh = .5 * cuboid.length_meters, .5 * cuboid.width_meters, .5 * cuboid.height_meters
  in_box = (#np.where(
      (cloud_obj[:, 0] >= -hl) & (cloud_obj[:, 0] <= hl) &
      (cloud_obj[:, 1] >= -hw) & (cloud_obj[:, 1] <= hw) &
      (cloud_obj[:, 2] >= -hh) & (cloud_obj[:, 2] <= hh))
#     print(in_box, hl, hw, hh, np.mean(cloud_obj, axis=0))
#     print('in_box', in_box[0].sum())
  return in_box
    # cloud_obj = cloud_obj[in_box]
    
    # return cloud_obj


# def iter_cleaned_world_clouds(SD_Table, task):
#     pcs = [T.from_row(rr) for rr in task.pcs]
#     cuboids = [T.from_row(c) for c in task.cuboids]
#     for pc in pcs:
        
#         # for nusc we gotta filter the returns off ego vehicle !!!!!!!!!!!! -- note we may get these in lidarseg
#         cloud = pc.cloud[:, :3]
# #         cloud = cloud[np.where(  ~(
# #                         (cloud[:, 0] <= 1.5) & (cloud[:, 0] >= -1.5) &  # Nusc lidar +x is +right
# #                         (cloud[:, 1] <= 2.5) & (cloud[:, 0] >= -2.5) &  # Nusc lidar +y is +forward
# #                         (cloud[:, 1] <= 1.5) & (cloud[:, 0] >= -1.5)   # Nusc lidar +z is +up
# #         ))]
#         # KITTI EDIT
        
        
#         cloud_ego = pc.ego_to_sensor.get_inverse().apply(cloud[:, :3]).T
    
#         # Filter out all cuboids
#         n_before = cloud_ego.shape[0]
#         for cuboid in cuboids:
#             xform = cuboid.obj_from_ego.get_inverse() # TODO check with bev plots ...
#             cloud_obj = xform.apply(cloud_ego).T 
    
#             # Filter to just object
#             hl, hw, hh = .5 * cuboid.length_meters, .5 * cuboid.width_meters, .5 * cuboid.height_meters
#             outside_box = np.where(
#                     np.logical_not(
#                         (cloud_obj[:, 0] >= -hl) & (cloud_obj[:, 0] <= hl) &
#                         (cloud_obj[:, 1] >= -hw) & (cloud_obj[:, 1] <= hw) &
#                         (cloud_obj[:, 2] >= -hh) & (cloud_obj[:, 2] <= hh)))
#             cloud_obj = cloud_obj[outside_box]
            
#             cloud_ego = xform.get_inverse().apply(cloud_obj).T
        
#         T_world_to_ego = pc.ego_pose
#         cloud_world = T_world_to_ego.apply(cloud_ego).T # why is this name backwards?? -- hmm works for nusc too

#         print('filtered', cloud_world.shape[0] - n_before)
#         yield cloud_world
    
# # iclouds = culi_tasks_df.repartition(5000).rdd.flatMap(iter_cleaned_world_clouds).toLocalIterator(prefetchPartitions=True) # KITTI EDIT iterator


class FusedWorldCloudTableBase(StampedDatumTableBase):
  """
  read SD table and emit a topic lidar|world_cloud; write plys to disk

  """

  SRC_SD_TABLE = None

  FUSER_ALGO_NAME = 'naive_cuboid_scrubber'

  SPLITS = ['train']

  # Subclass API

  @classmethod
  def _get_task_lidar_cuboid_rdd(cls, spark, segment_uri):
    assert False, "need RDD of Row(task_id | list[Point_cloud] | list[cuboids])"

  @classmethod
  def _filter_ego_vehicle(cls, cloud_ego):
    """Optionally filter self-returns in cloud in the ego frame for some
    datasets (e.g. NuScenes)"""
    return cloud_ego

  @classmethod
  def _get_cleaned_world_cloud(cls, point_clouds, cuboids):
    cleaned_clouds = []
    for pc in point_clouds:
      cloud = pc.get_cloud()[:, :3] # TODO: can we keep colors?
      cloud_ego = pc.ego_to_sensor.get_inverse().apply(cloud).T
    
      cloud_ego = cls._filter_ego_vehicle(cloud_ego)

      # Filter out all cuboids
      n_before = cloud_ego.shape[0]
      for cuboid in cuboids:
        in_box = get_point_idx_in_cuboid(cuboid, cloud_ego=cloud_ego)
        cloud_ego = cloud_ego[~in_box]
      n_after = cloud_ego.shape[0]

      T_world_to_ego = pc.ego_pose
      cloud_world = T_world_to_ego.apply(cloud_ego).T # why is this name backwards?? -- hmm works for nusc too

      # util.log.info("Filtered %s -> %s" % (n_before, n_after))
      cleaned_clouds.append(cloud_world)
    return np.vstack(cleaned_clouds)

  @classmethod
  def _task_to_clean_world_cloud(cls, task_row):
    T = cls.SRC_SD_TABLE
    pcs = [T.from_row(rr) for rr in task_row.point_clouds]
    cuboids = [T.from_row(c) for c in task_row.cuboids]
    world_cloud = cls._get_cleaned_world_cloud(pcs, cuboids)
    return world_cloud


  # Core Impl

  @classmethod
  def world_clouds_base_path(cls):
    return C.DATA_ROOT / 'fused_world_clouds' / cls.FUSER_ALGO_NAME

  @classmethod
  def world_cloud_path(cls, segment_uri):
    return (cls.world_clouds_base_path() / 
              segment_uri.dataset / segment_uri.split / 
              segment_uri.segment_id / 'fused_world.ply')

  @classmethod
  def _build_world_clouds(cls, spark, segment_uris=None):
    """
    to do painted lidar: just pre-paint clouds in DF?

    make a dir dataroot / fused_world_clouds / dataset / split / segment_id
    for segment in segments:
      culi_tasks_df = cls._get_task_lidar_cuboid_df(segment)
    
      import numpy as np
      cfcloud = np.vstack(clouds)
      print('cleaned_fcloud', cfcloud.shape)

      

    """

    util.log.info("%s building fused world clouds ..." % cls.__name__)

    segment_uris = segment_uris or cls.SRC_SD_TABLE.get_all_segment_uris()
    n_segs = len(segment_uris)
    util.log.info("... have %s segments to fuse ..." % n_segs)
    isegs = oputil.ThruputObserver.to_monitored_generator(
              iter(segment_uris), name='FuseSegments', n_total=n_segs)
    for suri in isegs:
      util.log.info("... working on %s ..." % suri.segment_id)
      dest_path = cls.world_cloud_path(suri)
      if dest_path.exists():
        util.log.info("... have fused cloud; skipping! %s" % dest_path)
        continue
      oputil.mkdir(dest_path.parent)

      culi_tasks_rdd = cls._get_task_lidar_cuboid_rdd(spark, suri)
      world_cloud_rdd = culi_tasks_rdd.map(cls._task_to_clean_world_cloud)

      iclouds = world_cloud_rdd.toLocalIterator(prefetchPartitions=True)
      iclouds = oputil.ThruputObserver.to_monitored_generator(
                  iclouds, name='ComputeWorldClouds', log_freq=500)
      clouds = list(iclouds)
      if len(clouds) > 0:
        world_cloud = np.vstack(clouds)
      else:
        world_cloud = np.zeros((0, 3))
      util.log.info(
        "... computed world cloud for %s of shape %s (%.2f GB) ..." % (
          suri.segment_id, world_cloud.shape,
          1e-9 * oputil.get_size_of_deep(world_cloud)))
      
      util.log.info("... writing ply to %s ..." % dest_path)
      import open3d as o3d
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(world_cloud)
      o3d.io.write_point_cloud(str(dest_path), pcd)
      util.log.info("... done writing ply ...")

    util.log.info("... %s done fusing clouds." % cls.__name__)


  @classmethod
  def _get_all_segment_uris(cls):
    uris = cls.SRC_SD_TABLE.get_all_segment_uris()
    uris = [u for u in uris if u.split in cls.SPLITS]
    return uris

  @classmethod
  def _create_datum_rdds(
    cls, spark, existing_uri_df=None, only_segments=None):

    if existing_uri_df is not None:
      util.log.warn("Note: resume mode not supported in %s" % cls.__name__)

    seg_uris = cls.get_all_segment_uris()
    if only_segments:
        util.log.info("Filtering to only %s segments" % len(only_segments))
        seg_uris = [
            uri for uri in seg_uris
            if any(
              suri.soft_matches_segment(uri) for suri in only_segments)
        ]
    
    cls._build_world_clouds(spark, segment_uris=seg_uris)

    world_cloud_sds = []
    for seg_uri in seg_uris:
      uri = copy.deepcopy(seg_uri)
      uri.topic = 'lidar|world_fused|' + cls.FUSER_ALGO_NAME
      
      wcloud_path = cls.world_cloud_path(seg_uri)
      
      def _load_cloud(path):
        import open3d as o3d
        import numpy as np
        pcd = o3d.io.read_point_cloud(str(path))
        return np.asarray(pcd.points)

      pc = datum.PointCloud(
        sensor_name=uri.topic,
        timestamp=uri.timestamp,
        cloud_factory=lambda: _load_cloud(wcloud_path),
        ego_to_sensor=datum.Transform(), # Hack! cloud is in world frame
        ego_pose=datum.Transform())
      world_cloud_sds.append(datum.StampedDatum(uri=uri, point_cloud=pc))

    datum_rdds = [spark.sparkContext.parallelize(world_cloud_sds)]
    return datum_rdds



class FusedObjectFromCuboidTable(StampedDatumTableBase):
  """

  read SD table and emit a topic lidar|obj_clouds; write plys to disk

  """
  
  @classmethod
  def get_cleaned(cls):
    pass

