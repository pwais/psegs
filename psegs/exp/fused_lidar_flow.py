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

import numpy as np
import pandas as pd

from oarphpy import util as oputil

from psegs import datum
from psegs import util
from psegs.conf import C
from psegs.table.sd_table import StampedDatumTableBase

import numpy as np

###############################################################################
### Lidar Fusion


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
  return in_box, cloud_obj
    # cloud_obj = cloud_obj[in_box]
    
    # return cloud_obj

def _move_clouds_to_ego_and_concat(point_clouds):
  clouds_ego = []
  for pc in point_clouds:
    c = pc.get_cloud()[:, :3] # TODO: can we keep colors?
    c_ego = pc.ego_to_sensor.get_inverse().apply(c).T
    clouds_ego.append(c_ego)
  if clouds_ego:
    cloud_ego = np.vstack(clouds_ego)
  else:
    cloud_ego = np.zeros((0, 3))
  return cloud_ego

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

class TaskLidarCuboidCameraDFFactory(object):
  """Adapt a `StampedDatumTable` to a table of "tasks" where each task has
  all point clouds, cuboids, and camera images associated with a specific
  time point or event.  (Some datasets, like KITTI and Waymo OD, refer to
  these as "frames"; we use the word "task" to distinguish these groupings
  from the unrelated frames-of-reference e.g. lidar frame, world frame, etc).

  The Task IDs implicitly represent numerical timestamps (but need not be
  real timestamps-- the Stamped Datum URIs have real timestamps).  Task IDs
  must be in chronological order: task T+1 should be an event after task T.
  The IDs need not be dense (there can be gaps e.g. 1, 2, 3, 7, 8, 9) but
  any gaps may impact synthetic flow generation.

  Create a DataFrame here so that it's cheap to omit columns when needed.
  """

  SRC_SD_TABLE = None

  def build_df_for_segment(cls, spark, segment_uri):
    """The DF should have rows like:
    Row(task_id | list[Point_cloud] | list[cuboids] | list[camera_image])"""
    raise NotImplementedError()



class FusedLidarCloudTableBase(StampedDatumTableBase):
  """
  read SD table and emit a topic lidar|world_cloud; write plys to disk

  """

  TASK_DF_FACTORY = None

  FUSER_ALGO_NAME = 'naive_cuboid_scrubber'

  # Some datasets are not amenable to fused object clouds; use this member
  # to opt those datasets out of object clouds.
  HAS_OBJ_CLOUDS = True

  SPLITS = ['train']

  ### Subclass API

  @classmethod
  def _should_build_world_cloud(cls, segment_uri):
    return not cls.world_cloud_path(segment_uri).exists()

  @classmethod
  def _should_build_obj_clouds(cls, segment_uri):
    if not cls.HAS_OBJ_CLOUDS:
      return False
    seg_basepath = cls.obj_cloud_seg_basepath(segment_uri)
    return oputil.missing_or_empty(str(seg_basepath))


  ## Utils

  @classmethod
  def SRC_SD_T(cls):
    return cls.TASK_DF_FACTORY.SRC_SD_TABLE

  @classmethod
  def _get_task_lidar_cuboid_rdd(cls, spark, segment_uri):
    # "need RDD of Row(task_id | list[Point_cloud] | list[cuboids])"
    df = cls.TASK_DF_FACTORY.build_df_for_segment(spark, segment_uri)
    df = df.select('task_id', 'point_clouds', 'cuboids')
    T = cls.SRC_SD_T()
    unpacked_rdd = df.rdd.map(T.from_row)
    return unpacked_rdd

  ## World Cloud Fusion

  @classmethod
  def _filter_ego_vehicle(cls, cloud_ego):
    """Optionally filter self-returns in cloud in the ego frame for some
    datasets (e.g. NuScenes)"""
    return cloud_ego

  @classmethod
  def _get_cleaned_world_cloud(cls, point_clouds, cuboids):
    cleaned_clouds = []
    pruned_counts = []
    for pc in point_clouds:
      cloud = pc.get_cloud()[:, :3] # TODO: can we keep colors?
      cloud_ego = pc.ego_to_sensor.get_inverse().apply(cloud).T
    
      cloud_ego = cls._filter_ego_vehicle(cloud_ego)

      # Filter out all cuboids
      n_before = cloud_ego.shape[0]
      for cuboid in cuboids:
        in_box, _ = get_point_idx_in_cuboid(cuboid, cloud_ego=cloud_ego)
        cloud_ego = cloud_ego[~in_box]
      n_after = cloud_ego.shape[0]

      T_world_to_ego = pc.ego_pose
      cloud_world = T_world_to_ego.apply(cloud_ego).T # why is this name backwards?? -- hmm works for nusc too

      cleaned_clouds.append(cloud_world)
      pruned_counts.append(n_before - n_after)
    return np.vstack(cleaned_clouds), pruned_counts

  @classmethod
  def _task_to_clean_world_cloud(cls, task_row):
    pcs = task_row.point_clouds
    cuboids = task_row.cuboids
    world_cloud, pruned_counts = cls._get_cleaned_world_cloud(pcs, cuboids)
    return world_cloud, pruned_counts


  # Object Cloud Fusion

  @classmethod
  def _get_object_cloud(cls, cuboid, point_clouds=None, cloud_ego=None):
    if cloud_ego is None:
      assert point_clouds is not None
      cloud_ego = _move_clouds_to_ego_and_concat(point_clouds)
    cloud_ego = cloud_ego.copy()

    in_box, cloud_obj = get_point_idx_in_cuboid(cuboid, cloud_ego=cloud_ego)
    return cloud_obj[in_box]
  
  @classmethod
  def _task_to_obj_cloud_rows(cls, task_row):
    pcs = task_row.point_clouds
    cuboids = task_row.cuboids

    cloud_ego = _move_clouds_to_ego_and_concat(pcs)
    for cuboid in cuboids:
      obj_cloud = cls._get_object_cloud(cuboid, cloud_ego=cloud_ego)
      from pyspark import Row
      from oarphpy.spark import RowAdapter
      yield RowAdapter.to_row(
              Row(
                track_id=cuboid.track_id,
                task_id=task_row.task_id,
                obj_cloud=obj_cloud))

  @classmethod
  def _get_fused_cloud(cls, obj_cloud_rows):
    T = cls.SRC_SD_T()
    obj_clouds = [T.from_row(r.obj_cloud) for r in obj_cloud_rows]
    return np.vstack(obj_clouds)

  ### Core Impl

  ## World Clouds

  @classmethod
  def world_clouds_base_path(cls):
    return C.DATA_ROOT / 'fused_world_clouds' / cls.FUSER_ALGO_NAME

  @classmethod
  def world_cloud_path(cls, segment_uri):
    return (cls.world_clouds_base_path() / 
              segment_uri.dataset / segment_uri.split / 
              segment_uri.segment_id / 'fused_world.ply')

  @classmethod
  def _build_world_cloud(cls, spark, segment_uri, culi_tasks_rdd):
    if not cls._should_build_world_cloud(segment_uri):
      return
    
    dest_path = cls.world_cloud_path(segment_uri)
    oputil.mkdir(dest_path.parent)
    util.log.info("Building world cloud to %s ..." % dest_path)

    n_tasks = culi_tasks_rdd.count()
    util.log.info("... fusing %s tasks ..." % n_tasks)
    world_cloud_rdd = culi_tasks_rdd.map(cls._task_to_clean_world_cloud)
    
    # Force fusion before we pull clouds to the driver (prevent an OOM)
    from pyspark import StorageLevel
    world_cloud_rdd = world_cloud_rdd.persist(StorageLevel.MEMORY_AND_DISK)
    t = oputil.ThruputObserver(name='FuseWorldClouds', n_total=n_tasks)
    t.start_block()
    n_bytes_n_pruned_rdd = world_cloud_rdd.map(
      lambda c_pc: (oputil.get_size_of_deep(c_pc[0]), c_pc[1]))
    n_bytes_n_pruned = n_bytes_n_pruned_rdd.collect()
    n_bytes = sum(nn[0] for nn in n_bytes_n_pruned)
    t.stop_block(n=n_tasks, num_bytes=n_bytes)    
    t.maybe_log_progress(every_n=1)
    n_pruned = np.array([nn[1] for nn in n_bytes_n_pruned])
    util.log.info("Total points pruned: %s" % np.sum(n_pruned))
    util.log.info("Avg pts pruned per cloud: %s" % np.mean(n_pruned))

    iclouds = world_cloud_rdd.toLocalIterator(prefetchPartitions=True)
    iclouds = oputil.ThruputObserver.to_monitored_generator(
                iclouds, name='CollectWorldClouds',
                log_freq=500, n_total=n_tasks, log_on_del=True) # fixme log_on_del!~~~~~~~~~
    
    # Pull one partition at a time to avoid a driver OOM
    clouds = list(iclouds)#world_cloud_rdd.collect()
    if len(clouds) > 0:
      world_cloud = np.vstack(clouds)
    else:
      world_cloud = np.zeros((0, 3))
    util.log.info(
      "... computed world cloud for %s of shape %s (%.2f GB) ..." % (
        segment_uri.segment_id, world_cloud.shape,
        1e-9 * oputil.get_size_of_deep(world_cloud)))
    
    util.log.info("... writing ply to %s ..." % dest_path)
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_cloud)
    o3d.io.write_point_cloud(str(dest_path), pcd)
    util.log.info("... done writing ply.")

  ## Object Clouds

  @classmethod
  def obj_cloud_base_path(cls):
    return C.DATA_ROOT / 'fused_obj_clouds' / cls.FUSER_ALGO_NAME
  
  @classmethod
  def obj_cloud_seg_basepath(cls, segment_uri):
    return (cls.obj_cloud_base_path() / 
              segment_uri.dataset / segment_uri.split / 
              segment_uri.segment_id )

  @classmethod
  def obj_cloud_path(cls, segment_uri, track_id):
    base_path = cls.obj_cloud_seg_basepath(segment_uri)
    track_id = track_id.replace(':', '-') # slugify ................................
    fname = 'fused_obj.%s.ply' % track_id
    return base_path / fname

  @classmethod
  def obj_cloud_idx_path(cls, segment_uri):
    base_path = cls.obj_cloud_seg_basepath(segment_uri)
    return base_path / "cloud_idx.csv"

  @classmethod
  def _build_object_clouds(cls, spark, segment_uri, culi_tasks_rdd):
    if not cls._should_build_obj_clouds(segment_uri):
      return

    # Map task rows to rows of (partial) obj cloud s.  We create a dataframe
    # from the result because it will better help Spark budget memory.
    util.log.info("Pruning object clouds ...")
    obj_cloud_row_rdd = culi_tasks_rdd.flatMap(cls._task_to_obj_cloud_rows)
    obj_cloud_df = spark.createDataFrame(obj_cloud_row_rdd)
    obj_cloud_df = obj_cloud_df.persist()
    n_rows = obj_cloud_df.count()
    n_tracks = obj_cloud_df.select('track_id').distinct().count()
    util.log.info("... have %s clouds of %s objects to fuse ..." % (
      n_rows, n_tracks))
    
    
    # Now fuse object clouds and save to disk
    seg_basepath = cls.obj_cloud_seg_basepath(segment_uri)
    util.log.info("... fusing obj clouds, saving to %s ..." % seg_basepath)
    grouped = obj_cloud_df.rdd.groupBy(lambda r: r.track_id)
    
    def _fuse_and_save(track_id_irows):
      track_id, irows = track_id_irows
      obj_cloud = cls._get_fused_cloud(irows)
      n_points = obj_cloud.shape[0]

      dest_path = cls.obj_cloud_path(segment_uri, track_id)
      oputil.mkdir(dest_path.parent)
      
      if n_points > 0:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_cloud)
        o3d.io.write_point_cloud(str(dest_path), pcd)

      idx_row = {
        'track_id': str(track_id),
        'n_points': n_points,
        'cloud_shape': obj_cloud.shape,
        'cloud_MBytes': 1e-6 * oputil.get_size_of_deep(obj_cloud),
        'path': dest_path,
      }
      return idx_row

    all_idx_rows = grouped.map(_fuse_and_save).collect()
    idx_df = pd.DataFrame(all_idx_rows)

    util.log.info("Saved fused clouds to %s. Wrote %2.f MBytes" % (
      seg_basepath, idx_df['cloud_MBytes'].sum()))
    util.log.info("Stats:")

    with pd.option_context('display.max_colwidth', None):
      util.log.info(str(idx_df))
    
    idx_df.to_csv(cls.obj_cloud_idx_path(segment_uri))

  @classmethod
  def _build_fused_clouds(cls, spark, segment_uris=None):
    util.log.info("%s building fused clouds ..." % cls.__name__)

    segment_uris = segment_uris or cls.SRC_SD_T().get_all_segment_uris()
    n_segs = len(segment_uris)
    util.log.info("... have %s segments to fuse ..." % n_segs)

    t = oputil.ThruputObserver(name='FuseEachSegment', n_total=n_segs)
    for suri in segment_uris:
      t.start_block()# TODO add a log to stop block or give a loop body wrapper ....
      util.log.info("... working on %s ..." % suri.segment_id)

      need_to_work = (
        cls._should_build_world_cloud(suri) or 
        cls._should_build_obj_clouds(suri))
      if need_to_work:
        culi_tasks_rdd = cls._get_task_lidar_cuboid_rdd(spark, suri)
        cls._build_world_cloud(spark, suri, culi_tasks_rdd)
        cls._build_object_clouds(spark, suri, culi_tasks_rdd)
      else:
        util.log.info(
          "... skipping %s; world and obj clouds done" % suri.segment_id)
        util.log.info("World Cloud: %s" % cls.world_cloud_path(suri))
        util.log.info("Obj Clouds: %s" % cls.obj_cloud_seg_basepath(suri))
      
      t.stop_block(n=1)
      t.maybe_log_progress(every_n=1)

    util.log.info("... %s done fusing clouds." % cls.__name__)

  ### StampedDatumTable Impl

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
    
    cls._build_fused_clouds(spark, segment_uris=seg_uris)

    sds = []
    for seg_uri in seg_uris:
      sds.append(cls._create_world_cloud_sd(seg_uri))
      if cls.HAS_OBJ_CLOUDS:
        sds.extend(cls._create_obj_cloud_sds(seg_uri))
    datum_rdds = [spark.sparkContext.parallelize(sds)]
    return datum_rdds

  @classmethod
  def _create_world_cloud_sd(cls, segment_uri):
    uri = copy.deepcopy(segment_uri)
    uri.topic = 'lidar|world_fused|' + cls.FUSER_ALGO_NAME
      
    wcloud_path = cls.world_cloud_path(segment_uri)
      
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
    return datum.StampedDatum(uri=uri, point_cloud=pc)

  @classmethod
  def _create_obj_cloud_sds(cls, segment_uri):
    idx_df = pd.read_csv(cls.obj_cloud_idx_path(segment_uri))
    for _, row in idx_df.iterrows():
      track_id = str(row['track_id'])
      cloud_path = row['path']
      n_points = row['n_points']

      uri = copy.deepcopy(segment_uri)
      uri.topic = 'lidar|objects_fused|' + cls.FUSER_ALGO_NAME
      uri.track_id = track_id

      def _load_cloud(path=cloud_path): # force capture by copy
        import open3d as o3d
        import numpy as np
        pcd = o3d.io.read_point_cloud(str(path))
        return np.asarray(pcd.points)

      if n_points > 0:
        cloud_factory = _load_cloud
      else:
        cloud_factory = lambda: np.zeros((0, 3))

      pc = datum.PointCloud(
        sensor_name=track_id,
        timestamp=uri.timestamp,
        cloud_factory=cloud_factory,
        ego_to_sensor=datum.Transform(), # Hack! cloud is in world frame
        ego_pose=datum.Transform())
      yield datum.StampedDatum(uri=uri, point_cloud=pc)



###############################################################################
### Optical Flow from Fused Lidar

class OpticalFlowRender(object):

  TASK_DF_FACTORY = None

  RENDERER_ALGO_NAME = 'naive_shortest_ray'

  MAX_TASKS_SEED = 1337
  MAX_TASKS_PER_SEGMENT = -1
  TASK_OFFSETS = (1, 5)

  render_func = None # TODO for python notebook drafting .........................

  @classmethod
  def 
