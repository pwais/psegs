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
import pandas as pd

from oarphpy import util as oputil

from psegs import datum
from psegs import util
from psegs.conf import C
from psegs.spark import Spark
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

  The (integer) Task IDs implicitly represent numerical timestamps (but need
  not be real timestamps-- the Stamped Datum URIs have real timestamps).  
  Task IDs must be in chronological order: task T+1 should be an event one 
  time-step after task T. The IDs need not be dense (there can be gaps
  e.g. 1, 2, 3, 7, 8, 9) but any gaps may impact synthetic flow generation.

  Create a DataFrame here so that it's cheap to omit columns when needed.
  """

  SRC_SD_TABLE = None

  @classmethod
  def table_schema(cls):
    """Return a copy of the expected table schema.  Subclasses only need this
    in rare cases, e.g. if one of the columns will always be empty / null"""
    if not hasattr(cls, '_schema'):
      from psegs.datum.stamped_datum import STAMPED_DATUM_PROTO
      from oarphpy.spark import RowAdapter
      from pyspark.sql import Row
      PROTO_ROW = Row(
                    task_id=0,
                    pc_sds=[STAMPED_DATUM_PROTO],
                    cuboids_sds=[STAMPED_DATUM_PROTO],
                    ci_sds=[STAMPED_DATUM_PROTO])
      cls._schema = RowAdapter.to_schema(PROTO_ROW)
    return cls._schema

  @classmethod
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
    uris = cls.SRC_SD_T().get_all_segment_uris()
    uris = [u for u in uris if (u.split in cls.SPLITS)]
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
      path = str(path)
      util.log.info("Reading world cloud %s GB at %s" % (
        1e-9 * os.path.getsize(path), path))
      pcd = o3d.io.read_point_cloud(path)
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
        sensor_name=uri.topic + '|' + track_id,
        timestamp=uri.timestamp,
        cloud_factory=cloud_factory,
        ego_to_sensor=datum.Transform(), # Hack! cloud is in world frame
        ego_pose=datum.Transform(),
        extra={'track_id': track_id})
      yield datum.StampedDatum(uri=uri, point_cloud=pc)



###############################################################################
### Optical Flow from Fused Lidar


###############################################################################
## FROM PAPER SCRATCH


## PSEGS


def color_to_opencv(color):
  r, g, b = np.clip(color, 0, 255).astype(int).tolist()
  return b, g, r

def rgb_for_distance(d_meters, period_meters=10.):
  """Given a distance `d_meters` or an array of distances, return an
  `np.array([r, g, b])` color array for the given distance (or a 2D array
  of colors if the input is an array)).  We choose a distinct hue every
  `period_meters` and interpolate between hues for `d_meters`.
  """
  from oarphpy.plotting import hash_to_rbg

  if not isinstance(d_meters, np.ndarray):
    d_meters = np.array([d_meters])
  
  SEED = 10 # Colors for 0 and 1 look too similar otherwise
  max_bucket = int(np.ceil(d_meters.max() / period_meters))
  bucket_to_color = np.array(
    [hash_to_rbg(bucket + SEED) for bucket in range(max_bucket + 2)])

  # Use numpy's indexing for fast "table lookup" of bucket ids (bids) in
  # the "table" bucket_to_color
  bucket_below = np.floor(d_meters / period_meters)
  bucket_above = bucket_below + 1

  color_below = bucket_to_color[bucket_below.astype(int)]
  color_above = bucket_to_color[bucket_above.astype(int)]

  # For each distance, interpolate to *nearest* color based on L1 distance
  d_relative = d_meters / period_meters
  l1_dist_below = np.abs(d_relative - bucket_below)
  l1_dist_above = np.abs(d_relative - bucket_above)

  colors = (
    (1. - l1_dist_below) * color_below.T + 
    (1. - l1_dist_above) * color_above.T)

  colors = colors.T
  if len(d_meters) == 1:
    return colors[0]
  else:
    return colors

def draw_xy_depth_px_in_image(img, pts, alpha=.7):
  """
  new!
  Draw a point cloud `pts` in `img`. Point color interpolates between
  standard colors for each 10-meter tick.

  Args:
    img (np.array): Draw in this image.
    pts (np.array): An array of N by 3 points in form
      (pixel x, pixel y, depth meters).
    dot_size (int): Size of the dot to draw for each point.
    alpha (float): Blend point color using weight [0, 1].
  """

  import cv2

  # OpenCV can't draw transparent colors, so we use the 'overlay image' trick:
  # First draw dots an an overlay...
  overlay = img.copy()

  pts = pts.copy()
  pts = pts[-pts[:, -1].argsort()]
    # short by distance descending; let colors of nearer points
    # override colors of farther points
  print(pts.shape)

  colors = rgb_for_distance(pts[:, 2])
  # print(colors.shape)
  colors = np.clip(colors, 0, 255).astype(int)
  # print(colors.shape)
  for i, ((x, y), color) in enumerate(zip(pts[:, :2].tolist(), colors.tolist())):
    x = int(round(x))
    y = int(round(y))
    if y >= overlay.shape[0] or x >= overlay.shape[1]:
        continue
    overlay[y, x, :] = color
#     print(color)
    
    # if i > 0 and ((i % 500000) == 0):
    #     print(i, flush=True)

  # Now blend!
  img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)











## Common Support

# collect img1, pose1
# collect img2, pose2
# collect all clouds (save in RAM)
# collect all labels / cuboids

def make_homo(cloud):
  if cloud.shape[-1] == 4:
    return cloud
  else:
    out = np.ones((cloud.shape[0], 4))
    out[:, :3] = cloud[:, :3]
    return out

import numba
from numba import jit

@jit(nopython=True)
def get_nearest_idx(uvd):
  max_u = int(np.rint(np.max(uvd[:, 0])))
  max_v = int(np.rint(np.max(uvd[:, 1])))
  nearest = np.full((max_u + 1, max_v + 1, 2), np.Inf)
  for r in range(uvd.shape[0]):
    d = uvd[r, 2]
    u = int(np.rint(uvd[r, 0]))
    v = int(np.rint(uvd[r, 1]))
    if d < nearest[u, v, 1]:
      nearest[u, v, 1] = d
      nearest[u, v, 0] = r

  rs = nearest[:, :, 0].flatten()
  return rs[rs != np.Inf].astype(np.int64)


def compute_optical_flows(
      world_cloud=np.zeros((0, 3)),
      T_ego2lidar=np.eye(4),
      T_lidar2cam=np.eye(4),
      P=np.eye(4),
      cam_height_pixels=0,
      cam_width_pixels=0,

      ego_pose1=np.eye(4),
      ego_pose2=np.eye(4),
      moving_1=np.zeros((0, 3)),
      moving_2=np.zeros((0, 3)),

      img1_factory=lambda: np.zeros((1, 1, 3)),
      img2_factory=lambda: np.zeros((1, 1, 3)),
      debug_title=''):
  
  h, w = cam_height_pixels, cam_width_pixels
  
  pose1 = np.linalg.inv(ego_pose1) # FIXME for Semantic KITTI ??
  pose2 = np.linalg.inv(ego_pose2) # FIXME for Semantic KITTI ??
  print('diff Tx_pose1->Tx_pose2', pose2[:, -1] - pose1[:, -1])
  
  hfused_world_cloud = make_homo(world_cloud)
  is_moving_ignore = np.zeros((hfused_world_cloud.shape[0], 1))
  

  # Add all moving things at t1 and t2 to environment; we'll mask them
  # TODO NO MASK!! ?? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
  hfused_world_cloud = np.vstack([
    hfused_world_cloud,
    make_homo(moving_1),
    make_homo(moving_2),
  ])
#     is_moving_ignore = np.vstack([
#       is_moving_ignore,
#       np.ones((m1.shape[0], 1)),
#       np.ones((m2.shape[0], 1)),
#     ])

  def world_to_uvdij_visible_t(pose):
    import time
    start = time.time()
    
    print('hfused_world_cloud', hfused_world_cloud.shape, 1e-9 * hfused_world_cloud.nbytes)
    xyz_ego_t = np.matmul(pose, hfused_world_cloud.T)
    print(time.time() - start, 'projected to xyz_ego_t')
#     print('xyz_ego_t mean min max', np.mean(xyz_ego_t, axis=1), np.min(xyz_ego_t, axis=1), np.max(xyz_ego_t, axis=1))


    uvd = P.dot(T_lidar2cam.dot( T_ego2lidar.dot( xyz_ego_t ) ) )
    uvd[0:2, :] /= uvd[2, :]
    uvd = uvd.T
    uvd = uvd[:, :3]
    ij = np.rint(uvd[:, (0, 1)]) # Group by rounded pixel coord; need orig (u, v) for sub-pixel flow
    uvdij = np.hstack([uvd, ij])
    print(time.time() - start, 'projected to uvd')

    in_cam_frustum = np.where(
        (uvdij[:, 0] >= 0) & 
        (uvdij[:, 0] <= w - 1) &
        (uvdij[:, 1] >= 0) & 
        (uvdij[:, 1] <= h - 1) &
        (uvdij[:, 2] >= 0.01))

    uvdij_in_cam = uvdij[in_cam_frustum]
#     uvdij, uvdij_in_cam = project_to_uvd(P, pose, hfused_world_cloud, T_lidar2cam, T_ego2lidar, w, h)
    print(time.time() - start, 'projected to uvd in cam', uvdij_in_cam.shape, 1e-9 * uvdij_in_cam.nbytes)

    
#     print('render using pandas %s ...' % (uvdij_in_cam.shape,))
#     import pandas as pd
#     start = time.time()
#     df = pd.DataFrame(uvdij_in_cam[:, 2:], columns=['d', 'i', 'j'])
#     df['id'] = df.index
#     nearest_idx = df.groupby(['i', 'j'])['d'].idxmin().to_numpy()
#     print(time.time() - start, 'done pandas, %s winners' % (nearest_idx.shape,))
#     print(nearest_idx[:10])
    
    print('render using numba %s ...' % (uvdij_in_cam.shape,))
    start = time.time()
    nearest_idx = get_nearest_idx(uvdij_in_cam)
    print(time.time() - start, 'done numba, %s winners' % (nearest_idx.shape,))
#     print(nearest_idx[:10])
    
    uvdij_visible = np.hstack([uvdij, np.zeros((uvdij.shape[0], 1))])
    idx = np.arange(uvdij_visible.shape[0])[in_cam_frustum][nearest_idx]
    uvdij_visible[idx, -1] += 1
       # visible: in the camera frustum, AND is nearest point for the pixel.
       # then we'll flow from that pt. TODO: try to average flows for a single pixel?
    print(time.time() - start, 'done select visible from numba', uvdij_visible.shape, 1e-9 * uvdij_visible.nbytes)

    # OK next task is to allow fused tables to have mask / ignore clouds
    # that blot out stuff in the flow.  add that and then we can run this junk
    # if len(all_moving_clouds_t1t2): # TODO NEED THIS FOR SEMANTIC KITTI ~~~~~~~~~~~~~~~~~
    #   uvdij_visible[np.where(is_moving_ignore == 1)[0], -1] = 0
    
    return uvdij_visible
    
    
  uvdij_visible1 = world_to_uvdij_visible_t(pose1)
  uvdij_visible2 = world_to_uvdij_visible_t(pose2)
  
  if debug_title:
    import imageio
    basepath = '/opt/psegs/test_run_output/' + debug_title
    debug = img1_factory().copy()
    draw_xy_depth_px_in_image(debug, uvdij_visible1[uvdij_visible1[:, -1] == 1][:, :3])
    print('project1')
    # imshow(debug)
    imageio.imwrite(basepath + '.img1.png' , debug)

    debug = img2_factory().copy()
    draw_xy_depth_px_in_image(debug, uvdij_visible2[uvdij_visible2[:, -1] == 1][:, :3])
    print('project2')
    # imshow(debug)
    imageio.imwrite(basepath + '.img2.png' , debug)
  
  # # old format -- need this to make flow map
  # visible_both = ((uvdij_visible1[:, -1] == 1) & (uvdij_visible2[:, -1] == 1))
  
  # visboth_uv1 = uvdij_visible1[visible_both, :2]
  # visboth_uv2 = uvdij_visible2[visible_both, :2]
  # ij_flow = np.hstack([
  #   uvdij_visible1[visible_both, 3:5], visboth_uv2 - visboth_uv1
  # ])
  # v2v_flow = np.zeros((h, w, 2))
  # xx = ij_flow[:, 0].astype(np.int)
  # yy = ij_flow[:, 1].astype(np.int)
  # v2v_flow[yy, xx] = ij_flow[:, 2:4]
  
  # v2o_flow = np.zeros((h, w, 2)) # ignore for now TODO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # o2v_flow = np.zeros((h, w, 2)) # ignore for now TODO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  # return v2v_flow, v2o_flow, o2v_flow

  # new format
  import time
  start = time.time()
  visible_either = ((uvdij_visible1[:, -1] == 1) | (uvdij_visible2[:, -1] == 1))
  uvdij1_visible_uvdij2_visible = np.hstack([
    uvdij_visible1[visible_either], uvdij_visible2[visible_either]
  ])
  print('uvdij1_visible_uvdij2_visible', time.time() - start, uvdij1_visible_uvdij2_visible.shape, 1e-9 * uvdij1_visible_uvdij2_visible.nbytes)

  return uvdij1_visible_uvdij2_visible



  
  
  
  # build array, each row is a 3d point at times t1 and t2
  # xyz t1 | xyz t2 | (is_moving - are xyz diff?) | img1 uvd | img2 uvd | is visible img1 | is visible img2
  # ** for moving stuff, careful to append correct xyz ...
  # ** for semantic kitti, need to make moving stuff verboten at both timesteps ...
  
  # then compute:
  # vis change | uvd img1 | uvd img2
  
  # the break that above into 3 flows
  
  # create fused unproj for img1
  # place moving objs for img1
  
  # create fused unproj for img2
  # place moving objs for img2


# def compute_optical_flows(
#     t1_i=0, img1=None, 
#     t2_i=1, img2=None,
#     pose1=None,
#     pose2=None,
#     fused_world_cloud=None, # pre-filter these to remove moving stuff !!
#     all_moving_objs=[],
#     all_moving_clouds_t1t2=[],
#     P=None,
#     T_lidar2cam=None,
#     T_ego2lidar=None):
  
#   h, w, c = img1.shape
  
#   pose1 = np.linalg.inv(pose1) # FIXME for Semantic KITTI
#   pose2 = np.linalg.inv(pose2) # FIXME for Semantic KITTI
#   print('diff Tx_pose1->Tx_pose2', pose2[:, -1] - pose1[:, -1])
  
#   hfused_world_cloud = make_homo(fused_world_cloud)
#   is_moving_ignore = np.zeros((hfused_world_cloud.shape[0], 1))
  
#   if len(all_moving_clouds_t1t2):
#     # Add all moving things at t1 and t2 to environment; we'll mask them
#     m1 = all_moving_clouds_t1t2[0]
#     m2 = all_moving_clouds_t1t2[1]
#     hfused_world_cloud = np.vstack([
#       hfused_world_cloud,
#       make_homo(m1),
#       make_homo(m2),
#     ])
# #     is_moving_ignore = np.vstack([
# #       is_moving_ignore,
# #       np.ones((m1.shape[0], 1)),
# #       np.ones((m2.shape[0], 1)),
# #     ])

#   def world_to_uvdij_visible_t(pose):
#     import time
#     start = time.time()
    
#     xyz_ego_t = np.matmul(pose, hfused_world_cloud.T)
#     print(time.time() - start, 'projected to xyz_ego_t')
# #     print('xyz_ego_t mean min max', np.mean(xyz_ego_t, axis=1), np.min(xyz_ego_t, axis=1), np.max(xyz_ego_t, axis=1))


#     uvd = P.dot(T_lidar2cam.dot( T_ego2lidar.dot( xyz_ego_t ) ) )
#     uvd[0:2, :] /= uvd[2, :]
#     uvd = uvd.T
#     uvd = uvd[:, :3]
#     ij = np.rint(uvd[:, (0, 1)]) # Group by rounded pixel coord; need orig (u, v) for sub-pixel flow
#     uvdij = np.hstack([uvd, ij])
#     print(time.time() - start, 'projected to uvd')

#     in_cam_frustum = np.where(
#         (uvdij[:, 0] >= 0) & 
#         (uvdij[:, 0] <= w - 1) &
#         (uvdij[:, 1] >= 0) & 
#         (uvdij[:, 1] <= h - 1) &
#         (uvdij[:, 2] >= 0.01))

#     uvdij_in_cam = uvdij[in_cam_frustum]
# #     uvdij, uvdij_in_cam = project_to_uvd(P, pose, hfused_world_cloud, T_lidar2cam, T_ego2lidar, w, h)
#     print(time.time() - start, 'projected to uvd in cam')

    
# #     print('render using pandas %s ...' % (uvdij_in_cam.shape,))
# #     import pandas as pd
# #     start = time.time()
# #     df = pd.DataFrame(uvdij_in_cam[:, 2:], columns=['d', 'i', 'j'])
# #     df['id'] = df.index
# #     nearest_idx = df.groupby(['i', 'j'])['d'].idxmin().to_numpy()
# #     print(time.time() - start, 'done pandas, %s winners' % (nearest_idx.shape,))
# #     print(nearest_idx[:10])
    
#     print('render using numba %s ...' % (uvdij_in_cam.shape,))
#     start = time.time()
#     nearest_idx = get_nearest_idx(uvdij_in_cam)
#     print(time.time() - start, 'done numba, %s winners' % (nearest_idx.shape,))
# #     print(nearest_idx[:10])
    
#     uvdij_visible = np.hstack([uvdij, np.zeros((uvdij.shape[0], 1))])
#     idx = np.arange(uvdij_visible.shape[0])[in_cam_frustum][nearest_idx]
#     uvdij_visible[idx, -1] += 1
#        # visible: in the camera frustum, AND is nearest point for the pixel.
#        # then we'll flow from that pt. TODO: try to average flows for a single pixel?

#     if len(all_moving_clouds_t1t2):
#       uvdij_visible[np.where(is_moving_ignore == 1)[0], -1] = 0
    
#     return uvdij_visible
    
    
#   uvdij_visible1 = world_to_uvdij_visible_t(pose1)
#   uvdij_visible2 = world_to_uvdij_visible_t(pose2)
  
#   if True:
#     debug = img1.copy()
#     draw_xy_depth_px_in_image(debug, uvdij_visible1[uvdij_visible1[:, -1] == 1][:, :3])
#     print('project1')
#     imshow(debug)
    
#     debug = img2.copy()
#     draw_xy_depth_px_in_image(debug, uvdij_visible2[uvdij_visible2[:, -1] == 1][:, :3])
#     print('project2')
#     imshow(debug)
  
#   visible_both = ((uvdij_visible1[:, -1] == 1) & (uvdij_visible2[:, -1] == 1))
  
#   visboth_uv1 = uvdij_visible1[visible_both, :2]
#   visboth_uv2 = uvdij_visible2[visible_both, :2]
#   ij_flow = np.hstack([
#     uvdij_visible1[visible_both, 3:5], visboth_uv2 - visboth_uv1
#   ])
#   v2v_flow = np.zeros((h, w, 2))
#   xx = ij_flow[:, 0].astype(np.int)
#   yy = ij_flow[:, 1].astype(np.int)
#   v2v_flow[yy, xx] = ij_flow[:, 2:4]
  
#   v2o_flow = np.zeros((h, w, 2)) # ignore for now
#   o2v_flow = np.zeros((h, w, 2)) # ignore for now
#   return v2v_flow, v2o_flow, o2v_flow
  
#   # build array, each row is a 3d point at times t1 and t2
#   # xyz t1 | xyz t2 | (is_moving - are xyz diff?) | img1 uvd | img2 uvd | is visible img1 | is visible img2
#   # ** for moving stuff, careful to append correct xyz ...
#   # ** for semantic kitti, need to make moving stuff verboten at both timesteps ...
  
#   # then compute:
#   # vis change | uvd img1 | uvd img2
  
#   # the break that above into 3 flows
  
#   # create fused unproj for img1
#   # place moving objs for img1
  
#   # create fused unproj for img2
#   # place moving objs for img2
    
    

## END FROM PAPER SCRATCH
###############################################################################


class RenderOFlowTasksWorker(object):
  
  def __init__(self, T_ego2lidar, fused_datum_sample, render_func):
    import threading
    self._shared = threading.Lock()
    self._track_id_to_fused_cloud = None
    self._world_cloud = None
    self.T_ego2lidar = T_ego2lidar
    self.fused_datum_sample = fused_datum_sample
    self.render_func = render_func
  
  def __getstate__(self):
    d = dict(self.__dict__)
    d.pop('_shared')
    d.pop('_track_id_to_fused_cloud')
    d.pop('_world_cloud')
    return d

  def __setstate__(self, d):
    for k, v in d.items():
      setattr(self, k, v)
    import threading
    self._shared = threading.Lock()
    self._track_id_to_fused_cloud = None
    self._world_cloud = None

  def get_track_id_to_fused_cloud(self):
    with self._shared:
      from oarphpy.spark import RowAdapter
      FROM_ROW = RowAdapter.from_row
      if self._track_id_to_fused_cloud is None:
        print('track_id_to_fused_cloud loading')
        track_id_to_fused_cloud = {}
        for pc in self.fused_datum_sample.lidar_clouds:
          if 'lidar|objects_fused' in pc.sensor_name:
            cucloud = FROM_ROW(pc)
            track_id = cucloud.extra['track_id']
            track_id_to_fused_cloud[track_id] = cucloud.get_cloud()
            print(track_id, track_id_to_fused_cloud[track_id].shape)
        print('track_id_to_fused_cloud', len(track_id_to_fused_cloud))
        self._track_id_to_fused_cloud = track_id_to_fused_cloud
      return self._track_id_to_fused_cloud

  def get_world_cloud(self):
    with self._shared:
      if self._world_cloud is None:
        print('get_world_cloud loading')
        from oarphpy.spark import RowAdapter
        FROM_ROW = RowAdapter.from_row
        world_cloud = None
        for pc in self.fused_datum_sample.lidar_clouds:
          if 'lidar|world_fused' in pc.sensor_name:
            pc = FROM_ROW(pc)
            print('loading cloud', pc.sensor_name)
            world_cloud = pc.get_cloud()
            print('loaded')
            break
        assert world_cloud is not None, fused_datum_sample.get_topics()
        print('cfcloud', world_cloud.shape)
        self._world_cloud = world_cloud
      return self._world_cloud

  def __call__(self, trow):
    T_ego2lidar = self.T_ego2lidar
    track_id_to_fused_cloud = self.get_track_id_to_fused_cloud()
    world_cloud = self.get_world_cloud()

    from oarphpy.spark import RowAdapter
    FROM_ROW = RowAdapter.from_row

    def union_all(it):
      import itertools
      return list(itertools.chain.from_iterable(it))
      
    cuboids1 = union_all(FROM_ROW(sd.cuboids) for sd in trow.cuboids_sds_t1)
    cuboids2 = union_all(FROM_ROW(sd.cuboids) for sd in trow.cuboids_sds_t2)
    ci1_sds = [FROM_ROW(sd) for sd in trow.ci_sds_t1]
    ci2_sds = [FROM_ROW(sd) for sd in trow.ci_sds_t2]
    cname_to_cisd1 = dict((sd.camera_image.sensor_name, sd) for sd in ci1_sds)
    cname_to_cisd2 = dict((sd.camera_image.sensor_name, sd) for sd in ci2_sds)
    all_cams = sorted(set(cname_to_cisd1.keys()) & set(cname_to_cisd2.keys()))
    
    rows_out = []
    for sensor_name in all_cams:
      import time
      start = time.time()
      
      ci_sd1 = cname_to_cisd1[sensor_name]
      ci_sd2 = cname_to_cisd2[sensor_name]
      ci1 = ci_sd1.camera_image
      ci2 = ci_sd2.camera_image
      print('starting cam', sensor_name, str(ci_sd1.uri))
      
      cam_height_pixels = ci1.height
      cam_width_pixels = ci1.width
      assert (ci1.width, ci1.height) == (ci2.width, ci2.height)

      # Pose all objects for t1 and t2
      moving_1 = np.zeros((0, 3))
      for cuboid in cuboids1:
        cloud_obj = track_id_to_fused_cloud[cuboid.track_id]
        cloud_ego = cuboid.obj_from_ego['ego', 'obj'].apply(cloud_obj).T
        cloud_world = cuboid.ego_pose.apply(cloud_ego).T
        moving_1 = np.vstack([moving_1, cloud_world])
      print('moving_1', moving_1.shape)
      
      moving_2 = np.zeros((0, 3))
      for cuboid in cuboids2:
        cloud_obj = track_id_to_fused_cloud[cuboid.track_id]
        cloud_ego = cuboid.obj_from_ego['ego', 'obj'].apply(cloud_obj).T
        cloud_world = cuboid.ego_pose.apply(cloud_ego).T
        moving_2 = np.vstack([moving_2, cloud_world])
      print('moving_2', moving_2.shape)
      
  
      movement = ci1.ego_pose.translation - ci2.ego_pose.translation
      print('movement', movement)
      if np.linalg.norm(movement) < 0.01:
          print('less than 1cm movement...')
          continue
  
      # T_ego2cam = ci1.ego_to_sensor.get_transformation_matrix(homogeneous=True)
      # T_lidar2cam = T_ego2cam @ np.linalg.inv(T_ego2lidar)
  
      P = np.eye(4)
      P[:3, :3] = ci1.K[:3, :3]
  
      pose1 = ci1.ego_pose.get_transformation_matrix(homogeneous=True)
      pose2 = ci2.ego_pose.get_transformation_matrix(homogeneous=True)
      uvdij1_visible_uvdij2_visible = self.render_func(
                  world_cloud=world_cloud,
                  T_ego2lidar=np.eye(4), # T_ego2lidar nope this is np.eye(4) for kitti and nusc
          
                  # KITTI-360 and nusc too wat i guess ego is lidar?
                  T_lidar2cam=ci1.ego_to_sensor.get_transformation_matrix(homogeneous=True),

                  P=P,
                  cam_height_pixels=cam_height_pixels,
                  cam_width_pixels=cam_width_pixels,

                  ego_pose1=pose1,
                  ego_pose2=pose2,
                  moving_1=moving_1,
                  moving_2=moving_2,


                  img1_factory=lambda: ci1.image,
                  img2_factory=lambda: ci2.image,
                  debug_title=trow.oflow_task_id)
      
      print('render_func in', time.time() - start)

      row_out = {
        'ci1_uri': ci_sd1.uri,
        'ci2_uri': ci_sd1.uri,
        'uvdij1_visible_uvdij2_visible': uvdij1_visible_uvdij2_visible,
      }

      rows_out.append(row_out)
    return rows_out
  
  FLOCK_PATH = '/tmp/psegs_RenderOFlowTasksWorker.lock'
  def single_machine_map_rows(self, trows):
    trows = list(trows)
    print('single_machine_map_rows working on', len(trows))
    if not trows:
      return []
    assert os.path.exists(self.FLOCK_PATH)
    import fasteners
    lock = fasteners.InterProcessLock(self.FLOCK_PATH)
    with lock:
      print(os.getpid(), 'starting with flock', self.FLOCK_PATH)
      
      # Hack: each worker thread needs temp space proportional to
      # world cloud size, so choose number of threads to not spill
      # into swap too badly.
      world_cloud = self.get_world_cloud()
      world_cloud_bytes = world_cloud.nbytes
      print('world_cloud_bytes', world_cloud_bytes)
      import psutil
      total_mem_bytes = psutil.virtual_memory().total # is exclusive of swap
      print('total_mem', total_mem_bytes)
      num_thread = max(1, int(total_mem_bytes / (10. * world_cloud_bytes)))
      print('num_thread', num_thread)
      
      # import multiprocessing
      # num_thread = 4 # semantic kitti
      # num_thread = 1 # kitti-360
      # num_thread = multiprocessing.cpu_count() # nusc keyframes only

      import concurrent.futures
      with concurrent.futures.ThreadPoolExecutor(max_workers=num_thread) as executor:
        results = list(executor.map(self, trows))

      # p = ThreadPool(num_thread)
      # print('num_thread', num_thread)
      # ress = p.map(self.__call__, trows)

      # import itertools
      # results = list(itertools.chain.from_iterable(ress))

      print('single_machine_map_rows done with', len(results))
      return results

class OpticalFlowRenderBase(object):

  FUSED_LIDAR_SD_TABLE = None

  RENDERER_ALGO_NAME = 'naive_shortest_ray'

  MAX_TASKS_SEED = 1337
  MAX_TASKS_PER_SEGMENT = -1
  TASK_OFFSETS = (1,)# 5)

  render_func = compute_optical_flows # TODO for python notebook drafting .........................

  @classmethod
  def TASK_DF_FACTORY(cls):
    return cls.FUSED_LIDAR_SD_TABLE.TASK_DF_FACTORY

  @classmethod
  def SRC_SD_T(cls):
    return cls.FUSED_LIDAR_SD_TABLE.SRC_SD_T()

  @classmethod
  def _get_T_ego2lidar(cls, task_df):
    from pyspark.sql import functions as F
    LIDAR_TOPIC = 'lidar'

    # Hacky! we just pick the first point cloud ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    pc_df = task_df.select(F.explode(task_df.pc_sds).alias('pc_sd'))
    row = pc_df.first()['pc_sd']
    sd = cls.SRC_SD_T().from_row(row)
    pc = sd.point_cloud
    T_ego2lidar = pc.ego_to_sensor.get_transformation_matrix(homogeneous=True)
    return T_ego2lidar

  @classmethod
  def _get_oflow_task_df(cls, spark, task_df):
    # Optionally limit by number of tasks.
    # We do this by filtering on task_id because it's much cheaper than e.g.
    # trying to sort the table below by rand() and then doing a LIMIT.
    task_id_filter_clause = ''
    if cls.MAX_TASKS_PER_SEGMENT > 0:
      print('restrict to', cls.MAX_TASKS_PER_SEGMENT)
      task_ids = [r.task_id for r in task_df.select('task_id').collect()]
      from random import Random
      r = Random(cls.MAX_TASKS_SEED)
      r.shuffle(task_ids)
      task_ids = task_ids[:cls.MAX_TASKS_PER_SEGMENT]
      tid_str = ", ".join(str(tid) for tid in task_ids)
      task_id_filter_clause = "AND cuci_1.task_id in ( %s )" % tid_str

    # Compute tasks pairs for flow
    task_id_join_clauses = [
      "( cuci_1.task_id = (cuci_2.task_id + %s) )" % offset
      for offset in cls.TASK_OFFSETS
    ]
    task_id_join_clause = " OR ".join(task_id_join_clauses)

    # Build the flow pair task table
    spark.catalog.dropTempView('oflow_culici_tasks_df')
    task_df.registerTempTable('oflow_culici_tasks_df')
    oflow_task_df = spark.sql(
      """
        SELECT
          CONCAT(cuci_1.task_id, '->', cuci_2.task_id)
            AS oflow_task_id,
          
          cuci_1.task_id AS task_id_1,
          cuci_1.cuboids_sds AS cuboids_sds_t1,
          cuci_1.ci_sds AS ci_sds_t1,

          cuci_2.task_id AS task_id_2,
          cuci_2.cuboids_sds AS cuboids_sds_t2,
          cuci_2.ci_sds AS ci_sds_t2
        
        FROM
          oflow_culici_tasks_df AS cuci_1, oflow_culici_tasks_df AS cuci_2
        
        WHERE
          SIZE(cuci_1.ci_sds) > 0 AND
          SIZE(cuci_2.ci_sds) > 0 AND
          ( {task_id_join_clause} ) {task_id_filter_clause}
      """.format(
            task_id_join_clause=task_id_join_clause,
            task_id_filter_clause=task_id_filter_clause))

    return oflow_task_df

  


  # @classmethod
  # def _render_oflow_tasks(cls, T_ego2lidar, fused_datum_sample, itask_rows):
  #   from oarphpy.spark import RowAdapter
  #   FROM_ROW = RowAdapter.from_row
    
  #   track_id_to_fused_cloud = {}
  #   for pc in fused_datum_sample.lidar_clouds:
  #     if 'lidar|objects_fused' in pc.sensor_name:
  #       cucloud = FROM_ROW(pc)
  #       track_id = cucloud.extra['track_id']
  #       track_id_to_fused_cloud[track_id] = cucloud.get_cloud()
  #       print(track_id, track_id_to_fused_cloud[track_id].shape)
  #   print('track_id_to_fused_cloud', len(track_id_to_fused_cloud))

  #   world_cloud = None
  #   for pc in fused_datum_sample.lidar_clouds:
  #     if 'lidar|world_fused' in pc.sensor_name:
  #       pc = FROM_ROW(pc)
  #       print('loading cloud', pc.sensor_name)
  #       world_cloud = pc.get_cloud()
  #       print('loaded')
  #       break
  #   assert world_cloud is not None, fused_datum_sample.get_topics()
  #   print('cfcloud', world_cloud.shape)
    
  #   def union_all(it):
  #     import itertools
  #     return list(itertools.chain.from_iterable(it))
  #   for trow in itask_rows:
  #     cuboids1 = union_all(FROM_ROW(sd.cuboids) for sd in trow.cuboids_sds_t1)
  #     cuboids2 = union_all(FROM_ROW(sd.cuboids) for sd in trow.cuboids_sds_t2)
  #     ci1s = [FROM_ROW(sd.camera_image) for sd in trow.ci_sds_t1]
  #     ci2s = [FROM_ROW(sd.camera_image) for sd in trow.ci_sds_t2]
  #     cname_to_ci1 = dict((c.sensor_name, c) for c in ci1s)
  #     cname_to_ci2 = dict((c.sensor_name, c) for c in ci2s)
  #     all_cams = sorted(set(cname_to_ci1.keys()) & set(cname_to_ci2.keys()))
  #     for sensor_name in all_cams:
  #       print(sensor_name)
  #       import time
  #       start = time.time()
        
  #       ci1 = cname_to_ci1[sensor_name]
  #       ci2 = cname_to_ci2[sensor_name]
        
  #       cam_height_pixels = ci1.height
  #       cam_width_pixels = ci1.width
  #       assert (ci1.width, ci1.height) == (ci2.width, ci2.height)

  #       # Pose all objects for t1 and t2
  #       moving_1 = np.zeros((0, 3))
  #       for cuboid in cuboids1:
  #         cloud_obj = track_id_to_fused_cloud[cuboid.track_id]
  #         cloud_ego = cuboid.obj_from_ego['ego', 'obj'].apply(cloud_obj).T
  #         cloud_world = cuboid.ego_pose.apply(cloud_ego).T
  #         moving_1 = np.vstack([moving_1, cloud_world])
  #       print('moving_1', moving_1.shape)
        
  #       moving_2 = np.zeros((0, 3))
  #       for cuboid in cuboids2:
  #         cloud_obj = track_id_to_fused_cloud[cuboid.track_id]
  #         cloud_ego = cuboid.obj_from_ego['ego', 'obj'].apply(cloud_obj).T
  #         cloud_world = cuboid.ego_pose.apply(cloud_ego).T
  #         moving_2 = np.vstack([moving_2, cloud_world])
  #       print('moving_2', moving_2.shape)
        
    
  #       movement = ci1.ego_pose.translation - ci2.ego_pose.translation
  #       print('movement', movement)
  #       if np.linalg.norm(movement) < 0.01:
  #           print('less than 1cm movement...')
  #           continue
    
  #       # T_ego2cam = ci1.ego_to_sensor.get_transformation_matrix(homogeneous=True)
  #       # T_lidar2cam = T_ego2cam @ np.linalg.inv(T_ego2lidar)
    
  #       P = np.eye(4)
  #       P[:3, :3] = ci1.K[:3, :3]
    
  #       pose1 = ci1.ego_pose.get_transformation_matrix(homogeneous=True)
  #       pose2 = ci2.ego_pose.get_transformation_matrix(homogeneous=True)
  #       result = cls.render_func(
  #                   world_cloud=world_cloud,
  #                   T_ego2lidar=np.eye(4), # T_ego2lidar nope this is np.eye(4) for kitti and nusc
            
  #                   # KITTI-360 and nusc too wat i guess ego is lidar?
  #                   T_lidar2cam=ci1.ego_to_sensor.get_transformation_matrix(homogeneous=True),

  #                   P=P,
  #                   cam_height_pixels=cam_height_pixels,
  #                   cam_width_pixels=cam_width_pixels,

  #                   ego_pose1=pose1,
  #                   ego_pose2=pose2,
  #                   moving_1=moving_1,
  #                   moving_2=moving_2,


  #                   img1_factory=lambda: ci1.get_image(),
  #                   img2_factory=lambda: ci2.get_image(),
  #                   debug_title=trow.oflow_task_id)
        
  #       print('did in', time.time() - start)

  #       yield result
        
        
  #       # import pickle
  #       # #path = "/opt/psegs/temp_out_fused/pair_%s_%s_%s.pkl" % (cam, t1, t2)
  #       # path = "/outer_root/media/seagates-ext4/au_datas/temp_out_fused/pair_%s_%s_%s.pkl" % (cam, t1, t2)
  #       # pickle.dump(data, open(path, 'wb'))
  #       # print(path)

  #   print()

  @classmethod
  def build(cls, spark=None, only_segments=None):
    with Spark.sess(spark) as spark:
      segment_uris = only_segments or cls.SRC_SD_T().get_all_segment_uris()
      
      for suri in segment_uris:
        task_df = cls.TASK_DF_FACTORY().build_df_for_segment(spark, suri)
        print('num tasks', task_df.count())
        fused_datum_sample = cls.FUSED_LIDAR_SD_TABLE.get_sample(
                                    suri, spark=spark)

        T_ego2lidar = cls._get_T_ego2lidar(task_df)

        oflow_task_df = cls._get_oflow_task_df(spark, task_df)
        print('oflow_task_df', oflow_task_df.count())
        worker = RenderOFlowTasksWorker(
          T_ego2lidar, fused_datum_sample, cls.render_func)

        # Hacky way to coalesce into CPU-intensive partitions
        from oarphpy.spark import num_executors
        n_tasks = oflow_task_df.count()
        n_parts = int(max(1, n_tasks / (10 * num_executors(spark))))
        print('coalesc to ', n_parts)
        oflow_task_df = oflow_task_df.coalesce(n_parts)
        result_rdd = oflow_task_df.rdd.mapPartitions(lambda irows: worker.single_machine_map_rows(irows))#(worker, preservesPartitioning=True)
                # lambda irows: cls._render_oflow_tasks(
                #       T_ego2lidar,
                #       fused_datum_sample,
                #       irows))
        OUT_PATH = '/tmp/oflow_out/'
        oputil.mkdir(OUT_PATH)
        import pickle
        t = oputil.ThruputObserver(name='BuildOFlow', n_total=n_tasks)
        t.start_block()
        for i, results in enumerate(result_rdd.toLocalIterator(prefetchPartitions=False)):
          for j, row in enumerate(results):
            path = os.path.join(OUT_PATH, 'oflow_%s_%s.pkl' % (i, j))
            with open(path, 'wb') as f:
              pickle.dump(row, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('saved to', path)

          t.update_tallies(n=1, num_bytes=oputil.get_size_of_deep(results), new_block=True)
          t.maybe_log_progress(every_n=1)


