import sys
sys.path.append('/opt/psegs')

import numpy as np

from psegs.exp.fused_lidar_flow import CloudFuser
from psegs.exp.fused_lidar_flow import SampleDFFactory
from psegs.exp.fused_lidar_flow import FusedFlowDFFactory

import IPython.display
import PIL.Image


from psegs.exp.fused_lidar_flow import SemanticKITTIFusedFlowDFFactory



# class SemanticKITTIOFlowRenderer(OpticalFlowRenderBase):
#     FUSED_LIDAR_SD_TABLE = SemanticKITTIFusedWorldCloudTable

# from psegs.exp.fused_lidar_flow import KITTI360_OurFused_FusedFlowDFFactory

from psegs.exp.fused_lidar_flow import KITTI360_OurFused_FusedFlowDFFactory
from psegs.exp.fused_lidar_flow import KITTI360_KITTIFused_SampleDFFactory

        
# from psegs.exp.fused_lidar_flow import WorldCloudCleaner



# class KITTI360OFlowRenderer(OpticalFlowRenderBase):
#     FUSED_LIDAR_SD_TABLE = KITTI360WorldCloudTableBase










# class NuscWorldCloudTableBase(CloudFuser):
#     SPLITS = ['train_detect', 'train_track']
    
#     @classmethod
#     def _filter_ego_vehicle(cls, cloud_ego):
#         # Note: NuScenes authors have already corrected clouds for ego motion:
#         # https://github.com/nutonomy/nuscenes-devkit/issues/481#issuecomment-716250423
#         # But have not filtered out ego self-returns
#         cloud_ego = cloud_ego[np.where(  ~(
#                         (cloud_ego[:, 0] <= 1.5) & (cloud_ego[:, 0] >= -1.5) &  # Nusc lidar +x is +right
#                         (cloud_ego[:, 1] <= 2.5) & (cloud_ego[:, 0] >= -2.5) &  # Nusc lidar +y is +forward
#                         (cloud_ego[:, 1] <= 1.5) & (cloud_ego[:, 0] >= -1.5)    # Nusc lidar +z is +up
#         ))]
#         return cloud_ego



# class NuscKFOnlyLCCDFFactory(TaskLidarCuboidCameraDFFactory):
    
#     SRC_SD_TABLE = NuscStampedDatumTableBase
    
#     @classmethod
#     def build_df_for_segment(cls, spark, segment_uri):
#         datum_df = cls.SRC_SD_TABLE.get_segment_datum_df(spark, segment_uri)
#         datum_df.registerTempTable('datums')
#         print('Building tasks table for %s ...' % segment_uri.segment_id)
        
#         # Nusc doesn't have numerical task_ids so we'll have to induce
#         # one via lidar timestamp.
#         # NB: for Nusc: can group by nuscenes-sample-token FOR KEYFRAMES-ONLY DATA
#         task_data_df = spark.sql("""
#             SELECT 
#               COLLECT_LIST(STRUCT(__pyclass__, uri, point_cloud)) 
#                   FILTER (WHERE uri.topic LIKE '%lidar%') AS pc_sds,
#               COLLECT_LIST(STRUCT(__pyclass__, uri, cuboids)) 
#                   FILTER (WHERE uri.topic LIKE '%cuboid%') AS cuboids_sds,
#               COLLECT_LIST(STRUCT(__pyclass__, uri, camera_image)) 
#                   FILTER (WHERE uri.topic LIKE '%camera%') AS ci_sds,
#               MIN(uri.timestamp) FILTER (WHERE uri.topic LIKE '%lidar%') AS lidar_time,
#               FIRST(uri.extra.`nuscenes-sample-token`) AS sample_token
#             FROM datums
#             WHERE 
#             uri.extra.`nuscenes-is-keyframe` = 'True' AND (
#               uri.extra['nuscenes-label-channel'] is NULL OR 
#               uri.extra['nuscenes-label-channel'] LIKE '%LIDAR%'
#             ) AND (
#               uri.topic LIKE '%cuboid%' OR
#               uri.topic LIKE '%lidar%' OR
#               uri.topic LIKE '%camera%'
#             )
#             GROUP BY uri.extra.`nuscenes-sample-token`
#             ORDER BY lidar_time
#         """)
#         sample_tokens_ordered = [r.sample_token for r in task_data_df.select('sample_token').collect()]
#         task_to_stoken = [
#             {'task_id': task_id, 'sample_token': sample_token}
#             for task_id, sample_token in enumerate(sample_tokens_ordered)
#         ]
#         task_id_rdd = spark.sparkContext.parallelize(task_to_stoken)
#         task_id_df = spark.createDataFrame(task_id_rdd)
#         tasks_df = task_data_df.join(task_id_df, on=['sample_token'], how='inner')
#         tasks_df = tasks_df.persist()
#         print('... done.')
#         return tasks_df


# class NuscAllFramesLCCDFFactory(TaskLidarCuboidCameraDFFactory):
    
#     SRC_SD_TABLE = NuscStampedDatumTableLabelsAllFrames
    
#     @classmethod
#     def build_df_for_segment(cls, spark, segment_uri):
#         datum_df = cls.SRC_SD_TABLE.get_segment_datum_df(spark, segment_uri)
#         datum_df.registerTempTable('datums')
#         print('Building tasks table for %s ...' % segment_uri.segment_id)
        
#         task_data_df = spark.sql("""
#             SELECT 
#               COLLECT_LIST(STRUCT(__pyclass__, uri, point_cloud)) 
#                   FILTER (WHERE uri.topic LIKE '%lidar%') AS pc_sds,
#               COLLECT_LIST(STRUCT(__pyclass__, uri, cuboids)) 
#                   FILTER (WHERE uri.topic LIKE '%cuboid%') AS cuboids_sds,
#               COLLECT_LIST(STRUCT(__pyclass__, uri, camera_image)) 
#                   FILTER (WHERE uri.topic LIKE '%camera%') AS ci_sds,
#               MIN(uri.timestamp) FILTER (WHERE uri.topic LIKE '%lidar%') AS lidar_time,
#               FIRST(uri.extra.`nuscenes-sample-token`) AS sample_token
#             FROM datums
#             WHERE 
#             (
#               uri.extra['nuscenes-label-channel'] is NULL OR 
#               uri.extra['nuscenes-label-channel'] LIKE '%LIDAR%'
#             ) AND (
#               uri.topic LIKE '%cuboid%' OR
#               uri.topic LIKE '%lidar%' OR
#               uri.topic LIKE '%camera%'
#             )
#             GROUP BY uri.extra.`nuscenes-sample-token`
#             ORDER BY lidar_time
#         """)
#         sample_tokens_ordered = [r.sample_token for r in task_data_df.select('sample_token').collect()]
#         task_to_stoken = [
#             {'task_id': task_id, 'sample_token': sample_token}
#             for task_id, sample_token in enumerate(sample_tokens_ordered)
#         ]
#         task_id_rdd = spark.sparkContext.parallelize(task_to_stoken)
#         task_id_df = spark.createDataFrame(task_id_rdd)
#         tasks_df = task_data_df.join(task_id_df, on=['sample_token'], how='inner')
#         tasks_df = tasks_df.persist()
#         print('... done.')
#         return tasks_df
        
# class NuscWorldCloudTableBase(CloudFuser):
#     SPLITS = ['train_detect', 'train_track']
    
#     @classmethod
#     def _filter_ego_vehicle(cls, cloud_ego):
#         # Note: NuScenes authors have already corrected clouds for ego motion:
#         # https://github.com/nutonomy/nuscenes-devkit/issues/481#issuecomment-716250423
#         # But have not filtered out ego self-returns
#         cloud_ego = cloud_ego[np.where(  ~(
#                         (cloud_ego[:, 0] <= 1.5) & (cloud_ego[:, 0] >= -1.5) &  # Nusc lidar +x is +right
#                         (cloud_ego[:, 1] <= 2.5) & (cloud_ego[:, 0] >= -2.5) &  # Nusc lidar +y is +forward
#                         (cloud_ego[:, 1] <= 1.5) & (cloud_ego[:, 0] >= -1.5)    # Nusc lidar +z is +up
#         ))]
#         return cloud_ego
    
# class NuscKFOnlyFusedWorldCloudTable(NuscWorldCloudTableBase):
#     FUSED_LIDAR_SD_TABLE = NuscKFOnlyLCCDFFactory

# class NuscAllFramesFusedWorldCloudTable(NuscWorldCloudTableBase):
#     FUSED_LIDAR_SD_TABLE = NuscAllFramesLCCDFFactory
    

# class NuscKeyframesOFlowRenderer(OpticalFlowRenderBase):
#     FUSED_LIDAR_SD_TABLE = NuscKFOnlyFusedWorldCloudTable

# class NuscAllFramesOFlowRenderer(OpticalFlowRenderBase):
#     FUSED_LIDAR_SD_TABLE = NuscAllFramesFusedWorldCloudTable

def build_sample_id_map(spark, outpath, only_segments=[]):
  from pathlib import Path
  from psegs import util
  from psegs import datum

  only_segments = [datum.URI.from_str(u) for u in only_segments]

  n_total = len(only_segments) if only_segments else None
  from oarphpy import util as oputil

  t = oputil.ThruputObserver(name='build_sample_idx', n_total=n_total)

  from psegs.exp.fused_lidar_flow import NuscFusedFlowDFFactory
  from psegs.exp.fused_lidar_flow import KITTI360_KITTIFused_FusedFlowDFFactory
  from psegs.exp.fused_lidar_flow import KITTI360_OurFused_FusedFlowDFFactory

  Rs = (
    # SemanticKITTIFusedFlowDFFactory -- none of this as of writing
    NuscFusedFlowDFFactory,
    KITTI360_KITTIFused_FusedFlowDFFactory,
    KITTI360_OurFused_FusedFlowDFFactory,
  )
  for R in Rs:
    seg_uris = R.SAMPLE_DF_FACTORY.SRC_SD_TABLE.get_all_segment_uris()
    if only_segments:
      seg_uris = [
        datum.URI.from_str(u)
        for u in (
          set(str(s) for s in only_segments) & set(str(s) for s in seg_uris)
        )
      ]
    for suri in seg_uris:
      t.start_block()

      sdest = (
        Path(outpath) / 
          ('dataset=' + suri.dataset) / 
          ('split=' + suri.split) / 
          ('segment_id=' + suri.segment_id))
      if sdest.exists():
        util.log.info("Have %s" % sdest)
        t.stop_block(n=1)
        continue

      util.log.info("Indexing %s" % suri)
      
      sample_df = R.SAMPLE_DF_FACTORY.build_df_for_segment(spark, suri)
      spark.catalog.dropTempView('sample_df')
      sample_df.registerTempTable('sample_df')

      uri_exps = (
        ('pc_sds', '0 AS ci_height, 0 AS ci_width'),
        ('cuboids_sds', '0 AS ci_height, 0 AS ci_width'),
        ('ci_sds',
          """
            sd.camera_image.height AS ci_height,
            sd.camera_image.width AS ci_width
          """),
      )
      index_df = None
      for expr in uri_exps:
        attrname, ci_expr = expr
        df = spark.sql("""
                SELECT
                  "{dataset}"     AS dataset,
                  "{split}"       AS split,
                  "{segment_id}"  AS segment_id,
                  BIGINT(sample_id) AS sample_id,
                  sd.uri AS uri,
                  {ci_expr}
                
                FROM (
                  SELECT sample_id, EXPLODE({attrname}) AS sd
                  FROM sample_df
                )
        """.format(
                  dataset=suri.dataset,
                  split=suri.split,
                  segment_id=suri.segment_id,
                  attrname=attrname,
                  ci_expr=ci_expr))
        if index_df is None:
          index_df = df
        else:
          index_df = index_df.union(df)

      index_df = index_df.persist()
      index_df = index_df.coalesce(5)
      index_df.write.save(
        mode='append',
        path=outpath,
        partitionBy=['dataset', 'split', 'segment_id'],
        format='parquet',
        compression='lz4')
      
      util.log.info("Done with %s" % suri)
      t.stop_block(n=1)
      t.maybe_log_progress(every_n=1)

def task_row_to_flow_record(task_row):
  pkl_path = task_row['pkl_path']
  import pickle
  with open(pkl_path, 'rb') as f:
    pkldata = pickle.load(f)
  ci1_uri = pkldata['ci1_uri']
  # ci2_uri = pkldata['ci2_uri'] # broken before 4/7
  uvdv1_uvdv2 = pkldata['uvdij1_visible_uvdij2_visible']

  # hacks we screwed up
  toks = pkl_path.split('->')
  assert len(toks) == 2, pkl_path
  ci1_sid_fname = int(toks[0].split('_')[-1])
  ci2_sid_fname = int(toks[1].split('_')[0])

  import itertools
  sampledata_rows = list(itertools.chain.from_iterable(
    rs for rs in task_row['collect_list(sample_datas)']))
  assert sampledata_rows
  
  from oarphpy.spark import RowAdapter
  sid_to_rows = dict()
  for r in sampledata_rows:
    r = RowAdapter.from_row(r)
    sid_to_rows.setdefault(r.sample_id, [])
    sid_to_rows[r.sample_id].append(r)
  assert ci1_sid_fname in sid_to_rows, (ci1_sid_fname, sid_to_rows.keys())
  assert ci2_sid_fname in sid_to_rows, (ci2_sid_fname, sid_to_rows.keys())
  ci1_sid = ci1_sid_fname
  ci2_sid = ci2_sid_fname
  
  ci1_rows = sid_to_rows[ci1_sid]
  ci2_rows = sid_to_rows[ci2_sid]


  ci1_recs = [r for r in ci1_rows if r.uri == ci1_uri]
  assert len(ci1_recs) == 1, ci1_recs
  ci1_rec = ci1_recs[0]

  ci2_recs = [r for r in ci2_rows if r.uri.topic == ci1_uri.topic]
    # b/c ci2_uri broken in pickles before 4/7
  assert len(ci2_recs) == 1, ci2_recs
  ci2_rec = ci2_recs[0]
  ci2_uri = ci2_rec.uri
    # b/c ci2_uri broken in pickles before 4/7

  ci1_h, ci1_w = ci1_rec.ci_height, ci1_rec.ci_width
  assert (ci1_h, ci1_w) != (0, 0), (ci1_h, ci1_w)
  ci2_h, ci2_w = ci2_rec.ci_height, ci2_rec.ci_width
  assert (ci2_h, ci2_w) != (0, 0), (ci2_h, ci2_w)
  
  from psegs.exp.fused_lidar_flow import RenderedCloud
  uvdvis1 = uvdv1_uvdv2[:, :4]
  uvdvis2 = uvdv1_uvdv2[:, 4:]

  clouds = [
    RenderedCloud(
      sample_id=ci1_sid,
      ego_pose_uri=ci1_uri,
      uvdvis=uvdvis1,
      ci_uris=[r.uri for r in ci1_rows if 'camera' in r.uri.topic],
      cuboids_uris=[r.uri for r in ci1_rows if 'cuboids' in r.uri.topic],
      pc_uris=[r.uri for r in ci1_rows if 'lidar' in r.uri.topic],
    ),
    RenderedCloud(
      sample_id=ci2_sid,
      ego_pose_uri=ci2_uri,
      uvdvis=uvdvis2,
      ci_uris=[r.uri for r in ci2_rows if 'camera' in r.uri.topic],
      cuboids_uris=[r.uri for r in ci2_rows if 'cuboids' in r.uri.topic],
      pc_uris=[r.uri for r in ci2_rows if 'lidar' in r.uri.topic],
    ),
  ]

  from psegs.exp.fused_lidar_flow import FlowRecord
  assert (ci1_h, ci1_w) == (ci2_h, ci2_w)
  uri = ci1_uri.to_segment_uri()
  sids_str = ','.join(str(c.sample_id) for c in clouds)
  uri = uri.replaced(extra={'psegs_flow_sids': sids_str})
  flow_record = FlowRecord(
                  uri=uri,
                  uri_key=str(uri),
                  clouds=clouds,
                  u_min=0.0, u_max=float(ci1_w),
                  v_min=0.0, v_max=float(ci1_h))
  
  return flow_record

def pickles_to_flow_records(
      pickles_path,
      dest_path,
      index_cache_path='/opt/psegs/dataroot/sample_idx/sidx.parquet',
      max_n=-1):

  from psegs import util

  from psegs.spark import Spark
  spark = Spark.getOrCreate()

  from oarphpy import util as oputil
  import os
  PSEGS_OFLOW_PKL_PATHS = [
      os.path.abspath(p)
      for p in oputil.all_files_recursive(
                  pickles_path,
                  pattern='*.pkl')
  ]
  util.log.info("Have %s pickles" % len(PSEGS_OFLOW_PKL_PATHS))

  import random
  r = random.Random(1337)
  r.shuffle(PSEGS_OFLOW_PKL_PATHS)
  if max_n > 0:
    PSEGS_OFLOW_PKL_PATHS = PSEGS_OFLOW_PKL_PATHS[:max_n]
  path_rdd = spark.sparkContext.parallelize(
                PSEGS_OFLOW_PKL_PATHS,
                numSlices=len(PSEGS_OFLOW_PKL_PATHS))

  ### Read pkl data in indexed format... 
  ### ... in the end we'll have to read the pickles twice.
  def to_join_key(uri, sample_id):
    return ''.join((
                uri.dataset, uri.split, uri.segment_id,
                uri.topic, str(sample_id)))
  def to_pkl_idx(path):
    import pickle
    with open(path, 'rb') as f:
      row = pickle.load(f)
    ci1_uri = row['ci1_uri']
    ci2_uri = row['ci2_uri']
    
    # hacks we screwed up
    toks = path.split('->')
    assert len(toks) == 2, path
    ci1_sid_fname = int(toks[0].split('_')[-1])
    ci2_sid_fname = int(toks[1].split('_')[0])

    return {
      'ci1_uri_seg': str(row['ci1_uri'].to_segment_uri()),
      'ci1_uri_key': to_join_key(ci1_uri, ci1_sid_fname),
      # 'ci2_uri_seg': str(row['ci2_uri'].to_segment_uri()),
      'ci2_uri_key': to_join_key(ci2_uri, ci2_sid_fname),
      'pkl_path': path,
    }

  pkl_idx_rdd = path_rdd.map(to_pkl_idx)
  pkl_idx_rdd = pkl_idx_rdd.cache()
  pkl_idx_df = spark.createDataFrame(pkl_idx_rdd, samplingRatio=1.0)
  pkl_idx_df = pkl_idx_df.persist()
  util.log.info("Have pickle index of size %s" % pkl_idx_df.count())


  ### Build Sample Index as necessary
  seg_uris = [
    r.ci1_uri_seg
    for r in pkl_idx_df.select('ci1_uri_seg').distinct().collect()
  ]
  util.log.info("Have %s segments to do %s" % (len(seg_uris), sorted(seg_uris)))

  build_sample_id_map(spark, index_cache_path, only_segments=seg_uris)
  sample_idx_df = spark.read.parquet(index_cache_path)
  
  from psegs import datum
  segs = set(datum.URI.from_str(s).segment_id for s in seg_uris)
  sample_idx_df = sample_idx_df.filter(sample_idx_df.segment_id.isin(segs))
  sample_idx_df = sample_idx_df.persist()

  util.log.info(
    "Read index for %s segments" % (
      sample_idx_df.select('segment_id').distinct().count()))
  # util.log.info(sample_idx_df.count())

  # create "map" of ci_uri (str) -> all sample data associated with that ci_uri
  from oarphpy.spark import RowAdapter
  jt_rdd = sample_idx_df.rdd.map(
                              lambda r: (str(r.dataset + r.split + r.segment_id + str(r.sample_id)), r)
                              ).groupByKey().flatMap(
                                  lambda kvs: [
                                    {
                                      'ci_uri_key': to_join_key(v.uri, v.sample_id),
                                      'sample_datas': kvs[1].data,
                                    }
                                    for v in kvs[1]
                                    if 'camera' in v.uri.topic
                                  ])
  # jt_rdd = jt_rdd.cache()
  jt_df = spark.createDataFrame(jt_rdd, samplingRatio=0.5)
  jt_df = jt_df.repartition('ci_uri_key').persist()
  util.log.info("Have %s sample data indexed by camera image" % jt_df.count())
  

  ### Join indices together and convert!

  joined = pkl_idx_df.join(
                jt_df,
                (pkl_idx_df.ci1_uri_key == jt_df.ci_uri_key) |
                  (pkl_idx_df.ci2_uri_key == jt_df.ci_uri_key) )
  task_df = joined.groupBy('pkl_path').agg({'sample_datas': 'collect_list'})
  task_df = task_df.persist()
  util.log.info("Have %s tasks to do" % task_df.count())

  # Do this in chunks because it keeps failing due to network otherwise
  task_paths = [r.pkl_path for r in task_df.select('pkl_path').collect()]

  from oarphpy import util as oputil
  t = oputil.ThruputObserver(name='save_chunks', n_total=len(task_paths))
  for task_chunk in oputil.ichunked(task_paths, 100):
    t.start_block()
    chunk_df = task_df.filter(task_df.pkl_path.isin(list(task_chunk)))

    frec_rdd = chunk_df.rdd.map(task_row_to_flow_record)
    frec_rdd = frec_rdd.map(RowAdapter.to_row)
    frec_rdd = frec_rdd.repartition(len(task_chunk))

    # import pyspark
    # frec_rdd = frec_rdd.persist(pyspark.StorageLevel.DISK_ONLY)
    
    from psegs.exp.fused_lidar_flow import FLOW_RECORD_PROTO  
    schema = RowAdapter.to_schema(FLOW_RECORD_PROTO)
    frec_df = spark.createDataFrame(frec_rdd, schema=schema)

    # frec_df = frec_df.persist()
    frec_df = frec_df.withColumn('dataset', frec_df['uri.dataset'])
    frec_df = frec_df.withColumn('split', frec_df['uri.split'])
    frec_df = frec_df.withColumn('segment_id', frec_df['uri.segment_id'])

    frec_df.write.save(
          path=dest_path,
          mode='append',
          format='parquet',
          partitionBy=['dataset', 'split', 'segment_id'],
          compression='lz4')
    util.log.info("Saved some to %s" % dest_path)
    t.stop_block(n=len(task_chunk))
    t.maybe_log_progress(every_n=1)
    # frec_df.unpersist()
    # frec_rdd.unpersist()


  # import pickle
  # pickle.dump(task_df.take(1)[0], open('/tmp/task_row.pkl', 'wb'))

  # import ipdb; ipdb.set_trace()
  # print()

  # def to_pkl_jrow(path):
  #   from oarphpy.spark import RowAdapter
  #   from pyspark.sql import Row
  #   import pickle
  #   pkldata_str = open(path, 'rb').read()
  #   pkldata = pickle.loads(pkldata_str)
    
  #   pkldata.pop('v2v_flow')

  #   jrow = Row(
  #           ci1_uri_str=str(pkldata['ci1_uri']),
  #           ci2_uri_str=str(pkldata['ci2_uri']),
  #           pkldata_str=pkldata_str)
  #   return jrow

  #   # # asdf = row.pop('uvdij1_visible_uvdij2_visible')
  #   # # row['uvd_viz1_uvd_viz2'] = asdf
  #   # # from psegs.datum import URI
  #   # # row['segment_uri'] = URI.from_str(row['ci1_uri']).to_segment_uri()
  #   # return RowAdapter.to_row(Row(**row))

  # pkl_rdd = path_rdd.map(to_pkl_jrow)
  # import pyspark
  # pkl_rdd = pkl_rdd.persist(pyspark.StorageLevel.DISK_ONLY)
  
  # # from psegs.datum.stamped_datum import URI_PROTO
  # # import numpy as np
  # # schema = RowAdapter.to_schema(Row(
  # #   ci1_uri=URI_PROTO,
  # #   ci2_uri=URI_PROTO,
  # #   uvd_viz1_uvd_viz2=np.zeros((1, 4 + 4)),
  # # ))
  # pkl_df = spark.createDataFrame(pkl_rdd, samplingRatio=0.5)
  
  # joined = pkl_df.join(
  #               jt_df,
  #               (pkl_df.ci1_uri_str == jt_df.ci_uri_str) |
  #                 (pkl_df.ci2_uri_str == jt_df.ci_uri_str) )
  # task_df = joined.groupBy('pkl_path').agg({'sample_datas': 'collect_list'})


  # import ipdb; ipdb.set_trace()
  # print()

  # # df = df.withColumn('dataset', df['ci1_uri.dataset'])
  # # df = df.withColumn('split', df['ci1_uri.split'])
  # # df = df.withColumn('segment_id', df['ci1_uri.segment_id'])
  
  # import ipdb; ipdb.set_trace()

  # df.write.save(
  #       path=dest_path,
  #       format='parquet',
  #       partitionBy=['dataset', 'split', 'segment_id'],
  #       compression='lz4')
  


"""

analysis:
  * cuboid hit rates
  * sample-to-sample could nearest neighbor "error" 

general (beyond-just-pairs) design:
sample_ids [s1, s2, ... ]
ci_uris [ [ ci1 ], [ ci2 ], ... ] <-- len-1 for oflow, could be len N for sflow
cu_uris [ [ cu11, cu12, .. ], [ cu21, cu22, ... ], ... ]
pc_uris [ [ pc1 ], [ pc2 ], ... ] 
ego_pose_uris [ ego1, ego2, ... ] <-- always len 1, for sflow these are the uvd origins ?

uvdvis [ uvdvis1, uvdvis2, ... ]

(these arrays could also include not just uvd but world frame xyz perhaps.  perhaps
even rgb-normal?  .. surfel...)



"""


# from psegs.exp.fused_lidar_flow import KITTI360_OurFused
# from psegs.exp.fused_lidar_flow import KITTI360_KITTIFused
# from psegs.exp.fused_lidar_flow import NuscFlowSDTable















### BEGIN HACK


import attr
import cv2
import imageio
import math
import os
import PIL.Image
import six

import numpy as np

from oarphpy import plotting as op_plt
from oarphpy.spark import CloudpickeledCallable
img_to_data_uri = lambda x: op_plt.img_to_data_uri(x, format='png')

@attr.s(slots=True, eq=False, weakref_slot=False)
class OpticalFlowPair(object):
    """A flyweight for a pair of images with an optical flow field.
    Supports lazy-loading of large data attributes."""
        
    dataset = attr.ib(type=str, default='')
    """To which dataset does this pair belong?"""
    
    id1 = attr.ib(type=str, default='')
    """Identifier or URI for the first image"""
    
    id2 = attr.ib(type=str, default='')
    """Identifier or URI for the second image"""
    
    img1 = attr.ib(default=None)
    """URI or numpy array or CloudPickleCallable for the first image (source image)"""

    img2 = attr.ib(default=None)
    """URI or numpy array or CloudpickeledCallable for the second image (target image)"""
    
    flow = attr.ib(default=None)
    """A numpy array or callable or CloudpickeledCallable representing optical flow from img1 -> img2"""
    
    ## Optional Attributes (For Select Datasets)
    
    diff_time_sec = attr.ib(type=float, default=0.0)
    """Difference in time (in seconds) between the views / poses depicted in `img1` and `img2`."""
    
    translation_meters = attr.ib(type=float, default=0.0)
    """Difference in ego translation (in meters) between the views / poses depicted in `img1` and `img2`."""

    # to add:
    # diff time seconds
    # semantic image for frame 1, frame 2 [could be painted by cuboids]
    # instance images for frame 1, frame 2 [could be painted by cuboids]
    #   -- for colored images, at first just pivot all oflow metrics by colors
    # get uvdviz1 uvdviz2 (scene flow)
    #   * for deepeform, their load_flow will work
    #   * for kitti, we have to read their disparity images
    # get uvd1 uvd2 (lidar for nearest neighbor stuff)
    # depth image for frame 1, frame 2 [could be interpolated by cuboids]
    #   -- at first bucket the depth coarsely and pivot al oflow by colors
    
    def get_img1(self):
        if isinstance(self.img1, CloudpickeledCallable):
            self.img1 = self.img1()
        if isinstance(self.img1, six.string_types):
            self.img1 = imageio.imread(self.img1)
        return self.img1
    
    def get_img2(self):
        if isinstance(self.img2, CloudpickeledCallable):
            self.img2 = self.img2()
        if isinstance(self.img2, six.string_types):
            self.img2 = imageio.imread(self.img2)
        return self.img2
    
    def get_flow(self):
        if not isinstance(self.flow, (np.ndarray, np.generic)):
            self.flow = self.flow()
        return self.flow
    
    def to_html(self):
        im1 = self.get_img1()
        im2 = self.get_img2()
        flow = self.get_flow()
        fviz = draw_flow(im1, flow)
        html = """
            <table>
            
            <tr><td style="text-align:left"><b>Dataset:</b> {dataset}</td></tr>
            
            <tr><td style="text-align:left"><b>Source Image:</b> {id1}</td></tr>
            <tr><td><img src="{im1}" /></td></tr>

            <tr><td style="text-align:left"><b>Target Image:</b> {id2}</td></tr>
            <tr><td><img src="{im2}" /></td></tr>

            <tr><td style="text-align:left"><b>Flow</b></td></tr>
            <tr><td><img src="{fviz}" /></td></tr>
            </table>
        """.format(
                dataset=self.dataset,
                id1=self.id1, id2=self.id2,
                im1=img_to_data_uri(im1), im2=img_to_data_uri(im2),
                fviz=img_to_data_uri(fviz))
        return html

def draw_flow(img, flow, step=8):
    """Based upon OpenCV sample: https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py"""
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = img.copy()
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis



#### END HACK




### BEGIN PROPOSE MODULE

# from cheap_optical_flow_eval_analysis.ofp import OpticalFlowPair

from oarphpy.spark import CloudpickeledCallable

from psegs.exp.fused_lidar_flow import FlowRecTable

PSEGS_SYNTHFLOW_DEMO_RECORD_URIS = (
  'psegs://dataset=kitti-360&split=train&segment_id=2013_05_28_drive_0000_sync&extra.psegs_flow_sids=4340,4339',
  'psegs://dataset=kitti-360&split=train&segment_id=2013_05_28_drive_0000_sync&extra.psegs_flow_sids=11219,11269',

  'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0501&extra.psegs_flow_sids=40009,40010',
  'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0501&extra.psegs_flow_sids=50013,50014',

  # 'psegs://dataset=kitti-360-fused&split=train&segment_id=2013_05_28_drive_0000_sync&extra.psegs_flow_sids=11103,11104',
  # 'psegs://dataset=kitti-360-fused&split=train&segment_id=2013_05_28_drive_0000_sync&extra.psegs_flow_sids=1181,1182',

  # 'psegs://dataset=nuscenes&split=train_detect&segment_id=scene-0002&extra.psegs_flow_sids=10016,10017',
  # 'psegs://dataset=nuscenes&split=train_detect&segment_id=scene-0582&extra.psegs_flow_sids=60035,60036',

  # 'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0393&extra.psegs_flow_sids=50017,50018',
  # 'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0501&extra.psegs_flow_sids=40019,40020',
)

def flow_rec_to_fp(flow_rec, sample):
  fr = flow_rec

  uri_str_to_datum = sample.get_uri_str_to_datum()

  # Find the camera_images associated with `flow_rec`
  ci1_url_str = str(flow_rec.clouds[0].ci_uris[0])
  ci1_sd = uri_str_to_datum[ci1_url_str]
  ci1 = ci1_sd.camera_image

  ci2_url_str = str(flow_rec.clouds[1].ci_uris[0])
  ci2_sd = uri_str_to_datum[ci2_url_str]
  ci2 = ci2_sd.camera_image

  import numpy as np
  world_T1 = ci1.ego_pose.translation
  world_T2 = ci2.ego_pose.translation
  translation_meters = np.linalg.norm(world_T2 - world_T1)

  id1 = ci1_url_str + '&extra.psegs_flow_sids=' + str(fr.clouds[0].sample_id)
  id2 = ci2_url_str + '&extra.psegs_flow_sids=' + str(fr.clouds[1].sample_id)

  fp = OpticalFlowPair(
          dataset=fr.uri.dataset + '/' + fr.uri.split,
          id1=id1,
          id2=id2,
          img1=CloudpickeledCallable(lambda: ci1.image),
          img2=CloudpickeledCallable(lambda: ci2.image),
          flow=CloudpickeledCallable(lambda: fr.to_optical_flow()),

          diff_time_sec=abs(ci2_sd.uri.timestamp - ci1_sd.uri.timestamp),
          translation_meters=translation_meters)
  return fp

def psegs_synthflow_create_fps(
        spark,
        flow_record_pq_table_path,
        record_uris,
        include_cuboids=False,
        include_point_clouds=False):

  T = FlowRecTable(spark, flow_record_pq_table_path)
  rec_sample_rdd = T.get_records_with_samples_rdd(
                          record_uris=record_uris,
                          include_cameras=True,
                          include_cuboids=include_cuboids,
                          include_point_clouds=include_point_clouds)

  fps = [
    flow_rec_to_fp(flow_rec, sample)
    for flow_rec, sample in rec_sample_rdd.collect()
  ]

  return fps

def psegs_synthflow_iter_fp_rdds(
        spark,
        flow_record_pq_table_path,
        fps_per_rdd=100,
        include_cuboids=False,
        include_point_clouds=False):
  
  T = FlowRecTable(spark, flow_record_pq_table_path)
  ruris = T.get_record_uris()

  # Ensure a sort so that pairs from similar segments will load in the same
  # RDD -- that makes joins smaller and faster
  ruris = sorted(ruris)

  from oarphpy import util as oputil
  for ruri_chunk in oputil.ichunked(ruris, fps_per_rdd):
    frec_sample_rdd = T.get_records_with_samples_rdd(
                          record_uris=rids,
                          include_cuboids=include_cuboids,
                          include_point_clouds=include_point_clouds)
    fp_rdd = frec_sample_rdd.map(flow_rec_to_fp)
    yield fp_rdd


### END PROPOSE MODULE



def convert_again(in_pq, out_pq):

  from psegs.spark import Spark
  spark = Spark.getOrCreate()

  df = spark.read.parquet(in_pq)

  def convert(fr):
    from pyspark import Row
    from oarphpy.spark import RowAdapter
    fr = RowAdapter.from_row(fr)

    uri = fr.uri

    key_uris = [c.ego_pose_uri for c in fr.clouds]
    uri = uri.replaced(sel_datums=key_uris)
    key = str(uri)

    fr.uri = uri
    fr.uri_key = key

    row = RowAdapter.to_row(fr)
    return row
    # row = row.asDict()
    # row['dataset'] = uri.dataset
    # row['split'] = uri.split
    # row['segment_id'] = uri.segment_id
    # return Row(
    #           dataset=uri.dataset,
    #           split=uri.split,
    #           segment_id=uri.segment_id,
    #           **row.asDict())
  
  

  from oarphpy.spark import RowAdapter
  from psegs.exp.fused_lidar_flow import FLOW_RECORD_PROTO  
  schema = RowAdapter.to_schema(FLOW_RECORD_PROTO)
  frec_df = spark.createDataFrame(df.rdd.map(convert), schema=schema)

  # frec_df = frec_df.persist()
  frec_df = frec_df.withColumn('dataset', frec_df['uri.dataset'])
  frec_df = frec_df.withColumn('split', frec_df['uri.split'])
  frec_df = frec_df.withColumn('segment_id', frec_df['uri.segment_id'])

  frec_df.write.save(
          path=out_pq,
          format='parquet',
          partitionBy=['dataset', 'split', 'segment_id'],
          compression='lz4')




if __name__ == '__main__':
  # from psegs.spark import Spark
  # spark = Spark.getOrCreate()

  # # R = KITTI360_OurFused_FusedFlowDFFactory
  # # R = KITTI360_KITTIFused_FusedFlowDFFactory

  # R = NuscFusedFlowDFFactory

  # seg_uris = R.SRC_SD_T().get_all_segment_uris()
  # # R.build(spark=spark, only_segments=['psegs://segment_id=scene-0594'])#seg_uris[0]])
  # R.build(spark=spark, only_segments=seg_uris[150:200])

  # import pickle
  # task_row = pickle.load(open('/tmp/task_row.pkl', 'rb'))
  # rec = task_row_to_flow_record(task_row)
  # import ipdb; ipdb.set_trace()




  # pickles_to_flow_records(
  #   '/opt/psegs/dataroot/oflow_pickles',
  #   '/outer_root/media/Costco8000/psegs_synthflow.parquet/',
  #   max_n=-1)
  # print('yay!')
  # assert False

  from psegs.spark import Spark
  spark = Spark.getOrCreate()


  # fps = psegs_synthflow_create_fps(
  #           spark,
  #           '/outer_root/media/rocket4q/psegs_flow_records_short',
  #           PSEGS_SYNTHFLOW_DEMO_RECORD_URIS)
  # import ipdb; ipdb.set_trace()






  convert_again(
     '/outer_root/media/rocket4q/psegs_synthflow.parquet',
     '/outer_root/media/Costco8000/psegs_flow_records_FULL_fixed'
  )
  import ipdb; ipdb.set_trace()



  T = FlowRecTable(spark, '/outer_root/media/rocket4q/psegs_flow_records_short_fixed')
  rids = T.get_record_uris()
  print('rids', len(rids))
  
  # import pprint
  # pprint.pprint([str(r) for r in rids])
  # assert False
  # rids = [r for r in rids if 'kitti' in r.dataset]
  # rids = rids[:100]

  
  rdd = T.get_records_with_samples_rdd(record_uris=rids)

  #print('second now')
  #big_rdd = T.get_records_with_samples_rdd(record_uris=rids[10:12])
  #print(big_rdd.count())
  #print('done')

  flow_rec, sample = rdd.take(1)[0]

  # print(flow_rec.to_html(camera_images=sample.camera_images))

  import ipdb; ipdb.set_trace()
  print()

  flow_df = spark.read.parquet('/outer_root/media/rocket4q/psegs_flow_records_short')

  rec = flow_df.take(1)[0]
  from oarphpy.spark import RowAdapter
  rec = RowAdapter.from_row(rec)

  rec.to_html()

  import ipdb; ipdb.set_trace()
  print()



  # R = NuscKeyframesOFlowRenderer

  # R = SemanticKITTIOFlowRenderer

  # R = KITTI360OFlowRenderer

  # # R.MAX_TASKS_PER_SEGMENT = 2

  # seg_uris = R.FUSED_LIDAR_SD_TABLE.get_all_segment_uris()
  # R.build(spark=spark, only_segments=[seg_uris[0]])

  



