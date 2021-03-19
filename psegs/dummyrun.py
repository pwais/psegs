import sys
sys.path.append('/opt/psegs')

import numpy as np

from psegs.exp.fused_lidar_flow import FusedLidarCloudTableBase
from psegs.exp.fused_lidar_flow import SampleDFFactory
from psegs.exp.fused_lidar_flow import FusedFlowDFFactory

import IPython.display
import PIL.Image

from psegs.exp.semantic_kitti import SemanticKITTISDTable

class SemanticKITTSampleDFFactory(SampleDFFactory):
    
    SRC_SD_TABLE = SemanticKITTISDTable
    
    @classmethod
    def build_df_for_segment(cls, spark, segment_uri):
        seg_rdd = cls.SRC_SD_TABLE.get_segment_datum_rdd(spark, segment_uri)
        
        def to_task_row(scan_id_iter_sds):
            scan_id, iter_sds = scan_id_iter_sds
            camera_images = []
            point_clouds = []
            for sd in iter_sds:
                if sd.camera_image is not None:
                    camera_images.append(sd)
                elif sd.point_cloud is not None:
                    point_clouds.append(sd)
            
            from pyspark import Row
            r = Row(
                    sample_id=int(scan_id),
                    pc_sds=point_clouds,
                    cuboids_sds=[], # SemanticKITTI has no cuboids
                    ci_sds=camera_images) 
            from oarphpy.spark import RowAdapter
            return RowAdapter.to_row(r)
            
        grouped = seg_rdd.groupBy(lambda sd: sd.uri.extra['semantic_kitti.scan_id'])
        row_rdd = grouped.map(to_task_row)

        df = spark.createDataFrame(row_rdd, schema=cls.table_schema())
        df = df.persist()
        return df

class SemanticKITTIFusedWorldCloudTable(FusedLidarCloudTableBase):
    TASK_DF_FACTORY = SemanticKITTSampleDFFactory

    # SemanticKITTI has no cuboids, so we skip this step.
    HAS_OBJ_CLOUDS = False


class SemanticKITTIFusedFlowDFFactory(FusedFlowDFFactory):
  SAMPLE_DF_FACTORY = SemanticKITTSampleDFFactory
  FUSED_LIDAR_SD_TABLE = SemanticKITTIFusedWorldCloudTable



# class SemanticKITTIOFlowRenderer(OpticalFlowRenderBase):
#     FUSED_LIDAR_SD_TABLE = SemanticKITTIFusedWorldCloudTable

from psegs.datasets.kitti_360 import KITTI360SDTable
class KITTI360OurFusedClouds(KITTI360SDTable):
    INCLUDE_FISHEYES = False
    INCLUDE_FUSED_CLOUDS = False  # Use our own fused clouds

class KITTI360LCCDFFactory(SampleDFFactory):
    
    SRC_SD_TABLE = KITTI360OurFusedClouds

    @classmethod
    def build_df_for_segment(cls, spark, segment_uri):
        from psegs import util
        
        datum_df = cls.SRC_SD_TABLE.get_segment_datum_df(spark, segment_uri)
        datum_df.registerTempTable('datums')
        
        util.log.info('Building sample table for %s ...' % segment_uri.segment_id)
        
        spark.catalog.dropTempView('kitti360_sample_df')
        spark.sql("""
            CACHE TABLE kitti360_sample_df OPTIONS ( 'storageLevel' 'DISK_ONLY' ) AS
            SELECT 
              INT(uri.extra.`kitti-360.frame_id`) AS sample_id,
              COLLECT_LIST(STRUCT(__pyclass__, uri, point_cloud)) 
                  FILTER (WHERE uri.topic LIKE '%lidar%') AS pc_sds,
              COLLECT_LIST(STRUCT(__pyclass__, uri, cuboids)) 
                  FILTER (WHERE uri.topic LIKE '%cuboid%') AS cuboids_sds,
              COLLECT_LIST(STRUCT(__pyclass__, uri, camera_image)) 
                  FILTER (WHERE uri.topic LIKE '%camera%') AS ci_sds
            FROM datums
            WHERE (
              uri.topic LIKE '%cuboid%' OR
              uri.topic LIKE '%lidar%' OR
              uri.topic LIKE '%camera%'
            ) AND (
              camera_image is NULL OR (camera_image.extra.`kitti-360.has-valid-ego-pose` = 'True')
            ) AND (
              point_cloud is NULL OR (point_cloud.extra.`kitti-360.has-valid-ego-pose` = 'True')
            )
            GROUP BY sample_id
            HAVING SIZE(pc_sds) > 0 AND SIZE(ci_sds) > 0
        """)
        
        sample_df = spark.sql('SELECT * FROM kitti360_sample_df')
        n_parts = int(max(10, sample_df.count() // 10))
        sample_df = sample_df.repartition(n_parts, 'sample_id')
        util.log.info('... done.')
        return sample_df

class KITTI360WorldCloudTableBase(FusedLidarCloudTableBase):
  TASK_DF_FACTORY = KITTI360LCCDFFactory

class KITTI360FusedFlowDFFactory(FusedFlowDFFactory):
  SAMPLE_DF_FACTORY = KITTI360LCCDFFactory
  FUSED_LIDAR_SD_TABLE = KITTI360WorldCloudTableBase
    

        
# class KITTI360OFlowRenderer(OpticalFlowRenderBase):
#     FUSED_LIDAR_SD_TABLE = KITTI360WorldCloudTableBase



# from psegs.datasets.nuscenes import NuscStampedDatumTableBase
# from psegs.datasets.nuscenes import NuscStampedDatumTableLabelsAllFrames


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
        
# class NuscWorldCloudTableBase(FusedLidarCloudTableBase):
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
#     TASK_DF_FACTORY = NuscKFOnlyLCCDFFactory

# class NuscAllFramesFusedWorldCloudTable(NuscWorldCloudTableBase):
#     TASK_DF_FACTORY = NuscAllFramesLCCDFFactory
    

# class NuscKeyframesOFlowRenderer(OpticalFlowRenderBase):
#     FUSED_LIDAR_SD_TABLE = NuscKFOnlyFusedWorldCloudTable

# class NuscAllFramesOFlowRenderer(OpticalFlowRenderBase):
#     FUSED_LIDAR_SD_TABLE = NuscAllFramesFusedWorldCloudTable




if __name__ == '__main__':
  from psegs.spark import Spark
  spark = Spark.getOrCreate()

  # R = KITTI360FusedFlowDFFactory

  # seg_uris = R.SRC_SD_T().get_all_segment_uris()
  # R.build(spark=spark, only_segments=[seg_uris[3]])




  # R = NuscKeyframesOFlowRenderer

  # R = SemanticKITTIOFlowRenderer

  # R = KITTI360OFlowRenderer

  # # R.MAX_TASKS_PER_SEGMENT = 2

  # seg_uris = R.FUSED_LIDAR_SD_TABLE.get_all_segment_uris()
  # R.build(spark=spark, only_segments=[seg_uris[0]])





  # from oarphpy import util as oputil
  # import os
  # PSEGS_OFLOW_PKL_PATHS = [
  #     os.path.abspath(p)
  #     for p in oputil.all_files_recursive('test_run_output', pattern='refactor*.pkl')
  # ]
  # print('len PSEGS_OFLOW_PKL_PATHS', len(PSEGS_OFLOW_PKL_PATHS))
  # # print(PSEGS_OFLOW_PKL_PATHS)

  # # PSEGS_OFLOW_PKL_PATHS = PSEGS_OFLOW_PKL_PATHS[:10]
  # path_rdd = spark.sparkContext.parallelize(
  #               PSEGS_OFLOW_PKL_PATHS,
  #               numSlices=len(PSEGS_OFLOW_PKL_PATHS))

  # from oarphpy.spark import RowAdapter
  # from pyspark.sql import Row
  # def to_row(path):
  #   import pickle
  #   with open(path, 'rb') as f:
  #       row = pickle.load(f)
    
  #   asdf = row.pop('uvdij1_visible_uvdij2_visible')
  #   row['uvd_viz1_uvd_viz2'] = asdf
  #   # from psegs.datum import URI
  #   # row['segment_uri'] = URI.from_str(row['ci1_uri']).to_segment_uri()
  #   return RowAdapter.to_row(Row(**row))

  # row_rdd = path_rdd.map(to_row)
  # import pyspark
  # row_rdd = row_rdd.persist(pyspark.StorageLevel.DISK_ONLY)
  
  # from psegs.datum.stamped_datum import URI_PROTO
  # import numpy as np
  # schema = RowAdapter.to_schema(Row(
  #   ci1_uri=URI_PROTO,
  #   ci2_uri=URI_PROTO,
  #   uvd_viz1_uvd_viz2=np.zeros((1, 4 + 4)),
  #   v2v_flow=np.zeros((10, 20, 2))
  # ))
  # df = spark.createDataFrame(row_rdd, schema=schema)
  
  # df = df.withColumn('dataset', df['ci1_uri.dataset'])
  # df = df.withColumn('segment_id', df['ci1_uri.segment_id'])
  
  # df.write.save(
  #       path='test_run_output/psegs_oflow.parquet',
  #       format='parquet',
  #       partitionBy=['dataset', 'segment_id'],
  #       compression='lz4')
  