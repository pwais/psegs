import sys
sys.path.append('/opt/psegs')

import numpy as np

from psegs.exp.fused_lidar_flow import CloudFuser
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

class SemanticKITTIFusedWorldCloudTable(CloudFuser):
    FUSED_LIDAR_SD_TABLE = SemanticKITTSampleDFFactory

    # SemanticKITTI has no cuboids, so we skip this step.
    HAS_OBJ_CLOUDS = False


class SemanticKITTIFusedFlowDFFactory(FusedFlowDFFactory):
  SAMPLE_DF_FACTORY = SemanticKITTSampleDFFactory
  FUSED_LIDAR_SD_TABLE = SemanticKITTIFusedWorldCloudTable



# class SemanticKITTIOFlowRenderer(OpticalFlowRenderBase):
#     FUSED_LIDAR_SD_TABLE = SemanticKITTIFusedWorldCloudTable

from psegs.datasets.kitti_360 import KITTI360SDTable
class KITTI360_OurFused(KITTI360SDTable):
    INCLUDE_FISHEYES = False
    INCLUDE_FUSED_CLOUDS = False  # Use our own fused clouds

class KITTI360_OurFused_SampleDFFactory(SampleDFFactory):
    
    SRC_SD_TABLE = KITTI360_OurFused

    @classmethod
    def build_df_for_segment(cls, spark, segment_uri):
        from psegs import util
        
        datum_df = cls.SRC_SD_TABLE.get_segment_datum_df(spark, segment_uri)
        datum_df.registerTempTable('datums')
        
        util.log.info('Building sample table for %s ...' % segment_uri)
        
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

class KITTI360_OurFused_WorldCloudTableBase(CloudFuser):
  FUSED_LIDAR_SD_TABLE = KITTI360_OurFused_SampleDFFactory

class KITTI360_OurFused_FusedFlowDFFactory(FusedFlowDFFactory):
  SAMPLE_DF_FACTORY = KITTI360_OurFused_SampleDFFactory
  FUSED_LIDAR_SD_TABLE = KITTI360_OurFused_WorldCloudTableBase
    

        




class KITTI360_KITTIFused(KITTI360SDTable):
    INCLUDE_FISHEYES = False
    INCLUDE_FUSED_CLOUDS = True  # Use KITTI's fused clouds
    DATASET_NAME = 'kitti-360-fused'

class KITTI360_KITTIFused_SampleDFFactory(SampleDFFactory):
    
    SRC_SD_TABLE = KITTI360_KITTIFused

    @classmethod
    def build_df_for_segment(cls, spark, segment_uri):
        from psegs import util
        
        util.log.info(
          'Building sample table for %s ...' % segment_uri)

        datum_df = cls.SRC_SD_TABLE.get_segment_datum_df(spark, segment_uri)
        # datum_df = datum_df.persist()
        
        spark.catalog.dropTempView('datums')
        datum_df.registerTempTable('datums')

        # KITTI-360 fused clouds have data for multiple frames in each
        # individual file.  We only want to read each file once, so let's
        # prune the 'datums' table to contain only *distinct* clouds from
        # the available datums (and all other non-cloud data).
        valid_frames_df = spark.sql("""
            SELECT
              FIRST(uri.extra.`kitti-360.frame_id`) AS frame_id,
              uri.extra.`kitti-360.fused_cloud_path` AS cloud_path
            FROM datums
            WHERE uri.topic LIKE '%lidar|fused_static%'
            GROUP BY uri.extra.`kitti-360.fused_cloud_path`
            """)
        valid_fids = set(
          r.frame_id for r in valid_frames_df.collect()
          if r.cloud_path is not None
        )
        util.log.info(
          "... found %s distinct world clouds ..." % len(valid_fids))
        valid_fids_str = ','.join("'%s'" % fid for fid in valid_fids)

        datum_df = spark.sql("""
            SELECT *
            FROM datums
            WHERE
              uri.topic NOT LIKE '%lidar%' OR
              (
                uri.topic LIKE '%lidar|fused_static%' AND
                uri.extra.`kitti-360.frame_id` IN ( {valid_fids_str} )
              )
        """.format(valid_fids_str=valid_fids_str))
        datum_df.registerTempTable('kitti360_kfused_datums')


        # Now collect datums, using only the distinct fused clouds we
        # collected above
        spark.catalog.dropTempView('kitti360_kfused_sample_df')
        spark.sql("""
            CACHE TABLE 
              kitti360_kfused_sample_df
              OPTIONS ( 'storageLevel' 'DISK_ONLY' ) AS
            SELECT 
              INT(uri.extra.`kitti-360.frame_id`) AS sample_id,
              COLLECT_LIST(STRUCT(__pyclass__, uri, point_cloud)) 
                  FILTER (WHERE uri.topic LIKE '%lidar%') AS pc_sds,
              COLLECT_LIST(STRUCT(__pyclass__, uri, cuboids)) 
                  FILTER (WHERE uri.topic LIKE '%cuboid%') AS cuboids_sds,
              COLLECT_LIST(STRUCT(__pyclass__, uri, camera_image)) 
                  FILTER (WHERE uri.topic LIKE '%camera%') AS ci_sds
            FROM kitti360_kfused_datums
            WHERE (
              uri.topic LIKE '%cuboid%' OR
              uri.topic LIKE '%lidar|fused_static%' OR
              uri.topic LIKE '%camera%'
            ) AND (
              camera_image is NULL OR (camera_image.extra.`kitti-360.has-valid-ego-pose` = 'True')
            ) AND (
              point_cloud is NULL OR (point_cloud.extra.`kitti-360.has-valid-ego-pose` = 'True')
            )
            GROUP BY sample_id
        """)
        
        sample_df = spark.sql('SELECT * FROM kitti360_kfused_sample_df')
        util.log.info('... done.')
        return sample_df

from psegs.exp.fused_lidar_flow import WorldCloudCleaner
class PassThruWorldCloudCleaner(WorldCloudCleaner):
  
  def get_cleaned_world_cloud(self, point_clouds, cuboids):
    # TODO need to prune isVisible for optical flow ............................................................
    cleaned_clouds = []
    n_pruned = 0
    for pc in point_clouds:
      # self._thruput().start_block()

      cloud = pc.get_cloud()

      # actually this doesn't make a huge difference
      # # Only keep visible points.  These are points visible to at least one
      # # camera.  
      # vis_idx = pc.get_col_idx('is_visible')
      # cloud = cloud[cloud[:, vis_idx] == 1]

      cloud = cloud[:, :3] # TODO: can we keep colors?
      cloud_ego = pc.ego_to_sensor.get_inverse().apply(cloud).T
    
      cloud_ego = self._filter_ego_vehicle(cloud_ego)

      # skip filtering cuboids

      T_world_to_ego = pc.ego_pose
      cloud_world = T_world_to_ego.apply(cloud_ego).T # why is this name backwards?? -- hmm works for nusc too

      cleaned_clouds.append(cloud_world)
      
      # self._thruput().stop_block(
      #         n=1, num_bytes=oputil.get_size_of_deep(cloud_world))
      # self._thruput().maybe_log_progress(every_n=1)
      # self.__log_pruned()
    if not cleaned_clouds:
      return np.zeros((0, 3)), 0
    else:
      return np.vstack(cleaned_clouds), n_pruned



class KITTI360_KITTIFused_FusedFlowDFFactory(FusedFlowDFFactory):
  SAMPLE_DF_FACTORY = KITTI360_KITTIFused_SampleDFFactory
  WORLD_CLEANER_CLS = PassThruWorldCloudCleaner
  FUSED_LIDAR_SD_TABLE = KITTI360_OurFused_WorldCloudTableBase
    # We'll use our own fused *dynamic objects* but use
    # KITTI-360's fused *static world clouds*





# class KITTI360OFlowRenderer(OpticalFlowRenderBase):
#     FUSED_LIDAR_SD_TABLE = KITTI360WorldCloudTableBase






from psegs.datasets.nuscenes import NuscStampedDatumTableBase
from psegs.datasets.nuscenes import NuscStampedDatumTableLabelsAllFrames

class NuscFlowSDTable(NuscStampedDatumTableBase):
  SENSORS_KEYFRAMES_ONLY = False
  LABELS_KEYFRAMES_ONLY = False
  INCLUDE_LIDARSEG = False

  # @classmethod
  # def _get_all_segment_uris(cls):
  #   segment_uris = super(cls, NuscFlowSDTable)._get_all_segment_uris()
    
  #   # Simplify 'train' for 'train-detect' and 'train-track'
  #   for suri in segment_uris:  
  #     if 'train' in suri.split:
  #       suri.split = 'train'
    
  #   return segment_uris

class NuscSampleDFFactory(SampleDFFactory):
  SRC_SD_TABLE = NuscFlowSDTable

  # For NuScenes, labels (and keyframes) are only available at 2Hz and are
  # otherwise interpolated. The lidar scans at 2x the frequence of the cameras.
  # For best label alignment, we:
  # * Create some sample groups just for fusion (loose camera-lidar
  #      constraints for decent lidar painting).
  # * Create sample groups for rendering with only exact label-sensor alignment.
  # You can control which sensors are grouped for rendering below.
  GROUP_LIDAR_FOR_RENDERING = False
  GROUP_CAMERAS_FOR_RENDERING = True
  INCLUDE_LOOSE_LIDAR_CAMERA_FOR_FUSION = True
  LOOSE_FUSION_TIME_WINDOW_SEC = 0.1

  SAMPLE_IDS_BETWEEN_SENSORS = 10000

  @classmethod
  def build_df_for_segment(cls, spark, segment_uri):
      from psegs import util
      
      util.log.info(
        'Building sample table for %s ...' % segment_uri)

      datum_df = cls.SRC_SD_TABLE.get_segment_datum_df(spark, segment_uri)
      spark.catalog.dropTempView('nusc_datums')
      datum_df.registerTempTable('nusc_datums')

      all_topics = datum_df.select('uri.topic').distinct().collect()
      all_topics = [r[0] for r in all_topics]
      lidar_topics = sorted(t for t in all_topics if 'lidar' in t)
      camera_topics = sorted(t for t in all_topics if 'camera' in t)
      
      from pyspark.sql import Row
      samples_by_time = spark.sql("""
        SELECT
          uri.extra.`nuscenes-sample-token` AS sample_token,
          MIN(uri.timestamp) AS first_time
          FROM nusc_datums
          GROUP BY sample_token
        """).collect()
      samples_by_time = sorted(samples_by_time, key=lambda r: r.first_time)
      samples = [
        Row(sample_n=n, sample_token=r.sample_token)
        for n, r in enumerate(samples_by_time)
      ]
      spark.catalog.dropTempView('nusc_samples')
      nusc_samples_df = spark.createDataFrame(samples)
      nusc_samples_df.registerTempTable('nusc_samples')
      util.log.info("... have %s Nusc samples ..." % nusc_samples_df.count())


      ## Fusion Samples
      fusion_dfs = []
      sample_id_base = -1
      if cls.INCLUDE_LOOSE_LIDAR_CAMERA_FOR_FUSION:
        lidar_times_df = spark.sql("""
          SELECT
            uri.topic AS lidar_topic,
            uri.timestamp AS lidar_time 
          FROM nusc_datums
          WHERE uri.topic like '%lidar%'
          """)
        spark.catalog.dropTempView('nusc_lidar_times_df')
        lidar_times_df.registerTempTable('nusc_lidar_times_df')
        util.log.info(
          "... adding %s lidar clouds for rendering only ..." % (
            lidar_times_df.count(),))

        fusion_df = spark.sql("""
              SELECT 
                -1 * lidar_time AS sample_id,
                COLLECT_LIST(STRUCT(__pyclass__, uri, point_cloud)) 
                    FILTER (WHERE uri.topic LIKE '%lidar%') AS pc_sds,
                COLLECT_LIST(STRUCT(__pyclass__, uri, cuboids)) 
                    FILTER (WHERE uri.topic LIKE '%cuboid%') AS cuboids_sds,
                COLLECT_LIST(STRUCT(__pyclass__, uri, camera_image)) 
                    FILTER (WHERE uri.topic LIKE '%camera%') AS ci_sds
              FROM nusc_datums, nusc_lidar_times_df
              WHERE
                (
                  uri.topic = nusc_lidar_times_df.lidar_topic AND
                  uri.timestamp = nusc_lidar_times_df.lidar_time
                ) OR
                (
                  uri.topic LIKE '%cuboid%' AND
                  uri.timestamp = nusc_lidar_times_df.lidar_time AND
                  uri.extra.`nuscenes-label-channel` = 
                            SUBSTRING(
                              nusc_lidar_times_df.lidar_topic,
                              LENGTH('lidar|') + 1,
                              100)
                ) OR
                (
                  uri.topic LIKE '%camera%' AND
                  nusc_lidar_times_df.lidar_time - {buf} <= uri.timestamp AND
                  uri.timestamp <= nusc_lidar_times_df.lidar_time + {buf}
                )
              GROUP BY sample_id
          """.format(
            buf=int(1e9 * cls.LOOSE_FUSION_TIME_WINDOW_SEC)))
        fusion_dfs.append(fusion_df)

      ## Render Samples
      render_dfs = []
      sample_id_base = 0
      topics_to_render = []
      if cls.GROUP_LIDAR_FOR_RENDERING:
        topics_to_render += lidar_topics
      if cls.GROUP_CAMERAS_FOR_RENDERING:
        topics_to_render += camera_topics
      for topic in topics_to_render:
        sample_id_base += cls.SAMPLE_IDS_BETWEEN_SENSORS
        if 'camera' in topic:
          channel = topic[len('camera|'):]
        elif 'lidar' in topic:
          channel = topic[len('lidar|'):]
        else:
          raise ValueError(topic)

        topics_clause = """
            (uri.topic LIKE '%cuboid%' OR uri.topic = '{topic}')
          """.format(topic=topic)

        util.log.info(
          "... adding rendering for %s with sample id base %s ..." % (
            topic, sample_id_base))
        render_df = spark.sql("""
              SELECT 
                {sample_id_base} + nusc_samples.sample_n AS sample_id,
                COLLECT_LIST(STRUCT(__pyclass__, uri, point_cloud)) 
                    FILTER (WHERE uri.topic LIKE '%lidar%') AS pc_sds,
                COLLECT_LIST(STRUCT(__pyclass__, uri, cuboids)) 
                    FILTER (
                      WHERE uri.topic LIKE '%cuboid%' AND
                            uri.extra.`nuscenes-label-channel` = '{channel}'
                      ) AS cuboids_sds,
                COLLECT_LIST(STRUCT(__pyclass__, uri, camera_image)) 
                    FILTER (WHERE uri.topic LIKE '%camera%') AS ci_sds
              FROM nusc_datums, nusc_samples
              WHERE
                uri.extra.`nuscenes-sample-token` = nusc_samples.sample_token AND
                {topics_clause} AND
                uri.extra.`nuscenes-is-keyframe` = 'True'
              GROUP BY sample_id
          """.format(
            sample_id_base=sample_id_base,
            channel=channel,
            topics_clause=topics_clause))
        render_dfs.append(render_df)
      assert render_dfs, "Nothing to render?"

      from oarphpy import spark as S
      all_dfs = render_dfs + fusion_dfs
      sample_df = S.union_dfs(*all_dfs)
      sample_df = sample_df.repartition('sample_id')
      sample_df = sample_df.persist()
      
      n_samples = sample_df.count()
      util.log.info(
        '... done building %s dataframes for %s total samples.' % (
          len(all_dfs), n_samples))
      return sample_df


      # if cls.GROUP_CAMERAS_FOR_RENDERING:


      # kf_filter = "AND uri.extra.`nuscenes-is-keyframe` = 'True'"
      # pc_kf_filter = kf_filter if cls.LIDAR_KEYFRAMES_ONLY else ''
      # ci_kf_filter = kf_filter if cls.CAMERAS_KEYFRAMES_ONLY else ''
      # cu_kf_filter = kf_filter if cls.CUBOIDS_KEYFRAMES_ONLY else ''

      # spark.catalog.dropTempView('nusc_sample_df')
      # spark.sql("""
      #     CACHE TABLE 
      #       nusc_sample_df
      #       OPTIONS ( 'storageLevel' 'DISK_ONLY' ) AS
      #     SELECT 
      #       INT(uri.extra.`nuscenes-sample-offset`) AS sample_id,
      #       COLLECT_LIST(STRUCT(__pyclass__, uri, point_cloud)) 
      #           FILTER (WHERE uri.topic LIKE '%lidar%' {pc_kf_filter}) AS pc_sds,
      #       COLLECT_LIST(STRUCT(__pyclass__, uri, cuboids)) 
      #           FILTER (WHERE uri.topic LIKE '%cuboid%' {cu_kf_filter}) AS cuboids_sds,
      #       COLLECT_LIST(STRUCT(__pyclass__, uri, camera_image)) 
      #           FILTER (WHERE uri.topic LIKE '%camera%' {ci_kf_filter}) AS ci_sds
      #     FROM nusc_datums
      #     WHERE (
      #       uri.topic LIKE '%cuboid%' OR
      #       uri.topic LIKE '%lidar%' OR
      #       uri.topic LIKE '%camera%'
      #     )
      #     GROUP BY sample_id
      #     ORDER BY sample_id
      # """.format(
      #   pc_kf_filter=pc_kf_filter,
      #   ci_kf_filter=ci_kf_filter,
      #   cu_kf_filter=cu_kf_filter))

      # import pdb; pdb.set_trace()

      # sample_df = spark.sql('SELECT * FROM nusc_sample_df')
      # util.log.info('... done.')
      # return sample_df

from psegs.exp.fused_lidar_flow import WorldCloudCleaner
class NuscWorldCloudCleaner(WorldCloudCleaner):
  
  @classmethod
  def _filter_ego_vehicle(cls, cloud_ego):
      # Note: NuScenes authors have already corrected clouds for ego motion:
      # https://github.com/nutonomy/nuscenes-devkit/issues/481#issuecomment-716250423
      # But have not filtered out ego self-returns
      cloud_ego = cloud_ego[np.where(  ~(
                      (cloud_ego[:, 0] <= 1.5) & (cloud_ego[:, 0] >= -1.5) &  # Nusc lidar +x is +right
                      (cloud_ego[:, 1] <= 3) & (cloud_ego[:, 0] >= -3) &  # Nusc lidar +y is +forward
                      (cloud_ego[:, 1] <= 2.5) & (cloud_ego[:, 0] >= -2.5)    # Nusc lidar +z is +up
      ))]
      return cloud_ego


class NuscWorldCloudTableBase(CloudFuser):
  pass

class NuscFusedFlowDFFactory(FusedFlowDFFactory):
  SAMPLE_DF_FACTORY = NuscSampleDFFactory
  WORLD_CLEANER_CLS = NuscWorldCloudCleaner
  FUSED_LIDAR_SD_TABLE = NuscWorldCloudTableBase



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

  Rs = (
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
  flow_record = FlowRecord(
                  segment_uri=ci1_uri.to_segment_uri(),
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
  jt_rdd = jt_rdd.cache()
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

  frec_rdd = task_df.rdd.map(task_row_to_flow_record)
  frec_rdd = frec_rdd.repartition(task_df.count())
  
  from psegs.exp.fused_lidar_flow import FLOW_RECORD_PROTO  
  schema = RowAdapter.to_schema(FLOW_RECORD_PROTO)
  frec_df = spark.createDataFrame(
    frec_rdd.map(RowAdapter.to_row), schema=schema)

  frec_df = frec_df.persist()
  frec_df = frec_df.withColumn('dataset', frec_df['segment_uri.dataset'])
  frec_df = frec_df.withColumn('split', frec_df['segment_uri.split'])
  frec_df = frec_df.withColumn('segment_id', frec_df['segment_uri.segment_id'])

  frec_df.write.save(
        path=dest_path,
        mode='append',
        format='parquet',
        partitionBy=['dataset', 'split', 'segment_id'],
        compression='lz4')
  util.log.info("Saved to %s" % dest_path)

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

  pickles_to_flow_records(
    '/opt/psegs/dataroot/oflow_pickles',
    '/opt/psegs/dataroot/psegs_flow_records/records.parquet',
    max_n=-1)


  # R = NuscKeyframesOFlowRenderer

  # R = SemanticKITTIOFlowRenderer

  # R = KITTI360OFlowRenderer

  # # R.MAX_TASKS_PER_SEGMENT = 2

  # seg_uris = R.FUSED_LIDAR_SD_TABLE.get_all_segment_uris()
  # R.build(spark=spark, only_segments=[seg_uris[0]])

  



