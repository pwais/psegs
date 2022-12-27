# Copyright 2022 Maintainers of PSegs
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

# from psegs.datasets.kitti import KITTISDTable
# from psegs.datasets.kitti_360 import KITTI360SDTable
from psegs.datasets.ios_lidar import IOSLidarSDTFactory
# from psegs.datasets.nuscenes import NuscStampedDatumTableLabelsAllFrames
from psegs.table.union_factory import UnionFactory


class CanonicalFactory(UnionFactory):
  SDT_FACTORIES = [
    IOSLidarSDTFactory,
  ]


if True:
  # For now we'll just wire in our own data in Psegs.
  # User libraries and/or notebooks should set this up
  # in their own code after importing psegs but before 
  # using any psegs code that might list canonical stuff.

  from psegs.datasets.ios_lidar import Fixtures as IOS_Fixtures
  class MyFixtures(IOS_Fixtures):
    DATASET = 'pwais'
    SPLIT = 'private'
    @classmethod
    def threeDScannerApp_data_root(cls):
      from pathlib import Path
      return Path('/outer_root/media/magdaraid0/lidarphone_lidar_scans/')
    
    @classmethod
    def get_all_seg_uris(cls):
      seg_uris = []
      seg_uris += cls.get_threeDScannerApp_segment_uris()
        # Room for other recording sources ...

      BROKEN_SEGMENTS = (
        'Untitled Scan', 'amiot-crow-bar', 'headlands-downhill-2',
        'headlands-long-descent', 'san-anselmo-rock-fort-broken',
        'amiot-catcher-short', 'amiot-hot-dog-broken',
      )
      seg_uris = [
        suri for suri in seg_uris
        if suri.segment_id not in BROKEN_SEGMENTS
      ]
      return seg_uris


  class MyIOSLidarSDTableFactory(IOSLidarSDTFactory):
    FIXTURES = MyFixtures

  CanonicalFactory.SDT_FACTORIES += [MyIOSLidarSDTableFactory]
  
  # from psegs.datasets.adhoc_pixels import AdhocVideosSDTFactory
  # CanonicalFactory.SDT_FACTORIES += [
  #   AdhocVideosSDTFactory.create_factory_for_video(p)
  #   for p in (
  #     '/outer_root/media/magdaraid0/iphone_vids_to_sfm/vids_to_sfm/san-bruno-ridge-sunset-lidar-comparison-IMG_5652.MOV',
  #     '/outer_root/media/magdaraid0/iphone_vids_to_sfm/vids_to_sfm/san-bruno-ridge-sunset-lidar-comparison-IMG_5654.MOV',
  #     '/outer_root/media/magdaraid0/iphone_vids_to_sfm/vids_to_sfm/san-bruno-ridge-sunset-lidar-comparisonIMG_5653.MOV',
  #     '/outer_root/media/magdaraid0/iphone_vids_to_sfm/vids_to_sfm/dubs-gym-bluetiful-subie-lidar-comparison.MOV',
  #   )]

  from pathlib import Path
  from psegs.datasets.adhoc_pixels import AdhocImagePathsSDTFactory
  CanonicalFactory.SDT_FACTORIES += [
    AdhocImagePathsSDTFactory.create_factory_for_images(
        images_dir=p,
        dataset='pwais',
        split='private',
        segment_id=seg_name)
    for seg_name, p in {
      'dubs-gym-bluetiful-subie-lidar-comparison.MOV':
        '/outer_root/media/magdaraid0/vids_to_hloc_cache/dubs-gym-bluetiful-subie-lidar-comparison.MOV/',
      'lidar_hero10_winter_stinsin_GX010018.MP4_cache':
        '/outer_root/media/magdaraid0/iphone_vids_to_sfm/vids_to_sfm/lidar_hero10_winter_stinsin_GX010018.MP4_cache/frames',
      'hero10_calib3':
        '/outer_root/media/magdaraid0/iphone_vids_to_sfm/hero10_1/calib3_frames_short/keeps',
      'hero10_calib5':
        '/outer_root/media/magdaraid0/iphone_vids_to_sfm/hero10_1/calib5_frames/keeps',
      'winter-stinsin-just-the-nappie':
        '/outer_root/media/magdaraid0/iphone_vids_to_sfm/winter-stinsin-just-the-nappie/',
      'winter-stinsin-just-the-nappie-oneside':
        '/outer_root/media/magdaraid0/iphone_vids_to_sfm/winter-stinsin-just-the-nappie-oneside/',
    }.items()
  ]

  CanonicalFactory.SDT_FACTORIES += [
    AdhocImagePathsSDTFactory.create_factory_for_images(
        image_paths=sorted(
          p for p in 
          Path('/outer_root/media/magdaraid0/lidarphone_lidar_scans/2021_10_12_13_53_29/').iterdir()
          if 'frame_' in p.name),
        dataset='pwais',
        split='private',
        segment_id='phoenix-dry-long-broken'),
    
    AdhocImagePathsSDTFactory.create_factory_for_images(
        image_paths=sorted(
          p for i, p in 
          enumerate(Path('/outer_root/home/pwais/ORB_SLAM3/mav0/cam0/data').iterdir())
          if 'jpg' in p.name and ((i % 30) == 0)),
        dataset='pwais',
        split='private',
        segment_id='winter-stinsin-jpegged-30th'),
    
    AdhocImagePathsSDTFactory.create_factory_for_images(
        image_paths=sorted(
          p for i, p in 
          enumerate(Path('/outer_root/home/pwais/ORB_SLAM3/mav0/cam0/data').iterdir())
          if 'jpg' in p.name and ((i % 25) == 0)),
        dataset='pwais',
        split='private',
        segment_id='winter-stinsin-jpegged-25th'),
    
    AdhocImagePathsSDTFactory.create_factory_for_images(
        image_paths=sorted(
          p for i, p in 
          enumerate(Path('/outer_root/media/magdaraid0/iphone_vids_to_sfm/hero10_1/tl-buttercup-gpm-tasty').iterdir())
          if 'jpg' in p.name ),
        dataset='pwais',
        split='private',
        segment_id='tl-buttercup-gpm-tasty'),
    
    AdhocImagePathsSDTFactory.create_factory_for_images(
        image_paths=sorted(
          p for i, p in 
          enumerate(Path('/outer_root/media/magdaraid0/iphone_vids_to_sfm/hero10_1/king-theo-heart-tasty').iterdir())
          if 'jpg' in p.name ),
        dataset='pwais',
        split='private',
        segment_id='king-theo-heart-tasty'),

    AdhocImagePathsSDTFactory.create_factory_for_images(
        image_paths=sorted(
          p for i, p in 
          enumerate(Path('/outer_root/media/magdaraid0/iphone_vids_to_sfm/hero10_1/gabi-library-tasty').iterdir())
          if 'jpg' in p.name ),
        dataset='pwais',
        split='private',
        segment_id='gabi-library-tasty'),
    
    AdhocImagePathsSDTFactory.create_factory_for_images(
        image_paths=sorted(
          p for i, p in 
          enumerate(Path('/outer_root/media/magdaraid0/iphone_vids_to_sfm/jane-monkey-adhoc1').iterdir())
          if 'jpg' in p.name and 'frame_' in p.name),
        dataset='pwais',
        split='private',
        segment_id='jane-monkey-adhoc1'),
    
    AdhocImagePathsSDTFactory.create_factory_for_images(
        image_paths=sorted(
          p for i, p in 
          enumerate(Path('/outer_root/media/magdaraid0/iphone_vids_to_sfm/jane-monkey-adhoc2').iterdir())
          if 'jpg' in p.name and 'frame_' in p.name),
        dataset='pwais',
        split='private',
        segment_id='jane-monkey-adhoc2'),
    
    AdhocImagePathsSDTFactory.create_factory_for_images(
        image_paths=sorted(
          p for i, p in 
          enumerate(Path('/outer_root/media/magdaraid0/iphone_vids_to_sfm/bruno-bumble').iterdir())
          if 'jpg' in p.name and 'output_' in p.name),
        dataset='pwais',
        split='private',
        segment_id='bruno-bumble'),
    
    AdhocImagePathsSDTFactory.create_factory_for_images(
        image_paths=sorted(
          p for i, p in 
          enumerate(Path('/outer_root/media/magdaraid0/iphone_vids_to_sfm/dipsea-banana-slug').iterdir())
          if 'jpg' in p.name and 'output_' in p.name),
        dataset='pwais',
        split='private',
        segment_id='dipsea-banana-slug'),
    
    AdhocImagePathsSDTFactory.create_factory_for_images(
        image_paths=sorted(
          p for i, p in 
          enumerate(Path('/outer_root/media/magdaraid0/iphone_vids_to_sfm/dipsea-iris-family-gpm-1').iterdir())
          if 'jpg' in p.name and 'output_' in p.name),
        dataset='pwais',
        split='private',
        segment_id='dipsea-iris-family-gpm-1'),
    
    AdhocImagePathsSDTFactory.create_factory_for_images(
        image_paths=sorted(
          p for i, p in 
          enumerate(Path('/outer_root/media/magdaraid0/iphone_vids_to_sfm/mimi-frogs-test-1').iterdir())
          if 'jpg' in p.name and 'output_' in p.name),
        dataset='pwais',
        split='private',
        segment_id='mimi-frogs-test-1'),
	
    AdhocImagePathsSDTFactory.create_factory_for_images(
        image_paths=sorted(
          p for i, p in 
          enumerate(Path('/outer_root/media/magdaraid0/iphone_vids_to_sfm/gabigarden-rose-spider-test').iterdir())
          if 'jpg' in p.name and 'output_' in p.name),
        dataset='pwais',
        split='private',
        segment_id='gabigarden-rose-spider-test'),

    AdhocImagePathsSDTFactory.create_factory_for_images(
        image_paths=sorted(
          p for i, p in 
          enumerate(Path('/outer_root/media/magdaraid0/iphone_vids_to_sfm/gabigarden-baby-strawberry-test-1').iterdir())
          if 'jpg' in p.name and 'output_' in p.name),
        dataset='pwais',
        split='private',
        segment_id='gabigarden-baby-strawberry-test-1'),

    AdhocImagePathsSDTFactory.create_factory_for_images(
        image_paths=sorted(
          p for i, p in 
          enumerate(Path('/outer_root/media/magdaraid0/iphone_vids_to_sfm/dipsea-sleepie-bumble-pink-test-1').iterdir())
          if 'jpg' in p.name and 'output_' in p.name),
        dataset='pwais',
        split='private',
        segment_id='dipsea-sleepie-bumble-pink-test-1'),

    AdhocImagePathsSDTFactory.create_factory_for_images(
        image_paths=sorted(
          p for i, p in 
          enumerate(Path('/outer_root/media/magdaraid0/iphone_vids_to_sfm/san-bruno-ridge-spring-moth-wings-closed-iphone').iterdir())
          if 'jpg' in p.name and 'output_' in p.name),
        dataset='pwais',
        split='private',
        segment_id='san-bruno-ridge-spring-moth-wings-closed-iphone'),

  ]

  CanonicalFactory.SDT_FACTORIES += [

    AdhocImagePathsSDTFactory.create_factory_for_images(
        image_paths=sorted(
          p for i, p in 
          enumerate(Path('/outer_root/media/magdaraid0/iphone_vids_to_sfm/mission_bay_streatfood_bee').iterdir())
          if 'jpg' in p.name and f'mission_bay_streatfood_bee{s}' in p.name),
        dataset='pwais',
        split='private',
        segment_id=f'mission_bay_streatfood_bee{s}')

      for s in range(1, 7)

  ]

  # from psegs.table.sd_table_factory import ParquetSDTFactory
  # CanonicalFactory.SDT_FACTORIES += [
  #   ParquetSDTFactory.factory_for_sd_subdirs(
  #     '/outer_root/media/magdaraid0/hloc_out/')
  # ]


  # F = CanonicalFactory.SDT_FACTORIES[-1]
  # print(F.get_all_segment_uris()[-1])
  # t = F.get_segment_sd_table(F.get_all_segment_uris()[-1])
  # datum_rdd = t.to_datum_rdd()
  # print(datum_rdd.take(1))
  # df = t.to_spark_df()
  # df.show()
  # breakpoint()
  # print()
