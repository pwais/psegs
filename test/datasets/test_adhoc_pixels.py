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

import os
import datetime

from psegs import datum
from psegs.datasets import adhoc_pixels as ap

from test import testutil

def test_AdhocImagePathsSDTFactory_create_factory_for_images():
  FIXTURE_PQ = (testutil.test_fixtures_dir() / 
    'test_AdhocImagePathsSDTFactory_create_factory_for_images.parquet')

  # Borrow the COLMAP test images
  IMAGES_DIR = testutil.test_fixtures_dir() / 'test_colmap' / 'images'

  F = ap.AdhocImagePathsSDTFactory.create_factory_for_images(
            images_dir=IMAGES_DIR,
            timestamp_use=None)
              # Force sequential timestamps for reproducibility

  with testutil.LocalSpark.sess() as spark:
    sdt = F.create_sd_table(spark=spark)
    
    # Let's do a quick URI check, in part for the reader to see what we expect:
    expected_uris = [
      'psegs://dataset=anon&split=anon&segment_id=images&timestamp=1&topic=camera_adhoc&extra.AdhocImagePathsSDTFactory.image_path=/opt/psegs/test/fixtures/test_colmap/images/frame_00000.jpg',
      'psegs://dataset=anon&split=anon&segment_id=images&timestamp=2&topic=camera_adhoc&extra.AdhocImagePathsSDTFactory.image_path=/opt/psegs/test/fixtures/test_colmap/images/frame_00003.jpg',
      'psegs://dataset=anon&split=anon&segment_id=images&timestamp=3&topic=camera_adhoc&extra.AdhocImagePathsSDTFactory.image_path=/opt/psegs/test/fixtures/test_colmap/images/frame_00006.jpg',
      'psegs://dataset=anon&split=anon&segment_id=images&timestamp=4&topic=camera_adhoc&extra.AdhocImagePathsSDTFactory.image_path=/opt/psegs/test/fixtures/test_colmap/images/frame_00009.jpg',
      'psegs://dataset=anon&split=anon&segment_id=images&timestamp=5&topic=camera_adhoc&extra.AdhocImagePathsSDTFactory.image_path=/opt/psegs/test/fixtures/test_colmap/images/frame_00012.jpg',
      'psegs://dataset=anon&split=anon&segment_id=images&timestamp=6&topic=camera_adhoc&extra.AdhocImagePathsSDTFactory.image_path=/opt/psegs/test/fixtures/test_colmap/images/frame_00015.jpg',
      'psegs://dataset=anon&split=anon&segment_id=images&timestamp=7&topic=camera_adhoc&extra.AdhocImagePathsSDTFactory.image_path=/opt/psegs/test/fixtures/test_colmap/images/frame_00030.jpg',
      'psegs://dataset=anon&split=anon&segment_id=images&timestamp=8&topic=camera_adhoc&extra.AdhocImagePathsSDTFactory.image_path=/opt/psegs/test/fixtures/test_colmap/images/frame_00033.jpg'
    ]
    actual_uris = sdt.as_uri_rdd().map(lambda x: str(x)).collect()
    actual_uris.sort()
    assert actual_uris == expected_uris

    sd_df_actual = sdt.to_spark_df()
    testutil.check_stamped_datum_dfs_equal(
      spark,
      sd_df_actual,
      sd_df_expected_path=FIXTURE_PQ)


def test_AdhocVideosSDTFactory_create_factory_for_video():
  FIXTURE_PQ = (testutil.test_fixtures_dir() / 
    'test_AdhocVideosSDTFactory_create_factory_for_video.parquet')

  # Create a test video borrowing the COLMAP test images
  IMAGES_DIR = testutil.test_fixtures_dir() / 'test_colmap' / 'images'
  VID_DIR = testutil.test_tempdir(
            'test_AdhocVideosSDTFactory_create_factory_for_video')
  VID_PATH = VID_DIR / 'my_video.mp4'

  import imageio
  FPS = 2
  w = imageio.get_writer(VID_PATH, fps=FPS)
  for p in sorted(IMAGES_DIR.iterdir()):
    im = imageio.imread(p)
    w.append_data(im)
  w.close()

  F = ap.AdhocVideosSDTFactory.create_factory_for_video(VID_PATH)
  with testutil.LocalSpark.sess() as spark:
    sdt = F.create_sd_table(spark=spark)
    
    sd_df_actual = sdt.to_spark_df()
    sd_df_actual.show()

    testutil.check_stamped_datum_dfs_equal(
      spark,
      sd_df_actual,
      sd_df_expected_path=FIXTURE_PQ)


def test_DiskCachedFramesVideoSegmentFactory_create_factory_for_video():

  ## Setup

  CLS_CACHE_TEST_DIR = testutil.test_tempdir(
        'test_DiskCachedFramesVideoSegmentFactory_cache')
  IMAGE_CACHE_DIR = testutil.test_tempdir(
        'test_DiskCachedFramesVideoSegmentFactory_images')

  TEST_IMG_CACHE_CLS = testutil.PSegsTestLocalDiskCache.cache_cls_for_testroot(
                    IMAGE_CACHE_DIR)

  FIXTURE_PQ = (testutil.test_fixtures_dir() / 
    'test_DiskCachedFramesVideoSegmentFactory_create_factory_for_video.parquet')

  # Create a test video borrowing the COLMAP test images
  IMAGES_DIR = testutil.test_fixtures_dir() / 'test_colmap' / 'images'
  VID_DIR = testutil.test_tempdir(
        'test_DiskCachedFramesVideoSegmentFactory_create_factory_for_video')
  VID_PATH = VID_DIR / 'my_video.mp4'

  import imageio
  FPS = 4
  w = imageio.get_writer(VID_PATH, fps=FPS)
  for p in sorted(IMAGES_DIR.iterdir()):
    im = imageio.imread(p)
    w.append_data(im)
  w.close()
  EXPECTED_NUM_FRAMES = len(sorted(IMAGES_DIR.iterdir()))

  # DiskCachedFramesVideoSegmentFactory will use the mtime as a base timestamp
  # for the video datums, so set that to a fixed value for our test fixture
  mtime = datetime.datetime(2023, 1, 1, 1, 0, 0)
  os.utime(
    str(VID_PATH),
    (os.stat(str(VID_PATH)).st_atime,
    mtime.timestamp()))


  ## Test!
  F = ap.DiskCachedFramesVideoSegmentFactory.create_factory_for_video(
            VID_PATH,
            cls_cache_dir=CLS_CACHE_TEST_DIR,
            img_cache_cls=TEST_IMG_CACHE_CLS)

  expected_base_uri = datum.URI(
                    dataset='anon',
                    split='anon',
                    segment_id='my_video.mp4_3e859aae95',
                    topic='video_camera|max_hw_-1|ext_png')
  assert F.BASE_URI == expected_base_uri

  assert F.VIDEO_METADATA.video_uri == VID_PATH
  assert F.VIDEO_METADATA.frames_per_second == float(FPS)
  assert F.VIDEO_METADATA.n_frames == EXPECTED_NUM_FRAMES
  assert F.VIDEO_METADATA.height == 240
  assert F.VIDEO_METADATA.width == 320
  assert F.VIDEO_METADATA.is_10bit_hdr == False

  # Check the cache was used
  expected_cache_pkl_path = (
    CLS_CACHE_TEST_DIR /
    'anon' / 'anon' / 'my_video.mp4_3e859aae95' /
    'video_camera|max_hw_-1|ext_png' /
    'DiskCachedFramesVideoSegmentFactory_cls.cpkl')
  assert expected_cache_pkl_path.exists()

  # Reload should use cache
  F = ap.DiskCachedFramesVideoSegmentFactory.create_factory_for_video(
            VID_PATH,
            cls_cache_dir=CLS_CACHE_TEST_DIR)
  

  # Test explode 
  EF = F.explode_frames()

  assert EF.EXPLODED_FRAME_PATHS is not None
  assert len(EF.EXPLODED_FRAME_PATHS) == EXPECTED_NUM_FRAMES

  # Re-loading from cache should have the frames
  F = ap.DiskCachedFramesVideoSegmentFactory.create_factory_for_video(
            VID_PATH,
            cls_cache_dir=CLS_CACHE_TEST_DIR)
  assert F.EXPLODED_FRAME_PATHS is not None
  assert len(F.EXPLODED_FRAME_PATHS) == EXPECTED_NUM_FRAMES

  # Test SDT
  with testutil.LocalSpark.sess() as spark:
    # Use a factory freshly loaded from cache
    F = ap.DiskCachedFramesVideoSegmentFactory.create_factory_for_video(
            VID_PATH,
            cls_cache_dir=CLS_CACHE_TEST_DIR)

    sdt = F.create_sd_table(spark=spark)
    
    sd = sdt.to_datum_rdd().first()
    ci = sd.camera_image
    image = ci.image
    h, w = image.shape[:2]
    assert h == 240
    assert w == 320

    sd_df_actual = sdt.to_spark_df()
    sd_df_actual.show()

    # Ensure we got URIs for all of the frames
    expected_frame_ids = sorted(str(i) for i in range(EXPECTED_NUM_FRAMES))
    actual_extra_rows = sd_df_actual.select('uri.extra').collect()
    actual_frame_ids = sorted(
      r.extra['DiskCachedFramesVideoSegmentFactory.frame_index']
      for r in actual_extra_rows)
    assert actual_frame_ids == expected_frame_ids

    # Ensure the camera_images have distinct paths too
    actual_ci_extra_rows = sd_df_actual.select('camera_image.extra').collect()
    actual_ci_extra_fpaths = set(
      r.extra['DiskCachedFramesVideoSegmentFactory.frame_path']
      for r in actual_ci_extra_rows)
    assert len(actual_ci_extra_fpaths) == EXPECTED_NUM_FRAMES

    sd_df_actual = sd_df_actual.repartition(1)
    testutil.check_stamped_datum_dfs_equal(
      spark,
      sd_df_actual,
      sd_df_expected_path=FIXTURE_PQ)


def test_DiskCachedFramesVideoSegmentFactory_resized_create_factory_for_video():
  from psegs.util.video import VideoExplodeParams

  ## Setup

  CLS_CACHE_TEST_DIR = testutil.test_tempdir(
        'test_DiskCachedFramesVideoSegmentFactory_cache_resized')
  IMAGE_CACHE_DIR = testutil.test_tempdir(
        'test_DiskCachedFramesVideoSegmentFactory_images_resized')

  TEST_IMG_CACHE_CLS = testutil.PSegsTestLocalDiskCache.cache_cls_for_testroot(
                    IMAGE_CACHE_DIR)

  # Create a test video borrowing the COLMAP test images
  IMAGES_DIR = testutil.test_fixtures_dir() / 'test_colmap' / 'images'
  VID_DIR = testutil.test_tempdir(
    'test_DiskCachedFramesVideoSegmentFactory_resized_create_factory_for_video')
  VID_PATH = VID_DIR / 'my_video.mp4'

  import imageio
  FPS = 4
  w = imageio.get_writer(VID_PATH, fps=FPS)
  for p in sorted(IMAGES_DIR.iterdir()):
    im = imageio.imread(p)
    w.append_data(im)
  w.close()
  EXPECTED_NUM_FRAMES = len(sorted(IMAGES_DIR.iterdir()))


  ## Test!
  F = ap.DiskCachedFramesVideoSegmentFactory.create_factory_for_video(
            VID_PATH,
            explode_params=VideoExplodeParams(
              max_hw=300,
              image_file_extension='jpg'),
            cls_cache_dir=CLS_CACHE_TEST_DIR,
            img_cache_cls=TEST_IMG_CACHE_CLS)

  expected_base_uri = datum.URI(
                    dataset='anon',
                    split='anon',
                    segment_id='my_video.mp4_86e7a426d9',
                    topic='video_camera|max_hw_300|ext_jpg')
  assert F.BASE_URI == expected_base_uri

  assert F.VIDEO_METADATA.video_uri == VID_PATH
  assert F.VIDEO_METADATA.height == 240
  assert F.VIDEO_METADATA.width == 320

  # Test explode
  EF = F.explode_frames()

  assert EF.EXPLODED_FRAME_PATHS is not None
  assert len(EF.EXPLODED_FRAME_PATHS) == EXPECTED_NUM_FRAMES

  with testutil.LocalSpark.sess() as spark:
    sdt = EF.create_sd_table(spark=spark)
    
    sd_df_actual = sdt.to_spark_df()
    sd_df_actual.show()

    hw_sdf = sd_df_actual.select(['camera_image.height', 'camera_image.width'])
    hw_pdf = hw_sdf.toPandas()
    assert all(hw_pdf['height'] == 226)
    assert all(hw_pdf['width'] == 300)

    sd = sdt.to_datum_rdd().first()
    ci = sd.camera_image
    image = ci.image
    h, w = image.shape[:2]
    assert h == 226
    assert w == 300
