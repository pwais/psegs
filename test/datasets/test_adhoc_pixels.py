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

from psegs.datasets import adhoc_pixels as ap

from test import testutil

def test_AdhocImagePathsSDTFactory_create_factory_for_images():
  FIXTURE_PQ = (testutil.test_fixtures_dir() / 
    'test_AdhocImagePathsSDTFactory_create_factory_for_images.parquet')

  # Borrow the COLMAP test images
  IMAGES_DIR = testutil.test_fixtures_dir() / 'test_colmap' / 'images'


  F = ap.AdhocImagePathsSDTFactory.create_factory_for_images(
            images_dir=IMAGES_DIR)

  with testutil.LocalSpark.sess() as spark:
    sdt = F.create_sd_table()
    
    sd_df_actual = sdt.to_spark_df()
    sd_df_actual.show()

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
    sdt = F.create_sd_table()
    
    sd_df_actual = sdt.to_spark_df()
    sd_df_actual.show()

    testutil.check_stamped_datum_dfs_equal(
      spark,
      sd_df_actual,
      sd_df_expected_path=FIXTURE_PQ)
