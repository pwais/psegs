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



from psegs.datasets.colmap import COLMAP_SDTFactory

from test import testutil


def test_colmap_create_sd_table_for_reconstruction():
  FIXTURES_DIR = testutil.test_fixtures_dir() / 'test_colmap'

  # Dump numpy cached assets to a temp dir
  PSEGS_ASSET_DIR = testutil.test_tempdir(
      'test_colmap_create_sd_table_for_reconstruction')

  with testutil.LocalSpark.sess() as spark:
    sdt = COLMAP_SDTFactory.create_sd_table_for_reconstruction(
              FIXTURES_DIR / 'sparse' / '0',
              FIXTURES_DIR / 'images',
              PSEGS_ASSET_DIR,
              spark=spark)
    
    sd_df_actual = sdt.to_spark_df()
    
    testutil.check_stamped_datum_dfs_equal(
      spark,
      sd_df_actual,
      sd_df_expected_path=FIXTURES_DIR / 'test_colmap_sdt_expected.parquet')
    