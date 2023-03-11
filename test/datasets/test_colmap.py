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


import pytest

import numpy as np

from psegs.datasets import colmap as pscolmap

from test import testutil


def test_colmap_create_camera_image():
  pytest.importorskip("pycolmap")

  FIXTURES_DIR = testutil.test_fixtures_dir() / 'test_colmap'
  
  ci = pscolmap.colmap_recon_create_camera_image(
                  'frame_00012.jpg',
                  FIXTURES_DIR / 'sparse' / '0',
                  FIXTURES_DIR / 'images')
  assert ci.image.shape == (240, 320, 3)
  assert ci.extra['colmap.image_id'] == '3'
  assert ci.K[0][0] == 262.71597626202475


def test_colmap_create_depth_image():
  pytest.importorskip("pycolmap")

  FIXTURES_DIR = testutil.test_fixtures_dir() / 'test_colmap'

  dci = pscolmap.colmap_recon_create_camera_image(
                  'frame_00012.jpg',
                  FIXTURES_DIR / 'sparse' / '0',
                  FIXTURES_DIR / 'images',
                  create_depth_image=True)
  assert dci.image.shape == (240, 320, 3)
  assert dci.get_depth().min() == 0
  np.testing.assert_allclose(dci.get_depth().max(), 54.110733)
  assert dci.get_chan('colmap_err').min() == 0
  np.testing.assert_allclose(dci.get_chan('colmap_err').max(), 2.194288)
  assert dci.get_chan('num_views_visible').min() == 0
  assert dci.get_chan('num_views_visible').max() == 14
  assert dci.extra['colmap.image_id'] == '3'
  np.testing.assert_allclose(dci.K[0][0], 262.71597626202475)


def test_colmap_create_sd_table_for_reconstruction():
  pytest.importorskip("pycolmap")

  FIXTURES_DIR = testutil.test_fixtures_dir() / 'test_colmap'

  # Dump numpy cached assets to a temp dir
  PSEGS_ASSET_DIR = testutil.test_tempdir(
      'test_colmap_create_sd_table_for_reconstruction')

  with testutil.LocalSpark.sess() as spark:
    sdt = pscolmap.COLMAP_SDTFactory.create_sd_table_for_reconstruction(
              FIXTURES_DIR / 'sparse' / '0',
              FIXTURES_DIR / 'images',
              PSEGS_ASSET_DIR,
              spark=spark)
    
    sd_df_actual = sdt.to_spark_df()
    
    testutil.check_stamped_datum_dfs_equal(
      spark,
      sd_df_actual,
      sd_df_expected_path=FIXTURES_DIR / 'test_colmap_sdt_expected.parquet')
    