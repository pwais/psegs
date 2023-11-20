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
  assert ci.extra['colmap.image_name'] == 'frame_00012.jpg'
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


def test_colmap_get_image_name_to_covis_names():
  pytest.importorskip("pycolmap")

  FIXTURES_DIR = testutil.test_fixtures_dir() / 'test_colmap'
  recon_dir = FIXTURES_DIR / 'sparse' / '0'

  import pycolmap
  recon = pycolmap.Reconstruction(recon_dir)

  image_name_to_covis_names = (
    pscolmap.colmap_get_image_name_to_covis_names(recon))

  assert image_name_to_covis_names == EXPECTD_COVIS


def test_colmap_create_matched_pair():
  pytest.importorskip("pycolmap")

  FIXTURES_DIR = testutil.test_fixtures_dir() / 'test_colmap'
  recon_dir = FIXTURES_DIR / 'sparse' / '0'

  import pycolmap
  recon = pycolmap.Reconstruction(recon_dir)
  image_name_to_covis_names = (
    pscolmap.colmap_get_image_name_to_covis_names(recon))

  EXPECTED_PAIRS_TO_TEST = (
    ('frame_00033.jpg', 'frame_00003.jpg'),
    ('frame_00012.jpg', 'frame_00003.jpg'),
  )

  for image1_name, image2_name in EXPECTED_PAIRS_TO_TEST:
    assert image1_name in image_name_to_covis_names[image2_name]
    assert image2_name in image_name_to_covis_names[image1_name]

    mp = pscolmap.colmap_recon_create_matched_pair(
          image1_name,
          image2_name,
          recon_dir,
          img1='not_null_sentinel',
          img2='not_null_sentinel')

    matches = mp.get_matches()

    # yapf: disable
    assert mp.matches_colnames == [
      'x1', 'y1', 'x2', 'y2',
      'r', 'g', 'b',
      'world_x', 'world_y', 'world_z',
      'error', 'track_length', 'colmap_p3id',
    ]# yapf: enable

    assert matches.shape[1] == len(mp.matches_colnames)
    
    # Spot check some numbers we pulled manually from 
    # yapf: disable
    EXPECTED_IM1_TO_MATCHES_ROWS = {
      'frame_00033.jpg': [
        (129.14500427246094, 169.54025268554688,
              109.0311279296875, 176.9521484375,
         102., 102., 90.,
         -2.2979449902111684, 6.357728828532021, 28.658129770725225,
         0.29433191072803566, 4., 1.),
      ],
    }
    # yapf: enable
    
    expected_match_rows = EXPECTED_IM1_TO_MATCHES_ROWS.get(image1_name, [])
    actual_match_rows = set(tuple(r) for r in matches)
    for expected_row in expected_match_rows:
      assert expected_row in actual_match_rows


    assert mp.img1 == 'not_null_sentinel'
    assert mp.img2 == 'not_null_sentinel'
    assert mp.extra['colmap.image1_name'] == image1_name
    assert mp.extra['colmap.image2_name'] == image2_name

    # Now ensure the image parsing works
    src_images_dir = FIXTURES_DIR / 'images'

    mp = pscolmap.colmap_recon_create_matched_pair(
          image1_name,
          image2_name,
          recon_dir,
          src_images_dir=src_images_dir)

    assert mp.img1.extra['colmap.image_name'] == image1_name
    assert mp.img2.extra['colmap.image_name'] == image2_name

  # TODO we don't have any pairs that are not covisible in this fixture


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



EXPECTD_COVIS = {'frame_00000.jpg': [
                     'frame_00003.jpg',
                     'frame_00006.jpg',
                     'frame_00009.jpg',
                     'frame_00012.jpg',
                     'frame_00015.jpg',
                     'frame_00030.jpg',
                     'frame_00033.jpg'],
 'frame_00003.jpg': ['frame_00000.jpg',
                     'frame_00006.jpg',
                     'frame_00009.jpg',
                     'frame_00012.jpg',
                     'frame_00015.jpg',
                     'frame_00030.jpg',
                     'frame_00033.jpg'],
 'frame_00006.jpg': ['frame_00000.jpg',
                     'frame_00003.jpg',
                     'frame_00009.jpg',
                     'frame_00012.jpg',
                     'frame_00015.jpg',
                     'frame_00030.jpg',
                     'frame_00033.jpg'],
 'frame_00009.jpg': ['frame_00000.jpg',
                     'frame_00003.jpg',
                     'frame_00006.jpg',
                     'frame_00012.jpg',
                     'frame_00015.jpg',
                     'frame_00030.jpg',
                     'frame_00033.jpg'],
 'frame_00012.jpg': ['frame_00000.jpg',
                     'frame_00003.jpg',
                     'frame_00006.jpg',
                     'frame_00009.jpg',
                     'frame_00015.jpg',
                     'frame_00030.jpg',
                     'frame_00033.jpg'],
 'frame_00015.jpg': ['frame_00000.jpg',
                     'frame_00003.jpg',
                     'frame_00006.jpg',
                     'frame_00009.jpg',
                     'frame_00012.jpg',
                     'frame_00030.jpg',
                     'frame_00033.jpg'],
 'frame_00030.jpg': ['frame_00000.jpg',
                     'frame_00003.jpg',
                     'frame_00006.jpg',
                     'frame_00009.jpg',
                     'frame_00012.jpg',
                     'frame_00015.jpg',
                     'frame_00033.jpg'],
 'frame_00033.jpg': ['frame_00000.jpg',
                     'frame_00003.jpg',
                     'frame_00006.jpg',
                     'frame_00009.jpg',
                     'frame_00012.jpg',
                     'frame_00015.jpg',
                     'frame_00030.jpg']}