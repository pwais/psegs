# Copyright 2023 Maintainers of PSegs
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

from psegs import datum

from test import testutil

def test_matched_pair_stereo_rect_viz_html():
  
  
  from psegs.datasets import colmap as pscolmap

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
    
    sd_df = sdt.to_spark_df()




    sd_rdd = sdt.get_datum_rdd_matching(only_types=['matched_pair'])

    from pyspark import StorageLevel
    sd_rdd = sd_rdd.persist(StorageLevel.MEMORY_AND_DISK)

    def get_cam_key(ci):
      # Try to get a segment-distinct, if not globally distinct, key for a
      # `camera_image`.  TODO: include affordance for user override (maybe
      # a dataset / segment has an image id in `extra` ?) and/or include
      # `URI`s in `MatchedPair` (but )
      return (
        ci.sensor_name, ci.timestamp, ci.width, ci.height,
        tuple(ci.ego_to_sensor.get_transformation_matrix().flatten().tolist()),
        tuple(ci.ego_pose.get_transformation_matrix().flatten().tolist()),
      )

    def sd_to_key_plotdatas(sd):
      mp = sd.matched_pair
      lkey = get_cam_key(mp.img1)
      rkey = get_cam_key(mp.img2)
      return [
        (lkey, ('key_is_1', sd.uri, mp)),
        (rkey, ('key_is_2', sd.uri, mp)),
      ]
    
    key_plotdata_rdd = sd_rdd.flatMap(sd_to_key_plotdatas)
    key_to_plotdatas_rdd = key_plotdata_rdd.groupByKey()

    
    # distinct_ci_key_rdd = data_rdd.flatMap(
    #   lambda lkey_rkey_mpuri_mp: 
    #     tuple(lkey_rkey_mpuri_mp[:2])).distinct()
    # distinct_keys = sorted(distinct_ci_key_rdd.collect())

    key, iter_plotdata = key_to_plotdatas_rdd.first()
    iter_plotdata = sorted(iter_plotdata)

    ci_left = None
    ci_rights = []
    lr_matches = []
    mp_uris = []
    for indicator, mp_uri, mp in iter_plotdata:
      
      if indicator == 'key_is_1':
        if ci_left is None:
          ci_left = mp.img1
        
        ci_right = mp.img2
        matches = mp.get_x1y1x2y2_extra()
        
      elif indicator == 'key_is_2':
        if ci_left is None:
          ci_left = mp.img2
        
        ci_right = mp.img1
        matches = mp.get_x1y1x2y2_extra()

        # Flip left-right x,y cols since the "left" image in this case is x2y2
        cols = list(range(matches.shape[1]))
        cols[0] = 2
        cols[1] = 3
        cols[2] = 0
        cols[3] = 1
        matches = matches[:, cols]
        
      else:
        raise ValueError(indicator)
      
      ci_rights.append(ci_right)
      lr_matches.append(matches)
      mp_uris.append(mp_uri)

    
    assert ci_left is not None

    from psegs.datum.matched_pair import create_stereo_rect_pair_debug_view_html
    html = create_stereo_rect_pair_debug_view_html(
              ci_left,
              ci_rights=ci_rights,
              lr_matches=lr_matches,
              mp_uris=mp_uris)

    with open('/opt/psegs/mp_test.html', 'w') as f:
      f.write(html)
    return
    breakpoint()
    sd_df.createOrReplaceTempView('')
    print()
  

