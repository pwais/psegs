# Copyright 2020 Maintainers of PSegs
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

try:
  import nuscenes
except ImportError:
  pytest.skip("Skipping nuscenes-only tests", allow_module_level=True)

from oarphpy import util as oputil

from psegs import datum
from psegs import util
from psegs.datasets import nuscenes as psnusc

from test import testutil


###############################################################################
### Test Utils

skip_if_no_nusc_mini = pytest.mark.skipif(
  not psnusc.NuscFixtures.version_exists('v1.0-mini'),
  reason="Requires NuScenes v1.0-trainval")

skip_if_no_nusc_trainval = pytest.mark.skipif(
  not psnusc.NuscFixtures.version_exists('v1.0-trainval'),
  reason="Requires NuScenes v1.0-mini")

###############################################################################
### Test NuScenes

@skip_if_no_nusc_mini
def test_nuscenes_mini_stats():
  nusc = psnusc.PSegsNuScenes(version='v1.0-mini', verbose=False)

  TABLE_TO_EXPECTED_LENGTH = {
    'attribute': 8,
    'calibrated_sensor': 120,
    'category': 32,
    'ego_pose': 31206,
    'instance': 911,
    'log': 8,
    'map': 4,
    'sample': 404,
    'sample_annotation': 18538,
    'sample_data': 31206,
    'scene': 10,
    'sensor': 12,
    'visibility': 4
  }

  actual = nusc.get_table_to_length()
  if 'lidarseg' in actual:
    TABLE_TO_EXPECTED_LENGTH['lidarseg'] = 404

  assert actual == TABLE_TO_EXPECTED_LENGTH


@skip_if_no_nusc_trainval
def test_nuscenes_trainval_stats():
  nusc = psnusc.PSegsNuScenes(version='v1.0-trainval', verbose=False)

  TABLE_TO_EXPECTED_LENGTH = {
    'attribute': 8,
    'calibrated_sensor': 10200,
    'category': 32,
    'ego_pose': 2631083,
    'instance': 64386,
    'log': 68,
    'map': 4,
    'sample': 34149,
    'sample_annotation': 1166187,
    'sample_data': 2631083,
    'scene': 850,
    'sensor': 12,
    'visibility': 4
  }

  actual = nusc.get_table_to_length()
  if 'lidarseg' in actual:
    TABLE_TO_EXPECTED_LENGTH['lidarseg'] = 34149

  assert actual == TABLE_TO_EXPECTED_LENGTH

def _check_sample(sample, testname):
  prefix = sample.uri.segment_id
  # outdir = testutil.test_tempdir(testname + '_' + prefix)

  from pathlib import Path
  outdir = Path('/opt/psegs/test_run_output/')
  # oputil.cleandir(outdir)

  def save(path, img):
    import imageio
    imageio.imwrite(path, img)
    print(path)

  cuboids = sample.cuboid_labels
  for pc in sample.lidar_clouds:
    path = outdir / ('%s_bev.png' % pc.sensor_name)
    save(path, pc.get_bev_debug_image(cuboids=cuboids))
    
    path = outdir / ('%s_rv.png' % pc.sensor_name)
    save(path, pc.get_front_rv_debug_image(cuboids=cuboids))

  for ci in sample.camera_images:
    path = outdir / ('%s_debug.png' % ci.sensor_name)
    save(
      path,
      ci.get_debug_image(
        clouds=sample.lidar_clouds,
        cuboids=cuboids))
  
  

  # datum_rdd = T.get_segment_datum_rdd(spark, myseg)
  # print('datum_rdd.count()', datum_rdd.count())
  # datums = datum_rdd.take(10)
  # import ipdb; ipdb.set_trace()

def test_nusenes_create_sd():
  # samples = ['psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&sel_datums=camera|CAM_BACK,1537292951937558000,camera|CAM_BACK_LEFT,1537292951947405000,camera|CAM_BACK_RIGHT,1537292951928113000,camera|CAM_FRONT,1537292951912404000,camera|CAM_FRONT_LEFT,1537292951904799000,camera|CAM_FRONT_RIGHT,1537292951920482000,ego_pose,1537292951904799000,ego_pose,1537292951912404000,ego_pose,1537292951920482000,ego_pose,1537292951928113000,ego_pose,1537292951933926000,ego_pose,1537292951937558000,ego_pose,1537292951945648000,ego_pose,1537292951947405000,ego_pose,1537292951949628000,ego_pose,1537292951954005000,ego_pose,1537292951954663000,ego_pose,1537292951976984000,labels|cuboids,1537292951904799000,labels|cuboids,1537292951912404000,labels|cuboids,1537292951920482000,labels|cuboids,1537292951928113000,labels|cuboids,1537292951933926000,labels|cuboids,1537292951937558000,labels|cuboids,1537292951945648000,labels|cuboids,1537292951947405000,labels|cuboids,1537292951949628000,labels|cuboids,1537292951954005000,labels|cuboids,1537292951954663000,labels|cuboids,1537292951976984000,lidar|LIDAR_TOP,1537292951949628000,radar|RADAR_BACK_LEFT,1537292951954005000,radar|RADAR_BACK_RIGHT,1537292951954663000,radar|RADAR_FRONT,1537292951945648000,radar|RADAR_FRONT_LEFT,1537292951976984000,radar|RADAR_FRONT_RIGHT,1537292951933926000', 'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0513&sel_datums=camera|CAM_BACK,1535478901787558000,camera|CAM_BACK_LEFT,1535478901797405000,camera|CAM_BACK_RIGHT,1535478901778113000,camera|CAM_FRONT,1535478901762404000,camera|CAM_FRONT_LEFT,1535478901754799000,camera|CAM_FRONT_RIGHT,1535478901770482000,ego_pose,1535478901754799000,ego_pose,1535478901762404000,ego_pose,1535478901770480000,ego_pose,1535478901770482000,ego_pose,1535478901778113000,ego_pose,1535478901787558000,ego_pose,1535478901796360000,ego_pose,1535478901797405000,ego_pose,1535478901803288000,ego_pose,1535478901813085000,ego_pose,1535478901815802000,ego_pose,1535478901832909000,labels|cuboids,1535478901754799000,labels|cuboids,1535478901762404000,labels|cuboids,1535478901770480000,labels|cuboids,1535478901770482000,labels|cuboids,1535478901778113000,labels|cuboids,1535478901787558000,labels|cuboids,1535478901796360000,labels|cuboids,1535478901797405000,labels|cuboids,1535478901803288000,labels|cuboids,1535478901813085000,labels|cuboids,1535478901815802000,labels|cuboids,1535478901832909000,lidar|LIDAR_TOP,1535478901796360000,radar|RADAR_BACK_LEFT,1535478901770480000,radar|RADAR_BACK_RIGHT,1535478901813085000,radar|RADAR_FRONT,1535478901815802000,radar|RADAR_FRONT_LEFT,1535478901803288000,radar|RADAR_FRONT_RIGHT,1535478901832909000', 'psegs://dataset=nuscenes&split=train_detect&segment_id=scene-0750&sel_datums=camera|CAM_BACK,1535656879787558000,camera|CAM_BACK_LEFT,1535656879797405000,camera|CAM_BACK_RIGHT,1535656879778113000,camera|CAM_FRONT,1535656879762404000,camera|CAM_FRONT_LEFT,1535656879754799000,camera|CAM_FRONT_RIGHT,1535656879770482000,ego_pose,1535656879754799000,ego_pose,1535656879762404000,ego_pose,1535656879770482000,ego_pose,1535656879778113000,ego_pose,1535656879781462000,ego_pose,1535656879787558000,ego_pose,1535656879797405000,ego_pose,1535656879801090000,ego_pose,1535656879805167000,ego_pose,1535656879819687000,ego_pose,1535656879823023000,ego_pose,1535656879832112000,labels|cuboids,1535656879754799000,labels|cuboids,1535656879762404000,labels|cuboids,1535656879770482000,labels|cuboids,1535656879778113000,labels|cuboids,1535656879781462000,labels|cuboids,1535656879787558000,labels|cuboids,1535656879797405000,labels|cuboids,1535656879801090000,labels|cuboids,1535656879805167000,labels|cuboids,1535656879819687000,labels|cuboids,1535656879823023000,labels|cuboids,1535656879832112000,lidar|LIDAR_TOP,1535656879801090000,radar|RADAR_BACK_LEFT,1535656879832112000,radar|RADAR_BACK_RIGHT,1535656879805167000,radar|RADAR_FRONT,1535656879819687000,radar|RADAR_FRONT_LEFT,1535656879781462000,radar|RADAR_FRONT_RIGHT,1535656879823023000']
  
  # T = psnusc.NuscStampedDatumTableBase
  # uris = T.iter_uris_for_segment('scene-0594') # with only keyframes!!!
  # uris = [str(u) for u in sorted(uris)]
  # first_cuboid = None
  # for u in uris:
  #   if 'cuboids' in u:
  #     first_cuboid = u
  #     break
  # assert first_cuboid
  # uris = [u for u in uris if 'cuboids' not in u]
  # uris = [first_cuboid] + uris

  # import pprint
  # pprint.pprint(uris[:30])
  # assert False

  SAMPLE_URIS = [
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943354799000&topic=labels|cuboids&extra.nuscenes-is-keyframe=True&extra.nuscenes-label-channel=CAM_FRONT_LEFT&extra.nuscenes-sample-token=fe6f79aed6ea4b7b9f87be3d68248f54&extra.nuscenes-token=sample_data|d141f680981f4c018e066751c2e8a489',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943354799000&topic=camera|CAM_FRONT_LEFT&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=sample_data|d141f680981f4c018e066751c2e8a489',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943354799000&topic=ego_pose&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=ego_pose|d141f680981f4c018e066751c2e8a489',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943362404000&topic=camera|CAM_FRONT&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=sample_data|8673669e2ece4fd2be37583b670d6c89',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943362404000&topic=ego_pose&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=ego_pose|8673669e2ece4fd2be37583b670d6c89',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943364426000&topic=ego_pose&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=ego_pose|5bb7e6318ccb4ed08d56ed23e0673d43',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943364426000&topic=radar|RADAR_FRONT_LEFT&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=sample_data|5bb7e6318ccb4ed08d56ed23e0673d43',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943367494000&topic=ego_pose&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=ego_pose|b595147868d74be9a7b8945c04cb36ee',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943367494000&topic=radar|RADAR_FRONT&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=sample_data|b595147868d74be9a7b8945c04cb36ee',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943370482000&topic=camera|CAM_FRONT_RIGHT&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=sample_data|770d389cc3964c3bae61ff2d032f621e',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943370482000&topic=ego_pose&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=ego_pose|770d389cc3964c3bae61ff2d032f621e',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943378113000&topic=camera|CAM_BACK_RIGHT&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=sample_data|8484e12be28f4795afe053f1ce82887d',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943378113000&topic=ego_pose&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=ego_pose|8484e12be28f4795afe053f1ce82887d',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943387558000&topic=camera|CAM_BACK&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=sample_data|43de58db5c714ae791e49712fab4be40',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943387558000&topic=ego_pose&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=ego_pose|43de58db5c714ae791e49712fab4be40',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943391357000&topic=ego_pose&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=ego_pose|b64f953571e4490eb909454bcb66a9f4',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943391357000&topic=radar|RADAR_BACK_LEFT&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=sample_data|b64f953571e4490eb909454bcb66a9f4',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943397405000&topic=camera|CAM_BACK_LEFT&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=sample_data|affd9e5eed1b480b97e411c3e473fe4a',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943397405000&topic=ego_pose&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=ego_pose|affd9e5eed1b480b97e411c3e473fe4a',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943398040000&topic=ego_pose&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=ego_pose|b99d95a1c87a4d5c9bb220f6f337203b',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943398040000&topic=radar|RADAR_BACK_RIGHT&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=sample_data|b99d95a1c87a4d5c9bb220f6f337203b',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943399770000&topic=ego_pose&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=ego_pose|f953a1ecb5a046a49d5d244a57820232',
    'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&timestamp=1537292943399770000&topic=lidar|LIDAR_TOP&extra.nuscenes-is-keyframe=True&extra.nuscenes-sample-token=ad4b2f2f60084f479261bfce1448af5e&extra.nuscenes-token=sample_data|f953a1ecb5a046a49d5d244a57820232',
  ]
  SAMPLE_URIS = [datum.URI.from_str(s) for s in SAMPLE_URIS]

  T = psnusc.NuscStampedDatumTableBase
  s = datum.Sample(datums=[T.create_stamped_datum(u) for u in SAMPLE_URIS])
  _check_sample(s, 'test_nusenes_create_sd')


def test_nuscenes_yay():

  # nusc = psnusc.PSegsNuScenes(
  #   version='v1.0-trainval',
  #   dataroot='/outer_root//media/seagates-ext4/au_datas/nuscenes_root/')

  # import pdb; pdb.set_trace()

  # from pprint import pprint
  # pprint(nusc.get_all_sensors())
  # pprint(nusc.get_all_classes())

  # pprint(('list_lidarseg_categories', nusc.list_lidarseg_categories(sort_by='count')))
  # pprint(('lidarseg_idx2name_mapping', nusc.lidarseg_idx2name_mapping))



  # KEYFRAMES_ONLY = True
  # with testutil.LocalSpark.getOrCreate() as spark:
  #   import random
  #   rand = random.Random(12)
  #   T = psnusc.NuscStampedDatumTableBase
  #   suris = rand.sample(T.get_all_segment_uris(), 3)
  #   nusc_samples = []
  #   for suri in suris:
  #     seg_df = T.get_segment_datum_df(spark, suri)
  #     if KEYFRAMES_ONLY:
  #       seg_df = seg_df.where('uri.extra.`nuscenes-is-keyframe` == "True"')
  #     row = seg_df.select('uri.extra.nuscenes-sample-token').first()
  #     sample_token = row[0]

  #     sample_df = seg_df.where(
  #       seg_df['uri.extra.nuscenes-sample-token'] == sample_token)
      
  #     sample_uri_df = sample_df.select('uri')
  #     sample_uris = [r.uri for r in T.sd_df_to_rdd(sample_uri_df).collect()]
  #     nusc_samples.append(datum.URI.segment_uri_from_datum_uris(sample_uris))

  samples = ['psegs://dataset=nuscenes&split=train_track&segment_id=scene-0594&sel_datums=camera|CAM_BACK,1537292951937558000,camera|CAM_BACK_LEFT,1537292951947405000,camera|CAM_BACK_RIGHT,1537292951928113000,camera|CAM_FRONT,1537292951912404000,camera|CAM_FRONT_LEFT,1537292951904799000,camera|CAM_FRONT_RIGHT,1537292951920482000,ego_pose,1537292951904799000,ego_pose,1537292951912404000,ego_pose,1537292951920482000,ego_pose,1537292951928113000,ego_pose,1537292951933926000,ego_pose,1537292951937558000,ego_pose,1537292951945648000,ego_pose,1537292951947405000,ego_pose,1537292951949628000,ego_pose,1537292951954005000,ego_pose,1537292951954663000,ego_pose,1537292951976984000,labels|cuboids,1537292951904799000,labels|cuboids,1537292951912404000,labels|cuboids,1537292951920482000,labels|cuboids,1537292951928113000,labels|cuboids,1537292951933926000,labels|cuboids,1537292951937558000,labels|cuboids,1537292951945648000,labels|cuboids,1537292951947405000,labels|cuboids,1537292951949628000,labels|cuboids,1537292951954005000,labels|cuboids,1537292951954663000,labels|cuboids,1537292951976984000,lidar|LIDAR_TOP,1537292951949628000,radar|RADAR_BACK_LEFT,1537292951954005000,radar|RADAR_BACK_RIGHT,1537292951954663000,radar|RADAR_FRONT,1537292951945648000,radar|RADAR_FRONT_LEFT,1537292951976984000,radar|RADAR_FRONT_RIGHT,1537292951933926000', 'psegs://dataset=nuscenes&split=train_track&segment_id=scene-0513&sel_datums=camera|CAM_BACK,1535478901787558000,camera|CAM_BACK_LEFT,1535478901797405000,camera|CAM_BACK_RIGHT,1535478901778113000,camera|CAM_FRONT,1535478901762404000,camera|CAM_FRONT_LEFT,1535478901754799000,camera|CAM_FRONT_RIGHT,1535478901770482000,ego_pose,1535478901754799000,ego_pose,1535478901762404000,ego_pose,1535478901770480000,ego_pose,1535478901770482000,ego_pose,1535478901778113000,ego_pose,1535478901787558000,ego_pose,1535478901796360000,ego_pose,1535478901797405000,ego_pose,1535478901803288000,ego_pose,1535478901813085000,ego_pose,1535478901815802000,ego_pose,1535478901832909000,labels|cuboids,1535478901754799000,labels|cuboids,1535478901762404000,labels|cuboids,1535478901770480000,labels|cuboids,1535478901770482000,labels|cuboids,1535478901778113000,labels|cuboids,1535478901787558000,labels|cuboids,1535478901796360000,labels|cuboids,1535478901797405000,labels|cuboids,1535478901803288000,labels|cuboids,1535478901813085000,labels|cuboids,1535478901815802000,labels|cuboids,1535478901832909000,lidar|LIDAR_TOP,1535478901796360000,radar|RADAR_BACK_LEFT,1535478901770480000,radar|RADAR_BACK_RIGHT,1535478901813085000,radar|RADAR_FRONT,1535478901815802000,radar|RADAR_FRONT_LEFT,1535478901803288000,radar|RADAR_FRONT_RIGHT,1535478901832909000', 'psegs://dataset=nuscenes&split=train_detect&segment_id=scene-0750&sel_datums=camera|CAM_BACK,1535656879787558000,camera|CAM_BACK_LEFT,1535656879797405000,camera|CAM_BACK_RIGHT,1535656879778113000,camera|CAM_FRONT,1535656879762404000,camera|CAM_FRONT_LEFT,1535656879754799000,camera|CAM_FRONT_RIGHT,1535656879770482000,ego_pose,1535656879754799000,ego_pose,1535656879762404000,ego_pose,1535656879770482000,ego_pose,1535656879778113000,ego_pose,1535656879781462000,ego_pose,1535656879787558000,ego_pose,1535656879797405000,ego_pose,1535656879801090000,ego_pose,1535656879805167000,ego_pose,1535656879819687000,ego_pose,1535656879823023000,ego_pose,1535656879832112000,labels|cuboids,1535656879754799000,labels|cuboids,1535656879762404000,labels|cuboids,1535656879770482000,labels|cuboids,1535656879778113000,labels|cuboids,1535656879781462000,labels|cuboids,1535656879787558000,labels|cuboids,1535656879797405000,labels|cuboids,1535656879801090000,labels|cuboids,1535656879805167000,labels|cuboids,1535656879819687000,labels|cuboids,1535656879823023000,labels|cuboids,1535656879832112000,lidar|LIDAR_TOP,1535656879801090000,radar|RADAR_BACK_LEFT,1535656879832112000,radar|RADAR_BACK_RIGHT,1535656879805167000,radar|RADAR_FRONT,1535656879819687000,radar|RADAR_FRONT_LEFT,1535656879781462000,radar|RADAR_FRONT_RIGHT,1535656879823023000']

  
  import imageio
  T = psnusc.NuscStampedDatumTableBase
  with testutil.LocalSpark.getOrCreate() as spark:
    for suri in samples:
      sample = T.get_sample(suri, spark=spark)
      prefix = sample.uri.segment_id

      cuboids = sample.cuboid_labels
      for pc in sample.lidar_clouds:
        img = pc.get_bev_debug_image(cuboids=cuboids)
        imageio.imwrite(
          '/opt/psegs/test_run_output/%s-%s-bev.png' % (prefix, pc.sensor_name), img)

        img = pc.get_front_rv_debug_image(cuboids=cuboids)
        imageio.imwrite(
          '/opt/psegs/test_run_output/%s-%s-rv.png' % (prefix, pc.sensor_name), img)

      for ci in sample.camera_images:
        img = ci.get_debug_image(clouds=sample.lidar_clouds, cuboids=cuboids)
        imageio.imwrite(
          '/opt/psegs/test_run_output/%s-%s-debug.png' % (prefix, ci.sensor_name), img)
        
          

      # datum_rdd = T.get_segment_datum_rdd(spark, myseg)
      # print('datum_rdd.count()', datum_rdd.count())
      # datums = datum_rdd.take(10)
      # import ipdb; ipdb.set_trace()
      print(suri)


  # for tests, let's look at both keyframes and interpolated stuff!
  # rename frame to sample; let's add way to get a sample
  # from table via URI (and this should be decent no matter backing
  # store)