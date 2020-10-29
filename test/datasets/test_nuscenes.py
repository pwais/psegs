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

from psegs.datasets import nuscenes as psnusc

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


def test_nuscenes_yay():

  # nusc = psnusc.PSegsNuScenes(
  #   version='v1.0-trainval',
  #   dataroot='/outer_root//media/seagates-ext4/au_datas/nuscenes_root/')

  # from pprint import pprint
  # pprint(nusc.get_all_sensors())
  # pprint(nusc.get_all_classes())

  # pprint(('list_lidarseg_categories', nusc.list_lidarseg_categories(sort_by='count')))
  # pprint(('lidarseg_idx2name_mapping', nusc.lidarseg_idx2name_mapping))


  psnusc.NuscStampedDatumTableBase.build()
