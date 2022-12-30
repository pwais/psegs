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

from psegs.datasets import nuscenes
from psegs.datasets import kitti
from psegs.datasets import ios_lidar
from psegs.datasets import tanks_and_temples

DS_TO_UTIL_IMPL = {
  'kitti': kitti.DSUtil,
  'nuscenes': nuscenes.NuscDSUtil,
  'ios_lidar': ios_lidar.DSUtil,
  'tanks_and_temples': tanks_and_temples.DSUtil,
}

def run(dataset):
  assert dataset in DS_TO_UTIL_IMPL, (
    "Unknown dataset %s, choices: %s" % (
      dataset, sorted(DS_TO_UTIL_IMPL.keys())))
  
  dsutil_impl = DS_TO_UTIL_IMPL[dataset]

  assert dsutil_impl.emplace()
  assert dsutil_impl.test()
  #assert dsutil_impl.demo()
  # assert dsutil_impl.build_table()
