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
      return Path('/outer_root/media/970-evo-plus-raid0/lidarphone_lidar_scans/')
    
    @classmethod
    def get_all_seg_uris(cls):
      seg_uris = []
      seg_uris += cls.get_threeDScannerApp_segment_uris()
        # Room for other recording sources ...

      BROKEN_SEGMENTS = (
        'Untitled Scan', 'amiot-crow-bar', 'headlands-downhill-2',
        'headlands-long-descent', 'san-anselmo-rock-fort-broken',
      )
      seg_uris = [
        suri for suri in seg_uris
        if suri.segment_id not in BROKEN_SEGMENTS
      ]
      return seg_uris


  class MyIOSLidarSDTableFactory(IOSLidarSDTFactory):
    FIXTURES = MyFixtures

  CanonicalFactory.SDT_FACTORIES += [MyIOSLidarSDTableFactory]
