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

from test import testutil

from psegs.datasets import tanks_and_temples as tnt

###############################################################################
## Stamped Datum Table


EXPECTED_SEGMENTS = (
  'psegs://dataset=tanks-and-temples&split=train&segment_id=Barn&extra.tnt.scene=Barn',
  'psegs://dataset=tanks-and-temples&split=train&segment_id=Caterpillar&extra.tnt.scene=Caterpillar',
  'psegs://dataset=tanks-and-temples&split=train&segment_id=Church&extra.tnt.scene=Church',
  'psegs://dataset=tanks-and-temples&split=train&segment_id=Courthouse&extra.tnt.scene=Courthouse',
  'psegs://dataset=tanks-and-temples&split=train&segment_id=Ignatius&extra.tnt.scene=Ignatius',
  'psegs://dataset=tanks-and-temples&split=train&segment_id=Meetingroom&extra.tnt.scene=Meetingroom',
  'psegs://dataset=tanks-and-temples&split=train&segment_id=Truck&extra.tnt.scene=Truck',
)

def test_tnt_all_segment_uris():
  testutil.skip_if_fixture_absent(tnt.Fixtures.ROOT)
  actual = tnt.TanksAndTemplesSDTable.get_all_segment_uris()
  assert sorted(EXPECTED_SEGMENTS) == sorted(str(uri) for uri in actual)

###############################################################################
## DSUtil Tests

def test_tnt_dsutil_smoke():
  testutil.skip_if_fixture_absent(tnt.Fixtures.ROOT)
  testutil.skip_if_fixture_absent(tnt.Fixtures.EXTERNAL_FIXTURES_ROOT)

  # The above are preconditions, so this should succeed:
  assert tnt.DSUtil.emplace()
