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

import copy

from psegs.datum.stamped_datum import StampedDatum
from psegs.datum.transform import Transform
from psegs.datum.uri import URI


# def test_sd_to_from_uri():
#   def check_eq(sd, s):
#     # Check StampedDatum -> URI
#     assert str(sd.uri) == s
#     assert sd.uri == URI.from_str(s)

#     # Check URI -> StampedDatum
#     sd_bare = copy.deepcopy(sd)
#     for k in StampedDatum.__slots__:
#       default = getattr(StampedDatum(), k)
#       setattr(sd_bare, k, default)
#     uri_bare = sd_bare.uri
#     assert StampedDatum.from_uri(uri_bare) == sd_bare
#     assert StampedDatum.from_str(str(uri_bare)) == sd_bare

#   check_eq(StampedDatum(), URI.PREFIX)

#   check_eq(
#     StampedDatum(
#       dataset='d', split='s', segment_id='s', timestamp=1, topic='t'),
#     'psegs://dataset=d&split=s&segment_id=s&timestamp=1&topic=t')
  
#   check_eq(
#     StampedDatum(
#       dataset='d', split='s', segment_id='s', timestamp=1, topic='t',
#       transform=Transform()),
#     'psegs://dataset=d&split=s&segment_id=s&timestamp=1&topic=t')

#   # Ensure extra works
#   check_eq(
#     StampedDatum(dataset='d', extra={'a': 'foo', 'b': 'bar'}),
#     'psegs://dataset=d&extra.a=foo&extra.b=bar')

