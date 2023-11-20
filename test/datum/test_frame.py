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

from psegs.datum.frame import Frame
from psegs.datum.stamped_datum import StampedDatum
from psegs.datum.uri import URI


def test_frame_to_from_uri():
  def check_eq(frame, uri_str):
    f_uri = frame.uri
    assert f_uri == URI.from_str(uri_str)
    assert str(f_uri) == uri_str

  check_eq(
    Frame(
        uri=URI(dataset='d'),
        datums=[
          StampedDatum(uri=URI(topic='t', timestamp=1)),
        ]),
    'psegs://dataset=d&sel_datums=t,1')

