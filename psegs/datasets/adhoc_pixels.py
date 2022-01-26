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

import os
from pathlib import Path

from psegs import datum
from psegs import util
from psegs.table.sd_table import StampedDatumTable
from psegs.table.sd_table_factory import StampedDatumTableFactory

def video_to_sdt(
      video_uri,
      dataset='anon',
      split='anon',
      segment_id=None,
      limit=-1):
  
  import imageio

  if segment_id is None:
    from urllib.parse import urlparse
    res = urlparse(video_uri)
    path = res.path
    fname = os.path.split(path)[-1]
    segment_id = fname

  r = imageio.get_reader(video_uri)
  n_frames = r.get_length()
  if n_frames == float('inf'):
    pass

  out_dir = Path(out_dir)

  
  n_frames = r.get_length()
  util.log.info(
    "About to extract %s from %s to %s ..." % (n_frames, video_uri, out_dir))

  out_dir.mkdir(parents=True, exist_ok=True)
