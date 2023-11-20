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


class LocalDiskCache(object):
  """Defines the API that PSegs expects to cache clients and provides
  a simple local adhoc disk cache."""

  def __init__(self):
    """Cache clients must have a zero-arg ctor"""
    pass

  def new_filepath(self, fname, t=None):
    from psegs.conf import C
    dest = C.DATA_ROOT / 'psegs_local_disk_cache' / 'adhoc_files' / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest

  def new_dirpath(self, dirpath, t=None):
    from psegs.conf import C
    dest = C.DATA_ROOT / 'psegs_local_disk_cache' / 'adhoc_dirs' / dirpath
    dest.mkdir(parents=True, exist_ok=True)
    return dest


class AssetDiskCache(LocalDiskCache):

  def __init__(self, config=None):
    """get canonical psegs config from somewhere or write to /opt/psegs/psegs_temp / dataroot stuff
    """
    self.yay = None

  def new_filepath(self, fname, t=None):
    raise NotImplementedError

  def new_dirpath(self, dirpath, t=None):
    raise NotImplementedError

