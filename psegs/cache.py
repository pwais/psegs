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


class IAssetCache(object):

  def new_filepath(self, fname, t=None):
    raise NotImplementedError

  def new_dirpath(self, dirpath, t=None):
    raise NotImplementedError



class AssetDiskCache(object):

  def __init__(self, config=None):
    """get canonical psegs config from somewhere or write to /opt/psegs/psegs_temp / dataroot stuff
    """
    self.yay = None

  def new_filepath(self, fname, t=None):
    raise NotImplementedError

  def new_dirpath(self, dirpath, t=None):
    raise NotImplementedError

