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

import shelve

from oarphpy import util as oputil

from psegs import util
from psegs.conf import C



###############################################################################
### NuScenes Fixtures & Other Constants

class Fixtures(object):

  NUSC_ROOT = C.EXT_DATA_ROOT / 'kitti_archives'

  @classmethod
  def index_root(cls, nusc_version):
    """A r/w place to cache any temp / index data"""
    return C.PS_TEMP / 'nuscenes' / nusc_version


from nuscenes.nuscenes import NuScenes
class PSegsNuScenes(NuScenes):
  """A Psegs wrappers around the NuScenes dataset handle.
  
  The base NuScenes object uses 8GB resident RAM (each instance) due to
  the "tables" of JSON data that it loads.  Below we replace these "tables"
  with disk-based `shelve`s in order to dramatically reduce memory usage.
  This change is needed in order to support instantiating multiple
  NuScenes readers per machine (e.g. in Spark or any parallel use case).

  To warm the disk--based caches (requires one-time temporary 8GB+ memory
  usage), simply instantiate an instance of this class.
  """

  FIXTURES = Fixtures

  def _get_cache_path(self, table_name):
    """Return a path to a shelve cache of the given nuscenes `table_name`"""
    return self.FIXTURES.index_root(self.version) / table_name

  def __init__(self, **kwargs):
    """FMI see NuScenes.__init__().  The parent class will read JSON blobs
    and load 8GB+ data into resident memory.  In this override, we load data
    using the parent NuScenes implemenation (thus, temporarily, using the same
    amount of resident memory) but then cache the table data to disk using
    python shelve`.  We then free the resident memory and use the disk-based
    for token-based access.
    """

    self.version = kwargs['version']
      # Base ctor does this, but we'll do it here so that path-resolving
      # utils below work properly
    
    if util.missing_or_empty(self._get_cache_path('')):
      util.log.info("Creating shelve caches.  Reading source JSON ...")
      nusc = NuScenes(**kwargs)
        # NB: The above ctor call not only loads all JSON but also runs
        # 'reverse indexing', which **EDITS** the data loaded into memory.
        # We'll then write the edited data below using `shelve` so that we
        # don't have to try to make `PSegsNuScenes` support reverse indexing
        # itself.
      util.log.info("... NuScenes done loading & indexing JSON data ...")
      
      for table_name in nusc.table_names:
        cache_path = self._get_cache_path(table_name)
        oputil.mkdir(cache_path.parent)

        util.log.info(
          "Building shelve cache for %s (in %s) ..." % (
            table_name, cache_path))
        
        import pickle
        d = shelve.open(str(cache_path), protocol=pickle.HIGHEST_PROTOCOL)
        rows = getattr(nusc, table_name) # E.g. self.sample_data
        d.update((r['token'], r) for r in rows)
        d.close()
      util.log.info("... done.")
      del nusc # Free several GB memory
    
    super(PSegsNuScenes, self).__init__(**kwargs)

  def _get_table(self, table_name):
    attr = '_cached_' + table_name
    if not hasattr(self, attr):
      cache_path = self._get_cache_path(table_name)
      util.log.info(
        "Using shelve cache for %s at %s" % (table_name, cache_path))
      d = shelve.open(str(cache_path))
      setattr(self, attr, d)
    return getattr(self, attr)

  def __load_table__(self, table_name):
    return self._get_table(table_name).values()
      # NB: Despite the type annotation in the parent class, the base method
      # actually returns a list of dicts and not a single dict.  This
      # subclass method returns a Values View (a generator-like thing)
      # and does not break any core NuScenes functionality.
  
  def __make_reverse_index__(self, verbose):
    # NB: Shelve data files have, built-in, the reverse indicies that the
    # base `NuScenes` creates.  See above.  This override allows the subclass
    # to safely invoke the parent class CTor.
    pass
  
  def get(self, table_name, token):
    assert table_name in self.table_names, \
      "Table {} not found".format(table_name)
    return self._get_table(table_name)[token]
  
  def getind(self, table_name, token):
    # This override should be safe due to our override of `get()` above"""
    raise ValueError("Unsupported / unnecessary; provided by shelve")


def test_nusc_yay():

  nusc = PSegsNuScenes(
    version='v1.0-trainval',
    dataroot='/outer_root//media/seagates-ext4/au_datas/nuscenes_root/')
