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

from psegs.conf import C
from psegs.datasets.idsutil import IDatasetUtil



###############################################################################
### KITTI-360 Fixtures & Other Constants

class Fixtures(object):

  ROOT = C.EXT_DATA_ROOT / 'KITTI-360'

  @classmethod
  def filepath(cls, rpath):
    return cls.ROOT / rpath


###############################################################################
### IDatasetUtil Impl

class DSUtil(IDatasetUtil):

  FIXTURES = Fixtures

  REQUIRED_DIRS = (
    'calibration',
    'data_2d_raw',
    'data_3d_raw',
    'data_3d_semantics',
    'data_3d_bboxes',
    'data_poses',
  )

  OPTIONAL_DIRS = (
    'data_2d_semantics',
  )

  @classmethod
  def emplace(cls):
    DIRS_REQUIRED = set(cls.FIXTURES.filepath(d) for d in cls.REQUIRED_DIRS)
    has_all_req = all(p.exists() for p in DIRS_REQUIRED)
    if not has_all_req:
      req = '\n        '.join('  * %s' % fname for fname in cls.all_zips())
      opt = '\n        '.join('  * %s' % fname for fname in cls.all_zips())
      cls.show_md("""
        Due to KITTI-360 license constraints, you need to manually accept the
        KITTI-360 license and download the files at
        [the KITTI-360 website](http://www.cvlibs.net/datasets/kitti-360/download.php).
        
        The KITTI-360 team provides download scripts that will help unzip
        files into place.  The total dataset is about 650GB unzipped
        (spinning disk OK).

        Required KITTI-360 data dirs:

        %s

        Optioanl KITTI-360 data dirs:

        %s
        """ % (req, opt))
      
      kitti_root = input(
        "Please enter the directory containing your KITTI zip archives; "
        "PSegs will create a (read-only) symlink to them: ")
      kitti_root = Path(kitti_root.strip())
      assert kitti_root.exists()
      assert kitti_root.is_dir()

      from oarphpy import util as oputil
      oputil.mkdir(str(cls.FIXTURES.ROOT.parent))

      cls.show_md("Symlink: \n%s <- %s" % (kitti_root, cls.FIXTURES.ROOT))
      os.symlink(kitti_root, cls.FIXTURES.ROOT)

      # Make symlink read-only
      import stat
      os.chmod(
        kitti_root,
        stat.S_IREAD|stat.S_IRGRP|stat.S_IROTH,
        follow_symlinks=False)

  cls.show_md("Validating KITTI archives ...")
  dirs_needed = set(cls.all_zips())
  dirs_have = set()
  for entry in cls.FIXTURES.ROOT.iterdir():
    if entry.name in cls.REQUIRED_DIRS:
      dirs_needed.remove(entry.name)
      dirs_have.add(entry.name)
  
  if dirs_needed:
    s_have = \
      '\n        '.join('  * %s' % fname for fname in dirs_have)
    s_needed = \
      '\n        '.join('  * %s' % fname for fname in dirs_needed)
    cls.show_md("""
      Missing some expected data dirs!

      Found:
      
      %s

      Missing:

      %s
    """ % (s_have, s_needed))
    return False
  
  cls.show_md("... all KITTI-360 data found!")
  return True
