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

import copy
from pathlib import Path

from psegs import datum
from psegs import util
from psegs.conf import C
from psegs.datasets.idsutil import IDatasetUtil
from psegs.table.sd_table_factory import StampedDatumTableFactory


###############################################################################
### TanksAndTemples Fixtures & Other Constants

class Fixtures(object):

  ROOT = C.EXT_DATA_ROOT / 'tanks_and_temples_archives'

  EXTERNAL_FIXTURES_ROOT = C.EXTERNAL_TEST_FIXTURES_ROOT / 'tanks_and_temples'

  TRAINING_SCENES = (
    'Barn',
    'Caterpillar',
    'Church',
    'Courthouse',
    'Ignatius',
    'Meetingroom',
    'Truck',
  )

  TRAINING_DATA_MASTER_ZIP = 'trainingdata.zip'

  @classmethod
  def zip_path(cls, scene):
    return cls.ROOT / f'{scene}.zip'

  ### DSUtil Auto-download ####################################################

  @classmethod
  def maybe_emplace_psegs_kitti_ext(cls):
    print('todo')
    return

###############################################################################
### StampedDatum Table

class TanksAndTemplesSDTable(StampedDatumTableFactory):
  
  FIXTURES = Fixtures

  ## Subclass API

  @classmethod
  def _get_all_segment_uris(cls):
    train_segs = [
      datum.URI(
            dataset='tanks-and-temples',
            split='train',
            segment_id=scene,
            extra={
              'tnt.scene': scene,
            })
      for scene in cls.FIXTURES.TRAINING_SCENES
    ]
    return sorted(train_segs)

  @classmethod
  def _create_datum_rdds(cls, spark, existing_uri_df=None, only_segments=None):

    ## First build indices (saves several minutes per worker per chunk) ...
    class SDBenchmarkToRawMapper(BenchmarkToRawMapper):
      FIXTURES = cls.FIXTURES
    SDBenchmarkToRawMapper.setup(spark=spark)

    ## ... now build a set of tasks to do ...
    archive_paths = cls._get_all_archive_paths()
    task_rdd = _rdd_of_all_archive_datafiles(spark, archive_paths)
    task_rdd = task_rdd.cache()
    util.log.info("Discovered %s tasks ..." % task_rdd.count())
    
    ## ... convert to URIs and filter those tasks if necessary ...
    if existing_uri_df is not None:
      # Since we keep track of the original archives and file names, we can
      # just filter on those.  We'll collect them in this process b/c the
      # maximal set of URIs is smaller than RAM.
      def to_task(row):
        return (row.extra.get('kitti.archive'),
                row.extra.get('kitti.archive.file'))
      skip_tasks = set(
        existing_uri_df.select('extra').rdd.map(to_task).collect())
      
      task_rdd = task_rdd.filter(lambda t: t not in skip_tasks)
      util.log.info(
        "Resume mode: have datums for %s datums; dropped %s tasks" % (
          existing_uri_df.count(), len(skip_tasks)))
    
    uri_rdd = task_rdd.map(lambda task: kitti_archive_file_to_uri(*task))
    if only_segments:
      util.log.info(
        "Filtering to only %s segments" % len(only_segments))
      uri_rdd = uri_rdd.filter(
        lambda uri: any(
          suri.soft_matches_segment(uri) for suri in only_segments))

    ## ... run tasks and create stamped datums.
    # from oarphpy.spark import cluster_cpu_count
    URIS_PER_CHUNK = os.cpu_count() * 64 # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ make class member so can configure to RAM
    uris = uri_rdd.collect()
    util.log.info("... creating datums for %s URIs." % len(uris))

    datum_rdds = []
    for chunk in oputil.ichunked(uris, URIS_PER_CHUNK):
      chunk_uri_rdd = spark.sparkContext.parallelize(chunk)
      datum_rdd = chunk_uri_rdd.flatMap(cls._iter_datums_from_uri)
      datum_rdds.append(datum_rdd)
      # if len(datum_rdds) >= 10:
      #   break # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return datum_rdds
  
  @classmethod
  def _get_all_archive_paths(cls):
    archives = []
    if cls.INCLUDE_OBJECT_BENCHMARK:
      archives += list(cls.FIXTURES.OBJECT_BENCHMARK_FNAMES)
      if not cls.INCLUDE_OBJ_PREV_FRAMES:
        archives = [arch for arch in archives if 'prev' not in arch]
    if cls.INCLUDE_TRACKING_BENCHMARK:
      archives += list(cls.FIXTURES.TRACKING_BENCHMARK_FNAMES)
    archives = [arch for arch in archives if 'calib' not in arch]
    paths = [cls.FIXTURES.zip_path(arch) for arch in archives]
    return paths


  ## Datum Construction Support

  @classmethod
  def _get_uris_for_segment_uri(cls, seg_uri):
    import zipfile

    archive_path = cls.FIXTURES.zip_path(seg_uri.extra['tnt.scene'])
    names = zipfile.ZipFile(archive_path).namelist()
    base_uri = seg_uri
    uris = [
      datum.URI
    ]
    for name in sorted(names):
      assert False, "TODO"


  @classmethod
  def _get_file_bytes(cls, uri=None, archive=None, entryname=None):
    """Read bytes for the file referred to by `uri`"""

    if uri is not None:
      archive = uri.extra['kitti.archive']
      entryname = uri.extra['kitti.archive.file']
    assert archive and entryname

    # Cache the Zipfiles for faster loading
    if not hasattr(cls, '_get_file_bytes_archives'):
      cls._get_file_bytes_archives = {}
    if archive not in cls._get_file_bytes_archives:
      import zipfile
      path = cls.FIXTURES.zip_path(archive)
      cls._get_file_bytes_archives[archive] = zipfile.ZipFile(path)
      
    
    try:
      return cls._get_file_bytes_archives[archive].read(entryname)
    except Exception as e:
        raise Exception((e, archive, uri))

###############################################################################
### IDatasetUtil Impl

class DSUtil(IDatasetUtil):

  FIXTURES = Fixtures

  @classmethod
  def all_training_zips(cls):
    return [cls.FIXTURES.TRAINING_DATA_MASTER_ZIP] + [
      f'{scene}.zip' for scene in cls.FIXTURES.TRAINING_SCENES
    ]

  @classmethod
  def emplace(cls):
    import os
    from pathlib import Path

    cls.FIXTURES.maybe_emplace_psegs_kitti_ext()

    if not cls.FIXTURES.ROOT.exists():
      cls.show_md("""
        The Tanks And Temples data files are offered via Google Drive links
        at [https://tanksandtemples.org/download/](https://tanksandtemples.org/download/).
        You must be signed-in with your own Google account in order to download
        these files.  
        
        The authors supply a download script, however this might only work
        for you if you have a non-headless and authenticated terminal session:
          https://github.com/isl-org/TanksAndTemples/blob/3c2c2125e9b16f32790c96a8953611de785d91d6/python_toolbox/download_t2_dataset.py#L1
        
        We recommend you download the data manually.  Please go to the download
        page and download at least the following:
          * `trainingdata.zip`-- The link for this file is embedded in a
             text comment on the page.  Direct link:
               https://drive.google.com/file/d/1jAr3IDvhVmmYeDWi0D_JfgiHcl70rzVE
          * For each Training Data scene, download the "image set" archive;
             here is a direct link for the Barn sequence:
               https://drive.google.com/file/d/0B-ePgl6HF260NzQySklGdXZyQzA/
        
        You'll want to download all the following zip files (do not decompress
        them) to a single directory on a local disk (spinning disk OK).
        Once you've downloaded the archives, we'll need the path to where
        you put them.  Enter that below, or exit this program.

      """)
      tnt_root = input(
        "Please enter the directory containing your TanksAndTemples zip archives; "
        "PSegs will create a (read-only) symlink to them: ")
      tnt_root = Path(tnt_root.strip())
      assert tnt_root.exists()
      assert tnt_root.is_dir()

      cls.FIXTURES.ROOT.parent.mkdir(parents=True, exist_ok=True)

      cls.show_md("Symlink: \n%s <- %s" % (tnt_root, cls.FIXTURES.ROOT))
      os.symlink(tnt_root, cls.FIXTURES.ROOT)

      # Make symlink read-only
      import stat
      os.chmod(
        tnt_root,
        stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH,
        follow_symlinks=False)

    cls.show_md("Validating TanksAndTemples archives ...")
    zips_needed = set(cls.all_training_zips())
    zips_have = set()
    for entry in cls.FIXTURES.ROOT.iterdir():
      if entry.name in zips_needed:
        zips_needed.remove(entry.name)
        zips_have.add(entry.name)
    
    if zips_needed:
      s_have = \
        '\n        '.join('  * %s' % fname for fname in zips_have)
      s_needed = \
        '\n        '.join('  * %s' % fname for fname in zips_needed)
      cls.show_md("""
        Missing some expected archives!

        Found:
        
        %s

        Missing:

        %s
      """ % (s_have, s_needed))
      return False
    
    cls.show_md("... all Tanks and Temples archives found!")
    return True

  @classmethod
  def test(cls):
    from oarphpy import util as oputil
    oputil.run_cmd("cd %s && pytest -s -vvv -k test_tanks_and_temples" % C.PS_ROOT)
    return True

  # @classmethod
  # def build_table(cls):
  #   TanksAndTemplesSDTable.build()
  #   return True
