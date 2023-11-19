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

import os
import tempfile
from pathlib import Path

from oarphpy import util as oputil
from oarphpy.spark import SessionFactory

import psegs
from psegs import util
from psegs.spark import Spark
from psegs.cache import LocalDiskCache

# FIXME put in mount!
# PS_TEST_TEMPDIR_ROOT = os.path.join(tempfile.gettempdir(), 'psegs_test')
PS_TEST_TEMPDIR_ROOT = '/opt/psegs/psegs_test'


class LocalSpark(Spark):
  """A local Spark session; should result in only one session being created
  per testing run"""

  SRC_ROOT_MODULES = ['psegs', 'test']

def test_tempdir(name, clean=True):
  path = os.path.join(PS_TEST_TEMPDIR_ROOT, name)
  if clean:
    from oarphpy.util import cleandir
    cleandir(path)
  return Path(path)

def skip_if_fixture_absent(path):
  if not os.path.exists(path):
    import pytest
    pytest.skip("This test requires %s" % path)


def assert_img_directories_equal(actual_dir, expected_dir):
  util.log.info("Inspecting artifacts in %s ..." % expected_dir)
  for actual in oputil.all_files_recursive(actual_dir):
    actual = Path(actual)
    expected_dir = Path(expected_dir)
    expected = expected_dir / actual.name

    match = (open(actual, 'rb').read() == open(expected, 'rb').read())
    if not match:
      import imageio
      actual_img = imageio.imread(actual)
      expected_img = imageio.imread(expected)
      
      diff = expected_img - actual_img
      n_pixels = (diff != 0).sum() / 3
      diff_path = str(actual) + '.diff.png'
      imageio.imwrite(diff_path, diff)
      assert False, \
        "File mismatch \n%s != %s ,\n %s pixels different, diff: %s" % (
          actual, expected, n_pixels, diff_path)
  
  util.log.info("Good! %s == %s" % (actual_dir, expected_dir))


def check_sample_debug_images(sample, expected_dir, testname=''):
  outdir = test_tempdir(testname or sample.uri.segment_id)

  def save(path, img):
    import imageio
    imageio.imwrite(path, img)
    util.log.info("Saved %s" % path)

  cuboids = sample.cuboid_labels
  for pc in sample.lidar_clouds:
    path = outdir / ('%s_bev.png' % pc.sensor_name)
    save(path, pc.get_bev_debug_image(cuboids=cuboids))
    
    path = outdir / ('%s_rv.png' % pc.sensor_name)
    save(path, pc.get_front_rv_debug_image(cuboids=cuboids))

    path = outdir / ('%s_bev_painted.png' % pc.sensor_name)
    save(path, pc.get_bev_debug_image(
                    camera_images=sample.camera_images))
    
    path = outdir / ('%s_rv_painted.png' % pc.sensor_name)
    save(path, pc.get_front_rv_debug_image(
                    camera_images=sample.camera_images))

  for ci in sample.camera_images:
    path = outdir / ('%s_debug.png' % ci.sensor_name)
    save(
      path,
      ci.get_debug_image(
        clouds=sample.lidar_clouds,
        cuboids=cuboids))

  assert_img_directories_equal(outdir, expected_dir)


def check_stamped_datum_dfs_equal(
        spark,
        sd_df_actual,
        sd_df_expected_path='',
        testname='',
        sd_df_expected=None):
  
  from psegs.table.sd_table import StampedDatumTable
  from psegs.spark import save_df_thunks

  if not testname:
    seg_df = sd_df_actual.select('uri.segment_id').orderBy('uri.segment_id')
    seg_name = seg_df.first().segment_id
    testname = seg_name
  
  # Make tests faster and artifacts more compact
  sd_df_actual = sd_df_actual.repartition(5).persist()

  actual_path = test_tempdir(testname)
  util.log.info("Testing serialization of actual to %s ..." % actual_path)
  save_df_thunks(
        [lambda: sd_df_actual],
        spark_save_opts=dict(
          path=str(actual_path),
          format='parquet',
          partitionBy=StampedDatumTable.PARTITION_KEYS,
          compression='lz4',
        ))
  
  if sd_df_expected is None:
    util.log.info("Fetching expected from %s ..." % sd_df_expected_path)
    assert (
      sd_df_expected_path and 
        (not oputil.missing_or_empty(str(sd_df_expected_path))))
    sd_df_expected = spark.read.parquet(str(sd_df_expected_path))

  difftxt = StampedDatumTable.find_diff(sd_df_actual, sd_df_expected)
  assert difftxt == '', \
        "Non-zero diff!\nActual path %s\nExpected path %s\nDiff:\n%s" % (
          actual_path, sd_df_expected_path, difftxt)

def test_fixtures_dir():
  # Path to fixtures *included* with PSegs
  return Path(__file__).parent / 'fixtures'


class PSegsTestLocalDiskCache(LocalDiskCache):
  """A `LocalDiskCache` that uses a test-defined root dir"""

  TEST_ROOT = Path('/tmp')

  @classmethod
  def cache_cls_for_testroot(cls, testroot):
    class MyTestLocalDiskCache(cls):
      TEST_ROOT = testroot
    return MyTestLocalDiskCache

  def new_filepath(self, fname, t=None):
    dest = self.TEST_ROOT / 'psegs_local_disk_cache' / 'adhoc_files' / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest

  def new_dirpath(self, dirpath, t=None):
    dest = self.TEST_ROOT / 'psegs_local_disk_cache' / 'adhoc_dirs' / dirpath
    dest.mkdir(parents=True, exist_ok=True)
    return dest
