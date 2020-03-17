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

from psegs.datum.camera_image import CameraImage
from psegs.datum.cuboid import Cuboid
from psegs.datum.point_cloud import PointCloud
from psegs.datum.stamped_datum import StampedDatum
from psegs.datum.transform import Transform
from psegs.datum.uri import URI
from psegs.table.sd_table import StampedDatumTableBase

import test.testutil as testutil


class TestStampedDatumTableBase(StampedDatumTableBase):
  """Create a clean temp directory for each test table"""

  @classmethod
  def table_root(cls):
    if not hasattr(cls, '_table_tempdir'):
      cls._table_tempdir = testutil.test_tempdir('sd_test_' + cls.__name__)
    return cls._table_tempdir


def test_sd_table_simple():
  test_datums = [
    StampedDatum(
      uri=URI(
        dataset='d',
        split='s',
        segment_id='segment'),
      transform=Transform()),
  ]

  class Simple(TestStampedDatumTableBase):
    @classmethod
    def _create_datum_rdds(
          cls, spark, existing_uri_df=None, only_segments=None):
      return [spark.sparkContext.parallelize(test_datums)]
    
  with testutil.LocalSpark.sess() as spark:
    Simple.build(spark)
    df = Simple.as_df(spark)
    assert df.count() == 1
    assert df.filter((df.uri.dataset=='d')).count() == 1

    datum_rdd = Simple.as_datum_rdd(spark)
    datums = datum_rdd.collect()
    assert len(datums) == 1
    assert datums[0] == test_datums[0]


def test_sd_table_one_of_every():
  BASE_URI = URI(dataset='d', split='s', segment_id='seg')
  test_datums = [
    StampedDatum(
      uri=BASE_URI.replaced(topic='camera|front', timestamp=1),
      camera_image=CameraImage()),
    StampedDatum(
      uri=BASE_URI.replaced(topic='labels|cuboids', timestamp=1),
      cuboids=[Cuboid()]),
    StampedDatum(
      uri=BASE_URI.replaced(topic='lidar|front', timestamp=1),
      point_cloud=PointCloud()),
    StampedDatum(
      uri=BASE_URI.replaced(topic='ego_pose', timestamp=1),
      transform=Transform()),
  ]

  class OneOfEvery(TestStampedDatumTableBase):
    @classmethod
    def _create_datum_rdds(
          cls, spark, existing_uri_df=None, only_segments=None):
      return [spark.sparkContext.parallelize(test_datums)]
    
  with testutil.LocalSpark.sess() as spark:
    OneOfEvery.build(spark)
    df = OneOfEvery.as_df(spark)
    assert df.count() == len(test_datums)
    
    # Let's do a basic query
    TOPICS = [datum.uri.topic for datum in test_datums]
    assert (
      sorted(TOPICS) ==
      sorted(r.topic for r in df.select('uri.topic').collect()))

    datum_rdd = OneOfEvery.as_datum_rdd(spark)
    datums = datum_rdd.collect()
    assert sorted(datums) == sorted(test_datums)
