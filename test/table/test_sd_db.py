# Copyright 2021 Maintainers of PSegs
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

import pytest

from psegs.datum.camera_image import CameraImage
from psegs.datum.cuboid import Cuboid
from psegs.datum.point_cloud import PointCloud
from psegs.datum.stamped_datum import StampedDatum
from psegs.datum.transform import Transform
from psegs.datum.uri import URI
from psegs.table.sd_table import StampedDatumTableBase

import test.testutil as testutil



from psegs.table.sd_db import NoKnownTable
from psegs.table.sd_db import StampedDatumDB
from psegs.table.sd_db import to_seg_uri_str

def test_seg_to_uri_str():
  def _check(actual, expected):
    assert to_seg_uri_str(actual) == expected
  
  _check(URI(), 'psegs://')
  _check('psegs://', 'psegs://')
  _check(URI(dataset='a'), 'psegs://dataset=a')
  _check('psegs://dataset=a', 'psegs://dataset=a')
  _check('psegs://dataset=a&topic=b', 'psegs://dataset=a')

  from pyspark import Row
  _check(Row(moof=1), 'psegs://')
  _check(Row(dataset='a'), 'psegs://dataset=a')
  _check(Row(dataset='a', topic='b'), 'psegs://dataset=a')

  with pytest.raises(Exception):
    to_seg_uri_str('')
    to_seg_uri_str(object())


class TestTableBase(StampedDatumTableBase):
  TEST_DATUMS = []

  @classmethod
  def _create_datum_rdds(
        cls, spark, existing_uri_df=None, only_segments=None):
    if only_segments:
      datums = []
      for suri in only_segments:
        datums += [
          sd for sd in cls.TEST_DATUMS
          if suri.soft_matches_segment_of(sd.uri)
        ]
    else:
      datums = cls.TEST_DATUMS
    return [spark.sparkContext.parallelize(datums)]

  @classmethod
  def _get_all_segment_uris(cls):
    suris_strs = set(str(sd.uri.to_segment_uri()) for sd in cls.TEST_DATUMS)
    return sorted(suris_strs)

class T1(TestTableBase):
  BASE_URI = URI(dataset='t1', split='s')
  TEST_DATUMS = [
    StampedDatum(
      uri=BASE_URI.replaced(
        segment_id='segt1.1', topic='c1', timestamp=1),
      camera_image=CameraImage()),

    StampedDatum(
      uri=BASE_URI.replaced(
        segment_id='segt1.2', topic='c', timestamp=1),
      camera_image=CameraImage()),
    StampedDatum(
      uri=BASE_URI.replaced(
        segment_id='segt1.2', topic='c', timestamp=2),
      camera_image=CameraImage()),
  ]

class T2(TestTableBase):
  BASE_URI = URI(dataset='t2', split='s')
  TEST_DATUMS = [
    StampedDatum(
      uri=BASE_URI.replaced(
        segment_id='segt2.1', topic='c1', timestamp=1),
      camera_image=CameraImage()),
    StampedDatum(
      uri=BASE_URI.replaced(
        segment_id='segt2.2', topic='c1', timestamp=1),
      camera_image=CameraImage()),
  ]

def _create_db_simple(spark=None):
  spark = spark or testutil.LocalSpark.getOrCreate()
  db = StampedDatumDB([T1, T2], spark=spark)
  return db



def test_db_get_sample():
  def _check_datums(sample, expected_tt):
    actual_tt = [(sd.uri.topic, sd.uri.timestamp) for sd in sample.datums]
    assert sorted(actual_tt) == sorted(expected_tt)

  db = _create_db_simple()
  
  uri = 'psegs://dataset=t1&segment_id=segt1.1&sel_datums=c1,1'
  sample = db.get_sample(uri)
  assert sample.uri == URI.from_str(uri)
  _check_datums(sample, [('c1', 1)])

  uri = 'psegs://dataset=t1&segment_id=segt1.2&sel_datums=c,2'
  sample = db.get_sample(uri)
  assert sample.uri == URI.from_str(uri)
  _check_datums(sample, [('c', 2)])
  
  uri = 'psegs://dataset=t1&segment_id=segt1.2&sel_datums=c,2,c,1'
  sample = db.get_sample(uri)
  assert sample.uri == URI.from_str(uri)
  _check_datums(sample, [('c', 2), ('c', 1)])

  uri = 'psegs://segment_id=segt1.2'
  sample = db.get_sample(uri)
  assert URI.from_str(uri).soft_matches_segment_of(sample.uri)
  _check_datums(sample, [('c', 1), ('c', 2)])

  with pytest.raises(NoKnownTable):
    uri = 'psegs://dataset=no-existe&segment_id=segt1.2'
    sample = db.get_sample(uri)



def _get_actual_uris(datum_df):
    return [
      r.uri for r in datum_df.select('uri').rdd.map(T1.from_row).collect()
    ]

def test_db_get_datum_df_uri_list():
  db = _create_db_simple()

  def _get_actual_uris(datum_df):
    return [
      r.uri for r in datum_df.select('uri').rdd.map(T1.from_row).collect()
    ]

  uris_exist = [
    URI(dataset='t1', split='s', segment_id='segt1.2', timestamp=1, topic='c'),
    URI(dataset='t1', split='s', segment_id='segt1.2', timestamp=2, topic='c'),
    URI(dataset='t2', split='s', segment_id='segt2.2', timestamp=1, topic='c1'),
  ]
  datum_df = db.get_datum_df(uris=uris_exist)
  actual_uris = _get_actual_uris(datum_df)
  assert sorted(uris_exist) == sorted(actual_uris)

  uris_no_exist = [
    URI(dataset='no-exist', segment_id='no-exist', timestamp=1, topic='c1')
  ]
  with pytest.raises(NoKnownTable):
    db.get_datum_df(uris=uris_no_exist)



def test_db_get_datum_df_uri_rdd():
  uris_exist = [
    URI(dataset='t1', split='s', segment_id='segt1.2', timestamp=1, topic='c'),
    URI(dataset='t1', split='s', segment_id='segt1.2', timestamp=2, topic='c'),
    URI(dataset='t2', split='s', segment_id='segt2.2', timestamp=1, topic='c1'),
  ]
  uris_no_exist = [
    URI(dataset='no-exist', segment_id='no-exist', timestamp=1, topic='c1')
  ]
  with testutil.LocalSpark.sess() as spark:
    db = _create_db_simple(spark=spark)

    uri_rdd = spark.sparkContext.parallelize(uris_exist + uris_no_exist)

    datum_df = db.get_datum_df(uris=uri_rdd)
    actual_uris = _get_actual_uris(datum_df)
    assert sorted(uris_exist) == sorted(actual_uris)
    

def test_db_get_datum_df_uri_df():
  uris_exist = [
    URI(dataset='t1', split='s', segment_id='segt1.2', timestamp=1, topic='c'),
    URI(dataset='t1', split='s', segment_id='segt1.2', timestamp=2, topic='c'),
    URI(dataset='t2', split='s', segment_id='segt2.2', timestamp=1, topic='c1'),
  ]
  uris_no_exist = [
    URI(dataset='no-exist', segment_id='no-exist', timestamp=1, topic='c1')
  ]
  with testutil.LocalSpark.sess() as spark:
    db = _create_db_simple(spark=spark)

    from oarphpy.spark import RowAdapter
    uris = uris_exist + uris_no_exist
    schema = RowAdapter.to_schema(URI())
    uri_df = spark.createDataFrame(
              [RowAdapter.to_row(u) for u in uris], schema=schema)
    datum_df = db.get_datum_df(uris=uri_df)
    actual_uris = _get_actual_uris(datum_df)
    assert sorted(uris_exist) == sorted(actual_uris)


















class TestStampedDatumDBBase(StampedDatumTableBase):
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
