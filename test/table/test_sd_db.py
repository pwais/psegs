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
      camera_image=CameraImage(sensor_name='c1', timestamp=1)),
    StampedDatum(
      uri=BASE_URI.replaced(
        segment_id='segt1.1', topic='l1', timestamp=1),
      point_cloud=PointCloud(sensor_name='l1', timestamp=1)),

    StampedDatum(
      uri=BASE_URI.replaced(
        segment_id='segt1.2', topic='c', timestamp=1),
      camera_image=CameraImage(sensor_name='c', timestamp=1)),
    StampedDatum(
      uri=BASE_URI.replaced(
        segment_id='segt1.2', topic='c', timestamp=2),
      camera_image=CameraImage(sensor_name='c', timestamp=2)),
  ]

class T2(TestTableBase):
  BASE_URI = URI(dataset='t2', split='s')
  TEST_DATUMS = [
    StampedDatum(
      uri=BASE_URI.replaced(
        segment_id='segt2.1', topic='c1', timestamp=1),
      camera_image=CameraImage(sensor_name='c1', timestamp=1)),
    StampedDatum(
      uri=BASE_URI.replaced(
        segment_id='segt2.2', topic='c1', timestamp=1),
      camera_image=CameraImage(sensor_name='c1', timestamp=1)),
  ]

class T3(TestTableBase):
  TEST_DATUMS = ([
      StampedDatum(
        uri=URI(dataset='t3', split='s',
          segment_id='segt3.1', topic='c1', timestamp=t+1),
        camera_image=CameraImage(sensor_name='c1', timestamp=t+1))
      for t in range(10)
    ] + [ 
      StampedDatum(
        uri=URI(dataset='t3', split='s',
          segment_id='segt3.2', topic='c1', timestamp=t+1),
        camera_image=CameraImage(sensor_name='c1', timestamp=t+1))
      for t in range(20)
    ])

def _create_db_simple(spark=None):
  spark = spark or testutil.LocalSpark.getOrCreate()
  db = StampedDatumDB([T1, T2, T3], spark=spark)
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


def test_db_get_keyed_sample_df():
  key_uris_exist = [
    ('k-span-segs',
      URI(dataset='t1', split='s', segment_id='segt1.2', timestamp=1, topic='c')),
    ('k-span-segs',
      URI(dataset='t2', split='s', segment_id='segt2.2', timestamp=1, topic='c1')),

    ('solo-seg-datum',
      URI(dataset='t1', split='s', segment_id='segt1.2', timestamp=2, topic='c')),
    
    ('two-topic',
      URI(dataset='t1', split='s', segment_id='segt1.1', timestamp=1, topic='c1')),
    ('two-topic',
      URI(dataset='t1', split='s', segment_id='segt1.1', timestamp=1, topic='l1')),
  ]
  key_uris_exist += [
    ('many',
      URI(
        dataset='t3',
        split='s',
        segment_id='segt3.2',
        timestamp=t+1,
        topic='c1'))
    for t in range(20)
  ]
  uris_no_exist = [
    ('solo-seg-datum',
      URI(dataset='no-exist', segment_id='no-exist', timestamp=1, topic='c1')),
  ]
  uris_no_exist += [
    ('no-exist-many',
      URI(
        dataset='t3',
        split='s',
        segment_id='segt3.1',
        timestamp=t+1,
        topic='c1'))
    for t in range(30, 100)
  ]
  with testutil.LocalSpark.sess() as spark:
    db = _create_db_simple(spark=spark)

    rows = [{'key': k, 'uri': u} for k, u in key_uris_exist]
    rows += [{'key': k, 'uri': u} for k, u in uris_no_exist]
    from oarphpy.spark import RowAdapter
    rows = [RowAdapter.to_row(r) for r in rows]
    schema = RowAdapter.to_schema({'key': 's', 'uri': URI()})
    df = spark.createDataFrame(rows, schema=schema)

    key_sample_df = db.get_keyed_sample_df(df)

    expected_key_to_uris_exist = {}
    for k, u in key_uris_exist:
      expected_key_to_uris_exist.setdefault(k, [])
      expected_key_to_uris_exist[k].append(u)
    
    key_sample_df = key_sample_df.persist()

    actual_keys = set(r.key for r in key_sample_df.select('key').collect())
    assert actual_keys == set(expected_key_to_uris_exist.keys())

    for key, expected_uris in expected_key_to_uris_exist.items():
      row_df = key_sample_df.filter(key_sample_df.key == key)

      datum_rows = row_df.collect()[0].asDict()['datums']
      assert len(datum_rows) == len(expected_uris)

      datums = [RowAdapter.from_row(rr) for rr in datum_rows]
      assert sorted(d.uri for d in datums) == sorted(expected_uris)
    
      samp = db.datum_rows_to_sample(datum_rows)
      assert len(samp.uri.sel_datums) == len(expected_uris)
      assert len(samp.datums) == len(expected_uris)
