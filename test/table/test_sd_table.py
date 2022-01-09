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


###############################################################################
## Diff Tests

def create_sd_table_and_df(spark, datums):

  class DiffTable(TestStampedDatumTableBase):
    @classmethod
    def _create_datum_rdds(
          cls, spark, existing_uri_df=None, only_segments=None):
      return [spark.sparkContext.parallelize(datums)]
  
  df = DiffTable.as_df(spark, force_compute=True)

  # Make tests faster; default number of partitions is 
  # usually larger than number of rows
  df = df.repartition(10).cache()

  return DiffTable, df


def test_sd_table_diff_empty():
  with testutil.LocalSpark.sess() as spark:
    _, df1 = create_sd_table_and_df(spark, [])
    _, df2 = create_sd_table_and_df(spark, [])
    
    difftxt = StampedDatumTableBase.find_diff(df1, df2)
    assert difftxt == ''

    
def test_sd_table_diff_identical():
  with testutil.LocalSpark.sess() as spark:
    BASE_URI = URI(dataset='d', split='s', segment_id='seg')
    one_of_every_datum = [
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
    _, df1 = create_sd_table_and_df(spark, one_of_every_datum)
    _, df2 = create_sd_table_and_df(spark, one_of_every_datum)
    
    difftxt = StampedDatumTableBase.find_diff(df1, df2)
    assert difftxt == ''


def test_sd_table_diff_mismatch_dataset():
  with testutil.LocalSpark.sess() as spark:
    BASE_URI = URI(dataset='d', split='s', segment_id='seg')
    t1 = [
      StampedDatum(
        uri=BASE_URI.replaced(dataset='d1', topic='camera|front', timestamp=t),
        camera_image=CameraImage())
      for t in range(10)
    ]
    t2 = [
      StampedDatum(
        uri=BASE_URI.replaced(dataset='d2', topic='camera|front', timestamp=t),
        camera_image=CameraImage())
      for t in range(10)
    ]
    _, df1 = create_sd_table_and_df(spark, t1)
    _, df2 = create_sd_table_and_df(spark, t2)

    difftxt = StampedDatumTableBase.find_diff(df1, df2)
    assert "Dataset/Split Mismatch" in difftxt
    assert "- [('d1', 's')]" in difftxt
    assert "+ [('d2', 's')]" in difftxt


def test_sd_table_diff_mismatch_segments():
  with testutil.LocalSpark.sess() as spark:
    BASE_URI = URI(dataset='d', split='s', segment_id='seg')
    t1 = [
      StampedDatum(
        uri=BASE_URI.replaced(segment_id='seg1', topic='camera|front', timestamp=t),
        camera_image=CameraImage())
      for t in range(10)
    ]
    t2 = [
      StampedDatum(
        uri=BASE_URI.replaced(segment_id='seg2', topic='camera|front', timestamp=t),
        camera_image=CameraImage())
      for t in range(10)
    ]
    _, df1 = create_sd_table_and_df(spark, t1)
    _, df2 = create_sd_table_and_df(spark, t2)

    difftxt = StampedDatumTableBase.find_diff(df1, df2)
    assert "Segment Mismatch" in difftxt
    assert "- [('d', 's', 'seg1')]" in difftxt
    assert "+ [('d', 's', 'seg2')]" in difftxt


def test_sd_table_diff_mismatch_uri_count_many():
  with testutil.LocalSpark.sess() as spark:
    BASE_URI = URI(dataset='d', split='s', segment_id='seg')
    t1 = [
      StampedDatum(
        uri=BASE_URI.replaced(topic='camera|front', timestamp=t),
        camera_image=CameraImage())
      for t in range(1000)
    ]
    t2 = [
      StampedDatum(
        uri=BASE_URI.replaced(topic='camera|front', timestamp=t),
        camera_image=CameraImage())
      for t in range(2000)
    ]
    _, df1 = create_sd_table_and_df(spark, t1)
    _, df2 = create_sd_table_and_df(spark, t2)

    difftxt = StampedDatumTableBase.find_diff(df1, df2)
    assert "URI Count Mismatch" in difftxt
    assert "left count: 1000" in difftxt
    assert "right count: 2000" in difftxt


def test_sd_table_diff_mismatch_uri_content():
  with testutil.LocalSpark.sess() as spark:
    BASE_URI = URI(dataset='d', split='s', segment_id='seg')
    t1 = [
      StampedDatum(
        uri=BASE_URI.replaced(topic='camera|front', timestamp=t),
        camera_image=CameraImage())
      for t in range(10)
    ]
    t2 = [
      StampedDatum(
        uri=BASE_URI.replaced(topic='camera|rear', timestamp=t),
        camera_image=CameraImage())
      for t in range(10)
    ]
    _, df1 = create_sd_table_and_df(spark, t1)
    _, df2 = create_sd_table_and_df(spark, t2)

    difftxt = StampedDatumTableBase.find_diff(df1, df2)
    assert "Missing URIs" in difftxt
    assert "Missing left (10): ['psegs://dataset=d&split=s&segment_id=seg&timestamp=1&topic=camera|rear'" in difftxt
    assert "Missing right (10): ['psegs://dataset=d&split=s&segment_id=seg&timestamp=1&topic=camera|front'" in difftxt


def test_sd_table_diff_mismatch_uri_dupes():
  with testutil.LocalSpark.sess() as spark:
    BASE_URI = URI(dataset='d', split='s', segment_id='seg')
    one_of_every_datum = [
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
    _, df1 = create_sd_table_and_df(spark, one_of_every_datum)
    _, df2 = create_sd_table_and_df(
                spark, one_of_every_datum + one_of_every_datum)
    
    difftxt = StampedDatumTableBase.find_diff(df1, df2)
    assert "Dupe URIs" in difftxt
    assert "Dupes left (0): []" in difftxt
    assert "Dupes right (4): " in difftxt
    assert "psegs://dataset=d&split=s&segment_id=seg&timestamp=1&topic=ego_pose', 2" in difftxt


def test_sd_table_diff_mismatch_sd_content():
  with testutil.LocalSpark.sess() as spark:
    BASE_URI = URI(dataset='d', split='s', segment_id='seg')
    t1 = [
      StampedDatum(
        uri=BASE_URI.replaced(topic='camera|front', timestamp=1),
        camera_image=CameraImage(sensor_name='c1')),
      StampedDatum(
        uri=BASE_URI.replaced(topic='labels|cuboids', timestamp=1),
        cuboids=[Cuboid(track_id='track_id1')]),
      StampedDatum(
        uri=BASE_URI.replaced(topic='lidar|front', timestamp=1),
        point_cloud=PointCloud(sensor_name='p1')),
      StampedDatum(
        uri=BASE_URI.replaced(topic='ego_pose', timestamp=1),
        transform=Transform(src_frame='src_frame1')),
    ]
    t2 = [
      StampedDatum(
        uri=BASE_URI.replaced(topic='camera|front', timestamp=1),
        camera_image=CameraImage(sensor_name='c2')),
      StampedDatum(
        uri=BASE_URI.replaced(topic='labels|cuboids', timestamp=1),
        cuboids=[Cuboid(track_id='track_id2')]),
      StampedDatum(
        uri=BASE_URI.replaced(topic='lidar|front', timestamp=1),
        point_cloud=PointCloud(sensor_name='p2')),
      StampedDatum(
        uri=BASE_URI.replaced(topic='ego_pose', timestamp=1),
        transform=Transform(src_frame='src_frame2')),
    ]
    _, df1 = create_sd_table_and_df(spark, t1)
    _, df2 = create_sd_table_and_df(spark, t2)
    
    difftxt = StampedDatumTableBase.find_diff(df1, df2)

    assert "Datum mismatch" in difftxt
    
    assert "-                   'sensor_name': 'c1'" in difftxt
    assert "+                   'sensor_name': 'c2'" in difftxt

    assert "-               'track_id': 'track_id1'" in difftxt
    assert "+               'track_id': 'track_id2'" in difftxt
    
    assert "-                  'sensor_name': 'p1'" in difftxt
    assert "+                  'sensor_name': 'p2'" in difftxt

    assert "-                'src_frame': 'src_frame1'" in difftxt
    assert "+                'src_frame': 'src_frame2'" in difftxt

