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

import copy

from oarphpy.spark import RowAdapter
from pyspark.sql import Row

from psegs import util
from psegs.conf import C
from psegs.datum import URI
from psegs.datum.stamped_datum import Sample
from psegs.datum.stamped_datum import STAMPED_DATUM_PROTO
from psegs.spark import Spark


class StampedDatumTableBase(object):

  ## Public API

  PARTITION_KEYS = ('dataset', 'split', 'segment_id')

  @classmethod
  def table_root(cls):
    return C.SD_TABLE_ROOT / 'stamped_datums'
  
  @classmethod
  def get_all_segment_uris(cls):
    return [URI.from_str(u) for u in cls._get_all_segment_uris()]

  @classmethod
  def build(cls, spark=None, only_segments=None):
    with Spark.sess(spark) as spark:
      existing_uri_df = None
      if not util.missing_or_empty(cls.table_root()):
        existing_uri_df = cls.as_uri_df(spark)
      sd_rdds = cls._create_datum_rdds(
                        spark, 
                        existing_uri_df=existing_uri_df,
                        only_segments=only_segments)
      class StampedDatumDFThunk(object):
        def __init__(self, sd_rdd):
          self.sd_rdd = sd_rdd
        def __call__(self):
          return cls._sd_rdd_to_sd_df(spark, self.sd_rdd)
      df_thunks = [StampedDatumDFThunk(sd_rdd) for sd_rdd in sd_rdds]
      Spark.save_df_thunks(
        df_thunks,
        path=str(cls.table_root()),
        format='parquet',
        partitionBy=cls.PARTITION_KEYS,
        compression='lz4')

  @classmethod
  def as_df(cls, spark):
    # TODO REPLACE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """ comments ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    util.log.info("Loading %s ..." % cls.table_root())
    df = spark.read.option("mergeSchema", "true").parquet(str(cls.table_root()))
    # df = spark.read.schema(cls.table_schema()).option("mergeSchema", "true").load(path=cls.table_root())
    # read = read.schema(
    # df = spark.read.parquet(cls.table_root(), schema=cls.table_schema())
    return df

  @classmethod
  def as_datum_rdd(cls, spark, df=None):
    df = df or cls.as_df(spark)
    return df.rdd.map(StampedDatumTableBase.from_row)

  # @classmethod
  # def get_segment_datum_rdd(cls, spark, segment_uri, time_ordered=True):
  #   if util.missing_or_empty(cls.table_root()):
  #     datum_rdds = cls._create_datum_rdds(spark, only_segments=[segment_uri])
  #     if not datum_rdds:
  #       return spark.sparkContext.parallelize([])
  #     datum_rdd = spark.sparkContext.union(datum_rdds)
  #     from pyspark import StorageLevel
  #     datum_rdd = datum_rdd.persist(StorageLevel.MEMORY_AND_DISK)
  #     if time_ordered:
  #       datum_rdd = datum_rdd.sortBy(lambda sd: sd.uri.timestamp)
  #     return datum_rdd
  #   else:
  #     df = cls.as_df(spark)
  #     assert segment_uri.segment_id
  #     seg_df = df.filter(df.segment_id == segment_uri.segment_id)
  #     if segment_uri.dataset:
  #       seg_df = seg_df.filter(df.dataset == segment_uri.dataset)
  #     if segment_uri.split:
  #       seg_df = seg_df.filter(df.split == segment_uri.split)

  #     seg_df = seg_df.persist()
  #     if time_ordered:
  #       seg_df = seg_df.orderBy('uri.timestamp')
  #     return seg_df.rdd.map(StampedDatumTableBase.from_row)

  @classmethod
  def _get_segment_datum_rdd_or_df(cls, spark, segment_uri):
    segment_uri = URI.from_str(segment_uri)
    if util.missing_or_empty(cls.table_root()):
      datum_rdds = cls._create_datum_rdds(spark, only_segments=[segment_uri])
      if not datum_rdds:
        return spark.sparkContext.parallelize([])
      datum_rdd = spark.sparkContext.union(datum_rdds)
      if segment_uri.sel_datums:
        selected = set(
          (sd.topic, sd.timestamp) for sd in segment_uri.sel_datums)
        datum_rdd = datum_rdd.filter(
          lambda sd: (sd.uri.topic, sd.uri.timestamp) in selected)
      return datum_rdd
    else:
      df = cls.as_df(spark)
      assert segment_uri.segment_id, "Bad URI %s" % segment_uri
      seg_df = df.filter(df.segment_id == segment_uri.segment_id)
      if segment_uri.dataset:
        seg_df = seg_df.filter(df.dataset == segment_uri.dataset)
      if segment_uri.split:
        seg_df = seg_df.filter(df.split == segment_uri.split)
      if segment_uri.sel_datums:
        import pyspark.sql.functions as F
        from functools import reduce
        seg_df = seg_df.where(
          reduce(
            lambda a, b: a | b,
            ((F.col('uri.topic') == sd.topic) & 
              (F.col('uri.timestamp') == sd.timestamp)
            for sd in segment_uri.sel_datums)))
      return seg_df
    
  @classmethod
  def get_segment_datum_rdd(cls, spark, segment_uri):
    rdd_or_df = cls._get_segment_datum_rdd_or_df(spark, segment_uri)
    if hasattr(rdd_or_df, 'rdd'):
      return cls.sd_df_to_rdd(rdd_or_df)
    else:
      return rdd_or_df
  
  @classmethod
  def get_segment_datum_df(cls, spark, segment_uri):
    rdd_or_df = cls._get_segment_datum_rdd_or_df(spark, segment_uri)
    if hasattr(rdd_or_df, 'rdd'):
      return rdd_or_df
    else:
      return cls._sd_rdd_to_sd_df(spark, rdd_or_df)

  @classmethod
  def get_sample(cls, uri, spark=None):
    with Spark.sess(spark) as spark:
      datums = cls._get_segment_datum_rdd_or_df(spark, uri)
      if hasattr(datums, 'rdd'):
        datums = cls.sd_df_to_rdd(datums)
      return Sample(uri=uri, datums=datums.collect())
      

  # @classmethod
  # def get_segment_datum_rdd(cls, spark, segment_uri, time_ordered=True):
  #   if util.missing_or_empty(cls.table_root()):
  #     datum_rdds = cls._create_datum_rdds(spark, only_segments=[segment_uri])
  #     if not datum_rdds:
  #       return spark.sparkContext.parallelize([])
  #     datum_rdd = spark.sparkContext.union(datum_rdds)
      
  #     from pyspark import StorageLevel
  #     datum_rdd = datum_rdd.persist(StorageLevel.MEMORY_AND_DISK)
  #     if time_ordered:
  #       datum_rdd = datum_rdd.sortBy(lambda sd: sd.uri.timestamp)
  #     return datum_rdd
  #   else:
  #     df = cls.as_df(spark)
  #     assert segment_uri.segment_id
  #     seg_df = df.filter(df.segment_id == segment_uri.segment_id)
  #     if segment_uri.dataset:
  #       seg_df = seg_df.filter(df.dataset == segment_uri.dataset)
  #     if segment_uri.split:
  #       seg_df = seg_df.filter(df.split == segment_uri.split)

  #     seg_df = seg_df.persist()
  #     if time_ordered:
  #       seg_df = seg_df.orderBy('uri.timestamp')
  #     return seg_df.rdd.map(StampedDatumTableBase.from_row)

  @staticmethod
  def to_row(sd):
    """This method is FINAL! ~~~~~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    # TODO do we need this method or can we add partition keys in a dataframe step ? ~~~~~~~~~~~~~~~~~~~~~
    row = RowAdapter.to_row(sd)
    row = row.asDict()

    # TODO: ditch these partition things and do it in the df writer?
    for k in StampedDatumTableBase.PARTITION_KEYS:
      row[k] = getattr(sd.uri, k)
    return Row(**row)

  @staticmethod
  def from_row(row):
    """This method is FINAL! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    return RowAdapter.from_row(row)

  # @classmethod
  # def as_uri_df(cls, spark):
  #   df = cls.as_df(spark)
  #   COLS = list(URI.__slots__)
  #   uri_df = df.select(*COLS)
  #   return uri_df

  # @classmethod
  # def as_stamped_datum_rdd(cls, spark):
  #   df = cls.as_df(spark)
  #   sd_rdd = df.rdd.map(RowAdapter.from_row)
  #   return sd_rdd


  ## Subclass API - Each dataset should provide ETL to a StampedDatumTable

  @classmethod
  def _get_all_segment_uris(cls):
    return []

  @classmethod
  def _create_datum_rdds(cls, spark, existing_uri_df=None, only_segments=None):
    """Subclasses should create and return a list of `RDD[StampedDatum]`s
    
    only_segments must be segment uris
    TODO docs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    return []


  ## Support

  @classmethod
  def table_schema(cls):
    """ comments ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    if not hasattr(cls, '_schema'):
      to_row = StampedDatumTableBase.to_row
      cls._schema = RowAdapter.to_schema(to_row(STAMPED_DATUM_PROTO))
    return cls._schema

  @classmethod
  def _sd_rdd_to_sd_df(cls, spark, sd_rdd):
    """ comments ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    
    row_rdd = sd_rdd.map(StampedDatumTableBase.to_row)
    df = spark.createDataFrame(row_rdd, schema=cls.table_schema())
    return df

  @classmethod
  def sd_df_to_rdd(cls, sd_df):
    # TODO refactor this and above ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
    return sd_df.rdd.map(cls.from_row)