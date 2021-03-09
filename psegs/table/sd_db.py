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

from psegs import util
from psegs.datum.uri import URI
from psegs.table.sd_table import StampedDatumTableBase
from psegs.datum.stamped_datum import Sample



def to_seg_uri_str(obj):
  import six
  if isinstance(obj, URI):
    uri = obj
  elif isinstance(obj, six.string_types):
    uri = URI.from_str(obj)
  elif hasattr(obj, '__dict__'):
    uri = URI(**obj.__dict__)
  return str(uri.to_segment_uri())


class StampedDatumDB(object):

  # @classmethod
  # def all_tables(cls):
  #   if not hasattr(cls, '_all_tables'):
  #     from psegs.datasets import kitti
  #     cls._all_tables = (
  #       kitti.KITTISDTable,
  #     )
  #   return cls._all_tables
  
  # @classmethod
  # def show_all_segment_uris(cls):
  #   """FIXME Interface ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
  #   for table in cls.all_tables():
  #     print(table)
  #     for seg_uri in table.get_all_segment_uris():
  #       print(seg_uri)

  # @classmethod
  # def get_segment_datum_rdd(cls, spark, segment_uri, time_ordered=True):
  #   """FIXME Interface ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

  #   def has_segment(table_segs):
  #     if segment_uri.dataset and segment_uri.split:
  #       return segment_uri in table_segs
  #     else:
  #       table_seg_ids = set(uri.segment_id for uri in table_segs)
  #       return segment_uri.segment_id in table_seg_ids

  #   for table in cls.all_tables():
  #     if has_segment(table.get_all_segment_uris()):
  #       return table.get_segment_datum_rdd(
  #         spark, segment_uri, time_ordered=time_ordered)
  #   return spark.sparkContext.parallelize([])

  def __init__(self, tables=[], spark=None):
    self._tables = tables
    self._segments = set()
    self._df = None
    self._spark = spark
  
  def _add_datum_df(self, datum_df, seg_uri=None):
    if not seg_uri:
      row = datum_df.select('uri').take(1)
      row = StampedDatumTableBase.from_row(row)
      uri = row.uri
      seg_uri = uri.to_segment_uri()
    
    if self._df is None:
      self._df = datum_df
    else:
      from oarphpy.spark import union_dfs
      self._df = union_dfs(self._df, datum_df)
      
    self._segments.add(str(seg_uri))
    util.log.info("Added DF for %s" % seg_uri)
      
  def _build_datum_df(self, uri, spark=None):
    spark = self._spark or spark
    suri = URI.from_str(uri).to_segment_uri()
    ssuri = str(suri)
    
    T = None
    for table in self._tables:
      seg_uris = table.get_all_segment_uris()
      if ssuri in set(str(s) for s in seg_uris):
        T = table
        break
        
    if T is None:
      raise ValueError("No known table for %s" % uri)
    
    util.log.info("Building DF for %s" % uri)
    datum_df = T.get_segment_datum_df(spark, suri)
    return datum_df

  def _maybe_add_segment(self, uri, spark=None):
    spark = self._spark or spark
    suri = URI.from_str(uri).to_segment_uri()
    ssuri = str(suri)
    
    if ssuri not in self._segments:
      datum_df = self._build_datum_df(uri, spark=spark)
      self._add_datum_df(datum_df, seg_uri=suri)
  
  @staticmethod
  def select_datum_df_from_uris(uris, datum_df):
    import pyspark.sql.functions as F
    from functools import reduce
    df = datum_df.where(
          reduce(
            lambda a, b: a | b, (
            (
              (F.col('uri.dataset') ==      uri.dataset) & 
              (F.col('uri.split') ==        uri.split) & 
              (F.col('uri.segment_id') ==   uri.segment_id) & 
              (F.col('uri.topic') ==        uri.topic) & 
              (F.col('uri.timestamp') ==    uri.timestamp)
            )
            for uri in uris)))
    return df

  @staticmethod
  def select_datum_df_from_uri_df(uri_df, datum_df):
    df = datum_df.join(
          uri_df,
          (datum_df.dataset == uri_df.dataset) &
          (datum_df.split == uri_df.split) &
          (datum_df.segment_id == uri_df.segment_id) &
          (datum_df.topic == uri_df.topic) &
          (datum_df.timestamp == uri_df.timestamp))
    return df

  @staticmethod
  def select_datum_df_from_uri_rdd(spark, uri_rdd, datum_df):
    from oarphpy.spark import RowAdapter
    row_rdd = uri_rdd.map(RowAdapter.to_row)
    uri_df = spark.createDataFrame(row_rdd)
    return StampedDatumDB.select_datum_df_from_uri_df(uri_df, datum_df)

  def get_sample(self, uri, spark=None):
    self._maybe_add_segment(uri, spark=spark)
    
    datum_df = self.get_datum_df(uris=[uri], spark=spark)
    datum_rdd = StampedDatumTableBase.sd_df_to_rdd(datum_df)
    return Sample(uri=uri, datums=datum_rdd.collect())

  def get_datum_df(self, uris=None, spark=None):
    spark = self._spark or spark

    def _ensure_have_data(seg_uris):
      for u in seg_uris:
        self._maybe_add_segment(u, spark=spark)

    if hasattr(uris, '_jrdd'):
      uri_rdd = uris
      seg_uris = uri_rdd.map(to_seg_uri_str).distinct().collect()
      _ensure_have_data(seg_uris)
      return StampedDatumDB.select_datum_df_from_uri_rdd(
                spark or self._spark,
                uri_rdd,
                self._df)
    elif hasattr(uris, '_rdd'):
      uri_df = uris
      uri_df = uri_df.select('dataset', 'split', 'segment_id').distinct()
      seg_uris = uri_df.rdd.map(to_seg_uri_str).distinct().collect()
      _ensure_have_data(seg_uris)
      return StampedDatumDB.select_datum_df_from_uri_df(
                uri_df,
                self._df)
    else:
      seg_uris = list(set(to_seg_uri_str(u) for u in uris))
      _ensure_have_data(seg_uris)
      return StampedDatumDB.select_datum_df_from_uris(uris, self._df)
