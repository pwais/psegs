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

import itertools

import attr

from psegs import util
from psegs.datum.uri import URI
from psegs.table.sd_table import StampedDatumTableBase
from psegs.datum.stamped_datum import Sample, StampedDatum


URI_ATTRNAMES = set(a.name for a in attr.fields(URI))

def to_seg_uri_str(obj):
  import six
  if isinstance(obj, URI):
    uri = obj
  elif isinstance(obj, six.string_types):
    uri = URI.from_str(obj)
  elif hasattr(obj, 'asDict'):
    d = dict(
      (k, v)
      for k, v in obj.asDict().items()
      if k in URI_ATTRNAMES)
    uri = URI(**d)
  else:
    raise ValueError("Can't convert %s" % (obj,))
  return str(uri.to_segment_uri())

class NoKnownTable(Exception):
  pass

class StampedDatumDB(object):
  """

  TODO rename this UnionTable or something and put with sd_table

  """

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
    self._segments = []
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
      
    self._segments.append(seg_uri)
    util.log.info("Added DF for %s" % seg_uri)
      
  def _build_datum_df(self, uri, spark=None):
    spark = self._spark or spark
    suri = URI.from_str(uri).to_segment_uri()
    
    T = None
    for table in self._tables:
      t_seg_uris = table.get_all_segment_uris()
      if any(suri.soft_matches_segment_of(tu) for tu in t_seg_uris):
        T = table
        break
        
    if T is None:
      raise NoKnownTable("No known table for %s in %s" % (uri, self._tables))
    
    util.log.info("Building DF for %s" % uri)
    datum_df = T.get_segment_datum_df(spark, suri)
    return datum_df

  def _maybe_add_segment(self, uri, spark=None):
    spark = self._spark or spark
    suri = URI.from_str(uri).to_segment_uri()
    if not any(suri.soft_matches_segment_of(u) for u in self._segments):
      datum_df = self._build_datum_df(uri, spark=spark)
      self._add_datum_df(datum_df, seg_uri=suri)
  
  def _ensure_have_data(self, seg_uris, spark=None, ignore_unknown=False):
    for u in seg_uris:
      try:
        self._maybe_add_segment(u, spark=spark)
      except NoKnownTable as e:
        if not ignore_unknown:
          raise e

  @staticmethod
  def select_datum_df_from_uris(uris, datum_df):
    # Compile `uris` into the query itself for maximum speed
    
    import pyspark.sql.functions as F
    from functools import reduce

    def _to_match(uri):
      toks = []
      if uri.dataset:
        toks += [F.col('uri.dataset') == uri.dataset]
      if uri.split:
        toks += [F.col('uri.split') == uri.split]
      if uri.segment_id:
        toks += [F.col('uri.segment_id') == uri.segment_id]
      if uri.topic:
        # Topic implies both topic and timestamp are valid
        toks += [
          F.col('uri.topic') == uri.topic,
          F.col('uri.timestamp') == uri.timestamp
        ]
      
      return reduce(lambda a, b: a & b, toks)
      
    # Construct SELECT * FROM T WHERE uri = uri1 OR uri = uri2 OR ...
    print(reduce(
            lambda a, b: a | b,
            (_to_match(uri) for uri in uris)
          ))
    df = datum_df.where(
          reduce(
            lambda a, b: a | b,
            (_to_match(uri) for uri in uris)
          ))
    return df

  @staticmethod
  def select_datum_df_from_uri_df(uri_df, datum_df):
    df = datum_df.join(
          uri_df,
          (datum_df.uri.dataset == uri_df.dataset) &
          (datum_df.uri.split == uri_df.split) &
          (datum_df.uri.segment_id == uri_df.segment_id) &
          (datum_df.uri.topic == uri_df.topic) &
          (datum_df.uri.timestamp == uri_df.timestamp))
    return df

  @staticmethod
  def select_datum_df_from_uri_rdd(spark, uri_rdd, datum_df):
    from oarphpy.spark import RowAdapter
    row_rdd = uri_rdd.map(RowAdapter.to_row)
    schema = RowAdapter.to_schema(URI())
    uri_df = spark.createDataFrame(row_rdd, schema=schema)
    return StampedDatumDB.select_datum_df_from_uri_df(uri_df, datum_df)

  def get_sample(self, uri, spark=None):
    uri = URI.from_str(uri)
    self._maybe_add_segment(uri, spark=spark)
    
    uris = uri.get_datum_uris() or [uri]
    datum_df = self.get_datum_df(uris=uris, spark=spark)
    datum_rdd = StampedDatumTableBase.sd_df_to_rdd(datum_df)
    return Sample(uri=uri, datums=datum_rdd.collect())

  def get_datum_df(self, uris=None, spark=None):
    spark = self._spark or spark

    if hasattr(uris, '_jrdd'):
      uri_rdd = uris
      seg_uris = uri_rdd.map(to_seg_uri_str).distinct().collect()
      self._ensure_have_data(seg_uris, ignore_unknown=True, spark=spark)
      return StampedDatumDB.select_datum_df_from_uri_rdd(
                spark or self._spark,
                uri_rdd,
                self._df)
    elif hasattr(uris, 'rdd'):
      uri_df = uris
      suri_df = uri_df.select('dataset', 'split', 'segment_id').distinct()
      seg_uris = suri_df.rdd.map(to_seg_uri_str).distinct().collect()
      self._ensure_have_data(seg_uris, ignore_unknown=True, spark=spark)
      return StampedDatumDB.select_datum_df_from_uri_df(
                uri_df,
                self._df)
    else:
      seg_uris = list(set(to_seg_uri_str(u) for u in uris))
      self._ensure_have_data(seg_uris, spark=spark)

      uris = [URI.from_str(u) for u in uris]
      uris = list(itertools.chain.from_iterable(
        (u.get_datum_uris() or [u])
        for u in uris
      ))
      print('urisuris', uris)
      
      return StampedDatumDB.select_datum_df_from_uris(uris, self._df)

  def get_keyed_sample_df(self, df, key_col='key', uri_col='uri', spark=None):
    suri_df = df.select(
                    df[uri_col + '.dataset'],
                    df[uri_col + '.split'],
                    df[uri_col + '.segment_id']).distinct()
    seg_uris = suri_df.rdd.map(to_seg_uri_str).distinct().collect()
    self._ensure_have_data(seg_uris, ignore_unknown=True, spark=spark)

    datum_df = self._df
    key_uri_df = df.withColumnRenamed(uri_col, 'user_uri')
    key_datum_df = datum_df.join(
          key_uri_df,
          (datum_df.uri.dataset == key_uri_df['user_uri.dataset']) &
          (datum_df.uri.split == key_uri_df['user_uri.split']) &
          (datum_df.uri.segment_id == key_uri_df['user_uri.segment_id']) &
          (datum_df.uri.topic == key_uri_df['user_uri.topic']) &
          (datum_df.uri.timestamp == key_uri_df['user_uri.timestamp']))

    import attr
    datum_colnames = [f.name for f in attr.fields(StampedDatum)]
    aggs = dict((c, 'collect_list') for c in datum_colnames)
    key_sample_df = key_datum_df.groupBy(key_col).agg(aggs)
    return key_sample_df
