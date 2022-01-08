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
        if only_segments:
          seg_uris = [URI.from_str(s).to_segment_uri() for s in only_segments]

          existing_uri_df = existing_uri_df.filter(
                              existing_uri_df.dataset.isin(
                                [u.dataset for u in seg_uris]) &
                              existing_uri_df.split.isin(
                                [u.split for u in seg_uris]) &
                              existing_uri_df.segment_id.isin(
                                [u.segment_id for u in seg_uris]))

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
  def as_df(cls, spark, force_compute=False):
    # TODO REPLACE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """ comments ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    if force_compute: # hacks
      sd_rdds = cls._create_datum_rdds(spark)
      sd_dfs = [cls._sd_rdd_to_sd_df(spark, rdd) for rdd in sd_rdds]
      import oarphpy.spark
      return oarphpy.spark.union_dfs(*sd_dfs)


    util.log.info("Loading %s ..." % cls.table_root())
    # df = spark.read.option("mergeSchema", "true").parquet(str(cls.table_root()))
    df = spark.read.parquet(str(cls.table_root()))
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
    if True:#util.missing_or_empty(cls.table_root()):
      print('fixme')
      return cls._create_segment_datum_rdd(spark, segment_uri)
    else:
      return cls._get_segment_df(spark, segment_uri)
  
  @classmethod
  def _create_segment_datum_rdd(cls, spark, segment_uri):
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

  @classmethod
  def _get_segment_datum_df_from_disk(cls, spark, segment_uri):
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

  @classmethod
  def as_uri_df(cls, spark):
    if util.missing_or_empty(cls.table_root()):
      return spark.sparkContext.parallelize([])
    df = cls.as_df(spark)

    import attr
    colnames = [
      'uri.' + f.name
      for f in attr.fields(URI)
      if f.name not in cls.PARTITION_KEYS
    ]
    colnames += [c for c in cls.PARTITION_KEYS]
      # Use the partition columns for faster filters
    uri_df = df.select(colnames)
    return uri_df
    # COLS = list(URI.__slots__)
    # uri_df = df.select(*COLS)
    # return uri_df

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
  
  @classmethod
  def find_diff(cls, sd_df1, sd_df2):
    """Compare all entries of Spark DataFrames of `StampedDatum`s `sd_df1`
    and `sd_df2` and return a string report on the first major difference.
    Return the empty string if the tables are equal.
    """

    import pprint
    from operator import add

    import attr
    from pyspark.sql import functions as F

    from psegs.util import misc
    from psegs.datum import datumutils as du
    from psegs.datum.stamped_datum import StampedDatum


    ## First, do we have the same datasets / splits?
    def get_dataset_splits(df):
      rows = df.select(df.uri.dataset, df.uri.split).distinct().collect()
      return sorted(tuple(r) for r in rows)
    
    ds1 = get_dataset_splits(sd_df1)
    ds2 = get_dataset_splits(sd_df2)
    if ds1 != ds2:
      return "Dataset/Split Mismatch: %s" % misc.diff_of_pprint(ds1, ds2)
    
    
    ## Next, do we have the same segments?
    def get_seg_uris(df):
      rows = df.select(
              df.uri.dataset,
              df.uri.split,
              df.uri.segment_id).distinct().collect()
      return sorted(tuple(r) for r in rows)

    segs1 = get_seg_uris(sd_df1)
    segs2 = get_seg_uris(sd_df2)
    if segs1 != segs2:
      return "Segment Mismatch: %s" % misc.diff_of_pprint(segs1, segs2)
    

    ## Next, let's compare URIs.  
    def get_uri_rdd(df):
      uri_rdd = df.select(df.uri).rdd.map(lambda row: cls.from_row(row.uri))
      return uri_rdd
    
    # First in number ...
    uri_rdd1 = get_uri_rdd(sd_df1).cache()
    uri_rdd2 = get_uri_rdd(sd_df2).cache()
    c1 = uri_rdd1.count()
    c2 = uri_rdd2.count()

    if c1 == 0 and c2 == 0:
      return '' # Short-circuit: Spark is slow for empty data below
    elif c1 == 0 and c2 > 0:
      return "Left table is EMPTY but right table has %s rows" % c2
    elif c1 > 0 and c2 == 0:
      return "Right table is EMPTY but left table has %s rows" % c1
    elif c1 != c2 and (abs(c1 - c2) >= 1000):
      return "URI Count Mismatch: left count: %s right count: %s" % (c1, c2)

    # ... then in content ...
    to_key = lambda uri: (uri.to_str(), 1)
    kv1 = uri_rdd1.map(to_key).reduceByKey(add)
    kv2 = uri_rdd2.map(to_key).reduceByKey(add)
    missing_rdd1 = kv1.subtractByKey(kv2).values().collect()
    missing_rdd2 = kv2.subtractByKey(kv1).values().collect()

    if missing_rdd1 or missing_rdd2:
      return "Missing URIs: %s" % misc.diff_of_pprint(
        missing_rdd1, missing_rdd2)

    # ... and check for dupes!!
    has_dupes = lambda kv: kv[-1] > 1
    rdd1_dupes = kv1.filter(has_dupes).collect()
    rdd2_dupes = kv2.filter(has_dupes).collect()
    if rdd1_dupes or rdd2_dupes:
      return "Dupe URIs (first 100): %s" % misc.diff_of_pprint(
                rdd1_dupes[:100], rdd2_dupes[:100])


    ## Finally, let's compare actual Datums.
    SD_COLS = [f.name for f in attr.fields(StampedDatum)]

    # Do key-value mapping and join using the Dataframe API
    # because it's faster
    def to_key_datum_df(df):
      URI_KEYS = (
        'dataset',        # Use partition col
        'split',          # Use partition col
        'segment_id',     # Use partition col
        'uri.timestamp',  # Must read from uri
        'uri.topic')      # Must read from uri
      kv_df = df.select(
                F.concat(*URI_KEYS).alias('key'),
                F.struct(*SD_COLS).alias('datum'))
      return kv_df
    
    kv_df1 = to_key_datum_df(sd_df1)
    kv_df2 = to_key_datum_df(sd_df2)
    kv_df1 = kv_df1.withColumn('datum1', kv_df1.datum)
    kv_df2 = kv_df2.withColumn('datum2', kv_df2.datum)
    joined = kv_df1.join(kv_df2, on='key', how='inner')

    def get_diff_string(row):
      return du.get_datum_diff_string(row.datum1, row.datum2)

    diffs = joined.rdd.map(get_diff_string)
    nonzero_diffs = diffs.filter(lambda s: bool(s))
    nonzero_diffs_sample = nonzero_diffs.take(10)
    if nonzero_diffs_sample:
      return "Datum mismatch, first 10: %s" % (
        pprint.pformat(nonzero_diffs_sample))
    
    # No diffs!
    return ''
