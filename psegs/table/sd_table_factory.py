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

from pathlib import Path

import attr

from psegs import util
from psegs.datum import URI
from psegs.table.sd_table import StampedDatumTable


class StampedDatumTableFactory(object):

  ## Public API

  PARTITION_KEYS = ('dataset', 'split', 'segment_id')
  
  @classmethod
  def get_all_segment_uris(cls):
    return [URI.from_str(u) for u in cls._get_all_segment_uris()]

  @classmethod
  def get_segment_sd_table(cls, segment_uri, spark=None):
    from psegs.spark import Spark
    with Spark.sess(spark) as spark:
      datum_rdds = cls._create_datum_rdds(
                        spark, 
                        only_segments=[segment_uri])
      datum_rdd = spark.sparkContext.union(datum_rdds)
      
      from pyspark import StorageLevel
      datum_rdd = datum_rdd.persist(StorageLevel.MEMORY_AND_DISK)

      return StampedDatumTable.from_datum_rdd(datum_rdd, spark=spark)

  @classmethod
  def save_parquet(
        cls,
        dest_dir,
        only_segments=None,
        existing_uri_df=None,
        auto_resume_incomplete=True,
        spark=None):
    
    from psegs import util
    from psegs.spark import Spark
    
    dest_dir = Path(dest_dir)

    with Spark.sess(spark) as spark:
      if auto_resume_incomplete and existing_uri_df is None and dest_dir.exists():
        util.log.info(f"Attempting to resume from {dest_dir} ...")

        F = ParquetSDTFactory.factory_for_sd_subdirs(dest_dir)
        existing_uri_df = F.read_uri_df(spark=spark)
        
        if existing_uri_df is not None:
          util.log.info(
            f"... found {existing_uri_df.count()} datums in {dest_dir} ...")
        else:
          util.log.info(f"... found no datum data in {dest_dir} ...")
    
      if only_segments:
        only_segments = [URI.from_str(s).to_segment_uri() for s in only_segments]
    
      sd_rdds = cls._create_datum_rdds(
                            spark, 
                            existing_uri_df=existing_uri_df,
                            only_segments=only_segments)
      sd_tables = [
        StampedDatumTable.from_datum_rdd(sd_rdd) for sd_rdd in sd_rdds
      ]

      from psegs.spark import save_sd_tables
      save_sd_tables(
        sd_tables,
        spark=spark,
        spark_save_opts=dict(
          path=dest_dir / 'stamped_datums',
          format='parquet',
          partitionBy=cls.PARTITION_KEYS,
          compression='zstd'))

  ## Subclass API - Datasets should provide ETL to lists of RDD[StampedDatum]

  @classmethod
  def _get_all_segment_uris(cls):
    return []

  @classmethod
  def _create_datum_rdds(cls, spark, existing_uri_df=None, only_segments=None):
    """Subclasses should create and return a list of `RDD[StampedDatum]`s
    
    only_segments must be segment uris
    TODO docs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    return []


  # @classmethod
  # def as_df(cls, spark, force_compute=False, only_segments=None):
  #   # TODO REPLACE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #   """ comments ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
  #   if force_compute: # hacks
  #     sd_rdds = cls._create_datum_rdds(spark, only_segments=only_segments)
  #     sd_dfs = [cls._sd_rdd_to_sd_df(spark, rdd) for rdd in sd_rdds]
  #     import oarphpy.spark
  #     return oarphpy.spark.union_dfs(*sd_dfs)


  #   util.log.info("Loading %s ..." % cls.table_root())
  #   # df = spark.read.option("mergeSchema", "true").parquet(str(cls.table_root()))
  #   df = spark.read.parquet(str(cls.table_root()))
  #   # df = spark.read.schema(cls.table_schema()).option("mergeSchema", "true").load(path=cls.table_root())
  #   # read = read.schema(
  #   # df = spark.read.parquet(cls.table_root(), schema=cls.table_schema())
  #   return df

  # @classmethod
  # def as_datum_rdd(cls, spark, df=None):
  #   df = df or cls.as_df(spark)
  #   return df.rdd.map(StampedDatumTableFactory.from_row)

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
  #     return seg_df.rdd.map(StampedDatumTableFactory.from_row)

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
  #     return seg_df.rdd.map(StampedDatumTableFactory.from_row)

  @staticmethod
  def to_row(sd):
    """This method is FINAL! ~~~~~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    # TODO do we need this method or can we add partition keys in a dataframe step ? ~~~~~~~~~~~~~~~~~~~~~
    row = RowAdapter.to_row(sd)
    row = row.asDict()

    # TODO: ditch these partition things and do it in the df writer?
    for k in StampedDatumTableFactory.PARTITION_KEYS:
      row[k] = getattr(sd.uri, k)
    return Row(**row)

  @staticmethod
  def from_row(row):
    """This method is FINAL! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    return RowAdapter.from_row(row)

  # @classmethod
  # def as_uri_df(cls, spark):
  #   if util.missing_or_empty(cls.table_root()):
  #     return spark.sparkContext.parallelize([])
  #   df = cls.as_df(spark)

  #   import attr
  #   colnames = [
  #     'uri.' + f.name
  #     for f in attr.fields(URI)
  #     if f.name not in cls.PARTITION_KEYS
  #   ]
  #   colnames += [c for c in cls.PARTITION_KEYS]
  #     # Use the partition columns for faster filters
  #   uri_df = df.select(colnames)
  #   return uri_df
    # COLS = list(URI.__slots__)
    # uri_df = df.select(*COLS)
    # return uri_df

  # @classmethod
  # def as_stamped_datum_rdd(cls, spark):
  #   df = cls.as_df(spark)
  #   sd_rdd = df.rdd.map(RowAdapter.from_row)
  #   return sd_rdd


  

  ## Support

  @classmethod
  def table_schema(cls):
    """ comments ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    if not hasattr(cls, '_schema'):
      to_row = StampedDatumTableFactory.to_row
      cls._schema = RowAdapter.to_schema(to_row(STAMPED_DATUM_PROTO))
    return cls._schema

  @classmethod
  def _sd_rdd_to_sd_df(cls, spark, sd_rdd):
    """ comments ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    
    row_rdd = sd_rdd.map(StampedDatumTableFactory.to_row)
    df = spark.createDataFrame(row_rdd, schema=cls.table_schema())
    return df

  @classmethod
  def sd_df_to_rdd(cls, sd_df):
    # TODO refactor this and above ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
    return sd_df.rdd.map(cls.from_row)
  
  
class ParquetSDTFactory(StampedDatumTableFactory):
  PQ_DIRS = []

  @classmethod
  def factory_for_sd_subdirs(cls, pq_base_path, sd_dir_name='stamped_datums'):
    from pathlib import Path
    pq_dirs = sorted(
                p
                for p in Path(pq_base_path).rglob('*')
                if (p.is_dir and p.name == sd_dir_name))

    class MyPQSDTFactory(cls):
      PQ_DIRS = pq_dirs

    return MyPQSDTFactory

  @classmethod
  def read_spark_df(cls, spark=None):
    from psegs.spark import Spark
    with Spark.sess(spark) as spark:
      df = None
      for p in cls.PQ_DIRS:
        p_df = spark.read.parquet(str(p))
        df = p_df if df is None else df.union(p_df)
      return df

  @classmethod
  def read_seg_uri_rdd(cls, spark=None):
    print("TODO ?? uri_df = cls.as_uri_df(spark=spark)")
    df = cls.read_spark_df(spark=spark)
    if df is None:
      return None

    seg_col_df = df.select(
                    df['uri.dataset'].alias('dataset'),
                    df['uri.split'].alias('split'),
                    df['uri.segment_id'].alias('segment_id'))
    seg_col_df = seg_col_df.distinct()

    def to_uri(row):
      return URI(
              dataset=row.dataset,
              split=row.split,
              segment_id=row.segment_id)
    
    uri_rdd = seg_col_df.rdd.map(to_uri)
    return uri_rdd

  @classmethod
  def read_uri_df(cls, spark=None):
    print("TODO move to superclass if possible? tho is generally expensive")
    df = cls.read_spark_df(spark=spark)
    if df is None:
      return None

    colnames = [
      'uri.' + f.name
      for f in attr.fields(URI)
      if f.name not in cls.PARTITION_KEYS
    ]
    colnames += [c for c in cls.PARTITION_KEYS]
      # Use the partition columns for faster filters
    colnames += ['uri.__pyclass__']
    uri_df = df.select(colnames)
    return uri_df

#     return uri_df

# def as_uri_df(self, spark=None):
    

#   def as_uri_rdd(self, spark=None):
#     if self._spark_df or self._sample:
#       uri_df = self.as_uri_df(spark=spark)
#       return uri_df.rdd.map(self.uri_from_row)
#     else:
#       datum_rdd = self.to_datum_rdd(spark=spark)
#       return datum_rdd.map(lambda sd: sd.uri)


#     df = cls.read_spark_df(spark=spark)
#     if df.rdd.isEmpty():
#       return None

#     def uri_from_row(row):
#       from oarphpy.spark import RowAdapter
#       return RowAdapter.from_row(row.uri)
    
#     uri_rdd = df.select(df['uri']).rdd.map(uri_from_row)
#     return uri_rdd

  ## StampedDatumTableFactory Impl

  @classmethod
  def _get_all_segment_uris(cls):
    seg_uri_rdd = cls.read_seg_uri_rdd()
    return sorted(seg_uri_rdd.collect())

  @classmethod
  def _create_datum_rdds(cls, spark, existing_uri_df=None, only_segments=None):
    if existing_uri_df is not None:
      util.log.info(
        f"Note: resume mode unsupported, got existing_uri_df {existing_uri_df}")
    
    seg_uri_rdd = cls.read_seg_uri_rdd(spark=spark)
    if only_segments:
      def has_match(s):
        return any(
              URI.from_str(suri).soft_matches_segment_of(s)
              for suri in only_segments)
      seg_uri_rdd = seg_uri_rdd.filter(has_match)
    segs_to_load = seg_uri_rdd.collect()

    util.log.info(f"Creating datum RDDs for {len(segs_to_load)} segments ...")

    union_df = cls.read_spark_df(spark=spark)
    datum_rdds = []
    for suri in segs_to_load:
      seg_df = union_df.filter(
                (union_df.dataset == suri.dataset) &
                (union_df.split == suri.split) &
                (union_df.segment_id == suri.segment_id))

      from psegs.table.sd_table import StampedDatumTable
      datum_rdd = StampedDatumTable.datum_df_to_datum_rdd(seg_df)
      datum_rdds.append(datum_rdd)

    util.log.info(f"... created datum RDDs.")

    return datum_rdds
