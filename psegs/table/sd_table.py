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


from psegs.datum import URI
from psegs.datum.stamped_datum import Sample
from psegs.datum.stamped_datum import STAMPED_DATUM_PROTO
from psegs.spark import Spark

class StampedDatumTable(object):

  PARTITION_KEYS = ('dataset', 'split', 'segment_id')

  ## Datums -> StampedDatumTable

  def __init__(self):
    self._sample = None
    self._spark_df = None
    self._datum_rdd = None
    self._spark = None

  @classmethod
  def from_spark_df(cls, spark_df, spark=None):
    sdt = cls()
    sdt._spark_df = spark_df
    sdt._spark = spark
    return sdt

  @classmethod
  def from_sample(cls, sample):
    sdt = cls()
    sdt._sample = sample
    return sdt

  @classmethod
  def from_datum_rdd(cls, datum_rdd, spark=None):
    sdt = cls()
    sdt._datum_rdd = datum_rdd
    sdt._spark = spark
    return sdt


  ## StampedDatumTable -> Datums

  def to_spark_df(self, spark=None):
    spark = spark or self._spark
    if self._sample:
      with Spark.sess(spark) as spark:
        datum_rdd = spark.sparkContext.parallelize(self._sample.datums)
        self._spark_df = self._sd_rdd_to_sd_df(spark, datum_rdd)
    elif self._spark_df:
      return self._spark_df
    elif self._datum_rdd:
      with Spark.sess(spark) as spark:
        return self._sd_rdd_to_sd_df(spark, self._datum_rdd)
    else:
      # Create an empty Spark DF
      with Spark.sess(spark) as spark:
        self._spark_df = self._sd_rdd_to_sd_df(spark, [])
        return self._spark_df

  def to_sample(self):
    if self._sample:
      return self._sample
    elif self._spark_df:
      datum_rdd = self.datum_df_to_datum_rdd(self._spark_df)
      return Sample(datums=datum_rdd.collect())
    elif self._datum_rdd:
      return Sample(datums=self._datum_rdd.collect())
    else:
      return Sample()

  def to_datum_rdd(self, spark=None):
    if self._datum_rdd:
      return self._datum_rdd
    elif self._sample:
      spark = spark or self._spark
      with Spark.sess(spark) as spark:
        return spark.sparkContext.parallelize(
                    self._sample.datums, numSlices=len(self._sample.datums))
    elif self._spark_df:
      return self.datum_df_to_datum_rdd(self._spark_df)
    else:
      with Spark.sess(spark) as spark:
        return spark.sparkContext.parallelize([])
  

  ## Accessors

  def get_all_segment_uris(self):
    if self._sample:
      return [self._sample.uri.to_segment_uri()]
    elif self._spark_df:
      uri_df = self.as_uri_df()
      cols = ['dataset', 'split', 'segment_id']
      distinct_uri_df = uri_df.select(*cols).distinct()
      uri_rows = distinct_uri_df.collect()
      return [
        URI(dataset=r.dataset, split=r.split, segment_id=r.segment_id)
        for r in uri_rows
      ]
    elif self._datum_rdd:
      uri_rdd = self.as_uri_rdd()
      sseg_uri_rdd = uri_rdd.map(lambda uri: str(uri.to_segment_uri()))
      distinct_ssegs = sseg_uri_rdd.distinct()
      distinct_segs = distinct_ssegs.map(lambda s: URI.from_str(s))
      return distinct_segs.collect()
    else:
      return []

  def select_from_uris(self, uris, spark=None):
    print('todo')
    pass # TODO


  # @classmethod
  # def get_sample(cls, uri, spark=None):
  #   with Spark.sess(spark) as spark:
  #     datums = cls._get_segment_datum_rdd_or_df(spark, uri)
  #     if hasattr(datums, 'rdd'):
  #       datums = cls.datum_df_to_datum_rdd(datums)
  #     return Sample(uri=uri, datums=datums.collect())

  def get_datum_rdd_matching(
          self,
          only_topics=None,
          only_types=None,
          spark=None):
    
    if self._spark_df or self._sample:
      sdf = self.to_spark_df(spark=spark)
      if only_types:
        for sd_type in only_types:
          if sd_type in ('cuboids',):
            sdf = sdf.where("ARRAY_SIZE(%s) > 0" % sd_type)
          else:
            sdf = sdf.where("%s IS NOT NULL" % sd_type)
      if only_topics:
        sdf = sdf.where(sdf['uri.topic'].isin(only_topics))
    
      datum_rdd = self.datum_df_to_datum_rdd(sdf)
      return datum_rdd
    else:
      datum_rdd = self.to_datum_rdd(spark=spark)
      def matches(sd):
        matches_topics = (not only_topics or (sd.uri.topic in only_topics))
        matches_types = True
        for sd_type in only_types:
          if not bool(getattr(sd, sd_type)):
            matches_types = False
            break
        return matches_topics and matches_types
      datum_rdd = datum_rdd.filter(matches)
      return datum_rdd





  def as_uri_df(self, spark=None):

    df = self.to_spark_df(spark=spark)

    import attr
    colnames = [
      'uri.' + f.name
      for f in attr.fields(URI)
      if f.name not in self.PARTITION_KEYS
    ]
    colnames += [c for c in self.PARTITION_KEYS]
      # Use the partition columns for faster filters
    colnames += ['uri.__pyclass__']
    uri_df = df.select(colnames)
    return uri_df

  def as_uri_rdd(self, spark=None):
    if self._spark_df or self._sample:
      uri_df = self.as_uri_df(spark=spark)
      return uri_df.rdd.map(self.uri_from_row)
    else:
      datum_rdd = self.to_datum_rdd(spark=spark)
      return datum_rdd.map(lambda sd: sd.uri)


  ## Viz

  def to_rich_html(self, spark=None, **html_kwargs):
    """Create and return an HTML visualization with (small) vidoes and
    3D plots"""
  
    from psegs.spark import Spark
    from psegs.util.plotting import sample_to_html

    spark = spark or self._spark
    with Spark.sess(spark) as spark:
      return sample_to_html(
                spark,
                self.to_spark_df(spark=spark),
                **html_kwargs)


  ## I/O

  def save_parquet(
        self,
        dest_dir,
        partition=True,
        mode='overwrite',
        compression='zstd',
        spark=None,
        num_partitions=-1):

    save_opts = dict(
      path=str(dest_dir),
      compression=compression,
      mode=mode,
    )

    spark_df = self.to_spark_df(spark=spark).persist()
    if partition:
      for k in self.PARTITION_KEYS:
        spark_df = spark_df.withColumn(k, spark_df['uri.' + k])
      save_opts['partitionBy'] = self.PARTITION_KEYS      
    
    if num_partitions > 0:
      spark_df = spark_df.repartition(num_partitions).persist()
    
    spark_df.write.save(**save_opts)
    spark_df.unpersist()
  


  ## StampedDatum <-> Table Rows

  @classmethod
  def sd_to_row(cls, sd):
    from oarphpy.spark import RowAdapter
    from pyspark.sql import Row
    
    row = RowAdapter.to_row(sd)
    row = row.asDict()

    for k in cls.PARTITION_KEYS:
      row[k] = getattr(sd.uri, k)
    return Row(**row)

  @classmethod
  def sd_from_row(cls, row):
    from oarphpy.spark import RowAdapter
    return RowAdapter.from_row(row)
  
  @classmethod
  def uri_from_row(cls, row):
    from oarphpy.spark import RowAdapter
    if hasattr(row, 'uri'):
      return RowAdapter.from_row(row.uri)
    else:
      return RowAdapter.from_row(row)

  @classmethod
  def table_schema(cls):
    """ comments ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    if not hasattr(cls, '_schema'):
      from oarphpy.spark import RowAdapter
      to_row = cls.sd_to_row
      cls._schema = RowAdapter.to_schema(to_row(STAMPED_DATUM_PROTO))
    return cls._schema

  @classmethod
  def _sd_rdd_to_sd_df(cls, spark, sd_rdd):
    """ comments ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    
    row_rdd = sd_rdd.map(cls.sd_to_row)
    df = spark.createDataFrame(row_rdd, schema=cls.table_schema())
    return df

  @classmethod
  def datum_df_to_datum_rdd(cls, sd_df):
    return sd_df.rdd.map(cls.sd_from_row)
  


  ## Diffing

  def diff_with(self, other_sdt, spark=None):
    this_df = self.to_spark_df(spark=spark)
    other_df = other_sdt.to_spark_df(spark=spark)
    return self.find_diff(this_df, other_df)

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
      return "Dataset/Split Mismatch: \n%s" % misc.diff_of_pprint(ds1, ds2)
    
    
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
      return "Segment Mismatch: \n%s" % misc.diff_of_pprint(segs1, segs2)
    

    ## Next, let's compare URIs.  
    def get_uri_rdd(df):
      uri_rdd = df.select(df.uri).rdd.map(lambda row: cls.sd_from_row(row.uri))
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
    missing_rdd2 = sorted(kv1.subtractByKey(kv2).keys().collect())
    missing_rdd1 = sorted(kv2.subtractByKey(kv1).keys().collect())

    if missing_rdd1 or missing_rdd2:
      return """
                Missing URIs (first 50):
                Missing left (%s): %s 
                Missing right (%s): %s""" % (
                  len(missing_rdd1),
                  pprint.pformat(missing_rdd1[:50]),
                  len(missing_rdd2),
                  pprint.pformat(missing_rdd2[:50]))

    # ... and check for dupes!!
    has_dupes = lambda kv: kv[-1] > 1
    rdd1_dupes = kv1.filter(has_dupes).collect()
    rdd2_dupes = kv2.filter(has_dupes).collect()
    if rdd1_dupes or rdd2_dupes:
      return """
                Dupe URIs (first 50):
                Dupes left (%s): %s 
                Dupes right (%s): %s""" % (
                  len(rdd1_dupes),
                  pprint.pformat(rdd1_dupes[:50]),
                  len(rdd2_dupes),
                  pprint.pformat(rdd2_dupes[:50]))

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
      return "Datum mismatch (%s datums), first 10: \n%s" % (
        nonzero_diffs.count(),
        pprint.pformat(nonzero_diffs_sample))
    
    # No diffs!
    return ''
