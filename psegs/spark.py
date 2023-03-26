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

import copy
import os

from oarphpy import spark
from oarphpy import util as oputil

import psegs
from psegs import util


class Spark(spark.SessionFactory):

  SRC_ROOT = os.path.dirname(psegs.__file__)

  SRC_ROOT_MODULES = ['psegs']

  CONF_KV = {
    'spark.driver.maxResultSize': '20g',
    'spark.driver.memory': '32g',
    'spark.executor.memory': '32g',
    # 'spark.driver.cores': '6',
    # 'spark.memory.offHeap.enabled': 'true',
    # 'spark.memory.offHeap.size': '100g',

    'spark.files.overwrite': 'true',
      # Needed for notebook-based development; FMI see oarphpy.spark.NBSpark

    'spark.python.worker.reuse': False,
    # 'spark.blockManager.port': '5555',
      # Helps reduce memory leaks related to matplotlib / tensorflow / etc
    # 'spark.driver.extraJavaOptions': '-Dlog4j.logger.org.apache.spark.api.python.PythonGatewayServer=DEBUG',
    # 'spark.driver.extraJavaOptions': '-Dlog4jspark.root.logger=DEBUG,console',
    'spark.sql.files.maxPartitionBytes': int(8 * 1e6),
      # Partitions need to be big enough to potentially fit
      # point clouds / images

    'spark.port.maxRetries': '256',
      # Allow lots of Spark drivers on a single machine (e.g. a dev machine)

  }

def save_sd_tables(
      sdts,
      spark=None,
      compute_df_sizes=True,
      spark_save_opts=None):
  
  class DFThunk:
    def __init__(self, t, spark):
      self.t = t
      self.spark = spark
    def __call__(self):
      return self.t.to_spark_df(spark=self.spark)

  df_thunks = [DFThunk(t, spark) for t in sdts]
  return save_df_thunks(
    df_thunks,
    compute_df_sizes=compute_df_sizes,
    spark_save_opts=spark_save_opts)

def save_df_thunks(df_thunks, compute_df_sizes=True, spark_save_opts=None):
  spark_save_opts = spark_save_opts or {}
  if 'path' in spark_save_opts:
    # JRDD bridge below requires string
    spark_save_opts['path'] = str(spark_save_opts['path'])
  
  t = oputil.ThruputObserver(name='save_df_thunks', n_total=len(df_thunks))
  util.log.info("Going to write in %s chunks ..." % len(df_thunks))
  while len(df_thunks):
    df_thunk = df_thunks.pop(0)
    t.start_block()
    df = df_thunk()
    # df = df.persist()
    # print('df size', df.count())
    # df.show()
    num_bytes = 0
    if compute_df_sizes:
      df = df.persist()
      # def getsize(x):#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #   y = oputil.get_size_of_deep(x)
      #   print(x.uri.topic, y / (2 **20))
      #   return y
      # num_bytes = df.rdd.map(getsize).sum()
      num_bytes = df.rdd.map(oputil.get_size_of_deep).sum()
    df.write.save(mode='append', **spark_save_opts)
    df.unpersist()
    
    t.stop_block(n=1, num_bytes=num_bytes)
    t.maybe_log_progress(every_n=1)

# Expose a NBSpark "subclass" configured for PSegs
NBSpark = copy.deepcopy(spark.NBSpark)
NBSpark.SRC_ROOT = Spark.SRC_ROOT
NBSpark.SRC_ROOT_MODULES = Spark.SRC_ROOT_MODULES
NBSpark.CONF_KV.update(Spark.CONF_KV)
