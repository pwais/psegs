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

import os

from oarphpy import spark
from oarphpy import util as oputil

import psegs
from psegs import util


class Spark(spark.SessionFactory):

  SRC_ROOT = os.path.dirname(psegs.__file__)

  SRC_ROOT_MODULES = ['psegs']

  CONF_KV = {
    'spark.driver.maxResultSize': '2g',
    'spark.driver.memory': '8g',
    'spark.memory.offHeap.enabled': 'true',
    'spark.memory.offHeap.size': '64g',

    'spark.files.overwrite': 'true',
      # Needed for notebook-based development; FMI see oarphpy.spark.NBSpark

    'spark.python.worker.reuse': False,
      # Helps reduce memory leaks related to matplotlib / tensorflow / etc

    'spark.sql.files.maxPartitionBytes': int(8 * 1e6),
      # Partitions need to be big enough to potentially fit
      # point clouds / images

    'spark.port.maxRetries': '256',
      # Allow lots of Spark drivers on a single machine (e.g. a dev machine)
    
    'spark.executorEnv.COLUMNS': '80',
    'spark.executorEnv.LINES': '80',
      # These settings make the `mdv` python package happy when we use it in a
      # Spark executor that has no terminal
  }


  # PSegs Utilities

  @staticmethod
  def save_df_thunks(df_thunks, compute_df_sizes=True, **save_opts):
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
      df.write.save(mode='append', **save_opts)
      df.unpersist()
      
      t.stop_block(n=1, num_bytes=num_bytes)
      t.maybe_log_progress(every_n=1)
