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


import itertools

from psegs import datum
from psegs.table.sd_table import StampedDatumTable
from psegs.table.sd_table_factory import StampedDatumTableFactory


class NoKnowStampedDatumTableFactory(Exception):
  pass


class UnionFactory(StampedDatumTableFactory):
  """Induce a union over a list of `StampedDatumTableFactory`s, with amortized
  O(1) lookup of factory by segment URI."""

  SDT_FACTORIES = []

  @classmethod
  def get_all_segment_uris(cls):
    iseg_uris = itertools.chain.from_iterable(
        F.get_all_segment_uris() for F in cls.SDT_FACTORIES)
    distinct_set_uris = set(str(suri) for suri in iseg_uris)
    return sorted(datum.URI.from_str(s) for s in distinct_set_uris)

  @classmethod
  def get_segment_sd_table(cls, segment_uri, spark=None):
    from psegs.spark import Spark
    Fs = cls._get_factories_for_seg_uri(segment_uri)
    with Spark.sess(spark) as spark:
      # TODO make this more flexible, for now we assume 
      # SDTF::get_segment_sd_table() gets a datum_rdd sdt and so it's 
      # fastest to just union those.
      union_datum_rdd = None
      for F in Fs:
        sdt = F.get_segment_sd_table(segment_uri=segment_uri, spark=spark)
        datum_rdd = sdt.to_datum_rdd()
        if union_datum_rdd is None:
          union_datum_rdd = datum_rdd
        else:
          union_datum_rdd = union_datum_rdd.union(datum_rdd)

      union_sdt = StampedDatumTable.from_datum_rdd(
        union_datum_rdd, spark=spark)
    return union_sdt

  # @classmethod
  # def build_cache(cls, spark=None, only_segments=None, table_root=''):
  #   raise NotImplementedError # TODO

  @classmethod
  def _seguri_to_factoryidxes(cls):
    if not hasattr(cls, '_seguri_to_factoryidxes_cache'):
      seguri_to_factoryidxes = {}
      for F_idx, F in enumerate(cls.SDT_FACTORIES):
        seg_uris = F.get_all_segment_uris()
        for seg_uri in seg_uris:
          key = str(seg_uri.to_segment_uri())
          if key not in seguri_to_factoryidxes:
            seguri_to_factoryidxes[key] = []
          seguri_to_factoryidxes[key].append(F_idx)
      cls._seguri_to_factoryidxes_cache = seguri_to_factoryidxes
    return cls._seguri_to_factoryidxes_cache
  
  @classmethod
  def _get_factories_for_seg_uri(cls, seg_uri):
    seg_uri = datum.URI.from_str(seg_uri)
    seg_uri_key = str(seg_uri.to_segment_uri())
    F_idxes = cls._seguri_to_factoryidxes().get(seg_uri_key)
    if F_idxes is None:
      raise NoKnowStampedDatumTableFactory(str(seg_uri))
    else:
      return [
        cls.SDT_FACTORIES[F_idx]
        for F_idx in F_idxes
      ]

