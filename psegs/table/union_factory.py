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
    return sorted(iseg_uris)

  @classmethod
  def get_segment_sd_table(cls, segment_uri, spark=None):
    F = cls._get_factory_for_seg_uri(segment_uri)
    return F.get_segment_sd_table(segment_uri=segment_uri, spark=spark)

  # @classmethod
  # def build_cache(cls, spark=None, only_segments=None, table_root=''):
  #   raise NotImplementedError # TODO

  @classmethod
  def _seguri_to_factoryidx(cls):
    if not hasattr(cls, '_seguri_to_factoryidx_cache'):
      seguri_to_factoryidx = {}
      for F_idx, F in enumerate(cls.SDT_FACTORIES):
        seg_uris = F.get_all_segment_uris()
        for seg_uri in seg_uris:
          seguri_to_factoryidx[str(seg_uri.to_segment_uri())] = F_idx
      cls._seguri_to_factoryidx_cache = seguri_to_factoryidx
    return cls._seguri_to_factoryidx_cache
  
  @classmethod
  def _get_factory_for_seg_uri(cls, seg_uri):
    seg_uri = datum.URI.from_str(seg_uri)    
    seg_uri_key = str(seg_uri.to_segment_uri())
    F_idx = cls._seguri_to_factoryidx().get(seg_uri_key)
    if F_idx is None:
      raise NoKnowStampedDatumTableFactory(seg_uri)
    else:
      return cls.SDT_FACTORIES[F_idx]

