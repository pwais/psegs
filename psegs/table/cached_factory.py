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

from psegs import datum
from psegs import util
from psegs.table.sd_table_factory import StampedDatumTableFactory


class SampleCachedFactory(StampedDatumTableFactory):
  """Cache each created `StampedDatumTable` in process memory as a `Sample`;
  useful when there is tons of OS memory / swap space available."""

  CACHED_FACTORY = None

  @classmethod
  def get_all_segment_uris(cls):
    if cls.CACHED_FACTORY:
      return cls.CACHED_FACTORY.get_all_segment_uris()
    else:
      return []

  @classmethod
  def get_segment_sd_table(cls, segment_uri, spark=None):
    from psegs.table.sd_table import StampedDatumTable

    assert cls.CACHED_FACTORY is not None

    if not hasattr(cls, '_seg_uri_to_sdt'):
      cls._seg_uri_to_sdt = {}
    
    segment_uri = str(datum.URI.from_str(segment_uri).to_segment_uri())
    if segment_uri not in cls._seg_uri_to_sdt:
      sdt = cls.CACHED_FACTORY.get_segment_sd_table(segment_uri, spark=spark)
      sample = sdt.to_sample()
      sdt_sample = StampedDatumTable.from_sample(sample)
      cls._seg_uri_to_sdt[segment_uri] = sdt_sample
      util.log.info(f'SampleCachedFactory: cache SAVE {segment_uri}')
    else:
      util.log.info(f'SampleCachedFactory: cache HIT {segment_uri}')

    return cls._seg_uri_to_sdt[segment_uri]
  