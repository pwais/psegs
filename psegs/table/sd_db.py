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

class StampedDatumDB(object):

  @classmethod
  def all_tables(cls):
    if not hasattr(cls, '_all_tables'):
      from psegs.datasets import kitti
      cls._all_tables = (
        kitti.KITTISDTable,
      )
    return cls._all_tables
  
  @classmethod
  def show_all_segment_uris(cls):
    """FIXME Interface ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    for table in cls.all_tables():
      print(table)
      for seg_uri in table.get_all_segment_uris():
        print(seg_uri)

  @classmethod
  def get_segment_datum_rdd(cls, spark, segment_uri, time_ordered=True):
    """FIXME Interface ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

    def has_segment(table_segs):
      if segment_uri.dataset and segment_uri.split:
        return segment_uri in table_segs
      else:
        table_seg_ids = set(uri.segment_id for uri in table_segs)
        return segment_uri.segment_id in table_seg_ids

    for table in cls.all_tables():
      if has_segment(table.get_all_segment_uris()):
        return table.get_segment_datum_rdd(
          spark, segment_uri, time_ordered=time_ordered)
    return spark.sparkContext.parallelize([])
