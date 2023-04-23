# Copyright 2021 Maintainers of PSegs
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

SEGXFORM_DESC = """
segxform - A script to processs one or more PSegs segments.

## Example


"""

import os

import six

from psegs.datum import URI


def configure_arg_parser(parser=None):
  """Configure the `ArgumentParser` instance `parser` with PSegs-related
  options and return the parser.  Create an `ArgumentParser` if needed.
  """
  
  from psegs.conf import DEFAULT_DATA_ROOT

  if parser is None:
    import argparse
    parser = argparse.ArgumentParser(
                      description="Default PSegs segxform job",
                      formatter_class=argparse.RawDescriptionHelpFormatter)
  segex_sel_group = parser.add_argument_group(
                        "PSegs Selection",
                        "Select the data to process")
  segex_env_group = parser.add_argument_group(
                        "PSegs Environment",
                        "Configure where PSegs looks for assets")

  # Pick your data
  segex_sel_group.add_argument(
    '--segment-id', default='',
    help='Select only this segment. Use --dataset and/or --split if you need '
         'to distinguish segments with the same name.')
  segex_sel_group.add_argument(
    '--segment-ids-with', default='',
    help='Select only segment IDs that contain this string.')
  segex_sel_group.add_argument(
    '--dataset', default='',
    help='Restrict to only this dataset')
  segex_sel_group.add_argument(
    '--split', default='',
    help='Restrict to only this split')

  # Configure PSegs environment fixtures
  segex_env_group.add_argument(
    '--ps-root', default=DEFAULT_DATA_ROOT,
    help='Use this as the PSegs root (where PSegs code and data '
         'fixtures live) [default %(default)s]')

  return parser


def get_matching_seg_uris(args):
  from psegs.table.canonical_factory import CanonicalFactory
  seg_uris = CanonicalFactory.get_all_segment_uris()

  if args.segment_id:
    seg_uris = [
      suri for suri in seg_uris
      if suri.segment_id == args.segment_id
    ]
  if args.segment_ids_with:
    seg_uris = [
      suri for suri in seg_uris
      if args.segment_ids_with in suri.segment_id
    ]
  if args.dataset:
    seg_uris = [
      suri for suri in seg_uris
      if suri.dataset == args.dataset
    ]
  if args.split:
    seg_uris = [
      suri for suri in seg_uris
      if suri.split == args.split
    ]
  return seg_uris


def get_partition_paths(seg_uris):
  part_keys = set(
    (uri.dataset, uri.split, uri.segment_id)
    for uri in seg_uris)
  return [
    os.path.join(
      "dataset=" + (dataset or "EMPTY_DATASET"),
      "split=" + (split or "EMPTY_SPLIT"),
      "segment_id=" + (segment_id or "EMPTY_SEGMENT_ID"))
    for (dataset, split, segment_id) in sorted(part_keys)
  ]


def get_partition_path(v):
  if isinstance(v, six.string_types):
    v = URI.from_str(v)
  
  if (not isinstance(v, URI)) and hasattr(v, '__iter__'):
    vs = [vv for vv in v]
    assert len(vs) == 1, \
      "Wanted exactly one value, but have %s" % (v,)
    v = vs[0]

  if not isinstance(v, URI):
    raise ValueError("Don't know what to do with %s" % (v,))
  
  return get_partition_paths([v])[0]


def get_segment_tables_for_uris(seg_uris, spark=None):
  from psegs.table.canonical_factory import CanonicalFactory
  return [
    CanonicalFactory.get_segment_sd_table(seg_uri, spark=spark)
    for seg_uri in seg_uris
  ]
