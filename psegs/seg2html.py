#!/usr/bin/env python3
# vim: tabstop=2 shiftwidth=2 expandtab

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


DESC = """
seg2html.py -- A library module of tools (as well as a script) to convert
PSegs segments to HTML visualizations.  Run this script in the PSegs dockerized
environment; FMI see ./psegs-util --help in the PSegs project.

## Example

python3 psegs/seg2html.py \
  --segment-id=charuco-lowres-test \
  --out-dir=./my_html_viz

This will render only the segment named `charuco-lowres-test` to HTML and
put rendered assets in ./my_html_viz .

"""

import os

from psegs import xform as psx


def save_htmls(
      sdtables=None,
      seg_uris=None,
      out_dir='/tmp',
      partition_by_segment=False):

  from pathlib import Path
  
  from oarphpy import util as oputil
  from tqdm import tqdm
  import six
  
  from psegs import table
  from psegs import datum
  from psegs import util

  out_dir = Path(out_dir)

  segs = seg_uris or sdtables
  assert segs, "Need either Segment URIs or StampedDatumTables"

  util.log.info(
    f"Saving to {out_dir}. Have {len(segs)} segments to HTMLize ...")
  segs = segs[155:]
  pbar = tqdm(segs)
  total_bytes = 0
  for seg in pbar:
    ## Fetch segment data
    if not isinstance(seg, table.StampedDatumTable):
      if isinstance(seg, six.string_types):
        seg_uri = datum.URI.from_str(seg)
      else:
        seg_uri = seg
      sdts = psx.get_segment_tables_for_uris([seg_uri])
      sdt = sdts[0]
    else:
      sdt = seg
    
    ## Decide where to output
    if partition_by_segment:
      partition_path = psx.get_partition_path(sdt.get_all_segment_uris())
      dest = out_dir / partition_path / "rich_viz.html"
    else:
      seg_uris = sdt.get_all_segment_uris()
      assert len(seg_uris) == 1, \
        "Table {sdt} has data for more than one segment: {seg_uris}"
      seg_uri = seg_uris[0]
      fname = '.'.join((
                    seg_uri.dataset or "anon_dataset",
                    seg_uri.split or "anon_split",
                    seg_uri.segment_id or "anon_segment_id"))
      fname = fname + '.html'
      dest = out_dir / fname

    ## Render!
    html = sdt.to_rich_html()
    total_bytes += len(html)
      
    oputil.mkdir(dest.parent)
    with open(dest, 'w') as f:
      f.write(html)

    pbar.set_description(f"Wrote total {(1e-6*total_bytes):.2f} MBytes")


def create_arg_parser():
  import argparse

  parser = argparse.ArgumentParser(
                    description=DESC,
                    formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
    '--out-dir', default=os.path.abspath('./my_html_viz'),
    help='Place all computed assets in this directory [default %(default)s].')
  parser.add_argument(
    '--partition-by-segment', default=False, action='store_true',
    help='Save rendered assets in a directory tree partitioned by dataset, '
         'split, and segment_id (as is done for PSegs StampedDatumTables).')

  psx.configure_arg_parser(parser)

  return parser

def main(args=None):
  if args is None:
    parser = create_arg_parser()
    args = parser.parse_args()
  
  seg_uris = psx.get_matching_seg_uris(args)
  save_htmls(
    seg_uris=seg_uris,
    out_dir=args.out_dir,
    partition_by_segment=args.partition_by_segment)

if __name__ == '__main__':
  main()
