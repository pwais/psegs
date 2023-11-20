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

BROWSER_DESC = """
browser - A script (and library) providing basic browsing functionality for
available PSegs data

## Examples

Show all known segments:
python3 psegs/browser.py --list-segments

Show all known segments only for 'my-dataset'
python3 psegs/browser.py --list-segments --dataset=my-dataset


"""

from psegs import xform as psx


def create_arg_parser():
  import argparse

  parser = argparse.ArgumentParser(
                    description=BROWSER_DESC,
                    formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument(
    '--list-segments', default=False, action='store_true',
    help='List all known segments')

  psx.configure_arg_parser(parser)

  return parser


def main(args=None):
  import pprint

  if args is None:
    parser = create_arg_parser()
    args = parser.parse_args()
  
  if args.list_segments:
    seg_uris = psx.get_matching_seg_uris(args)
    strs = [str(u) for u in seg_uris]
    pprint.pprint(strs)

if __name__ == '__main__':
  main()
