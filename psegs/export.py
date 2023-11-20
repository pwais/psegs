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

from psegs import xform as psx


def create_arg_parser():
  import argparse

  parser = argparse.ArgumentParser(
                    description=DESC,
                    formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument(
    '--viz-all-segments', default=False, action='store_true',
    help='Render all available segments')

  psx.configure_arg_parser(parser)

  return parser

def main(args=None):
  if args is None:
    parser = create_arg_parser()
    args = parser.parse_args()
  
  
  # hacks for now
  



if __name__ == '__main__':
  main(args=None)
