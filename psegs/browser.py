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



def main(args=None):
  from psegs.table.sd_db import StampedDatumDB
  StampedDatumDB.show_all_segment_uris()

if __name__ == '__main__':
  main()
