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
import tempfile
from pathlib import Path

import attr

DEFAULT_DATA_ROOT = Path('/opt/psegs/dataroot')
DEFAULT_EXTERNAL_TEST_FIXTURES_ROOT = Path('/opt/psegs/external_test_fixtures')
DEFAULT_EXT_DATA_ROOT = Path('/opt/psegs/ext_data')
DEFAULT_TEMP_DIR = Path(tempfile.gettempdir()) / 'psegs_temp'

@attr.s(eq=True)
class ProjConf(object):
  """A singleton that holds project-specific configuration (with defaults
  sensible for testing)"""

  DATA_ROOT = attr.ib(type=Path, default=DEFAULT_DATA_ROOT, converter=Path)
  """Path: root (perhaps local) for all input and output data."""

  EXTERNAL_TEST_FIXTURES_ROOT = attr.ib(
    type=Path, default=DEFAULT_EXTERNAL_TEST_FIXTURES_ROOT, converter=Path)
  """Path: root for externally-hosted test fixtures; these are required for
  some dataset-specific tests to run."""

  EXT_DATA_ROOT = attr.ib(
    type=Path, default=DEFAULT_EXT_DATA_ROOT, converter=Path)
  """Path: root for extention data (e.g. meta-labels mined from standard
  datasets).  These files are required for some specific features."""

  SD_TABLE_ROOT = attr.ib(
    type=Path,
    default=DEFAULT_DATA_ROOT / 'stamped_datum',
    converter=Path)
  """Path: store :class:`~psegs.table.sd_table.StampedDatumTableBase` data
  table(s) here.  Putting all tables in the same root 'directory' makes it
  easier to (virtually) concatenate them."""

  PS_TEMP = attr.ib(type=Path, default=DEFAULT_TEMP_DIR, converter=Path)
  """Path: cache any PSegs-specific files in this temp directory; co-locate
  them to make introspection and deletion easier."""


C = ProjConf()
