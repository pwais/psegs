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

import copy

import numpy as np

from psegs import datum
from psegs.datum import datumutils as du


def test_maybe_make_homogeneous():
  np.testing.assert_equal(
    du.maybe_make_homogeneous(np.array([[0, 0, 0]])),
    np.array([[0, 0, 0, 1]]))
  
  np.testing.assert_equal(
    du.maybe_make_homogeneous(np.array([[0, 0, 0, 1]])),
    np.array([[0, 0, 0, 1]]))


def test_datum_to_diffable_tree():

  sd1 = copy.deepcopy(datum.STAMPED_DATUM_PROTO)
  sd2 = copy.deepcopy(datum.STAMPED_DATUM_PROTO)

  tree1 = du.datum_to_diffable_tree(sd1)
  tree2 = du.datum_to_diffable_tree(sd2)

  assert tree1 == tree2


  sd1 = copy.deepcopy(datum.STAMPED_DATUM_PROTO)
  sd1.camera_image = None
  sd2 = copy.deepcopy(datum.STAMPED_DATUM_PROTO)

  tree1 = du.datum_to_diffable_tree(sd1)
  tree2 = du.datum_to_diffable_tree(sd2)

  assert tree1 != tree2

  difftxt = du.get_datum_diff_string(sd1, sd2)
  assert "-  'camera_image': None," in difftxt
  assert "+  'camera_image': {'K':" in difftxt


  sd1 = copy.deepcopy(datum.STAMPED_DATUM_PROTO)
  sd1.uri.dataset = 'foo'
  sd2 = copy.deepcopy(datum.STAMPED_DATUM_PROTO)

  assert tree1 != tree2

  difftxt = du.get_datum_diff_string(sd2, sd1)
  assert "-          'dataset': ''" in difftxt
  assert "+          'dataset': 'foo'" in difftxt

