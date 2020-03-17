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

import attr
import numpy as np
import pytest

from psegs.util import misc

def test_utils_attrs_eq_no_numpy():

  @attr.s(eq=False)
  class NoNumpy(object):
    x = attr.ib(default=1)
    def __eq__(self, other):
      return misc.attrs_eq(self, other)

  assert NoNumpy() == NoNumpy()
  assert NoNumpy(x=2) != NoNumpy(x=3)
  with pytest.raises(TypeError):
    assert NoNumpy() != object()


def test_utils_attrs_eq_one_numpy():

  @attr.s
  class OneNumpy(object):
    x = attr.ib(default=1)
    y = attr.ib(default=np.ones((1, 1)))
    def __eq__(self, other):
      return misc.attrs_eq(self, other)
  
  assert OneNumpy() == OneNumpy()
  assert OneNumpy(x=2) != OneNumpy(x=3)
  assert OneNumpy(y=np.zeros((1,))) != OneNumpy(y=np.ones((1,)))


def test_utils_get_png_wh():
  with pytest.raises(ValueError):
    misc.get_png_wh(bytearray(b''))
  
  from oarphpy import util as oputil
  img = np.zeros((20, 25, 3))
  png_bytes = oputil.to_png_bytes(img)

  w, h = misc.get_png_wh(bytearray(png_bytes))
  assert (h, w) == (20, 25)
