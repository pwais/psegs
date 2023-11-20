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

from psegs.datum.bbox2d import BBox2D

# BETTERME ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def test_serialization():
  b = BBox2D(x=1, y=2, width=3)

  import pickle
  s = pickle.dumps(b)
  bb = pickle.loads(s)

  assert b == bb


def test_x1_y1_x2_y2():
  b1 = BBox2D.from_x1_y1_x2_y2(0, 0, 9, 9) # Inclusive!!
  assert b1 == BBox2D(x=0, y=0, width=10, height=10)
  
  b2 = BBox2D(x=1, y=0, width=10, height=10)
  assert b2.get_x1_y1_x2_y2() == (1, 0, 10, 9)
  assert b2.get_r1_c1_r2_r2() == (0, 1, 9, 10)


def test_add_padding():
  b1 = BBox2D(x=0, y=0, width=1, height=1)
  b1.add_padding(1)
  assert b1 == BBox2D(x=-1, y=-1, width=1 + 1 + 1, height=1 + 1 + 1)

  b2 = BBox2D(x=0, y=0, width=1, height=1)
  b2.add_padding(1, 2)
  assert b2 == BBox2D(x=-1, y=-2, width=3, height=1 + 2 + 2)
