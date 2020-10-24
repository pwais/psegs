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

import numpy as np

from psegs.datum.cuboid import Cuboid
from psegs.datum.transform import Transform


def test_cuboid_box3d():
  c1 = Cuboid(
          length_meters=2,
          width_meters=2,
          height_meters=2,
          obj_from_ego=
            Transform(
              translation=[10, 0, 0],
              src_frame='ego',
              dest_frame='obj'))
  np.testing.assert_equal(
    c1.get_box3d(),
    np.array([
      [11,   1,   1],  # Front face
      [11,  -1,   1],
      [11,  -1,  -1],
      [11,   1,  -1],
      [ 9,   1,   1],  # Back face
      [ 9,  -1,   1],
      [ 9,  -1,  -1],
      [ 9,   1,  -1]
    ]))
  


def test_cuboid_union_merge():

  c1 = Cuboid(
          track_id='c1',
          category_name='c1',
          ps_category='c1',
          length_meters=2,
          width_meters=2,
          height_meters=2,
          obj_from_ego=
            Transform(
              translation=[10, 0, 0],
              src_frame='ego',
              dest_frame='obj'))

  c2 = Cuboid(
          track_id='c2',
          category_name='c2',
          ps_category='c2',
          length_meters=2,
          width_meters=2,
          height_meters=2,
          obj_from_ego=
            Transform(
              translation=[12, 0, 0],
              src_frame='ego',
              dest_frame='obj'))

  actual = Cuboid.get_merged(c1, c2, mode='union')

  expected_union = Cuboid(
          track_id='c1-union-c2',
          category_name='c1',
          ps_category='c1',
          length_meters=4,
          width_meters=2,
          height_meters=2,
          obj_from_ego=
            Transform(
              translation=[11, 0, 0],
              src_frame='ego',
              dest_frame='obj'))

  assert actual == expected_union


def test_cuboid_union_interpolate():

  c1 = Cuboid(
          track_id='c1',
          category_name='c1',
          ps_category='c1',
          length_meters=2,
          width_meters=2,
          height_meters=2,
          obj_from_ego=
            Transform(
              translation=[10, 0, 0],
              src_frame='ego',
              dest_frame='obj'))

  c2 = Cuboid(
          track_id='c2',
          category_name='c2',
          ps_category='c2',
          length_meters=2,
          width_meters=2,
          height_meters=2,
          obj_from_ego=
            Transform(
              translation=[12, 0, 0],
              src_frame='ego',
              dest_frame='obj'))

  actual = Cuboid.get_merged(c1, c2, mode='interpolate', alpha=0.5)

  expected_interp = Cuboid(
          track_id='c1-interpolate-c2',
          category_name='c1',
          ps_category='c1',
          length_meters=2,
          width_meters=2,
          height_meters=2,
          obj_from_ego=
            Transform(
              translation=[11, 0, 0],
              src_frame='ego',
              dest_frame='obj'))

  assert actual == expected_interp


def test_get_interpolated():
  for _ in range(10):
    print('todo test_get_interpolated')
