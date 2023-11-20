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

import pytest
import numpy as np

from psegs.datum.transform import Transform


def test_transform_force_shape():
  t = Transform(
        rotation=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        translation=np.array([1, 2, 3]))
  np.testing.assert_equal(
                t.rotation,
                np.array([
                  [1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]))
  np.testing.assert_equal(
                t.translation,
                np.array([
                  [1],
                  [2],
                  [3],
                ]))


def test_transform_apply_identity():
  t = Transform()
  pts = np.eye(3, 3)

  pts_out = t.apply(pts)
  np.testing.assert_equal(pts_out, pts)


def test_transform_apply_translation():
  t = Transform(translation=[1, 0, 0])
  pts = np.eye(3, 3)
  pts_out = t.apply(pts)
  xhat = np.array([[1, 0, 0]]).T
  np.testing.assert_equal(pts_out, pts + xhat)


def test_transform_apply_rotation():
  from scipy.spatial.transform import Rotation as R
  import math

  # A yaw of pi/4
  rot = R.from_euler('zxy', [math.pi / 4, 0, 0]).as_matrix()
  
  t = Transform(rotation=rot)
  pts = np.eye(3, 3)
  pts_out = t.apply(pts)
  
  np.testing.assert_almost_equal(
    pts_out,
    np.array([
      [math.sqrt(2) / 2, -math.sqrt(2) / 2, 0],
      [math.sqrt(2) / 2,  math.sqrt(2) / 2, 0],
      [               0,                 0, 1],
    ]))


def test_transform_get_xform():
  t = Transform(translation=[1, 0, 0], src_frame='f1', dest_frame='f2')
  assert t == t.get_xform('f1', 'f2')
  assert t.get_inverse() == t.get_xform('f2', 'f1')
  with pytest.raises(AssertionError):
    t.get_xform('a', 'b')
  
  assert t['f1', 'f2'] == t
  assert t['f2', 'f1'] == t.get_inverse()
  with pytest.raises(ValueError):
    t['moof']
  with pytest.raises(KeyError):
    t['a', 'b']


def test_transform_chained():

  t1 = Transform(
    translation=[1., 1., 1.], src_frame='t1_src', dest_frame='t1_dest')
  t2 = Transform(
    translation=[2., 2., 2.], src_frame='t2_src', dest_frame='t2_dest')
  
  t2_from_t1 = t2 @ t1
  assert \
    t2_from_t1 == Transform(
                    translation=[3., 3., 3.],
                    src_frame='t1_src',
                    dest_frame='t2_dest')

  t1_from_t2 = t1 @ t2
  assert \
    t1_from_t2 == Transform(
                    translation=[3., 3., 3.],
                    src_frame='t2_src',
                    dest_frame='t1_dest')
