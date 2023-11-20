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

import attr
import numpy as np

from psegs.datum import datumutils as du
from psegs.util import misc

def _force_shape(shape):
  def converter(arr):
    return np.reshape(np.array(arr), shape)
  return converter


@attr.s(slots=True, eq=False, weakref_slot=False)
class Transform(object):
  """An SE(3) / ROS Transform-like object.  Defaults to the identity
  transform.  Represents a transformation from `dest_frame` from
  `src_frame`.
  """

  rotation = attr.ib(
              default=np.eye(3, 3),
              converter=_force_shape((3, 3)))
  """np.ndarray: A 3x3 rotation matrix"""
  
  translation = attr.ib(
              default=np.zeros((3, 1)),
              converter=_force_shape((3, 1)))
  """np.ndarray: A 3x1 translation matrix"""
  
  src_frame = attr.ib(type=str, default="")
  """str: Name of the source frame"""
  
  dest_frame = attr.ib(type=str, default="")
  """str: Name of the destination frame"""
  
  def __eq__(self, other):
    return misc.attrs_eq(self, other)

  def apply(self, pts):
    """Apply this transform (i.e. right-multiply) to `pts` and return
    tranformed *homogeneous* points."""
    transform = self.get_transformation_matrix()
    pts = du.maybe_make_homogeneous(pts)
    return transform.dot(pts.T)

  @classmethod
  def from_transformation_matrix(cls, RT, **kwargs):
    """Create and return a `Transform` given the 3x4 [R|T] transformation
    matrix `RT`, and forward `kwargs` to `Transform` ctor."""
    R = RT[:3, :3]
    T = RT[:3, 3]
    return Transform(rotation=R, translation=T, **kwargs)

  def get_transformation_matrix(self, homogeneous=False):
    """Return a 3x4 [R|T] transform matrix (or a homogenous
    4x4 only if `homogeneous`)"""
    if homogeneous:
      RT = np.eye(4, 4)
    else:
      RT = np.eye(3, 4)
    RT[:3, :3] = self.rotation
    RT[:3, 3] = self.translation.reshape(3)
    return RT

  def get_inverse(self):
    """Create and return a new transform that is the inverse of this one."""
    return Transform(
      rotation=self.rotation.T,
      translation=self.rotation.T.dot(-self.translation),
      src_frame=self.dest_frame,
      dest_frame=self.src_frame)

  def get_xform(self, src, dest):
    """Return a transform from `src` frame to `dest` frame ; inverses this
    transform if necessary."""
    assert sorted((src, dest)) == sorted((self.src_frame, self.dest_frame)), \
      "Wanted frames (%s, %s) have frames (%s, %s)" % (
        src, dest, self.src_frame, self.dest_frame)
    if src == self.src_frame:
      return copy.deepcopy(self)
    else:
      return self.get_inverse()

  def __getitem__(self, index):
    """Syntactic sugar for `get_xform(src, dest)`.  Example:

    >>> t = Transform(src_frame='f1', dest_frame='f2')
    >>> t['f1', 'f2'] == t
    True

    >>> t['f2', 'f1'] == t.get_inverse()
    True

    >>> t['moof']
    ValueError

    >>> t['a', 'b']
    KeyError
    
    Creates and returns a new Transform instance.
    """
    try:
      src, dest = index
    except Exception as e:
      raise ValueError("Invalid input %s, error %s" % (index, e))
      
    try:
      return self.get_xform(src, dest)
    except Exception as e:
      raise KeyError("Can't get transform for %s, error %s" % (index, e))

  def compose_with(self, other):
    """Right-multiply (chain) this `Transform` with `other` and create a new
    gestault `Transform` that sends points from the source of `other` to
    the destination of this `Tranform`.
    """
    return Transform.from_transformation_matrix(
              self.get_transformation_matrix(homogeneous=True).dot(
                other.get_transformation_matrix(homogeneous=True)),
              src_frame=other.src_frame,
              dest_frame=self.dest_frame)

  def __matmul__(self, other):
    return self.compose_with(other)

  def is_identity(self):
    """Is this the identity transform?"""
    return (
      np.array_equal(self.rotation, np.eye(3, 3)) and
      np.array_equal(self.translation, np.zeros((3, 1))))
