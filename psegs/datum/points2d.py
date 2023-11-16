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

import typing

from oarphpy.spark import CloudpickeledCallable

import attr
import numpy as np

from psegs.util import misc
from psegs.datum.camera_image import CameraImage


@attr.s(slots=True, eq=True, weakref_slot=False)
class Points2D(object):
  """new todo docs
  """

  annotator_name = attr.ib(type=str, default="")
  """str: Name of the source of the points, e.g. "point_detector"; could be
  identical to the topic name or suffix."""

  timestamp = attr.ib(type=int, default=0)
  """int: Timestamp associated with this set of points; often the timestamp of
  `img1`."""

  img1 = attr.ib(default=None, type=CameraImage)
  """CameraImage: The image domain for these points."""

  points_array = attr.ib(type=np.ndarray, default=None)
  """numpy.ndarray: Matches as an n-by-d matrix (where `d` is *at least*
  2, i.e. (img x, img y))."""

  points_factory = attr.ib(
    type=CloudpickeledCallable,
    converter=CloudpickeledCallable,
    default=None)
  """CloudpickeledCallable: A serializable factory function that emits the
  values for `points_array` (if a realized array cannot be provided)"""

  points_colnames = attr.ib(default=['x', 'y'])
  """List[str]: Semantic names for the columns (or dimensions / attributes)
  of the `points_array`.  Typically points are just 2D points, but point data
  can include numeric class_id, score, distance, etc."""

  point_attributes = attr.ib(default=[], type=typing.List[typing.List[str]])
  """List[List[str]]: For each row / point in `points_array`, this member
  provides a *list* of string attributes (e.g. classnames) for the point.
  """

  extra = attr.ib(default={}, type=typing.Dict[str, str])
  """Dict[str, str]: A map for adhoc extra context"""

  def __eq__(self, other):
    return misc.attrs_eq(self, other)

