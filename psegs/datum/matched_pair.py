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

import attr
import numpy as np

from oarphpy.spark import CloudpickeledCallable

from psegs.datum.camera_image import CameraImage
from psegs.util import plotting as pspl


@attr.s(slots=True, eq=False, weakref_slot=False)
class MatchedPair(object):
  """A pair of `CameraImages` with pixelwise matches"""

  matcher_name = attr.ib(type=str, default='')
  """str: Name of the match source, e.g. SIFT_matches"""

  timestamp = attr.ib(type=int, default=0)
  """int: Timestamp associated with this matched pair; use the timestamp
  of `img1` or `img2` or the wall time of matching."""

  img1 = attr.ib(default=None, type=CameraImage)
  """CameraImage: The first (left, source) image"""

  img2 = attr.ib(default=None, type=CameraImage)
  """CameraImage: The second (right, target) image"""

  matches_array = attr.ib(type=np.ndarray, default=None)
  """numpy.ndarray: Matches as an n-by-d matrix (where `d` is *at least*
  4, i.e. (img1 x, img1 y, img2 x, img2 y))."""

  matches_factory = attr.ib(
    type=CloudpickeledCallable,
    converter=CloudpickeledCallable,
    default=None)
  """CloudpickeledCallable: A serializable factory function that emits the
  values for `matches_array` (if a realized array cannot be provided)"""

  matches_colnames = attr.ib(default=['x1', 'y1', 'x2', 'y2'])
  """List[str]: Semantic names for the columns (or dimensions / attributes)
  of the `matches_array`.  Typically matches are just 2D point pairs, but
  match data can include confidence, occlusion state, track ID, and/or 
  other data."""

  extra = attr.ib(default={}, type=typing.Dict[str, str])
  """Dict[str, str]: A map for adhoc extra context"""

  def get_matches(self):
    if self.matches_array is not None:
      return self.matches_array
    elif self.matches_factory != CloudpickeledCallable.empty():
      return self.matches_factory()
    else:
      raise ValueError("No matches data!")

  def to_point_cloud(self):
    print('todo')

  def get_debug_line_image(self):
    return pspl.create_matches_debug_line_image(
              self.img1.image,
              self.img2.image,
              self.get_matches())
