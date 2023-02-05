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
from psegs.datum.point_cloud import PointCloud
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

  def get_col_idx(self, colname):
    for i in range(len(self.matches_colnames)):
      if self.matches_colnames[i] == colname:
        return i
    raise ValueError(
      "Colname %s not found in %s" % (colname, self.matches_colnames))

  def get_x1y1x2y2_axes(self):
    return [
      self.get_col_idx('x1'),
      self.get_col_idx('y1'),
      self.get_col_idx('x2'),
      self.get_col_idx('y2'),
    ]
  
  def get_other_axes(self):
    x1y1x2y2c = set(['x1', 'y1', 'x2', 'y2'])
    all_c = set(self.matches_colnames)
    other_names = sorted(list(all_c - x1y1x2y2c))
    other_idx = [self.get_col_idx(n) for n in other_names]
    return other_names, other_idx

  def get_x1y1x2y2(self):
    matches = self.get_matches()
    x1y1x2y2 = matches[:, self.get_x1y1x2y2_axes()]
    return x1y1x2y2

  def get_debug_line_image(self):
    return pspl.create_matches_debug_line_image(
              self.img1.image,
              self.img2.image,
              self.get_matches())

  def get_point_cloud_in_world_frame(self):

    import cv2

    P_1 = self.img1.get_P()
    P_2 = self.img2.get_P()
    matches = self.get_matches()

    x1c, y1c, x2c, y2c = self.get_x1y1x2y2_axes()
    other_names, other_idx = self.get_other_axes()
    uv_1 = matches[:, [x1c, y1c]]
    uv_2 = matches[:, [x2c, y2c]]

    xyzh = cv2.triangulatePoints(P_1, P_2, uv_1.T, uv_2.T)
    xyz = xyzh.T.copy()
    xyz = xyz[:, :3] / xyz[:, (-1,)]

    other_vals = matches[:, other_idx]
    cloud = np.hstack([xyz, other_vals])
    return PointCloud(
              sensor_name=self.matcher_name,
              timestamp=self.timestamp,
              cloud=cloud,
              cloud_colnames = ['x', 'y', 'z'] + other_names)
