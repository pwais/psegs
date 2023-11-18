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


@attr.s(slots=True, eq=False, weakref_slot=False)
class Points2D(object):
  """new todo docs
  """

  annotator_name = attr.ib(type=str, default="")
  """str: Name of the source of the points, e.g. "point_detector"; could be
  identical to the topic name or suffix."""

  timestamp = attr.ib(type=int, default=0)
  """int: Timestamp associated with this set of points; often the timestamp of
  `img`."""

  img = attr.ib(default=None, type=CameraImage)
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

  point_attributes = attr.ib(default=[], type=typing.List[str])
  """List[str]: For each row / point in `points_array`, this member
  provides string attributes (e.g. classnames) for the point.
  """

  extra = attr.ib(default={}, type=typing.Dict[str, str])
  """Dict[str, str]: A map for adhoc extra context"""

  def __eq__(self, other):
    return misc.attrs_eq(self, other)

  def get_col_idx(self, colname):
    for i in range(len(self.points_colnames)):
      if self.points_colnames[i] == colname:
        return i
    raise ValueError(
      "Colname %s not found in %s" % (colname, self.points_colnames))

  def get_xy_axes(self):
    return [
      self.get_col_idx('x'),
      self.get_col_idx('y'),
    ]
  
  def get_other_axes(self):
    xyc = set(['x', 'y'])
    all_c = set(self.points_colnames)
    other_names = sorted(list(all_c - xyc))
    other_idx = [self.get_col_idx(n) for n in other_names]
    return other_names, other_idx

  def get_points(self):
    if self.points_array is not None:
      return self.points_array
    elif self.points_factory != CloudpickeledCallable.empty():
      return self.points_factory()
    else:
      raise ValueError("No points data!")

  def get_xy(self):
    points = self.get_points()
    xy = points[:, self.get_xy_axes()]
    return xy

  def get_xy_extra(self):
    matches = self.get_points()
    other_names, other_idx = self.get_other_axes()
    cols = self.get_xy_axes() + other_idx
    xy_extra = matches[:, cols]
    return xy_extra

  def get_debug_points_image(self, should_color_with_gid_col=True):
    from psegs.util import plotting as pspl
    from oarphpy.plotting import hash_to_rbg

    pts = self.get_xy()
    colors = None
    if len(self.points_colnames) > 2:
      colordata = None
      if should_color_with_gid_col:
        for i, colname in enumerate(self.points_colnames):
          if colname.endswith('gid'):
            colordata = self.get_xy_extra()
            colordata = colordata[:, i]    
      if colordata is None:
        colordata = self.get_xy_extra()
        colordata = colordata[:, 2:]

      colors = np.array([
        hash_to_rbg(r) for r in colordata
      ])

    if self.img is not None:
      debug_image = self.img.image.copy()
    else:
      w = int(pts[:, 0].max()) + 1
      h = int(pts[:, 1].max()) + 1
      debug_image = np.zeros((h, w, 3), dtype='uint8')
    
    pspl.draw_colored_2dpts_in_image(debug_image, pts, user_colors=colors)
    return debug_image
