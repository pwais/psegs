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

import typing

import attr
import numpy as np


@attr.s(slots=True, eq=True, weakref_slot=False)
class BBox2D(object):
  """An object in an image; in particular, an (ideally amodal) bounding box
  surrounding the object.  May include additional context.
  
  Note:
    Why this representation instead of ... ?  This class is:
     * Agnostic to origin (i.e. lower left or upper left)
     * Supports boxes of area Zero (width and height of zero)
     * Inclusivity / exclusivity of bounds are unambiguous
     * The image size is optionally associated, making it easy to convert to
       'relative' coordinates (i.e. `{x, y} in [0, 1]`) for the box encoders
       in detectors like FasterRCNN, SSD, Retinanet, etc.
     * We provide interop with `((x1, y1), (x2, y2))` corner encoding.
  """
  
  # NB: We explicitly disable `validator`s so that the user may temporarily
  # use floats

  x = attr.ib(type=int, default=0, validator=None)
  """int: Base x coordinate in pixels."""
  
  y = attr.ib(type=int, default=0, validator=None)
  """int: Base y coordinate in pixels."""
  
  width = attr.ib(type=int, default=0, validator=None)
  """int: Width of box in pixels."""

  height = attr.ib(type=int, default=0, validator=None)
  """int: Height of box in pixels."""

  im_width = attr.ib(type=int, default=0, validator=None)
  """int, optional: Width of enclosing image"""

  im_height = attr.ib(type=int, default=0, validator=None)
  """int, optional: Height of enclosing image"""
  
  category_name = attr.ib(type=str, default="")
  """str, optional: Class associated with this bounding box."""

  extra = attr.ib(default={}, type=typing.Dict[str, str])
  """Dict[str, str]: A map for adhoc extra context"""

  def update(self, **kwargs):
    """Update attributes of this `BBox2D` as specified in `kwargs`"""
    for k in self.__slots__:
      if k in kwargs:
        setattr(self, k, kwargs[k])

  @staticmethod
  def of_size(width, height):
    """Create a `BBox2D` of `width` by `height`"""
    return BBox(
            x=0, y=0,
            width=width, height=height,
            im_width=width, im_height=height)

  @staticmethod
  def from_x1_y1_x2_y2(x1, y1, x2, y2):
    """Create a `BBox2D` from corners `(x1, y1)` and `(x2, y2)` (inclusive)"""
    b = BBox2D()
    b.set_x1_y1_x2_y2(x1, y1, x2, y2)
    return b

  def set_x1_y1_x2_y2(self, x1, y1, x2, y2):
    """Update this `BBox2D` to have corners `(x1, y1)` and `(x2, y2)`
    (inclusive)"""
    self.update(x=x1, y=y1, width=x2 - x1 + 1, height=y2 - y1 + 1)

  def get_x1_y1_x2_y2(self):
    """Get the corners `(x1, y1)` and `(x2, y2)` (inclusive) of this `BBox2D`"""
    return self.x, self.y, self.x + self.width - 1, self.y + self.height - 1

  def get_r1_c1_r2_r2(self):
    """Get the row-major corners `(y1, x1)` and `(y2, x2)` (inclusive) of
    this `BBox2D`"""
    return self.y, self.x, self.y + self.height - 1, self.x + self.width - 1

  def get_x1_y1(self):
    """Return the origin"""
    return self.x, self.y

  def get_fractional_xmin_ymin_xmax_ymax(self, clip=True):
    """Get the corners `(x1, y1)` and `(x2, y2)` (inclusive) of this
    `BBox2D` in image-relative coordinates; i.e. each corner is scaled
    to [0, 1] based upon image size.  Forbid off-image corners only if
    `clip`."""
    xmin = float(self.x) / self.im_width
    ymin = float(self.y) / self.im_height
    xmax = float(self.x + self.width) / self.im_width
    ymax = float(self.y + self.height) / self.im_height
    if clip:
      xmin, ymin, xmax, ymax = \
        map(lambda x: float(np.clip(x, 0, 1)), \
          (xmin, ymin, xmax, ymax))
    return xmin, ymin, xmax, ymax

  def add_padding(self, *args):
    """Extrude this `BBox2D` with the given padding: either a single value
    in pixels or a `(pad_x, pad_y)` tuple."""
    if len(args) == 1:
      px, py = args[0], args[0]
    elif len(args) == 2:
      px, py = args[0], args[1]
    else:
      raise ValueError(len(args))
    self.x -= px
    self.y -= py
    self.width += 2 * px
    self.height += 2 * py

  def is_full_image(self):
    """Does this `BBox2D` cover the whole image?"""
    return (
      self.x == 0 and
      self.y == 0 and
      self.width == self.im_width and
      self.height == self.im_height)

  def get_corners(self):
    """Return all four corners, starting from the origin, in CCW order."""
    return (
      (self.x, self.y),
      (self.x + self.width, self.y),
      (self.x + self.width, self.y + self.height),
      (self.x, self.y + self.height),
    )

  def get_num_onscreen_corners(self):
    """Return the number (max four) of corners that are on the image."""
    return sum(
      1 for x, y in self.get_corners()
      if (0 <= x < self.im_width) and (0 <= y < self.im_height))

  def quantize(self):
    """Creating a `BBox2D` with float values is technically OK; use this
    method to round to integer values in-place."""
    ATTRS = ('x', 'y', 'width', 'height', 'im_width', 'im_height')
    def quantize(v):
      return int(round(v)) if v is not None else v
    for attr in ATTRS:
      setattr(self, attr, quantize(getattr(self, attr)))

  def clamp_to_screen(self):
    """Clamp any out-of-image corners to edges of the image."""
    def clip_and_norm(v, max_v):
      return int(np.clip(v, 0, max_v).round())
    
    x1, y1, x2, y2 = self.get_x1_y1_x2_y2()
    x1 = clip_and_norm(x1, self.im_width - 1)
    y1 = clip_and_norm(y1, self.im_height - 1)
    x2 = clip_and_norm(x2, self.im_width - 1)
    y2 = clip_and_norm(y2, self.im_height - 1)
    self.set_x1_y1_x2_y2(x1, y1, x2, y2)
    
  def get_intersection_with(self, other):
    """Create a new `BBox2D` containing the intersection with `other`."""
    x1, y1, x2, y2 = self.get_x1_y1_x2_y2()
    ox1, oy1, ox2, oy2 = other.get_x1_y1_x2_y2()
    ix1 = max(x1, ox1)
    ix2 = min(x2, ox2)
    iy1 = max(y1, oy1)
    iy2 = min(y2, oy2)
    
    import copy
    intersection = copy.deepcopy(self)
    intersection.set_x1_y1_x2_y2(ix1, iy1, ix2, iy2)
    return intersection

  def get_union_with(self, other):
    """Create a new `BBox2D` containing the union with `other`."""
    x1, y1, x2, y2 = self.get_x1_y1_x2_y2()
    ox1, oy1, ox2, oy2 = other.get_x1_y1_x2_y2()
    ux1 = min(x1, ox1)
    ux2 = max(x2, ox2)
    uy1 = min(y1, oy1)
    uy2 = max(y2, oy2)
    
    import copy
    union = copy.deepcopy(self)
    union.set_x1_y1_x2_y2(ux1, uy1, ux2, uy2)
    return union

  def overlaps_with(self, other):
    """Does this `BBox2D` overlap with `other`."""
    # TODO: faster
    return self.get_intersection_with(other).get_area() > 0

  def get_area(self):
    """Area in square pixels"""
    return self.width * self.height

  def translate(self, *args):
    """Move the origin of this `BBox2D` by the given `(x, y)` value;
    either a tuple or a `numpy.ndarray`."""
    if len(args) == 1:
      x, y = args[0].tolist()
    else:
      x, y = args
    self.x += x
    self.y += y

  def get_crop(self, img):
    """Given the `numpy` array image `img`, return a crop based on this
    `BBox2D`."""
    c, r, w, h = self.x, self.y, self.width, self.height
    return img[r:r+h, c:c+w, :]

  def draw_in_image(self, img, color=None, thickness=2, category=None):
    """Draw a bounding box in `np_image`.

    Args:
      img (numpy.ndarray): Draw in this image.
      color (tuple): an (r, g, b) tuple specifying the border color; by
        default use a category-determined color.
      thickness (int): thickness of the line in pixels.
      category (str): override the label text drawn for this box; otherwise
        use the `category` attribute; omit label text if either is empty
    """

    assert self.im_height == img.shape[0], (self.im_height, img.shape)
    assert self.im_width == img.shape[1], (self.im_width, img.shape)

    category = category or self.category_name
    if not color:
      from oarphpy.plotting import hash_to_rbg
      color = hash_to_rbg(category)

    from psegs.util.plotting import draw_bbox_in_image
    draw_bbox_in_image(
      img, self, color=color, thickness=thickness, label_txt=category)
