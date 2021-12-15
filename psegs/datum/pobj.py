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

from oarphpy.spark import CloudpickeledCallable

from psegs.datum import datumutils as du



@attr.s(slots=True, eq=True, weakref_slot=False)
class PUnion(object):
  """A union type representing the possible values for attributes of 
  `PObj` (see below)."""

  v_int     = attr.ib(type=int, default=0)
  """int: A integer value"""
  
  v_float   = attr.ib(type=float, default=0.)
  """float: A floating point value"""

  v_str     = attr.ib(type=str, default='')
  """str: A string blob value"""
  
  v_bytes   = attr.ib(type=bytearray, default=bytearray())
  """bytearray: A binary blob value"""

  v_arr     = attr.ib(type=np.ndarray, default=None)
  """numpy.ndarray: An array value"""

  v_factory = attr.ib(
                type=CloudpickeledCallable,
                converter=CloudpickeledCallable,
                default=None)
  """CloudpickeledCallable: A serializable factory function that returns
    some object."""

  v_method  = attr.ib(
                type=CloudpickeledCallable,
                converter=CloudpickeledCallable,
                default=None)
  """CloudpickeledCallable: A serializable unary function that accepts this
    PUnion instance as input and returns some object."""


  # Helpers for HTML-based visualization / report

  @classmethod
  def create_html_obj(cls, html='', html_factory=None, html_method=None):
    if html is not '':
      return cls(v_str=html)
    elif html_factory is not None:
      return cls(v_factory=html_factory)
    elif html_method is not None:
      return cls(v_method=html_method)
    else:
      raise ValueError("Don't know how to HTML-ize")

  def to_html_value(self):
    if self.v_str is not '':
      return self.v_str
    elif self.v_factory is not None:
      return self.v_factory()
    elif self.v_method is not None:
      return self.v_method(self)



@attr.s(slots=True, eq=True, weakref_slot=False)
class PObj(object):
  """A generic perception (pythonic) object container.  Use this datum
  for storing debug material, visualizations, and other hacks before
  promoting to a formal datum (or modifying an existing one).

  Good PObj material:
   * Evaluation metric metadata (e.g. matched bounding boxes or cuboids)--
      PSegs does not yet include any evaluation routines or metric
      implementations.
   * Rendered (or lazily-rendered) visualizations / debug content-- use PObj
      to associate this content with existing datums / segments.

  Bad PObj material:
   * Labels / Predictions-- Use Cuboid, BBox2D, etc., or add a new type.
   * Binary blob sensor data--  Use CameraImage, PointCloud, etc., perhaps
      using a factory function for dynamic I/O.
  """

  tag = attr.ib(type=str, default='')

  attr_name_to_value = attr.ib(default={}, type=typing.Dict[str, PUnion])

  extra = attr.ib(default={}, type=typing.Dict[str, str])
  """Dict[str, str]: A map for adhoc extra context"""

  @classmethod  
  def create_html(cls, html='', html_factory=None, html_method=None):
    o = PUnion.create_html_obj(
          html=html, html_method=html_method, html_factory=html_factory)
    return cls(tag='HTML', attr_name_to_value={'__html__': o})

  def to_html(self):
    if '__html__' in self.attr_name_to_value:
      o = self.attr_name_to_value['__html__']
      return o.to_html_value()
    else:
      import tabulate
      table = [
        ['tag', self.tag]
      ]
      table += [
        [attr, du.to_preformatted(o)]
        for attr, o in sorted(self.attr_name_to_value.items())
      ]
      table += [
        ['extra.' + k, v]
        for k, v in self.extra.items()
      ]
      return tabulate.tabulate(table, tablefmt='html')
