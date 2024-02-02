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
import itertools
import os
import typing

import attr
import six
import six.moves.urllib.parse


@attr.s(slots=True, eq=True, weakref_slot=False)
class DatumSelection(object):
  """A single topic-time pair to indicate one of several datums within a
  :class:`~psegs.datum.frame.Frame`"""
  
  topic = attr.ib(default='', type='str')
  """str: Name for a series of messages, e.g. '/ego_pose'"""

  timestamp = attr.ib(default=0, converter=int, type='int')
  """int: Some integer timestamp; most of `psegs` assumes a Unix time in
  nanoseconds."""

  @classmethod
  def selections_to_string(cls, fdatums):
    assert all(',' not in fd.topic for fd in fdatums), \
      "Need a different topic delimiter ..."
    datum_topic_ts = sorted((fd.topic, str(fd.timestamp)) for fd in fdatums)
      # NB: Sorting is not required but helps with comparing equal URIs
    datums_str = ','.join(itertools.chain.from_iterable(datum_topic_ts))
    return datums_str
  
  @classmethod
  def selections_from_value(cls, v):
    if isinstance(v, six.string_types):
      from oarphpy import util as oputil
      toks = v.split(',')
      assert len(toks) % 2 == 0, toks
      dss = [cls(*dtoks) for dtoks in oputil.ichunked(toks, 2)]
      return sorted(dss)
    elif hasattr(v, '__iter__'):
      def to_ds(vv):
        if isinstance(vv, DatumSelection):
          return vv
        elif all(hasattr(vv, attr) for attr in cls.__slots__):
          # It's `cls`-able! E.g. a URI
          attrvals = ((attr, getattr(vv, attr)) for attr in cls.__slots__)
          return DatumSelection(**dict(attrvals))
        elif hasattr(vv, 'uri') and isinstance(vv.uri, URI):
          attrvals = ((attr, getattr(vv.uri, attr)) for attr in cls.__slots__)
          return DatumSelection(**dict(attrvals))
        elif len(vv) == len(cls.__slots__):
          if isinstance(vv, dict):
            return DatumSelection(**vv)
          else:
            return DatumSelection(*vv)
        else:
          raise ValueError("Don't know what to do with %s" % (v,))

      dss = sorted(to_ds(vv) for vv in v)
      return dss
    else:
      raise ValueError("Don't know what to do with %s" % (v,))


@attr.s(slots=True, eq=True,  weakref_slot=False, order=False)
class URI(object):
  """A URI for one specifc datum, or a group of datums (e.g. a 
  :class:`~psegs.datum.frame.Frame`). All parameters are optional; more
  parameters address a more specific piece of all StampedDatum data available.
  """
  
  PREFIX = 'psegs://'
  """The URL prefix or scheme denoting the URL refers to `psegs` data"""


  ## Core Selection

  dataset = attr.ib(default='', type='str')
  """str: E.g. 'kitti'"""
  
  split = attr.ib(default='', type='str')
  """str: E.g. 'train'"""

  segment_id = attr.ib(default='', type='str')
  """str: String identifier for a drive segment, e.g. a UUID"""
  
  timestamp = attr.ib(default=0, converter=int, type='int')
  """int: Some integer timestamp; most of `psegs` assumes a Unix time in
  nanoseconds."""
  
  topic = attr.ib(default='', type='str')
  """str: Name for a series of messages, e.g. '/ego_pose'"""

  extra = attr.ib(default={}, type=typing.Dict[str, str])
  """Dict[str, str]: A map for adhoc extra context"""


  ## Extended Selection

  # # TODO dump this? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # track_id = attr.ib(default='', type='str')
  # """str: A string identifier of a specific track, e.g. a UUID"""

  sel_datums = attr.ib(
                default=[], type=typing.List[DatumSelection],
                converter=DatumSelection.selections_from_value)
  """List[DatumSelection]: A sequence of (topic, time) pairs that help
  encode a :class:`~psegs.datum.frame.Frame` -based `URI`."""
  

  def to_segment_uri(self):
    cls = self.__class__
    return cls(
            dataset=self.dataset,
            split=self.split,
            segment_id=self.segment_id)
  
  def soft_matches_segment_of(self, other):
    """Return true only if this URI soft-matches the segment_id, dataset,
    and/or split of URI `other`.  A soft match allows us to wildcard ('*')
    the one or all of the three components that define a distinct segment.
    Note that while most datasets have globally distinct segment_id
    names, this contraint isn't guaranteed; e.g. a segment_id name might 
    appear in more than one split (by error or by intention).
    
    For example:
      psegs://segment_id=s soft matches with psegs://segment_id=s&dataset=d
    BUT
      psegs://dataset=d&segment_id=s 
        DOES NOT
      soft match with psegs://segment_id=s 

    """
    if isinstance(other, six.string_types):
      other = self.__class__.from_str(other)

    return (
      ((not self.segment_id) or (self.segment_id == other.segment_id)) and
      ((not self.dataset) or (self.dataset == other.dataset)) and
      ((not self.split) or (self.split == other.split)))

  def as_tuple(self):
    def to_tokens(k, v):
      if bool(v):
        if k == 'extra':
          for ek, ev in sorted(v.items()):
            yield ('extra.%s' % ek, ev)
        else:
          yield (k, v)

    toks = itertools.chain.from_iterable(
      to_tokens(f.name, getattr(self, f.name))
      for f in attr.fields(self.__class__))
    return tuple(toks)

  def to_str(self):
    def encode(k, v):
      if k == 'sel_datums':
        return DatumSelection.selections_to_string(v)
      else:
        return v
    tup = self.as_tuple()
    toks = ('%s=%s' % (k, encode(k, v)) for k, v in tup)
    return '%s%s' % (self.PREFIX, '&'.join(toks))
  
  def to_urlsafe_str(self):
    return six.moves.urllib.parse.quote_plus(self.to_str())

  def to_segment_partition_relpath(self):
    return os.path.join(
      "dataset=%s" % (self.dataset or 'EMPTY_DATASET'),
      "split=%s" % (self.split or 'EMPTY_SPLIT'),
      "segment_id=%s" % (self.segment_id or 'EMPTY_SEGMENT_ID'))

  def __str__(self):
    return self.to_str()
  
  # def __hash__(self):
  #   # NB: read attrs warnings: https://www.attrs.org/en/stable/hashing.html#fn1
  #   # Consequences here:
  #   # * We get URIs to hash like their tuple/string encoding, which is what
  #   #    we want.
  #   # * We do this instead of frozen=True so that URIs can be updated in-place
  #   #    (e.g. via oarphpy.spark.RowAdapter.from_row(), or updating `.extra`).
  #   #    Furthermore, frozen=True doesn't prevent updates inside mutable 
  #   #    members anyways.
  #   # * URIs will probaby never be mutated *after* being inserted into a 
  #   #     container, thus the update-causes-silent-hash-bugs issue is likely
  #   #     a rare edge case.
  #   return hash(self.as_tuple())

  # def __repr__(self):
  #   kvs = ((attr, getattr(self, attr)) for attr in self.__slots__)
  #   kwargs_str = ', '.join('%s=%s' % (k, repr(v)) for k, v in kvs)
  #   return 'URI(%s)' % kwargs_str

  # def __eq__(self, other):
  #   if type(other) is type(self):
  #     return all(
  #       getattr(self, attr) == getattr(other, attr)
  #       for attr in self.__slots__)
  #   return False

  def __lt__(self, other):
    assert type(other) is type(self)
    return self.as_tuple() < other.as_tuple()
    
    # import pdb; pdb.set_trace()
    # def a_extra_less_than_b_extra(a, b):
    #   # extra (dicts) are not comparable, so we need to handle them specially
    #   assert self.__slots__[-1] == 'extra', "Schema changed?"
    #   a_extra = a[-1]
    #   b_extra = b[-1]
    #   return sorted(a_extra.items()) < sorted(b_extra.items())
    
    # return (
    #   (self_t[:-1] < other_t[:-1]) or 
    #     a_extra_less_than_b_extra(self_t, other_t))


  # def __hash__(self): # breaks equality of containers ... ~~~~~~~~~~~~~~~~~~~~~~``
  #   return hash(self.as_tuple())

  def update(self, **kwargs):
    """Override this instance in-place with all values specified in `kwargs`.
    (Ignores invalid values)."""
    for k in self.__slots__:
      if k in kwargs:
        v = kwargs[k]
        if k == 'sel_datums':
          v = DatumSelection.selections_from_value(v)
        setattr(self, k, v)
  
  def replaced(self, **kwargs):
    """Create and return a copy with all values updated to those specified in
    `kwargs`.  Similar to `namedtuple._replace()`.  Ignores invalid keys in
    `kwargs`.  Useful for constructing a derivative URI given a base URI."""
    uri = copy.deepcopy(self)
    uri.update(**kwargs)
    return uri



  # def set_crop(self, bbox):#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #   self.update(
  #     crop_x=bbox.x,
  #     crop_y=bbox.y,
  #     crop_w=bbox.width,
  #     crop_h=bbox.height)

  # def has_crop(self):
  #   return all(
  #     getattr(self, 'crop_%s' % a)
  #     for a in ('x', 'y', 'w', 'h'))

  # def get_crop_bbox(self):
  #   return BBox(
  #           x=self.crop_x, y=self.crop_y,
  #           width=self.crop_w, height=self.crop_h)

  # def get_viewport(self):
  #   if self.has_crop():
  #     return self.get_crop_bbox()

  @classmethod
  def from_str(cls, s, **overrides):
    """Create and return a `URI` from string `s`.

    Args:
      s (string): String form of the `URI`, e.g. `psegs://dataset=test`
      overrides (dict, optional): Override any parameters specified in `s`;
        you can also values for otherwise unset parameters this way.
    
    Returns:
      URI: The constructed instance
    """

    if isinstance(s, cls) or not bool(s):
      return s

    if s.startswith(six.moves.urllib.parse.quote_plus(URI.PREFIX)):
      s = six.moves.urllib.parse.unquote_plus(s)

    assert s.startswith(URI.PREFIX), "Missing %s in %s" % (URI.PREFIX, s)
    toks_s = s[len(URI.PREFIX):]
    if not toks_s:
      return cls()
    toks = toks_s.split('&')
    assert all('=' in tok for tok in toks), "Bad token in %s" % (toks,)
    
    kwargs = {}
    for tok in toks:
      k, v = tok.split('=')
      if k.startswith('extra.'):
        k = k[len('extra.'):]
        kwargs.setdefault('extra', {})
        kwargs['extra'][k] = v
      else:
        kwargs[k] = v
    kwargs.update(**overrides)
    
    return cls(**kwargs)
  
  def get_datum_uris(self):
    """If this `URI` has `DatumSelection`'s, create and return
    `URI` instances referencing each `StampedDatum` selected."""
    return [
      self.replaced(sel_datums=[], **attr.asdict(ds))
      for ds in self.sel_datums
    ]
  
  @classmethod
  def segment_uri_from_datum_uris(cls, uris):
    """Given a list of `uris`, construct and return a single (segment) `URI`
    instance that references the given `uris` as `DatumSelection`s"""
    assert uris, "Empty selection"
    uris = sorted(cls.from_str(uri) for uri in uris)
    out = uris[0].to_segment_uri()
    return out.replaced(sel_datums=DatumSelection.selections_from_value(uris))
