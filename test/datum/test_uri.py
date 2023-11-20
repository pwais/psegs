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

from psegs.datum.uri import URI
from psegs.datum.uri import DatumSelection


def check_eq(uri, s):
  assert str(uri) == s
  
  # Also exercise URI.__eq__() as well as string deserialization
  assert uri == URI.from_str(s)


def test_uri_basic():
  check_eq(URI(), URI.PREFIX)

  check_eq(
    URI(dataset='d', split='s', segment_id='s', timestamp=1, topic='t'),
    'psegs://dataset=d&split=s&segment_id=s&timestamp=1&topic=t')
  
  # String timestamps get converted
  check_eq(
    URI(dataset='d', split='s', segment_id='s', timestamp='1', topic='t'),
    'psegs://dataset=d&split=s&segment_id=s&timestamp=1&topic=t')

  # Special handling for extra
  check_eq(
    URI(dataset='d', extra={'a': 'foo', 'b': 'bar'}),
    'psegs://dataset=d&extra.a=foo&extra.b=bar')
  

def test_uri_from_str_extended():
  with_added_extra = URI.from_str('psegs://dataset=d', extra={'key': 'value'})
  check_eq(
    with_added_extra,
    'psegs://dataset=d&extra.key=value')


def test_uri_datums():

  check_eq(
    URI(dataset='d', sel_datums=[DatumSelection(topic='t1', timestamp=1)]),
    'psegs://dataset=d&sel_datums=t1,1')

  check_eq(
    URI(dataset='d', sel_datums='t1,1'),
    'psegs://dataset=d&sel_datums=t1,1')

  check_eq(
    URI(dataset='d', sel_datums=[('t1','1')]),
    'psegs://dataset=d&sel_datums=t1,1')

  check_eq(
    URI(dataset='d', sel_datums=[{'topic': 't1', 'timestamp': 1}]),
    'psegs://dataset=d&sel_datums=t1,1')


def test_uri_datum_sorting():
  dss = [
    DatumSelection(topic='t1', timestamp=1),
    DatumSelection(topic='t1', timestamp=2),
    DatumSelection(topic='t2', timestamp=1),
  ]
  assert dss == sorted(dss)


def test_uri_datum_to_datum_uris():
  def check_eqs(uri, sd_uris):
    actual_sd_uris = uri.get_datum_uris()
    assert len(actual_sd_uris) == len(sd_uris)
    for asd_uri, esd_uri in zip(actual_sd_uris, sd_uris):
      check_eq(asd_uri, esd_uri)
  
  check_eqs(URI(), [])
  check_eqs(URI(dataset='d'), [])

  check_eqs(
    URI(dataset='d', sel_datums=[DatumSelection(topic='t1', timestamp=1)]),
    ['psegs://dataset=d&timestamp=1&topic=t1'])

  sel_datums = [
    DatumSelection(topic='t1', timestamp=1),
    DatumSelection(topic='t1', timestamp=2),
    DatumSelection(topic='t2', timestamp=1),
  ]
  check_eqs(
    URI(dataset='d', sel_datums=sel_datums),
    [
      'psegs://dataset=d&timestamp=1&topic=t1',
      'psegs://dataset=d&timestamp=2&topic=t1',
      'psegs://dataset=d&timestamp=1&topic=t2',
    ])


def test_segment_uri_from_datum_uris():

  with pytest.raises(Exception):
    URI.segment_uri_from_datum_uris([])

  assert (
    URI.from_str('psegs://dataset=d&sel_datums=t1,1') ==
    URI.segment_uri_from_datum_uris([
      'psegs://dataset=d&timestamp=1&topic=t1'
    ]))
  
  assert (
    URI.from_str('psegs://dataset=d&sel_datums=t1,1,t2,1') ==
    URI.segment_uri_from_datum_uris([
      'psegs://dataset=d&timestamp=1&topic=t1',
      'psegs://dataset=d&timestamp=1&topic=t2',
    ]))

  assert (
    URI(
      dataset='d',
      sel_datums=[URI(topic='t1', timestamp=1), URI(topic='t2', timestamp=1)]) 
        ==
    URI.segment_uri_from_datum_uris([
      'psegs://dataset=d&timestamp=1&topic=t1',
      'psegs://dataset=d&timestamp=1&topic=t2',
    ]))
  
  from pyspark import Row
  assert (
    URI(
      dataset='d',
      sel_datums=[
        Row(topic='t1', timestamp=1, alt='yay'),
        Row(topic='t2', timestamp=1, moof='foo')]) 
          ==
    URI.segment_uri_from_datum_uris([
      'psegs://dataset=d&timestamp=1&topic=t1',
      'psegs://dataset=d&timestamp=1&topic=t2',
    ]))
  
  assert (
    URI(
      dataset='d',
      sel_datums=[
        Row(uri=URI(topic='t1', timestamp=1), alt='yay'),
        Row(uri=URI(topic='t2', timestamp=1), moof='foo')])
          ==
    URI.segment_uri_from_datum_uris([
      'psegs://dataset=d&timestamp=1&topic=t1',
      'psegs://dataset=d&timestamp=1&topic=t2',
    ]))


def test_uri_sorting():
  # A less-complete URI is always less than a more-complete one
  assert URI() < URI(dataset='d', timestamp=0, topic='t')
  
  # Ties are broken using tuple-based encoding
  u1 = URI(dataset='d', timestamp=1, topic='t')
  u2 = URI(dataset='d', timestamp=2, topic='t')
  assert u1 < u2
  assert u1.as_tuple() < u2.as_tuple()
  assert str(u1) < str(u2)  # Usually true, but NB timestamps are NOT padded!


def test_uri_soft_match():
  def soft_matches(left, right):
    left = URI.from_str(left)
    right = URI.from_str(right)
    return left.soft_matches_segment_of(right)
  
  # Empty URI is a wildcard match for any
  assert soft_matches(
            'psegs://',
            'psegs://segment_id=s1')
  assert soft_matches(
            'psegs://',
            'psegs://dataset=s1')
  assert soft_matches(
            'psegs://',
            'psegs://split=s1')

  # Typically we just match on segment_id
  assert soft_matches(
            'psegs://segment_id=s1',
            'psegs://segment_id=s1')
  assert not soft_matches(
            'psegs://segment_id=s1',
            'psegs://segment_id=nopenope')
  
  # lhs can be less precise, but not rhs
  assert soft_matches(
            'psegs://segment_id=s1',
            'psegs://segment_id=s1&dataset=d')
  assert soft_matches(
            'psegs://segment_id=s1',
            'psegs://segment_id=s1&dataset=d2')
  assert not soft_matches(
            'psegs://segment_id=s1&dataset=d',
            'psegs://segment_id=s1')

  assert soft_matches(
            'psegs://dataset=d',
            'psegs://dataset=d&segment_id=s1')
  assert soft_matches(
            'psegs://split=s',
            'psegs://split=s&segment_id=s1')
  assert not soft_matches(
            'psegs://dataset=d&segment_id=s1',
            'psegs://dataset=d')
  assert not soft_matches(
            'psegs://dataset=d&segment_id=s1',
            'psegs://segment_id=s1')
  assert not soft_matches(
            'psegs://split=s&segment_id=s1',
            'psegs://split=s')
