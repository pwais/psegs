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



import numpy as np

from psegs.util import misc


##############################################################################
## Misc

def maybe_make_homogeneous(pts, dim=3):
  """Convert numpy n-by-d array `pts` to Homogeneous coordinates of target
  `dim` if necessary"""
  if pts.shape[-1] != (dim + 1):
    pts = np.hstack((pts, np.ones((pts.shape[0], 1))))
  return pts

def l2_normalized(v):
  if len(v.shape) > 1:
    # Normalize row-wise
    return v / np.linalg.norm(v, axis=1)[:, np.newaxis]
  else:
    return v / np.linalg.norm(v)

def theta_signed(axis, v):
  return np.arctan2(np.cross(axis, v), np.dot(axis, v.T))

def to_preformatted(v):
  import pprint
  import html
  return '<pre>%s</pre>' % html.escape(pprint.pformat(v))



##############################################################################
## Datum Diffing

def datum_to_diffable_tree(datum):
  """Given a `StampedDatum` instance, return a diff-able tree for use
  in verification / diffing.

  To efficiently diff `StampedDatum`s, we use the following approach:
    1) Most datums are attrs-based classes without (auto-generated) equality
        methods, so we compare datums in a dict-like form.
    2) Many datums have numpy arrays, and those are not easily comparable. 
        However, the OarphPy `RowAdapter`-ified form of a `numpy` array (i.e.
        `oarphpy.spark.Tensor`) is easily comparable.  
    3) Some datums have embedded `oarphpy.spark.CloudpickeledCallable`
        instances, and these might have local filesystem paths embedded and
        thus are not directly comparable.  For these, we can only diff the
        function name.
    4) For any binary data fields, we just want to compare the hashes of
        the data.
  """

  import hashlib

  from oarphpy.spark import RowAdapter


  def to_sha1_str(v):
    return 'SHA1:' + hashlib.sha1(v).hexdigest()
      # Give a prefix so diffs make more sense

  def cpc_get_pyclass(cpc):
    # `cpc`` is the dict form of a Row-ified CloudpickeledCallable
    if cpc is not None:
      return 'CloudpickeledCallable:func_pyclass=' + cpc['func_pyclass']
        # Give a prefix so diffs make more sense
    return None

  DATUM_MEMBER_TO_FIELD_FORMATTER = {
    'camera_image': {
      'image_jpeg': to_sha1_str,
      'image_png': to_sha1_str,
      'image_factory': cpc_get_pyclass,
    },
    'point_cloud': {
      'cloud_factory': cpc_get_pyclass,
    }
  }

  row = RowAdapter.to_row(datum)
  rowdict = row.asDict(recursive=True)
  for membername, field_to_formatter in DATUM_MEMBER_TO_FIELD_FORMATTER.items():
    if rowdict[membername] is not None:
      d = rowdict[membername]
      for fieldname, formatter in field_to_formatter.items():
        d[fieldname] = formatter(d[fieldname])
  return rowdict


def get_datum_diff_string(sd1, sd2):
  """Return a string showing the diff between `StampedDatum`s `sd1` and
  `sd2` (if any).
  """
  tree1 = datum_to_diffable_tree(sd1)
  tree2 = datum_to_diffable_tree(sd2)
  if tree1 == tree2:
    return ''
  else:
    return misc.diff_of_pprint(tree1, tree2)
  






