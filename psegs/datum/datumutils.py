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

import numpy as np

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
