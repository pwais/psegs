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

import difflib
import pprint
from pathlib import Path

import attr
import numpy as np
from oarphpy import util as oputil


# A global logger, just for PSegs
log = oputil.create_log(name='ps')


def missing_or_empty(path):
  # TODO: support S3 and GCS paths ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return oputil.missing_or_empty(path)


def attrs_eq(o1, o2):
  """A utility for providing an `__eq__()` method to `attrs`-based classes
  that contain `numpy`-valued attributes.  

  Notes:
    See also https://github.com/python-attrs/attrs/issues/435
  
  Args:
    o1 (object): An `attr.s()`-based object.
    o2 (object): An object of the same type as `o1`.
  
  Returns:
    bool: True if `o1 == o2`.
  """

  if not type(o1) == type(o2):
    raise TypeError

  o1t = attr.astuple(o1, recurse=False)
  o2t = attr.astuple(o2, recurse=False)
  
  def eq(a1, a2):
    if isinstance(a1, np.ndarray):
      return np.array_equal(a1, a2)
    else:
      return a1 == a2

  return all(eq(*ats) for ats in zip(o1t, o2t))


def unarchive_entries(archive_path, archive_files, dest):
  """Un-tar or un-zip a specific subset of files to `dest`.

  Args:
    archive_path (str or pathlib.Path): Use this archive.
    archive_files (List[str] or List[pathlib.Path]): Extract only these files
      (archive entries) from the given archive.
    dest (str or pathlib.Path): The root director for extracted files.
  """
  log.info("Trying to extract %s entries from %s ..." % (
                                    len(archive_files), archive_path))


  fws = oputil.ArchiveFileFlyweight.fws_from(str(archive_path))
  archive_files = set(str(fname) for fname in archive_files)
  fws = [fw for fw in fws if fw.name in archive_files]
  
  
  oputil.mkdir(dest)
  dest = Path(dest)
  for fw in fws:
    fw_dest = dest / fw.name
    oputil.mkdir(str(fw_dest.parent))
    with open(fw_dest, 'wb') as f:
      f.write(fw.data)
  
  log.info("... saved %s entries to %s" % (len(fws), dest))


def get_png_wh(png_bytes):
  """Return the dimensions for a PNG image.

  Based upon `get_image_size <https://github.com/scardine/image_size/blob/fb25377f42fc6c90c280462a87a41cf20cc2ac0e/get_image_size.py#L107>`_

  Args:
    png_bytes (bytearray): Bytes of a PNG file buffer
  
  Returns:
    int: width of the image in pixels
    int: height of the image in pixels
  """
  
  from io import BytesIO
  buf = BytesIO(png_bytes)
  head = buf.read(24)

  if not head.startswith(b'\211PNG\r\n\032\n'):
    raise ValueError("Not a PNG")

  import struct
  w, h = struct.unpack(">LL", head[16:24])
  return int(w), int(h)


def diff_of_pprint(v1, v2):
  """Return a human-readable diff string of the diff of `v1` and `v2`"""
  # Based upon pytest:
  # https://github.com/pytest-dev/pytest/blob/55debfad1f690d11da3b33022d55c49060460e44/src/_pytest/assertion/util.py#L236
  # https://docs.python.org/3/library/difflib.html#difflib.ndiff

  lines1 = pprint.pformat(v1).splitlines(keepends=True)
  lines2 = pprint.pformat(v2).splitlines(keepends=True)
  difftxt = ''.join(difflib.ndiff(lines1, lines2))
  return difftxt


# TODO: make these work for classes... rowadapter will only do objects
def save_rowized_pkl(obj, path):
  import pickle
  from oarphpy.spark import RowAdapter
  row = RowAdapter.to_row(obj)
  with open(path, 'wb') as f:
    pickle.dump(row, f, protocol=3) # Support older python


def load_rowized_pkl(path):
  import pickle
  from oarphpy.spark import RowAdapter
  with open(path, 'rb') as f:
    row = pickle.load(f)
  return RowAdapter.from_row(row)


# Stolen from oarphpy RowAdapter ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def _get_classname_from_obj(o):
#   # Based upon https://stackoverflow.com/a/2020083
#   module = o.__class__.__module__
#   # NB: __module__ might be null
#   if module is None or module == str.__class__.__module__:
#     return o.__class__.__name__  # skip "__builtin__"
#   else:
#     return module + '.' + o.__class__.__name__
# def _get_class_from_path(path):
#   # Pydoc is a bit safer and more robust than anything we can write
#   import pydoc
#   obj_cls, obj_name = pydoc.resolve(path)
#   assert obj_cls
#   return obj_cls

# _LAZY_SLOTS = (
#   # '__pyclass',
#   '__thunktor_bytes',
#   # '__pyclass_bytes',
#   '__value',
#   '__lock',
# )

# class LazyThunktor(object):
#   """
#   design: thunktor can get invoked:
#     * once per process when impl is called
#     * the above, but once after deser
#     * the big value is never serialized ...
#   """

#   __slots__ = _LAZY_SLOTS

#   def __init__(self, thunktor):#, embed_cls=True):
#     self.__value = None
#     # self.__pyclass = _get_classname_from_obj(thunktor)

#     import cloudpickle
#     self.__thunktor_bytes = cloudpickle.dumps(thunktor)

#     import threading
#     self.__lock = threading.Lock()
  
#   def __getstate__(self):
#     d = dict(
#       (k, getattr(self, k))
#       for k in self.__slots__
#       if k not in ('__lock', '__value'))
#     return d
  
#   def __setstate__(self, d):
#     for k, v in d.items():
#       if k not in ('__lock', '__value'):
#         setattr(self, k, v)
#     import threading
#     self.__lock = threading.Lock()
  
#   @property
#   def impl(self):
#     if self.__value is not None:
#       return self.__value
    
#     with self.__lock:
#       import cloudpickle
#       thunktor = cloudpickle.loads(self.__thunktor_bytes)
#       self.__value = thunktor()
#     return self.__value

#   def __getattribute__(self, name):
#     if name in _LAZY_SLOTS or name in ('impl', '__slots__'):
#       return object.__getattribute__(self, name)
#     else:
#       return object.__getattribute__(self.impl, name)
           
#   def __setattr__(self, name, value):
#     if name in _LAZY_SLOTS or name in ('impl', '__slots__'):
#       return object.__setattr__(self, name)
#     else:
#       return object.__setattr__(self.impl, name)
  
#   def __delattr__(self, name, value):
#     if name in _LAZY_SLOTS or name in ('impl', '__slots__'):
#       return object.__delattr__(self, name)
#     else:
#       return object.__delattr__(self.impl, name)
