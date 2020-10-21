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

import textwrap
import mdv


def is_ipython():
  try:
    from IPython import get_ipython
    if 'IPKernelApp' not in get_ipython().config:
      return False
  except Exception:
      return False
  return True


class IDatasetUtil(object):

  @classmethod
  def emplace(cls):
    """Emplacing a dataset means downloading it any any other dependent
    fixtures, including test fixtures (see e.g. PSegs Extensions in the root
    README).  This method attempts to do that initial set-up for a given 
    dataset.
    
    In many cases, dataset availability and licensing will require manual
    effort from the user.  This method will explain necessary action (via
    the terminal or notebook) and attempt to help interactively.

    This method should be re-entrant (multiple attempts to emplace should be
    safe) and will return True only when all emplacing has suceeded.
    """
    return False

  @classmethod
  def test(cls):
    return False

  @classmethod
  def build_table(cls):
    return False

  @classmethod
  def show_md(cls, txt):
    txt = textwrap.dedent(txt)

    if is_ipython():
      from IPython.display import display, Markdown
      display(Markdown(txt))
    else:
      print()
      print(mdv.main(txt, cols=80))
      print()

