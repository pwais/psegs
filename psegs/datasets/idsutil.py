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

class IDatasetUtil(object):

  @classmethod
  def emplace(cls):
    return False

  @classmethod
  def test(cls):
    return False

  @classmethod
  def build_table(cls):
    return False

  @classmethod
  def show_md(cls, txt):
    import textwrap
    txt = textwrap.dedent(txt)

    import mdv
    print()
    print(mdv.main(txt, cols=80))
    print()

