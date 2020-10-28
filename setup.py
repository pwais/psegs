#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import os
import sys
import re

try:
  from setuptools import setup
except ImportError:
  from distutils.core import setup

# Function to parse __version__ in `psegs/__init__.py`
def find_version():
  here = os.path.abspath(os.path.dirname(__file__))
  with open(os.path.join(here, 'psegs', '__init__.py'), 'r') as fp:
    version_file = fp.read()
  version_match = re.search(
    r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
  if version_match:
    return version_match.group(1)
  raise RuntimeError("Unable to find version string.")


NUSC_DEPS = [
  # TODO try to use v1.1 when they create a formal release ...
  'nuscenes-devkit==1.1.0'
]

# SPARK_DEPS = [
#   'findspark==1.3.0',
#   'numpy',
#   'pandas>=0.19.2'
# ]
# HAVE_SYSTEM_SPARK = (
#   os.environ.get('SPARK_HOME') or
#   os.path.exists('/opt/spark'))
# if not HAVE_SYSTEM_SPARK:
#   SPARK_DEPS += ['pyspark>=2.4.4']

# TF_DEPS = [
#   'crcmod',
#   'tensorflow<=1.15.0',
# ]

# UTILS = [
#   # For various
#   'six',

#   # For SystemLock
#   # 'fasteners==0.14.1', TODO clean up util.SystemLock
  
#   # For lots of things
#   'pandas',

#   # For ThruputObserver
#   'humanfriendly',
#   'tabulate',
#   'tabulatehelper',

#   # For misc image utils
#   'imageio'
# ]

# ALL_DEPS = UTILS + SPARK_DEPS + TF_DEPS

dist = setup(
  name='psegs',
  version=find_version(),
  description='A library for normalized autonomous vehicle datasets',
  author='Paul Wais',
  author_email='u@oarph.me',
  url='https://github.com/pwais/psegs',
  license='Apache License 2.0',
  packages=['psegs'],
  long_description=open('README.md').read(),
  long_description_content_type="text/markdown",
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3',
    'Topic :: Software Development :: Libraries',
    'Topic :: Scientific/Engineering',
    'Topic :: System :: Distributed Computing',
  ],
  
  test_suite='test',
  setup_requires=['pytest-runner'],
  tests_require=['pytest'],
  
  extras_require={
    # 'all': ALL_DEPS,
    # 'utils': UTILS,
    # 'spark': SPARK_DEPS,
    # 'tensorflow': TF_DEPS,
  },
)
