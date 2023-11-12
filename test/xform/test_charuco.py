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

from psegs.xform import charuco as psc

from test import testutil

# try:
#   psc.from_cv2_import_aruco()
#   HAVE_ARUCO = True
# except ImportError:
#   HAVE_ARUCO = False

# skip_if_no_aruco = pytest.mark.skipif(not HAVE_ARUCO, reason="Requires opencv-contrib Aruco")


# @skip_if_no_aruco
def test_charuco_detect_board():
  FIXTURE_IMG_PATH = (testutil.test_fixtures_dir() / 
    'test_charuco' / 'frame_00000.jpg')
  
  import cv2

  board = psc.CharucoBoard()
  img_gray = cv2.imread(str(FIXTURE_IMG_PATH), cv2.IMREAD_GRAYSCALE)
  ret = psc.detect_charuco_board(board, img_gray)
  breakpoint()
  print()