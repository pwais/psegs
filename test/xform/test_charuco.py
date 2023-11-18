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

from oarphpy import util as oputil
from psegs import datum
from psegs.xform import charuco as psc

from test import testutil

try:
  psc.check_opencv_version_for_aruco()
  HAVE_OBJDET_ARUCO = True
except ImportError:
  HAVE_OBJDET_ARUCO = False

skip_if_no_objdet_aruco = pytest.mark.skipif(
  not HAVE_OBJDET_ARUCO,
  reason="Requires modern OpenCV Aruco, see `check_opencv_version_for_aruco()`")


def check_img(actual, fixture_name, actual_output_dir):
  FIXTURES_DIR = testutil.test_fixtures_dir() / 'test_charuco_output'
  oputil.mkdir(actual_output_dir)
  
  # First dump actual, in case the fixture doesn't exist yet and we're
  # writing a new test
  actual_bytes = oputil.to_png_bytes(actual)
  actual_path = actual_output_dir / ('actual_' + fixture_name)
  with open(actual_path, 'wb') as f:
    f.write(actual_bytes)
  print(actual_path)

  return
  expected_bytes = open(FIXTURES_DIR / fixture_name, 'rb').read()
  assert actual_bytes == expected_bytes, "Check %s" % actual_path


@skip_if_no_objdet_aruco
def test_charuco_detect_board():
  import cv2
  
  ACTUAL_OUTPUT_DIR = testutil.test_tempdir('test_charuco_detect_board')

  FIXTURE_INPUT_DIR = testutil.test_fixtures_dir() / 'test_charuco' 
  
  board = psc.CharucoBoardParams(
            dict_key='DICT_6X6_1000',
            cols=11,
            rows=8,
            square_length_meters=0.022,
            marker_length_meters=0.017,
            is_legacy_pattern=True)
  
  img_gray = cv2.imread(
    str(FIXTURE_INPUT_DIR / 'frame_00000.jpg'), cv2.IMREAD_GRAYSCALE)

  FRAMES_TO_CHECK = (
    'frame_00000.jpg',
    'frame_00021.jpg',
    'frame_00057.jpg',
  )

  for frame_fname in FRAMES_TO_CHECK:

    img_gray = cv2.imread(
      str(FIXTURE_INPUT_DIR / frame_fname), cv2.IMREAD_GRAYSCALE)
    dets = psc.charuco_detect_board(board, img_gray)
    debug_images = psc.charuco_create_debug_images(img_gray, dets)

    # todo moveme
    psc.charuco_should_use_board_marker_corners(dets)
    psc.charuco_detections_to_point2ds(dets)

    # TODO moveme

    p2ds = psc.charuco_detections_to_point2ds(dets)
    p2d = p2ds[0]
    import imageio
    imageio.imwrite('/opt/psegs/p2d_test.jpg', p2d.get_debug_points_image())
    return


    # need cases where some detections are null
    import copy
    dets_markers_null = copy.deepcopy(dets)
    dets_markers_null.aruco_marker_corners = None
    dets_markers_null.aruco_marker_ids = None
    psc.charuco_detections_to_point2ds(dets_markers_null)

    dets_bmarkers_null = copy.deepcopy(dets)
    dets_bmarkers_null.charuco_marker_corners = None
    dets_bmarkers_null.charuco_marker_ids = None
    psc.charuco_detections_to_point2ds(dets_bmarkers_null)

    dets_board_null = copy.deepcopy(dets)
    dets_board_null.charuco_corners = None
    dets_board_null.charuco_ids = None
    psc.charuco_detections_to_point2ds(dets_board_null)

    dets_ch_markers_has_diff = copy.deepcopy(dets)
    dets_ch_markers_has_diff.charuco_marker_ids = (
      copy.deepcopy(dets_ch_markers_has_diff.aruco_marker_ids))
    dets_ch_markers_has_diff.charuco_marker_corners = (
      copy.deepcopy(dets_ch_markers_has_diff.aruco_marker_corners))
    for c in dets_ch_markers_has_diff.charuco_marker_corners:
      c += 0.1
    psc.charuco_detections_to_point2ds(dets_board_null)



    # TODO moveme
    p2ds = psc.charuco_detections_to_point2ds(dets)
    datums = [
      datum.StampedDatum(uri=datum.URI(), points_2d=p2d) for p2d in p2ds
    ]
    from psegs.datum.stamped_datum import Sample
    from psegs.table.sd_table import StampedDatumTable
    sdt = StampedDatumTable.from_sample(Sample(datums=datums))
    sdf = sdt.to_spark_df()
    sdf.show()



    DEBUGS_TO_CHECK = (
      'debug_marker_detections',
      'debug_marker_rejections',
      'debug_board_image',
      'debug_board_detections')
    for debug_to_check in DEBUGS_TO_CHECK:
      actual = getattr(debug_images, debug_to_check)
      fixture_name = f'{frame_fname}.{debug_to_check}.png'
      check_img(actual, fixture_name, ACTUAL_OUTPUT_DIR)



  

  breakpoint()
  print()