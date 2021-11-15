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
import tempfile
from pathlib import Path

import numpy as np
from oarphpy import util

from psegs.util import plotting as pspl

from test import testutil


def check_img(actual, fixture_name):
  FIXTURES_DIR = Path(__file__).parent / '../fixtures'
  OUTPUT_DIR = testutil.test_tempdir('test_plotting')
  util.mkdir(OUTPUT_DIR)
  
  # First dump actual, in case the fixture doesn't exist yet and we're
  # writing a new test
  actual_bytes = util.to_png_bytes(actual)
  actual_path = OUTPUT_DIR / ('actual_' + fixture_name)
  open(actual_path, 'wb').write(actual_bytes)

  expected_bytes = open(FIXTURES_DIR / fixture_name, 'rb').read()
  assert actual_bytes == expected_bytes, "Check %s" % actual_path


def test_draw_xy_depth_in_image():
  # Create points for a test image:
  #  * One point every 10 pixels in x- and y- directions
  #  * The depth value of the pixel is the scalar value of the y-coord
  #      interpreted as meters
  h, w = 1000, 100
  pts = []
  for y in range(int(h / 10)):
    for x in range(int(w / 10)):
      pts.append((x * 10, y * 10, y))
  
  apts = np.array(pts)
  actual = np.zeros((h, w, 3))
  pspl.draw_xy_depth_in_image(actual, apts, marker_radius=0)
  check_img(actual, 'test_draw_xy_depth_in_image.png')

  actual_2 = np.zeros((h, w, 3))
  pspl.draw_xy_depth_in_image(actual_2, apts, marker_radius=1)
  check_img(actual_2, 'test_draw_xy_depth_in_image_radius_2.png')

  # Test user colors
  colors = 255 * np.cos(apts / 10)
  actual_3 = np.zeros((h, w, 3))
  pspl.draw_xy_depth_in_image(actual_3, apts, marker_radius=1, user_colors=colors)
  check_img(actual_3, 'test_draw_xy_depth_in_image_user_colors.png')


def test_draw_depth_in_image():
  # Create a depth channel for a test image:
  #  * Top half depth is just a linear function of xy coord
  #  * Bottom half is all invalid depth
  h, w = 1000, 100
  depth = np.zeros((h, w))
  for y in range(h):
    for x in range(w):
      if y < .5 * h:
        depth[y, x] = x + y
      else:
        depth[y, x] = -1 if (x < .5 * w) else float('nan')

  
  actual = np.zeros((h, w, 3))
  pspl.draw_depth_in_image(actual, depth)
  import imageio
  imageio.imwrite('/opt/psegs/yay.png', actual)
  check_img(actual, 'test_draw_depth_in_image.png')


def test_draw_cuboid_xy_in_image():
  cube = np.array([
    # Front
    [50, 50],
    [50, 75],
    [75, 75],
    [75, 50],
    
    # Back
    [15, 15],
    [15, 40],
    [40, 40],
    [40, 15],
  ])

  h, w = 100, 100
  actual = np.zeros((h, w, 3))
  pspl.draw_cuboid_xy_in_image(actual, cube, (128, 0, 128))

  check_img(actual, 'test_draw_cuboid_xy_in_image.png')


def test_draw_bbox_in_image():
  from psegs.datum.bbox2d import BBox2D

  img = np.zeros((100, 200, 3))
  
  center = BBox2D(x=80, y=40, width=20, height=20, category_name='center')
  pspl.draw_bbox_in_image(img, center)

  up_left = BBox2D(x=5, y=5, width=40, height=20, category_name='up_left')
  pspl.draw_bbox_in_image(img, up_left)
  
  low_right = BBox2D(x=150, y=75, width=40, height=20, category_name='low_right')
  pspl.draw_bbox_in_image(img, low_right)

  no_txt = BBox2D(x=5, y=75, width=10, height=10, category_name='')
  pspl.draw_bbox_in_image(img, no_txt)

  check_img(img, 'test_draw_bbox_in_image.png')


def test_get_ortho_debug_image():
  
  # Create a circular spiral in 3-d
  t = np.arange(0, 2 * np.pi, 2 * np.pi / 100)
  r = (t / (2 * np.pi))
  uvd = np.column_stack([r * np.cos(t), r * np.sin(t), t])
  
  def draw_window(uvd, bounds):
    kwargs = dict(
                pixels_per_meter=100,
                marker_radius=2,
                period_meters=2 * np.pi / 10,
                min_u=bounds[0],
                min_v=bounds[1],
                max_u=bounds[2],
                max_v=bounds[3])
    return pspl.get_ortho_debug_image(uvd, **kwargs)
  
  check_img(
    draw_window(uvd, [-1.25, -1.25, 1.25, 1.25]),
    'test_get_ortho_debug_image_all_manual.png')
  check_img(
    draw_window(uvd, [None, None, None, None]),
    'test_get_ortho_debug_image_autobound.png')
  check_img(
    draw_window(uvd, [0, 0, 1.25, 1.25]),
    'test_get_ortho_debug_image_q1.png')
  check_img(
    draw_window(uvd, [-1.25, 0, 0, 1.25]),
    'test_get_ortho_debug_image_q2.png')
  check_img(
    draw_window(uvd, [-1.25, -1.25, 0, 0]),
    'test_get_ortho_debug_image_q3.png')
  check_img(
    draw_window(uvd, [0, -1.25, 1.25, 0]),
    'test_get_ortho_debug_image_q4.png')
  check_img(
    draw_window(uvd, [1, 1, 2, 2]),
    'test_get_ortho_debug_image_empty_space.png')
  