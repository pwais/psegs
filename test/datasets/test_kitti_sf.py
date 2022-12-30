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

from psegs.datasets import kitti_sf

from test import testutil


def test_kitti_sf_stereo_3d_viz():
  testutil.skip_if_fixture_absent(kitti_sf.Fixtures.EXTERNAL_FIXTURES_ROOT)

  base_dir = kitti_sf.Fixtures.stereo_fixture_dir()
  outdir = testutil.test_tempdir('test_kitti_sf_stereo_3d_viz')

  for frame in kitti_sf.Fixtures.STEREO_TEST_FRAMES:
    assert False, frame

    """
     * load disp, k and baseline
     * compute uvd viz
     * save as a point cloud, check binary as well as viz / GLTF

     * create MatchedPair instance
     * save trimesh viz and test

     
    """

    calib_path = base_dir / ('training/calib/%s.txt' % frame)
    calib = kitti.Calibration.from_kitti_str(open(calib_path, 'r').read())
    save_projected_lidar(
      base_dir, outdir, frame, 'image_2', calib.K2, calib.velo_to_cam_2_rect)
    save_projected_lidar(
      base_dir, outdir, frame, 'image_3', calib.K3, calib.velo_to_cam_3_rect)

  # Now test!
  expected_base = (
    kitti.Fixtures.EXTERNAL_FIXTURES_ROOT / 
      'test_kitti_object_lidar_camera_projection')
  testutil.assert_img_directories_equal(outdir, expected_base)


###############################################################################
## DSUtil Tests

def test_kitti_dsutil_smoke():
  testutil.skip_if_fixture_absent(kitti_sf.Fixtures.ROOT)

  # The above are preconditions, so this should succeed:
  assert kitti_sf.DSUtil.emplace()
