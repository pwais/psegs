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
    
    disp_path = base_dir / f'training/disp_occ_0/{frame}.png'
    disp = kitti_sf.kittisf15_load_disp(open(disp_path, 'rb'))

    cam_to_cam_path = (
      base_dir / f'training/calib_cam_to_cam/{frame.replace("_10", "")}.txt')
    
    K_2, K_3, baseline, T_00, T_01, P_2, P_3 = kitti_sf.kittisf15_load_K_baseline(open(cam_to_cam_path, 'r').read())

    uv_2_uv_3_depth = kitti_sf.kittisf15_to_stereo_matches(disp, baseline, K_2)

    import trimesh
    import numpy as np
    vs = uv_2_uv_3_depth[:, (0, 1, -1)]
    vs = vs[vs[:,-1] > 0]
    f_x = K_2[0, 0]
    f_y = K_2[1, 1]
    c_x = K_2[0, 2]
    c_y = K_2[1, 2]
    # breakpoint()
    uvd2_x = (vs[:, 0] - c_x) / f_x
    uvd2_y = (vs[:, 1] - c_y) / f_y
    uvd2_z = np.ones_like(uvd2_y)
    uvd2xyz = np.hstack([uvd2_x[:, None], uvd2_y[:, None], uvd2_z[:, None]])
    uvd2xyz *= vs[:, (-1,)]
    pc_tmesh_uvd = trimesh.points.PointCloud(vertices=uvd2xyz, colors=np.zeros_like(uvd2xyz))
    scene = trimesh.Scene()
    scene.add_geometry(pc_tmesh_uvd)
    b = trimesh.exchange.gltf.export_glb(scene)
    with open('/opt/psegs/debug.glb', 'wb') as f:
      print('debug.glb')
      f.write(b)
    
    import cv2
    import numpy as np
    # P_2 = np.eye(3, 4)
    # P_2[:3, :3] = K_2
    # P_2[:, 3] = T_00
    # P_3 = np.eye(3, 4)
    # P_3[:3, :3] = K_3
    # P_3[:, 3] = T_01
    uv_2 = uv_2_uv_3_depth[:, 0:2][uv_2_uv_3_depth[:, -1] > 0]
    uv_3 = uv_2_uv_3_depth[:, 2:4][uv_2_uv_3_depth[:, -1] > 0]
    
    xyzh = cv2.triangulatePoints(P_2, P_3, uv_2.T, uv_3.T)
    xyz = xyzh.T.copy()
    # xyz = xyz[:, :3] / xyz[:, (-1,)]
    xyz = xyz[:, :3] / xyz[:, (-1,)]
    # xyz = xyz[:, :3]
    # xyz = cv2.convertPointsFromHomogeneous(xyzh.T)
    
    pc_tmesh_xyz = trimesh.points.PointCloud(vertices=xyz.squeeze(), colors=.3 * np.ones_like(xyz))
    scene = trimesh.Scene()
    scene.add_geometry(pc_tmesh_xyz)
    b = trimesh.exchange.gltf.export_glb(scene)
    with open('/opt/psegs/debug.xyz.glb', 'wb') as f:
      print('debug.xyz.glb')
      f.write(b)
    
    
    scene = trimesh.Scene()
    scene.add_geometry(pc_tmesh_uvd)
    scene.add_geometry(pc_tmesh_xyz)
    b = trimesh.exchange.gltf.export_glb(scene)
    with open('/opt/psegs/debug.comp.glb', 'wb') as f:
      print('debug.xyz.glb')
      f.write(b)
    
    
    breakpoint()
    assert False


    """
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
