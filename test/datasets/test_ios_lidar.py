# Copyright 2021 Maintainers of PSegs
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

import json

import imageio

from psegs import datum
from psegs.datasets import ios_lidar

from test import testutil


def test_threeDScannerApp_json_parsing():
  testutil.skip_if_fixture_absent(
    ios_lidar.Fixtures.threeDScannerApp_data_root())

  json_data_path = (
    ios_lidar.Fixtures.threeDScannerApp_data_root() / 
      'charuco-test-fixture-lowres' / 'frame_00000.json')

  assert json_data_path.exists()
  with open(json_data_path, 'r') as f:
    json_data = json.load(f)

  xform = ios_lidar.threeDScannerApp_get_ego_pose(json_data)
  assert xform.src_frame == 'ego'
  assert xform.dest_frame == 'world'

  K = ios_lidar.threeDScannerApp_get_K(json_data)
  assert K[0][0] != 0
  assert K[1][1] != 0
  assert K[0][2] != 0
  assert K[1][2] != 0

  frame_id = ios_lidar.threeDScannerApp_frame_id_from_fname(json_data_path)
  assert frame_id == '00000'


def test_threeDScannerApp_timestamps():
  testutil.skip_if_fixture_absent(
    ios_lidar.Fixtures.threeDScannerApp_data_root())
  
  scene_dir = (
    ios_lidar.Fixtures.threeDScannerApp_data_root() /
      'charuco-test-fixture-lowres')

  frame_id_to_nanostamp = ios_lidar.threeDScannerApp_create_frame_to_timestamp(
                            scene_dir)
  
  assert len(frame_id_to_nanostamp) == 92

  # These will break if we have wrong image mtimes
  assert frame_id_to_nanostamp['00000'] == 1637363832000000000
  assert frame_id_to_nanostamp['00001'] == 1637363832227758083
  assert frame_id_to_nanostamp['00002'] == 1637363832428882749
  assert frame_id_to_nanostamp['00003'] == 1637363832527052333
  assert frame_id_to_nanostamp['00004'] == 1637363832728491833
  assert frame_id_to_nanostamp['00005'] == 1637363832929434458

  assert frame_id_to_nanostamp['00087'] == 1637363846836126208
  assert frame_id_to_nanostamp['00088'] == 1637363847036753166
  assert frame_id_to_nanostamp['00089'] == 1637363847268963208
  assert frame_id_to_nanostamp['00090'] == 1637363847369206458
  assert frame_id_to_nanostamp['00091'] == 1637363847570889541

  # In human-readable form
  import datetime
  EXPECTED_STAMPS = {
    '00000': datetime.datetime(2021, 11, 19, 23, 17, 12),
    '00001': datetime.datetime(2021, 11, 19, 23, 17, 12, 227758),

    '00005': datetime.datetime(2021, 11, 19, 23, 17, 12, 929435),
    '00087': datetime.datetime(2021, 11, 19, 23, 17, 26, 836126),

    '00090': datetime.datetime(2021, 11, 19, 23, 17, 27, 369207),
    '00091': datetime.datetime(2021, 11, 19, 23, 17, 27, 570889),
  }
  for frame_id, expected_t in EXPECTED_STAMPS.items():
    actual_t = datetime.datetime.fromtimestamp(
                1e-9 * frame_id_to_nanostamp[frame_id])
    assert expected_t == actual_t


### Test PointCloud from Mesh #################################################

def test_threeDScannerApp_create_point_cloud_from_mesh():
  testutil.skip_if_fixture_absent(
    ios_lidar.Fixtures.threeDScannerApp_data_root())
  
  scene_dir = (
    ios_lidar.Fixtures.threeDScannerApp_data_root() /
      'charuco-test-fixture-highres')
  
  pc = ios_lidar.threeDScannerApp_create_point_cloud_from_mesh(
          scene_dir / 'export.obj')
  
  cloud = pc.get_cloud()
  assert cloud.shape == (113955, 3)

  outdir = testutil.test_tempdir(
            'test_threeDScannerApp_create_point_cloud_from_mesh')
  imageio.imwrite(
    outdir / 'debug_mesh.png',
    datum.PointCloud.get_ortho_debug_image(
              cloud,
              flatten_axis='+y', # For iOS, +y is up
              u_axis='+x', # For this scene, +x is "right"
              v_axis='-z', # For this scene, -z is "up"
              u_bounds=(-.75, .75),
              v_bounds=(-.75, .75),
              filter_behind=False,
              pixels_per_meter=400))
  
  expected_base = (
    ios_lidar.Fixtures.threeDScannerApp_test_data_root() / 
      'test_threeDScannerApp_create_point_cloud_from_mesh')
  
  testutil.assert_img_directories_equal(outdir, expected_base)



### Test CameraImage ##########################################################

def test_threeDScannerApp_create_camera_image_lowres():
  testutil.skip_if_fixture_absent(
    ios_lidar.Fixtures.threeDScannerApp_data_root())
  
  scene_dir = (
    ios_lidar.Fixtures.threeDScannerApp_data_root() /
      'charuco-test-fixture-lowres')
  

  ### Test RGB

  frame_json_path = scene_dir / 'frame_00045.json'
  ci = ios_lidar.threeDScannerApp_create_camera_image(frame_json_path)
  EXTRA_EXPECTED = {
    'threeDScannerApp.averageAngularVelocity': '0.1675194501876831',
    'threeDScannerApp.averageVelocity': '0.09594733268022537',
    'threeDScannerApp.cameraGrain': '0',
    'threeDScannerApp.exposureDuration': '0.016393441706895828',
    'threeDScannerApp.frame_id': '00045',
    'threeDScannerApp.frame_index': '45',
    'threeDScannerApp.frame_json_name': 'frame_00045.json',
    'threeDScannerApp.img_path': 'frame_00045.jpg',
    'threeDScannerApp.intrinsics': '[1453.939453125, 0, 973.69287109375, 0, '
                              '1453.939453125, 714.6398315429688, 0, 0, 1]',
    'threeDScannerApp.motionQuality': '0.952137291431427',
    'threeDScannerApp.projectionMatrix': '[1.514520287513733, 0, '
                                      '-0.01478421688079834, 0, 0, '
                                      '2.019360303878784, '
                                      '-0.006750226020812988, 0, 0, 0, '
                                      '-0.9999997615814209, '
                                      '-0.0009999998146668077, 0, 0, -1, 0]',
    'threeDScannerApp.scan_dir': 'charuco-test-fixture-lowres',
    'threeDScannerApp.time': '942363.2627448752'
  }
  assert ci.extra == EXTRA_EXPECTED
  
  img = ci.image
  assert (ci.height, ci.width) == (1440, 1920)
  assert img.shape[:3] == (1440, 1920, 3)


  ### Test Depth

  dci = ios_lidar.threeDScannerApp_create_camera_image(
          frame_json_path, sensor_name='depth|front')
  
  assert dci.channel_names == ['depth', 'confidence']
  assert (dci.height, dci.width) == (192, 256)
  dimg = dci.image
  assert dimg.shape[:3] == (192, 256, 2)

  dpc = dci.depth_image_to_point_cloud()
  assert dpc.cloud_colnames == ['x', 'y', 'z', 'confidence']
  cloud = dpc.get_cloud()
  assert cloud.shape == (49152, 4)


  ### Test Projection / Calibration

  outdir = testutil.test_tempdir(
            'test_threeDScannerApp_create_camera_image_lowres')
  frame_id = ci.extra['threeDScannerApp.frame_id']

  imageio.imwrite(
    outdir / ('depth_debug_%s_5mm.png' % frame_id),
    dci.get_debug_image(period_meters=0.005))

  imageio.imwrite(
    outdir / ('projected_lidar_%s_5cm.png' % frame_id),
    ci.get_debug_image(clouds=[dpc], period_meters=0.05))

  imageio.imwrite(
    outdir / ('front_rv_debug_%s.png' % frame_id),
    dpc.get_front_rv_debug_image(
            camera_images=[ci],
            z_bounds_meters=(-1, 1),
            y_bounds_meters=(-1.5, 1.5),
            pixels_per_meter=400))

  imageio.imwrite(
    outdir / ('bev_debug_%s.png' % frame_id),
    dpc.get_bev_debug_image(
            camera_images=[ci],
            x_bounds_meters=(-.4, .4),
            y_bounds_meters=(-.6, .6),
            pixels_per_meter=400))


  expected_base = (
    ios_lidar.Fixtures.threeDScannerApp_test_data_root() / 
      'test_threeDScannerApp_create_camera_image_lowres')
  
  testutil.assert_img_directories_equal(outdir, expected_base)


def test_threeDScannerApp_create_camera_image_high():
  testutil.skip_if_fixture_absent(
    ios_lidar.Fixtures.threeDScannerApp_data_root())
  
  scene_dir = (
    ios_lidar.Fixtures.threeDScannerApp_data_root() /
      'charuco-test-fixture-highres')
  

  ### Test RGB

  frame_json_path = scene_dir / 'frame_00012.json'
  ci = ios_lidar.threeDScannerApp_create_camera_image(frame_json_path)
  EXTRA_EXPECTED = {
    'threeDScannerApp.altitude': '44.56772631313652',
    'threeDScannerApp.averageAngularVelocity': '0.06802098453044891',
    'threeDScannerApp.averageVelocity': '0.05275433138012886',
    'threeDScannerApp.cameraGrain': '0',
    'threeDScannerApp.exposureDuration': '0.016393441706895828',
    'threeDScannerApp.frame_id': '00012',
    'threeDScannerApp.frame_index': '12',
    'threeDScannerApp.frame_json_name': 'frame_00012.json',
    'threeDScannerApp.gpsTime': '1637363870.0003262',
    'threeDScannerApp.hasGPS': 'true',
    'threeDScannerApp.horizontalAccuracy': '16.303965550780777',
    'threeDScannerApp.img_path': 'frame_00012.jpg',
    'threeDScannerApp.intrinsics': '[1455.00341796875, 0, 980.0227661132812, 0, '
                                    '1455.00341796875, 713.0078125, 0, 0, 1]',
    'threeDScannerApp.latitude': '37.77783127898064',
    'threeDScannerApp.longitude': '-122.39396648365398',
    'threeDScannerApp.motionQuality': '0.9805654287338257',
    'threeDScannerApp.projectionMatrix': '[1.5156285762786865, 0, '
                                          '-0.021377921104431152, 0, 0, '
                                          '2.0208380222320557, '
                                          '-0.009016871452331543, 0, 0, 0, '
                                          '-0.9999997615814209, '
                                          '-0.0009999998146668077, 0, 0, -1, 0]',
    'threeDScannerApp.scan_dir': 'charuco-test-fixture-highres',
    'threeDScannerApp.time': '942393.5939737086',
    'threeDScannerApp.verticalAccuracy': '6.213013810869289'
  }
  assert ci.extra == EXTRA_EXPECTED
  
  img = ci.image
  assert (ci.height, ci.width) == (1440, 1920)
  assert img.shape[:3] == (1440, 1920, 3)

  frame_id = ci.extra['threeDScannerApp.frame_id']
  assert int(frame_id) == 12

  ### Test Mesh Projection / Calibration

  pc = ios_lidar.threeDScannerApp_create_point_cloud_from_mesh(
          scene_dir / 'export.obj')
  
  # Cloud is in world frame, put it in the ego frame of the camera
  cloud = pc.get_cloud()
  pc.cloud = ci.ego_pose['world', 'ego'].apply(cloud).T
  
  outdir = testutil.test_tempdir(
            'test_threeDScannerApp_create_camera_image_highres')

  imageio.imwrite(
    outdir / ('projected_lidar_%s_5mm.png' % frame_id),
    ci.get_debug_image(clouds=[pc], period_meters=0.005))

  imageio.imwrite(
    outdir / ('front_rv_debug_%s.png' % frame_id),
    pc.get_front_rv_debug_image(
            camera_images=[ci],
            z_bounds_meters=(-1.5, 1.5),
            y_bounds_meters=(-2.0, 2.0),
            pixels_per_meter=400))
  imageio.imwrite(
    outdir / ('bev_debug_%s.png' % frame_id),
    pc.get_bev_debug_image(
            camera_images=[ci],
            x_bounds_meters=(-.5, .5),
            y_bounds_meters=(-.6, .6),
            pixels_per_meter=400))


  expected_base = (
    ios_lidar.Fixtures.threeDScannerApp_test_data_root() / 
      'test_threeDScannerApp_create_camera_image_highres')
  testutil.assert_img_directories_equal(outdir, expected_base)



def test_threeDScannerApp_sd_table():
  testutil.skip_if_fixture_absent(
      ios_lidar.Fixtures.threeDScannerApp_data_root())
  
  with testutil.LocalSpark.sess() as spark:
    suri = datum.URI.from_str(
      'psegs://dataset=psegs-ios-lidar-ext&split=threeDScannerApp_data&segment_id=charuco-test-fixture-lowres')
    sd_df_actual = ios_lidar.IOSLidarSDTable.as_df(spark, force_compute=True, only_segments=[suri])
    
    testutil.check_stamped_datum_dfs_equal(
      spark,
      sd_df_actual,
      sd_df_expected_path=(
        ios_lidar.Fixtures.threeDScannerApp_test_data_root() / 
          'test_threeDScannerApp_charuco-test-fixture-lowres-sd.parquet'))


    suri = datum.URI.from_str(
      'psegs://dataset=psegs-ios-lidar-ext&split=threeDScannerApp_data&segment_id=charuco-test-fixture-highres')
    sd_df_actual = ios_lidar.IOSLidarSDTable.as_df(spark, force_compute=True, only_segments=[suri])
    
    testutil.check_stamped_datum_dfs_equal(
      spark,
      sd_df_actual,
      sd_df_expected_path=(
        ios_lidar.Fixtures.threeDScannerApp_test_data_root() / 
          'test_threeDScannerApp_charuco-test-fixture-highres-sd.parquet'))

