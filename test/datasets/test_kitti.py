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
import unittest
from pathlib import Path

import imageio
import numpy as np

from oarphpy import util as oputil

from psegs import datum
from psegs import util
from psegs.datasets import kitti
from test import testutil


###############################################################################
## Utils

def assert_img_directories_equal(actual_dir, expected_dir):
  util.log.info("Inspecting artifacts in %s ..." % expected_dir)
  for actual in oputil.all_files_recursive(actual_dir):
    actual = Path(actual)
    expected = expected_dir / actual.name

    match = (open(actual, 'rb').read() == open(expected, 'rb').read())
    if not match:
      import imageio
      actual_img = imageio.imread(actual)
      expected_img = imageio.imread(expected)
      
      diff = expected_img - actual_img
      n_pixels = (diff != 0).sum() / 3
      diff_path = str(actual) + '.diff.png'
      imageio.imwrite(diff_path, diff)
      assert False, \
        "File mismatch \n%s != %s ,\n %s pixels different, diff: %s" % (
          actual, expected, n_pixels, diff_path)
  
  util.log.info("Good! %s == %s" % (actual_dir, expected_dir))


def save_projected_lidar(base_dir, outdir, frame, camera, K, lidar_to_cam):
  with open(base_dir / ('training/velodyne/%s.bin' % frame), 'rb') as f:
    raw_lidar = np.frombuffer(f.read(), dtype=np.float32).reshape((-1, 4))
  xyz = raw_lidar[:, :3]
  # unused: reflectance = raw_lidar[:, 3:]

  img = imageio.imread(base_dir / 'training' / camera / ('%s.png' % frame))

  xyd = K.dot(lidar_to_cam.apply(xyz))
  xyd[0, :] /= xyd[2, :]
  xyd[1, :] /= xyd[2, :]
  xyd = xyd.T

  def filter_behind_cam(my_xyd):
    my_xyd = my_xyd.T
    idx_ = np.where(my_xyd[2, :] > 0)
    idx_ = idx_[0]
    my_xyd = my_xyd[:, idx_]
    return my_xyd.T
  
  xyd = filter_behind_cam(xyd)

  from psegs.util import plotting as pspl
  util.log.info("Projecting %s %s ..." % (frame, camera))
  pspl.draw_xy_depth_in_image(img, xyd, alpha=0.5)
  imageio.imwrite(
    outdir / ('projected_lidar_%s_%s.png' % (frame.replace('/', '_'), camera)),
    img)


def save_projected_cuboids(
    base_dir, cuboids, outdir, frame, camera, K, lidar_to_cam):
  
  img = imageio.imread(base_dir / 'training' / camera / ('%s.png' % frame))

  for cuboid in cuboids:
    # NB: For simplicity, we do NOT filter off-camera cuboids; these will plot
    # oddly but consistently.
    cuboid.obj_from_ego.src_frame = 'ego' # In KITTI, lidar = ego
    cxyz = cuboid.get_box3d()
    # from psegs.datum.datumutils import maybe_make_homogeneous ~~~~~~~~~~~~~~~~~~~~~
    # cxyd = calib.P2.dot(maybe_make_homogeneous(cxyz).T)
    cxyd = K.dot(lidar_to_cam.apply(cxyz))
    cxyd[0, :] /= cxyd[2, :]
    cxyd[1, :] /= cxyd[2, :]
    cxyd = cxyd.T
    from psegs.util import plotting as pspl
    from oarphpy.plotting import hash_to_rbg
    pspl.draw_cuboid_xy_in_image(
      img, cxyd[:,:2], hash_to_rbg(cuboid.category_name))
  
  fname = 'projected_cuboids_%s_%s.png' % (frame.replace('/', '_'), camera)
  imageio.imwrite(outdir / fname, img)


def save_labels_projected_to_lidar(
  base_dir, outdir, frame, calib, cloud, cuboids):

  ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ## Note: KITTI Cuboids are in the *camera* frame and must be projected
  ## into the lidar frame for plotting. This test helps document and 
  ## ensure this assumption holds.
  ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  lidar_to_cam = calib.R0_rect @ calib.velo_to_cam_unrectified
  cam_to_lidar = lidar_to_cam.get_inverse()

  cuboids = [c for c in cuboids if c.category_name != 'DontCare']
  for c in cuboids:
    from psegs.datum.transform import Transform
    obj_from_ego_lidar = cam_to_lidar @ c.obj_from_ego
    c.obj_from_ego = obj_from_ego_lidar
    c.obj_from_ego.src_frame = 'ego'
    c.obj_from_ego.dest_frame = 'obj'

  ## Now create debug images
  pc = datum.PointCloud(cloud=cloud)

  util.log.info("Projecting BEV %s ..." % frame)
  bev_img = pc.get_bev_debug_image(cuboids=cuboids)
  fname = 'projected_lidar_labels_bev_%s.png' % frame.replace('/', '_')
  imageio.imwrite(outdir / fname, bev_img)

  util.log.info("Projecting Front RV %s ..." % frame)
  rv_img = pc.get_front_rv_debug_image(cuboids=cuboids)
  fname = 'projected_lidar_labels_front_rv_%s.png' % frame.replace('/', '_')
  imageio.imwrite(outdir / fname, rv_img)



###############################################################################
## Common

MOCK_CALIBRATION = """
P0: 1e+02 2e+02 3e+02 4e+02 5e+02 6e+02 7e+02 8e+02 9e+02 10e+02 11e+02 12e+02
P1: 1e+02 2e+02 3e+02 4e+02 5e+02 6e+02 7e+02 8e+02 9e+02 10e+02 11e+02 12e+02
P2: 1e+02 2e+02 3e+02 4e+02 5e+02 6e+02 7e+02 8e+02 9e+02 10e+02 11e+02 12e+02
P3: 1e+02 2e+02 3e+02 4e+02 5e+02 6e+02 7e+02 8e+02 9e+02 10e+02 11e+02 12e+02
R0_rect: 1e+02 2e+02 3e+02 4e+02 5e+02 6e+02 7e+02 8e+02 9e+02
Tr_velo_to_cam: 1e+02 2e+02 3e+02 4e+02 5e+02 6e+02 7e+02 8e+02 9e+02 10e+02 11e+02 12e+02
Tr_imu_to_velo: 1e+02 2e+02 3e+02 4e+02 5e+02 6e+02 7e+02 8e+02 9e+02 10e+02 11e+02 12e+02
"""


def test_kitti_load_calibration():
  calib = kitti.Calibration.from_kitti_str(MOCK_CALIBRATION)
  
  MOCK_3x4 = np.array([
       [ 100.,  200.,  300.,  400.],
       [ 500.,  600.,  700.,  800.],
       [ 900., 1000., 1100., 1200.],
  ])
  MOCK_3x3 = np.array([
       [100., 200., 300.],
       [400., 500., 600.],
       [700., 800., 900.],
  ])
  MOCK_R = MOCK_3x4[:3, :3]
  MOCK_T = MOCK_3x4[:3, 3]
  MOCK_R0_rect = datum.Transform(
                  rotation=MOCK_3x3,
                  src_frame='camera|left_raw',
                  dest_frame='camera|left_sensor')
  MOCK_velo_to_cam_unrectified = datum.Transform(
                  rotation=MOCK_R,
                  translation=MOCK_T,
                  src_frame='lidar',
                  dest_frame='camera|left_grey_raw')

  # Fixtures for derived attributes
  MOCK_DERIVED_T = np.array([[-3596., -1398.66666667, 1200.]]).T
  vel_to_cam_left_grey = MOCK_R0_rect @ MOCK_velo_to_cam_unrectified

  RT_left_color = datum.Transform(translation=MOCK_DERIVED_T)
  MOCK_velo_to_cam_2_rect = RT_left_color @ vel_to_cam_left_grey
  MOCK_velo_to_cam_2_rect.src_frame = 'ego'
  MOCK_velo_to_cam_2_rect.dest_frame = 'camera|left'

  RT_right_color = datum.Transform(translation=MOCK_DERIVED_T)
  MOCK_velo_to_cam_3_rect = RT_right_color @ vel_to_cam_left_grey
  MOCK_velo_to_cam_3_rect.src_frame = 'ego'
  MOCK_velo_to_cam_3_rect.dest_frame = 'camera|right'


  EXPECTED = kitti.Calibration(
              P2=MOCK_3x4,
              P3=MOCK_3x4,

              K2=MOCK_3x4[:3, :3],
              K3=MOCK_3x4[:3, :3],
              T2=MOCK_DERIVED_T,
              T3=MOCK_DERIVED_T,
              
              R0_rect=MOCK_R0_rect,
              velo_to_cam_unrectified=MOCK_velo_to_cam_unrectified,
              imu_to_velo=
                datum.Transform(
                  rotation=MOCK_R,
                  translation=MOCK_T,
                  src_frame='oxts',
                  dest_frame='lidar'),
              
              velo_to_cam_2_rect=MOCK_velo_to_cam_2_rect,
              velo_to_cam_3_rect=MOCK_velo_to_cam_3_rect,
  )

  assert calib == EXPECTED



MOCK_OXTS = """
49.0 8.0 115.0 0.3 0.4 0.5 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25 26 27 28 29
49.0 8.0 115.0 0.3 0.4 0.5 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25 26 27 28 29
"""


def test_kitti_load_transforms_from_oxts():
  from scipy.spatial.transform import Rotation as R

  frame_to_xform = kitti.load_transforms_from_oxts(MOCK_OXTS)
  assert sorted(frame_to_xform.keys()) == [0, 1]

  EXPECT_R = R.from_euler('xyz', [0.3, 0.4, 0.5]).as_matrix()
  EXPECT_T = np.array([[5.84257256e+05, 4.11667947e+06, 115.0]]).T

  xform = frame_to_xform[0]
  np.testing.assert_allclose(EXPECT_R, xform.rotation)
  np.testing.assert_allclose(EXPECT_T, xform.translation)
  assert xform.src_frame == 'world'
  assert xform.dest_frame == 'oxts'


def test_kitti_archive_file_to_uri():
  INPUT_EXPECTED_OUT = {
    ## Object Benchmark
    ('data_object_label_2.zip', 'training/label_2/006192.txt'):
      'psegs://dataset=kitti-object&split=train&segment_id=kitti-object-benchmark-train&topic=labels|cuboids&extra.kitti.archive=data_object_label_2.zip&extra.kitti.archive.file=training/label_2/006192.txt&extra.kitti.frame=006192',
    ('data_object_image_2.zip', 'training/image_2/006192.png'):
      'psegs://dataset=kitti-object&split=train&segment_id=kitti-object-benchmark-train&topic=camera|left&extra.kitti.archive=data_object_image_2.zip&extra.kitti.archive.file=training/image_2/006192.png&extra.kitti.frame=006192',
    ('data_object_image_3.zip', 'training/image_3/006192.png'):
      'psegs://dataset=kitti-object&split=train&segment_id=kitti-object-benchmark-train&topic=camera|right&extra.kitti.archive=data_object_image_3.zip&extra.kitti.archive.file=training/image_3/006192.png&extra.kitti.frame=006192',
    ('data_object_prev_2.zip', 'training/prev_2/006192_02.png'):
      'psegs://dataset=kitti-object&split=train&segment_id=kitti-object-benchmark-train&topic=camera|left&extra.kitti.archive=data_object_prev_2.zip&extra.kitti.archive.file=training/prev_2/006192_02.png&extra.kitti.frame=006192&extra.kitti.prev=02',
    ('data_object_prev_3.zip', 'training/prev_3/006192_02.png'):
      'psegs://dataset=kitti-object&split=train&segment_id=kitti-object-benchmark-train&topic=camera|right&extra.kitti.archive=data_object_prev_3.zip&extra.kitti.archive.file=training/prev_3/006192_02.png&extra.kitti.frame=006192&extra.kitti.prev=02',
    ('data_object_velodyne.zip', 'training/velodyne/006192.bin'):
      'psegs://dataset=kitti-object&split=train&segment_id=kitti-object-benchmark-train&topic=lidar&extra.kitti.archive=data_object_velodyne.zip&extra.kitti.archive.file=training/velodyne/006192.bin&extra.kitti.frame=006192',
    
    ## Tracking Benchmark
    ('data_tracking_label_2.zip', 'training/label_02/0005.txt'):
      'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0005&topic=labels|cuboids&extra.kitti.archive=data_tracking_label_2.zip&extra.kitti.archive.file=training/label_02/0005.txt',
    ('data_tracking_image_2.zip', 'training/image_02/0005/000039.png'):
      'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0005&topic=camera|left&extra.kitti.frame=000039&extra.kitti.archive=data_tracking_image_2.zip&extra.kitti.archive.file=training/image_02/0005/000039.png',
    ('data_tracking_image_3.zip', 'training/image_03/0005/000039.png'):
      'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0005&topic=camera|right&extra.kitti.frame=000039&extra.kitti.archive=data_tracking_image_3.zip&extra.kitti.archive.file=training/image_03/0005/000039.png',
    ('data_tracking_velodyne.zip', 'training/velodyne/0005/000039.bin'):
      'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0005&topic=lidar&extra.kitti.frame=000039&extra.kitti.archive=data_tracking_velodyne.zip&extra.kitti.archive.file=training/velodyne/0005/000039.bin',
    ('data_tracking_oxts.zip', 'training/oxts/0005.txt'):
      'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0005&topic=ego_pose&extra.kitti.archive=data_tracking_oxts.zip&extra.kitti.archive.file=training/oxts/0005.txt',
  }

  for args, expected_out in INPUT_EXPECTED_OUT.items():
    assert (
      datum.URI.from_str(expected_out) == 
        kitti.kitti_archive_file_to_uri(*args))


## test_kitti_benchmark_to_raw_mapper Fixtures

# Breadcrumbs: 
# pdf.query(' or '.join("(benchmark == '%s' and b_filename == '%s')" % k for k in keys))
MOCK_BENCH_TO_RAW_CSV = """
pandas_id,b_digest,r_digest,benchmark,segment_category,b_filename,r_filename,split,filename,frame,nanostamp,segment,topic
2719,mock-sha-1,mock-sha-1,data_tracking_image_2.zip,road,training/image_02/0005/000039.png,2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000039.png,train,2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000039.png,39,1317042727991676,2011_09_26_drive_0015_sync,image_02
24296,mock-sha-1,mock-sha-1,data_tracking_velodyne.zip,road,training/velodyne/0005/000039.bin,2011_09_26/2011_09_26_drive_0015_sync/velodyne_points/data/0000000039.bin,train,2011_09_26/2011_09_26_drive_0015_sync/velodyne_points/data/0000000039.bin,39,1317042727981173,2011_09_26_drive_0015_sync,velodyne_points
71594,mock-sha-1,,data_tracking_image_2.zip,,testing/image_02/0009/000039.png,,test,,0,0,,
83489,mock-sha-1,mock-sha-1,data_object_image_3.zip,residential,training/image_3/006192.png,2011_09_26/2011_09_26_drive_0061_sync/image_03/data/0000000117.png,train,2011_09_26/2011_09_26_drive_0061_sync/image_03/data/0000000117.png,117,1317047755608174,2011_09_26_drive_0061_sync,image_03
96234,mock-sha-1,mock-sha-1,data_tracking_image_3.zip,road,training/image_03/0005/000039.png,2011_09_26/2011_09_26_drive_0015_sync/image_03/data/0000000039.png,train,2011_09_26/2011_09_26_drive_0015_sync/image_03/data/0000000039.png,39,1317042727991186,2011_09_26_drive_0015_sync,image_03
146436,mock-sha-1,mock-sha-1,data_object_image_2.zip,residential,training/image_2/006192.png,2011_09_26/2011_09_26_drive_0061_sync/image_02/data/0000000117.png,train,2011_09_26/2011_09_26_drive_0061_sync/image_02/data/0000000117.png,117,1317047755608672,2011_09_26_drive_0061_sync,image_02
199206,mock-sha-1,,data_object_image_2.zip,,testing/image_2/006192.png,,test,,0,0,,
278639,mock-sha-1,mock-sha-1,data_object_velodyne.zip,residential,training/velodyne/006192.bin,2011_09_26/2011_09_26_drive_0061_sync/velodyne_points/data/0000000117.bin,train,2011_09_26/2011_09_26_drive_0061_sync/velodyne_points/data/0000000117.bin,117,1317047755598192,2011_09_26_drive_0061_sync,velodyne_points
300477,mock-sha-1,mock-sha-1,data_object_prev_3.zip,residential,training/prev_3/006192_02.png,2011_09_26/2011_09_26_drive_0061_sync/image_03/data/0000000115.png,train,2011_09_26/2011_09_26_drive_0061_sync/image_03/data/0000000115.png,115,131704775540113,2011_09_26_drive_0061_sync,image_03
337804,mock-sha-1,mock-sha-1,data_object_prev_2.zip,residential,training/prev_2/006192_02.png,2011_09_26/2011_09_26_drive_0061_sync/image_02/data/0000000115.png,train,2011_09_26/2011_09_26_drive_0061_sync/image_02/data/0000000115.png,115,131704775540175,2011_09_26_drive_0061_sync,image_02
"""

def test_kitti_benchmark_to_raw_mapper_mock():
  test_tempdir = testutil.test_tempdir(
                              'test_kitti_benchmark_to_raw_mapper_mock')

  class Fixtures(kitti.Fixtures):
    @classmethod
    def bench_to_raw_path(cls):
      return test_tempdir / 'mock_bench_to_raw_df'

    @classmethod
    def index_root(cls):
      return test_tempdir / 'kitti_index_root'

  # We'll test this class that uses mock data
  class MockBenchmarkToRawMapper(kitti.BenchmarkToRawMapper):
    FIXTURES = Fixtures

  # First create a mock bench_to_raw parquet table
  MOCK_BENCH_TO_RAW_PATH = test_tempdir / 'mock_bench_to_raw_df'
  with testutil.LocalSpark.getOrCreate() as spark:
    from io import StringIO
    import pandas as pd
    pdf = pd.read_csv(StringIO(MOCK_BENCH_TO_RAW_CSV))
    pdf.fillna(value='', inplace=True)
    df = spark.createDataFrame(pdf)
    df.write.parquet(str(Fixtures.bench_to_raw_path()))

  # Now build index ...
  with testutil.LocalSpark.getOrCreate() as spark:
    MockBenchmarkToRawMapper.setup(spark)

  # ... and create a mapper to test.
  mapper = MockBenchmarkToRawMapper()
  
  # Now test BenchmarkToRawMapper logic!
  INPUT_EXPECTED_OUT = {
    ## Object Benchmark
    ('data_object_label_2.zip', 'training/label_2/006192.txt'):
      (1317047755608672,
      {'kitti.raw.segment': '2011_09_26_drive_0061_sync',
        'kitti.raw.segment_category': 'residential',
        'kitti.raw.timestamp': '1317047755608672'}),
    
    ('data_object_image_2.zip', 'training/image_2/006192.png'):
      (1317047755608672,
      {'kitti.raw.filename': '2011_09_26/2011_09_26_drive_0061_sync/image_02/data/0000000117.png',
        'kitti.raw.segment': '2011_09_26_drive_0061_sync',
        'kitti.raw.segment_category': 'residential',
        'kitti.raw.sha-1': 'mock-sha-1',
        'kitti.raw.timestamp': '1317047755608672'}),
    
    ('data_object_image_3.zip', 'training/image_3/006192.png'):
      (1317047755608174,
      {'kitti.raw.filename': '2011_09_26/2011_09_26_drive_0061_sync/image_03/data/0000000117.png',
        'kitti.raw.segment': '2011_09_26_drive_0061_sync',
        'kitti.raw.segment_category': 'residential',
        'kitti.raw.sha-1': 'mock-sha-1',
        'kitti.raw.timestamp': '1317047755608174'}),
    
    ('data_object_image_2.zip', 'testing/image_2/006192.png'):
      (620200000000,
      {}),

    ('data_object_prev_2.zip', 'training/prev_2/006192_02.png'):
      (131704775540175,
      {'kitti.raw.filename': '2011_09_26/2011_09_26_drive_0061_sync/image_02/data/0000000115.png',
        'kitti.raw.segment': '2011_09_26_drive_0061_sync',
        'kitti.raw.segment_category': 'residential',
        'kitti.raw.sha-1': 'mock-sha-1',
        'kitti.raw.timestamp': '131704775540175'}),
    
    ('data_object_prev_3.zip', 'training/prev_3/006192_02.png'):
      (131704775540113,
      {'kitti.raw.filename': '2011_09_26/2011_09_26_drive_0061_sync/image_03/data/0000000115.png',
        'kitti.raw.segment': '2011_09_26_drive_0061_sync',
        'kitti.raw.segment_category': 'residential',
        'kitti.raw.sha-1': 'mock-sha-1',
        'kitti.raw.timestamp': '131704775540113'}),
    
    ('data_object_velodyne.zip', 'training/velodyne/006192.bin'):
      (1317047755598192,
      {'kitti.raw.filename': '2011_09_26/2011_09_26_drive_0061_sync/velodyne_points/data/0000000117.bin',
        'kitti.raw.segment': '2011_09_26_drive_0061_sync',
        'kitti.raw.segment_category': 'residential',
        'kitti.raw.sha-1': 'mock-sha-1',
        'kitti.raw.timestamp': '1317047755598192'}),


    ## Tracking Benchmark
    ('data_tracking_label_2.zip', 'training/label_02/0005.txt', '000039'):
      (1317042727991676,
      {'kitti.raw.segment': '2011_09_26_drive_0015_sync',
        'kitti.raw.segment_category': 'road',
        'kitti.raw.timestamp': '1317042727991676'}),
    
    ('data_tracking_image_2.zip', 'training/image_02/0005/000039.png'):
      (1317042727991676,
      {'kitti.raw.filename': '2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000039.png',
        'kitti.raw.segment': '2011_09_26_drive_0015_sync',
        'kitti.raw.segment_category': 'road',
        'kitti.raw.sha-1': 'mock-sha-1',
        'kitti.raw.timestamp': '1317042727991676'}),
    
    ('data_tracking_image_2.zip', 'testing/image_02/0009/000039.png'):
      (4900000000,
      {}),
    
    ('data_tracking_image_3.zip', 'training/image_03/0005/000039.png'):
      (1317042727991186,
      {'kitti.raw.filename': '2011_09_26/2011_09_26_drive_0015_sync/image_03/data/0000000039.png',
        'kitti.raw.segment': '2011_09_26_drive_0015_sync',
        'kitti.raw.segment_category': 'road',
        'kitti.raw.sha-1': 'mock-sha-1',
        'kitti.raw.timestamp': '1317042727991186'}),

    ('data_tracking_velodyne.zip', 'training/velodyne/0005/000039.bin'):
      (1317042727981173,
      {'kitti.raw.filename': '2011_09_26/2011_09_26_drive_0015_sync/velodyne_points/data/0000000039.bin',
        'kitti.raw.segment': '2011_09_26_drive_0015_sync',
        'kitti.raw.segment_category': 'road',
        'kitti.raw.sha-1': 'mock-sha-1',
        'kitti.raw.timestamp': '1317042727981173'}),

    ('data_tracking_oxts.zip', 'training/oxts/0005.txt', '000039'):
      (4900000000, {}),
  }
  
  for args, expected_out in INPUT_EXPECTED_OUT.items():
    uri = kitti.kitti_archive_file_to_uri(*args[:2])
    
    # For oxts and labels test, we include frame
    if len(args) == 3:
      uri.extra['kitti.frame'] = args[-1]
    
    extra = mapper.get_extra(uri)
    mapper.fill_timestamp(uri)
    
    et, exp_extra = expected_out
    assert exp_extra == extra
    assert et == uri.timestamp


###############################################################################
## Object Benchmark

MOCK_OBJECT_LABEL = """
Cyclist 0.0 0 1. 2. 3. 14. 5. 6. 7. 8. 9. 10. 11. 3.14159 13.
Pedestrian 0.1 10 1. 2. 3. 14. 5. 6. 7. 8. 9. 10. 11. 1.570796 13.
"""


def test_kitti_object_load_label():
  from scipy.spatial.transform import Rotation as R
  
  cuboids, bboxes = kitti.parse_object_label_cuboids(MOCK_OBJECT_LABEL)

  EXPECTED_CUBOIDS = [
      datum.Cuboid(
        category_name='Cyclist', 
        extra={
          'kitti.truncated': '0.0',
          'kitti.occluded': '0',
          'kitti.score': '13.0',
          'kitti.cam_relative_yaw': '1.0',
        },
        length_meters=8.0,
        width_meters=6.0,
        height_meters=7.0,
        obj_from_ego=datum.Transform(
          rotation=R.from_euler('yzx', [3.14159, 0, 0]).as_matrix(),
          translation=[9., 7., 11.],
          src_frame='camera|left',
          dest_frame='obj'),
      ),
      datum.Cuboid(
        category_name='Pedestrian',
        extra={
          'kitti.truncated': '0.1',
          'kitti.occluded': '10',
          'kitti.score': '13.0',
          'kitti.cam_relative_yaw': '1.0',
        },
        length_meters=8.0,
        width_meters=6.0,
        height_meters=7.0,
        obj_from_ego=datum.Transform(
          rotation=R.from_euler('yzx', [1.570796, 0, 0]).as_matrix(),
          translation=[9., 7., 11.],
          src_frame='camera|left',
          dest_frame='obj'),
      ),
  ]
  assert cuboids == EXPECTED_CUBOIDS

  EXPECTED_BBOXES = [
    datum.BBox2D(
      x=2, y=3, width=13, height=3, category_name='Cyclist',
      extra={
        'kitti.truncated': '0.0',
        'kitti.occluded': '0',
        'kitti.score': '13.0',
        'kitti.cam_relative_yaw': '1.0'
    }),
    datum.BBox2D(
      x=2, y=3, width=13, height=3, category_name='Pedestrian', 
      extra={
        'kitti.truncated': '0.1', 
        'kitti.occluded': '10', 
        'kitti.score': '13.0', 
        'kitti.cam_relative_yaw': '1.0'}
  )]

  assert bboxes == EXPECTED_BBOXES


def test_kitti_object_lidar_camera_projection():
  testutil.skip_if_fixture_absent(kitti.Fixtures.EXTERNAL_FIXTURES_ROOT)

  base_dir = kitti.Fixtures.object_fixture_dir()
  outdir = testutil.test_tempdir('test_kitti_object_lidar_camera_projection')

  for frame in kitti.Fixtures.OBJ_TEST_FRAMES:
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
  assert_img_directories_equal(outdir, expected_base)


def test_kitti_object_label_camera_projection():
  testutil.skip_if_fixture_absent(kitti.Fixtures.EXTERNAL_FIXTURES_ROOT)
  
  base_dir = kitti.Fixtures.object_fixture_dir()
  outdir = testutil.test_tempdir('test_kitti_object_label_camera_projection')

  for frame in kitti.Fixtures.OBJ_TEST_FRAMES:
    calib_path = base_dir / ('training/calib/%s.txt' % frame)
    calib = kitti.Calibration.from_kitti_str(open(calib_path, 'r').read())

    cuboids, bboxes = kitti.parse_object_label_cuboids(
      open(base_dir / ('training/label_2/%s.txt' % frame), 'r').read())

    def save_projected_bboxes(camera):
      img = imageio.imread(base_dir / 'training' / camera / ('%s.png' % frame))
      for bbox in bboxes:
        h, w = img.shape[:2]
        bbox.im_height = h
        bbox.im_width = w
        bbox.draw_in_image(img)
      imageio.imwrite(
        outdir / ('projected_bboxes_%s_%s.png' % (frame, camera)), img)

    save_projected_cuboids(
      base_dir, cuboids, 
      outdir, frame,
      'image_2', calib.K2, datum.Transform(translation=calib.T2))
    save_projected_cuboids(
      base_dir, cuboids, 
      outdir, frame,
      'image_3', calib.K3, datum.Transform(translation=calib.T3))
    save_projected_bboxes('image_2')
    # We don't bother projecting bboxes to the right camera

  # Now test!
  expected_base = (
    kitti.Fixtures.EXTERNAL_FIXTURES_ROOT / 
      'test_kitti_object_label_camera_projection')
  assert_img_directories_equal(outdir, expected_base)


def test_kitti_object_label_lidar_projection():
  testutil.skip_if_fixture_absent(kitti.Fixtures.EXTERNAL_FIXTURES_ROOT)

  base_dir = kitti.Fixtures.object_fixture_dir()
  outdir = testutil.test_tempdir('test_kitti_object_label_lidar_projection')

  for frame in kitti.Fixtures.OBJ_TEST_FRAMES:
    calib_path = base_dir / ('training/calib/%s.txt' % frame)
    calib = kitti.Calibration.from_kitti_str(open(calib_path, 'r').read())

    cuboids, bboxes = kitti.parse_object_label_cuboids(
      open(base_dir / ('training/label_2/%s.txt' % frame), 'r').read())

    with open(base_dir / ('training/velodyne/%s.bin' % frame), 'rb') as f:
      raw_lidar = np.frombuffer(f.read(), dtype=np.float32).reshape((-1, 4))
    xyz = raw_lidar[:, :3]
    # unused: reflectance = raw_lidar[:, 3:]

    save_labels_projected_to_lidar(
      base_dir, outdir, frame, calib, xyz, cuboids)
    
  # Now test!
  expected_base = (
    kitti.Fixtures.EXTERNAL_FIXTURES_ROOT / 
      'test_kitti_object_label_lidar_projection')
  assert_img_directories_equal(outdir, expected_base)


###############################################################################
## Tracking Benchmark

MOCK_TRACKING_LABEL = """
0 1 Cyclist 0.0 0 1. 2. 3. 14. 5. 6. 7. 8. 9. 10. 11. 3.14159 13.
0 2 Pedestrian 0.1 10 1. 2. 3. 14. 5. 6. 7. 8. 9. 10. 11. 1.570796 13.
1 1 Cyclist 0.0 0 1. 2. 3. 14. 5. 6. 7. 8. 9. 10. 11. 3.14159 13.
1 2 Pedestrian 0.1 10 1. 2. 3. 14. 5. 6. 7. 8. 9. 10. 11. 1.570796 13.
"""


def test_kitti_tracking_load_label():
  from scipy.spatial.transform import Rotation as R
  
  f_to_cuboids, f_to_bboxes = kitti.parse_tracking_label_cuboids(
                                                  MOCK_TRACKING_LABEL)

  assert 0 in f_to_bboxes
  assert 1 in f_to_bboxes
  assert 0 in f_to_cuboids
  assert 1 in f_to_cuboids
  cuboids = f_to_cuboids[0]

  # Note: these are identical to the cuboids listed in
  # test_kitti_object_load_label(), except these
  # have track_id and frame_num populated.
  EXPECTED_CUBOIDS = [
      datum.Cuboid(
        category_name='Cyclist', 
        track_id='1',
        extra={
          'kitti.truncated': '0.0',
          'kitti.occluded': '0',
          'kitti.score': '13.0',
          'kitti.cam_relative_yaw': '1.0',
          'kitti.track_id': '1',
          'kitti.frame_num': '0'
        },
        length_meters=8.0,
        width_meters=6.0,
        height_meters=7.0,
        obj_from_ego=datum.Transform(
          rotation=R.from_euler('yzx', [3.14159, 0, 0]).as_matrix(),
          translation=[9., 7., 11.],
          src_frame='camera|left',
          dest_frame='obj'),
      ),
      datum.Cuboid(
        category_name='Pedestrian',
        track_id='2',
        extra={
          'kitti.truncated': '0.1',
          'kitti.occluded': '10',
          'kitti.score': '13.0',
          'kitti.cam_relative_yaw': '1.0',
          'kitti.track_id': '2',
          'kitti.frame_num': '0'
        },
        length_meters=8.0,
        width_meters=6.0,
        height_meters=7.0,
        obj_from_ego=datum.Transform(
          rotation=R.from_euler('yzx', [1.570796, 0, 0]).as_matrix(),
          translation=[9., 7., 11.],
          src_frame='camera|left',
          dest_frame='obj'),
      ),
  ]
  assert cuboids == EXPECTED_CUBOIDS


def test_kitti_tracking_lidar_camera_projection():
  testutil.skip_if_fixture_absent(kitti.Fixtures.EXTERNAL_FIXTURES_ROOT)

  base_dir = kitti.Fixtures.tracking_fixture_dir()
  outdir = testutil.test_tempdir('test_kitti_tracking_lidar_camera_projection')

  for frame in kitti.Fixtures.TRACKING_TEST_FRAMES:
    seq, frame_num = frame.split('/')
    calib_path = base_dir / ('training/calib/%s.txt' % seq)
    calib = kitti.Calibration.from_kitti_str(open(calib_path, 'r').read())
    
    save_projected_lidar(
      base_dir, outdir, frame, 'image_02', calib.K2, calib.velo_to_cam_2_rect)
    save_projected_lidar(
      base_dir, outdir, frame, 'image_03', calib.K3, calib.velo_to_cam_3_rect)

  # Now test!
  expected_base = (
    kitti.Fixtures.EXTERNAL_FIXTURES_ROOT / 
      'test_kitti_tracking_lidar_camera_projection')
  assert_img_directories_equal(outdir, expected_base)


def test_kitti_tracking_label_camera_projection():
  testutil.skip_if_fixture_absent(kitti.Fixtures.EXTERNAL_FIXTURES_ROOT)

  base_dir = kitti.Fixtures.tracking_fixture_dir()
  outdir = testutil.test_tempdir('test_kitti_tracking_label_camera_projection')

  for frame in kitti.Fixtures.TRACKING_TEST_FRAMES:
    seq, frame_num = frame.split('/')
    frame_num = int(frame_num)
    
    calib_path = base_dir / ('training/calib/%s.txt' % seq)
    calib = kitti.Calibration.from_kitti_str(open(calib_path, 'r').read())
    
    f_to_cuboids, f_to_bboxes = kitti.parse_tracking_label_cuboids(
      open(base_dir / ('training/label_02/%s.txt' % seq), 'r').read())

    assert frame_num in f_to_bboxes
    assert frame_num in f_to_cuboids
    cuboids = f_to_cuboids[frame_num]

    save_projected_cuboids(
      base_dir, cuboids,
      outdir, frame, 
      'image_02', calib.K2, datum.Transform(translation=calib.T2))
    save_projected_cuboids(
      base_dir, cuboids, 
      outdir, frame, 
      'image_03', calib.K3, datum.Transform(translation=calib.T3))
    
    # We don't bother testing bboxes

  # Now test!
  expected_base = (
    kitti.Fixtures.EXTERNAL_FIXTURES_ROOT / 
      'test_kitti_tracking_label_camera_projection')
  assert_img_directories_equal(outdir, expected_base)


def test_kitti_tracking_label_lidar_projection():
  testutil.skip_if_fixture_absent(kitti.Fixtures.EXTERNAL_FIXTURES_ROOT)

  base_dir = kitti.Fixtures.tracking_fixture_dir()
  outdir = testutil.test_tempdir('test_kitti_tracking_label_lidar_projection')

  for frame in kitti.Fixtures.TRACKING_TEST_FRAMES:
    seq, frame_num = frame.split('/')
    frame_num = int(frame_num)
    
    calib_path = base_dir / ('training/calib/%s.txt' % seq)
    calib = kitti.Calibration.from_kitti_str(open(calib_path, 'r').read())
    
    f_to_cuboids, _ = kitti.parse_tracking_label_cuboids(
      open(base_dir / ('training/label_02/%s.txt' % seq), 'r').read())

    assert frame_num in f_to_cuboids
    cuboids = f_to_cuboids[frame_num]

    with open(base_dir / ('training/velodyne/%s.bin' % frame), 'rb') as f:
      raw_lidar = np.frombuffer(f.read(), dtype=np.float32).reshape((-1, 4))
    xyz = raw_lidar[:, :3]
    # unused: reflectance = raw_lidar[:, 3:]

    save_labels_projected_to_lidar(base_dir, outdir, frame, calib, xyz, cuboids)
    
  # Now test!
  expected_base = (
    kitti.Fixtures.EXTERNAL_FIXTURES_ROOT / 
      'test_kitti_tracking_label_lidar_projection')
  assert_img_directories_equal(outdir, expected_base)


###############################################################################
## Stamped Datum Table


def test_kitti_sd_table_tracking():
  testutil.skip_if_fixture_absent(kitti.Fixtures.ROOT)
  
  TEST_TEMPDIR = testutil.test_tempdir('test_kitti_sd_table_tracking')

  class Fixtures(kitti.Fixtures):
    @classmethod
    def index_root(cls):
      return TEST_TEMPDIR / 'kitti_index_root'
    
  class TrackingTestTable(kitti.KITTISDTable):
    INCLUDE_OBJECT_BENCHMARK = False
    FIXTURES = Fixtures

    @classmethod
    def table_root(cls):
      return TEST_TEMPDIR / 'sd_table'
  
  with testutil.LocalSpark.sess() as spark:
    suri = datum.URI.from_str(
      'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0012')
    TrackingTestTable.build(spark, only_segments=[suri])

    df = TrackingTestTable.as_df(spark)
    df.createOrReplaceTempView('seg')
    spark.sql(""" SELECT uri.topic AS topic, count(*) AS N, MAX(uri.timestamp),  MIN(uri.timestamp) FROM seg GROUP BY topic """).show()
    import pdb; pdb.set_trace()
    print() # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


EXPECTED_SEGMENTS = (
  'psegs://dataset=kitti-object&split=test&segment_id=kitti-object-benchmark-test',
  'psegs://dataset=kitti-object&split=train&segment_id=kitti-object-benchmark-train',
  'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0000', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0001', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0002', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0003', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0004', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0005', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0006', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0007', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0008', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0009', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0010', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0011', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0012', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0013', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0014', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0015', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0016', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0017', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0018', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0019', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0020', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0021', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0022', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0023', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0024', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0025', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0026', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0027', 'psegs://dataset=kitti-tracking&split=test&segment_id=kitti-tracking-test-0028',
  'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0000', 'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0001', 'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0002', 'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0003', 'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0004', 'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0005', 'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0006', 'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0007', 'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0008', 'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0009', 'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0010', 'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0011', 'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0012', 'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0013', 'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0014', 'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0015', 'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0016', 'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0017', 'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0018', 'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0019', 'psegs://dataset=kitti-tracking&split=train&segment_id=kitti-tracking-train-0020',
)

def test_kitti_all_segment_uris():
  testutil.skip_if_fixture_absent(kitti.Fixtures.ROOT)
  actual = kitti.KITTISDTable.get_all_segment_uris()
  assert sorted(EXPECTED_SEGMENTS) == sorted(str(uri) for uri in actual)


###############################################################################
## DSUtil Tests

def test_kitti_dsutil_smoke():
  testutil.skip_if_fixture_absent(kitti.Fixtures.ROOT)
  testutil.skip_if_fixture_absent(kitti.Fixtures.EXTERNAL_FIXTURES_ROOT)

  # The above are preconditions, so this should succeed:
  assert kitti.DSUtil.emplace()
