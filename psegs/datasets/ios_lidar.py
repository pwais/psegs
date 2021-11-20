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



## 3DScannerApp Data Parsing
# The utilities below help parse "raw" data exported from 3DScannerApp:
# https://www.3dscannerapp.com/
#
# To create data:
#  * Download the app and do a scan
#  * Share your scan as "All Data" or connect your device to a computer and
#      download the files over USB
#  * The downloaded directory has files like:
#      frame_XXXXX.json - pose of the device at frame N
#      frame_XXXXX.jpg  - image captured at frame N
#      info.json - has GPS data and other context
#      export.obj - raw fused mesh (unclear if this is from 
#         Apple's fusion or not)
# 
#    If the capture was recorded in "low res" mode, the directory additionally
#    has files like:
#      depth_XXXXX.png - raw depth info as a 16-bit png (depth in millimeters)
#      conf_XXXXX.png - sensor confidence for depth (Apple's ARConfidenceLevel)
#
# Parsing code references:
#  * ARBodyPoseRecorder from the developer of 3DScannerApp:
#     https://github.com/laanlabs/ARBodyPoseRecorder/blob/9e7a37cdfdb44bc223f7b983481841696a763782/ARBodyPoseRecorder/ViewController.swift#L233
#  * rtabmap ( http://introlab.github.io/rtabmap/ ) code that appears to
#     parse 3DScannerApp output:
#     https://docs.ros.org/en/api/rtabmap/html/CameraImages_8cpp_source.html

import json
import os
from pathlib import Path

import numpy as np

from psegs import datum
from psegs import util
from psegs.conf import C
from psegs.table.sd_table import StampedDatumTableBase


"""
Apple camera frame:
 +x is right
"""


def threeDScannerApp_get_ego_pose(json_data):
  # The device ego pose is in a GPS-based coordinate frame where:
  #  +y is "up" based upon *gravity*
  #  +x is GPS East
  #  +z is GPS South
  # https://developer.apple.com/documentation/arkit/arconfiguration/worldalignment/gravityandheading
  T_raw = json_data['cameraPoseARFrame'] # A row-major 4x4 matrix
  T_arr_raw = np.array(T_raw).reshape([4, 4])

  # Based upon rtabmap noted above
  # Rotate from OpenGL coordinate frame to 
  # +x = forward, +y = left, +z = up
  OPENGTL_T_WORLD = np.array([
    [ 0, -1,  0,  0],
    [ 0,  0,  1,  0],
    [-1,  0,  0,  0],
    [ 0,  0,  0,  1],
  ])

  WORLD_T_OPENGL = np.array([
    [ 0,  0, -1,  0],
    [-1,  0,  0,  0],
    [ 0,  1,  0,  0],
    [ 0,  0,  0,  1],
  ])

  WORLD_T_PSEGS = np.array([
    [ 0,  0, -1,  0],
    [-1,  0,  0,  0],
    [ 0,  1,  0,  0],
    [ 0,  0,  0,  1],
  ])

  # pose = T_arr_raw @ WORLD_T_PSEGS
  # pose = WORLD_T_OPENGL * T_arr_raw[:3, :4]
  pose = T_arr_raw[:3, :4]
  # assert False, (T_arr_raw, pose)
  return datum.Transform.from_transformation_matrix(
            pose,
            src_frame='ego',
            dest_frame='world')
  

def threeDScannerApp_get_K(json_data):
  K_raw = json_data['intrinsics']
  f_x = K_raw[0]
  f_y = K_raw[4]
  c_x = K_raw[2]
  c_y = K_raw[5]

  K = np.array([
        [f_x,   0, c_x],
        [0,   f_y, c_y],
        [0,     0,   1],
  ])

  return K


def threeDScannerApp_frame_id_from_fname(path):
    path = os.path.basename(path)
    prefix = path.split('.')[0]
    toks = prefix.split('_')
    assert len(toks) == 2, toks
    frame_id = toks[-1]
    return frame_id


def threeDScannerApp_create_frame_to_timestamp(scene_dir):
  """Unfortunately, 3D Scanner App only provides 'timestamps' in the form of
  CACurrentMediaTime (which is mach_absolute_time or *system uptime*).
  We want more accurate unix nanostamps in order to:
    (1) be consistent with the rest of PSegs
    (2) join 3D Scanner App data with other data sources / sensors external
          to the iPhone
  
  So can we resolve nanostamps?
  3D Scanner App does name scenes (by default) using a second-resolution 
  timestamp, but that's in the local timezone, and the name (i.e. folder name)
  is not hard to change / break.

  It appears that the very first image recorded (i.e. frame_00000.jpg) has
  the same unix mtime as the timestamp used for the default scene name.  
  Unfortunately, this timstamp only has 1-second resolution, but without
  further information, we assume that the mtime of frame_00000.jpg
  corresponds to the uptime recorded in frame_00000.json (the info / pose)
  file for the 0th frame).
  
  This helper creates and returns a map of frame id -> nanostamp, including
  for frames that do not have images (e.g. in low-res capture mode).

  """

  import os
  import json
  from oarphpy import util as oputil

  frame_0_img_path = Path(scene_dir) / 'frame_00000.jpg'
  assert os.path.exists(frame_0_path), \
    f"Can't estimate time, no image {frame_0_img_path}"

  frame_0_info_path = Path(scene_dir) / 'frame_00000.json'
  assert os.path.exists(frame_0_info_path), \
    f"Can't estimate time, no info {frame_0_info_path}"

  base_stamp = os.path.getmtime(frame_0_img_path)
  base_nanostamp = int(1e9 * base_stamp)

  with open(frame_0_info_path, 'r') as f:
    info = json.load(f)
  base_CACurrentMediaTime = info['time']
    # CACurrentMediaTime, which is mach_absolute_time, which is
    # *system uptime*. Has microsecond resolution, due to being provided
    # here as a time in seconds (float format)?
  
  # Now infer timestamps for all frames
  frame_info_paths = oputil.all_paths_recursive(
                        scene_dir, pattern='frame_*.json')
  frame_info_paths.sort()
  
  frame_id_to_nanostamp = {}
  for path in frame_info_paths:
    frame_id = threeDScannerApp_frame_id_from_fname(path)
    
    with open(path, 'r') as f:
      info = json.load(f)
    frame_CACurrentMediaTime = info['time']
    frame_offset_sec = frame_CACurrentMediaTime - base_CACurrentMediaTime
    frame_nanostamp = int(1e9 * frame_offset_sec) + base_nanostamp
      
    frame_id_to_nanostamp[frame_id] = frame_nanostamp
  return frame_id_to_nanostamp


def threeDScannerApp_create_camera_image(frame_json_path, timestamp=None):

  assert os.path.exists(frame_json_path), frame_json_path
  frame_img_path = frame_json_path.replace('.json', '.jpg')
  assert os.path.exists(frame_img_path), frame_img_path

  with open(frame_json_path, 'r') as f:
    json_data = json.load(f)
  
  ego_pose = threeDScannerApp_get_ego_pose(json_data)
  K = threeDScannerApp_get_K(json_data)

  if timestamp is None:
    timestamp = int(json_data['time'] * 1e9)
      # CACurrentMediaTime, which is mach_absolute_time, which is
      # *system uptime*.  We use this as a fallback unless the caller
      # has resolved timestamps for the whole scene.

  REQUIRED_KEYS = (
    'averageAngularVelocity',
    'averageVelocity',
    'exposureDuration',
    'frame_index',
  )
  SKIP_KEYS = (
    'cameraPoseARFrame',
    'intrinsics'
    'time',
  )
  
  extra = dict(
            ('threeDScannerApp.' + k, json.dumps(v))
            for k, v in json_data.items()
            if k not in SKIP_KEYS)
  assert set(REQUIRED_KEYS) - set(json_data.keys()) == set(), \
    "Have %s wanted %s" % (extra.keys(), REQUIRED_KEYS)

  extra['threeDScannerApp.frame_json_name'] = os.path.basename(frame_json_path)

  WORLD_T_PSEGS = np.array([
    [ 0,  0, -1,  0],
    [-1,  0,  0,  0],
    [ 0,  1,  0,  0],
    [ 0,  0,  0,  1],
  ])

  PSEGS_T_IOS_CAM = np.array([
    [ 0, -1,  0,  0],
    [ 0,  0,  1,  0],
    [-1,  0,  0,  0],
    [ 0,  0,  0,  1],
  ])


  # https://docs.ros.org/en/api/rtabmap/html/classrtabmap_1_1CameraModel.html#a0853af9d0117565311da4ffc3965f8d2
  # https://developer.apple.com/documentation/arkit/arcamera/2866108-transform
  #   Apple camera frame is:
  #     +x is right when device is in lanscape; along the device long edge
  #     +y is up when device is in landscape
  #     +z is out of the device screen
  ego_to_sensor = datum.Transform(
            rotation=np.array([
              # [ 0,  0,  1],
              # [-1,  0,  0],
              # [ 0, -1,  0],
              # [ 0,   0,  -1],
              # [-1,   0,   0],
              # [ 0,  -1,   0],
              
              # [ 0,  -1,   0],
              # [ 0,   0,   1],
              # [-1,   0,   0],
              [ 1,   0,   0],
              [ 0,  -1,   0],
              [ 0,   0,  -1],
            ]),
            src_frame='camera_front',
            dest_frame='ego')
  
  from oarphpy import util as oputil
  with open(frame_img_path, 'rb') as f:
    w, h = oputil.get_jpeg_size(f.read(1024))

  import imageio
  image_factory = lambda: imageio.imread(frame_img_path)

  ci = datum.CameraImage(
                sensor_name='camera_front',
                image_factory=image_factory,
                height=h,
                width=w,
                timestamp=timestamp,
                ego_pose=ego_pose,
                ego_to_sensor=ego_to_sensor,
                K=K,
                extra=extra)

  return ci


def threeDScannerApp_get_segment_id(scan_dir='', info_path=''):
  if not info_path:
    info_path = str(Path(scan_dir) / 'info.json')
  seg_dir = os.path.dirname(info_path)
  with open(info_path, 'r') as f:
    info = json.load(f)
  segment_id = info.get('title', os.path.split(seg_dir)[-1])
  return segment_id


def threeDScannerApp_get_uris_from_scan_dir(scan_dir):
  from oarphpy import util as oputil

  uris = []
  segment_id = threeDScannerApp_get_segment_id(scan_dir=scan_dir)

  frame_to_t = threeDScannerApp_create_frame_to_timestamp(scan_dir)

  mesh_paths = oputil.all_files_recursive(scan_dir, pattern='*.obj')
  for mesh_path in mesh_paths:
    start_t = min(frame_to_t.values())
    frame_id = min(frame_to_t.keys())
    mesh_uri = datum.URI(
                  segment_id=segment_id,
                  topic='lidar|mesh',
                  timestamp=start_t,
                  extra={
                    'threeDScannerApp.frame_id': frame_id,
                    'threeDScannerApp.mesh_path': os.path.basename(mesh_path),
                  })
    uris.append(mesh_uri)

  # Sometimes the frame json info data gets lost.  Without that data,
  # we can't deduce timestamps nor transforms.  So just ignore dropped
  # frames.
  info_paths = oputil.all_files_recursive(scan_dir, pattern='frame*.json')
  for info_path in info_paths:
    frame_id = threeDScannerApp_frame_id_from_fname(info_path)
    t = frame_to_t[frame_id]

    xform_uri = datum.URI(
                  segment_id=segment_id,
                  topic='ego_pose',
                  timestamp=t,
                  extra={
                    'threeDScannerApp.frame_id': frame_id,
                    'threeDScannerApp.json_path': os.path.basename(info_path),
                  })
    uris.append(xform_uri)

    img_path = info_path.replace('.json', '.jpg')
    if os.path.exists(img_path):
      # NB: for 'low-res' capture mode, Depth gets recorded at ~6Hz but images
      # only at ~2Hz.  Also, sometimes images just don't get recorded
      # (dropped frames)
      ci_uri = datum.URI(
                  segment_id=segment_id,
                  topic='camera|front',
                  timestamp=t,
                  extra={
                    'threeDScannerApp.frame_id': frame_id,
                    'threeDScannerApp.img_path': os.path.basename(img_path),
                  })
      uris.append(ci_uri)
    
    depth_path = scan_dir / f'depth_{frame_id}.png'
    conf_path = scan_dir / f'conf_{frame_id}.png'
    if os.path.exist(depth_path):
      # NB: raw depth only available when app is in 'low-res' mode
      pc_uri = datum.URI(
                  segment_id=segment_id,
                  topic='lidar|front',
                  timestamp=t,
                  extra={
                    'threeDScannerApp.frame_id': frame_id,
                    'threeDScannerApp.depth_path': os.path.basename(depth_path),
                    'threeDScannerApp.conf_path': os.path.basename(conf_path),
                  })
      uris.append(pc_uri)
    
  return uris

  
  


def threeDScannerApp_convert_raw_to_opend3d_rgbd(input_dir, output_dataset_dir):
  from oarphpy import util as oputil
  import imageio

  output_dir_image = os.path.join(output_dataset_dir, 'image')
  output_dir_depth = os.path.join(output_dataset_dir, 'depth')
  output_dir_debug = os.path.join(output_dataset_dir, 'debug')

  threeDScannerApp_convert_raw_to_sync_rgbd(
    input_dir,
    output_dir_image,
    output_dir_depth=output_dir_depth,
    output_dir_debug=output_dir_debug)
  
  # Pick a frame and get the intrinstics
  input_rgb_paths = oputil.all_files_recursive(input_dir, pattern='frame*.jpg')
  
  sample_image = imageio.imread(input_rgb_paths[0])
  h, w = sample_image.shape[:2]

  K = None
  frame_jsons = oputil.all_files_recursive(input_dir, pattern='frame*.json')
  for path in frame_jsons:
    with open(path, 'r') as f:
      json_data = json.load(f)
    if 'intrinsics' not in json_data:
      continue
    else:
      K = threeDScannerApp_get_K(json_data)
      break
  
  assert K is not None
  f_x = K[0][0]
  f_y = K[1][1]
  c_x = K[0][2]
  c_y = K[1][2]

  # See example https://github.com/isl-org/Open3D/blob/a27456cc9f4cd43744e87c3e65a9bf196c0e5526/examples/python/reconstruction_system/sensors/realsense_recorder.py#L69
  opend3d_calib_data = {
    'width': w,
    'height': h,
    'intrinsic_matrix': [
        # Column-major !
        f_x, 0, 0, 0, f_y, 0, c_x, c_y, 1
    ],
  }

  output_intrinsics_path = os.path.join(output_dataset_dir, 'intrinsic.json')
  with open(output_intrinsics_path, 'w') as f:
    json.dump(opend3d_calib_data, f, indent=2)
  util.log.info("Saved intrinsics to %s" % output_intrinsics_path)


def threeDScannerApp_convert_raw_to_sync_rgbd(
      input_dir,
      output_dir,
      scale_depth_to_match_visible=True,
      out_id_zfill=8,
      rgb_prefix='image_',
      depth_prefix='depth_',
      ignore_depth_below_ARConfidenceLevel=1, # ARConfidenceLevel.medium
      include_debug=True,
      output_dir_depth=None,
      output_dir_debug=None,
      parallel=-1):
  
  ## Get Input
  from oarphpy import util as oputil
  input_rgb_paths = oputil.all_files_recursive(
                          input_dir, pattern='frame*.jpg')
  input_depth_paths = oputil.all_files_recursive(
                          input_dir, pattern='depth*.png')
  assert input_rgb_paths
  assert input_depth_paths

  if output_dir_depth is None:
    output_dir_depth = output_dir
  if output_dir_debug is None:
    output_dir_debug = output_dir
  oputil.mkdir(str(output_dir))
  oputil.mkdir(str(output_dir_depth))
  oputil.mkdir(str(output_dir_debug))

  ## Get Input Dimensions
  import imageio
  sample_img = imageio.imread(input_rgb_paths[0])
  rgb_hw = sample_img.shape[:2]
  util.log.info("Have RGB of resolution %s" % (rgb_hw,))

  sample_depth = imageio.imread(input_depth_paths[0])
  depth_hw = sample_depth.shape[:2]
  util.log.info("Have depth of resolution %s" % (depth_hw,))

  ## Define what we need to do
  def convert(in_rgb, in_depth, out_id):
    import shutil
    import cv2
    import imageio

    out_id_str = str(out_id).zfill(out_id_zfill)

    rgb_suffix = in_rgb.split('.')[-1]
    rgb_suffix = '.' + rgb_suffix
    rgb_dest = os.path.join(output_dir, rgb_prefix + out_id_str + rgb_suffix)
    shutil.copyfile(in_rgb, rgb_dest)
    util.log.info("%s -> %s" % (in_rgb, rgb_dest))

    depth = imageio.imread(in_depth)
    confidence = imageio.imread(in_depth.replace('depth_', 'conf_'))

    if scale_depth_to_match_visible:
      w, h = rgb_hw[1], rgb_hw[0]
      depth = cv2.resize(depth, (w, h))
      confidence = cv2.resize(confidence, (w, h))

    # Zero out depth with low confidence
    depth[ confidence < ignore_depth_below_ARConfidenceLevel ] = 0

    depth_dest = os.path.join(
                    output_dir_depth, depth_prefix + out_id_str + '.png')
    imageio.imwrite(depth_dest, depth)
    util.log.info("%s -> %s" % (in_depth, depth_dest))

    if include_debug:
      from psegs.util import plotting as pspl

      if depth is None:
        depth = imageio.imread(depth_dest)
      
      # millimeters -> meters
      depth = depth.astype(np.float32) * .001
      
      debug = imageio.imread(in_rgb)
      pspl.draw_depth_in_image(debug, depth, period_meters=.1)

      debug_dest = os.path.join(
                    output_dir_debug, 'debug_' + out_id_str + '.jpg')  
      imageio.imwrite(debug_dest, debug)
      util.log.info("Saved debug %s" % debug_dest)
    
    if True:
      print('hacks!')
      frame_json_path = in_rgb.replace('.jpg', '.json')
      if os.path.exists(frame_json_path):
        with open(frame_json_path, 'r') as f:
          json_data = json.load(f)
        
        ego_pose = threeDScannerApp_get_ego_pose(json_data)
        xform = ego_pose.get_transformation_matrix(homogeneous=True)
        xform_dest = rgb_dest + '.xform.npz'
        with open(xform_dest, 'wb') as f:
          np.save(f, xform)
        print('wrote', xform_dest)

    return rgb_dest, depth_dest

  ## Set up conversion jobs
  def get_frame_idx(path):
    fname = os.path.basename(path)
    return int(fname.split('.')[0].split('_')[1])

  frame_to_img = dict((get_frame_idx(p), p) for p in input_rgb_paths)
  frame_to_d = dict((get_frame_idx(p), p) for p in input_depth_paths)
  
  matched_frames = set(frame_to_img.keys()) & set(frame_to_d.keys())
  util.log.info("Have %s frames to convert ..." % len(matched_frames))

  frame_out_id = [
    (frame_id, out_id)
    for out_id, frame_id in enumerate(sorted(matched_frames))
  ]
  jobs = [
    (frame_to_img[f], frame_to_d[f], out_id)
    for f, out_id in frame_out_id
  ]

  ## Run conversion!
  out_path_pairs = []
  if parallel is None:
    for j in jobs:
      result = convert(*j)
      out_path_pairs.append(result)
  else:
    from psegs.spark import Spark  
    
    with Spark.sess() as spark:
      if parallel < 0:
        import multiprocessing
        parallel = multiprocessing.cpu_count()
      job_rdd = spark.sparkContext.parallelize(jobs, numSlices=parallel)
      out_path_pairs = job_rdd.map(lambda j: convert(*j)).collect()
  
  util.log.info("... converted %s." % len(jobs))
  return out_path_pairs





###############################################################################
### iOS Lidar Fixtures & Other Constants

class Fixtures(object):

  # ROOT = C.EXT_DATA_ROOT / 'kitti_archives'

  # OBJECT_BENCHMARK_FNAMES = (
  #   'data_object_label_2.zip',
  #   'data_object_image_2.zip',
  #   'data_object_image_3.zip',
  #   'data_object_prev_2.zip',
  #   'data_object_prev_3.zip',
  #   'data_object_velodyne.zip',
  #   'data_object_calib.zip',
  # )

  # TRACKING_BENCHMARK_FNAMES = (
  #   'data_tracking_label_2.zip',
  #   'data_tracking_image_2.zip',
  #   'data_tracking_image_3.zip',
  #   'data_tracking_velodyne.zip',
  #   'data_tracking_oxts.zip',
  #   'data_tracking_calib.zip',
  # )

  # @classmethod
  # def zip_path(cls, zipname):
  #   return cls.ROOT / zipname


  ### Extension Data ##########################################################
  ### See https://github.com/pwais/psegs-ios-lidar-ext

  EXT_DATA_ROOT = C.EXT_DATA_ROOT / 'psegs-ios-lidar-ext'

  @classmethod
  def threeDScannerApp_data_root(cls):
    return cls.EXT_DATA_ROOT / 'threeDScannerApp_data'

  @classmethod
  def get_threeDScannerApp_uri_to_segment_dir(cls):
    from oarphpy import util as oputil
    all_info_paths = oputil.all_files_recursive(
                        str(cls.threeDScannerApp_data_root()),
                        pattern='info.json')
    uri_to_segment_dir = {}
    for info_path in all_info_paths:
      uri_to_segment_dir threeDScannerApp_get_segment_id(info_path=info_path)
      uri = datum.URI(
              dataset='psegs-ios-lidar-ext',
              split='threeDScannerApp_data',
              segment_id=segment_id)
      uri_to_segment_dir[str(uri.to_segment_uri())] = seg_dir
    return uri_to_segment_dir

  # @classmethod
  # def bench_to_raw_path(cls):
  #   return cls.EXT_DATA_ROOT / 'bench_to_raw_df'

  @classmethod
  def index_root(cls):
    """A r/w place to cache any temp / index data"""
    return C.PS_TEMP / 'psegs_ios_lidar'


  ### Testing #################################################################

  TEST_FIXTURES_ROOT = Path('/tmp/psegs_ios_lidar_test_fixtures')
  

  ### DSUtil Auto-download ####################################################

  @classmethod
  def maybe_emplace_psegs_ios_lidar_ext(cls):
    from oarphpy import util as oputil

    if not cls.EXT_DATA_ROOT.exists():
      util.log.info("Emplacing PSegs iOS Lidar Extension data ...")
      oputil.mkdir(str(cls.EXT_DATA_ROOT))

      util.log.info("... downloading PSegs iOS Lidar Extension data ...")
      oputil.run_cmd(
        "git clone https://github.com/pwais/psegs-ios-lidar-ext %s" % \
          cls.EXT_DATA_ROOT)

    # if not cls.TEST_FIXTURES_ROOT.exists():
    #   from oarphpy import util as oputil
    #   util.log.info("Emplacing PSegs iOS Lidar Extension data ...")
    #   oputil.mkdir(str(cls.index_root()))
    #   oputil.mkdir(str(cls.TEST_FIXTURES_ROOT))
    #   ext_root = cls.index_root() / 'ext_tmp'
    #   if not ext_root.exists():
    #     util.log.info("... downloading PSegs iOS Lidar Extension data ...")
    #     oputil.run_cmd(
    #       "git clone https://github.com/pwais/psegs-ios-lidar-ext %s" % \
    #         ext_root)

    #   util.log.info("... emplacing PSegs iOS Lidar Extension data ...")
    #   def move(src, dest):
    #     oputil.mkdir(dest.parent)
    #     oputil.run_cmd("mv %s %s" % (src, dest))
    #   move(
    #     ext_root / 'threeDScannerApp_data',
    #     cls.EXTERNAL_FIXTURES_ROOT / 'threeDScannerApp_data')
    
    #   util.log.info("... emplace success!")
    #   util.log.info("(You can remove %s if needed)" % ext_root)


class KITTISDTable(StampedDatumTableBase):
  
  FIXTURES = Fixtures

  ## Subclass API

  @classmethod
  def _get_all_segment_uris(cls):
    uris = set()
    
    uri_to_seg_dir = cls.FIXTURES.get_threeDScannerApp_uri_to_segment_dir()
    uris |= set(uri_to_seg_dir.keys())

    return sorted(datum.URI.from_str(uri) for uri in uris)

  @classmethod
  def _create_datum_rdds(cls, spark, existing_uri_df=None, only_segments=None):



    ## First build indices (saves several minutes per worker per chunk) ...
    class SDBenchmarkToRawMapper(BenchmarkToRawMapper):
      FIXTURES = cls.FIXTURES
    SDBenchmarkToRawMapper.setup(spark=spark)

    ## ... now build a set of tasks to do ...
    archive_paths = cls._get_all_archive_paths()
    task_rdd = _rdd_of_all_archive_datafiles(spark, archive_paths)
    task_rdd = task_rdd.cache()
    util.log.info("Discovered %s tasks ..." % task_rdd.count())
    
    ## ... convert to URIs and filter those tasks if necessary ...
    if existing_uri_df:
      # Since we keep track of the original archives and file names, we can
      # just filter on those.  We'll collect them in this process b/c the
      # maximal set of URIs is smaller than RAM.
      def to_task(row):
        return (row.extra.get('kitti.archive'),
                row.extra.get('kitti.archive.file'))
      skip_tasks = set(
        existing_uri_df.select('extra').rdd.map(to_task).collect())
      
      task_rdd = task_rdd.filter(lambda t: t not in skip_tasks)
      util.log.info(
        "Resume mode: have datums for %s datums; dropped %s tasks" % (
          existing_uri_df.count(), len(skip_tasks)))
    
    uri_rdd = task_rdd.map(lambda task: kitti_archive_file_to_uri(*task))
    if only_segments:
      util.log.info(
        "Filtering to only %s segments" % len(only_segments))
      uri_rdd = uri_rdd.filter(
        lambda uri: any(
          suri.soft_matches_segment(uri) for suri in only_segments))

    ## ... run tasks and create stamped datums.
    # from oarphpy.spark import cluster_cpu_count
    URIS_PER_CHUNK = os.cpu_count() * 64 # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ make class member so can configure to RAM
    uris = uri_rdd.collect()
    util.log.info("... creating datums for %s URIs." % len(uris))

    datum_rdds = []
    for chunk in oputil.ichunked(uris, URIS_PER_CHUNK):
      chunk_uri_rdd = spark.sparkContext.parallelize(chunk)
      datum_rdd = chunk_uri_rdd.flatMap(cls._iter_datums_from_uri)
      datum_rdds.append(datum_rdd)
      # if len(datum_rdds) >= 10:
      #   break # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return datum_rdds
  
  @classmethod
  def _get_all_archive_paths(cls):
    archives = []
    if cls.INCLUDE_OBJECT_BENCHMARK:
      archives += list(cls.FIXTURES.OBJECT_BENCHMARK_FNAMES)
      if not cls.INCLUDE_OBJ_PREV_FRAMES:
        archives = [arch for arch in archives if 'prev' not in arch]
    if cls.INCLUDE_TRACKING_BENCHMARK:
      archives += list(cls.FIXTURES.TRACKING_BENCHMARK_FNAMES)
    archives = [arch for arch in archives if 'calib' not in arch]
    paths = [cls.FIXTURES.zip_path(arch) for arch in archives]
    return paths


  ## Datum Construction Support







  @classmethod
  def _get_file_bytes(cls, uri=None, archive=None, entryname=None):
    """Read bytes for the file referred to by `uri`"""

    if uri is not None:
      archive = uri.extra['kitti.archive']
      entryname = uri.extra['kitti.archive.file']
    assert archive and entryname

    # Cache the Zipfiles for faster loading
    if not hasattr(cls, '_get_file_bytes_archives'):
      cls._get_file_bytes_archives = {}
    if archive not in cls._get_file_bytes_archives:
      import zipfile
      path = cls.FIXTURES.zip_path(archive)
      cls._get_file_bytes_archives[archive] = zipfile.ZipFile(path)
      
    
    try:
      return cls._get_file_bytes_archives[archive].read(entryname)
    except Exception as e:
        raise Exception((e, archive, uri))

  @classmethod
  def _get_segment_frame_to_pose(cls, segment_id):
    """Get the frame -> pose map for the given `segment_id`.  Cache these since
    multiple datum constructors will need to look up poses."""
    if not hasattr(cls, '_seg_to_poses'):
      cls._seg_to_poses = {}
    if segment_id not in cls._seg_to_poses:
      split, segnum = segment_id.split('-')[-2:]
      entryname = split + 'ing/oxts/' + segnum + '.txt'
      oxts_str = cls._get_file_bytes(
        archive='data_tracking_oxts.zip', entryname=entryname)
      oxts_str = oxts_str.decode()
      frame_to_xform = load_transforms_from_oxts(oxts_str)
      cls._seg_to_poses[segment_id] = frame_to_xform
    return cls._seg_to_poses[segment_id]

  @classmethod
  def _get_ego_pose(cls, uri):
    # Pose information for Object Benchmark not available
    if 'kitti-object-benchmark' in uri.segment_id:
      return datum.Transform(src_frame='world', dest_frame='ego')
    else:
      frame_to_xform = cls._get_segment_frame_to_pose(uri.segment_id)
      return frame_to_xform[int(uri.extra['kitti.frame'])]

  @classmethod
  def _get_calibration(cls, uri):
    """Get the `Calibration` instance for the given `uri`.  Cache these since
    multiple datum constructors will need to look up calibration."""

    if not hasattr(cls, '_obj_frame_to_calib'):
      cls._obj_frame_to_calib = {}
    if not hasattr(cls, '_tracking_seg_to_calib'):
      cls._tracking_seg_to_calib = {}
    
    if 'kitti-object-benchmark' in uri.segment_id:
      frame = uri.extra['kitti.frame']
      if frame not in cls._obj_frame_to_calib:
        entryname = uri.split + 'ing/calib/' + frame + '.txt'
        calib_str = cls._get_file_bytes(
          archive='data_object_calib.zip', entryname=entryname)
        calib_str = calib_str.decode()
        calib = Calibration.from_kitti_str(calib_str)
        cls._obj_frame_to_calib[frame] = calib
      return cls._obj_frame_to_calib[frame]
    
    else: # Tracking
      if uri.segment_id not in cls._tracking_seg_to_calib:
        split, segnum = uri.segment_id.split('-')[-2:]
        entryname = split + 'ing/calib/' + segnum + '.txt'
        calib_str = cls._get_file_bytes(
          archive='data_tracking_calib.zip', entryname=entryname)
        calib_str = calib_str.decode()
        calib = Calibration.from_kitti_str(calib_str)
        cls._tracking_seg_to_calib[uri.segment_id] = calib
      return cls._tracking_seg_to_calib[uri.segment_id]

  @classmethod
  def _project_cuboids_to_lidar_frame(cls, uri, cuboids):
    """Project the given `cuboids` from the camera frame to the lidar frame
    (using calibration for `uri`) and return a transformed copy.

    See also the tests:
     * `test_kitti_object_label_lidar_projection()`
     * `test_kitti_tracking_label_lidar_projection()`
    """
    import copy

    ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ## Note: KITTI Cuboids are in the *camera* frame and must be projected
    ## into the lidar frame for plotting. This test helps document and 
    ## ensure this assumption holds.
    ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    calib = cls._get_calibration(uri)
    lidar_to_cam = calib.R0_rect @ calib.velo_to_cam_unrectified
    cam_to_lidar = lidar_to_cam.get_inverse()

    cuboids = copy.deepcopy(cuboids)
    for c in cuboids:
      from psegs.datum.transform import Transform
      obj_from_ego_lidar = cam_to_lidar @ c.obj_from_ego
      c.obj_from_ego = obj_from_ego_lidar
      c.obj_from_ego.src_frame = 'ego' # In KITTI, lidar is the ego frame ~~~~~~~~~~
      c.obj_from_ego.dest_frame = 'obj'

    return cuboids

  @classmethod
  def _get_bench2raw_mapper(cls):
    if not hasattr(cls, '_bench2raw_mapper'):
      class SDBenchmarkToRawMapper(BenchmarkToRawMapper):
        FIXTURES = cls.FIXTURES
      cls._bench2raw_mapper = SDBenchmarkToRawMapper()
    return cls._bench2raw_mapper


  ## Datum Construction

  @classmethod
  def _iter_datums_from_uri(cls, uri):
    if uri.topic.startswith('camera'):
      yield cls._create_camera_image(uri)
    elif uri.topic.startswith('lidar'):
      yield cls._create_point_cloud(uri)
    elif uri.topic.startswith('labels'):
      for sd in cls._iter_labels(uri):
        yield sd
    elif uri.topic == 'ego_pose':
      for sd in cls._iter_ego_poses(uri):
        yield sd
    else:
      raise ValueError(uri)
  
  @classmethod
  def _create_camera_image(cls, uri):
    from psegs.util import misc

    image_png = cls._get_file_bytes(uri=uri)
    width, height = misc.get_png_wh(image_png)

    mapper = cls._get_bench2raw_mapper()
    mapper.fill_timestamp(uri)

    # timestamp = int(int(uri.extra['kitti.frame']) * 1e8)
    # # TODO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
    # uri.timestamp = timestamp

    ego_pose = cls._get_ego_pose(uri)

    calib = cls._get_calibration(uri)
    K = calib.K2
    ego_to_sensor = calib.velo_to_cam_2_rect
    if 'right' in uri.topic:
      K = calib.K3
      ego_to_sensor = calib.velo_to_cam_3_rect

    extra = mapper.get_extra(uri)

    ci = datum.CameraImage(
          sensor_name=uri.topic,
          image_png=bytearray(image_png),
          width=width,
          height=height,
          timestamp=uri.timestamp,
          ego_pose=ego_pose,
          K=K,
          ego_to_sensor=ego_to_sensor,
          extra=extra)
    return datum.StampedDatum(uri=uri, camera_image=ci)

  @classmethod
  def _create_point_cloud(cls, uri):
    lidar_bytes = cls._get_file_bytes(uri=uri)
    raw_lidar = np.frombuffer(lidar_bytes, dtype=np.float32).reshape((-1, 4))
    cloud = raw_lidar[:, :3]
    # unused: reflectance = raw_lidar[:, 3:]

    # timestamp = int(int(uri.extra['kitti.frame']) * 1e8)
    # # TODO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
    # uri.timestamp = timestamp
    mapper = cls._get_bench2raw_mapper()
    mapper.fill_timestamp(uri)

    # In KITTI, lidar is the ego frame
    ego_to_sensor = Transform(src_frame='ego', dest_frame='lidar')

    ego_pose = cls._get_ego_pose(uri)

    extra = mapper.get_extra(uri)

    pc = datum.PointCloud(
          sensor_name=uri.topic,
          timestamp=uri.timestamp,
          cloud=cloud,
          ego_to_sensor=ego_to_sensor,
          ego_pose=ego_pose,
          extra=extra)
    return datum.StampedDatum(uri=uri, point_cloud=pc)

  @classmethod
  def _iter_labels(cls, uri):
    # KITTI has no labels for test.
    # FMI see https://github.com/pwais/psegs-kitti-ext
    if uri.split == 'test':
      return
    
    if 'kitti-object-benchmark' in uri.segment_id:
      yield cls._get_object_labels(uri)
    else: # Tracking
      for sd in cls._iter_tracking_labels(uri):
        yield sd
  
  @classmethod
  def _get_object_labels(cls, uri):
    frame = uri.extra['kitti.frame']
    entryname = uri.split + 'ing/label_2/' + frame + '.txt'
    label_str = cls._get_file_bytes(
        archive='data_object_label_2.zip', entryname=entryname)
    label_str = label_str.decode()
    cuboids, bboxes = parse_object_label_cuboids(label_str)

    # FIXME bboxes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    cuboids = cls._project_cuboids_to_lidar_frame(uri, cuboids)

    # timestamp = int(int(frame) * 1e8)
    # # TODO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
    # uri.timestamp = timestamp
    mapper = cls._get_bench2raw_mapper()
    mapper.fill_timestamp(uri)

    for c in cuboids:
      c.timestamp = uri.timestamp
      c.ego_pose = cls._get_ego_pose(uri)
      c.extra = mapper.get_extra(uri)
    
    return datum.StampedDatum(uri=uri, cuboids=cuboids)
  
  @classmethod
  def _iter_tracking_labels(cls, uri):
    import copy
    
    split, segnum = uri.segment_id.split('-')[-2:]
    entryname = split + 'ing/label_02/' + segnum + '.txt'
    labels_str = cls._get_file_bytes(
      archive='data_tracking_label_2.zip', entryname=entryname)
    labels_str = labels_str.decode()

    f_to_cuboids, _ = parse_tracking_label_cuboids(labels_str)
      # NB: We ignore bboxes for the Tracking Benchmark
    
    mapper = cls._get_bench2raw_mapper()
    for frame, cuboids in f_to_cuboids.items():
      datum_uri = copy.deepcopy(uri)
      datum_uri.extra['kitti.frame'] = str(frame).zfill(6)

      cuboids = cls._project_cuboids_to_lidar_frame(uri, cuboids)

      # timestamp = int(int(frame) * 1e8)
      # # TODO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
      # datum_uri.timestamp = timestamp
      mapper.fill_timestamp(datum_uri)

      for c in cuboids:
        c.timestamp = datum_uri.timestamp
        c.ego_pose = cls._get_ego_pose(datum_uri)
        c.extra = mapper.get_extra(datum_uri)

      yield datum.StampedDatum(uri=datum_uri, cuboids=cuboids)

  @classmethod
  def _iter_ego_poses(cls, uri):
    import copy

    # Pose information for Object Benchmark not available
    if 'kitti-object-benchmark' in uri.segment_id:
      return
    
    mapper = cls._get_bench2raw_mapper()
    frame_to_xform = cls._get_segment_frame_to_pose(uri.segment_id)
    for frame, xform in frame_to_xform.items():
      datum_uri = copy.deepcopy(uri)
      # datum_uri.timestamp = int(int(frame) * 1e8) # FIXME ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      datum_uri.extra['kitti.frame'] = str(frame).zfill(6)
      mapper.fill_timestamp(datum_uri)
      yield datum.StampedDatum(uri=datum_uri, transform=xform)


###############################################################################
### IDatasetUtil Impl

class DSUtil(IDatasetUtil):

  FIXTURES = Fixtures

  @classmethod
  def all_zips(cls):
    import itertools
    all_zips = itertools.chain(
                  cls.FIXTURES.OBJECT_BENCHMARK_FNAMES,
                  cls.FIXTURES.TRACKING_BENCHMARK_FNAMES)
    return list(all_zips)

  @classmethod
  def emplace(cls):
    cls.FIXTURES.maybe_emplace_psegs_kitti_ext()

    if not cls.FIXTURES.ROOT.exists():
      zips = '\n        '.join('  * %s' % fname for fname in cls.all_zips())
      cls.show_md("""
        Due to KITTI license constraints, you need to manually accept the KITTI
        license to obtain the download URLs for the
        [Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) and
        [Object Benchmark](http://www.cvlibs.net/datasets/kitti/eval_object.php)
        zip files.  But once you have the URL, it's easy to write a short bash
        loop with `wget` to fetch them in parallel.

        You'll want to download all the following zip files (do not decompress
        them) to a single directory on a local disk (spinning disk OK):

        %s

        Once you've downloaded the archives, we'll need the path to where
        you put them.  Enter that below, or exit this program.

      """ % (zips,))
      kitti_root = input(
        "Please enter the directory containing your KITTI zip archives; "
        "PSegs will create a (read-only) symlink to them: ")
      kitti_root = Path(kitti_root.strip())
      assert kitti_root.exists()
      assert kitti_root.is_dir()

      from oarphpy import util as oputil
      oputil.mkdir(str(cls.FIXTURES.ROOT.parent))

      cls.show_md("Symlink: \n%s <- %s" % (kitti_root, cls.FIXTURES.ROOT))
      os.symlink(kitti_root, cls.FIXTURES.ROOT)

      # Make symlink read-only
      import stat
      os.chmod(
        kitti_root,
        stat.S_IREAD|stat.S_IRGRP|stat.S_IROTH,
        follow_symlinks=False)

    cls.show_md("Validating KITTI archives ...")
    zips_needed = set(cls.all_zips())
    zips_have = set()
    for entry in cls.FIXTURES.ROOT.iterdir():
      if entry.name in zips_needed:
        zips_needed.remove(entry.name)
        zips_have.add(entry.name)
    
    if zips_needed:
      s_have = \
        '\n        '.join('  * %s' % fname for fname in zips_have)
      s_needed = \
        '\n        '.join('  * %s' % fname for fname in zips_needed)
      cls.show_md("""
        Missing some expected archives!

        Found:
        
        %s

        Missing:

        %s
      """ % (s_have, s_needed))
      return False
    
    cls.show_md("... all KITTI archives found!")
    return True

  @classmethod
  def test(cls):
    from oarphpy import util as oputil
    oputil.run_cmd("cd %s && pytest -s -vvv -k test_kitti" % C.PS_ROOT)
    return True

  @classmethod
  def build_table(cls):
    KITTISDTable.build()
    return True
