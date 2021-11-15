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
#      depth_XXXXX.png - raw depth info as a 16-bit png (depth in millimeters?)
#      conf_XXXXX.png - sensor confidence for depth?
#
# Parsing code references:
#  * ARBodyPoseRecorder from the developer of 3DScannerApp:
#     https://github.com/laanlabs/ARBodyPoseRecorder/blob/9e7a37cdfdb44bc223f7b983481841696a763782/ARBodyPoseRecorder/ViewController.swift#L233
#  * rtabmap ( http://introlab.github.io/rtabmap/ ) code that appears to
#     parse 3DScannerApp output:
#     https://docs.ros.org/en/api/rtabmap/html/CameraImages_8cpp_source.html

import os

import numpy as np

from psegs import datum
from psegs import util
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


def threeDScannerApp_create_camera_image(frame_json_path):
  import os
  import json

  assert os.path.exists(frame_json_path), frame_json_path
  frame_img_path = frame_json_path.replace('.json', '.jpg')
  assert os.path.exists(frame_img_path), frame_img_path

  with open(frame_json_path, 'r') as f:
    json_data = json.load(f)
  
  ego_pose = threeDScannerApp_get_ego_pose(json_data)
  K = threeDScannerApp_get_K(json_data)

  timestamp = int(json_data['time'] * 1e9)
    # CACurrentMediaTime, which is mach_absolute_time, which is *system uptime*
    # TODO: re-write time using exif / file timestamps

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


def threeDScannerApp_convert_raw_to_opend3d_rgbd(input_dir, output_dataset_dir):
  import json

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
      import json
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
