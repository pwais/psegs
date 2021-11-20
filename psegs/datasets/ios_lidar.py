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
from multiprocessing import Value
import os
from pathlib import Path

import numpy as np

from psegs import datum
from psegs import util
from psegs.conf import C
from psegs.datasets.idsutil import IDatasetUtil
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
  if not frame_0_img_path.exists():
    return {}

  frame_0_info_path = Path(scene_dir) / 'frame_00000.json'
  if not frame_0_info_path.exists():
    return {}

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


def threeDScannerApp_create_camera_image(
        frame_json_path,
        sensor_name='camera|front',
        timestamp=None):

  assert os.path.exists(frame_json_path), frame_json_path

  scan_dir = Path(os.path.dirname(frame_json_path))
  frame_id = threeDScannerApp_frame_id_from_fname(frame_json_path)

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
  extra['threeDScannerApp.frame_id'] = str(frame_id)
  extra['threeDScannerApp.scan_dir'] = str(os.path.basename(scan_dir))

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
  
  if 'depth' in sensor_name:
  
    depth_path = scan_dir / f'depth_{frame_id}.png'
    assert os.path.exists(depth_path), depth_path
  
    conf_path = scan_dir / f'conf_{frame_id}.png'
    assert os.path.exists(conf_path), conf_path

    # Get dimensions from the conf image, which is a smaller file
    from psegs.util import misc
    with open(conf_path, 'rb') as f:
      w, h = misc.get_png_wh(f.read(1024))
  
    def _get_depth_conf_image(depth_path, conf_path):
      import imageio
      import numpy as np
      depth = imageio.imread(depth_path)
      
      # millimeters -> meters
      depth = depth.astype(np.float32) * .001

      conf = imageio.imread(conf_path)
      depth_image = np.concatenate([depth, conf.astype(np.float32)], axis=-1)
      return depth_image
    
    image_factory = lambda: _get_depth_conf_image(depth_path, conf_path)

    extra['threeDScannerApp.depth_path'] = os.path.basename(depth_path)
    extra['threeDScannerApp.conf_path'] = os.path.basename(conf_path)

  else:
    frame_img_path = frame_json_path.replace('.json', '.jpg')
    assert os.path.exists(frame_img_path), frame_img_path

    from oarphpy import util as oputil
    with open(frame_img_path, 'rb') as f:
      w, h = oputil.get_jpeg_size(f.read(1024))

    def _load_image(path):
      import imageio
      return imageio.imread(path)
    image_factory = lambda: _load_image(frame_img_path)

    extra['threeDScannerApp.img_path'] = os.path.basename(frame_img_path)

  ci = datum.CameraImage(
                sensor_name=sensor_name,
                image_factory=image_factory,
                height=h,
                width=w,
                timestamp=timestamp,
                ego_pose=ego_pose,
                ego_to_sensor=ego_to_sensor,
                K=K,
                extra=extra)

  return ci


def threeDScannerApp_create_point_cloud_from_mesh(
        mesh_path,
        sensor_name='lidar|mesh'):
  
  assert os.path.exists(mesh_path), mesh_path

  scan_dir = Path(os.path.dirname(mesh_path))

  extra = {
    'threeDScannerApp.mesh_path': os.path.basename(mesh_path),
    'threeDScannerApp.scan_dir': os.path.basename(scan_dir),
  }

  # The meshes are in the world frame
  ego_to_sensor = datum.Transform(
            src_frame='world',
            dest_frame=sensor_name)
  
  def _get_cloud(mesh_path):
    import open3d as o3d
    import numpy as np
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    xyz = np.asarray(mesh.vertices)
    return xyz
  cloud_factory = lambda: _get_cloud(mesh_path)

  pc = datum.PointCloud(
          sensor_name=sensor_name,
          cloud_factory=cloud_factory,
          ego_to_sensor=ego_to_sensor,
          extra=extra)
  return pc


def threeDScannerApp_get_segment_id(scan_dir='', info_path=''):
  if not info_path:
    info_path = str(Path(scan_dir) / 'info.json')
  seg_dir = os.path.dirname(info_path)
  if info_path:
    with open(info_path, 'r') as f:
      info = json.load(f)
    segment_id = info.get('title', os.path.split(seg_dir)[-1])
  else:
    segment_id = seg_dir
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
                    'threeDScannerApp.scan_dir': scan_dir,
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
                    'threeDScannerApp.scan_dir': scan_dir,
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
                    'threeDScannerApp.scan_dir': scan_dir,
                    'threeDScannerApp.frame_id': frame_id,
                    'threeDScannerApp.img_path': os.path.basename(img_path),
                    'threeDScannerApp.json_path': os.path.basename(info_path),
                  })
      uris.append(ci_uri)
    
    depth_path = scan_dir / f'depth_{frame_id}.png'
    conf_path = scan_dir / f'conf_{frame_id}.png'
    if os.path.exist(depth_path) and os.path.exists(conf_path):
      # NB: raw depth only available when app is in 'low-res' mode
      pc_uri = datum.URI(
                  segment_id=segment_id,
                  topic='camera|front|depth',
                  timestamp=t,
                  extra={
                    'threeDScannerApp.scan_dir': scan_dir,
                    'threeDScannerApp.frame_id': frame_id,
                    'threeDScannerApp.depth_path': os.path.basename(depth_path),
                    'threeDScannerApp.conf_path': os.path.basename(conf_path),
                    'threeDScannerApp.json_path': os.path.basename(info_path),
                  })
      uris.append(pc_uri)
    
  return uris


def threeDScannerApp_create_stamped_datum(uri):
  if 'threeDScannerApp.scan_dir' not in uri.extra:
    raise ValueError(uri)
  scan_dir = Path(uri.extra['threeDScannerApp.scan_dir'])
  frame_json_path = scan_dir / uri.extra['threeDScannerApp.json_path']
  if uri.topic.startswith('camera'):
    ci = threeDScannerApp_create_camera_image(
            frame_json_path,
            sensor_name=uri.topic,
            timestamp=uri.timestamp)
    return datum.StampedDatum(uri=uri, camera_image=ci)
  elif uri.topic == 'lidar|mesh':
    scan_dir = Path(uri.extra['threeDScannerApp.scan_dir'])
    mesh_path = scan_dir / uri.extra['threeDScannerApp.mesh_path']
    pc = threeDScannerApp_create_point_cloud_from_mesh(
      mesh_path, sensor_name=uri.topic)
    pc.timestamp = uri.timestamp
    return datum.StampedDatum(uri=uri, point_cloud=pc)
  elif uri.topic == 'ego_pose':
    with open(frame_json_path, 'r') as f:
      json_data = json.load(f)
    xform = threeDScannerApp_get_ego_pose(json_data)
    return datum.StampedDatum(uri=uri, transform=xform)
  else:
    raise ValueError(uri)


###############################################################################
### Single-Scene Research Utils

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
      include_raw_xform=True,
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
    
    if include_raw_xform:
      frame_json_path = in_rgb.replace('.jpg', '.json')
      if os.path.exists(frame_json_path):
        with open(frame_json_path, 'r') as f:
          json_data = json.load(f)
        
        ego_pose = threeDScannerApp_get_ego_pose(json_data)
        xform = ego_pose.get_transformation_matrix(homogeneous=True)
        xform_dest = rgb_dest + '.xform.npz'
        with open(xform_dest, 'wb') as f:
          np.save(f, xform)

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
      seg_dir = os.path.basename(info_path)
      segment_id = threeDScannerApp_get_segment_id(info_path=info_path)
      uri = datum.URI(
              dataset='psegs-ios-lidar-ext',
              split='threeDScannerApp_data',
              segment_id=segment_id)
      uri_to_segment_dir[str(uri.to_segment_uri())] = seg_dir
    return uri_to_segment_dir

  # @classmethod
  # def index_root(cls):
  #   """A r/w place to cache any temp / index data"""
  #   return C.PS_TEMP / 'psegs_ios_lidar'

  @classmethod
  def get_uri_to_seg_dir(cls):
    """Build and return a map of segment uri -> segment data dir
    for all known segments"""
    uri_to_seg_dir = dict()
    uri_to_seg_dir.update(
      cls.get_threeDScannerApp_uri_to_segment_dir())
    return uri_to_seg_dir

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


class IOSLidarSDTable(StampedDatumTableBase):
  
  FIXTURES = Fixtures

  ## Subclass API

  @classmethod
  def _get_all_segment_uris(cls):   
    uri_to_seg_dir = cls.FIXTURES.get_uri_to_seg_dir()
    return sorted(datum.URI.from_str(uri) for uri in uri_to_seg_dir.keys())

  @classmethod
  def _create_datum_rdds(cls, spark, existing_uri_df=None, only_segments=None):
    from oarphpy import util as oputil

    ## First get the data dirs for the segments we need ...
    uri_to_seg_dir = cls.FIXTURES.get_uri_to_seg_dir()
    if only_segments:
      util.log.info(
        "Filtering to only %s segments" % len(only_segments))
      uri_to_seg_dir = dict(
                        (uri, seg_dir)
                        for uri, seg_dir in uri_to_seg_dir.items()
                        if any(
                            suri.soft_matches_segment(uri)
                            for suri in only_segments))
    
    ## ... generate URIs for those segments ...
    seg_dirs = sorted(uri_to_seg_dir.values())
    seg_dir_rdd = spark.sparkContext.parallelize(
                    seg_dirs, numSlices=len(seg_dirs))
    uri_rdd = seg_dir_rdd.flatMap(cls.get_uris_for_seg_dir)

    ## ... filter if necessary ...
    if existing_uri_df is not None:
      def to_datum_id(obj):
          return (
            obj.dataset,
            obj.split,
            obj.segment_id,
            obj.topic,
            obj.timestamp)

      key_uri_rdd = uri_rdd.map(lambda u: (to_datum_id(u), u))
      existing_keys_nulls = existing_uri_df.rdd.map(to_datum_id).map(
                                  lambda t: (t, None))
      uri_rdd = key_uri_rdd.subtractByKey(existing_keys_nulls).map(
                                      lambda kv: kv[1])

    ## ... now build Datum RDDs ...
    URIS_PER_CHUNK = os.cpu_count() * 128
    uris = uri_rdd.collect()
    util.log.info("... creating datums for %s URIs." % len(uris))

    datum_rdds = []
    for chunk in oputil.ichunked(uris, URIS_PER_CHUNK):
      chunk_uri_rdd = spark.sparkContext.parallelize(chunk)
      datum_rdd = chunk_uri_rdd.map(cls.create_stamped_datum)
      datum_rdds.append(datum_rdd)
    return datum_rdds
  

  ## Datum Construction Support

  @classmethod
  def get_bad_seg_dirs(cls):
    """Some captures are bad / cannot be parsed / are incomplete.
    Return a list of those segments."""
    uri_to_seg_dir = cls.FIXTURES.get_uri_to_seg_dir()
    bad_seg_dirs = [
      seg_dir for seg_dir in uri_to_seg_dir.values()
      if not cls.get_uris_for_seg_dir(seg_dir)
    ]
    return bad_seg_dirs

  @classmethod
  def get_uris_for_seg_dir(cls, seg_dir):
    # For now, we don't need to sniff the seg_dir type, we only
    # support threeDScannerApp format.  In the future, we'll need
    # to condition on seg_dir type.
    uris = threeDScannerApp_get_uris_from_scan_dir(seg_dir)
    return uris

  @classmethod
  def create_stamped_datum(cls, uri):
    # For now, we don't need to sniff the uri, we only
    # support threeDScannerApp format.  In the future, we'll need
    # to condition on uri type.
    return threeDScannerApp_create_stamped_datum(uri)



###############################################################################
### IDatasetUtil Impl

class DSUtil(IDatasetUtil):

  FIXTURES = Fixtures

  @classmethod
  def emplace(cls):
    cls.FIXTURES.maybe_emplace_psegs_ios_lidar_ext()
    return True

  @classmethod
  def test(cls):
    from oarphpy import util as oputil
    oputil.run_cmd("cd %s && pytest -s -vvv -k test_ios_lidar" % C.PS_ROOT)
    return True

  @classmethod
  def build_table(cls):
    # IOSLidarSDTable.build()
    return True
