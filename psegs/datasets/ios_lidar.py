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
# Parsing code references:
#  * ARBodyPoseRecorder from the developer of 3DScannerApp:
#     https://github.com/laanlabs/ARBodyPoseRecorder/blob/9e7a37cdfdb44bc223f7b983481841696a763782/ARBodyPoseRecorder/ViewController.swift#L233
#  * rtabmap ( http://introlab.github.io/rtabmap/ ) code that appears to
#     parse 3DScannerApp output:
#     https://docs.ros.org/en/api/rtabmap/html/CameraImages_8cpp_source.html

from psegs import datum
from psegs.table.sd_table import StampedDatumTableBase

def threeDScannerApp_get_camera_pose(json_data):
  import numpy as np

  T_raw = json_data['cameraPoseARFrame'] # A row-major 4x4 matrix
  T_arr_raw = np.array(T_raw).reshape([4, 4])

  # Based upon rtabmap noted above
  # Rotate from OpenGL coordinate frame to 
  # +x = forward, +y = left, +z = up
  OPENGTL_T_WORLD = np.array([
    [ 0, -1,  0,  0],
    [ 0,  0,  1,  0],
    [-1,  0,  0,  0],
  ])

  WORLD_T_OPENGL = np.array([
    [ 0,  0, -1,  0],
    [-1,  0,  0,  0],
    [ 0,  1,  0,  0],
  ])

  pose = WORLD_T_OPENGL * T_arr_raw[:3, :4] * OPENGTL_T_WORLD
  return datum.Transform.from_transformation_matrix(
            pose,
            src_frame='world',
            dest_frame='ego')
  
def threeDScannerApp_get_K(json_data):
  import numpy as np

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
  
  import numpy as np

  assert os.path.exists(frame_json_path)
  frame_img_path = frame_json_path.replace('.json', '.jpg')
  assert os.path.exists(frame_img_path)

  with open(frame_json_path, 'r') as f:
    json_data = json.load(f)
  
  ego_pose = threeDScannerApp_get_camera_pose(json_data)
  K = threeDScannerApp_get_K(json_data)

  timestamp = int(json_data['time'] * 1e9)
    # CACurrentMediaTime, which is mach_absolute_time, which is system uptime

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
  
  assert set(REQUIRED_KEYS) - set(json_data.keys()) == set(), \
    "Have %s wanted %s" % (extra.keys(), REQUIRED_KEYS)
  extra = dict(
            ('threeDScannerApp.' + k, json.dumps(v))
            for k, v in json_data.items()
            if k not in SKIP_KEYS)

  # https://docs.ros.org/en/api/rtabmap/html/classrtabmap_1_1CameraModel.html#a0853af9d0117565311da4ffc3965f8d2
  ego_to_sensor = datum.Transform(
            rotation=np.array([
              [ 0,  0,  1],
              [-1,  0,  0],
              [ 0, -1,  0],
            ]),
            src_frame='ego',
            dest_frame='camera_front')
  
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


