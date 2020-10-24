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

from psegs.datasets.kitti_360 import *


# NB: cv2 File Storage in _python_ does not support integers, only floats :(
# import cv2
# CV2_FILE_NODE_TYPE_TO_NAME = dict(
#   (getattr(cv2, attr), attr)
#   for attr in dir(cv2)
#   if attr.startswith('FILE_NODE_'))
# CV2_FILE_NODE_INTEGRAL_TYPE_TO_GETTER = {
#   'FILE_NODE_FLOAT':  lambda n: n.real(),
#   'FILE_NODE_REAL':   lambda n: n.real(),
#   'FILE_NODE_INT':    lambda n: n.real(),
#   'FILE_NODE_STRING': lambda n: n.string(),
#   'FILE_NODE_MAT':    lambda n: n.mat(),
# }

# def cvnode_to_python(n):
#   node_type_name = CV2_FILE_NODE_TYPE_TO_NAME[n.type()]
#   if node_type_name in CV2_FILE_NODE_INTEGRAL_TYPE_TO_GETTER:
#     f = CV2_FILE_NODE_INTEGRAL_TYPE_TO_GETTER[node_type_name]
#     return f(n)
#   elif node_type_name == 'FILE_NODE_MAP':
#     return dict(
#       (k, cvnode_to_python(n.getNode(k)))
#       for k in n.keys()
#     )
#   elif node_type_name == 'FILE_NODE_SEQ':
#     return [cvnode_to_python(n.at(i)) for i in range(n.size())]
#   else:
#     raise ValueError("Don't know how to handle node of type %s: %s" % (
#       node_type_name, n))


def kitti_360_get_parsed_node(d):

  def to_ndarray(d):
    import numpy as np
    r = int(d['rows'])
    c = int(d['cols'])
    dtype = str(d['dt'])
    parse = float if dtype == 'f' else int
    data = [parse(t) for t in d['data'].split() if t]
    a = np.array(data)
    return a.reshape((r, c))

  def fill_cuboid(d):
    # Appears the cuboid bounds are encoded in the RT; in the raw XML, the
    # vertices are +/- 0.5m for all objects in the XML
    # FMI https://github.com/autonomousvision/kitti360Scripts/blob/081c08b34a14960611f459f23a0ad049542205c6/kitti360scripts/helpers/annotation.py#L125
    R = d['transform'][:3, :3]
    T = d['transform'][:3, 3]
    v = d['vertices']
    d['cuboid'] = np.matmul(R, v.T).T + T

  # ??? Not sure what this is about
  # FMI https://github.com/autonomousvision/kitti360Scripts/blob/081c08b34a14960611f459f23a0ad049542205c6/kitti360scripts/helpers/annotation.py#L154
  def to_class(label_value):
    K360_CLASSMAP = {
      'driveway': 'parking',
      'ground': 'terrain',
      'unknownGround': 'ground', 
      'railtrack': 'rail track'
    }
    if label_value in K360_CLASSMAP:
      return K360_CLASSMAP[label_value]
    else:
      return label_value

  out = {
    'index':            int(d['index']),
    'label':            str(d['label']),
    'k360_class_name':  to_class(str(d['label'])),
    'semanticId_orig':  int(d['semanticId_orig']),
    'semanticId':       int(d['semanticId']),
    'instanceId':       int(d['instanceId']),
    'category':         str(d['category']),
    'timestamp':        int(d['timestamp']),
    'dynamic':          int(d['dynamic']),
    'start_frame':      int(d['start_frame']),
    'end_frame':        int(d['end_frame']),
    'transform':        to_ndarray(d['transform']),
    'vertices':         to_ndarray(d['vertices']),
    'faces':            to_ndarray(d['faces']),
  }

  fill_cuboid(out)
  return out


def test_kitti350_play():

  import imageio

  import numpy as np

  from pathlib import Path

  

  import xmltodict
  d = xmltodict.parse(open("/outer_root/media/seagates-ext4/au_datas/kitti/kitti-360/KITTI-360/data_3d_bboxes/train/2013_05_28_drive_0000_sync.xml").read())

  objects = d['opencv_storage']
  obj_name_to_value = dict(
    (k, kitti_360_get_parsed_node(v)) for (k, v) in objects.items())

  FRAMEID = 8096
  obs_in_frame = dict(
    (k, v) for (k, v) in obj_name_to_value.items()
    if v['start_frame'] <= FRAMEID <= v['end_frame']
  )

  ROOT = Path('/outer_root/media/seagates-ext4/au_datas/kitti/kitti-360/KITTI-360/')

  calib_cam_to_pose = open(ROOT / 'calibration/calib_cam_to_pose.txt').read()
  calib_cam_to_velo = open(ROOT / 'calibration/calib_cam_to_velo.txt').read()
  calib_sick_to_velo = open(ROOT / 'calibration/calib_sick_to_velo.txt').read()
  perspective = open(ROOT / 'calibration/perspective.txt').read()

  calib = Calibration.from_kitti_360_strs(
            calib_cam_to_pose,
            calib_cam_to_velo,
            calib_sick_to_velo,
            perspective)
  
  img = imageio.imread(ROOT / 'data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000008096.png')
  img2 = imageio.imread(ROOT / 'data_2d_raw/2013_05_28_drive_0000_sync/image_01/data_rect/0000008096.png')

  vel_path = ROOT / 'data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/0000008096.bin'
  cloud = np.fromfile(vel_path, dtype=np.float32)
  cloud = np.reshape(cloud, [-1, 4])
  cloud = cloud[:, :3]

  # https://github.com/autonomousvision/kitti360Scripts/blob/fc4e92bfe7d7da0a404e58bca3b98660147ca09c/kitti360scripts/helpers/project.py#L65
  cam0_to_world = ROOT / 'data_poses/2013_05_28_drive_0000_sync/cam0_to_world.txt'
  poses = np.loadtxt(cam0_to_world)
  frames = poses[:,0]
  poses_raw = np.reshape(poses[:, 1:],[-1, 4, 4])
  cam0_to_world = dict(zip(frames, poses_raw))

  poses_idk = ROOT / 'data_poses/2013_05_28_drive_0000_sync/poses.txt'
  poses = np.loadtxt(poses_idk)
  frames = poses[:,0]
  poses_raw = np.reshape(poses[:, 1:],[-1, 3, 4])
  poses_idk = dict(zip(frames, poses_raw))


  cuboids_cam = []
  for obj_name, obj in obs_in_frame.items():
    from psegs import datum

    front_world = obj['cuboid'][:3, :]
    rear_world = obj['cuboid'][3:, :]

    l = abs(front_world[0, 0] - rear_world[0, 0])
    w = abs(front_world[0, 1] - front_world[1, 1])
    h = abs(front_world[0, 2] - front_world[2, 2])

    T_world = np.mean(front_world - rear_world, axis=0)

    from scipy.spatial.transform import Rotation as R
    heading = front_world[0, :] - rear_world[0, :]
    heading /= np.linalg.norm(heading)
    R_world = R.from_rotvec(heading).as_matrix()

    T_world_to_obj = datum.Transform.from_transformation_matrix(
                          np.hstack([R_world, T_world]),
                          src_frame='world',
                          dest_frame='obj')

    RT_cam0_to_world = cam0_to_world[FRAMEID]
    T_cam0_to_world = datum.Transform.from_transformation_matrix(
                          RT_cam0_to_world,
                          src_frame='camera|left_raw',
                          dest_frame='world')
    
    T_obj_from_ego = (
      T_world_to_obj['obj', 'world'] @ 
      T_cam0_to_world['world', 'camera|left_raw'])
        # left camera is ego?  or do we need `poses` that has lidar?
    T_obj_from_ego.src_frame = 'ego'
    T_obj_from_ego.dest_frame = 'obj'

    c = datum.Cuboid(
          track_id=obj_name,
          category_name=obj['k360_class_name'],
          length_meters=l,
          width_meters=w,
          height_meters=h,
          obj_from_ego=datum.Transform())
    cuboids_cam.append(c)


  from psegs import util
  from psegs import datum
  
  outdir = Path('/opt/psegs/test_run_output')

  frame = 'yay'

  pc = datum.PointCloud(cloud=cloud)
  util.log.info("Projecting BEV %s ..." % frame)
  import time
  start = time.time()
  bev_img = pc.get_bev_debug_image(cuboids=cuboids_cam)
  print('bev', time.time() - start)
  fname = 'projected_lidar_labels_bev_%s.png' % frame.replace('/', '_')
  imageio.imwrite(outdir / fname, bev_img)

  util.log.info("Projecting Front RV %s ..." % frame)
  import time
  start = time.time()
  rv_img = pc.get_front_rv_debug_image(cuboids=cuboids_cam)
  print('rv', time.time() - start)
  fname = 'projected_lidar_labels_front_rv_%s.png' % frame.replace('/', '_')
  imageio.imwrite(outdir / fname, rv_img)


  from psegs.util import plotting as pspl

  util.log.info("Projecting cloud %s ..." % frame)

  cloud2 = calib.cam_left_raw_to_velo.get_inverse().apply(cloud).T

  uvd = calib.cam0_K.dot(cloud2.T).T
  uvd[:, 1] /= uvd[:, 2]
  uvd[:, 0] /= uvd[:, 2]

  # import pdb; pdb.set_trace()


  debug = img.copy()
  pspl.draw_xy_depth_in_image(debug, uvd, marker_radius=0, alpha=0.7)
  fname = 'projected_pts_front_cam_%s.png' % frame.replace('/', '_')
  imageio.imwrite(outdir / fname, debug)


  frame = 'yay_right'
  util.log.info("Projecting cloud %s ..." % frame)

  cloud2 = calib.cam_left_raw_to_velo.get_inverse().apply(cloud).T
  cloud2 = calib.RT_01.apply(cloud2).T

  uvd = calib.cam1_K.dot(cloud2.T).T
  uvd[:, 1] /= uvd[:, 2]
  uvd[:, 0] /= uvd[:, 2]

  # import pdb; pdb.set_trace()


  debug = img2.copy()
  pspl.draw_xy_depth_in_image(debug, uvd, marker_radius=0, alpha=0.7)
  fname = 'projected_pts_front_cam_%s.png' % frame.replace('/', '_')
  imageio.imwrite(outdir / fname, debug)

