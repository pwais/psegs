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

from psegs.datasets import kitti_360


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


def test_kitti360_uris():
  T = kitti_360.KITTI360SDTable
  uris = T.get_uris_for_sequence('2013_05_28_drive_0000_sync')
  uris = [u for u in uris if u.extra['kitti-360.frame_id'] == '106']

  for uri in uris:
    sd = T.create_stamped_datum(uri)
    print(sd.uri)

def test_kitti350_play():

  import imageio

  import numpy as np

  from pathlib import Path


  # FRAMEID = 8112#8096
  FRAMEID = 8096
  FRAMENAME = str(FRAMEID).rjust(10, '0')
  

  import xmltodict
  d = xmltodict.parse(open("/outer_root/media/seagates-ext4/au_datas/kitti/kitti-360/KITTI-360/data_3d_bboxes/train/2013_05_28_drive_0000_sync.xml").read())

  objects = d['opencv_storage']
  obj_name_to_value = dict(
    (k, kitti_360_get_parsed_node(v)) for (k, v) in objects.items())

  
  obs_in_frame = dict(
    (k, v) for (k, v) in obj_name_to_value.items()
    if (
      ((not v['dynamic']) and (v['start_frame'] <= FRAMEID <= v['end_frame'])) or
      False)   #)(v['dynamic'] and v['index'] == FRAMEID))
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
  
  img = imageio.imread(ROOT / ('data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/%s.png' % FRAMENAME))
  img2 = imageio.imread(ROOT / ('data_2d_raw/2013_05_28_drive_0000_sync/image_01/data_rect/%s.png'% FRAMENAME))

  vel_path = ROOT / ('data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/%s.bin' % FRAMENAME)
  cloud = np.fromfile(vel_path, dtype=np.float32)
  cloud = np.reshape(cloud, [-1, 4])
  cloud = cloud[:, :3]

  static_path =  ROOT / 'data_3d_semantics/2013_05_28_drive_0000_sync/static/007968_008291.ply'
  dynamic_path = ROOT / 'data_3d_semantics/2013_05_28_drive_0000_sync/dynamic/007968_008291.ply'

  import open3d
  static_cloud = open3d.io.read_point_cloud(str(static_path))
  dynamic_cloud = open3d.io.read_point_cloud(str(dynamic_path))


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
  cuboids_lidar = []
  print('obs_in_frame', len(obs_in_frame))

  close = None
  for obj_name, obj in obs_in_frame.items():
    from psegs import datum

    if obj['k360_class_name'] in ('building', 'garage'):
      continue

    # IMU frame: x = forward, y = right, z = down
    # +x +y +z
    # +x +y -z
    # +x -y +z
    # +x -y -z
    # -x +y -z
    # -x +y +z
    # -x -y -z
    # -x -y +z
    
    front_world = obj['cuboid'][[0, 1, 2, 3], :]
    rear_world = obj['cuboid'][[5, 4, 7, 6], :]

    # Now:
    # +x +y +z
    # +x +y -z
    # +x -y +z
    # +x -y -z
    # -x +y +z
    # -x +y -z
    # -x -y +z
    # -x -y -z

    print(front_world - np.mean(obj['cuboid'], axis=0))
    print(rear_world - np.mean(obj['cuboid'], axis=0))

    # w = 1.5#abs(front_world[0, 0] - rear_world[0, 0])
    # l = 2.5#abs(front_world[0, 1] - front_world[2, 1])
    # h = 1.5#abs(front_world[0, 2] - front_world[1, 2])
    w = np.linalg.norm(front_world[0, :] - front_world[2, :])
    l = np.linalg.norm(front_world[0, :] - rear_world[0, :])
    h = np.linalg.norm(front_world[0, :] - front_world[1, :])

    T_world = np.mean(obj['cuboid'], axis=0)
    # print(obj['cuboid'] - T_world)

    from scipy.spatial.transform import Rotation as R
    import math
    heading = front_world[0, :] - rear_world[0, :]
    heading_hat = heading / np.linalg.norm(heading)
    X_HAT = np.array([1, 0, 0])
    cos_theta = heading_hat.dot(X_HAT)
    rot_axis = np.cross(heading_hat, X_HAT)
    R_world2 = R.from_rotvec(
      math.acos(cos_theta) * rot_axis / np.linalg.norm(rot_axis)).as_matrix()
    # WTF why doesn't this work??
    
    # KITTI-360 Transform confounds R and S; we need to separate them.
    # See also https://math.stackexchange.com/a/1463487
    obj_sR = obj['transform'][:3, :3]
    sx = np.linalg.norm(obj_sR[:, 0])
    sy = np.linalg.norm(obj_sR[:, 1])
    sz = np.linalg.norm(obj_sR[:, 2])
    R_world = obj_sR.copy()
    R_world[:, 0] *= 1. / sx
    R_world[:, 1] *= 1. / sy
    R_world[:, 2] *= 1. / sz

    print('R_world - R_world2', R_world - R_world2)

    # heading *= 2 * np.pi # effectively zero rotation about axis
    # R_world = R.from_rotvec(heading).as_matrix()
    # R_world = np.eye(3, 3)

    T_world_to_obj = datum.Transform.from_transformation_matrix(
                          np.column_stack([R_world, T_world]),
                          src_frame='world',
                          dest_frame='obj')

    RT_cam0_to_world = cam0_to_world[FRAMEID]
    
    RT_world_to_ego = poses_idk[FRAMEID]
    T_world_to_ego = datum.Transform.from_transformation_matrix(
                          RT_world_to_ego,
                          src_frame='world',
                          dest_frame='ego')

    T_ego_to_velo = (
        calib.cam_left_raw_to_ego @
      calib.cam_left_raw_to_velo.get_inverse())
    T_world_to_velo = (
      T_world_to_ego @ T_ego_to_velo).get_inverse()


    # print('little', cam0_to_world[FRAMEID + 1][:3, 3] - cam0_to_world[FRAMEID][:3, 3])
    # print('big', cam0_to_world[8122][:3, 3] - cam0_to_world[FRAMEID][:3, 3])

    
    # R_cam0_to_world = np.linalg.inv(wtf_fwd) @ RT_cam0_to_world[:3, :3] @ wtf_fwd

    R_cam0_to_world = RT_cam0_to_world[:3, :3]
    T_cam0_to_world = RT_cam0_to_world[:3, 3]

    # if close is None:
    #   close = obj
    # else:
    #   dist = np.linalg.norm(obj['transform'][:3, 3] - T_cam0_to_world)
    #   cdist = np.linalg.norm(close['transform'][:3, 3] - T_cam0_to_world)
    #   if dist < cdist:
    #     close = obj

    # T_cam0_to_world = -T_cam0_to_world[[0, 2, 1]]
    # R_cam0_to_world = R_cam0_to_world.T
      # World y and z axes are flipped vs lidar/ego
      # https://github.com/autonomousvision/kitti360Scripts/blob/081c08b34a14960611f459f23a0ad049542205c6/kitti360scripts/helpers/project.py#L17
    # import pdb; pdb.set_trace()
    Tr_cam0_to_world = datum.Transform(
                          rotation=R_cam0_to_world,
                          translation=T_cam0_to_world,
                          src_frame='world',
                          dest_frame='camera|left_raw')
                            # Name should be "from" not "to" ?
                            # https://github.com/autonomousvision/kitti360Scripts/blob/081c08b34a14960611f459f23a0ad049542205c6/kitti360scripts/helpers/project.py#L106

    # print('T_world_to_obj.translation - T_cam0_to_world.translation', T_world_to_obj.translation - T_cam0_to_world.translation)
    # if np.linalg.norm(T_world_to_obj.translation - T_cam0_to_world.translation) > 10:
    #   continue
    
    # Tr_cam0_to_world name is backwards?
    T_obj_from_cam = (
      T_world_to_obj.get_inverse() @ Tr_cam0_to_world).get_inverse()
    T_obj_from_cam.src_frame = 'ego' # actually cam but tag this way to make psegs happy
    T_obj_from_cam.dest_frame = 'obj'


    # T_obj_from_ego = (
    #   T_world_to_obj @ 
    #   Tr_cam0_to_world['world', 'camera|left_raw'])
    #     # left camera is ego?  or do we need `poses` that has lidar?
    # T_obj_from_ego.src_frame = 'ego'
    # T_obj_from_ego.dest_frame = 'obj'

    c = datum.Cuboid(
          track_id=obj_name,
          category_name=obj['k360_class_name'], #obj['label'],
          length_meters=l,
          width_meters=w,
          height_meters=h,
          obj_from_ego=T_obj_from_cam)
    cuboids_cam.append(c)
    # import pdb; pdb.set_trace()

    ### Lidar

    # From kitti tracking
    # kitti_Tr_imu_to_velo = np.array([
    #   9.999976000000e-01, 7.553071000000e-04, -2.035826000000e-03, 
    #   -8.086759000000e-01, -7.854027000000e-04, 9.998898000000e-01,
    #    -1.482298000000e-02, 3.195559000000e-01, 2.024406000000e-03,
    #     1.482454000000e-02, 9.998881000000e-01, -7.997231000000e-01])

    # kitti_imu_to_velo = datum.Transform.from_transformation_matrix(
    #   np.reshape(kitti_Tr_imu_to_velo, (3, 4)),
    #   src_frame='oxts', dest_frame='lidar')

    
    T_obj_from_velo = (
      T_world_to_obj.get_inverse() @ T_world_to_ego @ T_ego_to_velo).get_inverse()

      #calib.cam_left_raw_to_velo)
      # @ )
          # cam_left_raw_to_velo is really opposite?
    # print('T_obj_from_ego', T_obj_from_ego.translation)
    # T_lidar_to_world = (
    #   T_world_to_obj['obj', 'world'] @ 
    #   T_cam0_to_world['world', 'camera|left_raw'])
    
    # (
    #   calib.cam_left_raw_to_velo.get_inverse() @ T_cam0_to_world)
        # TODO: fix name?
    # T_obj_from_ego = (
    #   T_world_to_obj['obj', 'world'] @ 
    #   T_lidar_to_world['world', 'lidar'])
    #     # left camera is ego?  or do we need `poses` that has lidar?
    # T_obj_from_ego.rotation = np.eye(3, 3)#T_obj_from_ego.rotation.T
    # T_obj_from_ego.translation = T_obj_from_ego.translation[[1, 0, 2]]
    # T_obj_from_ego.translation[1] *= -1
    # T_obj_from_ego.translation[0] *= -1

    # T_obj_from_ego.translation -= calib.cam_left_raw_to_velo.translation

    # T_obj_from_ego.translation = T_obj_from_ego.rotation.T
    T_obj_from_velo.src_frame = 'ego' # actually velo but tag this way to make psegs happy
    T_obj_from_velo.dest_frame = 'obj'

    c = datum.Cuboid(
          track_id=obj_name,
          category_name=obj['k360_class_name'],
          length_meters=l,
          width_meters=w,
          height_meters=h,
          obj_from_ego=T_obj_from_velo)
    cuboids_lidar.append(c)
  
  # print('close', close)
  # print('T_cam0_to_world', T_cam0_to_world)
  # print('cdist', cdist, close['transform'][:3, 3] - T_cam0_to_world)
  # print('cdist future', close['transform'][:3, 3] - cam0_to_world[8122][:3, 3])
  # assert False

  from psegs import util
  from psegs import datum
  outdir = Path('/opt/psegs/test_run_output')

  

  frame = 'yay'
  pc = datum.PointCloud(cloud=cloud)
  util.log.info("Projecting BEV %s ..." % frame)
  import time
  start = time.time()
  bev_img = pc.get_bev_debug_image(cuboids=cuboids_lidar)
  print('bev', time.time() - start)
  fname = 'projected_lidar_labels_bev_%s.png' % frame.replace('/', '_')
  imageio.imwrite(outdir / fname, bev_img)

  util.log.info("Projecting Front RV %s ..." % frame)
  import time
  start = time.time()
  rv_img = pc.get_front_rv_debug_image(cuboids=cuboids_lidar)
  print('rv', time.time() - start)
  fname = 'projected_lidar_labels_front_rv_%s.png' % frame.replace('/', '_')
  imageio.imwrite(outdir / fname, rv_img)



  frame = 'static'
  static_cloud_arr = np.asarray(static_cloud.points)
  
  # xform = vel_from_cam @ Tr_cam0_to_world['camera|left_raw', 'world']
  static_cloud_arr = T_world_to_velo.apply(static_cloud_arr).T
  # static_cloud_arr -= np.mean(static_cloud_arr, axis=0)
  pc = datum.PointCloud(cloud=static_cloud_arr)
  util.log.info("Projecting BEV %s ..." % frame)
  import time
  start = time.time()
  bev_img = pc.get_bev_debug_image(cuboids=cuboids_lidar)
  print('bev', time.time() - start)
  fname = 'projected_lidar_labels_bev_%s.png' % frame.replace('/', '_')
  imageio.imwrite(outdir / fname, bev_img)

  util.log.info("Projecting Front RV %s ..." % frame)
  import time
  start = time.time()
  rv_img = pc.get_front_rv_debug_image(cuboids=cuboids_lidar)
  print('rv', time.time() - start)
  fname = 'projected_lidar_labels_front_rv_%s.png' % frame.replace('/', '_')
  imageio.imwrite(outdir / fname, rv_img)



  frame = 'dynamic'
  dynamic_cloud_arr = np.asarray(dynamic_cloud.points)
  dynamic_cloud_arr -= np.mean(dynamic_cloud_arr, axis=0)
  pc = datum.PointCloud(cloud=dynamic_cloud_arr)
  util.log.info("Projecting BEV %s ..." % frame)
  import time
  start = time.time()
  bev_img = pc.get_bev_debug_image(cuboids=cuboids_lidar)
  print('bev', time.time() - start)
  fname = 'projected_lidar_labels_bev_%s.png' % frame.replace('/', '_')
  imageio.imwrite(outdir / fname, bev_img)

  util.log.info("Projecting Front RV %s ..." % frame)
  import time
  start = time.time()
  rv_img = pc.get_front_rv_debug_image(cuboids=cuboids_lidar)
  print('rv', time.time() - start)
  fname = 'projected_lidar_labels_front_rv_%s.png' % frame.replace('/', '_')
  imageio.imwrite(outdir / fname, rv_img)



  frame = 'left'

  from psegs.util import plotting as pspl

  util.log.info("Projecting cloud %s ..." % frame)

  cloud2 = calib.cam_left_raw_to_velo.get_inverse().apply(cloud).T

  uvd = calib.cam0_K.dot(cloud2.T).T
  uvd[:, 1] /= uvd[:, 2]
  uvd[:, 0] /= uvd[:, 2]

  # import pdb; pdb.set_trace()


  debug = img.copy()
  pspl.draw_xy_depth_in_image(debug, uvd, marker_radius=0, alpha=0.7)
  for c in cuboids_cam:
    pts = c.get_box3d()
    uvd = calib.cam0_K.dot(pts.T).T
    uvd[:, 1] /= uvd[:, 2]
    uvd[:, 0] /= uvd[:, 2]
    if (uvd[:, 2] <= 1e-3).any():
      continue

    from oarphpy.plotting import hash_to_rbg
    color = pspl.color_to_opencv(
      np.array(hash_to_rbg(c.category_name)))
    pspl.draw_cuboid_xy_in_image(debug, uvd[:, :2], color)
  fname = 'projected_pts_front_cam_%s.png' % frame.replace('/', '_')
  imageio.imwrite(outdir / fname, debug)




  frame = 'yay_right'
  util.log.info("Projecting cloud %s ..." % frame)

  cloud2 = calib.cam_left_raw_to_velo.get_inverse().apply(cloud).T
  cloud2 = calib.RT_01.apply(cloud2).T

  uvd = calib.cam1_K.dot(cloud2.T).T
  uvd[:, 1] /= uvd[:, 2]
  uvd[:, 0] /= uvd[:, 2]


  debug = img2.copy()
  pspl.draw_xy_depth_in_image(debug, uvd, marker_radius=0, alpha=0.7)
  fname = 'projected_pts_front_cam_%s.png' % frame.replace('/', '_')
  imageio.imwrite(outdir / fname, debug)

