

import numpy as np

# import tf
# import os
# import cv2
import rospy
# import rosbag
# import progressbar

# from datetime import datetime
# from std_msgs.msg import Header
# from sensor_msgs.msg import CameraInfo, Imu, PointField, NavSatFix
# import sensor_msgs.point_cloud2 as pcl2
# from geometry_msgs.msg import TransformStamped, TwistStamped, Transform
# from cv_bridge import CvBridge
# import numpy as np
# import argparse

# https://github.com/tomas789/kitti2bag/blob/master/kitti2bag/kitti2bag.py





###############################################################################
## PSegs -> ROS: Utils

def nanostamp_to_rostime(nanostamp):
  # For a good time https://github.com/pgao/roscpp_core/commit/dffa31afe8d7f1268a3fa227408aeb6e04a28b87#diff-65b9485bd6b5d3fb4b7a84cd975c3967L157
  return rospy.Time(
            secs=int(nanostamp / 1000000000),
            nsecs=int(nanostamp % 1000000000))


def to_ros_frame(s):
  s = s.replace('|', '_')
  s = s.replace('/', '_')
  return s


def to_ros_topic(s):
  s = s.replace('|', '_')
  return '/' + s


def to_ros_arr(arr):
  return arr.flatten(order='C').tolist()


###############################################################################
## PSegs -> ROS: datum conversion

def transform_to_ros(xform, nanostamp=None):
  import tf
  from tf2_msgs.msg import TFMessage
  from geometry_msgs.msg import Transform
  from geometry_msgs.msg import TransformStamped

  tf_msg = TFMessage()
  tf_transform = TransformStamped()
  if nanostamp is not None:
    tf_transform.header.stamp = nanostamp_to_rostime(nanostamp)
  
  tf_transform.header.frame_id = to_ros_frame(xform.src_frame)
  tf_transform.child_frame_id = to_ros_frame(xform.dest_frame)

  transform = Transform()
  r_4x4 = np.ones((4, 4))
  r_4x4[:3, :3] = xform.rotation
  q = tf.transformations.quaternion_from_matrix(r_4x4)
  transform.rotation.x = q[0]
  transform.rotation.y = q[1]
  transform.rotation.z = q[2]
  transform.rotation.w = q[3]

  transform.translation.x = xform.translation[0]
  transform.translation.y = xform.translation[1]
  transform.translation.z = xform.translation[2]

  tf_transform.transform = transform
  tf_msg.transforms.append(tf_transform)
  return tf_msg


def ci_to_ros_camera_info(ci):
  from sensor_msgs.msg import CameraInfo
  info = CameraInfo()
  info.header.frame_id = to_ros_frame(ci.sensor_name)
  info.header.stamp = nanostamp_to_rostime(ci.timestamp)
  info.width = ci.width
  info.height = ci.height
  info.distortion_model = 'plumb_bob'

  info.K = to_ros_arr(ci.K)
  P = np.zeros((3, 4))
  P[:3, :3] = ci.K
  info.P = to_ros_arr(P)

  return info


def ci_to_ros_image(ci):
  import cv2
  from cv_bridge import CvBridge
  bridge = CvBridge()

  img_arr = np.asarray(ci.image_png, dtype=np.uint8)
  cv_img = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)
  ros_img_msg = bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
  ros_img_msg.header.frame_id = to_ros_frame(ci.sensor_name)
  ros_img_msg.header.stamp = nanostamp_to_rostime(ci.timestamp)
  
  return ros_img_msg


def pc_to_ros_pcl(pc):
  from sensor_msgs.msg import PointField
  from std_msgs.msg import Header
  import sensor_msgs.point_cloud2 as pcl2
  
  header = Header()
  header.frame_id = 'ego' # fixme? ~~~~~~~~~~~~~~~~~~~~~~~~` to_ros_frame(pc.sensor_name)
  header.stamp = nanostamp_to_rostime(pc.timestamp)

  cloud = pc.get_cloud()
  xyz = cloud.astype(np.float32)
  assert xyz.shape[-1] == 3

  from psegs.util.plotting import rgb_for_distance
  colors = rgb_for_distance(np.linalg.norm(xyz, axis=1))

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb#file-create_cloud_xyzrgb-py-L28
  # https://github.com/cruise-automation/webviz/blob/2e7db3aafffec39b541728668c97ce7d83eee007/packages/webviz-core/src/panels/ThreeDimensionalViz/commands/Pointclouds/PointCloudBuilder.js#L117
  colors = colors.astype(int)
  colors_uint32 = (
    (2**16) * colors[:, 0] + (2**8) * colors[:, 1] + 1 * colors[:, 2])


  points = [pt + [c] for pt, c in zip(xyz.tolist(), colors_uint32)]

  # import pdb; pdb.set_trace()

  # points = np.hstack([xyz, colors_uint32[:, np.newaxis]])
  # import pdb; pdb.set_trace()

  fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1)]

  pcl_msg = pcl2.create_cloud(header, fields, points)

  return pcl_msg


def color_to_ros(color):
  """color in [0, 1] -> ROS color"""
  from std_msgs.msg import ColorRGBA
  r, g, b = np.clip(color, 0, 1).tolist()
  ros_color = ColorRGBA()
  ros_color.r = r
  ros_color.g = g
  ros_color.b = b
  ros_color.a = 1.
  return ros_color


def _box_face_marker(face_pts, color):
  from geometry_msgs.msg import Point
  from visualization_msgs.msg import Marker
  
  m = Marker()
  m.type = Marker.LINE_STRIP  # each point in points is part of the line
  m.action = Marker.MODIFY    # or add
  m.color = color
  m.scale.x = 0.1
  for i in range(5):
    p = Point()
    p.x, p.y, p.z = face_pts[i % 4, :].tolist()
    m.points.append(p)
  return m


def _box_sides_marker(front_pts, back_pts, color):
  from geometry_msgs.msg import Point
  from visualization_msgs.msg import Marker

  m = Marker()
  m.type = Marker.LINE_LIST # pairs of points create a line
  m.action = Marker.MODIFY  # or add
  m.color = color
  m.scale.x = 0.1
  for start, end in zip(front_pts.tolist(), back_pts.tolist()):
    startp = Point()
    startp.x, startp.y, startp.z = start
    endp = Point()
    endp.x, endp.y, endp.z = end
    m.points += [startp, endp]
  return m


def cuboids_to_ros_marker_array(cuboids):
  
  from visualization_msgs.msg import MarkerArray
  marray = MarkerArray()
  
  # We'll use the Line List and Line Strip Markers instead of the Cube marker
  # so that we can highlight the front face of the cuboid.
  for obj_id, cuboid in enumerate(cuboids):
    from std_msgs.msg import Header
    header = Header()
    header.frame_id = 'ego' # fixme? ~~~~~~~~~~~~~~~~~~~~~~~~` to_ros_frame(pc.sensor_name)
    header.stamp = nanostamp_to_rostime(cuboid.timestamp)

    from oarphpy.plotting import hash_to_rbg
    base_color = np.array(hash_to_rbg(cuboid.category_name)) / 255.
    front_color = color_to_ros(base_color + 0.3)
    back_color = color_to_ros(base_color - 0.3)
    sides_color = color_to_ros(base_color)

    box_xyz = cuboid.get_box3d()
    front = box_xyz[:4, :]
    back = box_xyz[4:, :]

    box_markers = [
      _box_face_marker(front, front_color),
      _box_sides_marker(front, back, sides_color),
      _box_face_marker(back, back_color),
    ]
    for mid, m in enumerate(box_markers):
      m.id = obj_id * 10 + mid
      m.ns = cuboid.track_id
      m.header = header
    
    marray.markers += box_markers
  
  return marray


###############################################################################
## PSegs -> ROS: RDD[StampedDatum] conversion

import attr

@attr.s(slots=True)
class ROSMsg(object):
  topic = attr.ib(default='')
  timestamp = attr.ib(default=0)
  msg = attr.ib(default=None)

  @classmethod
  def iter_rosmsgs_from_datum(cls, sd):
    msg_t = nanostamp_to_rostime(sd.uri.timestamp)

    transforms = []
    if sd.camera_image:
      namespace = to_ros_topic(sd.uri.topic)
      yield ROSMsg(
        timestamp=msg_t,
        topic=namespace + '/image',
        msg=ci_to_ros_image(sd.camera_image))
      yield ROSMsg(
        timestamp=msg_t,
        topic=namespace + '/camera_info',
        msg=ci_to_ros_camera_info(sd.camera_image))

      transforms += [sd.camera_image.ego_pose, sd.camera_image.ego_to_sensor]

    elif sd.point_cloud:
      yield ROSMsg(
        timestamp=msg_t,
        topic=to_ros_topic(sd.uri.topic),
        msg=pc_to_ros_pcl(sd.point_cloud))
      
      transforms += [sd.point_cloud.ego_pose, sd.point_cloud.ego_to_sensor]

    elif sd.cuboids:
      yield ROSMsg(
        timestamp=msg_t,
        topic=to_ros_topic(sd.uri.topic),
        msg=cuboids_to_ros_marker_array(sd.cuboids))
      transforms += [c.ego_pose for c in sd.cuboids]

    elif sd.transform:
      transforms += [sd.transform]
    
    for transform in transforms:
      yield ROSMsg(
        timestamp=msg_t,
        topic='/tf',
        msg=transform_to_ros(transform))


def segment_to_bag(spark, sd_table, segment_id, dest):
  sd_rdd = sd_table.get_segment_datum_rdd(
                  spark, segment_id, time_ordered=True) # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # rosmsg_rdd = sd_rdd.flatMap(ROSMsg.iter_rosmsgs_from_datum)

  # from oarphpy import util as oputil
  # thru = oputil.ThruputObserver(name='seg_to_bag')

  import rosbag
  bag = rosbag.Bag(dest, 'w', compression='lz4')
  n = 0
  for sd in sd_rdd.toLocalIterator():
    for rosmsg in ROSMsg.iter_rosmsgs_from_datum(sd):
      bag.write(rosmsg.topic, rosmsg.msg, t=rosmsg.timestamp)
      n += 1
      if n % 100 == 0:
        print(bag)
  print('done')
  print(bag)
  bag.close()


class DynamicPubNode(object):
  def __init__(self):
    self._topic_to_pub = {}
    rospy.init_node('dynamicpubnode')

  def publish(self, topic, msg):
    if topic not in self._topic_to_pub:
      pub = rospy.Publisher(topic, type(msg), queue_size=0)
      self._topic_to_pub[topic] = pub
    pub = self._topic_to_pub[topic]
    pub.publish(msg)


def publish_segment(spark, sd_table, segment_id):
  sd_rdd = sd_table.get_segment_datum_rdd(
                  spark, segment_id, time_ordered=True)
  
  ros_node = DynamicPubNode()

  # should_play = [True]
  # import keyboard
  # def toggle():
  #   print('toggle')
  #   should_play[0] = not should_play[0]
  # keyboard.add_hotkey('space', toggle)
  
  n = 0
  for sd in sd_rdd.toLocalIterator():
    for rosmsg in ROSMsg.iter_rosmsgs_from_datum(sd):
      # if not should_play[0]:
      import time
      time.sleep(.1)
        # print('waited')
      ros_node.publish(rosmsg.topic, rosmsg.msg)
      n += 1
      if n % 100 == 0:
        print('published %s' % n)


if __name__ == '__main__':
  import copy
  print('moof')

  import numpy as np
  ds = np.array([10., 12., 15., 17., 20., 25., 34.])
  print(ds)
  from psegs.util.plotting import rgb_for_distance
  print(rgb_for_distance(ds))


  from psegs.datum import stamped_datum as sd

  info = ci_to_ros_camera_info(sd.CAMERAIMAGE_PROTO)

  ros_tf = transform_to_ros(sd.TRANSFORM_PROTO)

  ci = copy.deepcopy(sd.CAMERAIMAGE_PROTO)
  ci.image_png = bytearray(open('/outer_root/home/au/psegs/yay.png', 'rb').read())
  image_msg = ci_to_ros_image(ci)

  pc = copy.deepcopy(sd.POINTCLOUD_PROTO)
  pc.cloud = np.random.rand(10, 3)
  pcl_msg = pc_to_ros_pcl(pc)

  cuboids = [copy.deepcopy(sd.CUBOID_PROTO)] * 4
  for c in cuboids:
    c.obj_from_ego.src_frame = 'ego'
    c.obj_from_ego.dest_frame = 'obj'
  cube_markers = cuboids_to_ros_marker_array(cuboids)


  # ros_node = DynamicPubNode()
  # ros_node.publish('camera', image_msg)
  # ros_node.publish('pc', pcl_msg)
  # ros_node.publish('cube', cube_markers)
  # import sys
  # sys.exit()


  from psegs.spark import Spark
  from psegs.datasets import kitti
  with Spark.sess() as spark:
    # segment_to_bag(
    #   spark,
    #   kitti.KITTISDTable,
    #   'train-0009',
    #   '/outer_root/home/au/psegs/testbag.bag')
  
    publish_segment(
      spark,
      kitti.KITTISDTable,
      'train-0009')


  # import pdb; pdb.set_trace()
  print()
