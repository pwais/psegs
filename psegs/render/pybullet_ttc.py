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

# TODO https://pybullet.org/Bullet/phpBB3/viewtopic.php?f=24&t=13269 

import tempfile

import attr

from psegs.util import misc


@attr.s(slots=True, eq=False, weakref_slot=False)
class CuboidAgent(object):
  obj_id = attr.ib(default=-1)

  material_color = attr.ib(default='white')

  mass_kg = attr.ib(default=1)

  size_xyz = attr.ib(default=[1., 1., 1.])

  init_xyz = attr.ib(default=[0., 0., 0.])
  init_rpy = attr.ib(default=[0., 0., 0.])

  init_velocity = attr.ib(default=[0., 0., 0.])
  init_angular_velocity = attr.ib(default=[0., 0., 0.])

  constant_acceleration = attr.ib(default=[0., 0., 0.])


  def pybullet_init(self, p):

    sz_x, sz_y, sz_z = self.size_xyz
    CUBE_URDF = f"""
      <?xml version="1.0"?>
      <robot name="psegs_cuboid">
        <link name="base_link">
          <visual>
            <geometry>
              <box size="{sz_x} {sz_y} {sz_z}"/>
            </geometry>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <material name="{self.material_color}"/>
          </visual>
          <collision>
            <geometry>
              <box size="{sz_x} {sz_y} {sz_x}"/>
            </geometry>
            <origin rpy="0 0 0" xyz="0 0 0"/>
          </collision>
        </link>
      </robot>
      """
    
    urdf_path = tempfile.NamedTemporaryFile(suffix='.urdf').name
    with open(urdf_path, 'w') as f:
      f.write(CUBE_URDF)
    
    self.obj_id = p.loadURDF(urdf_path)

    p.changeDynamics(self.obj_id, -1, mass=self.mass_kg)

    p.resetBaseVelocity(
      objectUniqueId=self.obj_id,
      linearVelocity=self.init_velocity,
      angularVelocity=self.init_angular_velocity)
    
    p.resetBasePositionAndOrientation(
      bodyUniqueId=self.obj_id,
      posObj=self.init_xyz,
      ornObj=p.getQuaternionFromEuler(self.init_rpy))

  
  def step_acceleration(self, p):
    p.applyExternalForce(
          objectUniqueId=self.obj_id,
          linkIndex=-1,
          forceObj=self.constant_acceleration,
          posObj=[0, 0, 0],
          flags=p.LINK_FRAME)


@attr.s(slots=True, eq=False, weakref_slot=False)
class PyBulletSim(object):

  ground_plane_id = attr.ib(default=-1)

  cuboid_agents = attr.ib(default=[])

  time_step_Hz = attr.ib(default=20)

  duration_sec = attr.ib(default=2)

  debug_cam_distance = attr.ib(default=5)
  debug_cam_look_at_agent = attr.ib(default=0)
  debug_image_width = attr.ib(default=900)
  debug_image_height = attr.ib(default=900)

  @classmethod
  def start_direct(cls):
    import pybullet as p
    p.connect(p.DIRECT)
    return p

  def _set_up_world(self, p):
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.resetSimulation()

    self.ground_plane_id = p.loadURDF('plane.urdf')

    for aa in self.cuboid_agents:
      aa.pybullet_init(p)


  def run(self, p=None, debug_video_out='/tmp/pybullet_debug.mp4'):
    if p is None:
      p = self.start_direct()
      self._set_up_world(p)

    p.setTimeStep(1. / self.time_step_Hz)

    misc.log.info(
      f"Running pybullet sim with {p.getNumBodies()} bodies "
      "for {self.duration_sec} seconds at "
      "{self.time_step_Hz} Hz ...")

    debug_writer = None
    debug_look_at = [0., 0., 0.]
    if debug_video_out:
      import imageio
      debug_writer = imageio.get_writer(debug_video_out, fps=self.time_step_Hz)

      aa = self.cuboid_agents[self.debug_cam_look_at_agent]
      debug_look_at = aa.init_xyz

    t_sec = 0
    n_steps = 0
    while t_sec < self.duration_sec:

      # Update accelerations
      for aa in self.cuboid_agents:
        aa.step_acceleration(p)

      p.stepSimulation()
      p.performCollisionDetection()

      for aa1 in self.cuboid_agents:
        for aa2 in self.cuboid_agents:
          if aa1.obj_id < aa2.obj_id:
            print(aa1.obj_id, aa2.obj_id)
            pts = p.getClosestPoints(aa1.obj_id, aa2.obj_id, float('inf'))
            print('len(pts)', len(pts))
            distance = pts[0][8]
            #print("distance=",distance)
            ptA = pts[0][5]
            ptB = pts[0][6]
            print('distance', distance, 'ptA', ptA, 'ptB', ptB)
      
      if debug_writer is not None:
        import numpy as np
        from PIL import Image
        # from IPython.display import display

        result = p.getCameraImage(
            self.debug_image_width,
            self.debug_image_height,
            viewMatrix=
              p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=debug_look_at,
                distance=self.debug_cam_distance,
                yaw=20,
                pitch=-10,
                roll=0,
                upAxisIndex=2),
            projectionMatrix=p.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(self.debug_image_width) / self.debug_image_height,
                nearVal=0.01,
                farVal=500.),
            shadow=True,
            lightDirection=[1, 1, 1])

        width, height, rgba, depth, mask = result

        # print(f"rgba shape={rgba.shape}, dtype={rgba.dtype}")
        debug_writer.append_data(rgba[:, :, :3].astype(np.uint8))
        
        # display(Image.fromarray(rgba, 'RGBA'))
        # print(f"depth shape={depth.shape}, dtype={depth.dtype}, as values from 0.0 (near) to 1.0 (far)")
        # display(Image.fromarray((depth*255).astype('uint8')))
        # print(f"mask shape={mask.shape}, dtype={mask.dtype}, as unique values from 0 to N-1 entities, and -1 as None")
        # display(Image.fromarray(np.interp(mask, (-1, mask.max()), (0, 255)).astype('uint8')))


      t_sec += 1. / self.time_step_Hz
      n_steps += 1
      if (n_steps % 100) == 0:
        misc.log.info(f"... rendered {n_steps} steps ...")
    
    if debug_writer is not None:
      debug_writer.close()



