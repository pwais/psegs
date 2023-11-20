# Copyright 2023 Maintainers of PSegs
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

import typing

import attr
import numpy as np

from psegs.datum import datumutils as du
from psegs.datum.transform import Transform


@attr.s(slots=True, eq=True, weakref_slot=False)
class Cuboid(object):
  """An 8-vertex cuboid"""

  ## Context

  track_id = attr.ib(type=str, default='')
  """str: String identifier; same object across many frames has same
  track_id"""

  category_name = attr.ib(type=str, default='')
  """str: Category of the cuboid, can be using the dataset category domain"""

  ps_category = attr.ib(type=str, default='')
  """str: `psegs` Category (typically coarser than `category_name`)"""

  timestamp = attr.ib(type=int, default=0)
  """int: Timestamp associated with this cuboid; typically a Unix stamp in
  nanoseconds.  Probably a Lidar timestamp."""

  extra = attr.ib(default={}, type=typing.Dict[str, str])
  """Dict[str, str]: A map for adhoc extra context"""

  ## Cuboid orientation and size

  length_meters = attr.ib(type=float, default=0.)
  """float: Length in ego frame, where +x is forward"""

  width_meters = attr.ib(type=float, default=0.)
  """float: Width in ego frame, where +y is left"""
  
  height_meters = attr.ib(type=float, default=0.)
  """float: Height in ego frame, where +z is up"""

  obj_from_ego = attr.ib(type=Transform, default=Transform())
  """Transform: From center of cuboid frame to ego / robot frame"""

  ego_pose = attr.ib(type=Transform, default=Transform())
  """Transform: From world to ego / robot frame at the cuboid's `timestamp`"""


  # ## Extra Context

  # distance_meters = attr.ib(type=float, default=0.)
  # """float: Distance from ego / robot to closest cuboid point"""

  # ## In robot / ego frame
  #   'length_meters',        # Cuboid frame: +x forward
  #   'width_meters',         #               +y left
  #   'height_meters',        #               +z up    
  #   'distance_meters',      # Dist from ego to closest cuboid point


  #   ## Points # TODO keep ? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #   'box3d',                # Points in ego / robot frame defining the cuboid.
  #                           # Given in order:
  #                           #   (+x +y +z)  [Front face CW about +x axis]
  #                           #   (+x -y +z)
  #                           #   (+x -y -z)
  #                           #   (+x +y -z)
  #                           #   (-x +y +z)  [Rear face CW about +x axis]
  #                           #   (-x -y +z)
  #                           #   (-x -y -z)
  #                           #   (-x +y -z)
  #   'motion_corrected',     # Is `3d_box` corrected for ego motion?

  #   ## In robot / ego frame
  #   'length_meters',        # Cuboid frame: +x forward
  #   'width_meters',         #               +y left
  #   'height_meters',        #               +z up    
  #   'distance_meters',      # Dist from ego to closest cuboid point
    
  #   # TODO
  #   # 'yaw',                  # +yaw to the left (right-handed)
  #   # 'pitch',                # +pitch up from horizon
  #   # 'roll',                 # +roll towards y axis (?); usually 0

  #   'obj_from_ego',         # type: Transform from ego / robot frame to object
  #   'ego_pose',             # type: Transform (ego from world)
    
  #   'extra',                # type: string -> string extra metadata

  # __slots__ = (
  #   ## Core
  #   'track_id',             # String identifier; same object across many frames
  #                           #   has same track_id
    
  # )

  # def __init__(self, **kwargs):
  #   _set_defaults(self, kwargs, {})
  #     # Default all to None

  # def __eq__(self, other):
  #   return _slotted_eq(self, other)

  @classmethod
  def merge_extras(cls, e1, e2):
    merged = dict(e1)
    for k, v in e2.items():
      if k == 'motion_corrected':
        merged[k] = str(bool(merged.get('motion_corrected')) or bool(v))
      else:
        merged[k] = v
    return merged

  def get_box3d(self):
    """Return the 3d box in ego / robot frame defining the cuboid.
        Given in order:
            (+x +y +z)  [Front face CW about +x axis]
            (+x -y +z)
            (+x -y -z)
            (+x +y -z)
            (-x +y +z)  [Rear face CW about +x axis]
            (-x -y +z)
            (-x -y -z)
            (-x +y -z)
    """
    l, w, h = self.length_meters, self.width_meters, self.height_meters
    CORNERS_IN_CUBE_FRAME = .5 * np.array([
                  [ l,  w,  h],  # Front
                  [ l, -w,  h],
                  [ l, -w, -h],
                  [ l,  w, -h],

                  [-l,  w,  h],  # Back
                  [-l, -w,  h],
                  [-l, -w, -h],
                  [-l,  w, -h],
    ])

    to_ego = self.obj_from_ego['ego', 'obj']
    corners_in_ego = to_ego.apply(CORNERS_IN_CUBE_FRAME)
    return corners_in_ego.T

  def to_html(self):
    import tabulate
    table = [
      [attr, du.to_preformatted(getattr(self, attr))]
      for attr in self.__slots__
    ]
    return tabulate.tabulate(table, tablefmt='html')

  @classmethod
  def get_merged(cls, c1, c2, mode='union', alpha=None):
    """Return a new cuboid via merging `c1` and `c2`.

    Args:
      c1 (Cuboid): Merge this cuboid with `c2`. Retain category and other
        context of `c1`.  
      c2 (Cuboid): Merge this cuboid with `c1`.
      mode (str): Merging mode. Choices:
        `union`: Pick a mean position and orientation and scale to fit points
        of both `c1` and `c2`.  Use to merge two objects (e.g. bicycle and its
        rider)
        `interpolate`: Interpolate (using `alpha`) between the positions and
        orientations of `c1` and `c2` and use the size of `c1`.  Use to 
        compute the interpolated position / cuboid of a track between positions
        at time `c1.timestamp` and `c2.timestamp`.
      alpha (float, optional): For interpolation, weight `c1` with 1-`alpha`
        and `c2` with `alpha`, where `alpha in [0, 1]`
    
    Returns:
      Cuboid: The merged cuboid
    """
    
    ## Find new box3d, maintaining orientation of old box.
    # Step 1: Compute mean centroid and pose
    if mode == 'union':
      alpha = 0.5

    c1_obj_from_ego = c1.obj_from_ego['ego', 'obj']
    c2_obj_from_ego = c2.obj_from_ego['ego', 'obj']

    merged_translation = (
      (1 - alpha) * c1_obj_from_ego.translation + 
      alpha * c2_obj_from_ego.translation)
        # NB: use alpha blend consistent with the definition of Slerp

    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp

    # # DELETEME WHEN NEW DATUMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # c1.obj_from_ego.rotation = np.reshape(c1.obj_from_ego.rotation, (3,3))
    # c2.obj_from_ego.rotation = np.reshape(c2.obj_from_ego.rotation, (3,3))

    rots = R.from_matrix([
      c1_obj_from_ego.rotation,
      c2_obj_from_ego.rotation,
    ])
    slerp = Slerp([0, 1], rots)
    merged_rot = slerp([alpha]).as_matrix()

    merged_transform = Transform(
      rotation=merged_rot,
      translation=merged_translation,
      dest_frame='obj',
      src_frame='ego',
    )

    # Step 2: Compute cuboid bounds given new pose
    if mode == 'union':
      # Project all the points of the cubes `c1` and `c2` into the new
      # merged frame
      all_pts_in_ego = np.concatenate((c1.get_box3d(), c2.get_box3d()))
      to_merged = merged_transform['obj', 'ego']
      all_pts_in_merged = to_merged.apply(all_pts_in_ego).T

      lwh = all_pts_in_merged.max(axis=0) - all_pts_in_merged.min(axis=0)
      length, width, height = lwh.tolist()

      # length = all_pts_in_merged[:,0].max() - all_pts_in_merged[:,0].min()
      # width = c1.width_meters
      # height = c1.height_meters

      # # A cube with each corner touches a point of unity in each dimension
      # UNIT_CUBE = np.array([
      #               [ 1,  1.,  1.],  # Front
      #               [ 1, -1.,  1.],
      #               [ 1, -1., -1.],
      #               [ 1,  1., -1.],

      #               [-1,  1.,  1.],  # Back
      #               [-1, -1.,  1.],
      #               [-1, -1., -1.],
      #               [-1,  1., -1.],
      # ])

      # # Send the unit cube into the object frame
      # cube_in_merged_frame = merged_transform['ego', 'obj'].apply(UNIT_CUBE).T
      # # import pdb; pdb.set_trace()
      # # Stretch the cuboid to fit all points
      # all_pts = np.concatenate((c1.get_box3d(), c2.get_box3d()))
      # merged_box3d = []
      # for i in range(8):
      #   corner = cube_in_merged_frame[i, :3]
      #   corner /= np.linalg.norm(corner)
      #   merged_box3d.append(
      #     # Scale corner by the existing point with the greatest projection
      #     corner * all_pts.dot(corner).max()
      #   )
      #   # import pdb; pdb.set_trace()
      # merged_box3d = np.array(merged_box3d)
    
    elif mode == 'interpolate':
      # Just fit the box from the first cuboid; assume the track is not
      # deformable
      length = c1.length_meters
      width = c1.width_meters
      height = c1.height_meters
      # radius = 0.5 * np.array([
      #   c1.length_meters, c1.width_meters, c1.height_meters])
      # box_in_cube_frame = UNIT_CUBE * radius
      # merged_box3d = merged_transform.apply(box_in_cube_frame).T

    else:
      raise ValueError(mode)
    
    # width = np.linalg.norm(merged_box3d[1] - merged_box3d[0])
    # length = np.linalg.norm(merged_box3d[4] - merged_box3d[0])
    # height = np.linalg.norm(merged_box3d[3] - merged_box3d[0])

    timestamp = c1.timestamp
    if mode == 'interpolate':
      diff = abs(c1.timestamp - c2.timestamp)
      timestamp += int((1 - alpha) * diff)

    return Cuboid(
      track_id=c1.track_id + '-' + mode + '-' + c2.track_id,
      category_name=c1.category_name,
      ps_category=c1.ps_category,
      timestamp=timestamp,
      # box3d=merged_box3d,~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      length_meters=float(length),
      width_meters=float(width),
      height_meters=float(height),
      # distance_meters=float(np.min(np.linalg.norm(merged_box3d, axis=-1))),~~~~~~~~
      obj_from_ego=merged_transform,
      extra=cls.merge_extras(c1.extra, c2.extra),
    )

  @classmethod
  def get_interpolated(cls, cuboids, target_timestamp, allow_future=False):
    """For each distrinct track in `cuboids`, return a single cuboid
    interpolated to have estimated pose at time `target_timestamp` based
    on cuboids before and after that target.  If the track does not have
    cuboids that straddle `target_timestamp`, then return the most
    recent cuboid, if there is one.  If `allow_future`, then return the
    cuboid closest in time, even if it's in the future (i.e. `cuboid.timestamp`
    is after [greater than] `target_timestamp`)."""

    track_id_to_cuboid = {}
    for cuboid in cuboids:
      track_id = cuboid.track_id
      track_id_to_cuboid.setdefault(track_id, [])
      track_id_to_cuboid[track_id].append(cuboid)
    
    cuboids_out = []
    for track_id in track_id_to_cuboid.keys():
      cuboids = track_id_to_cuboid[track_id]

      ## Nothing to interpolate
      if len(cuboids) == 1:
        c = cuboids[0]
        if c.timestamp < target_timestamp or allow_future:
          cuboids_out.append(c)
        continue
      
      ## Are there cuboids straddling `target_timestamp`?
      diff_cuboid = [(target_timestamp - c.timestamp, c) for c in cuboids]
      before = None
      after = None
      for c in cuboids:
        diff_t = target_timestamp - c.timestamp
        if diff_t <= 0:
          if not before or diff_t < abs(target_timestamp - before.timestamp):
            before = c
        else: # diff_t > 0; for after we use strictly after
          if not after or diff_t < abs(target_timestamp - after.timestamp):
            after = c

      if before is None:
        if allow_future:
          cuboids_out.append(after)
        continue
      if after is None:
        cuboids_out.append(before)
        continue
      
      ## Interpolate!
      alpha = (
        float(target_timestamp - before.timestamp) / 
          (after.timestamp - before.timestamp))
      assert 0 <= alpha <= 1, alpha
      interpolated = Cuboid.get_merged(
                        after, before, mode='interpolate', alpha=alpha)
      cuboids_out.append(interpolated)
    return cuboids_out
