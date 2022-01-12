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

import copy
import itertools
import typing

import attr
import numpy as np

from psegs.datum.camera_image import CameraImage
from psegs.datum.cuboid import Cuboid
from psegs.datum.point_cloud import PointCloud
from psegs.datum.transform import Transform
from psegs.datum.uri import DatumSelection
from psegs.datum.uri import URI


@attr.s(slots=True, eq=True, weakref_slot=False)
class StampedDatum(object):
  """A union-like class containing a single piece of data associated with a
  specific time ("stamp" or "timestamp") from a specific segment of data
  (i.e. a single `segment_id`).  Represents a single row in a
  `StampedDatumTable`.  While the object has multiple attributes, only one
  data attribute typically has non-vacuous value.
  """

  ## Every datum can be addressed
  
  uri = attr.ib(type=URI, default=None, converter=URI.from_str)
  """URI: The URI addressing this datum; also defines sort order"""

  def __lt__(self, other):
    """Ordering is by URI (and not by data content)"""
    # TODO Fixme it turns out when there's a tie, attrs will look at content anyways, we need to fix or dump this ............
    assert type(other) is type(self)
    return self.uri < other.uri

  ## A datum should contain exactly one of the following:

  camera_image = attr.ib(type=CameraImage, default=None)
  """CameraImage: A single camera image"""

  point_cloud = attr.ib(type=PointCloud, default=None)
  """PointCloud: A single point cloud"""

  cuboids = attr.ib(type=typing.List[Cuboid], default=[])
  """List[Cuboid]: Zero or more cuboids; topic name may indicate label or
  prediction."""

  transform = attr.ib(type=Transform, default=None)
  """Transform: A transform such as ego pose; topic indicates semantics"""



@attr.s(slots=True, eq=True, weakref_slot=False)
class Sample(object):
  """A `Sample` is a group of :class:`~psegs.datum.stamped_datum.StampedDatum`
  instances that centers around a specific event or purpose.  For example, a
  `Sample` may group all datums around a specific timestamp; in particular,
  a `Sample` may be used to synchronized camera, lidar, and label
  data.  A `Sample` is a container for a set of data specified in a
  :class:`~psegs.datum.uri.DatumSelection`.
  """

  datums = attr.ib(type=typing.List[StampedDatum], default=[])
  """List[StampedDatum]: All datums associated with this `Sample`"""

  uri = attr.ib(type=URI, default=None)
  """URI: The URI addressing this `Sample` (and group of datums)"""

  def __attrs_post_init__(self):
    if not self.uri:
      if self.datums:
        # Note this might effectively select a segment_id and/or uri.extra
        # data that is not consistent with the rest of the `datums`.  You
        # probably want to specify your own `uri`.
        base_uri = sorted(d.uri for d in self.datums)[0]
        self.uri = copy.deepcopy(base_uri)
    
    if self.uri:
      self.uri = copy.deepcopy(URI.from_str(self.uri))

    if self.uri and not self.uri.sel_datums:
      self.uri.sel_datums = DatumSelection.selections_from_value(self.datums)

  ## Topic selectors

  def topic_datums(self, topic=None, prefix=None):
    """Return all `StampedDatum` instances for the given topic.

    Args:
      topic (str): Select all datums from this topic, e.g. `camera|front`.
      prefix (str): Select all datums with this topic prefix; E.g.
        `camera` selects `camera|front` and `camera|back`.
    
    Returns:
      List[StampedDatum]: The selected datums
    """
    
    def is_from_topic(datum):
      if topic is not None:
        return datum.uri.topic == topic
      elif prefix is not None:
        return datum.uri.topic.startswith(prefix)
      else:
        raise ValueError("Must specify a topic or prefix")
    
    return [
      sd for sd in self.datums
      if is_from_topic(sd)
    ]

  @property
  def ego_poses(self):
    """Normalized selector for the `ego_pose`
    :class:`~psegs.datum.transform.Transform` canonical topic.
    Returns a list of transforms.
    """
    return [
      sd.transform for sd in self.topic_datums(topic='ego_pose')
    ]
  
  @property
  def camera_images(self):
    """Normalized selector for all camera
    :class:`~psegs.datum.camera_image.CameraImage` canonical topics.
    Returns a list of camera images.
    """
    return [
      sd.camera_image for sd in self.topic_datums(prefix='camera')
    ]
  
  @property
  def lidar_clouds(self):
    """Normalized selector for all lidar
    :class:`~psegs.datum.point_cloud.PointCloud` canonical topics.
    Returns a list of point clouds.
    """
    return [
      sd.point_cloud for sd in self.topic_datums(prefix='lidar')
    ]
  
  @property
  def cuboid_labels(self):
    """Normalized selector for the *label* :class:`~psegs.datum.Cuboid`
    canonical topic.  Returns a list of cuboids flattened from all available
    datums.
    """
    return list(itertools.chain.from_iterable(
      sd.cuboids for sd in self.topic_datums(topic='labels|cuboids')))

  ## Utils

  def get_topics(self):
    return sorted(set(sd.uri.topic for sd in self.datums))
  
  def get_uri_str_to_datum(self):
    return dict((str(datum.uri), datum) for datum in self.datums)


###
### Prototypes
###

# Spark (and `RowAdapter`) can automatically deduce schemas from object
# heirarchies, but these tools need non-null, non-empty members to deduce
# proper types.  Creating a DataFrame with an explicit schema can also
# improve efficiently dramatically, because then Spark can skip row sampling
# and parallelized auto-deduction.  The Prototypes below serve to provide
# enough type information for `RowAdapter` to deduce the full av.Frame schema.
# In the future, Spark may perhaps add support for reading Python 3 type
# annotations, in which case the Protoypes will be obviated.

# URI_PROTO_KWARGS = dict(
#   # Core spec; most URIs will have these set
#   dataset='proto',
#   split='train',
#   segment_id='proto_segment',
#   topic='topic',
#   timestamp=int(100 * 1e9), # In nanoseconds
  
#   # Uris can identify more specific things in a Frame
#   camera='camera_1',
#   camera_timestamp=int(100 * 1e9), # In nanoseconds
  
#   crop_x=0, crop_y=0,
#   crop_w=10, crop_h=10,
  
#   track_id='track-001',

#   extra={'key': 'value'},
# )
URI_PROTO_KWARGS = dict(
  extra={'key': 'value'},
  sel_datums=[DatumSelection(topic='t', timestamp=1)],
)
URI_PROTO = URI(**URI_PROTO_KWARGS)

TRANSFORM_PROTO = Transform()

CUBOID_PROTO = Cuboid(
  extra={'key': 'value'},
)

# BBOX_PROTO = BBox(
#   x=0, y=0,
#   width=10, height=10,
#   im_width=100, im_height=100,
#   category_name='vehicle',
#   au_category='car',

#   cuboid=CUBOID_PROTO,
#   cuboid_pts=np.ones((8, 3), dtype=np.float32),
#   cuboid_center=np.array([1., 2., 3.]),
#   cuboid_in_cam=np.ones((8, 3), dtype=np.float32),

#   has_offscreen=False,
#   is_visible=True,

#   cuboid_from_cam=np.array([1., 0., 1.]),
#   ypr_camera_local=np.ones((1, 3)),
# )

POINTCLOUD_PROTO = PointCloud(
  cloud=np.ones((10, 3), dtype=np.float32),
  extra={'key': 'value'},
)

CAMERAIMAGE_PROTO = CameraImage(
  extra={'key': 'value'},
)

STAMPED_DATUM_PROTO = StampedDatum(
  uri=URI_PROTO,
  camera_image=CAMERAIMAGE_PROTO,
  point_cloud=POINTCLOUD_PROTO,
  cuboids=[CUBOID_PROTO],
  transform=TRANSFORM_PROTO,
)

