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
  """A single piece of data associated with a specific time and a specific
  segment of data (i.e. a single `segment_id`).  Represents a single row in a
  `StampedDatumTable`.  While the object has multiple attributes, only one
  data attribute typically has non-vacuous value.
  """


  ## Every datum can be addressed
  
  uri = attr.ib(type=URI, default=None)
  """URI: The URI addressing this datum; also defines sort order"""


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

  # __slots__ = tuple(list(URI.__slots__) + [
  #   # Inherit everything from a URI; we'll use URIs to address StampedDatums.
  #   # Python users can access URI attributes directly or thru the `uri`
  #   # property below.
  #   # Parquet users can partition data using URI attributes.

  #   # The actual Data
  #   'camera_image',       # type: CameraImage
  #   'point_cloud',        # type: PointCloud
  #   'cuboids',            # type: List[Cuboid]
  #   # 'bboxes',             # type: List[BBox]
  #   'transform',          # type: Transform
  # ])

  # def __init__(self, **kwargs):
  #   # Handle subclass attributes first to let superclass control defaults for
  #   # its own attributes
  #   DEFAULTS = {
  #     'cuboids': [],
  #   }
  #   _set_defaults(self, kwargs, DEFAULTS)
  #   super(StampedDatum, self).__init__(**kwargs)

  # @property
  # def uri(self):
  #   kwargs = dict((attr, getattr(self, attr)) for attr in URI.__slots__)
  #   return URI(**kwargs)

  # @classmethod
  # def from_uri(cls, uri, **init_kwargs):
  #   sd = cls(**attr.asdict(uri))
  #   sd.update(**init_kwargs)
  #   return sd


###
### Prototypes
###

# TODO make this easier to establish schema ? ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# or just doc why we have to spec some members so that types can be inferred

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
