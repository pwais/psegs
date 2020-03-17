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
import typing

import attr

from psegs.datum.uri import DatumSelection
from psegs.datum.uri import URI
from psegs.datum.stamped_datum import StampedDatum


@attr.s(slots=True, eq=True, weakref_slot=False)
class Frame(object):
  """A `Frame` is a group of :class:`~psegs.datum.stamped_datum.StampedDatum`
  instances that centers around a single event or purpose.  For example, a
  `Frame` may group all datums around a labels for a specific timestamp; in
  particular, a `Frame` may be used to synchronized camera, lidar, and label
  data.
  
  Notes:
   * `Frame`s are intended to be a utility for serialized `StampedDatum`s
     rather than serialized themselves.
  """

  datums = attr.ib(type=typing.List[StampedDatum], default=[])
  """List[StampedDatum]: All datums associated with this `Frame`"""

  uri = attr.ib(type=URI, default=None)
  """URI: The URI addressing this frame (and group of datums)"""

  def __attrs_post_init__(self):
    if not self.uri:
      if self.datums:
        # Note this is not safe ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        base_uri = sorted(self.datums)[0].uri
        self.uri = copy.deepcopy(base_uri)
    
    if self.uri and not self.uri.sel_datums:
      self.uri.sel_datums = DatumSelection.selections_from_value(self.datums)

  # @property
  # def uri(self):
  #   kwargs = dict((attr, getattr(self, attr)) for attr in URI.__slots__)
  #   kwargs['sel_datums'] = DatumSelection.selections_from_value(self.datums)
  #   return URI(**kwargs)


  ## Topic selectors

  def topic_datums(self, topic=None, prefix=None):
    """Return all `StampedDatum` instances for the given topic.

    Args:
      topic (str): Select all datums from this topic, e.g. `camera|front`.
      prefix (str): Select all datums with this topic prefix; E.g.
        `camera` selects `camera|front` and `camera|back`.
    
    Returns
      List[StampedDatum]: The selected datums
    """
    
    def is_from_topic(datum):
      if topic is not None:
        return datum.topic == topic
      elif prefix is not None:
        return datum.topic.startswith(prefix)
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

  # def to_html(self):
  #   from datetime import datetime
  #   import tabulate
  #   import pprint
  #   uri = self.uri
    
  #   def get_topic_offset_html(datums):
  #     topic_time = [(d.topic, d.timestamp) for d in datums]
  #     topic_time.sort(key=lambda t: -t[-1])
  #     end = 0
  #     if topic_time:
  #       end = topic_time[0][-1]
  #     table = [['Topic', 'Relative to Oldest Datum (msec)']]
  #     table += [[topic, '-%5.2f' % (1e-6 * (end - t))] for topic, t in topic_time]
  #     return tabulate.tabulate(table, tablefmt='html')

  #   table = [
  #     ['URI', to_preformatted(uri)],
  #     ['Timestamp', 
  #       datetime.utcfromtimestamp(uri.timestamp * 1e-9).strftime('%Y-%m-%d %H:%M:%S')],
  #     ['Extra', to_preformatted(self.extra)],
  #     ['Datums', to_preformatted(sorted(str(d.uri) for d in self.datums))],
  #     ['Offsets', get_topic_offset_html(self.datums)],
  #   ]
  #   html = tabulate.tabulate(table, tablefmt='html')
  #   table = [['<h2>Camera Images</h2>']]
  #   for c in self.camera_images:
  #     c = copy.deepcopy(c)
  #     # TODO: find a way to get rid of clouds from camera_image ~~~~~~~~~~~~~~~~~~~~~
  #     c.clouds += self.lidar_clouds
  #     for cuboid in self.cuboids:
  #       bbox = c.project_cuboid_to_bbox(cuboid)
  #       if not bbox.is_visible:
  #         continue
  #       c.bboxes.append(bbox)
  #     table += [[c.to_html()]]
    
  #   table += [['<h2>Point Clouds</h2>']]
  #   for c in self.lidar_clouds:
  #     table += [[c.to_html(cuboids=self.cuboids)]]

  #   html += tabulate.tabulate(table, tablefmt='html')
  #   return html
