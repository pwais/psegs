# Copyright 2022 Maintainers of PSegs
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


from pathlib import Path

import six

from psegs import datum
from psegs import util
from psegs.spark import Spark
from psegs.table.sd_table_factory import StampedDatumTableFactory


###############################################################################
## StampedDatumTableFactory for an Video

def load_video_frame(video_uri, frame_idx):
  # Defined here for easier serializeation
  import imageio
  r = imageio.get_reader(video_uri)
  return r.get_data(frame_idx)

class AdhocVideosSDTFactory(StampedDatumTableFactory):
  """This `StampedDatumTableFactory` wraps a single video and exposes the
  video frames as a segment of (lazy-loaded) PSegs `CameraImage`s.  See
  `create_factory_for_video()` below for quickstart.
  """

  VIDEO_URI = ''
  START_NANOSTAMP = 0
  FRAMES_PER_SECOND = 1
  N_FRAMES = 0
  HEIGHT = 0
  WIDTH = 0

  DATASET = 'anon'
  SPLIT = 'anon'
  SEGMENT_ID = 'anon_segment'
  TOPIC = 'camera_video_adhoc'

  @classmethod
  def create_factory_for_video(
        cls,
        video_uri,
        dataset='anon',
        split='anon',
        topic='camera_video_adhoc',
        segment_id=None,
        start_timestamp_use='st_mtime',
        limit=-1):
    """Create and return a `StampedDatumTableFactory` class instance
    for the given video.  We will read part of the video to assess
    image dimensions, length, etc.

    If `start_timestamp_use` is a string, we will infer timestamps by not
    just the frame index but by stat-ing the video and using this 
    stat attribute.

    If `start_timestamp_use` is an integer, we'll use that nanostamp
    offset instead.

    Use this factory-factory function to produce a factory for your images,
    and then:
     * Use `create_sd_table()` to directly get a `StampedDatumTable` with your
        wrapped `CameraImage` instances.
     * Use `to_image_dir_sd_table_factory()` to dump the video as images
        to disk and create a `AdhocImagePathsSDTFactory` (see below).
     * Register the returned class to a PSegs `UnionFactory` as part of a
        larger collection of segments.
    """
  
    import imageio

    if segment_id is None:
      from urllib.parse import urlparse
      res = urlparse(str(video_uri))
      path = res.path
      fname = Path(path).name
      segment_id = fname

    r = imageio.get_reader(video_uri)
    n_frames = r.get_length()
    if n_frames == float('inf'):
      raise ValueError(
        "Don't currently support infinite streams: %s" % video_uri)
    if limit > 0:
      n_frames = limit
    
    fps = r.get_meta_data()['fps']
    h, w = load_video_frame(video_uri, 0).shape[:2]

    if isinstance(start_timestamp_use, six.string_types):
      res = Path(video_uri).lstat()
      start_time_sec = getattr(res, start_timestamp_use)
      start_time = int(1e9 * start_time_sec)
    else:
      start_time = int(start_timestamp_use)

    class MyAdhocVideosSDTFactory(cls):
      VIDEO_URI = str(video_uri)
      START_NANOSTAMP = start_time
      FRAMES_PER_SECOND = fps
      N_FRAMES = n_frames
      HEIGHT = h
      WIDTH = w

      DATASET = dataset
      SPLIT = split
      SEGMENT_ID = segment_id
      TOPIC = topic
    
    return MyAdhocVideosSDTFactory

  @classmethod
  def to_image_dir_sd_table_factory(cls, image_dest_dir):
    assert False, 'todo'

  @classmethod
  def get_segment_uri(cls):
    return datum.URI(
            dataset=cls.DATASET,
            split=cls.SPLIT,
            segment_id=cls.SEGMENT_ID)

  @classmethod
  def get_image_uris(cls):
    base_uri = cls.get_segment_uri()
    uris = []
    for i in range(cls.N_FRAMES):
      t = int(cls.START_NANOSTAMP + i * (1e9 / cls.FRAMES_PER_SECOND))
      uri = base_uri.replaced(topic=cls.TOPIC, timestamp=t)
      uri.extra['AdhocVideosSDTFactory.video_uri'] = str(cls.VIDEO_URI)
      uri.extra['AdhocVideosSDTFactory.frame_index'] = str(i)
      uris.append(uri)
    return uris

  @classmethod
  def create_stamped_datum(cls, uri):
    video_uri = uri.extra['AdhocVideosSDTFactory.video_uri']
    frame_index = int(uri.extra['AdhocVideosSDTFactory.frame_index'])
    ci = datum.CameraImage.create_world_frame_ci(
          sensor_name=uri.topic,
          width=cls.WIDTH,
          height=cls.HEIGHT,
          timestamp=uri.timestamp,
          image_factory=lambda: load_video_frame(video_uri, frame_index),
          extra=dict(uri.extra))

    return datum.StampedDatum(uri=uri, camera_image=ci)
  
  @classmethod
  def create_sd_table(cls, spark=None):
    with Spark.sess(spark) as spark:
      seg_uri = cls.get_segment_uri()
      sdt = cls.get_segment_sd_table(seg_uri, spark=spark)
      return sdt


  ## StampedDatumTableFactory Impl

  @classmethod
  def _get_all_segment_uris(cls):
    return [cls.get_segment_uri()]

  @classmethod
  def _create_datum_rdds(cls, spark, existing_uri_df=None, only_segments=None):
    if existing_uri_df is not None:
      util.log.info(
        f"Note: resume mode unsupported, got existing_uri_df {existing_uri_df}")
    
    if only_segments is not None:
      has_match = any(
              suri.soft_matches_segment_of(cls.get_segment_uri())
              for suri in only_segments)
      if not has_match:
        return []

    # Generate URIs ...
    uris = cls.get_image_uris()
    uri_rdd = spark.sparkContext.parallelize(uris, numSlices=len(uris))
    util.log.info(f"Creating datums for {len(uris)} images ...")

    datum_rdd = uri_rdd.map(cls.create_stamped_datum)

    return [datum_rdd]



###############################################################################
## StampedDatumTableFactory for an Image Collection

def load_image(path):
  # Defined at package-level for easier serialization
  import imageio
  return imageio.imread(path)


class AdhocImagePathsSDTFactory(StampedDatumTableFactory):
  """This `StampedDatumTableFactory` wraps a single collection of image paths
  and exposes them as a segment of PSegs `CameraImage`s.  See
  `create_factory_for_images()` below for quickstart.
  """

  IMAGE_PATHS = []
  IMAGE_TIMESTAMPS = None
    # Defaults to index in IMAGE_PATHS

  DATASET = 'anon'
  SPLIT = 'anon'
  SEGMENT_ID = 'anon_segment'
  TOPIC = 'camera_adhoc'

  @classmethod
  def create_factory_for_images(
        cls,
        images_dir=None,
        image_exts=('.jpg', '.png'),
        image_paths=None,
        dataset='anon',
        split='anon',
        topic='camera_adhoc',
        segment_id=None,
        timestamp_use='st_mtime',
        limit=-1):
    """Create and return a `StampedDatumTableFactory` class instance
    for the given set of images.  Provide either a directory of images
    `images_dir` and chosen file extensions (case-insensitive) `image_exts`
    OR a list of paths `image_paths`.

    If `timestamp_use` is not null, we will infer timestamps by stat-ing the
    files.

    Use this factory-factory function to produce a factory for your images,
    and then:
     * Use `create_sd_table()` to directly get a `StampedDatumTable` with your
        wrapped `CameraImage` instances.
     * Register the returned class to a PSegs `UnionFactory` as part of a
        larger collection of segments.
    """
  
    assert image_paths is not None or images_dir is not None

    if images_dir is not None:
      images_dir = Path(images_dir)
      image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in image_exts)
      if segment_id is None:
        segment_id = images_dir.name
    else:
      image_paths = [Path(p) for p in image_paths]
    
    if segment_id is None:
      if images_dir is not None:
        segment_id = images_dir.name
      else:
        assert len(image_paths) > 0
        segment_id = image_paths[0].parent.name

    if limit > 0:
      image_paths = image_paths[:limit]

    image_timestamps = None
    if timestamp_use is not None:
      def get_nanostamp(path):
        res = path.lstat()
        t_sec = getattr(res, timestamp_use)
        return int(t_sec * 1e9)

      image_timestamps = [
        get_nanostamp(p) for p in image_paths
      ]

    class MyAdhocImagePathsSDTFactory(cls):
      IMAGE_PATHS = image_paths
      IMAGE_TIMESTAMPS = image_timestamps
      DATASET = dataset
      SPLIT = split
      SEGMENT_ID = segment_id
      TOPIC = topic
    
    return MyAdhocImagePathsSDTFactory

  @classmethod
  def get_segment_uri(cls):
    return datum.URI(
            dataset=cls.DATASET,
            split=cls.SPLIT,
            segment_id=cls.SEGMENT_ID)

  @classmethod
  def get_image_uris(cls):
    base_uri = cls.get_segment_uri()
    uris = []
    for i, p in enumerate(cls.IMAGE_PATHS):
      if cls.IMAGE_TIMESTAMPS is None:
        t = i
      else:
        t = cls.IMAGE_TIMESTAMPS[i]

      uri = base_uri.replaced(topic=cls.TOPIC, timestamp=t)
      uri.extra['AdhocImagePathsSDTFactory.image_path'] = str(p)
      uris.append(uri)
    return uris

  @classmethod
  def create_stamped_datum(cls, uri):
    img_path = Path(uri.extra['AdhocImagePathsSDTFactory.image_path'])
    if img_path.suffix.lower == '.jpg':
      from oarphpy import util as oputil
      with open(rbg_path, 'rb') as f:
        w, h = oputil.get_jpeg_size(f.read(1024))
    elif img_path.suffix.lower == '.png':
      from psegs.util import misc
      with open(img_path, 'rb') as f:
        w, h = misc.get_png_wh(f.read(1024))
    else:
      import imageio
      w, h = imageio.imread(img_path).shape[:2]

    ci = datum.CameraImage.create_world_frame_ci(
          sensor_name=uri.topic,
          width=w,
          height=h,
          timestamp=uri.timestamp,
          image_factory=lambda: load_image(img_path),
          extra=dict(uri.extra))

    return datum.StampedDatum(uri=uri, camera_image=ci)
  
  @classmethod
  def create_sd_table(cls, spark=None):
    with Spark.sess(spark) as spark:
      seg_uri = cls.get_segment_uri()
      sdt = cls.get_segment_sd_table(seg_uri, spark=spark)
      return sdt


  ## StampedDatumTableFactory Impl

  @classmethod
  def _get_all_segment_uris(cls):
    return [cls.get_segment_uri()]

  @classmethod
  def _create_datum_rdds(cls, spark, existing_uri_df=None, only_segments=None):
    if existing_uri_df is not None:
      util.log.info(
        f"Note: resume mode unsupported, got existing_uri_df {existing_uri_df}")
    
    if only_segments is not None:
      has_match = any(
              suri.soft_matches_segment_of(cls.get_segment_uri())
              for suri in only_segments)
      if not has_match:
        return []

    # Generate URIs ...
    uris = cls.get_image_uris()
    uri_rdd = spark.sparkContext.parallelize(uris, numSlices=len(uris))
    util.log.info(f"Creating datums for {len(uris)} images ...")

    datum_rdd = uri_rdd.map(cls.create_stamped_datum)

    return [datum_rdd]
