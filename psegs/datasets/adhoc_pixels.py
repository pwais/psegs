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

from pathlib import Path

import six
import attr

from psegs import datum
from psegs import util
from psegs.spark import Spark
from psegs.cache import AssetDiskCache
from psegs.util.video import VideoMeta
from psegs.table.sd_table_factory import StampedDatumTableFactory


###############################################################################
## StampedDatumTableFactory for a Video


def read_frame(video_uri, frame_idx):
  import imageio
  r = imageio.get_reader(video_uri)
  return r.get_data(frame_idx)


def lazy_load_cached_frame(video_uri, frame_idx, cache_path):
  import imageio
  from pathlib import Path
  cache_path = Path(cache_path)
  if not cache_path.exists():
    im = read_frame(video_uri, frame_idx)
    imageio.imwrite(cache_path, im, compress_level=1) # Fastest
  return imageio.imread(cache_path)


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

  ASSET_CACHE_DIR = None

  @classmethod
  def create_factory_for_video(
        cls,
        video_uri,
        dataset='anon',
        split='anon',
        topic='camera_video_adhoc',
        segment_id=None,
        start_timestamp_use='st_mtime',
        asset_cache_dir='',
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
     * Register the returned class to a PSegs `UnionFactory` as part of a
        larger collection of segments.
      


    TODO:
    for f in `ls /outer_root/media/mai-tank/vids_to_sfm_temp/` ; do echo $f ; \
      mkdir -p /outer_root/media/970evo_2/vids_to_sfm_temp_expanded2/${f}_expanded ; \
        cd /outer_root/media/970evo_2/vids_to_sfm_temp_expanded2/${f}_expanded && \
          ffmpeg -i /outer_root/media/mai-tank/vids_to_sfm_temp/${f} -qscale:v 2 -framerate 5 -vf scale=-1:1024 ffmpeg_extracted_${f}_1024_%9d.jpg
          cd - ; \
       done


    """
  
    import imageio

    F = cls.maybe_load_factory(asset_cache_dir=asset_cache_dir)
    if F is not None:
      return F

    if segment_id is None:
      from urllib.parse import urlparse
      res = urlparse(str(video_uri))
      path = res.path
      fname = Path(path).name
      segment_id = fname

    r = imageio.get_reader(video_uri)
    n_frames = r.get_meta_data()['nframes']
    if n_frames == float('inf'):
      # For some python / imageio versions, you have to use this API:
      n_frames = r.count_frames()
      if n_frames == float('inf'):
        raise ValueError(
          "Don't currently support infinite streams: %s %s" % (
            r.get_meta_data(), video_uri))
    if limit > 0:
      n_frames = limit
    
    fps = r.get_meta_data()['fps']
    h, w = r.get_data(0).shape[:2]

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

      ASSET_CACHE_DIR = Path(asset_cache_dir) if asset_cache_dir else None
    
    if asset_cache_dir:
      MyAdhocVideosSDTFactory.maybe_save_factory()

    return MyAdhocVideosSDTFactory


  @classmethod
  def create_factories_for_videos(
        cls,
        root_search_dir,
        video_extensions=('.mov', '.mp4'),
        spark=None,
        **create_factory_kwargs):

    from oarphpy.util.misc import is_stupid_mac_file

    root_search_dir = Path(root_search_dir)
    video_paths = [
      p
      for p in root_search_dir.rglob('**/*')
      if (
        not p.is_dir() and
        not is_stupid_mac_file(p) and
        any(p.name.lower().endswith(ext) for ext in video_extensions)
      )
    ]

    with Spark.sess(spark) as spark:
      util.log.info(
        f"AdhocVideosSDTFactory: Creating factories for "
        f"{len(video_paths)} videos ...")
      path_rdd = spark.sparkContext.parallelize(
                        video_paths, numSlices=len(video_paths))
      def create_factory(p):
        return cls.create_factory_for_video(p, **create_factory_kwargs)
      factory_rdd = path_rdd.map(create_factory)

      sdt_factories = factory_rdd.collect()
      util.log.info("... done.")
    return sdt_factories

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
    
    if cls.ASSET_CACHE_DIR is None:
      image_factory = lambda: read_frame(video_uri, frame_index)
    else:
      frame_fname = '.'.join([uri.topic, str(uri.timestamp)])
      frame_fname = frame_fname + '.png'
      frames_dir = cls.ASSET_CACHE_DIR / 'frames'
      if not frames_dir.exists():
        frames_dir.mkdir(parents=True, exist_ok=True)
      lazy_path = frames_dir / frame_fname
      image_factory = lambda: lazy_load_cached_frame(
                                  video_uri,
                                  frame_index,
                                  lazy_path)
    
    ci = datum.CameraImage.create_world_frame_ci(
          sensor_name=uri.topic,
          width=cls.WIDTH,
          height=cls.HEIGHT,
          timestamp=uri.timestamp,
          image_factory=image_factory,
          extra=dict(uri.extra))

    return datum.StampedDatum(uri=uri, camera_image=ci)
  
  @classmethod
  def create_sd_table(cls, spark=None):
    with Spark.sess(spark) as spark:
      seg_uri = cls.get_segment_uri()
      sdt = cls.get_segment_sd_table(seg_uri, spark=spark)
      return sdt

  @classmethod
  def maybe_load_factory(cls, asset_cache_dir=''):
    if not asset_cache_dir:
      asset_cache_dir = cls.ASSET_CACHE_DIR
    
    if asset_cache_dir is None:
      return None

    asset_cache_dir = Path(asset_cache_dir)
    table_factory_path = asset_cache_dir / 'psegs_AdhocVideosSDTFactory_df.pkl'

    if not table_factory_path.exists():
      return None
    
    import cloudpickle
    with open(table_factory_path, 'rb') as f:
      return cloudpickle.load(f)
  
  @classmethod
  def maybe_save_factory(cls):
    if cls.ASSET_CACHE_DIR is None:
      return False
    
    cls.ASSET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    table_factory_path = (
      cls.ASSET_CACHE_DIR / 'psegs_AdhocVideosSDTFactory_df.pkl')
    
    import cloudpickle
    with open(table_factory_path, 'wb') as f:
      cloudpickle.dump(cls, f, protocol=3) # Support older python
    return True

  @classmethod
  def maybe_build_cache(cls, spark=None):
    if cls.ASSET_CACHE_DIR is None:
      return False

    util.log.info(f"Building cache for {cls.__name__} ...")
    cls.maybe_save_factory()
    with Spark.sess(spark) as spark:
      sdt = cls.create_sd_table(spark=spark)
      
      datum_rdd = sdt.to_datum_rdd(spark=spark)

      # Try to favor longer-lived python processes
      from oarphpy.spark import cluster_cpu_count
      n_cpus = min(cluster_cpu_count(spark), 8)
      util.log.info('fixme too many cpus doesnt work with ffmpeg')
      datum_rdd = datum_rdd.repartition(n_cpus)

      util.log.info(
        f"... exploding video {cls.VIDEO_URI} into {cls.N_FRAMES} images ...")
      def get_num_pixels(sd):
        if sd.camera_image:
          im = sd.camera_image.image
          h, w = im.shape[:2]
          return h * w
        else:
          return 0
      total_pixels = datum_rdd.map(get_num_pixels).sum()
      util.log.info(
        f"... exploded {1e-9 * total_pixels} total Gigapixels.")

  ## StampedDatumTableFactory Impl

  @classmethod
  def _get_all_segment_uris(cls):
    return [cls.get_segment_uri()]

  @classmethod
  def _create_datum_rdds(cls, spark, existing_uri_df=None, only_segments=None):
    if existing_uri_df is not None:
      util.log.info(
        f"Note: resume mode unsupported, got existing_uri_df {existing_uri_df}")
    
    if only_segments:
      has_match = any(
              suri.soft_matches_segment_of(cls.get_segment_uri())
              for suri in only_segments)
      if not has_match:
        return []

    # Generate URIs ...
    uris = cls.get_image_uris()
    uri_rdd = spark.sparkContext.parallelize(uris)
    util.log.info(f"Creating datums for {len(uris)} frames ...")

    datum_rdd = uri_rdd.map(cls.create_stamped_datum)

    return [datum_rdd]





class DiskCachedFramesVideoSegmentFactory(StampedDatumTableFactory):
  """This `StampedDatumTableFactory` TODO




  """

  # VIDEO_LOCATION = None
  # START_NANOSTAMP = 0
  # FRAMES_PER_SECOND = 1
  # N_FRAMES = 0
  # HEIGHT = 0
  # WIDTH = 0

  # DATASET = 'anon'
  # SPLIT = 'anon'
  # SEGMENT_ID = 'anon_segment'
  # TOPIC = 'camera_video_adhoc'

  # ASSET_CACHE_DIR = None


  # @classmethod
  # def _ffmpeg_asplode_lowfi_frames(cls, dest_root, max_hw=1080, jpeg_quality=2):
  #   video_path = Path(cls.VIDEO_LOCATION).resolve()

  #   from oarphpy import util as oputil

  #   FFMPEG_CMD = f"""
  #     cd {dest_root} && \
  #     ffmpeg \
  #       -y \
  #       -noautorotate \
  #       -i {video_path} \
  #       -vf 'scale=if(gte(iw\,ih)\,min({max_hw}\,iw)\,-2):if(lt(iw\,ih)\,min({max_hw}\,ih)\,-2)' \
  #       -vsync 0 \
  #       -qscale {jpeg_quality} \
  #         frame_%09d.jpg
  #   """
  #   oputil.run_cmd(FFMPEG_CMD)

  #   paths = sorted(
  #     Path(p)
  #     for p in oputil.all_files_recursive(dest_root, pattern='frame_*'))
  #   return paths


  ## Classloader / writer

  @classmethod
  def _maybe_load_F(cls, seg_uri):
    F_path = cls._cached_cls_path_for_seg_uri(seg_uri)
    if not F_path.exists():
      return None
    
    import cloudpickle
    with open(F_path, 'rb') as f:
      return cloudpickle.load(f)
  
  @classmethod
  def _save_F(cls, F, seg_uri):
    F_path = cls._cached_cls_path_for_seg_uri(seg_uri)
    F_path.parent.mkdir(parents=True, exist_ok=True)
   
    import cloudpickle
    with open(F_path, 'wb') as f:
      cloudpickle.dump(F, f, protocol=3) # Support older python


  ## Utils

  @classmethod
  def _default_segment_id_for_video_uri(cls, video_uri):
    from urllib.parse import urlparse
    import hashlib
  
    res = urlparse(str(video_uri))
    path = res.path
    fname = Path(path).name

    uri_hash = hashlib.md5(str(video_uri).encode('utf-8')).hexdigest()

    return f'{fname}_{uri_hash[:10]}'

  @classmethod
  def _cached_cls_path_for_seg_uri(cls, seg_uri):
    cls_cache_dir = cls.CLS_CACHE_DIR
    if not cls_cache_dir:
      from psegs.conf import C
      cls_cache_dir = C.DATA_ROOT / 'DiskCachedFramesVideoSegmentFactory_cache'
      
    cls_cache_dir = Path(cls_cache_dir)
    
    cls_cached_path = (
      cls_cache_dir / 
      seg_uri.dataset / 
      seg_uri.split / 
      seg_uri.segment_id / 
      'DiskCachedFramesVideoSegmentFactory_cls.cpkl')
    
    return cls_cached_path

  @classmethod
  def _get_segment_uri(cls, base_uri, video_uri):
    seg_uri = base_uri.to_segment_uri()
    if not seg_uri.segment_id:
      seg_uri.segment_id = cls._default_segment_id_for_video_uri(video_uri)
    return seg_uri


  ## User API

  # Cache pre-computed `DiskCachedFramesVideoSegmentFactory` here; avoids
  # having to re-read videos for metadata.
  CLS_CACHE_DIR = None

  IMAGE_CACHE_CLS = AssetDiskCache

  BASE_URI = datum.URI(
                  dataset='anon',
                  split='anon',
                  # NB: leave segment_id blank to deduce it from video_uri
                  topic='video_camera')

  @classmethod
  def create_for_video(
        cls,
        video_uri,
        base_uri=None,
        start_timestamp_lstat_attr='st_mtime',
        start_time_nanostamp=None,
        max_n_frames=-1,
        force_recompute=False,
        do_cache_factory=True):
    """Create and return a `StampedDatumTableFactory` class instance for the
    given `video_uri`.  Use a pre-computed and cached class instance if
    available (unless `force_recompute`); save the results of this function call
    to the cache only if `do_cache_factory`.  We will read part of the video to
    assess image dimensions, length, etc.  Use `explode_frames()` to also
    decode the entire video and fill the frame caches.

    If `start_timestamp_use` is a string, we will infer timestamps by not
    just the frame index but by stat-ing the video and using this 
    stat attribute.

    If `start_timestamp_use` is an integer, we'll use that nanostamp
    offset instead.

    """
  
    base_uri = base_uri or cls.BASE_URI
    seg_uri = cls._get_segment_uri(base_uri, video_uri)

    if not force_recompute:
      F = cls._maybe_load_F(seg_uri)
      if F is not None:
        return F

    video_meta = VideoMeta.create_for_video(
      video_uri, lstat_attr=start_timestamp_lstat_attr)
    if max_n_frames >= 0:
      video_meta.n_frames = max_n_frames
    if start_time_nanostamp is not None:
      video_meta.start_time_nanostamp = start_time_nanostamp
    
    

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

      ASSET_CACHE_DIR = Path(asset_cache_dir) if asset_cache_dir else None
    
    if asset_cache_dir:
      MyAdhocVideosSDTFactory.maybe_save_factory()

    return MyAdhocVideosSDTFactory

  ## StampedDatumTableFactory Impl

  @classmethod
  def _get_all_segment_uris(cls):
    return [cls.get_segment_uri()]

  @classmethod
  def _create_datum_rdds(cls, spark, existing_uri_df=None, only_segments=None):
    if existing_uri_df is not None:
      util.log.info(
        f"Note: resume mode unsupported, got an existing_uri_df")
    
    if only_segments:
      has_match = any(
              suri.soft_matches_segment_of(cls.get_segment_uri())
              for suri in only_segments)
      if not has_match:
        return []

    # Generate URIs ...
    uris = cls.get_image_uris()
    uri_rdd = spark.sparkContext.parallelize(uris)
    util.log.info(f"Creating datums for {len(uris)} frames ...")

    datum_rdd = uri_rdd.map(cls.create_stamped_datum)

    return [datum_rdd]



###############################################################################
## StampedDatumTableFactory for an Image Collection

def load_image(path):
  # Defined at package-level for easier serialization
  import imageio
  return imageio.imread(path)


def video_to_pngs(video_uri, out_dir, preserve_mtime=True):
  import math
  import imageio
  from tqdm import tqdm

  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  r = imageio.get_reader(video_uri)
  n_frames = r.count_frames()
  n_zfill = int(math.log10(n_frames)) + 1

  for i in tqdm(range(n_frames)):
    im = r.get_data(i)
    dest = out_dir / (f'frame_%s.png' % str(i).zfill(n_zfill))
    imageio.imwrite(dest, im, compress_level=1) # Fastest


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
    files and using the given stat attribute.  We may induce "fake" timestamps
    (off by a few nanoseconds) if many files have the same timestamp.  
    Alternatively, you can provide a list of nanostamps `timestamp_use` to use.

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
    if isinstance(timestamp_use, six.string_types):
      def get_nanostamp(path):
        res = path.lstat()
        t_sec = getattr(res, timestamp_use)
        return int(t_sec * 1e9)

      image_timestamps = [
        get_nanostamp(p) for p in image_paths
      ]

      # Ensure we don't induce any timestamp collisions.  All images are
      # distinct, so they should have distinct timestamps.  If the mitigation
      # below 
      t_to_count = {}
      distinct_t = []
      for t in image_timestamps:
        if t not in t_to_count:
          distinct_t.append(t)
          t_to_count[t] = 1
        else:
          distinct_t.append(t + t_to_count[t])
            # Add a few nanos to make distinct
          t_to_count[t] += 1
      image_timestamps = distinct_t

    elif timestamp_use is not None:
      image_timestamps = list(timestamp_use)

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
        t = i + 1
      else:
        t = cls.IMAGE_TIMESTAMPS[i]

      uri = base_uri.replaced(topic=cls.TOPIC, timestamp=t)
      uri.extra['AdhocImagePathsSDTFactory.image_path'] = str(p)
      uris.append(uri)
    return uris

  @classmethod
  def create_stamped_datum(cls, uri):
    img_path = Path(uri.extra['AdhocImagePathsSDTFactory.image_path'])
    if img_path.suffix.lower() in ('.jpg', '.jpeg'):
      from oarphpy import util as oputil
      try:
        with open(img_path, 'rb') as f:
          w, h = oputil.get_jpeg_size(f.read(1024))
      except ValueError:
        # Read entire image as fallback (slow)
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = 1e12

        import imageio
        w, h = imageio.imread(img_path).shape[:2]

    elif img_path.suffix.lower() == '.png':
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
    
    if only_segments:
      has_match = any(
              suri.soft_matches_segment_of(cls.get_segment_uri())
              for suri in only_segments)
      if not has_match:
        return []

    # Generate URIs ...
    uris = cls.get_image_uris()
    uri_rdd = spark.sparkContext.parallelize(uris)
    util.log.info(f"Creating datums for {len(uris)} images ...")

    datum_rdd = uri_rdd.map(cls.create_stamped_datum)
    return [datum_rdd]


###############################################################################
## Adhoc Directory Tree -> UnionFactory of Adhoc{ImagePaths,Video}SDTFactories

"""
video file -> segment
directory of images -> assume want sub-segment

future: directory of images for dedicated segment (e.g. camera after iphone)

"""



if __name__ == '__main__':
  F = AdhocVideosSDTFactory.create_factory_for_video(
        '/outer_root/media/970-evo-plus-raid0/iphone_vids_to_sfm/lidar_hero10/image_capture_continuous/GX010018.MP4',
        asset_cache_dir='/outer_root/media/970-evo-plus-raid0/iphone_vids_to_sfm/vids_to_sfm/lidar_hero10_winter_stinsin_GX010018.MP4_cache')
  F.maybe_build_cache()

  # video_to_pngs(
  #   '/outer_root/media/970-evo-plus-raid0/iphone_vids_to_sfm/vids_to_sfm/dubs-gym-bluetiful-subie-lidar-comparison.MOV',
  #   '/outer_root/media/970-evo-plus-raid0/hloc_out/anon.anon.dubs-gym-bluetiful-subie-lidar-comparison.MOV/images/')
