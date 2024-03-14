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

import copy
from pathlib import Path

import six

from psegs import datum
from psegs import util
from psegs.spark import Spark
from psegs.cache import AssetDiskCache
from psegs.util.misc import get_image_wh
from psegs.util.video import ffmpeg_explode
from psegs.util.video import VideoMeta
from psegs.util.video import VideoExplodeParams
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


# class DiskCachedFramesVideoSegmentFactory_load_image(object):
#   __slots__ = ['_path']
#   def __init__(self, path):
#     self._path = path
#   def __call__(self):
#     import imageio
#     return imageio.imread(self._path)
# def DiskCachedFramesVideoSegmentFactory_load_image(path):
#   import imageio
#   return imageio.imread(path)

class DiskCachedFramesVideoSegmentFactory(StampedDatumTableFactory):
  """This `StampedDatumTableFactory` TODO
  """

  ## Classloader / writer and persisted state

  # Base URI for all datums
  BASE_URI = None
  
  # Extracted video metadata
  VIDEO_METADATA = None

  # Cache pre-computed `DiskCachedFramesVideoSegmentFactory` here; avoids
  # having to re-read videos for metadata.  You probably want these
  # cached on the same disk (e.g. a local disk) that `IMAGE_CACHE_CLS` uses.
  CLS_CACHE_DIR = None

  # Cache the actual frame image paths after a call to `explode_frames()`
  EXPLODED_FRAME_PATHS = None

  EXPLODE_PARAMS = None

  IMAGE_CACHE_CLS = None

  AUTO_EXPLODE = True

  @classmethod
  def _maybe_load_F(cls, uri, cls_cache_dir=None):
    F_path = cls._cached_cls_path_for_uri(uri, cls_cache_dir=cls_cache_dir)
    if not F_path.exists():
      return None
    import cloudpickle
    with open(F_path, 'rb') as f:
      return cloudpickle.load(f)
  
  @classmethod
  def _save_F(cls, F):
    F_path = F._cached_cls_path_for_uri(
      F.BASE_URI, cls_cache_dir=F.CLS_CACHE_DIR)
    F_path.parent.mkdir(parents=True, exist_ok=True)
    import cloudpickle
    with open(F_path, 'wb') as f:
      cloudpickle.dump(F, f, protocol=3) # Support older python

  @classmethod
  def _needs_explode(cls):
    return not (
      cls.EXPLODED_FRAME_PATHS and
      all(Path(p).exists() for p in cls.EXPLODED_FRAME_PATHS)
    )
    
  ## User API / Factory-Factory API

  DEFAULT_BASE_URI = datum.URI(
                  dataset='anon',
                  split='anon',
                  # NB: leave segment_id blank to deduce it from video_uri
                  topic='video_camera')

  DEFAULT_EXPODE_PARAMS = VideoExplodeParams()

  @classmethod
  def create_factory_for_video(
        cls,
        video_uri,
        base_uri=None,
        explode_params=None,
        auto_explode=True,
        start_timestamp_lstat_attr='st_mtime',
        start_time_nanostamp=None,
        cls_cache_dir=None,
        img_cache_cls=None,
        force_recompute_cls=False,
        do_cache_factory=True):
    """Create and return a `StampedDatumTableFactory` class instance for the
    given `video_uri`.  Use a pre-computed and cached class instance if
    available (unless `force_recompute_cls`); save the results of this function
    call to the cache only if `do_cache_factory`.  We will read part of the
    video to assess image dimensions, length, etc.  Use `explode_frames()` to
    also decode the entire video and fill the frame caches.

    If `start_timestamp_use` is a string, we will infer timestamps by not
    just the frame index but by stat-ing the video and using this 
    stat attribute.

    If `start_timestamp_use` is an integer, we'll use that nanostamp
    offset instead.
    """
  
    cls_base_uri = base_uri or copy.deepcopy(cls.DEFAULT_BASE_URI)
    
    explode_params = explode_params or copy.deepcopy(cls.DEFAULT_EXPODE_PARAMS)
    topic_suffix = (
      f"|max_hw_{explode_params.max_hw}"
      f"|ext_{explode_params.image_file_extension}"
    )
    if not cls_base_uri.topic:
      cls_base_uri.topic = 'video_camera'
    cls_base_uri.topic = cls_base_uri.topic + topic_suffix    
    
    if not cls_base_uri.segment_id:
      cls_base_uri.segment_id = cls._default_segment_id_for_video_uri(video_uri)

    if not force_recompute_cls:
      F = cls._maybe_load_F(cls_base_uri, cls_cache_dir=cls_cache_dir)
      if F is not None:
        util.log.debug(
          f"Using cached {F.__name__} for {str(cls_base_uri)}")
        return F

    util.log.info(
      "DiskCachedFramesVideoSegmentFactory: Creating video meta for %s" % video_uri)
    video_meta = VideoMeta.create_for_video(
      video_uri, lstat_attr=start_timestamp_lstat_attr)
    if start_time_nanostamp is not None:
      video_meta.start_time_nanostamp = start_time_nanostamp
    
    img_cache_cls = img_cache_cls or cls.DEFAULT_IMAGE_CACHE_CLS

    class MyVideoSDTFactory(cls):
      BASE_URI = cls_base_uri
      VIDEO_METADATA = video_meta
      CLS_CACHE_DIR = cls_cache_dir
      EXPLODE_PARAMS = explode_params
      IMAGE_CACHE_CLS = img_cache_cls
      AUTO_EXPLODE = auto_explode

    if do_cache_factory:
      util.log.info(
        f"Saving cached {cls.__name__} for {str(cls_base_uri)}")
      cls._save_F(MyVideoSDTFactory)

    return MyVideoSDTFactory

  @classmethod
  def get_segment_uri(cls):
    # Add path if available, to help --list-and-exit
    suri = copy.deepcopy(cls.BASE_URI.to_segment_uri())
    if cls.VIDEO_METADATA is not None:
      video_uri = cls.VIDEO_METADATA.video_uri
      suri.extra[cls.__name__ + '.video_uri'] = str(video_uri)
    return suri
  

  DEFAULT_IMAGE_CACHE_CLS = AssetDiskCache

  @classmethod
  def explode_frames(
        cls,
        force_recompute=False,
        do_cache_factory=True,
        img_cache_now_time=None):
    
    if not (force_recompute or cls._needs_explode()):
      util.log.debug(
        f"Factory \n{str(cls.BASE_URI)}\n already has "
                    f"{len(cls.EXPLODED_FRAME_PATHS or [])} exploded frames.")
      return cls
      
    img_cache = cls.IMAGE_CACHE_CLS()
    cache_dirkey = str(
      Path('DiskCachedFramesVideoSegmentFactory_root') / 
      cls._uri_dirkey_for_uri(cls.BASE_URI)
    )
    dest_root = img_cache.new_dirpath(cache_dirkey, t=img_cache_now_time)

    util.log.info(
      f"Factory \n{str(cls.BASE_URI)}\n exploding frames to \n{dest_root} ...")
    exploded_frame_paths = ffmpeg_explode(
                              cls.EXPLODE_PARAMS,
                              cls.VIDEO_METADATA.video_uri,
                              dest_root)
    util.log.info("... explode complete!")

    class MyExplodedVideoSDTFactory(cls):
      EXPLODED_FRAME_PATHS = exploded_frame_paths
    
    if do_cache_factory:
      util.log.info(
        f"Saving updated cached {MyExplodedVideoSDTFactory.__name__}"
        f" for {str(MyExplodedVideoSDTFactory.BASE_URI)}")
      MyExplodedVideoSDTFactory._save_F(MyExplodedVideoSDTFactory)

    return MyExplodedVideoSDTFactory
    
  @classmethod
  def create_sd_table(cls, spark=None):
    """Create and return a `StampedDatumTable` for just this Factory's video"""
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
        f"Note: resume mode unsupported, got an existing_uri_df")
    
    if only_segments:
      has_match = any(
              suri.soft_matches_segment_of(cls.get_segment_uri())
              for suri in only_segments)
      if not has_match:
        return []

    if cls.AUTO_EXPLODE:
      if cls._needs_explode():
        util.log.info(
          f"Auto-exploding cached images for {cls.__name__}"
          f" for {str(cls.BASE_URI)} ...")
        EF = cls.explode_frames()
        util.log.info("... done auto-exploding!")     
        return EF._create_datum_rdds(
          spark,
          existing_uri_df=existing_uri_df,
          only_segments=only_segments)


    # Generate URIs ...
    uris = cls._get_image_uris()
    uri_rdd = spark.sparkContext.parallelize(uris)
    util.log.info(f"Creating datums for {len(uris)} frames ...")

    datum_rdd = uri_rdd.map(cls._create_stamped_datum)

    return [datum_rdd]


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
  def _uri_dirkey_for_uri(cls, uri):
    key = Path(uri.dataset) / uri.split / uri.segment_id / uri.topic
    return str(key)

  @classmethod
  def _cached_cls_path_for_uri(cls, uri, cls_cache_dir=None):
    cls_cache_dir = cls_cache_dir or cls.CLS_CACHE_DIR
    if not cls_cache_dir:
      from psegs.conf import C
      cls_cache_dir = C.DATA_ROOT / 'DiskCachedFramesVideoSegmentFactory_cache'
      
    cls_cache_dir = Path(cls_cache_dir)
    
    util.log.debug(f"{cls.__name__} using cls_cache_dir {cls_cache_dir} ...")
    cls_cached_path = (
      cls_cache_dir / 
      cls._uri_dirkey_for_uri(uri) /
      'DiskCachedFramesVideoSegmentFactory_cls.cpkl')
    
    return cls_cached_path
  
  @classmethod
  def _get_uri_extra(cls):
    import urllib.parse
    
    vm = cls.VIDEO_METADATA
    extra = dict(
      video_uri=urllib.parse.quote_plus(str(vm.video_uri)),
      start_time_nanostamp=vm.start_time_nanostamp,
      frames_per_second=vm.frames_per_second,
      n_frames=vm.n_frames,
      is_10bit_hdr=vm.is_10bit_hdr,
    )
    prefix = 'DiskCachedFramesVideoSegmentFactory.'
    return dict((prefix + k, str(v)) for k, v in extra.items())

  # @classmethod
  # def _ffmpeg_asplode_frames(
  #         cls,
  #         video_meta,
  #         dest_root,
  #         max_hw=-1,
  #         file_extension='png',
  #         jpeg_quality=2):
    
  #   import math
  #   from oarphpy import util as oputil
    
  #   video_path = Path(video_meta.video_uri).resolve()

  #   rescale_arg = ''
  #   if max_hw >= 0:
  #     rescale_arg = (
  #       f"-vf 'scale=if(gte(iw\,ih)\,min({max_hw}\,iw)\,-2):if(lt(iw\,ih)\,min({max_hw}\,ih)\,-2)' "
  #     )
  #   qscale_arg = ''
  #   if file_extension == 'jpg':
  #     qscale_arg = f" -qscale {jpeg_quality} "

  #   zfill = int(math.log10(video_meta.n_frames)) + 1

  #   FFMPEG_CMD = f"""
  #     cd {dest_root} && \
  #     ffmpeg \
  #       -y \
  #       -noautorotate \
  #       -vframes {video_meta.n_frames} \
  #       -i {video_path} \
  #       {rescale_arg} \
  #       -vsync 0 \
  #       {qscale_arg} \
  #         DiskCachedFramesVideoSegmentFactory_frame_%0{zfill}d.{file_extension}
  #   """
  #   oputil.run_cmd(FFMPEG_CMD)

  #   paths = sorted(
  #     Path(p)
  #     for p in oputil.all_files_recursive(
  #       dest_root, 
  #       pattern='DiskCachedFramesVideoSegmentFactory_frame_*'))
  #   return paths

  @classmethod
  def _get_image_uris(cls):
    base_uri = cls.BASE_URI
    vm = cls.VIDEO_METADATA
    uris = []
    frame_paths = cls.EXPLODED_FRAME_PATHS or []
    n_frames_exploded = len(frame_paths)
    for i, frame_path in enumerate(frame_paths):
      t = int(
        vm.start_time_nanostamp + i * (1e9 / float(vm.frames_per_second)))
      uri = base_uri.replaced(timestamp=t)
      uri.extra.update(cls._get_uri_extra())
      uri.extra['DiskCachedFramesVideoSegmentFactory.frame_path'] = str(frame_path)
      uri.extra['DiskCachedFramesVideoSegmentFactory.frame_index'] = str(i)
      uri.extra['DiskCachedFramesVideoSegmentFactory.n_frames_exploded'] = str(n_frames_exploded)
      uris.append(uri)
    return uris

  @classmethod
  def _create_stamped_datum(cls, uri):
    assert cls.EXPLODED_FRAME_PATHS, \
      (f"User must call explode_frames() before realizing a "
       f"StampedDatumTable, {cls} {cls.VIDEO_METADATA}")

    frame_idx = uri.extra['DiskCachedFramesVideoSegmentFactory.frame_index']
    frame_idx = int(frame_idx)
    n_frames_exploded = uri.extra['DiskCachedFramesVideoSegmentFactory.n_frames_exploded']
    n_frames_exploded = int(n_frames_exploded)

    frame_path = uri.extra['DiskCachedFramesVideoSegmentFactory.frame_path']

    assert Path(frame_path).exists(), frame_path
    extra = dict(uri.extra)
    extra['DiskCachedFramesVideoSegmentFactory.frame_path'] = str(frame_path)

    # image_factory = (
    #   lambda: DiskCachedFramesVideoSegmentFactory_load_image(frame_path))
    # image_factory = DiskCachedFramesVideoSegmentFactory_load_image(frame_path)

    def _load_image(path=None):
      import imageio
      return imageio.imread(path)
    image_factory = lambda: _load_image(path=frame_path)

    vm = cls.VIDEO_METADATA
    # w, h = vm.width, vm.height
    # if cls.EXPLODE_PARAMS.max_hw >= 0:
    #   if w > cls.EXPLODE_PARAMS.max_hw or h > cls.EXPLODE_PARAMS.max_hw:
    w, h = get_image_wh(frame_path)
    
    # All we know for sure is the frame index, so include a refined estimated
    # frame timestamp and context used for that estimate.  This estimate could
    # be WRONG tho if there are lots of dropped frames and the video metadata
    # is actually correct.
    extra['DiskCachedFramesVideoSegmentFactory.VideoMeta.start_time_nanostamp'] = str(vm.start_time_nanostamp)
    extra['DiskCachedFramesVideoSegmentFactory.VideoMeta.n_frames'] = str(vm.n_frames)
    extra['DiskCachedFramesVideoSegmentFactory.VideoMeta.frames_per_second'] = str(vm.frames_per_second)
    extra['DiskCachedFramesVideoSegmentFactory.VideoMeta.end_time_nanostamp'] = str(vm.end_time_nanostamp)

    duration_ns = vm.end_time_nanostamp - vm.start_time_nanostamp
    ns_per_frame = duration_ns / float(n_frames_exploded)
    estimated_frame_nanostamp = frame_idx * ns_per_frame
    extra['DiskCachedFramesVideoSegmentFactory.estimated_frame_nanostamp'] = str(estimated_frame_nanostamp)

    ci = datum.CameraImage.create_world_frame_ci(
          sensor_name=uri.topic,
          width=w,
          height=h,
          timestamp=uri.timestamp,
          image_factory=image_factory,
          extra=extra)

    return datum.StampedDatum(uri=uri, camera_image=ci)

  



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

  # For smaller datasets, use a smaller number of partitions based upon the
  # size of `IMAGE_PATHS`
  SPARK_AUTO_REPARTITION = True

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
    w, h = get_image_wh(img_path)
    
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
    num_slices = None
    if cls.SPARK_AUTO_REPARTITION:
      from oarphpy.spark import cluster_cpu_count
      import math
      n_cpus = cluster_cpu_count(spark)
      if len(uris) < n_cpus:
        num_slices = max(1, int(math.log(len(uris) or 1)))

    # ... now create RDD
    uri_rdd = spark.sparkContext.parallelize(uris, numSlices=num_slices)
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
