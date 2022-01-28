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



# This module helps convert COLMAP reconstructions to PSegs segments.
# pycolmap is a soft dependency for this module
# To enable: `pip3 install pycolmap>=0.1.0``
try:
  import pycolmap
except ImportError:
  util.log.error("This module requires pycolmap>=0.1.0")


import copy
import json
from pathlib import Path

import numpy as np

from psegs import datum
from psegs import util
from psegs.table.sd_table import StampedDatumTable
from psegs.table.sd_table_factory import StampedDatumTableFactory



def colmap_recon_create_world_cloud(
        recon_dir,
        sensor_name='colmap_sparse'):
  """Given a COLMAP reconstruction assets directory `recon_dir`, extract the
  sparse model 3D point cloud as a single `PointCloud` in the world frame.
  """

  recon = pycolmap.Reconstruction(recon_dir)
  ptid_to_info = recon.points3D

  xyzrgbErrViz = np.zeros((len(ptid_to_info), 8), dtype='float')
  for i, (ptid, info) in enumerate(sorted(ptid_to_info.items())):
    xyzrgbErrViz[i, :3] = info.xyz
    xyzrgbErrViz[i, 3:6] = info.color
    xyzrgbErrViz[i, 6] = info.error
    xyzrgbErrViz[i, 7] = info.track.length()
  
  # The points are in the world frame; provide identity transform(s)
  pc = datum.PointCloud.create_world_frame_cloud(
          sensor_name=sensor_name,
          cloud=xyzrgbErrViz,
          cloud_colnames=[
            'x', 'y', 'z',
            'r', 'g', 'b',
            'colmap_err', 'num_views_visible',
          ])
  return pc


def colmap_recon_create_camera_image(
            image_name,
            recon_dir,
            src_images_dir,
            uri='',
            sensor_name='colmap_sparse',
            timestamp=0,
            create_depth_image=False):
  """Given a COLMAP reconstruction assets directory `recon_dir`, create
  and return a single `CameraImage` from the COLMAP-computed camera pose
  (and perhaps the visible COLMAP 3D keypoints only if `create_depth_image`).

  `image_name` must be the (file) name of the image in the COLMAP
  reconstruction, and `src_images_dir` is the input image directory (which
  COLMAP typically requires be called "images") given to COLMAP.
  """

  recon = pycolmap.Reconstruction(recon_dir)
  
  # Find the image record
  iinfo = None
  iid = -1
  for ciid, cinfo in recon.images.items():
    if cinfo.name == image_name:
      iinfo = cinfo
      iid = ciid
  assert iinfo is not None, f"Could not find {image_name} in {recon_dir}"

  cameras = recon.cameras
  camera = cameras[iinfo.camera_id]
  if len(camera.params) < 4:
    # Probably SIMPLE_PINHOLE
    # FMI https://github.com/colmap/colmap/blob/9f3a75ae9c72188244f2403eb085e51ecf4397a8/scripts/python/visualize_model.py#L88
    fx, cx, cy = camera.params[:4]  
    fy = fx
  else:
    fx, fy, cx, cy = camera.params[:4]
  
  K = np.array([
        [fx,  0, cx],
        [0,  fy, cy],
        [0,   0,  1],
  ])
  h = camera.height
  w = camera.width

  extra = {
    'colmap.image_id': str(iid),
    'colmap.camera_params_raw_json': json.dumps(list(camera.params)),
    'colmap.camera_model_name': camera.model_name,
  }

  R = iinfo.rotation_matrix()
  T = iinfo.tvec
  ego_to_sensor = datum.Transform(
            src_frame='ego',
            dest_frame=sensor_name)
  ego_pose = datum.Transform(
                rotation=R,
                translation=T,
                src_frame='world',
                dest_frame='ego')
                  # COLMAP provides world-to-camera transforms

  if create_depth_image:
    
    ptid_to_info = recon.points3D
    p2ds = iinfo.get_valid_points2D()
    xyz_world = np.array(
      [ptid_to_info[p2d.point3D_id].xyz for p2d in p2ds]
    )
    z = (iinfo.rotation_matrix() @ xyz_world.T)[-1] + iinfo.tvec[-1]
    uv = np.array([p2d.xy for p2d in p2ds])
    errors = np.array(
      [ptid_to_info[p2d.point3D_id].error for p2d in p2ds]
    )
    n_visible = np.array(
      [ptid_to_info[p2d.point3D_id].track.length() for p2d in p2ds]
    )
    
    dev = np.zeros((h, w, 3), dtype=np.float32)
    channel_names = ['depth', 'colmap_err', 'num_views_visible']
    uu, vv = uv[:, 0].astype(int), uv[:, 1].astype(int)
      # TODO: bilinear interpolation ? 
      # the triangulation is already pretty noisy tho

    dev[vv, uu, 0] = z
    dev[vv, uu, 1] = errors
    dev[vv, uu, 2] = n_visible

    image_factory = lambda: dev

    uri = copy.deepcopy(uri)
    if uri:
      uri.topic = uri.topic + '|depth'
    dci = datum.CameraImage(
              sensor_name=sensor_name,
              image_factory=image_factory,
              channel_names=channel_names,
              height=h,
              width=w,
              timestamp=timestamp,
              ego_pose=ego_pose,
              ego_to_sensor=ego_to_sensor,
              K=K,
              extra=extra)
    return dci
  
  else:

    image_path = src_images_dir / iinfo.name
    assert image_path.exists(), image_path

    def _load_image(path):
      import imageio
      return imageio.imread(path)
    image_factory = lambda: _load_image(image_path)
    channel_names = ['r', 'g', 'b']

    ci = datum.CameraImage(
                sensor_name=sensor_name,
                image_factory=image_factory,
                channel_names=channel_names,
                height=h,
                width=w,
                timestamp=timestamp,
                ego_pose=ego_pose,
                ego_to_sensor=ego_to_sensor,
                K=K,
                extra=extra)
    return ci



def load_array(path):
  """Listed as a package-level function to improve clarity / portability."""
  import numpy as np
  return np.load(path)


class COLMAP_SDTFactory(StampedDatumTableFactory):
  """This `StampedDatumTableFactory` helps convert a single COLMAP 
  reconstruction into a single `StampedDatumTable` one-to-one.  While
  most `StampedDatumTableFactory` classes help transform multiple segments,
  This factory is agnostic to how the user stores multile COLMAP scene
  reconstructions and simply helps map between single scenes.

  For simple use with single scenes:
   * If your input is a PSegs segment, use
     `create_input_images_and_psegs_assets_for_colmap()` to prepare your
     data for input to COLMAP.
   # Run COLMAP as desired.
   * Use `create_sd_table_for_reconstruction()` to read back the COLMAP
     reconstruction as a PSegs segment.

  To use en masse (e.g. multiple scenes and multiple factories):
   * Subclass and configure the `*_DIR` members below for a *single*
     COLMAP reconstruction.  If your input is a PSegs segment, you should run
     `create_imgpath_to_uri_and_images()` before running COLMAP for each scene
     to save a PSegs uri <-> COLMAP image name mapping on disk.
  """

  COLMAP_RECON_DIR = Path('my_colmap_sparse')
  COLMAP_IMAGES_DIR = Path('my_colmap_images')
  PSEGS_ASSETS_DIR = Path('my_psegs_assets')

  INCLUDE_DEPTH_IMAGES = True
  INCLUDE_WORLD_CLOUD = True

  CI_RECON_TOPIC_SUFFIX = '|colmap_sparse'
  DCI_RECON_TOPIC_SUFFIX = '|depth'
  WORLD_CLOUD_TOPIC = 'fused_world_cloud|colmap_sparse'

  ## Support

  @classmethod
  def get_input_image_paths(cls):
    return sorted(
      pp for pp in cls.COLMAP_IMAGES_DIR.iterdir() if not pp.is_dir()
    )

  @classmethod
  def psegs_imgpath_to_uri_path(cls):
    return cls.PSEGS_ASSETS_DIR / 'psegs_imgpath_to_uri.json'

  @classmethod
  def psegs_npy_cache_dir(cls):
    return cls.PSEGS_ASSETS_DIR / 'npy_cached'

  @classmethod
  def create_imgpath_to_uri_and_images(cls, sd_table, only_topics=None):
    
    # Select the datums to export
    sdf = sd_table.to_spark_df()
    sdf = sdf.where("camera_image IS NOT NULL")
    if only_topics:
      sdf = sdf.where(sdf['uri.topic'].isin(only_topics))
    util.log.info(f"Selected {sdf.count()} input images ...")
    
    datum_rdd = sd_table.datum_df_to_datum_rdd(sdf)
    cls.COLMAP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    def save_image(stamped_datum):
      import imageio
      ci = stamped_datum.camera_image
      if not ci.has_rgb(): # TODO: can we support grey?
        return False
      fname = (
        str(stamped_datum.uri.topic) + "." + 
        str(stamped_datum.uri.timestamp) + ".png")
      dest = cls.COLMAP_IMAGES_DIR / fname
      image = ci.image 
      imageio.imsave(dest, image)
      return str(stamped_datum.uri), str(dest)
    
    uri_paths = datum_rdd.map(save_image).filter(lambda x: x).collect()
    util.log.info(f"... saved {len(uri_paths)} input images ...")
    data_path = cls.psegs_imgpath_to_uri_path()
    with open(data_path, 'w') as f:
      json.dump(uri_paths, f, indent=2)
    util.log.info(f"... saved PSegs uri<->image mapping to f{data_path}")
  
  @classmethod
  def get_imgpath_to_uri(cls):
    data_path = cls.psegs_imgpath_to_uri_path()
    if data_path.exists():
      with open(data_path) as f:
        uri_paths = json.load(f)
      return dict(
        (p, datum.URI.from_str(suri))
        for suri, p in sorted(uri_paths))
    else:
      # Create an anonymous segment if no prior PSegs data available
      base_uri = datum.URI(
                  dataset='anon',
                  split='anon',
                  segment_id='anon_colmap_recon')
      return dict(
        (path,
         base_uri.replaced(
           timestamp=int(1e9 * t),
           topic='camera|input'))
        for t, path in enumerate(cls.get_input_image_paths()))

  @classmethod
  def get_segment_uri(cls):
    """COLMAP Reconstructions act on only single segments"""
    p_to_uri = cls.get_imgpath_to_uri()
    if p_to_uri:
      for uri in p_to_uri.values():
        return uri.to_segment_uri()
    else:
      return None
  
  @classmethod
  def get_colmap_recon_uris(cls):
    if not cls.COLMAP_RECON_DIR.exists():
      return []

    imgpath_to_uri = cls.get_imgpath_to_uri()
    
    # Find registered images
    recon = pycolmap.Reconstruction(cls.COLMAP_RECON_DIR)
    registered_image_names = set(iinfo.name for iinfo in recon.images.values())

    all_uris = []
    for imgpath, input_uri in imgpath_to_uri.items():
      imgpath = Path(imgpath)
      if imgpath.name in registered_image_names:
        ci_uri = input_uri.replaced(
            topic=input_uri.topic + cls.CI_RECON_TOPIC_SUFFIX)
        ci_uri.extra['colmap.image_name'] = imgpath.name
        all_uris.append(ci_uri)
        if cls.INCLUDE_DEPTH_IMAGES:
          dci_uri = ci_uri.replaced(
            topic=ci_uri.topic + cls.DCI_RECON_TOPIC_SUFFIX)
          all_uris.append(dci_uri)

    if cls.INCLUDE_WORLD_CLOUD:
      seg_uri = cls.get_segment_uri()
      if seg_uri:
        all_uris.append(
          seg_uri.replaced(topic=cls.WORLD_CLOUD_TOPIC))
    
    return all_uris
  
  @classmethod
  def create_stamped_datum(cls, uri):
    if uri.topic == cls.WORLD_CLOUD_TOPIC:

      pc = colmap_recon_create_world_cloud(
            cls.COLMAP_RECON_DIR,
            sensor_name='colmap_sparse')
      return datum.StampedDatum(uri=uri, point_cloud=pc)
    
    elif uri.topic.endswith(cls.DCI_RECON_TOPIC_SUFFIX):

      ci = colmap_recon_create_camera_image(
              uri.extra['colmap.image_name'],
              cls.COLMAP_RECON_DIR,
              cls.COLMAP_IMAGES_DIR,
              uri=uri,
              sensor_name=uri.topic,
              timestamp=uri.timestamp,
              create_depth_image=True)
      return datum.StampedDatum(uri=uri, camera_image=ci)

    elif uri.topic.endswith(cls.CI_RECON_TOPIC_SUFFIX):

      dci = colmap_recon_create_camera_image(
              uri.extra['colmap.image_name'],
              cls.COLMAP_RECON_DIR,
              cls.COLMAP_IMAGES_DIR,
              uri=uri,
              sensor_name=uri.topic,
              timestamp=uri.timestamp,
              create_depth_image=False)
      return datum.StampedDatum(uri=uri, camera_image=dci)
    
    else:
      raise ValueError(f"Don't know what to do with {uri}")

  @classmethod
  def create_input_images_and_psegs_assets_for_colmap(
        cls,
        sd_table,
        colmap_input_images_dir,
        psegs_assets_dir):
    
    class MyCOLMAP_SDTFactory(cls):
      # COLMAP_RECON_DIR not needed
      COLMAP_IMAGES_DIR = Path(colmap_input_images_dir)
      PSEGS_ASSETS_DIR = Path(psegs_assets_dir)

    MyCOLMAP_SDTFactory.create_imgpath_to_uri_and_images(sd_table)

  @classmethod
  def get_reconstruction_sd_table(
        cls,
        use_np_assets=True,
        force_recompute_np_assets=False,
        spark=None):
    
    seg_uri = cls.get_segment_uri()
    if not seg_uri:
      return None
    
    sdt = cls.get_segment_sd_table(seg_uri, spark=spark)
    if use_np_assets:
      cls.psegs_npy_cache_dir().mkdir(parents=True, exist_ok=True)
      datum_rdd = sdt.to_datum_rdd(spark=spark)
      datum_rdd = datum_rdd.map(
                      lambda sd: 
                        cls.to_lazy_datum(
                          sd, force_recompute=force_recompute_np_assets))
      return StampedDatumTable.from_datum_rdd(spark, datum_rdd)
    else:
      return sdt


  @classmethod
  def create_sd_table_for_reconstruction(
        cls,
        spark,
        colmap_recon_dir,
        colmap_input_images_dir,
        psegs_assets_dir,
        use_np_assets=True,
        force_recompute_np_assets=False):
    
    class MyCOLMAP_SDTFactory(cls):
      COLMAP_RECON_DIR = Path(colmap_recon_dir)
      COLMAP_IMAGES_DIR = Path(colmap_input_images_dir)
      PSEGS_ASSETS_DIR = Path(psegs_assets_dir)
    
    return MyCOLMAP_SDTFactory.get_reconstruction_sd_table(
              use_np_assets=use_np_assets,
              force_recompute_np_assets=force_recompute_np_assets,
              spark=spark)


  @classmethod
  def to_lazy_datum(cls, sd, force_recompute=False):
    from oarphpy.spark import CloudpickeledCallable
        
    if sd.camera_image:
      ci = sd.camera_image
      if 'depth' not in ci.channel_names:
        return sd
      
      depth_npy_fname = f"{sd.uri.topic}_{sd.uri.timestamp}_depth.npy"
      depth_npy_path = cls.psegs_npy_cache_dir() / depth_npy_fname
      if force_recompute or (not depth_npy_path.exists()):
        with open(depth_npy_path, 'wb') as f:
          np.savez_compressed(f, image=ci.image)
            # Lots of zeros; compression helps
      ci.image_factory = CloudpickeledCallable(
        lambda: load_array(depth_npy_path)['image'])
      return sd
    
    elif sd.point_cloud:
      pc = sd.point_cloud

      cloud_npy_fname = f"{sd.uri.topic}_{sd.uri.timestamp}_cloud.npy"
      cloud_npy_path = cls.psegs_npy_cache_dir() / cloud_npy_fname
      if force_recompute or (not cloud_npy_path.exists()):
        with open(cloud_npy_path, 'wb') as f:
          np.save(f, pc.get_cloud()) # Compression doesn't help
        pc.cloud = None
      pc.cloud_factory = CloudpickeledCallable(
        lambda: load_array(cloud_npy_path))
      return sd
      

  ## StampedDatumTableFactory Impl

  @classmethod
  def _get_all_segment_uris(cls):
    suri = cls.get_segment_uri()
    if suri is not None:
      return [suri]
    else:
      return []

  @classmethod
  def _create_datum_rdds(cls, spark, existing_uri_df=None, only_segments=None):
    from oarphpy import util as oputil
    from psegs import util

    if existing_uri_df is not None:
      util.log.info(
        f"Note: resume mode unsupported, got existing_uri_df {existing_uri_df}")
    if only_segments is not None:
      util.log.info(
        f"Note: Segment filter unsupported, got only_segments {only_segments}")

    # Generate URIs ...
    colmap_uris = cls.get_colmap_recon_uris()
    uri_rdd = spark.sparkContext.parallelize(
                    colmap_uris, numSlices=len(colmap_uris))
    util.log.info(f"Creating datums for {len(colmap_uris)} URIs")

    datum_rdd = uri_rdd.map(cls.create_stamped_datum)

    return [datum_rdd]
