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

from psegs import util
try:
  import pycolmap
except ImportError:
  util.log.error("This module requires pycolmap>=0.1.0")


import copy
import json
import itertools
from pathlib import Path
from tqdm.auto import tqdm

import cv2
import numpy as np

from psegs import datum
from psegs.table.sd_table_factory import StampedDatumTableFactory


def _find_image_record(image_name, recon, err_msg=''):
  # pycolmap does not provide a map, so we need to do a linear find
  iinfo = None
  iid = -1
  for ciid, cinfo in recon.images.items():
    if cinfo.name == image_name:
      iinfo = cinfo
      iid = ciid
  if iinfo is None:
    err_msg = err_msg or f"Could not find image_name {image_name}"
    raise KeyError(err_msg)
  return iinfo, iid


def colmap_recon_create_world_cloud(
        recon_dir,
        sensor_name='colmap_sparse'):
  """Given a COLMAP reconstruction assets directory `recon_dir`, extract the
  sparse model 3D point cloud as a single `PointCloud` in the world frame.
  """

  def _get_cloud(recon_dir):

    recon = pycolmap.Reconstruction(recon_dir)
    ptid_to_info = recon.points3D

    xyzrgbErrViz = np.zeros((len(ptid_to_info), 8), dtype='float')
    for i, (ptid, info) in enumerate(sorted(ptid_to_info.items())):
      xyzrgbErrViz[i, :3] = info.xyz
      xyzrgbErrViz[i, 3:6] = info.color
      xyzrgbErrViz[i, 6] = info.error
      xyzrgbErrViz[i, 7] = info.track.length()
    return xyzrgbErrViz
  
  # The points are in the world frame; provide identity transform(s)
  pc = datum.PointCloud.create_world_frame_cloud(
          sensor_name=sensor_name,
          cloud_factory=lambda: _get_cloud(recon_dir),
          cloud_colnames=[
            'x', 'y', 'z',
            'r', 'g', 'b',
            'colmap_err', 'num_views_visible',
          ])
  return pc


def colmap_get_intrinsics(camera):
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
  
  camera_model = None
  if hasattr(camera, 'model'):
    camera_model = camera.model
  elif hasattr(camera, 'model_name'):
    # pycolmap >= 0.4.0
    camera_model = camera.model_name
  assert camera_model is not None

  distortion_kv = {}
  distortion_model = f'colmap_camera.model={camera_model}'
  if camera_model == 'OPENCV':
    distortion_model = 'OPENCV'
    distortion_kv = {
      'k1': float(camera.params[4]),
      'k2': float(camera.params[5]),
      'p1': float(camera.params[6]),
      'p2': float(camera.params[7]),
    }
  elif camera_model == 'FULL_OPENCV':
    distortion_model = 'FULL_OPENCV'
    distortion_kv = {
      'k1': float(camera.params[4]),
      'k2': float(camera.params[5]),
      'p1': float(camera.params[6]),
      'p2': float(camera.params[7]),
      'k3': float(camera.params[8]),
      'k4': float(camera.params[9]),
      'k5': float(camera.params[10]),
      'k6': float(camera.params[11]),
    }
  elif camera_model == 'OPENCV_FISHEYE':
    distortion_model = 'OPENCV_FISHEYE'
    distortion_kv = {
      'k1': float(camera.params[4]),
      'k2': float(camera.params[5]),
      'k3': float(camera.params[6]),
      'k4': float(camera.params[7]),
    }

  h = camera.height
  w = camera.width

  return K, h, w, camera_model, distortion_model, distortion_kv

def colmap_recon_create_camera_image(
            image_name,
            recon_dir,
            src_images_dir,
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
  iinfo, iid = _find_image_record(image_name, recon, err_msg=f"Could not find {image_name} in {recon_dir}")
  # iid = -1
  # for ciid, cinfo in recon.images.items():
  #   if cinfo.name == image_name:
  #     iinfo = cinfo
  #     iid = ciid
  # assert iinfo is not None, 

  cameras = recon.cameras
  camera = cameras[iinfo.camera_id]
  
  ret = colmap_get_intrinsics(camera)
  K, h, w, colmap_camera_model, distortion_model, distortion_kv = ret

  extra = {
    'colmap.image_id': str(iid),
    'colmap.image_name': image_name,
    'colmap.camera_params_raw_json': json.dumps(list(camera.params)),
    'colmap.camera_model_name': colmap_camera_model,
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

    # FIXME pycolmap `p2ds` segfaults in list comprehensions in python 3.10
    # xyz_world = []
    # errors = []
    # n_visible = []
    # uv = []
    # for i in range(len(p2ds)):
    #   p2d = p2ds[i]
    #   xyz_world.append(ptid_to_info[p2d.point3D_id].xyz)
    #   errors.append(ptid_to_info[p2d.point3D_id].error)
    #   n_visible.append(ptid_to_info[p2d.point3D_id].track.length())
    #   uv.append(p2d.xy)
    # xyz_world = np.array(xyz_world)
    # errors = np.array(errors)
    # n_visible = np.array(n_visible)
    # uv = np.array(uv)
    

    # for i in range(len(p2ds)):
    #   print(([p2d.point3D_id for p2d in p2ds[:i]], i))
    # breakpoint()
    # print([p2d.point3D_id for p2d in p2ds])
    xyz_world = np.array(
      [ptid_to_info[p2d.point3D_id].xyz for p2d in p2ds]
    )
    
    xyz_in_camera = (iinfo.rotation_matrix() @ xyz_world.T).T + iinfo.tvec
    dist = np.linalg.norm(xyz_in_camera, axis=-1)
    uv = np.array([p2d.xy for p2d in p2ds])
    errors = np.array(
      [ptid_to_info[p2d.point3D_id].error for p2d in p2ds]
    )
    n_visible = np.array(
      [ptid_to_info[p2d.point3D_id].track.length() for p2d in p2ds]
    )
        
    # Sometimes COLMAP includes points that are outside the image...
    # TODO where do these come from? should not be due to distortion
    idx = np.where(
        (uv[:, 0] >= 0) &
        (uv[:, 0] < w) &
        (uv[:, 1] >= 0) &
        (uv[:, 1] < h)
    )
    dist = dist[idx]
    errors = errors[idx]
    n_visible = n_visible[idx]
    uv = uv[idx]

    uu, vv = uv[:, 0].astype(int), uv[:, 1].astype(int)
      # TODO: bilinear interpolation ? 
      # the triangulation is already pretty noisy tho

    dev = np.zeros((h, w, 3), dtype=np.float32)
    dev[vv, uu, 0] = dist
    dev[vv, uu, 1] = errors
    dev[vv, uu, 2] = n_visible
    channel_names = ['depth', 'colmap_err', 'num_views_visible']

    image_factory = lambda: dev

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
              distortion_model=distortion_model,
              distortion_kv=distortion_kv,
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
                distortion_model=distortion_model,
                distortion_kv=distortion_kv,
                extra=extra)
    return ci


def colmap_get_image_name_to_covis_names(recon):
  """Create and return a dict of `image.name` -> all other `image.name`s with
  at least one co-visible 3D point (i.e. a matched point).  Returns an
  undirected graph; covisibility is bijective.
  """  
  
  points3d = recon.points3D
  image_id_to_name = dict(
    (image.image_id, image.name) for image in recon.images.values())
  image_name_to_covis_names = dict(
    (name, set())
    for name in image_id_to_name.values())
  
  iter_images = tqdm(recon.images.values(), desc="Collect covisible images")
  for image in iter_images:
    name = image.name
        
    # Get all tracked neighbors... this could be slow...
    for imp in image.points2D:
      if imp.has_point3D():
        ptid = imp.point3D_id
        p3d = points3d[ptid]
        for te in p3d.track.elements:
          track_image_id = te.image_id
          track_image_name = image_id_to_name[track_image_id]
          if track_image_name != name:
            
            # Ensure the covisibility graph is *undirected*
            image_name_to_covis_names[name].add(track_image_name)
            image_name_to_covis_names[track_image_name].add(name)

  # Reformat
  image_name_to_covis_names = dict(
    (name, sorted(neighbs))
    for name, neighbs in image_name_to_covis_names.items())
  return image_name_to_covis_names


def colmap_recon_create_matched_pair(
        image1_name,
        image2_name,
        recon_dir,
        matcher_name='colmap_sparse',
        timestamp=0,
        include_point3d_colors_uint8=True,
        include_point3d_world_xyz=True,
        include_point3d_extras=True,
        img1=None,
        img2=None,
        src_images_dir=None,
        camera_image_kwargs={}):
  """Given a COLMAP reconstruction assets directory `recon_dir`, extract the
  matched pair for given image names.  (Note that the image pair might not
  have any matches; see `colmap_get_image_name_to_covis_names()` to help
  restrict to only image pairs with covisible points).

  Optionally uses pre-filled `img1` and `img2`, else attempts to load them.
  """

  assert image1_name != image2_name, image2_name

  matches_colnames = [
          # Core required
          'x1', 'y1', 'x2', 'y2'
  ]
  if include_point3d_colors_uint8:
    matches_colnames += ['r', 'g', 'b']
  if include_point3d_world_xyz:
    matches_colnames += ['world_x', 'world_y', 'world_z']
  if include_point3d_extras:
    matches_colnames += ['error', 'track_length', 'colmap_p3id']

  def _get_matches(
          recon_dir, image1_name, image2_name,
          include_point3d_colors_uint8=True,
          include_point3d_world_xyz=True,
          include_point3d_extras=True):
    recon = pycolmap.Reconstruction(recon_dir)
    ii1nfo, iid1 = _find_image_record(image1_name, recon, 
                                err_msg=
                                  f"Could not find {image1_name} (_get_matches) in {recon_dir}")
    ii2nfo, iid2 = _find_image_record(image2_name, recon, 
                                err_msg=
                                  f"Could not find {image2_name} (_get_matches) in {recon_dir}")
    i1_p3id_to_p2d = dict(
      (p.point3D_id, p) for p in ii1nfo.points2D if p.has_point3D())
    i2_p3id_to_p2d = dict(
      (p.point3D_id, p) for p in ii2nfo.points2D if p.has_point3D())
    covis_p3ids = set(i1_p3id_to_p2d.keys()) & set(i2_p3id_to_p2d.keys())

    n_cols = (
      4 + (3 if include_point3d_colors_uint8 else 0) 
      + (3 if include_point3d_world_xyz else 0) 
      + (3 if include_point3d_extras else 0)
    )
    matches = np.zeros((len(covis_p3ids), n_cols), dtype='float')
    for i, p3id in enumerate(covis_p3ids):
      i1_p2d = i1_p3id_to_p2d[p3id]
      i2_p2d = i2_p3id_to_p2d[p3id]
      p3d = recon.points3D[p3id]
      c = 0

      x1 = i1_p2d.x
      y1 = i1_p2d.y
      x2 = i2_p2d.x
      y2 = i2_p2d.y
      matches[i, c:c+4] = x1, y1, x2, y2
      c += 4

      if include_point3d_colors_uint8:
        # Yes it's RGB with colors in [0, 255]
        # https://github.com/colmap/colmap/blob/a7b50e4d70888cb2c7e5a35fc44a6a1e1f82e69a/src/colmap/scene/point3d.h#L57
        r = p3d.color[0]
        g = p3d.color[1]
        b = p3d.color[2]
        matches[i, c:c+3] = r, g, b
        c+=3
      
      if include_point3d_world_xyz:
        wx = p3d.x
        wy = p3d.y
        wz = p3d.z
        matches[i, c:c+3] = wx, wy, wz
        c+=3

      if include_point3d_extras:
        error = p3d.error
        track_length = p3d.track.length()
        colmap_p3id = p3id
        matches[i, c:c+3] = error, track_length, colmap_p3id
        c+=3

    return matches

  extra = {
    'colmap.image1_name': image1_name,
    'colmap.image2_name': image2_name,
  }

  should_fill_images = (img1 is None and img2 is None)
  if should_fill_images:
    assert src_images_dir is not None, f"Programming error, need {src_images_dir}"

    img1 = colmap_recon_create_camera_image(
              image1_name,
              recon_dir,
              src_images_dir,
              **camera_image_kwargs)

    img2 = colmap_recon_create_camera_image(
              image2_name,
              recon_dir,
              src_images_dir,
              **camera_image_kwargs)

    extra['colmap.image1_id'] = img1.extra['colmap.image_id']
    extra['colmap.image2_id'] = img2.extra['colmap.image_id']
  else:
    recon = pycolmap.Reconstruction(recon_dir)
    ii1nfo, iid1 = _find_image_record(image1_name, recon, 
                      err_msg=
                        f"Could not find {image1_name} (cmp) in {recon_dir}")
    ii2nfo, iid2 = _find_image_record(image2_name, recon, 
                      err_msg=
                        f"Could not find {image2_name} (cmp) in {recon_dir}")
    extra['colmap.image1_id'] = str(iid1)
    extra['colmap.image2_id'] = str(iid2)

  extra.update({
    'colmap.image1_name': image1_name,
    'colmap.image2_name': image2_name,
  })

  mp = datum.MatchedPair(
        matcher_name=matcher_name,
        timestamp=timestamp,
        img1=img1,
        img2=img2,
        matches_factory=lambda: _get_matches(
          recon_dir, image1_name, image2_name,
          include_point3d_colors_uint8=include_point3d_colors_uint8,
          include_point3d_world_xyz=include_point3d_world_xyz,
          include_point3d_extras=include_point3d_extras),
        matches_colnames=matches_colnames,
        extra=extra,
  )
  return mp


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
  INCLUDE_MATCHED_PAIRS = True
  USE_NP_CACHED_ASSETS = True

  MP_INCLUDE_POINT3D_COLORS_UINT8 = True
  MP_INCLUDE_POINT3D_WORLD_XYS = True
  MP_INCLUDE_POINT3D_EXTRAS = True
  MP_INCLUDE_CAMERA_IMAGES = True

  CI_RECON_TOPIC_SUFFIX = '|colmap_sparse'
  DCI_RECON_TOPIC_SUFFIX = '|depth'
  WORLD_CLOUD_TOPIC = 'fused_world_cloud|colmap_sparse'
  MP_TOPIC_SUFFIX = '|matches'


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
  def create_imgpath_to_uri_and_images(
          cls,
          sd_table,
          only_topics=None,
          resize_image_max_height=-1,
          spark=None):
    
    # Select the datums to export
    datum_rdd = sd_table.get_datum_rdd_matching(
                    only_types=['camera_image'],
                    only_topics=only_topics)
    
    # Try to favor fewer, longer-lived python processes
    from oarphpy.spark import cluster_cpu_count
    from psegs.spark import Spark
    with Spark.sess(spark) as spark:
      n_cpus = cluster_cpu_count(spark)
    datum_rdd = datum_rdd.repartition(n_cpus).cache()

    util.log.info(f"Selected {datum_rdd.count()} input images ...")

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

      h, w = image.shape[:2]
      if (resize_image_max_height >= 0 and h > resize_image_max_height):
        scale = float(resize_image_max_height) / h
        th = int(scale * h)
        tw = int(scale * w)
        image = cv2.resize(image, (tw, th))

      imageio.imsave(dest, image)
      return str(stamped_datum.uri), str(dest)
    
    uri_paths = datum_rdd.map(save_image).filter(lambda x: x).collect()
    util.log.info(f"... saved {len(uri_paths)} input images ...")
    data_path = cls.psegs_imgpath_to_uri_path()
    with open(data_path, 'w') as f:
      json.dump(uri_paths, f, indent=2)
    util.log.info(f"... saved PSegs uri<->image mapping to {data_path}")
  
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
    _ci_uris = []
    for imgpath, input_uri in imgpath_to_uri.items():
      imgpath = Path(imgpath)
      if imgpath.name in registered_image_names:
        ci_uri = input_uri.replaced(
            topic=input_uri.topic + cls.CI_RECON_TOPIC_SUFFIX)
        ci_uri.extra['colmap.image_name'] = imgpath.name
        ci_uri.extra['colmap.input_uri'] = input_uri.to_urlsafe_str()
        all_uris.append(ci_uri)
        _ci_uris.append(ci_uri)
        if cls.INCLUDE_DEPTH_IMAGES:
          dci_uri = ci_uri.replaced(
            topic=ci_uri.topic + cls.DCI_RECON_TOPIC_SUFFIX)
          all_uris.append(dci_uri)

    if cls.INCLUDE_MATCHED_PAIRS:
      ci_image_name_to_uri = dict(
        (ci_uri.extra['colmap.image_name'], ci_uri) for ci_uri in _ci_uris)
      image_name_to_covis_names = colmap_get_image_name_to_covis_names(recon)
      # Build only one matched pair per distinct pair of images
      image_pair_names = set(itertools.chain.from_iterable(
        ((image_name1, in2) for in2 in image_name2s)
        for image_name1, image_name2s in image_name_to_covis_names.items()))
      for im1_name, im2_name in image_pair_names:
        ci1_uri = ci_image_name_to_uri[im1_name]
        ci2_uri = ci_image_name_to_uri[im2_name]

        c1_topic_base = ci1_uri.topic.replace(cls.CI_RECON_TOPIC_SUFFIX, '')
        c2_topic_base = ci2_uri.topic.replace(cls.CI_RECON_TOPIC_SUFFIX, '')
        mp_topic_base = '|'.join(sorted((c1_topic_base, c2_topic_base)))

        mp_uri = copy.deepcopy(ci1_uri)
        mp_uri = mp_uri.replaced(
            timestamp=int(0.5 * abs(ci1_uri.timestamp + ci2_uri.timestamp)),
            topic=(
              # E.g. 'camera-input|camera-input|colmap_sparse|matches'
              mp_topic_base + cls.CI_RECON_TOPIC_SUFFIX + cls.MP_TOPIC_SUFFIX
              ))
        mp_uri.extra['colmap.image1_name'] = im1_name
        mp_uri.extra['colmap.image2_name'] = im2_name
        mp_uri.extra['colmap.image1_uri'] = ci1_uri.to_urlsafe_str()
        mp_uri.extra['colmap.image2_uri'] = ci2_uri.to_urlsafe_str()

        all_uris.append(mp_uri)

    if cls.INCLUDE_WORLD_CLOUD:
      seg_uri = cls.get_segment_uri()
      if seg_uri:
        all_uris.append(
          seg_uri.replaced(topic=cls.WORLD_CLOUD_TOPIC))
    
    return all_uris
  
  @classmethod
  def _create_point_cloud(cls, uri):
    from oarphpy.spark import CloudpickeledCallable

    pc = colmap_recon_create_world_cloud(
            cls.COLMAP_RECON_DIR,
            sensor_name='colmap_sparse')

    if cls.USE_NP_CACHED_ASSETS:
      cloud_npy_fname = f"{uri.topic}_{uri.timestamp}_cloud.npy"
      cloud_npy_path = cls.psegs_npy_cache_dir() / cloud_npy_fname
      if not cloud_npy_path.exists():
        with open(cloud_npy_path, 'wb') as f:
          np.save(f, pc.get_cloud()) # Compression doesn't help
        pc.cloud = None
      pc.cloud_factory = CloudpickeledCallable(
        lambda: load_array(cloud_npy_path))
 
    return datum.StampedDatum(uri=uri, point_cloud=pc)

  @classmethod
  def _create_camera_image(cls, uri):
    ci = colmap_recon_create_camera_image(
              uri.extra['colmap.image_name'],
              cls.COLMAP_RECON_DIR,
              cls.COLMAP_IMAGES_DIR,
              sensor_name=uri.topic,
              timestamp=uri.timestamp,
              create_depth_image=False)
    return datum.StampedDatum(uri=uri, camera_image=ci)

  @classmethod
  def _create_depth_camera_image(cls, uri):
    from oarphpy.spark import CloudpickeledCallable

    dci = colmap_recon_create_camera_image(
              uri.extra['colmap.image_name'],
              cls.COLMAP_RECON_DIR,
              cls.COLMAP_IMAGES_DIR,
              sensor_name=uri.topic,
              timestamp=uri.timestamp,
              create_depth_image=True)

    ci_uri = copy.deepcopy(uri)
    ci_uri.topic = ci_uri.topic.replace(cls.DCI_RECON_TOPIC_SUFFIX, '')
    dci.extra['psegs.depth.rgb_uri'] = str(ci_uri)

    if cls.USE_NP_CACHED_ASSETS:
      # Make sure the filename is definitely distinct
      colmap_image_id = dci.extra['colmap.image_id']
      depth_npy_fname = (
        f"{uri.topic}_{uri.timestamp}_image_id.{colmap_image_id}_depth.npy")
      depth_npy_path = cls.psegs_npy_cache_dir() / depth_npy_fname
      if not depth_npy_path.exists():
        with open(depth_npy_path, 'wb') as f:
          np.savez_compressed(f, image=dci.image)
            # Lots of zeros; compression helps
      dci.image_factory = CloudpickeledCallable(
        lambda: load_array(depth_npy_path)['image'])
      
    return datum.StampedDatum(uri=uri, camera_image=dci)

  @classmethod
  def _create_matched_pair(cls, uri):
    from oarphpy.spark import CloudpickeledCallable

    img1, img2 = None, None
    if cls.MP_INCLUDE_CAMERA_IMAGES:
      ci1_uri = datum.URI.from_str(uri.extra['colmap.image1_uri'])
      sd1 = cls._create_camera_image(ci1_uri)
      img1 = sd1.camera_image

      ci2_uri = datum.URI.from_str(uri.extra['colmap.image2_uri'])
      sd2 = cls._create_camera_image(ci2_uri)
      img2 = sd2.camera_image

    mp = colmap_recon_create_matched_pair(
      uri.extra['colmap.image1_name'],
      uri.extra['colmap.image2_name'],
      cls.COLMAP_RECON_DIR,
      matcher_name=uri.topic,
      timestamp=uri.timestamp,
      include_point3d_colors_uint8=cls.MP_INCLUDE_POINT3D_COLORS_UINT8,
      include_point3d_world_xyz=cls.MP_INCLUDE_POINT3D_WORLD_XYS,
      include_point3d_extras=cls.MP_INCLUDE_POINT3D_EXTRAS,
      img1=img1,
      img2=img2,
    )
    mp.extra['colmap.image1_uri'] = uri.extra['colmap.image1_uri']
    mp.extra['colmap.image2_uri'] = uri.extra['colmap.image2_uri']

    if cls.USE_NP_CACHED_ASSETS:
      # Make sure the filename is definitely distinct
      iid1, iid2 = mp.extra['colmap.image1_id'], mp.extra['colmap.image2_id']
      matches_npy_fname = (
        f"{uri.topic}_{uri.timestamp}_iid1.{iid1}_iid2.{iid2}_matches.npy")
      matches_npy_path = cls.psegs_npy_cache_dir() / matches_npy_fname
      if not matches_npy_path.exists():
        with open(matches_npy_path, 'wb') as f:
          np.savez_compressed(f, matches=mp.get_matches())
            # Use compression since we use it for other npy assets
      mp.matches_factory = CloudpickeledCallable(
        lambda: load_array(matches_npy_path)['matches'])

    return datum.StampedDatum(uri=uri, matched_pair=mp)

  @classmethod
  def create_stamped_datum(cls, uri):
    if uri.topic == cls.WORLD_CLOUD_TOPIC:
      return cls._create_point_cloud(uri)
    elif uri.topic.endswith(cls.DCI_RECON_TOPIC_SUFFIX):
      return cls._create_depth_camera_image(uri)
    elif uri.topic.endswith(cls.CI_RECON_TOPIC_SUFFIX):
      return cls._create_camera_image(uri)
    elif uri.topic.endswith(cls.MP_TOPIC_SUFFIX):
      return cls._create_matched_pair(uri)
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
  def get_reconstruction_sd_table(cls, spark=None):
    seg_uri = cls.get_segment_uri()
    if not seg_uri:
      return None
    return cls.get_segment_sd_table(segment_uri=seg_uri, spark=spark)

  @classmethod
  def create_sd_table_for_reconstruction(
        cls,
        colmap_recon_dir,
        colmap_input_images_dir,
        psegs_assets_dir,
        spark=None):
    
    class MyCOLMAP_SDTFactory(cls):
      COLMAP_RECON_DIR = Path(colmap_recon_dir)
      COLMAP_IMAGES_DIR = Path(colmap_input_images_dir)
      PSEGS_ASSETS_DIR = Path(psegs_assets_dir)
    
    return MyCOLMAP_SDTFactory.get_reconstruction_sd_table(spark=spark)
      

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
      has_match = any(
              suri.soft_matches_segment_of(cls.get_segment_uri())
              for suri in only_segments)
      if not has_match:
        return []

    # Generate URIs ...
    colmap_uris = cls.get_colmap_recon_uris()
    uri_rdd = spark.sparkContext.parallelize(colmap_uris)
    util.log.info(f"Creating datums for {len(colmap_uris)} URIs ...")

    if cls.USE_NP_CACHED_ASSETS:
      util.log.info(
        f"... using numpy asset cache, may take moments to populate ...")
      if not cls.psegs_npy_cache_dir().exists():
        # Initial cache population may be memory intensive
        uri_rdd = uri_rdd.repartition(uri_rdd.count())
        cls.psegs_npy_cache_dir().mkdir(parents=True, exist_ok=True)

    datum_rdd = uri_rdd.map(cls.create_stamped_datum)

    return [datum_rdd]
