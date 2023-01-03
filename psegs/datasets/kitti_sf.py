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

import numpy as np
from oarphpy import util as oputil

from psegs import datum
from psegs import util
from psegs.conf import C
from psegs.datasets.idsutil import IDatasetUtil
from psegs.table.sd_table_factory import StampedDatumTableFactory


"""

first load just stereo, maybe get from scene flow..
  * (CameraImage left, CameraImage right, Matches uvleft_uv_right)

test trimesh viz

"""


###############################################################################
### KittiSceneFlow Fixtures & Other Constants

class Fixtures(object):

  ROOT = C.EXT_DATA_ROOT / 'kitti_sf_archives'

  ZIPS = (
    'data_scene_flow.zip',
    'data_scene_flow_calib.zip',
  )

  @classmethod
  def zip_path(cls, zipname):
    return cls.ROOT / zipname

  @classmethod
  def maybe_emplace_psegs_kitti_sf_ext(cls):
    print('todo')


  @classmethod
  def get_all_train_test_frame_ids(cls):
    import zipfile
    entries = zipfile.ZipFile(cls.zip_path('data_scene_flow.zip')).namelist()
    
    def get_frame_id(path):
      return Path(path).name.split('.')[0]
    
    train_frame_ids = [
      get_frame_id(p)
      for p in entries
      if ('training/image_2' in p and '.png' in p)
    ]

    test_frame_ids = [
      get_frame_id(p)
      for p in entries
      if ('training/image_2' in p and '.png' in p)
    ]

    return train_frame_ids, test_frame_ids



  ### Unit Test Support #######################################################

  TEST_FIXTURES_ROOT = Path('/tmp/psegs_kitti_sf_test_fixtures')

  EXTERNAL_FIXTURES_ROOT = C.EXTERNAL_TEST_FIXTURES_ROOT / 'kitti_sf'

  STEREO_TEST_FRAMES= ('000016_10', '000024_10', '000177_10')

  @classmethod
  def stereo_fixture_dir(cls):
    fixture_dir = cls.TEST_FIXTURES_ROOT / 'stereo'
    if util.missing_or_empty(fixture_dir):
      util.log.info(
        "Putting Stereo Benchmark test fixtures in %s" % fixture_dir)
      oputil.cleandir(fixture_dir)
      
      # Disparity
      util.unarchive_entries(
        cls.zip_path('data_scene_flow.zip'),
        ['training/disp_occ_0/%s.png' % f for f in cls.STEREO_TEST_FRAMES],
        fixture_dir)
      
      # RGB
      util.unarchive_entries(
        cls.zip_path('data_scene_flow.zip'),
        ['training/image_2/%s.png' % f for f in cls.STEREO_TEST_FRAMES],
        fixture_dir)
      util.unarchive_entries(
        cls.zip_path('data_scene_flow.zip'),
        ['training/image_3/%s.png' % f for f in cls.STEREO_TEST_FRAMES],
        fixture_dir)
      
      # Calib
      util.unarchive_entries(
        cls.zip_path('data_scene_flow_calib.zip'),
        [
          'training/calib_cam_to_cam/%s.txt' % f.replace('_10', '') 
          for f in cls.STEREO_TEST_FRAMES
        ],
        fixture_dir)
    
    return fixture_dir


###############################################################################
### KITTI Parsing Utils

def kittisf15_load_flow(path):
  # Based upon https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/flowlib.py#L559
  import png
  import numpy as np
  flow_object = png.Reader(filename=path)
  flow_direct = flow_object.asDirect()
  flow_data = list(flow_direct[2])
  w, h = flow_direct[3]['size']
  flow = np.zeros((h, w, 3), dtype=np.float64)
  for i in range(len(flow_data)):
    flow[i, :, 0] = flow_data[i][0::3]
    flow[i, :, 1] = flow_data[i][1::3]
    flow[i, :, 2] = flow_data[i][2::3]

  invalid_idx = (flow[:, :, 2] == 0)
  flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
  flow[invalid_idx, 0] = 0
  flow[invalid_idx, 1] = 0
  return flow[:, :, :2]


def kittisf15_load_disp(disp_data):
  import imageio
  
  # From KITTI SF Devkit:
  # "Disparity maps are saved as uint16 PNG images, which can be opened with
  # either MATLAB or libpng++. A 0 value indicates an invalid pixel (ie, no
  # ground truth exists, or the estimation algorithm didn't produce an estimate
  # for that pixel). Otherwise, the disparity for a pixel can be computed by
  # converting the uint16 value to float and dividing it by 256.0"

  img = imageio.imread(disp_data)
  disp = img.astype('float32') / 256.
  return disp


def kittisf15_load_calib(cam_to_cam_str):
  import numpy as np
  
  # TODO: See psegs.datasets.kitti.Calibration -- we want to migrate to that
  # data structure eventually.

  # Notes from KITTI Raw Devkit: https://github.com/pratikac/kitti/blob/eba7ba0f36917f72055060e9e59f344b72456cb9/readme.raw.txt#L169
  # calib_cam_to_cam.txt: Camera-to-camera calibration
  # --------------------------------------------------
  #   - S_xx: 1x2 size of image xx before rectification
  #   - K_xx: 3x3 calibration matrix of camera xx before rectification
  #   - D_xx: 1x5 distortion vector of camera xx before rectification
  #   - R_xx: 3x3 rotation matrix of camera xx (extrinsic)
  #   - T_xx: 3x1 translation vector of camera xx (extrinsic)
  #   - S_rect_xx: 1x2 size of image xx after rectification
  #   - R_rect_xx: 3x3 rectifying rotation to make image planes co-planar
  #   - P_rect_xx: 3x4 projection matrix after rectification
  # Note: When using this dataset you will most likely need to access only
  # P_rect_xx, as this matrix is valid for the rectified image sequences.
  #
  # And: https://github.com/pratikac/kitti/blob/eba7ba0f36917f72055060e9e59f344b72456cb9/readme.raw.txt#L206
  # example transformations
  # -----------------------
  # As the transformations sometimes confuse people, here we give a short
  # example how points in the velodyne coordinate system can be transformed
  # into the camera left coordinate system.
  #
  # In order to transform a homogeneous point X = [x y z 1]' from the velodyne
  # coordinate system to a homogeneous point Y = [u v 1]' on image plane of
  # camera xx, the following transformation has to be applied:
  # Y = P_rect_xx * R_rect_00 * (R|T)_velo_to_cam * X
  #
  # To transform a point X from GPS/IMU coordinates to the image plane:
  # Y = P_rect_xx * R_rect_00 * (R|T)_velo_to_cam * (R|T)_imu_to_velo * X
  #
  # The matrices are:
  # - P_rect_xx (3x4):         rectfied cam 0 coordinates -> image plane
  # - R_rect_00 (4x4):         cam 0 coordinates -> rectified cam 0 coord.
  # - (R|T)_velo_to_cam (4x4): velodyne coordinates -> cam 0 coordinates
  # - (R|T)_imu_to_velo (4x4): imu coordinates -> velodyne coordinates
  #
  # Note that the (4x4) matrices above are padded with zeros and:
  # R_rect_00(4,4) = (R|T)_velo_to_cam(4,4) = (R|T)_imu_to_velo(4,4) = 1.

  K2_line = None
  K3_line = None
  R_02_line = None
  R_03_line = None
  T_02_line = None
  T_03_line = None
  for l in cam_to_cam_str.split('\n'):
    if 'P_rect_02' in l:
      K2_line = l
    if 'P_rect_03' in l:
      K3_line = l
    if 'T_02' in l:
      T_02_line = l
    if 'T_03' in l:
      T_03_line = l
    if 'R_02' in l:
      R_02_line = l
    if 'R_03' in l:
      R_03_line = l
  
  assert K2_line
  params = K2_line.split('P_rect_02: ')[-1]
  params = [float(tok.strip()) for tok in params.split(' ') if tok]
  P_2 = np.array(params).reshape([3, 4])
  K_2 = P_2[:3, :3]
  
  assert K3_line
  params = K3_line.split('P_rect_03: ')[-1]
  params = [float(tok.strip()) for tok in params.split(' ') if tok]
  P_3 = np.array(params).reshape([3, 4])
  K_3 = P_3[:3, :3]

  assert R_02_line
  assert R_03_line
  params = R_02_line.split('R_02: ')[-1]
  params = [float(tok.strip()) for tok in params.split(' ') if tok]
  R_02 = np.array(params)
  params = R_03_line.split('R_03: ')[-1]
  params = [float(tok.strip()) for tok in params.split(' ') if tok]
  R_03 = np.array(params)

  assert T_02_line
  assert T_03_line
  params = T_02_line.split('T_02: ')[-1]
  params = [float(tok.strip()) for tok in params.split(' ') if tok]
  T_02 = np.array(params)
  params = T_03_line.split('T_03: ')[-1]
  params = [float(tok.strip()) for tok in params.split(' ') if tok]
  T_03 = np.array(params)
  
  # Baseline appears to be in meters, resulting images will have depth to
  # about 78 meters

  baseline = np.linalg.norm(T_02 - T_03)
  
  # Seems calibration is in mm, if we use this baseline then resulting
  # images have depth of ~56,000 (millimeters?)
  # baseline = np.linalg.norm(P_3[:, 3] - P_2[:, 3])
  
  return K_2, K_3, baseline, R_02, T_02, R_03, T_03, P_2, P_3

def kittisf15_to_stereo_matches(disp, baseline, K_2):
  fx = K_2[0, 0]
  
  disp_valid = disp[:, :] > 0
  depth = fx * baseline / (disp + 1e-5)
  depth[~disp_valid] = 0

  h, w = disp.shape[:2]
  px_y_2 = np.tile(np.arange(h)[:, np.newaxis], [1, w])
  px_x_2 = np.tile(np.arange(w)[np.newaxis, :], [h, 1])
  pyx_2 = np.concatenate([px_y_2[:,:,np.newaxis], px_x_2[:, :, np.newaxis]], axis=-1)
  pyx_2 = pyx_2.astype(np.float32)

  vud_2 = np.dstack([pyx_2, depth]).reshape([-1, 3])
  uvd_2 = np.zeros((vud_2.shape[0], 3))
  uvd_2[:, :3] = vud_2[:, (1, 0, 2)]
  
  uv_3 = uvd_2[:, :2].copy()
  uv_3[:, 0] -= disp.reshape(-1)
  uv_2_uv_3_depth = np.hstack([uvd_2[:, :2], uv_3, uvd_2[:, (-1,)]])
  return uv_2_uv_3_depth



# def kittisf15_load_sflow(flow, K, baseline, disp0_path, disp1_path):
#   fx = K[0, 0]
  
#   disp0 = kittisf15_load_disp(disp0_path)
#   disp0_valid = disp0[:, :] > 0
#   d0 = fx * baseline / (disp0 + 1e-5)
#   d0[~disp0_valid] = 0
  
#   disp1 = kittisf15_load_disp(disp1_path)
#   disp1_valid = disp1[:, :] > 0
#   d1 = fx * baseline / (disp1 + 1e-5)
#   d1[~disp1_valid] = 0
  
#   h, w = d1.shape[:2]
#   px_y = np.tile(np.arange(h)[:, np.newaxis], [1, w])
#   px_x = np.tile(np.arange(w)[np.newaxis, :], [h, 1])
#   pyx = np.concatenate([px_y[:,:,np.newaxis], px_x[:, :, np.newaxis]], axis=-1)
#   pyx = pyx.astype(np.float32)
  
#   vud1 = np.dstack([pyx, d0]).reshape([-1, 3])
#   uvdviz_im1 = np.zeros((vud1.shape[0], 4))
#   uvdviz_im1[:, :3] = vud1[:, (1, 0, 2)]
#   uvdviz_im1[:, -1] = np.logical_and(
#               (flow > 0).reshape([-1, 2])[:, 0], # Flow is valid
#               (d0 > 0).reshape([-1]))      # Depth is valid

#   vu2 = (pyx + flow[:, :, (1, 0)]).reshape([-1, 2])
#   d2_valid = (d1 > 0).reshape([-1])
#   invalid = np.where(
#       (np.rint(vu2[:, 0]) < 0) | (np.rint(vu2[:, 0]) >= h) |
#       (np.rint(vu2[:, 1]) < 0) | (np.rint(vu2[:, 1]) >= w) |
#       (flow[:, :, 0] == 0).reshape([-1]) |
#       (~d2_valid))
#   j2 = np.rint(vu2[:, 0]).astype(np.int64)
#   i2 = np.rint(vu2[:, 1]).astype(np.int64)
#   j2[invalid] = 0
#   i2[invalid] = 0
#   d2_col = d1[j2, i2]
#   vud2 = np.hstack([vu2, d2_col[:, np.newaxis]])
  
#   uvdviz_im2 = np.ones((vud1.shape[0], 4))
#   uvdviz_im2[:, :3] = vud2[:, (1, 0, 2)]
#   uvdviz_im2[invalid, -1] = 0
  
# #   vudviz_im2[:, -1] = (vudviz_im2[:, 0] != -np.Inf)
# #   vudviz_im1[:, -1] = np.logical_and(vudviz_im1[:, -1], (vudviz_im1[:, 2] > 0))
  
#   visible_either = ((uvdviz_im1[:, -1] == 1) | (uvdviz_im2[:, -1] == 1))
#   uvdviz_im1 = uvdviz_im1[visible_either]
#   uvdviz_im2 = uvdviz_im2[visible_either]
# #     xyz1 = uvd_to_xyzrgb(uvd1, fp.K)[:, :3]
# #     xyz2 = uvd_to_xyzrgb(uvd2, fp.K)[:, :3]   
  
#   return uvdviz_im1, uvdviz_im2

# def kittisf15_create_fp(uri):
#   flow = kittisf15_load_flow(os.path.join(KITTI_SF15_DATA_ROOT, uri.extra['ksf15.flow_gt']))
#   K, baseline = kittisf15_load_K_baseline(os.path.join(KITTI_SF15_DATA_ROOT, uri.extra['ksf15.K']))
#   uvdviz_im1, uvdviz_im2 = kittisf15_load_sflow(
#                                   flow, K, baseline,
#                                   os.path.join(KITTI_SF15_DATA_ROOT, uri.extra['ksf15.disp0']),
#                                   os.path.join(KITTI_SF15_DATA_ROOT, uri.extra['ksf15.disp1']))
  
#   return OpticalFlowPair(
#               uri=uri,
#               dataset="KITTI Scene Flow 2015",
#               id1=uri.extra['ksf15.input'],
#               img1='file://' + os.path.join(KITTI_SF15_DATA_ROOT, uri.extra['ksf15.input']),
#               id2=uri.extra['ksf15.expected_out'],
#               img2='file://' + os.path.join(KITTI_SF15_DATA_ROOT, uri.extra['ksf15.expected_out']),
#               flow=flow,
      
#               K=K,
#               uvdviz_im1=uvdviz_im1,
#               uvdviz_im2=uvdviz_im2)


###############################################################################
### StampedDatumTableFactory Impl

class KITTISF15SDTable(StampedDatumTableFactory):
  
  FIXTURES = Fixtures

  # The dataset has about 400 total frames; tune here to control memory usage
  FRAMES_PER_PARTITION = 50
  
  ## Public API

  @classmethod
  def _get_all_segment_uris(cls):
    # In KITTI SF15, there are no sequences, so we make every scene flow
    # example a distinct segment
    train_ids, test_ids = cls.FIXTURES.get_all_train_test_frame_ids()

    train_segs = [
      datum.URI(
            dataset='kitti-sf15',
            split='train',
            segment_id=frame_id,
            extra={
              'kitti_sf15.frame_id': frame_id,
            })
      for frame_id in train_ids
    ]

    test_segs = [
      datum.URI(
            dataset='kitti-sf15',
            split='test',
            segment_id=frame_id,
            extra={
              'kitti_sf15.frame_id': frame_id,
            })
      for frame_id in test_ids
    ]
    
    return train_segs + test_segs

  @classmethod
  def _create_datum_rdds(cls, spark, existing_uri_df=None, only_segments=None):

    ## For KITTI SF15, each frame ID becomes a segment / task ...
    seg_uris_to_build = cls._get_all_segment_uris()
    util.log.info(f"Discovered {len(seg_uris_to_build)} segments ...")
    
    ## ... skip any segments we already have ...
    if existing_uri_df is not None:
      def get_frame_id(row):
        return row.extra.get('kitti_sf15.frame_id')
      skip_frame_ids = set(
        existing_uri_df.select('extra').rdd.map(get_frame_id).collect())
      seg_uris_to_build = [
        suri for suri in seg_uris_to_build
        if suri.extra['kitti_sf15.frame_id'] not in skip_frame_ids
      ]
      util.log.info(
        f"Resume mode: have datums for {len(skip_frame_ids)} frames; "
        "reduced to f{len(seg_uris_to_build)} tasks")
    
    if only_segments:
      util.log.info(
        "Filtering to only %s segments" % len(only_segments))
      seg_uris_to_build = [
        uri for uri in seg_uris_to_build
        if any(suri.soft_matches_segment(uri) for suri in only_segments)
      ]

    ## ... now run tasks and create stamped datums.
    util.log.info(
      f"... creating datums for {len(seg_uris_to_build)} segments.")
    datum_rdds = []
    for chunk in oputil.ichunked(seg_uris_to_build, cls.FRAMES_PER_PARTITION):
      chunk_uri_rdd = spark.sparkContext.parallelize(chunk)
      datum_rdd = chunk_uri_rdd.flatMap(cls._create_datums_for_segement_uri)
      datum_rdds.append(datum_rdd)

    return datum_rdds


  ## Datum Construction Support

  @classmethod
  def _create_datums_for_segement_uri(cls, seg_uri):
    # TODO: add scene flow datums, camera image datums as optional, etc
    return [cls._create_matched_pair(seg_uri)]

  @classmethod
  def _get_file_bytes(cls, uri=None, archive=None, entryname=None):
    """Read bytes for the file referred to by `uri`"""

    if uri is not None:
      archive = uri.extra['kitti_sf15.archive']
      entryname = uri.extra['kitti_sf15.archive.path']
    assert archive and entryname

    # Cache the Zipfiles for faster loading
    if not hasattr(cls, '_get_file_bytes_archives'):
      cls._get_file_bytes_archives = {}
    if archive not in cls._get_file_bytes_archives:
      import zipfile
      path = cls.FIXTURES.zip_path(archive)
      cls._get_file_bytes_archives[archive] = zipfile.ZipFile(path)

    try:
      return cls._get_file_bytes_archives[archive].read(entryname)
    except Exception as e:
        raise Exception((e, archive, uri))

  @classmethod
  def _get_calib(cls, uri):
    frame_id = uri.extra['kitti_sf15.frame_id']
    ksplit = uri.split + 'ing'
    calib_key = frame_id.replace("_10", "").replace("_11", "")
    
    calib_uri = copy.deepcopy(uri)
    calib_uri.extra['kitti_sf15.archive.path'] = (
      f'{ksplit}/calib_cam_to_cam/{calib_key}.txt')
    calib_uri.extra['kitti_sf15.archive'] = 'data_scene_flow_calib.zip'
    cam_to_cam_str = cls._get_file_bytes(uri=calib_uri)
    cam_to_cam_str = cam_to_cam_str.decode('utf-8')
    calib = kittisf15_load_calib(cam_to_cam_str)
    return calib

  @classmethod
  def _create_camera_image(cls, uri):
    from psegs.util import misc

    image_png = cls._get_file_bytes(uri=uri)
    width, height = misc.get_png_wh(image_png)

    def _get_image(uri):
      import imageio
      im_bytes = cls._get_file_bytes(uri=uri)
      return imageio.imread(bytearray(im_bytes))

    calib = cls._get_calib(uri)
    K_2, K_3, baseline, R_02, T_02, R_03, T_03, P_2, P_3 = calib

    if uri.topic == 'camera|left':
      K = K_2
      ego_to_sensor = datum.Transform(
                  dest_frame='ego', # for KITTI SF15, left camera is ego
                  src_frame='camera|left')
    elif uri.topic == 'camera|right':
      K = K_3
      ego_to_sensor = datum.Transform(
                  rotation=todo,
                  translation=todo,
                  dest_frame='ego', # for KITTI SF15, left camera is ego
                  src_frame='camera|right')
    else:
      raise ValueError(uri.topic)

    # for KITTI SF15, left camera is ego
    ego_pose = datum.Transform(
                  src_frame='ego',
                  dest_frame=ego_to_sensor.src_frame)
    extra = uri.extra
    ci = datum.CameraImage(
          sensor_name=uri.topic,
          image_factory=lambda: _get_image(uri),
          width=width,
          height=height,
          timestamp=uri.timestamp,
          ego_pose=ego_pose,
          K=K,
          ego_to_sensor=ego_to_sensor,
          extra=extra)
    return ci
  
  @classmethod
  def _create_matched_pair(cls, uri):

    def _get_matches(base_uri):
      frame_id = base_uri.extra['kitti_sf15.frame_id']

      disp_uri = copy.deepcopy(uri)
      ksplit = uri.split + 'ing'
      disp_uri.extra['kitti_sf15.archive.path'] = (
        f'{ksplit}/disp_occ_0/{frame_id}.png')
      disp_uri.extra['kitti_sf15.archive'] = 'data_scene_flow.zip'
      disp_bytes = cls._get_file_bytes(uri=disp_uri)
      disp = kittisf15_load_disp(disp_bytes)
      
      calib = cls._get_calib(uri)
      K_2, K_3, baseline, R_02, T_02, R_03, T_03, P_2, P_3 = calib
      
      uv_2_uv_3_depth = kittisf15_to_stereo_matches(disp, baseline, K_2)
      return uv_2_uv_3_depth

    frame_id = uri.extra['kitti_sf15.frame_id']
    ksplit = uri.split + 'ing'

    img1_uri = copy.deepcopy(uri)
    img1_uri.topic = 'camera|left'
    img1_uri.extra['kitti_sf15.archive.path'] = (
        f'{ksplit}/image_2/{frame_id}.png')
    img1_uri.extra['kitti_sf15.archive'] = 'data_scene_flow.zip'
    
    img2_uri = copy.deepcopy(uri)
    img2_uri.topic = 'camera|right'
    img2_uri.extra['kitti_sf15.archive.path'] = (
        f'{ksplit}/image_3/{frame_id}.png')
    img2_uri.extra['kitti_sf15.archive'] = 'data_scene_flow.zip'

    mp = datum.MatchedPair(
                matcher_name='kitti_gt',
                timestamp=uri.timestamp,
                img1=cls._create_camera_image(img1_uri),
                img2=cls._create_camera_image(img2_uri),
                matches_factory=lambda: _get_matches(uri),
                matches_colnames=['x1', 'y1', 'x2', 'y2', 'depth_meters'],
                extra=uri.extra)
    
    sd_uri = copy.deepcopy(uri)
    sd_uri.topic = 'camera|matches'
    
    return datum.StampedDatum(uri=sd_uri, matched_pair=mp)




###############################################################################
### IDatasetUtil Impl

class DSUtil(IDatasetUtil):

  FIXTURES = Fixtures

  @classmethod
  def all_zips(cls):
    return cls.FIXTURES.ZIPS

  @classmethod
  def emplace(cls):
    import os

    cls.FIXTURES.maybe_emplace_psegs_kitti_sf_ext()

    if not cls.FIXTURES.ROOT.exists():
      zips = '\n        '.join('  * %s' % fname for fname in cls.all_zips())
      cls.show_md("""
        Due to KITTI license constraints, you need to manually accept the KITTI
        license to obtain the download URLs for the
        [Stereo / Scene Flow](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
        zip files.  But once you have the URL, it's easy to write a short bash
        loop with `wget` to fetch them in parallel.

        You'll want to download all the following zip files (do not decompress
        them) to a single directory on a local disk (spinning disk OK):

        %s

        Once you've downloaded the archives, we'll need the path to where
        you put them.  Enter that below, or exit this program.

      """ % (zips,))
      kitti_sf_root = input(
        "Please enter the directory containing your KITTI Scene Flow 2015 zip "
        "archives; PSegs will create a (read-only) symlink to them: ")
      kitti_sf_root = Path(kitti_sf_root.strip())
      assert kitti_sf_root.exists()
      assert kitti_sf_root.is_dir()

      from oarphpy import util as oputil
      oputil.mkdir(str(cls.FIXTURES.ROOT.parent))

      cls.show_md("Symlink: \n%s <- %s" % (kitti_sf_root, cls.FIXTURES.ROOT))
      os.symlink(kitti_sf_root, cls.FIXTURES.ROOT)

      # Make symlink read-only
      import stat
      os.chmod(
        kitti_sf_root,
        stat.S_IREAD|stat.S_IRGRP|stat.S_IROTH,
        follow_symlinks=False)

    cls.show_md("Validating KITTI SF 2015 archives ...")
    zips_needed = set(cls.all_zips())
    zips_have = set()
    for entry in cls.FIXTURES.ROOT.iterdir():
      if entry.name in zips_needed:
        zips_needed.remove(entry.name)
        zips_have.add(entry.name)
    
    if zips_needed:
      s_have = \
        '\n        '.join('  * %s' % fname for fname in zips_have)
      s_needed = \
        '\n        '.join('  * %s' % fname for fname in zips_needed)
      cls.show_md("""
        Missing some expected archives!

        Found:
        
        %s

        Missing:

        %s
      """ % (s_have, s_needed))
      return False
    
    cls.show_md("... all KITTI SF 2015 archives found!")
    return True

  @classmethod
  def test(cls):
    from oarphpy import util as oputil
    oputil.run_cmd("cd %s && pytest -s -vvv -k test_kitti_sf" % C.PS_ROOT)
    return True
