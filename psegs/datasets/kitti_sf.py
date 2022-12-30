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

from oarphpy import util as oputil

from psegs import util
from psegs.conf import C
from psegs.datasets.idsutil import IDatasetUtil


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


  ### Testing #################################################################

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

def kittisf15_load_disp(disp_path):
    import imageio
    
    # From KITTI SF Devkit:
    # "Disparity maps are saved as uint16 PNG images, which can be opened with
    # either MATLAB or libpng++. A 0 value indicates an invalid pixel (ie, no
    # ground truth exists, or the estimation algorithm didn't produce an estimate
    # for that pixel). Otherwise, the disparity for a pixel can be computed by
    # converting the uint16 value to float and dividing it by 256.0"

    img = imageio.imread(disp_path)
    disp = img.astype('float32') / 256.
    return disp

def kittisf15_load_K_baseline(cam_to_cam_path):
    import numpy as np
    
    K_line = None
    T_00_line = None
    T_01_line = None
    with open(cam_to_cam_path, 'r') as f:
        for l in f.readlines():
            if 'P_rect_02' in l:
                K_line = l
            if 'T_02' in l:
                T_00_line = l
            if 'T_03' in l:
                T_01_line = l
    
    assert K_line
    params = K_line.split('P_rect_02: ')[-1]
    params = [float(tok.strip()) for tok in params.split(' ') if tok]
    K = np.array(params).reshape([3, 4])
    K = K[:3, :3]
    
    assert T_00_line
    assert T_01_line
    params = T_00_line.split('T_02: ')[-1]
    params = [float(tok.strip()) for tok in params.split(' ') if tok]
    T_00 = np.array(params)
    params = T_01_line.split('T_03: ')[-1]
    params = [float(tok.strip()) for tok in params.split(' ') if tok]
    T_01 = np.array(params)
    baseline = np.linalg.norm(T_00 - T_01)
    
    return K, baseline

def kittisf15_load_sflow(flow, K, baseline, disp0_path, disp1_path):
    fx = K[0, 0]
    
    disp0 = kittisf15_load_disp(disp0_path)
    disp0_valid = disp0[:, :] > 0
    d0 = fx * baseline / (disp0 + 1e-5)
    d0[~disp0_valid] = 0
    
    disp1 = kittisf15_load_disp(disp1_path)
    disp1_valid = disp1[:, :] > 0
    d1 = fx * baseline / (disp1 + 1e-5)
    d1[~disp1_valid] = 0
    
    h, w = d1.shape[:2]
    px_y = np.tile(np.arange(h)[:, np.newaxis], [1, w])
    px_x = np.tile(np.arange(w)[np.newaxis, :], [h, 1])
    pyx = np.concatenate([px_y[:,:,np.newaxis], px_x[:, :, np.newaxis]], axis=-1)
    pyx = pyx.astype(np.float32)
    
    vud1 = np.dstack([pyx, d0]).reshape([-1, 3])
    uvdviz_im1 = np.zeros((vud1.shape[0], 4))
    uvdviz_im1[:, :3] = vud1[:, (1, 0, 2)]
    uvdviz_im1[:, -1] = np.logical_and(
                            (flow > 0).reshape([-1, 2])[:, 0], # Flow is valid
                            (d0 > 0).reshape([-1]))            # Depth is valid

    vu2 = (pyx + flow[:, :, (1, 0)]).reshape([-1, 2])
    d2_valid = (d1 > 0).reshape([-1])
    invalid = np.where(
            (np.rint(vu2[:, 0]) < 0) | (np.rint(vu2[:, 0]) >= h) |
            (np.rint(vu2[:, 1]) < 0) | (np.rint(vu2[:, 1]) >= w) |
            (flow[:, :, 0] == 0).reshape([-1]) |
            (~d2_valid))
    j2 = np.rint(vu2[:, 0]).astype(np.int64)
    i2 = np.rint(vu2[:, 1]).astype(np.int64)
    j2[invalid] = 0
    i2[invalid] = 0
    d2_col = d1[j2, i2]
    vud2 = np.hstack([vu2, d2_col[:, np.newaxis]])
    
    uvdviz_im2 = np.ones((vud1.shape[0], 4))
    uvdviz_im2[:, :3] = vud2[:, (1, 0, 2)]
    uvdviz_im2[invalid, -1] = 0
    
#     vudviz_im2[:, -1] = (vudviz_im2[:, 0] != -np.Inf)
#     vudviz_im1[:, -1] = np.logical_and(vudviz_im1[:, -1], (vudviz_im1[:, 2] > 0))
    
    visible_either = ((uvdviz_im1[:, -1] == 1) | (uvdviz_im2[:, -1] == 1))
    uvdviz_im1 = uvdviz_im1[visible_either]
    uvdviz_im2 = uvdviz_im2[visible_either]
#         xyz1 = uvd_to_xyzrgb(uvd1, fp.K)[:, :3]
#         xyz2 = uvd_to_xyzrgb(uvd2, fp.K)[:, :3]     
    
    return uvdviz_im1, uvdviz_im2

def kittisf15_create_fp(uri):
  flow = kittisf15_load_flow(os.path.join(KITTI_SF15_DATA_ROOT, uri.extra['ksf15.flow_gt']))
  K, baseline = kittisf15_load_K_baseline(os.path.join(KITTI_SF15_DATA_ROOT, uri.extra['ksf15.K']))
  uvdviz_im1, uvdviz_im2 = kittisf15_load_sflow(
                                  flow, K, baseline,
                                  os.path.join(KITTI_SF15_DATA_ROOT, uri.extra['ksf15.disp0']),
                                  os.path.join(KITTI_SF15_DATA_ROOT, uri.extra['ksf15.disp1']))
  
  return OpticalFlowPair(
              uri=uri,
              dataset="KITTI Scene Flow 2015",
              id1=uri.extra['ksf15.input'],
              img1='file://' + os.path.join(KITTI_SF15_DATA_ROOT, uri.extra['ksf15.input']),
              id2=uri.extra['ksf15.expected_out'],
              img2='file://' + os.path.join(KITTI_SF15_DATA_ROOT, uri.extra['ksf15.expected_out']),
              flow=flow,
      
              K=K,
              uvdviz_im1=uvdviz_im1,
              uvdviz_im2=uvdviz_im2)










###############################################################################
### IDatasetUtil Impl

class DSUtil(IDatasetUtil):

  FIXTURES = Fixtures

  @classmethod
  def all_zips(cls):
    return cls.FIXTURES.ZIPS

  @classmethod
  def emplace(cls):
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
