import os

from psegs import util

def save_sample_blender_format(
      cis,
      outdir='/tmp/test_nerf_blender_out',
      split='train',
      parallel=-1):
  """
  Given a list of `:class:`~psegs.datum.camera_image.CameraImage` instances,
  export the images in the "Blender format" that is compatible with most
  NeRF research.

  Args:
    cis (List[CameraImage]): Export these camera images.
    outdir (str): Dump all data to this directory.
    split (str): Export this split of the dataset; the Blender format
      accommodates 'train', 'test', and 'val.
    parallel (int): Use this many export workers (default to one per vcpu
      if negative)

  References:
   * Original NeRF Blender files: https://github.com/bmild/nerf/issues/59
   * Original NeRF Blender *dataset* transforms: https://drive.google.com/drive/folders/1LEDmMJ-rFRhl8CJKnLBeTePDBsmJJ8OP
   * Code that reads Blender data:
     * Original NeRF: https://github.com/bmild/nerf/blob/20a91e764a28816ee2234fcadb73bd59a613a44c/load_blender.py#L41
     * nerf_pl (pytorch lightning): https://github.com/kwea123/nerf_pl/blob/f4a072bc0dc49d2703d2a47da808432d228622e0/datasets/blender.py#L11
     * jaxnerf: https://github.com/google-research/google-research/blob/47795035fc374b9501bbf9a49a1ae05a4d3282e3/jaxnerf/nerf/datasets.py#L196
     * nerf_sh / plenoctrees: https://github.com/sxyu/plenoctree/blob/92ee5c1e367602d08f7eda77ed331f0f515d4b6f/nerf_sh/nerf/datasets.py#L189
     * mipnerf: https://github.com/google/mipnerf/blob/84c969e0a623edd183b75693aed72a7e7c22902d/internal/datasets.py#L311
     * D-nerf: https://github.com/albertpumarola/D-NeRF/blob/f16319df497105b71ac151d2c2ddd4de36a1493f/load_blender.py#L70 
     * related, Pixel-NeRF: https://github.com/sxyu/pixel-nerf/blob/a5a514224272a91e3ec590f215567032e1f1c260/src/data/MultiObjectDataset.py#L72
     * related, NeRF-- (without poses) & COLMAP loader: https://github.com/ActiveVisionLab/nerfmm/blob/27faab66a927ea14259125e1140231f0c8f6d14c/dataloader/with_colmap.py#L119
   * List of lots of projects: https://github.com/visonpon/New-View-Synthesis 
  """

  import json
  import imageio
  from oarphpy import util as oputil
  from oarphpy import spark as S

  from psegs.spark import Spark


  assert split in ('train', 'test', 'val')

  util.log.info("Exporting %s images to Blender format ..." % len(cis))

  img_dir_out = os.path.join(outdir, split)
  oputil.mkdir(str(img_dir_out))

  
  class SaveAndGetFrame(object):
    def __call__(self):
      i = self.i
      ci = self.ci
      img_dir_out = self.img_dir_out

      import imageio

      dest = os.path.join(img_dir_out, 'r_%s.png' % str(i).zfill(6))
      img = ci.image
      imageio.imwrite(dest, img) 
        # NOTE: some nerfs think this image is in [0, 1] ???
      
      c2w = ci.ego_pose['ego', 'world']
      # TODO FIXME  should be camera frame !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      transform_matrix = c2w.get_transformation_matrix(homogeneous=True)
      frame = {
        'transform_matrix': transform_matrix.tolist(),
        'file_path': dest.replace('.png', ''),
          # NB: Dataset readers are supposed to append the .png suffix :S
        # 'rotation': ??? don't know what this is but it's not read?
      }
      return frame

  camera_angle_x = None
  K = None
  cis = sorted(cis, key=lambda ci: ci.timestamp)
  callables = []
  for ci in cis:

    if camera_angle_x is None:
      fov_h, fov_v = ci.get_fov()
      camera_angle_x = fov_h
        # NB: Readers will compute somthing like:
        # focal = .5 * image_width / np.tan(.5 * camera_angle_x)
      
      K = ci.K
    
    c = SaveAndGetFrame()
    c.i = int(ci.extra['threeDScannerApp.frame_id'])
    c.ci = ci
    c.img_dir_out = img_dir_out
    callables.append(c)

  with Spark.sess() as spark:
    results = S.run_callables(spark, callables, parallel=parallel)
    frames = [f for obj, f in results]

  transforms_data = {
    'camera_angle_x': camera_angle_x,
    'frames': frames,
  }

  transforms_dest = os.path.join(outdir, 'transforms_%s.json' % split)
  with open(transforms_dest, 'w') as f:
    json.dump(transforms_data, f, indent=2)

  full_intrinsic = os.path.join(outdir, 'full_intrinsic.json')
  with open(full_intrinsic, 'w') as f:
    json.dump(K.tolist(), f, indent=2)

  util.log.info("... done writing to %s ." % outdir)



if __name__ == '__main__':


  from psegs.datasets import ios_lidar

  # base_dir = '/outer_root/home/au/lidarphone_scans/2021_06_27_12_37_38'
  base_dir = '/outer_root/media/970-evo-plus-raid0/lidarphone_lidar_scans/2021_10_31_20_24_07/'
  # base_dir = '/outer_root/home/au/lidarphone_scans/landscape_home_button_right_07_09_49'

  from oarphpy import util as oputil
  jpg_paths = oputil.all_files_recursive(base_dir, pattern='frame*.jpg')
  json_paths = [p.replace('jpg', 'json') for p in jpg_paths]
  json_paths = sorted(json_paths)
  cis = [ios_lidar.threeDScannerApp_create_camera_image(p) for p in json_paths]

  print(len(cis))
  cis[0]

  save_sample_blender_format(
    cis,
    '/outer_root/home/pwais/bundle-adjusting-NeRF/data/blender/soma-pizza-mural')


