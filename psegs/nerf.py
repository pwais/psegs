import os
from psegs.dummyrun import T

from psegs import util

def save_sample_blender_format(
      cis,
      outdir='/tmp/test_nerf_blender_out',
      split='train'):
  """
  Given a list of `:class:`~psegs.datum.camera_image.CameraImage` instances,
  export the images in the "Blender format" that is compatible with most
  NeRF research.

  Args:
    cis (List[CameraImage]): Export these camera images.
    outdir (str): Dump all data to this directory.
    split (str): Export this split of the dataset; the Blender format
      accommodates 'train', 'test', and 'val.

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

  assert split in ('train', 'test', 'val')

  util.log.info("Exporting %s images to Blender format ..." % len(cis))

  img_dir_out = os.path.join(outdir, split)
  oputil.mkdir(str(img_dir_out))

  camera_angle_x = None
  frames = []
  cis = sorted(cis, key=lambda ci: ci.timestamp)
  t = oputil.ThruputObserver(
                    name='save_sample_blender_format',
                    n_total=len(cis),
                    log_freq=10,
                    log_on_del=True)
  for i, ci in enumerate(cis):
    t.start_block()

    if camera_angle_x is None:
      fov_h, fov_v = ci.get_fov()
      camera_angle_x = fov_h
        # NB: Readers will compute somthing like:
        # focal = .5 * image_width / np.tan(.5 * camera_angle_x)
    
    dest = os.path.join(img_dir_out, 'r_%s.png' % i)
    img = ci.image
    imageio.imwrite(dest, img)
    
    c2w = ci.ego_pose[ci.sensor_name, 'world']
    transform_matrix = c2w.get_transformation_matrix(homogeneous=True)
    frames.append({
      'transform_matrix': transform_matrix.tolist(),
      'file_path': dest.replace('.png', ''),
        # NB: Dataset readers are supposed to append the .png suffix :S
      # 'rotation': ??? don't know what this is but it's not read?
    })
    
    t.update_tallies(n=1, num_bytes=oputil.get_size_of_deep(img))
    t.maybe_log_progress()

  transforms_data = {
    'camera_angle_x': camera_angle_x,
    'frames': frames,
  }
  
  transforms_dest = os.path.join(outdir, 'transforms_%s.json' % split)
  with open(transforms_dest, 'w') as f:
    json.dump(transforms_data, f, indent=2)

  util.log.info("... done writing to %s ." % outdir)
