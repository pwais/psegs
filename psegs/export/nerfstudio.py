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


import json
from pathlib import Path

from psegs import util

def export_sdt_to_nerfstudio_format(
      sd_table,
      outdir,
      downscales=(2, 4, 8),
      resize_max_h=-1,
      img_ext='png',
      only_cameras=None,
      limit=-1):
  """
  Given a `:class:`~psegs.table.StampedDatumTable` instance, export the 
  `CameraImage` images (and other metadata) to `outdir` in Nerfstudio format:
   * https://github.com/nerfstudio-project/nerfstudio/blob/5d640bfcfdf174922687c11a629a9eb7659a47ce/nerfstudio/data/dataparsers/nerfstudio_dataparser.py#L14

  Args:
    sd_table (StampedDatumTable): Export images from this table.
    outdir (Path or str): Dump all data to this directory.
    splits_to_write (List[str]): Export transform data for these splits
      (NB: we currently always ignore splits specified in the `sd_table` URIs).
    downscales (List[int]): Also export downsized copies of images, downscaled
      by these factors.
    resize_max_h (int): Resize input images to have this maximum
      height in pixels.
    img_ext (str): Save images in this format.
    only_cameras (List[str]): Only export these camera topics.
    limit (int): Sample this number of frames uniformly.

  """

  outdir = Path(outdir)

  images_outdir = outdir / 'images'


  # Select the datums to export
  datum_rdd = sd_table.get_datum_rdd_matching(
                  only_types=['camera_image'],
                  only_topics=only_cameras)
  
  def has_rgb(stamped_datum):
    ci = stamped_datum.camera_image
    return ci.has_rgb()
  datum_rdd = datum_rdd.filter(has_rgb)

  if limit >= 0:
    n_total = datum_rdd.count()
    frac = float(limit) / max(n_total, 1)
    datum_rdd = datum_rdd.sample(
                  fraction=frac,
                  withReplacement=False,
                  seed=1337)

  # Try to favor fewer, longer-lived python processes
  from oarphpy.spark import cluster_cpu_count
  from psegs.spark import Spark
  with Spark.sess() as spark:
    n_cpus = cluster_cpu_count(spark)
  datum_rdd = datum_rdd.repartition(n_cpus).cache()

  if datum_rdd.count() == 0:
    util.log.info(f"Nothing to export for {outdir}!")
    return
  
  util.log.info(f"Selected {datum_rdd.count()} input images ...")

  def save_image(stamped_datum):
    import imageio
    import cv2
    ci = stamped_datum.camera_image
    fname = (
      str(stamped_datum.uri.topic) + "." + 
      str(stamped_datum.uri.timestamp) + "." + img_ext)
    dest = images_outdir / fname
    image = ci.image.copy()
    h, w = image.shape[:2]

    frame = {}
    frame['psegs_uri'] = str(stamped_datum.uri)

    # Maybe rescale, and record dimensions
    th, tw = h, w
    frame['psegs_rescaled'] = 1.0
    if resize_max_h >= 0 and h > resize_max_h:
      th = int(resize_max_h)
      tw = int((float(w) / h) * th)
      image = cv2.resize(image, (tw, th))
      frame['psegs_rescaled'] = float(tw) / w
    frame['height'] = th
    frame['h'] = th
    frame['width'] = tw
    frame['w'] = tw
    
    imageio.imwrite(dest, image)

    for dfactor in downscales:
      d_sz = (int(tw / dfactor), int(th / dfactor))
      image_downscaled = cv2.resize(image, d_sz)

      d_base_dir = outdir / f'images_{dfactor}'
      d_dest = d_base_dir / fname
      imageio.imwrite(d_dest, image_downscaled)

    T_c2w = ci.get_world_to_sensor()#.get_inverse() DEBUG THIS
    c2w = T_c2w.get_transformation_matrix(homogeneous=True)
    
    import numpy as np
    OPENCV_2_OPENGL = np.diag([1, -1, -1, 1])
    c2w = c2w @ OPENCV_2_OPENGL

    frame['transform_matrix'] = c2w.tolist()
    frame['file_path'] = str(dest.relative_to(outdir))

    fx = ci.K[0, 0]
    fy = ci.K[1, 1]
    cx = ci.K[0, 2]
    cy = ci.K[1, 2]
    frame['fl_x'] = fx * frame['psegs_rescaled']
    frame['fl_y'] = fy * frame['psegs_rescaled']
    frame['cx'] = cx * frame['psegs_rescaled']
    frame['cy'] = cy * frame['psegs_rescaled']

    if 'colmap.camera_params_raw_json' in ci.extra:
      params_raw = ci.extra['colmap.camera_params_raw_json']
      
      # FMI https://github.com/colmap/colmap/blob/e180948665b03c4a12d45e2ca39a589f42fdbda6/src/base/camera_models.h#L235
      if ci.extra.get('colmap.camera_model_name') in ('OPENCV', 'FULL_OPENCV'):
        params = json.loads(params_raw)
        k1, k2, p1, p2 = params[4:8] # FULL_OPENCV has more ...

        frame['camera_model'] = 'OPENCV' # TODO support FULL_OPENCV
        frame['k1'] = k1
        frame['k2'] = k2
        frame['p1'] = p1
        frame['p2'] = p2

    return frame
  
  images_outdir.mkdir(parents=True, exist_ok=True)
  for dfactor in downscales:
    d_base_dir = outdir / f'images_{dfactor}'
    d_base_dir.mkdir(parents=True, exist_ok=True)

  frames = datum_rdd.map(save_image).collect()
  frames = [f for f in frames if f]
  frames = sorted(frames, key=lambda f: f['file_path'])
  util.log.info(f"... saved {len(frames)} input images frames ...")

  transforms_data = {
    'frames': frames,
  }

  KEYS_TO_MAKE_GLOBAL = (
    'fl_x', 'fl_y', 'cx', 'cy',
    'w', 'h',
    'camera_model',
    'k1', 'k2', 'p1', 'p2',
  )
  if frames:
    f0 = frames[0]
    for k in KEYS_TO_MAKE_GLOBAL:
      if k in f0:
        transforms_data[k] = f0[k]    

  transforms_dest = outdir / f'transforms.json' 
  with open(transforms_dest, 'w') as f:
    json.dump(transforms_data, f, indent=2)
  util.log.info(f"... saved {transforms_dest} .")

if __name__ == '__main__':
  # TODO: need to port these datas ...
  # from psegs.table.sd_table_factory import ParquetSDTFactory
  # F = ParquetSDTFactory.factory_for_sd_subdirs(
  #   '/outer_root/media/mai-tank/hloc_out/pwais.private.canepa-speedster/psegs/')
  # T = F.create_as_single_table()
  # export_sdt_to_nerfstudio_format(
  #   T,
  #   '/outer_root/media/mai-tank/ns-test-root')
  
  spath = Path('/outer_root/media/mai-tank/hloc_out_mrskylake/pwais.private.moms-strawberrys-comp')

  from psegs.datasets.colmap import COLMAP_SDTFactory
  sdt = COLMAP_SDTFactory.create_sd_table_for_reconstruction(
              spath/'sfm_out/sfm_superpoint+superglue/',
              spath/'images/',
              '/outer_root/media/mai-tank/ns-test-root/tastpsegs')
  export_sdt_to_nerfstudio_format(
    sdt,
    '/outer_root/media/mai-tank/ns-test-root')