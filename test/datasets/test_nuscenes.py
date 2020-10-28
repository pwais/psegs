# Copyright 2020 Maintainers of PSegs
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

import itertools
import shelve
from pathlib import Path

from oarphpy import util as oputil

from psegs import util
from psegs.conf import C
from psegs.datasets.idsutil import IDatasetUtil


###############################################################################
### NuScenes Fixtures & Other Constants

class FixturesBase(object):
  # Subclasses should configure
  FLAVOR = ''
  ROOT = None
  DATASET_VERSIONS = tuple()

  @classmethod
  def index_root(cls, nusc_ds_version):
    """A r/w place to cache any temp / index data"""
    return C.PS_TEMP / cls.FLAVOR / nusc_ds_version

  @classmethod
  def version_exists(cls, version):
    return (cls.ROOT / version).exists()

  ### DSUtil Auto-download ####################################################

  @classmethod
  def maybe_emplace_psegs_ext(cls):
    print('todo')
    print('todo')
    print('todo')
    print('todo') #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return 

    if (cls.bench_to_raw_path().exists() and 
          cls.EXTERNAL_FIXTURES_ROOT.exists()):
      return
    
    from oarphpy import util as oputil
    util.log.info("Downloading latest PSegs NuScenes Extension data ...")
    oputil.mkdir(str(cls.index_root()))
    psegs_kitti_ext_root = cls.index_root() / 'psegs_kitti_ext_tmp'
    if not psegs_kitti_ext_root.exists():
      oputil.run_cmd(
        "git clone https://github.com/pwais/psegs-kitti-ext %s" % \
          psegs_kitti_ext_root)

    util.log.info("... emplacing PSegs KITTI Extension data ...")
    def move(src, dest):
      oputil.mkdir(dest.parent)
      oputil.run_cmd("mv %s %s" % (src, dest))
    
    move(
      psegs_kitti_ext_root / 'assets' / 'bench_to_raw_df',
      cls.bench_to_raw_path())
    move(
      psegs_kitti_ext_root / 'ps_external_test_fixtures',
      cls.EXTERNAL_FIXTURES_ROOT)
    
    util.log.info("... emplace success!")
    util.log.info("(You can remove %s if needed)" % psegs_kitti_ext_root)


class NuscFixtures(FixturesBase):
  FLAVOR = 'nuscenes'
  ROOT = C.EXT_DATA_ROOT / 'nuscenes'
  DATASET_VERSIONS = ('v1.0-mini', 'v1.0-trainval', 'v1.0-test')

class LyftFixtures(FixturesBase):
  FLAVOR = 'lyft_level_5_detection'
  ROOT = C.EXT_DATA_ROOT / 'lyft_level_5_detection'
  DATASET_VERSIONS = ('train',)


from nuscenes.nuscenes import NuScenes
class PSegsNuScenes(NuScenes):
  """A PSegs wrapper around the NuScenes dataset handle featuring reduced
  memory usage and utilities for easier PSegs interop.

  ## Reduced Memory Usage & Faster Loading
  The base NuScenes object uses ~8GB resident RAM (each instance) and takes
  about 30sec to instantiate due to the "tables" of JSON data that it loads.  
  Below we replace these "tables" with disk-based `shelve`s in order to
  dramatically reduce memory usage (to less than 1GB resident) and make
  object instantiation faster (down to about 3 sec). We designed this change 
  to support instantiating multiple NuScenes readers per machine (e.g. in
  Spark or any parallel use case).

  To warm the disk--based caches (requires one-time temporary ~8GB memory
  usage), use `maybe_warm_caches()` below.  At the time of writing
  the cache for the `v1.0-trainval` split requires about 4GB of cache and takes
  about 2-3 minutes on a 3.9GHz Intel Core i7.

  ## Utils
  See `print_sensor_sample_rates()` in particular for important data about
  NuScenes / Lyft Level 5 dataset sample rates.
  """

  FIXTURES = NuscFixtures

  def _get_cache_path(self, table_name):
    """Return a path to a shelve cache of the given nuscenes `table_name`"""
    return self.FIXTURES.index_root(self.version) / table_name

  @classmethod
  def maybe_warm_caches(cls, versions=None):
    versions = versions or cls.FIXTURES.DATASET_VERSIONS
    for version in versions:
      if (cls.FIXTURES.ROOT / version).exists():
        # To warm the cache, simply instantiate and delete. To bust caches,
        # user needs to delete paths that will be logged to stdout.
        nusc = cls(version=version)
        del nusc

  def __init__(self, **kwargs):
    """FMI see NuScenes.__init__().  The parent class will read JSON blobs
    and load 8GB+ data into resident memory.  In this override, we load data
    using the parent NuScenes implemenation (thus, temporarily, using the same
    amount of resident memory) but then cache the table data to disk using
    python shelve`.  We then free the resident memory and use the disk-based
    for token-based access.
    """

    self.version = kwargs.get('version', 'v1.0-mini')
    self.dataroot = kwargs.get('dataroot', str(self.FIXTURES.ROOT))
      # Base ctor does this, but we'll do it here so that path-resolving
      # superclass methods below work properly
    
    if util.missing_or_empty(self._get_cache_path('')):
      util.log.info(
        "Creating shelve caches; will use ~8-10GB of RAM ...")
      nusc = NuScenes(**kwargs)
        # NB: The above ctor call not only loads all JSON but also runs
        # 'reverse indexing', which **EDITS** the data loaded into memory.
        # We'll then write the edited data below using `shelve` so that we
        # don't have to try to make `PSegsNuScenes` support reverse indexing
        # itself.
      util.log.info("... NuScenes done loading & indexing JSON data ...")
      
      table_names = list(nusc.table_names)
      if (Path(self.table_root) / 'lidarseg.json').exists():
        table_names += ['lidarseg']
      for table_name in nusc.table_names:
        cache_path = self._get_cache_path(table_name)
        oputil.mkdir(cache_path.parent)

        util.log.info(
          "Building shelve cache for %s (in %s) ..." % (
            table_name, cache_path))
        
        import pickle
        d = shelve.open(str(cache_path), protocol=pickle.HIGHEST_PROTOCOL)
        rows = getattr(nusc, table_name) # E.g. self.sample_data
        d.update((r['token'], r) for r in rows)
        d.close()
      util.log.info("... done.")
      del nusc # Free several GB memory
    
    super(PSegsNuScenes, self).__init__(**kwargs)

  def _get_table(self, table_name):
    attr = '_cached_' + table_name
    if not hasattr(self, attr):
      cache_path = self._get_cache_path(table_name)
      util.log.info(
        "Using shelve cache for %s at %s" % (table_name, cache_path))
      d = shelve.open(str(cache_path))
      setattr(self, attr, d)
    return getattr(self, attr)

  def __load_table__(self, table_name):
    return self._get_table(table_name).values()
      # NB: Despite the type annotation in the parent class, the base method
      # actually returns a list of dicts and not a single dict.  This
      # subclass method returns a Values View (a generator-like thing)
      # and does not break any core NuScenes functionality.
  
  def __make_reverse_index__(self, verbose):
    # NB: Shelve data files have, built-in, the reverse indicies that the
    # base `NuScenes` creates.  See above.  This override allows the subclass
    # to safely invoke the parent class CTor.
    pass
  
  def get(self, table_name, token):
    assert table_name in self.table_names, \
      "Table {} not found".format(table_name)
    return self._get_table(table_name)[token]
  
  def getind(self, table_name, token):
    # This override should be safe due to our override of `get()` above"""
    raise ValueError("Unsupported / unnecessary; provided by shelve")

  def get_sample_data_for_scene(self, scene_name):
    # print('expensive get rows')

    sample_to_scene = {}
    for sample in self.sample:
      scene = self.get('scene', sample['scene_token'])
      sample_to_scene[sample['token']] = scene['token']

    for sd in self.sample_data:
      if self.get('scene', sample_to_scene[sd['sample_token']])['name'] == scene_name:
        yield sd
    # df = self.sample_data_ts_df
    # df = df[df['scene_name'] == scene_name]
    # return df.to_dict(orient='records')

  #### PSegs Adhoc Utils

  def get_all_sensors(self):
    return set(itertools.chain.from_iterable(
      s['data'].keys() for s in self.sample))
    # NuScenes:
    # 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT',
    #   'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    # 'LIDAR_TOP',
    # 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'RADAR_FRONT',
    #   'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT'
    # Lyft Level 5:
    # 'CAM_FRONT_ZOOMED', 'CAM_BACK', 'LIDAR_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    # 'CAM_BACK_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'LIDAR_TOP',
    # 'LIDAR_FRONT_LEFT', 'CAM_BACK_RIGHT'
  
  def get_all_classes(self):
    return set(anno['category_name'] for anno in self.sample_annotation)
    # NuScenes:
    # 'animal',
    # 'human.pedestrian.adult',
    # 'human.pedestrian.child',
    # 'human.pedestrian.construction_worker',
    # 'human.pedestrian.personal_mobility',
    # 'human.pedestrian.police_officer',
    # 'human.pedestrian.stroller',
    # 'human.pedestrian.wheelchair',
    # 'movable_object.barrier',
    # 'movable_object.debris',
    # 'movable_object.pushable_pullable',
    # 'movable_object.trafficcone',
    # 'static_object.bicycle_rack',
    # 'vehicle.bicycle',
    # 'vehicle.bus.bendy',
    # 'vehicle.bus.rigid',
    # 'vehicle.car',
    # 'vehicle.construction',
    # 'vehicle.emergency.ambulance',
    # 'vehicle.emergency.police',
    # 'vehicle.motorcycle',
    # 'vehicle.trailer',
    # 'vehicle.truck'
    # Lyft Level 5:
    # 'other_vehicle', 'bus', 'truck', 'car', 'bicycle', 'pedestrian', 
    # 'animal', 'emergency_vehicle', 'motorcycle'

  def get_table_to_length(self):
    return dict(
      (table, len(getattr(self, table)))
      for table in self.table_names)

  def print_sensor_sample_rates(self, scene_names=None):
    """Print a report to stdout describing the sample rates of all sensors,
    labels, and localization objects."""

    if not scene_names:
      scene_names = [s['name'] for s in self.scene]
    scene_names = set(scene_names)

    scene_tokens = set(
      s['token'] for s in self.scene if s['name'] in scene_names)
    for scene_token in scene_tokens:
      scene_samples = [
        s for s in self.sample if s['scene_token'] == scene_token
      ]

      def to_sec(timestamp):
        # NuScenes and Lyft Level 5 timestamps are in microseconds
        return timestamp * 1e-6

      # Samples are supposed to be 'best groupings' of lidar and camera data.
      # Let's measure their sample rate.
      from collections import defaultdict
      name_to_tss = defaultdict(list)
      for sample in scene_samples:
        name_to_tss['sample (annos)'].append(to_sec(sample['timestamp']))
      
      # Now measure individual sensor sample rates.
      sample_toks = set(s['token'] for s in scene_samples)
      scene_sample_datas = [
        sd for sd in self.sample_data if sd['sample_token'] in sample_toks
      ]
      for sd in scene_sample_datas:
        name_to_tss[sd['channel']].append(to_sec(sd['timestamp']))
        ego_pose = self.get('ego_pose', sd['ego_pose_token'])
        name_to_tss['ego_pose'].append(to_sec(ego_pose['timestamp']))
        name_to_tss['sample_data'].append(to_sec(sd['timestamp']))
      
      annos = [
        a for a in self.sample_annotation if a['sample_token'] in sample_toks
      ]
      num_tracks = len(set(a['instance_token'] for a in annos))

      import itertools
      all_ts = sorted(itertools.chain.from_iterable(name_to_tss.values()))
      from datetime import datetime
      dt = datetime.utcfromtimestamp(all_ts[0])
      start = dt.strftime('%Y-%m-%d %H:%M:%S')
      duration = (all_ts[-1] - all_ts[0])

      # Print a report
      print('---')
      print('---')

      scene = self.get('scene', scene_token)
      print('Scene %s %s' % (scene['name'], scene['token']))
      print('Start %s \tDuration %s sec' % (start, duration))
      print('Num Annos %s (Tracks %s)' % (len(annos), num_tracks))
      import pandas as pd
      from collections import OrderedDict
      rows = []
      for name in sorted(name_to_tss.keys()):
        def get_series(name):
          return np.array(sorted(name_to_tss[name]))
        
        series = get_series(name)
        freqs = series[1:] - series[:-1]

        lidar_series = get_series('LIDAR_TOP')
        diff_lidar_ms = 1e3 * np.mean(
          [np.abs(lidar_series - t).min() for t in series])

        rows.append(OrderedDict((
          ('Series',              name),
          ('Freq Hz',             1. / np.mean(freqs)),
          ('Diff Lidar (msec)',   diff_lidar_ms),
          ('Duration',            series[-1] - series[0]),
          ('Support',             len(series)),
        )))
      print(pd.DataFrame(rows))

      print()
      print()

      # NuScenes:
      # ---
      # ---
      # Scene scene-1000 09f67057dd8346388b28f79d9bb1cf04
      # Start 2018-11-14 11:01:41       Duration 19.922956943511963 sec
      # Num Annos 493 (Tracks 27)
      #                Series     Freq Hz  Diff Lidar (msec)   Duration  Support
      # 0            CAM_BACK   11.738035          10.554540  19.850000      234
      # 1       CAM_BACK_LEFT   11.536524           0.918133  19.850000      230
      # 2      CAM_BACK_RIGHT   11.788413          20.070969  19.850000      235
      # 3           CAM_FRONT   11.637280          14.986247  19.850000      232
      # 4      CAM_FRONT_LEFT   11.536524           7.586523  19.850000      230
      # 5     CAM_FRONT_RIGHT   11.536524          22.528135  19.850000      230
      # 6           LIDAR_TOP   19.747937           0.000000  19.850175      393
      # 7     RADAR_BACK_LEFT   12.593898          12.635898  19.850883      251
      # 8    RADAR_BACK_RIGHT   13.211602          12.786931  19.831054      263
      # 9         RADAR_FRONT   12.976290          12.728528  19.882416      259
      # 10   RADAR_FRONT_LEFT   13.510020          12.710849  19.911148      270
      # 11  RADAR_FRONT_RIGHT   13.724216          12.571387  19.891846      274
      # 12           ego_pose  155.599393          11.128198  19.922957     3101
      # 13     sample (annos)    2.015096           0.000000  19.850175       41
      # 14        sample_data  155.599393          11.128198  19.922957     3101
      # ---
      # ---
      # Scene scene-0293 6308d6d934074a028fc3145eedf3e65f
      # Start 2018-08-31 15:25:42       Duration 19.525898933410645 sec
      # Num Annos 3548 (Tracks 277)
      #                Series     Freq Hz  Diff Lidar (msec)   Duration  Support
      # 0            CAM_BACK   11.773779          10.441362  19.450000      230
      # 1       CAM_BACK_LEFT   11.825193           1.573582  19.450000      231
      # 2      CAM_BACK_RIGHT   11.928021          19.624993  19.450000      233
      # 3           CAM_FRONT   11.876607          14.872485  19.450000      232
      # 4      CAM_FRONT_LEFT   11.876607           7.329719  19.450000      232
      # 5     CAM_FRONT_RIGHT   11.979434          22.875966  19.450000      234
      # 6           LIDAR_TOP   19.844216           0.000000  19.451512      387
      # 7     RADAR_BACK_LEFT   13.157897          12.698666  19.455997      257
      # 8    RADAR_BACK_RIGHT   13.490288          12.647669  19.421380      263
      # 9         RADAR_FRONT   13.082394          12.616018  19.491845      256
      # 10   RADAR_FRONT_LEFT   13.305139          12.709008  19.466163      260
      # 11  RADAR_FRONT_RIGHT   13.209300          12.723777  19.455989      258
      # 12           ego_pose  157.329504          11.144872  19.525899     3073
      # 13     sample (annos)    2.004986           0.000000  19.451512       40
      # 14        sample_data  157.329504          11.144872  19.525899     3073
      # ---
      # ---
      # Scene scene-1107 89f20737ec344aa48b543a9e005a38ca
      # Start 2018-11-21 11:59:53       Duration 19.820924997329712 sec
      # Num Annos 496 (Tracks 47)
      #                Series     Freq Hz  Diff Lidar (msec)   Duration  Support
      # 0            CAM_BACK   11.696203          11.014590  19.750000      232
      # 1       CAM_BACK_LEFT   11.848101           1.382666  19.750000      235
      # 2      CAM_BACK_RIGHT   11.746835          20.483792  19.750000      233
      # 3           CAM_FRONT   11.848103          14.481831  19.749997      235
      # 4      CAM_FRONT_LEFT   11.898734           7.063236  19.750000      236
      # 5     CAM_FRONT_RIGHT   11.898734          22.159666  19.750000      236
      # 6           LIDAR_TOP   19.797377           0.000000  19.750091      392
      # 7     RADAR_BACK_LEFT   13.578124          12.646209  19.811279      270
      # 8    RADAR_BACK_RIGHT   13.271911          12.721395  19.740940      263
      # 9         RADAR_FRONT   13.198245          12.553021  19.775357      262
      # 10   RADAR_FRONT_LEFT   13.730931          12.645984  19.736462      272
      # 11  RADAR_FRONT_RIGHT   13.778595          12.488227  19.740764      273
      # 12           ego_pose  158.317536          11.102567  19.820925     3139
      # 13     sample (annos)    2.025307           0.000000  19.750091       41
      # 14        sample_data  158.317536          11.102567  19.820925     3139

      # Lyft Level 5:
      # ---
      # ---
      # Scene host-a015-lidar0-1235423635198474636-1235423660098038666 755e4564756ad5c92243b7f77039d07ab1cce40662a6a19b67c820647666a3ef
      # Start 2019-02-28 21:13:55       Duration 24.99979877471924 sec
      # Num Annos 1637 (Tracks 44)
      #               Series    Freq Hz  Diff Lidar (msec)   Duration  Support
      # 0           CAM_BACK   5.020080          98.882582  24.900000      126
      # 1      CAM_BACK_LEFT   5.020080          16.919276  24.900000      126
      # 2     CAM_BACK_RIGHT   5.020080          82.887411  24.900000      126
      # 3          CAM_FRONT   5.020080          50.027272  24.900000      126
      # 4     CAM_FRONT_LEFT   5.020080          33.542296  24.900000      126
      # 5    CAM_FRONT_RIGHT   5.020080          66.427920  24.900000      126
      # 6   CAM_FRONT_ZOOMED   5.020080          50.096270  24.900000      126
      # 7          LIDAR_TOP   5.020156           0.000000  24.899626      126
      # 8           ego_pose  40.442375           0.000000  24.899626     1008
      # 9     sample (annos)   5.020156           0.000000  24.899626      126
      # 10       sample_data  40.280324          49.847878  24.999799     1008
      # ---
      # ---
      # Scene host-a004-lidar0-1233947108297817786-1233947133198765096 114b780b2efd6f73f134fc3a8f9db628e43131dc47f90e9b5dfdb886400d70f2
      # Start 2019-02-11 19:05:08       Duration 25.000741004943848 sec
      # Num Annos 4155 (Tracks 137)
      #               Series    Freq Hz  Diff Lidar (msec)   Duration  Support
      # 0           CAM_BACK   5.020080          98.790201  24.900000      126
      # 1      CAM_BACK_LEFT   5.020080          17.030725  24.900000      126
      # 2     CAM_BACK_RIGHT   5.020080          83.033195  24.900000      126
      # 3          CAM_FRONT   5.020080          50.151564  24.900000      126
      # 4     CAM_FRONT_LEFT   5.020080          33.667718  24.900000      126
      # 5    CAM_FRONT_RIGHT   5.020080          66.649443  24.900000      126
      # 6   CAM_FRONT_ZOOMED   5.020080          50.265691  24.900000      126
      # 7          LIDAR_TOP   5.019934           0.000000  24.900724      126
      # 8           ego_pose  40.440592           0.000000  24.900724     1008
      # 9     sample (annos)   5.019934           0.000000  24.900724      126
      # 10       sample_data  40.278806          49.948567  25.000741     1008
      # ---
      # ---
      # Scene host-a101-lidar0-1241886983298988182-1241887008198992182 7b4640d63a9c62d07a8551d4b430d0acd88eaba8249c843248feb888f4630070
      # Start 2019-05-14 16:36:23       Duration 25.002139806747437 sec
      # Num Annos 4777 (Tracks 173)
      #                Series    Freq Hz  Diff Lidar (msec)   Duration  Support
      # 0            CAM_BACK   5.020080          93.825297  24.900000      126
      # 1       CAM_BACK_LEFT   5.020080          85.205165  24.900000      126
      # 2      CAM_BACK_RIGHT   5.020080          19.099243  24.900000      126
      # 3           CAM_FRONT   5.020080          52.394347  24.900000      126
      # 4      CAM_FRONT_LEFT   5.020080          68.799780  24.900000      126
      # 5     CAM_FRONT_RIGHT   5.020080          35.769232  24.900000      126
      # 6    CAM_FRONT_ZOOMED   5.020080          52.394347  24.900000      126
      # 7    LIDAR_FRONT_LEFT   5.020060           0.000000  24.900101      126
      # 8   LIDAR_FRONT_RIGHT   5.020060           0.000000  24.900101      126
      # 9           LIDAR_TOP   5.020060           0.000000  24.900101      126
      # 10           ego_pose  50.562044           0.000000  24.900101     1260
      # 11     sample (annos)   5.020060           0.000000  24.900101      126
      # 12        sample_data  50.355690          40.748741  25.002140     1260


###############################################################################
### IDatasetUtil Impl

class NuscDSUtil(IDatasetUtil):
  """DSUtil for Nuscenes (only)"""

  FIXTURES = NuscFixtures

  REQUIRED_SUBDIRS = ('maps', 'samples', 'sweeps')

  WARM_CACHE_FOR_VERSIONS = ('v1.0-mini', 'v1.0-trainval', 'v1.0-test')

  @classmethod
  def emplace(cls):
    cls.FIXTURES.maybe_emplace_psegs_ext()
    if not cls.FIXTURES.ROOT.exists():
      
      req_subdirs = '\n        '.join(
        '  * %s' % fname for fname in cls.REQUIRED_SUBDIRS)
      cls.show_md("""
        Due to NuScenes license constraints, you need to manually accept the 
        NuScenes and download the `nuScenes Dataset` at 
        [nuscenes.org](https://www.nuscenes.org/).

        Furthermore, you need to untar / expand the downloaded files due
        to the way that the NuScenes python devkit uses the files.  See the
        `tar -xf` instructions here:
         * For NuScenes (core): https://render.githubusercontent.com/view/ipynb?commit=d8403d35a49f9a5f2b8707129c8af1eff6a8906c&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6e75746f6e6f6d792f6e757363656e65732d6465766b69742f643834303364333561343966396135663262383730373132396338616631656666366138393036632f707974686f6e2d73646b2f7475746f7269616c732f6e757363656e65735f7475746f7269616c2e6970796e62&nwo=nutonomy%2Fnuscenes-devkit&path=python-sdk%2Ftutorials%2Fnuscenes_tutorial.ipynb&repository_id=147720534&repository_type=Repository#Google-Colab-(optional)
         * For NuScenes-lidarseg (optional): https://render.githubusercontent.com/view/ipynb?commit=d8403d35a49f9a5f2b8707129c8af1eff6a8906c&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6e75746f6e6f6d792f6e757363656e65732d6465766b69742f643834303364333561343966396135663262383730373132396338616631656666366138393036632f707974686f6e2d73646b2f7475746f7269616c732f6e757363656e65735f6c696461727365675f7475746f7269616c2e6970796e62&nwo=nutonomy%2Fnuscenes-devkit&path=python-sdk%2Ftutorials%2Fnuscenes_lidarseg_tutorial.ipynb&repository_id=147720534&repository_type=Repository#Google-Colab-(optional)

        Your decompressed dataset directory (the NuSecenes `dataroot`) must
        have at least these subdirectories:

        %s

        Once you've downloaded and unpacked the NuScenes data, we'll need the
        path to that data.  Enter that below, or exit this program.

      """ % (req_subdirs,))
      nusc_root = input(
        "Please enter your NuScenes dataroot path; "
        "PSegs will create a (read-only) symlink to it: ")
      nusc_root = Path(nusc_root.strip())
      assert nusc_root.exists()
      assert nusc_root.is_dir()

      from oarphpy import util as oputil
      oputil.mkdir(str(cls.FIXTURES.ROOT.parent))

      cls.show_md("Symlink: \n%s <- %s" % (nusc_root, cls.FIXTURES.ROOT))
      os.symlink(nusc_root, cls.FIXTURES.ROOT)

      # Make symlink read-only
      import stat
      os.chmod(
        nusc_root,
        stat.S_IREAD|stat.S_IRGRP|stat.S_IROTH,
        follow_symlinks=False)

    cls.show_md("Validating NuScenes data ...")
    subdirs_needed = set(cls.REQUIRED_SUBDIRS)
    subdirs_have = set()
    for entry in cls.FIXTURES.ROOT.iterdir():
      if entry.name in subdirs_needed:
        subdirs_needed.remove(entry.name)
        subdirs_have.add(entry.name)
    
    if subdirs_needed:
      s_have = \
        '\n        '.join('  * %s' % fname for fname in subdirs_have)
      s_needed = \
        '\n        '.join('  * %s' % fname for fname in subdirs_needed)
      cls.show_md("""
        Missing some expected subdirs!

        Found:
        
        %s

        Missing:

        %s
      """ % (s_have, s_needed))
      return False
    
    cls.show_md("... core NuScenes data found!")

    cls.show_md("Warming NuScenes caches ...")
    class NuScenesWithMyFixtures(PSegsNuScenes):
      FIXTURES = cls.FIXTURES
    NuScenesWithMyFixtures.maybe_warm_caches()
    cls.show_md("... done warming caches.")
    return True

  @classmethod
  def test(cls):
    from oarphpy import util as oputil
    oputil.run_cmd("cd %s && pytest -s -vvv -k test_nuscenes" % C.PS_ROOT)
    return True

  @classmethod
  def build_table(cls):
    assert False, "TODO"


class LyftDSUtil(IDatasetUtil):
  """DSUtil for Lyft (only)"""

  FIXTURES = LyftFixtures

  NUSC_REQUIRED_SUBDIRS = ('maps', 'images', 'lidar')




















import pytest

###############################################################################
### Test Utils

skip_if_no_nusc_mini = pytest.mark.skipif(
  not NuscFixtures.version_exists('v1.0-mini'),
  reason="Requires NuScenes v1.0-trainval")

skip_if_no_nusc_trainval = pytest.mark.skipif(
  not NuscFixtures.version_exists('v1.0-trainval'),
  reason="Requires NuScenes v1.0-mini")

###############################################################################
### Test NuScenes

@skip_if_no_nusc_mini
def test_nuscenes_mini_stats():
  nusc = PSegsNuScenes(version='v1.0-mini')

  TABLE_TO_EXPECTED_LENGTH = {
    'yay': 100,
  }

  actual = nusc.get_table_to_length()

  assert actual == TABLE_TO_EXPECTED_LENGTH


@skip_if_no_nusc_trainval
def test_nuscenes_trainval_stats():
  nusc = PSegsNuScenes(version='v1.0-trainval')

  TABLE_TO_EXPECTED_LENGTH = {
    'yay': 100,
  }

  actual = nusc.get_table_to_length()

  assert actual == TABLE_TO_EXPECTED_LENGTH


def test_nuscenes_yay():

  nusc = PSegsNuScenes(
    version='v1.0-trainval',
    dataroot='/outer_root//media/seagates-ext4/au_datas/nuscenes_root/')

  from pprint import pprint
  pprint(nusc.get_all_sensors())
  pprint(nusc.get_all_classes())

  pprint(('list_lidarseg_categories', nusc.list_lidarseg_categories(sort_by='count')))
  pprint(('lidarseg_idx2name_mapping', nusc.lidarseg_idx2name_mapping))

