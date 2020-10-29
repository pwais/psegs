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
import os
import pickle
import shelve
from pathlib import Path

import numpy as np

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
    print('todo maybe_emplace_psegs_ext')
    print('todo maybe_emplace_psegs_ext')
    print('todo maybe_emplace_psegs_ext')
    print('todo maybe_emplace_psegs_ext') #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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


# nuscenes is a soft dependency
try:
  from nuscenes.nuscenes import NuScenes
  BASE = NuScenes
except ImportError:
  BASE = object

class PSegsNuScenes(BASE):
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

  DATASET = 'nuscenes'

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
    self.dataroot = kwargs.get('dataroot')
    self.verbose = kwargs.get('verbose', True)
      # Base ctor sets these members, but we'll do it here so that
      # path-resolving superclass methods below work properly
    
    # If needed, default to PSegs-configured dataroot
    if not (self.dataroot and os.path.exists(self.dataroot)):
      if self.FIXTURES.ROOT.exists():
        self.dataroot = str(self.FIXTURES.ROOT)
        kwargs['dataroot'] = self.dataroot
      else:
        raise FileNotFoundError(
          "Could not find provided dataroot %s nor default "
          "PSegs dataroot %s" % (self.dataroot, self.FIXTURES.ROOT))

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
        
        d = shelve.open(str(cache_path), protocol=pickle.HIGHEST_PROTOCOL)
        rows = getattr(nusc, table_name) # E.g. self.sample_data
        d.update((r['token'], r) for r in rows)
        d.close()
      
      # This cache helps speed up full-scene lookups by 10x
      util.log.info("Building scene name -> sample_data token cache ...")
      scene_name_to_sd_token = self._build_scene_name_to_sd_token(nusc)
      cache_path = self._get_cache_path('scene_name_to_sd_token')
      d = shelve.open(str(cache_path), protocol=pickle.HIGHEST_PROTOCOL)
      d.update(scene_name_to_sd_token)
      d.close()

      util.log.info("... done.")
      del nusc # Free several GB memory

    super(PSegsNuScenes, self).__init__(**kwargs)

  def _get_table(self, table_name):
    attr = '_cached_' + table_name
    if not hasattr(self, attr):
      cache_path = self._get_cache_path(table_name)
      if self.verbose:
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

  @classmethod
  def _build_scene_name_to_sd_token(cls, nusc):
    scene_name_to_sd_token = {}

    sample_to_scene = {}
    for sample in nusc.sample:
      scene = nusc.get('scene', sample['scene_token'])
      sample_to_scene[sample['token']] = scene['token']

    for sd in nusc.sample_data:
      scene_tok = sample_to_scene[sd['sample_token']]
      cur_scene = nusc.get('scene', scene_tok)['name']
      scene_name_to_sd_token.setdefault(cur_scene, [])
      scene_name_to_sd_token[cur_scene].append(sd['token'])
    
    # For NuScenes trainval-1.0, this is about 400MB of memory
    return scene_name_to_sd_token

  def iter_sample_data_for_scene(self, scene_name):
    sd_tokens = self._get_table('scene_name_to_sd_token')[scene_name]
    for sd_token in sd_tokens:
      yield self.get('sample_data', sd_token)


  #### PSegs Adhoc Utils

  @classmethod
  def get_split_for_scene(cls, scene):
    if not hasattr(cls, '_scene_to_split'):

      ## NuScenes
      from nuscenes.utils.splits import create_splits_scenes
      split_to_scenes = create_splits_scenes()

      scene_to_split = {}
      for split, scenes in split_to_scenes.items():
        # Ignore mini splits because they duplicate train/val
        if 'mini' not in split:
          for s in scenes:
            scene_to_split[s] = split
      
      # ## Lyft
      # # NB: Lyft Level 5 does not have a proper train/test split of labeled
      # # data, so we induced an arbitrary one for now.
      # from au.fixtures.datasets import lyft_level_5_constants as lyft_consts
      # AU_SCENES = lyft_consts.AU_VAL_SCENES + lyft_consts.AU_TRAIN_SCENES
      # assert sorted(AU_SCENES) == sorted(lyft_consts.TRAIN_SCENES)
      # scene_to_split.update(
      #   dict((s, 'train') for s in  lyft_consts.AU_TRAIN_SCENES))
      # scene_to_split.update(
      #   dict((s, 'val') for s in    lyft_consts.AU_VAL_SCENES))
      # scene_to_split.update(
      #   dict((s, 'test') for s in   lyft_consts.TEST_SCENES))

      ## Done!
      cls._scene_to_split = scene_to_split

    return cls._scene_to_split[scene]

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











from psegs import datum
from psegs.table.sd_table import StampedDatumTableBase

def transform_from_record(rec, src_frame='', dest_frame=''):
  from pyquaternion import Quaternion
  return datum.Transform(
          rotation=Quaternion(rec['rotation']).rotation_matrix,
          translation=np.array(rec['translation']),
          src_frame=src_frame,
          dest_frame=dest_frame)

def get_camera_normal(K, extrinsic):
    """FMI see au.fixtures.datasets.auargoverse.get_camera_normal()"""

    # Build P
    # P = |K 0| * | R |T|
    #             |000 1|
    K_h = np.zeros((3, 4))
    K_h[:3, :3] = K
    P = K_h.dot(extrinsic)

    # Zisserman pg 161 The principal axis vector.
    # P = [M | p4]; M = |..|
    #                   |m3|
    # pv = det(M) * m3
    pv = np.linalg.det(P[:3,:3]) * P[2,:3].T
    pv_hat = pv / np.linalg.norm(pv)
    return pv_hat

def to_nanostamp(timestamp_micros):
  return int(timestamp_micros) * 1000

class NuscStampedDatumTableBase(StampedDatumTableBase):

  API_CLS = PSegsNuScenes

  NUSC_VERSION = 'v1.0-trainval' # E.g. v1.0-mini, v1.0-trainval, v1.0-test

  SENSORS_KEYFRAMES_ONLY = False
    # NuScenes: If enabled, throttles sensor data to about 2Hz, in tune with
    #   samples; if disabled, samples at full res.
    # Lyft Level 5: all sensor data is key frames.
    # FMI see print_sensor_sample_rates() above.
  
  LABELS_KEYFRAMES_ONLY = True
    # If enabled, samples only raw annotations.  If disabled, will motion-
    # correct cuboids to every sensor reading.


  ## Subclass API
  
  @classmethod
  def _create_datum_rdds(cls, spark, existing_uri_df=None, only_segments=None):
    from psegs.spark import Spark
    from oarphpy.spark import cluster_cpu_count

    PARTITIONS_PER_SEGMENT = 7
    TASKS_PER_RDD = cluster_cpu_count(spark)

    # Are we resuming? We need to filter existing datums *before* computing
    # them for resume mode to save time.
    existing_ids = []
    def to_nusc_datum_id(row):
      return (
        row.dataset,
        row.split,
        row.segment_id,
        row.topic,
        row.timestamp)
    if existing_uri_df:
      existing_ids = set(existing_uri_df.rdd.map(to_nusc_datum_id).collect())
      util.log.info("Resume mode: have %s datums" % len(existing_ids))

    segment_ids = cls.get_segment_ids()
    if only_segments:
      segment_ids = [s for s in segment_ids if s in only_segments]

    datum_rdds = []
    iter_tasks = itertools.chain.from_iterable(
      ((segment_id, p) for p in range(PARTITIONS_PER_SEGMENT))
      for segment_id in segment_ids)
    for task_chunk in oputil.ichunked(iter_tasks, TASKS_PER_RDD):
      task_rdd = spark.sparkContext.parallelize(task_chunk)

      def gen_partition_datums(task):
        import time
        total_sd_time = 0
        segment_id, partition = task
        for i, uri in enumerate(cls.iter_uris_for_segment(segment_id)):
          if to_nusc_datum_id(uri) in existing_ids:
            continue
          if (i % PARTITIONS_PER_SEGMENT) == partition:
            start = time.time()
            yield cls.create_stamped_datum(uri)
            total_sd_time += time.time() - start
        print('total_sd_time', total_sd_time)
        
      datum_rdd = task_rdd.flatMap(gen_partition_datums)
      datum_rdds.append(datum_rdd)
    return datum_rdds

  
  ## Public API

  @classmethod
  def get_nusc(cls):
    if not hasattr(cls, '_nusc'):
      cls._nusc = cls.API_CLS(version=cls.NUSC_VERSION, verbose=False)
    return cls._nusc
  
  @classmethod
  def get_segment_ids(cls):
    nusc = cls.get_nusc()
    return sorted(s['name'] for s in nusc.scene)

  @classmethod
  def iter_uris_for_segment(cls, segment_id):
    import time
    start = time.time()
    nusc = cls.get_nusc()

    scene_split = nusc.get_split_for_scene(segment_id)

    ## Get sensor data and ego pose
    for sd in nusc.iter_sample_data_for_scene(segment_id):
      # Use these for creating frames based upon NuScenes / Lyft groupings
      sample_token = sd['sample_token']
      is_key_frame = str(sd['is_key_frame'])

      # Note all poses
      # DANGER: The timestamps of the pose records in Lyft Level 5 might be
      # broken, but the sensor timestamps look corect.
      # https://github.com/lyft/nuscenes-devkit/issues/73
      yield datum.URI(
              dataset=nusc.DATASET,
              split=scene_split,
              segment_id=segment_id,
              timestamp=to_nanostamp(sd['timestamp']),
              topic='ego_pose',
              extra={
                'nuscenes-token': 'ego_pose|' + sd['ego_pose_token'],
                'nuscenes-sample-token': sample_token,
                'nuscenes-is-keyframe': is_key_frame,
              })

      # Maybe skip the sensor data if we're only doing keyframes
      if cls.SENSORS_KEYFRAMES_ONLY:
        if sd['sensor_modality'] and not sd['is_key_frame']:
          continue

      yield datum.URI(
              dataset=nusc.DATASET,
              split=scene_split,
              segment_id=segment_id,
              timestamp=to_nanostamp(sd['timestamp']),
              topic=sd['sensor_modality'] + '|' + sd['channel'],
              extra={
                'nuscenes-token': 'sample_data|' + sd['token'],
                'nuscenes-sample-token': sample_token,
                'nuscenes-is-keyframe': is_key_frame,
              })

      # Get labels (non-keyframes; interpolated one per track)
      if not cls.LABELS_KEYFRAMES_ONLY:
        yield datum.URI(
                dataset=nusc.DATASET,
                split=scene_split,
                segment_id=segment_id,
                timestamp=to_nanostamp(row['timestamp']),
                topic='labels|cuboids',
                extra={
                  'nuscenes-token': 'sample_data|' + sd['token'],
                  'nuscenes-sample-token': sample_token,
                  'nuscenes-is-keyframe': is_key_frame,
                })

    ## Get labels (keyframes only)
    if cls.LABELS_KEYFRAMES_ONLY:

      # Get annos for *only* samples, which are keyframes
      scene_tokens = [
        s['token'] for s in nusc.scene if s['name'] == segment_id]
      assert scene_tokens
      scene_token = scene_tokens[0]

      scene_samples = [
        s for s in nusc.sample if s['scene_token'] == scene_token
      ]

      for sample in scene_samples:
        sample_token = sample['token']
        for channel, sample_data_token in sample['data'].items():
          sd = nusc.get('sample_data', sample_data_token)
          yield datum.URI(
                  dataset=nusc.DATASET,
                  split=scene_split,
                  segment_id=segment_id,
                  timestamp=to_nanostamp(sd['timestamp']),
                  topic='labels|cuboids',
                  extra={
                    'nuscenes-token': 'sample_data|' + sd['token'],
                    'nuscenes-sample-token': sample_token,
                    'nuscenes-is-keyframe': 'True',
                  })
    
    print('iter_uris_for_segment', time.time() - start)

  @classmethod
  def create_stamped_datum(cls, uri):
    if uri.topic.startswith('camera'):
      sample_data = cls.__get_row(uri)
      return cls.__create_camera_image(uri, sample_data)
    elif uri.topic.startswith('lidar') or uri.topic.startswith('radar'):
      sample_data = cls.__get_row(uri)
      return cls.__create_point_cloud(uri, sample_data)
    elif uri.topic == 'ego_pose':
      pose_record = cls.__get_row(uri)
      return cls.__create_ego_pose(uri, pose_record)
    elif uri.topic == 'labels|cuboids':
      sample_data = cls.__get_row(uri)
      # nusc = cls.get_nusc()
      # best_sd, diff_ns = nusc.get_nearest_sample_data(
      #                           uri.segment_id,
      #                           uri.timestamp)
      # assert best_sd
      # assert diff_ns < .01 * 1e9
      return cls.__create_cuboids_in_ego(uri, sample_data['token'])
    else:
      raise ValueError(uri)


  ## Support

  @classmethod
  def __get_row(cls, uri):
    if 'nuscenes-token' in uri.extra:
      record = uri.extra['nuscenes-token']
      table, token = record.split('|')
      nusc = cls.get_nusc()
      return nusc.get(table, token)
    raise ValueError
    # nusc = cls.get_nusc()
    # return nusc.get_row(uri.segment_id, uri.timestamp, uri.topic)
  
  @classmethod
  def __create_camera_image(cls, uri, sample_data):
    nusc = cls.get_nusc()

    camera_token = sample_data['token']
    cs_record = nusc.get(
      'calibrated_sensor', sample_data['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sample_data['ego_pose_token'])

    data_path, _, cam_intrinsic = nusc.get_sample_data(camera_token)
      # Ignore box_list, we'll get boxes in ego frame later
    
    w, h = sample_data['width'], sample_data['height']
    # viewport = uri.get_viewport()~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # if not viewport:
    #   from au.fixtures.datasets import common
    #   viewport = datum.BBox2D.of_size(w, h)

    timestamp = sample_data['timestamp']

    ego_from_cam = transform_from_record(
                      cs_record,
                      dest_frame='ego',
                      src_frame=sample_data['channel'])
    cam_from_ego = ego_from_cam.get_inverse()
    RT_h = cam_from_ego.get_transformation_matrix(homogeneous=True)
    principal_axis_in_ego = get_camera_normal(cam_intrinsic, RT_h)

    ego_pose = transform_from_record(
                      pose_record,
                      dest_frame='city',
                      src_frame='ego')  ## TODO check nusc, lyft ok ~~~~~~~~~~~~~
    ci = datum.CameraImage(
            sensor_name=sample_data['channel'],
            image_jpeg=bytearray(open(data_path, 'rb').read()),
            height=h,
            width=w,
            # viewport=viewport,~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            timestamp=to_nanostamp(timestamp),
            ego_pose=ego_pose,
            ego_to_sensor=ego_from_cam,
            K=cam_intrinsic,
            # principal_axis_in_ego=principal_axis_in_ego,~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    )
    return datum.StampedDatum(uri=uri, camera_image=ci)
  
  @classmethod
  def __create_point_cloud(cls, uri, sample_data):
    # Based upon nuscenes.py#map_pointcloud_to_image()

    from pyquaternion import Quaternion
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.data_classes import RadarPointCloud

    nusc = cls.get_nusc()

    target_pose_token = sample_data['ego_pose_token']

    pcl_path = os.path.join(nusc.dataroot, sample_data['filename'])
    if 'host-a011_lidar1_1233090652702363606.bin' in pcl_path:
      util.log.warn('Kaggle download has a broken file') #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if sample_data['sensor_modality'] == 'lidar':
      pc = LidarPointCloud.from_file(pcl_path)
    else:
      pc = RadarPointCloud.from_file(pcl_path)

    # Step 1: Points live in the point sensor frame.  First transform to
    # world frame:
    # 1a transform to ego
    # First step: transform the point-cloud to the ego vehicle frame for the
    # timestamp of the sweep.
    cs_record = nusc.get(
                  'calibrated_sensor', sample_data['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # 1b transform to the global frame.
    poserecord = nusc.get('ego_pose', sample_data['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Step 2: Send points into the ego frame at the target timestamp
    poserecord = nusc.get('ego_pose', target_pose_token)
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    n_xyz = pc.points[:3, :].T
      # Throw out intensity (lidar) and ... other stuff (radar) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    ego_pose = transform_from_record(
                      nusc.get('ego_pose', sample_data['ego_pose_token']),
                      dest_frame='city',
                      src_frame='ego')
    ego_to_sensor = transform_from_record(
                      cs_record,
                      src_frame='ego',
                      dest_frame=sample_data['channel']) # ?????????????????????????

    motion_corrected = (sample_data['ego_pose_token'] != target_pose_token)

    pc = datum.PointCloud(
            sensor_name=sample_data['channel'],
            timestamp=to_nanostamp(sample_data['timestamp']),
            cloud=n_xyz.astype(np.float32),
            # motion_corrected=motion_corrected,~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ego_to_sensor=ego_to_sensor,
            ego_pose=ego_pose,
    )
    return datum.StampedDatum(uri=uri, point_cloud=pc)
  
  @classmethod
  def __create_cuboids_in_ego(cls, uri, sample_data_token):
    nusc = cls.get_nusc()

    # NB: This helper always does motion correction (interpolation) unless
    # `sample_data_token` refers to a keyframe.
    boxes = nusc.get_boxes(sample_data_token)
  
    # Boxes are in world frame.  Move all to ego frame.
    from pyquaternion import Quaternion
    sd_record = nusc.get('sample_data', sample_data_token)
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    for box in boxes:
      # Move box to ego vehicle coord system
      box.translate(-np.array(pose_record['translation']))
      box.rotate(Quaternion(pose_record['rotation']).inverse)

    ego_pose = transform_from_record(
      pose_record, dest_frame='city', src_frame='ego')
    # from au.fixtures.datasets.av import NUSCENES_CATEGORY_TO_AU_AV_CATEGORY ~~~~~~~~~~~~~~~~~~~
    cuboids = []
    for box in boxes:
      cuboid = datum.Cuboid()

      # Core
      sample_anno = nusc.get('sample_annotation', box.token)
      cuboid.track_id = \
        'nuscenes_instance_token:' + sample_anno['instance_token']
      cuboid.category_name = box.name
      cuboid.timestamp = to_nanostamp(sd_record['timestamp'])
      
      cuboid.ps_category = 'todo' # ~~~~~~~~~~~~~~~~~~~~~~~`NUSCENES_CATEGORY_TO_AU_AV_CATEGORY[box.name]
      
      # Try to give bikes riders
      # NB: In Lyft Level 5, they appear to *not* label bikes without riders
      attribs = [
        nusc.get('attribute', attrib_token)['name']
        for attrib_token in sample_anno['attribute_tokens']
      ]
      if 'cycle.with_rider' in attribs:
        if cuboid.ps_category == 'bike_no_rider':
          cuboid.ps_category = 'bike_with_rider'
        elif cuboid.ps_category == 'motorcycle_no_rider':
          cuboid.ps_category = 'motorcycle_with_rider'
        else:
          # raise ValueError(
            # "Don't know how to give a rider to %s %s" % (cuboid, attribs))
          print("""TODO "Don't know how to give a rider """)#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      cuboid.extra = {
        'nuscenes_token': box.token,
        'nuscenes_attribs': '|'.join(attribs),
      }

      # Points
      # box3d = box.corners().T

      # cuboid.motion_corrected = (not sd_record['is_key_frame'])~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
      # cuboid.distance_meters = np.min(np.linalg.norm(cuboid.box3d, axis=-1))
      
      # Pose
      cuboid.width_meters = float(box.wlh[0])
      cuboid.length_meters = float(box.wlh[1])
      cuboid.height_meters = float(box.wlh[2])

      cuboid.obj_from_ego = datum.Transform(
          rotation=box.orientation.rotation_matrix,
          translation=box.center,
          src_frame='ego',
          dest_frame='obj')
      cuboid.ego_pose = ego_pose
      cuboids.append(cuboid)
    return datum.StampedDatum(uri=uri, cuboids=cuboids)

  @classmethod
  def __create_ego_pose(cls, uri, pose_record):
    nusc = cls.get_nusc()
    pose_record = nusc.get('ego_pose', pose_record['token'])
    ego_pose = transform_from_record(
                      pose_record,
                      dest_frame='city',
                      src_frame='ego')
    return datum.StampedDatum(uri=uri, transform=ego_pose)
      # Note: we use the timestamp in `uri` versus the one in the pose record.
      # DANGER: The timestamps of the pose records in Lyft Level 5 might be
      # broken, but the sensor timestamps look corect.
      # https://github.com/lyft/nuscenes-devkit/issues/73













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
         * [For NuScenes (core)](https://render.githubusercontent.com/view/ipynb?commit=d8403d35a49f9a5f2b8707129c8af1eff6a8906c&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6e75746f6e6f6d792f6e757363656e65732d6465766b69742f643834303364333561343966396135663262383730373132396338616631656666366138393036632f707974686f6e2d73646b2f7475746f7269616c732f6e757363656e65735f7475746f7269616c2e6970796e62&nwo=nutonomy%%2Fnuscenes-devkit&path=python-sdk%%2Ftutorials%%2Fnuscenes_tutorial.ipynb&repository_id=147720534&repository_type=Repository#Google-Colab-(optional))
         * [For NuScenes-lidarseg (optional)](https://render.githubusercontent.com/view/ipynb?commit=d8403d35a49f9a5f2b8707129c8af1eff6a8906c&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6e75746f6e6f6d792f6e757363656e65732d6465766b69742f643834303364333561343966396135663262383730373132396338616631656666366138393036632f707974686f6e2d73646b2f7475746f7269616c732f6e757363656e65735f6c696461727365675f7475746f7269616c2e6970796e62&nwo=nutonomy%%2Fnuscenes-devkit&path=python-sdk%%2Ftutorials%%2Fnuscenes_lidarseg_tutorial.ipynb&repository_id=147720534&repository_type=Repository#Google-Colab-(optional))

        Your decompressed dataset directory (the NuSecenes `dataroot`) must
        have at least these subdirectories:

        %s

        Once you've downloaded and unpacked the NuScenes data, we'll need the
        path to that data.  Enter that below, or exit this program.

      """ % req_subdirs)
      nusc_root = input(
        "Please enter your NuScenes dataroot path; "
        "PSegs will create a (read-only) symlink to it: ")
      nusc_root = Path(nusc_root.strip())
      assert nusc_root.exists()
      assert nusc_root.is_dir()

      oputil.mkdir(str(cls.FIXTURES.ROOT.parent))

      cls.show_md("Symlink: \n%s <- %s" % (nusc_root, cls.FIXTURES.ROOT))
      os.symlink(nusc_root, cls.FIXTURES.ROOT)

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
    return True
    assert False, "TODO"


class LyftDSUtil(IDatasetUtil):
  """DSUtil for Lyft (only)"""

  FIXTURES = LyftFixtures

  REQUIRED_SUBDIRS = ('maps', 'images', 'lidar')


# 2020-10-29 07:43:22,447 oarph 965634 : Progress for                             
# save_df_thunks [Pid:965634 Id:140090221242784]
# -----------------------  -------------------------------------------------------------------------------
# Thruput
# N thru                   3 (of 496)
# N chunks                 3
# Total time               8 minutes and 51.56 seconds
# Total thru               2.34 GB
# Rate                     4.4 MB / sec
# Hz                       0.0056438171817696624
# Progress
# Percent Complete         0.6048387096774194
# Est. Time To Completion  1 day, 15 minutes and 52.23 seconds
# Latency (per chunk)
# Avg                      2 minutes, 57 seconds, 185 milliseconds, 44 microseconds and 765.47 nanoseconds
# p50                      3 minutes, 1 second, 11 milliseconds, 476 microseconds and 755.14 nanoseconds
# p95                      3 minutes, 7 seconds, 486 milliseconds, 417 microseconds and 293.55 nanoseconds
# p99                      3 minutes, 8 seconds, 61 milliseconds, 967 microseconds and 563.63 nanoseconds
# -----------------------  -------------------------------------------------------------------------------

# with faster loops now
# save_df_thunks [Pid:1588334 Id:140225493356208]
# -----------------------  -------------------------------------------------------------------------------
# Thruput
# N thru                   3 (of 496)
# N chunks                 3
# Total time               3 minutes and 35.83 seconds
# Total thru               2.34 GB
# Rate                     10.82 MB / sec
# Hz                       0.013900148032916951
# Progress
# Percent Complete         0.6048387096774194
# Est. Time To Completion  9 hours, 51 minutes and 7.25 seconds
# Latency (per chunk)
# Avg                      1 minute, 11 seconds, 941 milliseconds, 679 microseconds and 875.06 nanoseconds
# p50                      1 minute, 15 seconds, 234 milliseconds, 795 microseconds and 331.95 nanoseconds
# p95                      1 minute, 16 seconds, 7 milliseconds, 751 microseconds and 536.37 nanoseconds
# p99                      1 minute, 16 seconds, 76 milliseconds, 458 microseconds and 754.54 nanoseconds
# -----------------------  -------------------------------------------------------------------------------