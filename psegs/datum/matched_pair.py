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

import typing

import attr
import numpy as np

from oarphpy.spark import CloudpickeledCallable

from psegs.datum.camera_image import CameraImage
from psegs.datum.point_cloud import PointCloud
from psegs.datum.transform import Transform
from psegs.util import plotting as pspl


@attr.s(slots=True, eq=False, weakref_slot=False)
class MatchedPair(object):
  """A pair of `CameraImages` with pixelwise matches"""

  matcher_name = attr.ib(type=str, default='')
  """str: Name of the match source, e.g. SIFT_matches"""

  timestamp = attr.ib(type=int, default=0)
  """int: Timestamp associated with this matched pair; use the timestamp
  of `img1` or `img2` or the wall time of matching."""

  img1 = attr.ib(default=None, type=CameraImage)
  """CameraImage: The first (left, source) image"""

  img2 = attr.ib(default=None, type=CameraImage)
  """CameraImage: The second (right, target) image"""

  matches_array = attr.ib(type=np.ndarray, default=None)
  """numpy.ndarray: Matches as an n-by-d matrix (where `d` is *at least*
  4, i.e. (img1 x, img1 y, img2 x, img2 y))."""

  matches_factory = attr.ib(
    type=CloudpickeledCallable,
    converter=CloudpickeledCallable,
    default=None)
  """CloudpickeledCallable: A serializable factory function that emits the
  values for `matches_array` (if a realized array cannot be provided)"""

  matches_colnames = attr.ib(default=['x1', 'y1', 'x2', 'y2'])
  """List[str]: Semantic names for the columns (or dimensions / attributes)
  of the `matches_array`.  Typically matches are just 2D point pairs, but
  match data can include confidence, occlusion state, track ID, and/or 
  other data."""

  extra = attr.ib(default={}, type=typing.Dict[str, str])
  """Dict[str, str]: A map for adhoc extra context"""

  def get_matches(self):
    if self.matches_array is not None:
      return self.matches_array
    elif self.matches_factory != CloudpickeledCallable.empty():
      return self.matches_factory()
    else:
      raise ValueError("No matches data!")

  def get_col_idx(self, colname):
    for i in range(len(self.matches_colnames)):
      if self.matches_colnames[i] == colname:
        return i
    raise ValueError(
      "Colname %s not found in %s" % (colname, self.matches_colnames))

  def get_x1y1x2y2_axes(self):
    return [
      self.get_col_idx('x1'),
      self.get_col_idx('y1'),
      self.get_col_idx('x2'),
      self.get_col_idx('y2'),
    ]
  
  def get_other_axes(self):
    x1y1x2y2c = set(['x1', 'y1', 'x2', 'y2'])
    all_c = set(self.matches_colnames)
    other_names = sorted(list(all_c - x1y1x2y2c))
    other_idx = [self.get_col_idx(n) for n in other_names]
    return other_names, other_idx

  def get_x1y1x2y2(self):
    matches = self.get_matches()
    x1y1x2y2 = matches[:, self.get_x1y1x2y2_axes()]
    return x1y1x2y2

  def get_x1y1x2y2_extra(self):
    matches = self.get_matches()
    other_names, other_idx = self.get_other_axes()
    cols = self.get_x1y1x2y2_axes() + other_idx
    x1y1x2y2_extra = matches[:, cols]
    return x1y1x2y2_extra

  def get_debug_line_image(self):
    return pspl.create_matches_debug_line_image(
              self.img1.image,
              self.img2.image,
              self.get_matches())

  def get_point_cloud_in_world_frame(self):

    import cv2

    P_1 = self.img1.get_P()
    P_2 = self.img2.get_P()
    matches = self.get_matches()

    x1c, y1c, x2c, y2c = self.get_x1y1x2y2_axes()
    other_names, other_idx = self.get_other_axes()
    uv_1 = matches[:, [x1c, y1c]]
    uv_2 = matches[:, [x2c, y2c]]

    if uv_1.shape[0] > 0:
      xyzh = cv2.triangulatePoints(P_1, P_2, uv_1.T, uv_2.T)
      xyz = xyzh.T.copy()
      xyz = xyz[:, :3] / xyz[:, (-1,)]
    else:
      xyz = np.zeros((0, 3), dtype=np.float64)

    other_vals = matches[:, other_idx]
    cloud = np.hstack([xyz, other_vals])
    return PointCloud(
              sensor_name=self.matcher_name,
              ego_to_sensor=Transform(
                src_frame='ego', dest_frame=self.matcher_name),
              ego_pose=Transform(
                src_frame='world', dest_frame='ego'),
              timestamp=self.timestamp,
              cloud=cloud,
              cloud_colnames = ['x', 'y', 'z'] + other_names)


def create_stereo_rect_pair_debug_view_html(
      ci_left,
      ci_rights=[],
      lr_matches=[],
      mp_uris=[],
      rect_image_wh=None,
      image_viz_max_size=-1,
      embed_images_root_path=None,
      embed_opencv_js=True):
  
  import cv2

  if lr_matches:
    assert len(ci_rights) == len(lr_matches), (
      f"{len(ci_rights)} != len(lr_matches)")

    if mp_uris:
      assert len(mp_uris) == len(lr_matches), (
        f"{len(mp_uris)} != len(lr_matches)")

  rightImageIdToInfo_entries = []
  for i in range(len(ci_rights)):
    ci_right = ci_rights[i]

    rect_image_wh = None or (ci_left.width, ci_right.height)
    matches = None if not lr_matches else lr_matches[i]
    mp_uri = '(unknown)' if not mp_uris else mp_uris[i]

    K1 = ci_left.K
    RT1 = ci_left.get_world_to_sensor().get_transformation_matrix(homogeneous=True)

    K2 = ci_right.K
    RT2 = ci_right.get_world_to_sensor().get_transformation_matrix(homogeneous=True)

    sRT1_inv = np.eye(4, 4)
    sRT1_inv[:3, :3] = RT2[:3, :3].T
    sRT1_inv[:3, 3] = RT2[:3, :3].T.dot(-1 * RT2[:3, 3])

    RT_diff = sRT1_inv @ RT1

    distCoeffs1 = ci_left.get_opencv_distcoeffs()
    if distCoeffs1 is None:
      distCoeffs1 = np.array([0., 0., 0., 0.,])
    distCoeffs2 = ci_right.get_opencv_distcoeffs()
    if distCoeffs2 is None:
      distCoeffs2 = np.array([0., 0., 0., 0.,])
    rect_output = cv2.stereoRectify(
                    K1, distCoeffs1, K2, distCoeffs2,
                    (ci_left.width, ci_right.height),
                    RT_diff[:3, :3], RT_diff[:3, 3],
                    newImageSize=rect_image_wh)
    sR1, sR2, sP1, sP2, sQ, sroi1, sroi2 = rect_output

    right_image_uri = "todo"

    def _mat2jsstr(name, mat):
      js = f"""(
        new cv.matFromArray(
          {mat.shape[0]}, // rows
          {mat.shape[1]}, // cols
          cv.CV_32F,
          {mat.flatten().tolist()}
        )
      )
      """
      return js
    
    rightImageIdToInfo_entries.append(f"""
      "{i}": // rightImageId
        {{
          "rightImageId": {i},
          "rightImageUri": "{right_image_uri}",
          "K1": {_mat2jsstr(K1)},
          "K2": {_mat2jsstr(K2)},
          "sR1": {_mat2jsstr(sR1)},
          "sR2": {_mat2jsstr(sR2)},
          "sP1": {_mat2jsstr(sP1)},
          "sP2": {_mat2jsstr(sP2)},
          "sroi1": {list(sroi1)},
          "sroi2": {list(sroi2)},
          "distCoeffs1": {_mat2jsstr(distCoeffs1)},
          "distCoeffs2": {_mat2jsstr(distCoeffs2)},
          "newImageSize": new cv.Size({rect_image_wh[0]}, {rect_image_wh[1]}),
          
          "mpURI": "{str(mp_uri)}"
        }}
    """)

  final_html = f"""

  <table>
    <tr>
      <td><canvas id="stereoRectVizLeft"></td>
      <td><canvas id="stereoRectVizRight"></td>
    </tr>
  </table>

  <script 
    async
    src="https://docs.opencv.org/3.4/opencv.js"
    type="text/javascript">
  </script>
  <script type="text/javascript">

    // BEGIN utils

    showRightImageId = function(i) {{
      console.log("Showing right image " + i);

      info = rightImageIdToInfo[i];
      let K1 = info["K1"];
      let K2 = info["K2"];
      let sR1 = info["sR1"];
      let sR2 = info["sR2"];
      let sP1 = info["sP1"];
      let sP2 = info["sP2"];
      let newImageSize = info["newImageSize"];

      let leftMap1 = new cv.Mat();
      let leftMap2 = new cv.Mat();
      let rightMap1 = new cv.Mat();
      let rightMap2 = new cv.Mat();
      cv.initUndistortRectifyMap(
        K1, distCoeffs1, sR1, sP1, newImageSize, cv.CV_32FC1,
        leftMap1, leftMap2);
      cv.initUndistortRectifyMap(
        K2, distCoeffs2, sR2, sP2, newImageSize, cv.CV_32FC1,
        rightMap1, rightMap2);

      let leftOrigImg = cv.imread(document.getElementById("inputLeft"));
      let rightOrigImg = cv.imread(document.getElementById("inputRight"));

      let leftRectImg = new cv.Mat();
      cv.remap(leftOrigImg, leftRectImg, leftMap1, leftMap2, cv.INTER_LANCZOS4);
      let rightRectImg = new cv.Mat();
      cv.remap(rightOrigImg, rightRectImg, rightMap1, rightMap2, cv.INTER_LANCZOS4);
      
      cv.imshow('stereoRectVizLeft', leftRectImg);
      cv.imshow('stereoRectVizRight', rightRectImg);

    }};

    // END utils

    // BEGIN embedded rect variables
    
    rightImageIdToInfo = {{
      { ",".join(rightImageIdToInfo_entries) }
    }};

    // END embedded rect variables

    // BEGIN opencv loaded hook
    currentRightImageId = 0;

    var Module = {{
      // https://emscripten.org/docs/api_reference/module.html#Module.onRuntimeInitialized
      onRuntimeInitialized() {{
        console.log("StereoRectViz Setup");



        console.log("StereoRectViz Setup Complete");
      }}
    }};
    // END opencv loaded hook

  </script>
  
  """

  return final_html

