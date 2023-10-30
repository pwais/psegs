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
from pathlib import Path

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
      max_matches_per_pair=10_000,
      embed_images_root_path='stereo_rect_pair_viz_images',
      embed_opencv_js=True):
  
  import json
  
  import attr
  import cv2

  from oarphpy.plotting import hash_to_rbg

  if lr_matches:
    assert len(ci_rights) == len(lr_matches), (
      f"{len(ci_rights)} != len(lr_matches)")

    if mp_uris:
      assert len(mp_uris) == len(lr_matches), (
        f"{len(mp_uris)} != len(lr_matches)")

  embed_images_root_path = Path(embed_images_root_path)
  embed_images_root_path.mkdir(parents=True, exist_ok=True)

  rightImageIdToInfo_entries = []
  default_right_image_uri = ""
  for right_id in range(len(ci_rights)):
    ci_right = ci_rights[right_id]

    rect_image_wh = None or (ci_left.width, ci_right.height)
    mp_uri = '(unknown)' if not mp_uris else mp_uris[right_id]

    match_left_xy = []
    match_right_xy = []
    match_color = []
    matches = None if not lr_matches else lr_matches[right_id]
    if matches is not None:
      rng = np.random.RandomState(1337)
      n = min(max_matches_per_pair, matches.shape[0])
      idx = rng.choice(np.arange(matches.shape[0]), n)
      for mid in idx:
        x1, y1, x2, y2 = matches[mid, :4]
        r, g, b = hash_to_rbg(mid)
        match_left_xy.append((x1, y1))
        match_right_xy.append((x2, y2))
        match_color.append((int(b), int(g), int(r)))

    K1 = ci_left.K
    #RT1 = ci_left.get_world_to_sensor() # FIXME!! this is giving ego to ego :(
    RT1 = ci_left.ego_pose

    K2 = ci_right.K
    RT2 = ci_right.ego_pose #get_world_to_sensor()

    invRT1 = RT1.get_inverse()
    invRT1h = invRT1.get_transformation_matrix(homogeneous=True)
    RT = RT2.get_transformation_matrix(homogeneous=True) @ invRT1h
    R = RT[:3, :3]
    T = RT[:3, 3]

    distCoeffs1 = ci_left.get_opencv_distcoeffs()
    if distCoeffs1 is None:
      distCoeffs1 = np.array([0., 0., 0., 0.,])
    distCoeffs2 = ci_right.get_opencv_distcoeffs()
    if distCoeffs2 is None:
      distCoeffs2 = np.array([0., 0., 0., 0.,])
    rect_output = cv2.stereoRectify(
                    K1, distCoeffs1, K2, distCoeffs2,
                    (ci_left.width, ci_right.height),
                    R, T,
                    newImageSize=rect_image_wh)
    sR1, sR2, sP1, sP2, sQ, sroi1, sroi2 = rect_output

    right_image_dest = embed_images_root_path / f"right_{right_id}.jpg"
    right_img = ci_right.image
    cv2.imwrite(
      str(right_image_dest),
      cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR),
      [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    right_image_uri = str(right_image_dest)
    if not default_right_image_uri:
      default_right_image_uri = right_image_uri

    def _mat2jsstr(mat):
      nrows = mat.shape[0]
      ncols = 1
      if len(mat.shape) > 1:
        ncols = mat.shape[1]
      js = f"""(
        new cv.matFromArray(
          {nrows},
          {ncols},
          cv.CV_32F,
          {mat.flatten().tolist()}
        )
      )
      """
      return js
    
    def _roundFloats(o, precision=2):
      """Save a bunch of JSON bytes where precision doesn't matter"""
      if isinstance(o, float): return round(o, precision)
      if isinstance(o, (list, tuple)): return [_roundFloats(x) for x in o]
      return o

    rightImageIdToInfo_entries.append(f"""
      "{right_id}": // rightImageId
        {{
          "rightImageId": {right_id},
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
          "mpURI": "{str(mp_uri)}",
          "mpURIPretty": ( {json.dumps(attr.asdict(mp_uri, recurse=True))} ),
          "matchLeftXY": ( {json.dumps(_roundFloats(match_left_xy))} ),
          "matchRightXY": ( {json.dumps(_roundFloats(match_right_xy))} ),
          "matchColor": ( {json.dumps(match_color)} )
        }}
    """)

  from oarphpy import plotting as opplot
  left_img = ci_left.image
  left_img_data_uri = opplot.img_to_data_uri(left_img, format='jpg', jpeg_quality=90)

  stereoRectVizSelectRight_body = "".join(
    f""" <option value="{i}">Image {i}</option> """
    for i in range(len(ci_rights))
  )

  final_html = f"""

  <div id="stereoRectVizRoot">

  <script 
    async
    src="https://docs.opencv.org/4.5.5/opencv.js"
    type="text/javascript">
  </script>
  <script type="text/javascript">

    // BEGIN opencv rectifier and load hook

    // Show first right image by default
    stereoRectVizCurrentRight = "0";

    var Module = {{
      // https://emscripten.org/docs/api_reference/module.html#Module.onRuntimeInitialized
      onRuntimeInitialized() {{
        console.log("StereoRectViz Setup");

        
        // BEGIN embedded rect variables
    
        rightImageIdToInfo = {{
          { ",".join(rightImageIdToInfo_entries) }
        }};

        // END embedded rect variables


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
          let sroi1 = info["sroi1"];
          let sroi2 = info["sroi2"];
          let distCoeffs1 = info["distCoeffs1"];
          let distCoeffs2 = info["distCoeffs2"];
          let newImageSize = info["newImageSize"];
          let mpURI = info["mpURI"];
          let mpURIPretty = info["mpURIPretty"];
          let matchLeftXY = info["matchLeftXY"];
          let matchRightXY = info["matchRightXY"];
          let matchColor = info["matchColor"];

          let leftMap1 = new cv.Mat();
          let leftMap2 = new cv.Mat();
          let rightMap1 = new cv.Mat();
          let rightMap2 = new cv.Mat();
          try {{
            cv.initUndistortRectifyMap(
              K1, distCoeffs1, sR1, sP1, newImageSize, cv.CV_32FC1,
              leftMap1, leftMap2);
            cv.initUndistortRectifyMap(
              K2, distCoeffs2, sR2, sP2, newImageSize, cv.CV_32FC1,
              rightMap1, rightMap2);
          }} catch(err) {{
            document.getElementById("stereoRectMPURI").innerHTML = (
              "Error rectifying, cameras are too far apart? " + err);
          }}

          let leftOrigImg = cv.imread(document.getElementById("inputLeft"));
          let rightOrigImg = cv.imread(document.getElementById("inputRight"));

          for (var i = 0; i < matchLeftXY.length; i++) {{
            let lxy = new cv.Point(matchLeftXY[i][0], matchLeftXY[i][1]);
            let rxy = new cv.Point(matchRightXY[i][0], matchRightXY[i][1]);
            let bgr = new cv.Scalar(
              matchColor[i][0], matchColor[i][1], matchColor[i][2], 128);
            cv.circle(leftOrigImg, lxy, 3, bgr, cv.FILLED);
            cv.circle(rightOrigImg, rxy, 3, bgr, cv.FILLED);
          }}

          let leftRectImg = new cv.Mat();
          cv.remap(
            leftOrigImg,
            leftRectImg, leftMap1, leftMap2, cv.INTER_LANCZOS4);
          let rightRectImg = new cv.Mat();
          cv.remap(
            rightOrigImg,
            rightRectImg, rightMap1, rightMap2, cv.INTER_LANCZOS4);
          
          cv.rectangle(
            leftRectImg,
            new cv.Point(sroi1[0], sroi1[1]), new cv.Point(sroi1[2], sroi1[3]),
            new cv.Scalar(0, 255, 0), 1);
          cv.rectangle(
            rightRectImg,
            new cv.Point(sroi2[0], sroi2[1]), new cv.Point(sroi2[2], sroi2[3]),
            new cv.Scalar(0, 255, 0), 1);

          cv.imshow('stereoRectVizLeft', leftRectImg);
          cv.imshow('stereoRectVizRight', rightRectImg);

          document.getElementById("stereoRectMPURI").innerHTML = mpURI;
          document.getElementById("stereoRectMPURIPretty").innerHTML = 
            JSON.stringify(mpURIPretty, undefined, 2);

        }};

        stereoRectVizSelectRightChanged = function () {{
          var rightId = 
            document.getElementById("stereoRectVizSelectRight").value;
        
          console.log("Selecting right image " + rightId);

          let info = rightImageIdToInfo[rightId];
          let rightImageUri = info["rightImageUri"];
          let rightImage = document.getElementById("inputRight");
          stereoRectVizCurrentRight = rightId;
          rightImage.src = rightImageUri;        
        }};
        
        stereoRectVizRightLoaded = function () {{
          console.log("Right image loaded " + stereoRectVizCurrentRight);
          showRightImageId(stereoRectVizCurrentRight);
        }};

        // END utils


        showRightImageId(stereoRectVizCurrentRight);

        console.log("StereoRectViz Setup Complete");

      }}
    }};
    // END opencv loaded hook
  </script>
    
  
  <!-- StereoRectViz HTML UI -->
  
  <img 
    src="{left_img_data_uri}"
    id="inputLeft"
    style="display: none;" />
  <img 
    src="{default_right_image_uri}"
    id="inputRight"
    style="display: none;"
    onload="stereoRectVizRightLoaded();" />

  <div id="stereoRectVizContainer">
    <div id="stereoRectVizContainerOverlayRoot" style="position: relative">
    
      <div 
        id="stereoRectVizPairViz" 
        style="position: absolute;">
        <table style="background-color: rgba(128, 128, 128, 0.5);">
          <tr>
            <td><canvas id="stereoRectVizLeft"></canvas></td>
            <td><canvas id="stereoRectVizRight"></canvas></td>
          </tr>
        </table>

        <div id="stereoRectControlsNInfo">

          <select
              id="stereoRectVizSelectRight"
              style="padding: 0.5em; font-size: large;"
              onchange="stereoRectVizSelectRightChanged();">
            {stereoRectVizSelectRight_body}
          </select>

          <pre>
            <div id="stereoRectMPURI">(not loaded)</div>
            <div id="stereoRectMPURIPretty">(not loaded)</div>
          </pre>

        </div>

      </div>
    
      <div
        id="stereoRectVizHorizontalLine"
        style="position: absolute; z-index: 10; background-color: red; width: 100%; height: 2px; translate(0px, 100px)"
        >
      </div>
    
    </div>
  </div>
    
  <script type="text/javascript">

    // BEGIN Mouse chaser lines

    mouseChaseDiv = document.getElementById("stereoRectVizPairViz");
    var drawLines = function(event) {{
      let rect = event.target.getBoundingClientRect();
      let x = event.clientX - rect.left; //x position within the element.
      let y = event.clientY - rect.top;  //y position within the element.
      let lineDiv = document.getElementById("stereoRectVizHorizontalLine");
      lineDiv.style.transform = 'translate(0px, ' + y + 'px)';
    }}
    mouseChaseDiv.addEventListener('mousemove', function(event) {{
      drawLines(event);
    }});
    mouseChaseDiv.addEventListener('mousedown', function(event) {{
      drawLines(event);   
    }});
    mouseChaseDiv.addEventListener('mouseup', function(event) {{
      drawLines(event);
    }});
    mouseChaseDiv.addEventListener('mouseout', function(event) {{
      drawLines(event);
    }});

    // END Mouse chaser lines

  </script>

  </div> <!-- END stereoRectVizRoot -->
  
  """

  return final_html

