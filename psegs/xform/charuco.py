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


## For Charuco board pattern generation see:
##  * https://github.com/opencv/opencv_contrib/tree/a26f71313009c93d105151094436eecd4a0990ed/modules/aruco/misc/pattern_generator 
##  * https://calib.io/pages/camera-calibration-pattern-generator 

from cmath import isfinite
import attr
import numpy as np

from psegs import util

# def from_cv2_import_aruco():
#   try:
#     from cv2 import aruco
#     return aruco
#   except ImportError as e:
#     raise ValueError(
#       f"This feature requires opencv-contrib-python>=4.5.5.62, error: {e}")

@attr.s()
class CharucoBoard(object):
  dict_key = attr.ib(default='DICT_6X6_1000', type='str')
  cols = attr.ib(default=11, type='int')
  rows = attr.ib(default=8, type='int')
  square_length_meters = attr.ib(default=0.022, type='float')
  marker_length_meters = attr.ib(default=0.017, type='float')
  
  # Important!!
  # https://github.com/opencv/opencv/issues/23873#issuecomment-1620504453
  is_legacy_pattern = attr.ib(default=True, type='bool')

  # dict_key = attr.ib(default='DICT_5X5_1000', type='str')
  # cols = attr.ib(default=4, type='int')
  # rows = attr.ib(default=4, type='int')
  # square_length_meters = attr.ib(default=0.040, type='float')
  # marker_length_meters = attr.ib(default=0.031, type='float')




  def create_aruco_dict(self):
    # aruco = from_cv2_import_aruco()
    from cv2 import aruco

    # TODO perhaps support custom_dictionary()
    flag = getattr(aruco, self.dict_key, None)
    if flag is None:
      valid_flags = sorted(k for k in dir(aruco) if k.startswith('DICT_'))
      raise ValueError(
        f"Requested {self.dict_key} but only support {valid_flags}")

    try:
      aruco_dict = aruco.Dictionary_get(flag)
    except AttributeError:
      aruco_dict = aruco.getPredefinedDictionary(flag)
    return aruco_dict

  def create_board_and_dict(self):
    # aruco = from_cv2_import_aruco()
    from cv2 import aruco
    
    aruco_dict = self.create_aruco_dict()

    if hasattr(aruco, 'CharucoBoard_create'):
      board = aruco.CharucoBoard_create(
              squaresX=self.cols,
              squaresY=self.rows,
              squareLength=self.square_length_meters,
              markerLength=self.marker_length_meters,
              dictionary=aruco_dict)
    else:
      board = aruco.CharucoBoard(
        (self.cols, self.rows),
        squareLength=self.square_length_meters,
        markerLength=self.marker_length_meters,
        dictionary=aruco_dict)
    return board, aruco_dict

  def create_board_image(self, height_pixels=2000, width_pixels=1000):
    board, _ = self.create_board_and_dict()
    img = board.draw((width_pixels, height_pixels))
    return img

  def detect_board_legacy(self, img_gray, K=None, dist_coeffs=None, refine=False):
    """
    TODO
    refine - Tutorials says do not use when detecting Charuco boards because
      it can confuse marker corners with checkerboard corners; use this
      feature for detecting markers in the wild.
      https://docs.opencv.org/3.4/df/d4a/tutorial_charuco_detection.html
    """
    # aruco = from_cv2_import_aruco()
    # import cv2
    from cv2 import aruco

    # from packaging import version
    # assert version.parse(cv2.__version__) >= version.parse('4.8.1'), (
    #   "Required cv2 version >= 4.8.1 b/c the aruco impl has changed "
    #   "dramatically")
     
    
    board, aruco_dict = self.create_board_and_dict()

    if hasattr(aruco, 'CharucoDetector'):
      detector = aruco.CharucoDetector(board)
      idk1, idk2, arucoCorners, arucoIds = detector.detectBoard(img_gray)
    else:
      aruco_params = aruco.DetectorParameters_create()
      arucoCorners, arucoIds, rejectedImgPoints = aruco.detectMarkers(
                                          img_gray,
                                          aruco_dict,
                                          parameters=aruco_params)

      
    # if refine:
    #   ret = aruco.refineDetectedMarkers(
    #             img_gray,
    #             board,
    #             arucoCorners,
    #             arucoIds,
    #             rejectedImgPoints)
    #   arucoCorners, arucoIds, rejectedCorners, recoveredIdxs = ret

    if arucoIds is None or len(arucoIds) == 0:
      print('FIXME no detections')
      return None

    """
    cv::VideoCapture inputVideo;
    inputVideo.open(0);
    cv::Mat cameraMatrix, distCoeffs;
    // You can read camera parameters from tutorial_camera_params.yml
    readCameraParameters(filename, cameraMatrix, distCoeffs);  // This function is implemented in aruco_samples_utility.hpp
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    // To use tutorial sample, you need read custom dictionaty from tutorial_dict.yml
    readDictionary(filename, dictionary); // This function is implemented in opencv/modules/objdetect/src/aruco/aruco_dictionary.cpp
    cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(5, 7, 0.04, 0.01, dictionary);
    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);
    


    import cv2
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    aruco_board = cv2.aruco.CharucoBoard((11, 8), 0.022, 0.017, dictionary=aruco_dict)
    detector_params = cv2.aruco.DetectorParameters()
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = True
    refine_params = cv2.aruco.RefineParameters()
    detector = cv2.aruco.CharucoDetector(board=aruco_board, charucoParams=charuco_params, detectorParams=detector_params, refineParams=refine_params)
    
    ret = detector.detectBoard(img)

    """

    breakpoint()
    ret = aruco.interpolateCornersCharuco(
            arucoCorners,
            arucoIds,
            img_gray,
            board,
            cameraMatrix=K,
            distCoeffs=dist_coeffs)

    retval, charucoCorners, charucoIds = ret
    return (arucoCorners, arucoIds), (charucoCorners, charucoIds)

  def calibrate_from_images(
        self,
        all_corners=[],
        all_ids=[],
        images=[],
        img_hw=(-1, -1),
        n_dist_coeffs=4):
    
    import cv2
    from cv2 import aruco


    debugs = []

    if not (all_corners and all_ids):
      assert len(images), "Need input images OR marker coordinates"
    
      util.log.info(f"Running Charuco detection on {len(images)} images ...")

      all_corners = []
      all_ids = []
      from tqdm import tqdm
      for img_gray in tqdm(images):
        if img_hw == (-1, -1):
          img_hw = img_gray.shape[:2]

        ret = self.detect_board(img_gray)
        if ret is None:
          continue

        (arucoCorners, arucoIds), (charucoCorners, charucoIds) = ret
        if charucoCorners is None or charucoIds is None:
          continue
        
        if len(charucoIds) < 6:
          continue

        if not all(np.isfinite(c).all() for c in charucoCorners):
          continue

        all_corners.append(charucoCorners)
        all_ids.append(charucoIds)

        if charucoCorners is not None and charucoIds is not None:
          debug = img_gray.copy()
          debug = aruco.drawDetectedMarkers(debug, arucoCorners, arucoIds)
          debug = aruco.drawDetectedCornersCharuco(debug, charucoCorners, charucoIds)
          debugs.append(debug)

    from tqdm import tqdm
    import imageio
    for i, debug in enumerate(tqdm(debugs)):
      imageio.imwrite(f"/opt/psegs/psegs_test/det_charuco_{i}.jpg", debug)


    assert img_hw != (-1, -1), "Need image size"

    board, _ = self.create_board_and_dict()
    h, w = img_hw
    K_init = np.eye(3, 3, dtype='float')
    K_init[0, 0] = float(w)
    K_init[1, 1] = float(h)
    
    import math
    dist_init = np.zeros((n_dist_coeffs, 1))
    # dist_init[0] =  -70. * math.pi / 180
    flags = cv2.CALIB_FIX_ASPECT_RATIO#cv2.CALIB_USE_QR#cv2.CALIB_RATIONAL_MODEL#cv2.CALIB_FIX_ASPECT_RATIO
    
    util.log.info(f"Running Charuco calibration ...")
    try:
      # breakpoint()

      # ret = cv2.calibrateCameraExtended(
      #               all_obj_points,
      #               all_corners,
      #               imageSize=(w, h),
      #               cameraMatrix=K_init,
      #               distCoeffs=dist_init,
      #               flags=0)

      # all_obj_points = []
      # for ids in all_ids:
      #   obj_points = np.array([board.chessboardCorners[i] for i in ids])
      #   all_obj_points.append(obj_points)

      # ret = cv2.fisheye.calibrate(
      #               all_obj_points,
      #               all_corners,
      #               image_size=(w, h),
      #               K=K_init,
      #               D=dist_init,
      #               flags=(
      #                 cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT | 
      #                 cv2.fisheye.CALIB_FIX_SKEW | 
      #                 cv2.fisheye.CALIB_CHECK_COND),
      #               criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 1e-9))
      # ( retval,
      # cameraMatrix,
      # distCoeffs,
      # rvecs,
      # tvecs) = ret


      ret = aruco.calibrateCameraCharucoExtended(
                    charucoCorners=all_corners,
                    charucoIds=all_ids,
                    board=board,
                    imageSize=(w, h),
                    cameraMatrix=K_init,
                    distCoeffs=dist_init,
                    flags=flags,
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 1e-9))
      ( retval,
        cameraMatrix,
        distCoeffs,
        cal_rvecs,
        cal_tvecs,
        stdDeviationsIntrinsics,
        stdDeviationsExtrinsics,
        perViewErrors) = ret
    except Exception as e:
      print(e)
      #breakpoint()
      print()
    util.log.info(f"... done")

    
    
    extra = {
      'rms_reproj_error': retval,
    }
    

    # ncameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w,h), 1, (w,h))

    from tqdm import tqdm
    rvecss = []
    tvecss = []
    debugs_est = []
    for ii in tqdm(range(len(images))):
      img = images[ii]

      

      ncameraMatrix = None
      img = cv2.undistort(src=img, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, newCameraMatrix=ncameraMatrix)
      # img = cv2.fisheye.undistortImage(img, cameraMatrix, distCoeffs, Knew=ncameraMatrix)

      ret = self.detect_board(img)
      if ret is None:
        import imageio
        imageio.imwrite(f"/opt/psegs/psegs_test/charuco_{ii}.jpg", img)
        continue
      (arucoCorners, arucoIds), (charucoCorners, charucoIds) = ret
      img = aruco.drawDetectedMarkers(img, arucoCorners, arucoIds)
      img = aruco.drawDetectedCornersCharuco(img, charucoCorners, charucoIds)
      rvecs, tvecs, objPts = aruco.estimatePoseSingleMarkers(
                              arucoCorners,
                              self.marker_length_meters,
                              cameraMatrix,
                              distCoeffs)
      
      if tvecs is not None:
        for i in range(len(tvecs)):
            img = aruco.drawAxis(
                        img,
                        cameraMatrix,
                        distCoeffs,
                        rvecs[i],
                        tvecs[i],
                        self.marker_length_meters)

      isValid, rvec, tvec = aruco.estimatePoseCharucoBoard(
                              charucoCorners,
                              charucoIds,
                              board=board,
                              cameraMatrix=cameraMatrix,
                              distCoeffs=distCoeffs,
                              rvec=np.array([]),
                              tvec=np.array([]))
      rvecss.append(rvec)
      tvecss.append(tvec)
      try:
        img = cv2.drawFrameAxes(img, cameraMatrix, distCoeffs, rvec, tvec, 0.1)
        debugs_est.append(img)
      except Exception as e:
        print('draw', e)
        debugs_est.append(None)
      
      import imageio
      imageio.imwrite(f"/opt/psegs/psegs_test/charuco_{ii}.jpg", img)

    # print(ret)
    # print(extra)
    # # breakpoint()
    # print()
    #return rvecss, tvecss, debugs_est, cameraMatrix
    return cal_rvecs, cal_tvecs, debugs_est, cameraMatrix


def check_opencv():
  import cv2

  from packaging import version
  assert version.parse(cv2.__version__) >= version.parse('4.8.1'), (
    "Required cv2 version >= 4.8.1 b/c the aruco impl has changed "
    "dramatically between versions; aruco was moved from opencv-contrib "
    "to mainly opencv objdetect and board patterns changed. See e.g. "
    "https://github.com/opencv/opencv/blob/9b97c97bd1a4726f84679618a586e7a6cc8b0909/modules/objdetect/misc/python/test/test_objdetect_aruco.py#L189 "
    "and "
    "https://github.com/opencv/opencv/issues/23873#issuecomment-1620504453")

def detect_charuco_board(
      board_params: CharucoBoard,
      img: np.ndarray) -> 'Any':
  
  check_opencv()

  import cv2
  import cv2.aruco

  if hasattr(cv2.aruco, board_params.dict_key):
    dict_key = getattr(cv2.aruco, board_params.dict_key)
  else:
    valid_flags = sorted(k for k in dir(cv2.aruco) if k.startswith('DICT_'))
    raise ValueError(
      f"Requested {board_params.dict_key} but only support {valid_flags}")

  aruco_dict = cv2.aruco.getPredefinedDictionary(dict_key)
  aruco_board = cv2.aruco.CharucoBoard(
                    (board_params.cols, board_params.rows), 
                    board_params.square_length_meters, 
                    board_params.marker_length_meters, 
                    dictionary=aruco_dict)
  
  # https://github.com/opencv/opencv/issues/23873#issuecomment-1620504453
  aruco_board.setLegacyPattern(board_params.is_legacy_pattern)
  
  detector_params = cv2.aruco.DetectorParameters()
  charuco_params = cv2.aruco.CharucoParameters()
  charuco_params.tryRefineMarkers = True
  refine_params = cv2.aruco.RefineParameters()
  
  marker_detector = cv2.aruco.ArucoDetector(
    dictionary=aruco_dict,
    detectorParams=detector_params,
    refineParams=refine_params)

  md_ret = marker_detector.detectMarkers(img)
  markerCorners, markerIds, rejectedImgPoints = md_ret
  debug_marker_detections = cv2.aruco.drawDetectedMarkers(
                              img.copy(), corners=markerCorners, ids=markerIds)
  
  cv2.imwrite('/opt/psegs/debug_marker_detections.png', debug_marker_detections)

  debug_marker_rejections = cv2.aruco.drawDetectedMarkers(
                                img.copy(), corners=rejectedImgPoints)

  cv2.imwrite('/opt/psegs/debug_marker_rejections.png', debug_marker_rejections)

  board_detector = cv2.aruco.CharucoDetector(
    board=aruco_board,
    charucoParams=charuco_params)
    # detectorParams=detector_params,
    # refineParams=refine_params)
  
  debug_board_image = aruco_board.generateImage((board_params.cols*50, board_params.rows*50), marginSize=10)
  cv2.imwrite('/opt/psegs/debug_board_image.png', debug_board_image)

  bdet_ret = board_detector.detectBoard(img)
  charucoCorners, charucoIds, bdet_markerCorners, bdet_markerIds = bdet_ret

  debug_board_detections = cv2.aruco.drawDetectedCornersCharuco(
    img.copy(), charucoCorners, charucoIds=charucoIds)
  cv2.imwrite('/opt/psegs/debug_board_detections.png', debug_board_detections)

  breakpoint()
  print()
  





















"""
w, h 5312 2988
K
array([[9.58550221e+02, 0.00000000e+00, 2.88703475e+03],
       [0.00000000e+00, 9.58550221e+02, 2.84402233e+03],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

distCoeffs
array([[ 0.01447736],
       [-0.00325806],
       [ 0.00411286],
       [-0.01069181],
       [ 0.00011799]])


based upon 1920x1080 /outer_root/media/970-evo-plus-raid0/hloc_out/pwais.private.lidar_hero10_winter_stinsin_GX010018.MP4_cache/images/camera_adhoc.1645923337846437120.png

w, h 1920 1080

K
array([[2.07996977e+03, 0.00000000e+00, 4.58423012e+02],
       [0.00000000e+00, 2.07996977e+03, 5.82382328e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

distCoeffs
array([[-1.88078561],
       [ 3.09708077],
       [ 0.01062191],
       [ 0.20261413],
       [-3.1799531 ]])


TODO:
 * add a video input ...
 * enable subsampling for ~200 images b/c seems to be properly N^2
 * input: images
 * output:
     * CameraImage with K, disp and RT for every image
     * cuboids for board and markers
     * rectified debug video
     * 3D scene with camera poses and cuboids
     * a hook to tutorial / readme for that

Then we can SLAM the gopro

"""









# def create_calibrated_cameras(images):
#   pass

# # def charuco_board_image(
# #       aruco_dict_key='DICT_6X6_250',

# #       squaresX=11, squaresY=8, squareLength=.022, markerLength=.017
# # ):

# def detect_charuco_board(
#           img_gray,
#           aruco_dict_key='DICT_6X6_250'):
  
  


#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
#     for corner in corners:
#         cv2.cornerSubPix(gray, corner, winSize = (3,3), zeroZone = (-1,-1), criteria = criteria)


# frame = cv2.imread(path)
#     img_undist = cv2.undistort(src = frame, cameraMatrix = mtx, distCoeffs = dist)
    
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
#     parameters =  aruco.DetectorParameters_create()
#     corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict,
#                                                           parameters=parameters)
#     # SUB PIXEL DETECTION
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
#     for corner in corners:
#         cv2.cornerSubPix(gray, corner, winSize = (3,3), zeroZone = (-1,-1), criteria = criteria)

#     frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    
#     size_of_marker =  0.012 # side lenght of the marker in meter
#     rvecs,tvecs,objPts = aruco.estimatePoseSingleMarkers(corners, size_of_marker , mtx, dist)
    
#     length_of_axis = 0.012
#     imaxis = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

#     if tvecs is not None:
#         for i in range(len(tvecs)):
#             imaxis = aruco.drawAxis(imaxis, mtx, dist, rvecs[i], tvecs[i], length_of_axis)
    
#     imaxis = cv2.resize(imaxis, (frame.shape[1] // 4, frame.shape[0] // 4))
#     writer.append_data(imaxis)
#     print('did frame', ii)

# """

# debug images of detected markers. unrect only at first

# then run calib, and plot detected markers on rectified

# """

