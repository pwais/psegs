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


def pytorch3d_iter_mesh2uvd_for_camera_images(
        cis,
        mesh_path='',
        batch_size=-1):
  
  import numpy as np

  from oarphpy import util as oputil

  import torch

  # Data structures and functions for rendering
  from pytorch3d.structures import Meshes
  from pytorch3d.io import load_obj
  from pytorch3d.renderer import (
      PointLights, 
      RasterizationSettings, 
      MeshRenderer, 
      MeshRasterizer,  
      HardPhongShader,
      TexturesVertex,
      FoVPerspectiveCameras,
      BlendParams,
      PointLights,
      TexturesVertex,
  )


  if not cis:
    return
  
  if batch_size < 0:
    batch_size = 10
  
  
  if torch.cuda.is_available():
      device = torch.device("cuda:0")
      torch.cuda.set_device(device)
  else:
      device = torch.device("cpu")

  verts, faces_idx, _ = load_obj(mesh_path)
  faces = faces_idx.verts_idx
  print('verts', verts.shape)
  print('faces', faces.shape)

  # nverts = verts.numpy()
  # nfaces = faces.numpy()

  # Initialize each vertex to be white in color.
  verts_rgb = .9 * torch.ones_like(verts)[None]  # (1, V, 3)

  # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
  verts = verts.to(device).tile((batch_size, 1, 1))
  faces = faces.to(device).tile((batch_size, 1, 1))
  verts_rgb = verts_rgb.to(device).tile((batch_size, 1, 1))
  mesh = Meshes(
      verts=verts,
      faces=faces,
      textures=TexturesVertex(verts_features=verts_rgb))
  print('mesh', mesh)

  import torch
  K = torch.zeros(batch_size, 4, 4, dtype=torch.float32, device=device)
  R = torch.zeros(batch_size, 3, 3, dtype=torch.float32, device=device)
  T = torch.zeros(batch_size, 3, dtype=torch.float32, device=device)
  image_size = torch.zeros(batch_size, 2, dtype=torch.float32, device=device)

  fov_y = None
  rasterizer_image_size = None
  import time
  for ci_chunk in oputil.ichunked(cis, batch_size):
    start = time.time()
    print('start batch')
    for i, ci in enumerate(ci_chunk):
      if fov_y is None:
        fov_x, fov_y = ci.get_fov()
      
      if rasterizer_image_size is None:
        rasterizer_image_size = (ci.height, ci.width)
      
      pose = ci.ego_pose['world', 'ego'].get_transformation_matrix(homogeneous=True)

      # For iOS !!!
      world2pytorch = np.array([
          [1, 0, 0, 0],
          [0, -1, 0, 0],
          [0, 0, -1, 0],
          [0, 0, 0, 1],
      ], dtype=np.float32)

      pose = world2pytorch @ pose

      K[i, :3, :3] = torch.from_numpy(ci.K)
      K[i, 3, 3] = 1
      R[i, :3, :3] = torch.from_numpy(pose[:3, :3])
      T[i, :3] = torch.from_numpy(pose[:3, 3])
      image_size[i, 0] = ci.height
      image_size[i, 1] = ci.width
      

    # # https://github.com/facebookresearch/pytorch3d/issues/522
    # from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
    # cameras = cameras_from_opencv_projection(R, T, tK, image_size, device=device).cuda()


    # # hack up cameras_from_opencv_projection
    tvec = T
    R_pytorch3d = R.clone().permute(0, 2, 1)
    T_pytorch3d = tvec.clone()
    R_pytorch3d[:, :, :2] *= -1
    T_pytorch3d[:, :2] *= -1

    cameras = FoVPerspectiveCameras(
                device=device,
                fov=fov_y,
                degrees=False,
                R=R_pytorch3d,
                T=T_pytorch3d,
                znear=0.01,
                zfar=100.)

    raster_settings = RasterizationSettings(
        image_size=rasterizer_image_size, 
        faces_per_pixel=1)
    lights = PointLights(
        device=device, 
        location=cameras.get_world_to_view_transform().transform_points(
          torch.tensor([[0., 0., -1.]]).cuda()))

    blend_params = BlendParams(
                      sigma=1e-4,
                      gamma=1e-4,
                      background_color=(0.1, 0.1, 0.1))
    rasterizer = MeshRasterizer(
          cameras=cameras, 
          raster_settings=raster_settings)
    # phong_renderer = MeshRenderer(
    #     rasterizer=rasterizer,
    #     shader=HardPhongShader(
    #               device=device,
    #               cameras=cameras,
    #               lights=lights,
    #               blend_params=blend_params))


    # image_ref = phong_renderer(meshes_world=mesh)
    # import torchvision.transforms.functional as F
    # import numpy as np
    # pil_img = F.to_pil_image((255.0*image_ref.cpu().numpy()[0]).astype(np.uint8))

    # from IPython.display import display
    # display(pil_img)


    fragments = rasterizer(meshes_world=mesh)

    zbuf = fragments.zbuf
    print('batch done', time.time() - start)
    for i in range(batch_size):
      depth = zbuf[i, :, :, 0].cpu().numpy()
      
      h, w = rasterizer_image_size
      px_y = np.tile(np.arange(h)[:, np.newaxis], [1, w])
      px_x = np.tile(np.arange(w)[np.newaxis, :], [h, 1])
      pyx = np.concatenate([px_y[:,:,np.newaxis], px_x[:, :, np.newaxis]], axis=-1)
      pyx = pyx.astype(np.float32)

      vud1 = np.dstack([pyx, depth]).reshape([-1, 3])

      vud1 = vud1[vud1[:, 2] > 0]
      uvd = vud1[:, (1, 0, 2)]

      
      print('yielding', batch_size)
      yield uvd


  
  # plt.imshow(zbuf[0, ..., 0].cpu().numpy())
  # plt.show()
  # print('zbuf', zbuf.min(), zbuf.max(), zbuf[zbuf > 0].min())
  # # display(F.to_pil_image(image_ref.cpu().numpy()[0].astype(np.uint8)[:, :, -1]))
  # # image_ref.cpu().numpy()[0]

  # depth = zbuf[0, ..., 0].cpu().numpy()
  # h, w = rasterizer_image_size
  # px_y = np.tile(np.arange(h)[:, np.newaxis], [1, w])
  # px_x = np.tile(np.arange(w)[np.newaxis, :], [h, 1])
  # pyx = np.concatenate([px_y[:,:,np.newaxis], px_x[:, :, np.newaxis]], axis=-1)
  # pyx = pyx.astype(np.float32)

  # vud1 = np.dstack([pyx, depth]).reshape([-1, 3])

  # vud1 = vud1[vud1[:, 2] > 0]
  # uvd = vud1[:, (1, 0, 2)]
  # yield uvd


  # from psegs.util.plotting import draw_xy_depth_in_image
  # draw_xy_depth_in_image(debug, uvd, period_meters=0.1)


  # import torchvision.transforms.functional as F
  # import numpy as np
  # pil_img = F.to_pil_image(debug.astype(np.uint8))

  # from IPython.display import display
  # display(pil_img)






# def pytorch3d_mesh2depth_for_camera_images(cis, mesh_path='', batch_size=-1):
#   if batch_size < 0:
#     batch_size = 10
  
#   from oarphpy import util as oputil

#   from pytorch3d.io import load_obj
#   obj_path = os.path.join(base_dir, 'export_refined.obj')
#   verts, faces_idx, _ = load_obj(obj_path)
#   faces = faces_idx.verts_idx
#   print('verts', verts.shape)
#   print('faces', faces.shape)

#   nverts = verts.numpy()
#   nfaces = faces.numpy()

#   import numpy as np
# # obj_path = os.path.join(base_dir, 'export_refined.obj')
# fov_x, fov_y = CI.get_fov()
# K = CI.K
# height, width = CI.height, CI.width
# pose = CI.ego_pose['ego', 'world'].get_inverse().get_transformation_matrix(homogeneous=True)
# pose = pose.astype(np.float32)
# K = K.astype(np.float32)



# import numpy as np
# world2pytorch = np.array([
#     [1, 0, 0, 0],
#     [0, -1, 0, 0],
#     [0, 0, -1, 0],
#     [0, 0, 0, 1],
# ], dtype=np.float32)

# pose = world2pytorch @ pose

# # pose[0, 0] *= -1
# # pose[1, 1] *= -1
# # pose[2, 2] *= -1

# import os
# import sys
# import torch

# import pytorch3d

# import os
# import torch
# import matplotlib.pyplot as plt

# from pytorch3d.utils import ico_sphere
# import numpy as np
# from tqdm.notebook import tqdm




# # Util function for loading meshes
# from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj

# from pytorch3d.loss import (
#     chamfer_distance, 
#     mesh_edge_loss, 
#     mesh_laplacian_smoothing, 
#     mesh_normal_consistency,
# )

# # Data structures and functions for rendering
# from pytorch3d.structures import Meshes
# from pytorch3d.renderer import (
#     look_at_view_transform,
#     OpenGLPerspectiveCameras, 
#     PointLights, 
#     DirectionalLights, 
#     Materials, 
#     RasterizationSettings, 
#     MeshRenderer, 
#     MeshRasterizer,  
#     SoftPhongShader,
#     SoftSilhouetteShader,
#     SoftPhongShader,
#     TexturesVertex,
#     AmbientLights
# )

# # add path for demo utils functions 
# import sys
# import os
# sys.path.append(os.path.abspath(''))


# # io utils
# from pytorch3d.io import load_obj

# # datastructures
# from pytorch3d.structures import Meshes

# # 3D transformations functions
# from pytorch3d.transforms import Rotate, Translate

# # rendering components
# from pytorch3d.renderer import (
#     FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
#     RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
#     SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
#     HardGouraudShader, SoftGouraudShader,HardFlatShader,PerspectiveCameras,FoVOrthographicCameras,
# )


# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
#     torch.cuda.set_device(device)
# else:
#     device = torch.device("cpu")
# # device = torch.device("cpu")

# # # Set paths
# # DATA_DIR = "./data"
# # obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")

# # Load obj file
# # mesh = load_objs_as_meshes([obj_path], device=device)

# # Load the obj and ignore the textures and materials.
# # verts, faces_idx, _ = load_obj(obj_path)
# # faces = faces_idx.verts_idx
# # print('verts', verts.shape)
# # print('faces', faces.shape)
# import torch
# verts = torch.from_numpy(nverts)
# faces = torch.from_numpy(nfaces)

# # Initialize each vertex to be white in color.
# verts_rgb = .9 * torch.ones_like(verts)[None]  # (1, V, 3)
# textures = TexturesVertex(verts_features=verts_rgb.to(device))

# # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
# teapot_mesh = Meshes(
#     verts=[verts.to(device)],   
#     faces=[faces.to(device)], 
#     textures=textures,
# )
# print('teapot_mesh', teapot_mesh)
# # teapot_mesh = load_objs_as_meshes([obj_path], device=device)


# import torch
# import numpy as np
# R = torch.from_numpy(pose[:3, :3].reshape([1, 3, 3])).to(device)
# T = torch.from_numpy(pose[:3, 3].reshape([1, 3])).to(device)


# # Select the viewpoint using spherical angles  
# distance = 5   # distance from camera to the object
# elevation = 50.0   # angle of elevation in degrees
# azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis. 

# # Get the position of the camera based on the spherical angles
# # R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
# print('R', R)
# print('T', T)

# tK = np.eye(4).astype(np.float32)
# tK[:3, :3] = K

# # # Great job pytorch3d!! 
# # # https://github.com/facebookresearch/pytorch3d/blob/103da63393d6bbb697835ddbfc86b07572ea4d0c/tests/test_camera_conversions.py#L116
# # tK[0, 0] = 1.1 * K[0, 0]
# # tK[1, 1] = 1.1 * K[1, 1]
# # tK[2, 0] = 1.1 * K[2, 0]
# # tK[2, 1] = 1.1 * K[2, 1]


# tK = torch.from_numpy(tK.reshape([1, 4, 4])).to(device)
# print('K', tK)

# image_size = torch.from_numpy(np.array([height, width]).reshape([1, 2])).to(device)
# print('image_size', image_size)

# # https://github.com/facebookresearch/pytorch3d/issues/522
# from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection




# cameras = cameras_from_opencv_projection(R, T, tK, image_size, device=device).cuda()

# # assert False, (cameras.R, cameras.T, cameras.get_world_to_view_transform().device)
# # assert False, cameras.get_world_to_view_transform().device
# # print('get_world_to_view_transform', xform.device, cameras.R, cameras.T)

# # cameras = FoVPerspectiveCameras(device=device, fov=fov_x, degrees=False)#, K=K)

# # # hack up cameras_from_opencv_projection
# # camera_matrix = tK
# tvec = T
# # focal_length = torch.stack([camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]], dim=-1)
# # principal_point = camera_matrix[:, :2, 2]

# # # Retype the image_size correctly and flip to width, height.
# # image_size_wh = image_size.to(R).flip(dims=(1,))

# # # Get the PyTorch3D focal length and principal point.
# # focal_pytorch3d = focal_length / (0.5 * image_size_wh)
# # p0_pytorch3d = -(principal_point / (0.5 * image_size_wh) - 1)

# # For R, T we flip x, y axes (opencv screen space has an opposite
# # orientation of screen axes).
# # We also transpose R (opencv multiplies points from the opposite=left side).
# R_pytorch3d = R.clone().permute(0, 2, 1)
# T_pytorch3d = tvec.clone()
# R_pytorch3d[:, :, :2] *= -1
# T_pytorch3d[:, :2] *= -1
# # cameras = PerspectiveCameras(
# #             device=device, R=R_pytorch3d,
# #             T=T_pytorch3d,
# #             focal_length=focal_pytorch3d,
# #             principal_point=p0_pytorch3d, image_size=image_size, in_ndc=True)

# fov_x, fov_y = CI.get_fov()
# cameras = FoVPerspectiveCameras(
#     device=device, fov=fov_y, degrees=False, R=R_pytorch3d, T=T_pytorch3d, aspect_ratio=1.0)


# # cameras = PerspectiveCameras(device=device, K=K, R=R, T=T, in_ndc=False, image_size=image_size)


# raster_settings = RasterizationSettings(
#     image_size=(height, width), 
#     faces_per_pixel=1, 
# )
# lights = PointLights(
#     device=device, 
#     location=cameras.get_world_to_view_transform().transform_points(torch.tensor([[0., 0., -1.]]).cuda()),
# )
# # lights = AmbientLights(device=device)
# blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.1, 0.1, 0.1))
# rasterizer = MeshRasterizer(
#         cameras=cameras, 
#         raster_settings=raster_settings
#     )
# phong_renderer = MeshRenderer(
#     rasterizer=rasterizer,
#     shader=HardPhongShader(device=device, cameras=cameras, lights=lights, blend_params=blend_params)
# )


# image_ref = phong_renderer(meshes_world=teapot_mesh)


# import torchvision.transforms.functional as F
# import numpy as np
# pil_img = F.to_pil_image((255.0*image_ref.cpu().numpy()[0]).astype(np.uint8))

# from IPython.display import display
# display(pil_img)

# import matplotlib.pyplot as plt

# fragments = rasterizer(meshes_world=teapot_mesh)

# zbuf = fragments.zbuf
# plt.imshow(zbuf[0, ..., 0].cpu().numpy())
# plt.show()
# print('zbuf', zbuf.min(), zbuf.max(), zbuf[zbuf > 0].min())
# # display(F.to_pil_image(image_ref.cpu().numpy()[0].astype(np.uint8)[:, :, -1]))
# # image_ref.cpu().numpy()[0]

# debug = CI.image
# depth = zbuf[0, ..., 0].cpu().numpy()
# h, w = debug.shape[:2]
# px_y = np.tile(np.arange(h)[:, np.newaxis], [1, w])
# px_x = np.tile(np.arange(w)[np.newaxis, :], [h, 1])
# pyx = np.concatenate([px_y[:,:,np.newaxis], px_x[:, :, np.newaxis]], axis=-1)
# pyx = pyx.astype(np.float32)

# vud1 = np.dstack([pyx, depth]).reshape([-1, 3])

# vud1 = vud1[vud1[:, 2] > 0]
# uvd = vud1[:, (1, 0, 2)]


# from psegs.util.plotting import draw_xy_depth_in_image
# draw_xy_depth_in_image(debug, uvd, period_meters=0.1)


# import torchvision.transforms.functional as F
# import numpy as np
# pil_img = F.to_pil_image(debug.astype(np.uint8))

# from IPython.display import display
# display(pil_img)

