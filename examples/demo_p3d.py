import os

try:
    from pytorch3d.io import load_obj, save_obj
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import ico_sphere
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.loss import (
        chamfer_distance,
        mesh_edge_loss,
        mesh_laplacian_smoothing,
        mesh_normal_consistency,
    )
    from alfred.vis.renders.render_p3d import Renderer
except ImportError:
    from alfred.vis.renders.render_prd import Renderer
    import trimesh
    import colorsys
    import pyrender
import cv2
import numpy as np

import os
import torch
import matplotlib.pyplot as plt
from skimage.io import imread

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

import numpy as np
from alfred.dl.torch.common import device, print_shape
# add path for demo utils functions 
import sys
import os


'''
wget -P data/cow_mesh https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.obj
wget -P data/cow_mesh https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.mtl
wget -P data/cow_mesh https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow_texture.png
'''

obj_filename = os.path.join('./cow_mesh', "cow.obj")

# Load obj file
mesh = load_objs_as_meshes([obj_filename], device=device)

R, T = look_at_view_transform(2.7, 0, 180)
print(R, T)
# R = torch.eye(3).unsqueeze(0)
R = torch.randn([1, 3, 3])
print_shape(R, T)
sfm_camera = PerspectiveCameras(device=device, R=R, T=T)
width = 500
height = 500
raster_settings = RasterizationSettings(
    image_size=(height, width), 
    blur_radius=0.0, 
    faces_per_pixel=1,
)
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
sfm_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=sfm_camera, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=sfm_camera,
        lights=lights
    )
)
images = sfm_renderer(mesh)
cv2.imshow('aa', images[0, ..., :3].cpu().numpy())
cv2.waitKey(0)