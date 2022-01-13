# -*- coding: utf-8 -*-
# brought from https://github.com/mkocabas/VIBE/blob/master/lib/utils/renderer.py
import sys
import os
import json
from typing import Tuple
from pytorch3d.renderer.cameras import PerspectiveCameras
import torch
from torch import nn
import pickle

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    OpenGLPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
)
# from pytorch3d.io.mesh
from pytorch3d.io import IO
import numpy as np


"""
source code adopt from ROMP
edited by alfred-py
"""


colors = {
    "pink": [0.7, 0.7, 0.9],
    "neutral": [0.9, 0.9, 0.8],
    "capsule": [0.7, 0.75, 0.5],
    "yellow": [0.5, 0.7, 0.75],
}


def get_projection_matrix_for_weak_perspective_camera(s_x, s_y, t_x, t_y):
    P = torch.eye(4)
    P[0, 0] = s_x
    P[1, 1] = s_y
    P[0, 3] = t_x * s_x
    P[1, 3] = -t_y * s_y
    P[2, 2] = -1
    print(P)
    return P[:-1, :-1], P[:, -1][:-1]


class Renderer(nn.Module):
    def __init__(
        self,
        smpl_faces,
        resolution=(512, 512),
        perps=True,
        R=None,
        T=None,
        use_gpu=False,
    ):
        super(Renderer, self).__init__()
        self.name = 'pytorch3d'
        self.perps = perps
        if use_gpu:
            self.device = torch.device('cuda')
            print("visualize in gpu mode")
        else:
            self.device = torch.device("cpu")
            print("visualize in cpu mode")

        if isinstance(smpl_faces, np.ndarray):
            smpl_faces = torch.from_numpy(smpl_faces.astype(np.float))
            smpl_faces.to(self.device)
        self.faces = smpl_faces.unsqueeze(0).to(
            self.device)  # add a BatchSize dim
        self.default_color = torch.as_tensor(colors["neutral"]).unsqueeze(0)
        print('default color: ', self.default_color.shape)
        self.save_io = IO()

        if R is None:
            self.default_R = torch.Tensor(
                [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        if T is None:
            self.default_T = torch.Tensor([[0.0, 0.0, 0.0]])
        self.fov = 60

        if self.perps:
            # self.cameras = FoVPerspectiveCameras(zfar=1000,
            #     R=self.default_R, T=self.default_T, fov=60, device=self.device
            # )
            self.cameras = PerspectiveCameras(focal_length=5000,
                R=self.default_R, T=self.default_T, device=self.device
            )
            self.lights = PointLights(
                ambient_color=((0.56, 0.56, 0.56),),
                location=torch.Tensor([[0.0, 0.0, 0.0]]),
                device=self.device,
            )
        else:
            self.cameras = FoVOrthographicCameras(
                R=self.default_R,
                T=self.default_T,
                znear=0.0,
                zfar=100.0,
                max_y=1.0,
                min_y=-1.0,
                max_x=1.0,
                min_x=-1.0,
                device=self.device,
            )
            self.lights = DirectionalLights(
                direction=torch.Tensor([[0.0, 1.0, 0.0]]), device=self.device
            )

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0.
        raster_settings = RasterizationSettings(
            image_size=(resolution[0], resolution[1]), blur_radius=0.0, faces_per_pixel=1
        )
        print('resolution: ', resolution)

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and
        # apply the Phong lighting model
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device, cameras=self.cameras, lights=self.lights
            ),
        )

    # def render(self, img, verts, cam, angle=None, axis=None, mesh_filename=None, color=[1.0, 1.0, 0.9], rotate=False):
    def render(
        self, img, verts, faces=None, color=None, merge_meshes=True, cam=None, mesh_filename=None, **kwargs
    ):
        assert len(
            verts.shape) >= 2, f"The input verts of visualizer is bounded to be 3-dims (Nx6890 x3) tensor, but got: {verts.shape}"
        if isinstance(verts, np.ndarray):
            verts = torch.from_numpy(verts)
        if len(verts.shape) == 2:
            verts = verts.unsqueeze(0)
        if faces is None:
            faces = self.faces
        
        BS = verts.shape[0]
        if verts.shape[0] != faces.shape[0]:
            faces = faces.repeat(BS, 1, 1).to(self.device)

        verts = verts.to(self.device)

        # print('vertis: ', verts.shape, verts.device)
        # print('faces: ', faces.shape, faces.device)
        # print('cam: ', cam)
        if isinstance(color, np.ndarray) or isinstance(color, tuple):
            # color = torch.from_numpy(np.array(color)).to(self.device).unsqueeze(1)
            color = torch.from_numpy(np.array(color)).to(
                self.device).unsqueeze(0)
        elif color is None:
            color = self.default_color

        verts = verts.to(self.device)
        verts = verts.float()
        verts_rgb = torch.ones_like(verts)
        verts_rgb[:, :] = color
        textures = TexturesVertex(verts_features=verts_rgb)
        verts[:, :, :2] *= -1
        meshes = Meshes(verts, faces, textures)
        if merge_meshes:
            meshes = join_meshes_as_scene(meshes)
        if mesh_filename is not None:
            self.save_io.save_mesh(meshes, mesh_filename)
        if cam is not None:
            # cam = cam.float()
            # print(cam)
            cam = torch.as_tensor(cam).to(self.device)
            if self.perps:
                # R, T, fov = cam
                # distance = 3   # distance from camera to the object
                # elevation = 50.0   # angle of elevation in degrees
                # azimuth = 0.0
                # R, T = look_at_view_transform(
                #     distance, elevation, azimuth, device=self.device)
                # print(R, T)
                # print(R.shape, T.shape)
                # fov = 60
                # R, T = get_projection_matrix_for_weak_perspective_camera(cam[0], cam[1], cam[2], cam[3])
                # R = R.unsqueeze(0).repeat(BS, 1, 1)
                T = cam
                T = T.unsqueeze(0).repeat(BS, 1)
                # new_cam = FoVPerspectiveCameras(zfar=1000, znear=0.05,
                #     R=self.default_R, T=T, fov=self.fov, device=self.device)
                new_cam = PerspectiveCameras(focal_length=5000,
                    R=self.default_R, T=T, device=self.device)
            else:
                R, T, xyz_ranges = cam
                new_cam = FoVOrthographicCameras(
                    R=R, T=self.default_T, **xyz_ranges, device=self.device
                )
            images = self.renderer(meshes, cameras=new_cam)
        else:
            images = self.renderer(meshes)
        images[:, :, :-1] *= 255
        images = images[:, :, :-1].cpu().numpy()
        return images


def get_renderer(test=False, **kwargs):
    renderer = Renderer(**kwargs)
    if test:
        import cv2

        dist = 1 / np.tan(np.radians(args().FOV / 2.0))
        print("dist:", dist)
        model = pickle.load(
            open(
                os.path.join(args().smpl_model_path, "smpl",
                             "SMPL_NEUTRAL.pkl"), "rb"
            ),
            encoding="latin1",
        )
        np_v_template = (
            torch.from_numpy(
                np.array(model["v_template"])).cuda().float()[None]
        )
        face = torch.from_numpy(model["f"].astype(np.int32)).cuda()[None]
        np_v_template = np_v_template.repeat(2, 1, 1)
        np_v_template[1] += 0.3
        np_v_template[:, :, 2] += dist
        face = face.repeat(2, 1, 1)
        result = renderer(np_v_template, face).cpu().numpy()
        for ri in range(len(result)):
            cv2.imwrite(
                "test{}.png".format(
                    ri), (result[ri, :, :, :3] * 255).astype(np.uint8)
            )
    return renderer


if __name__ == "__main__":
    get_renderer(test=True, perps=True)
