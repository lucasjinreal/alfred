import platform
import os
import math
import torch
import numpy as np
import cv2
import trimesh


# autopep8: off
os_name = platform.platform().lower()
if 'centos' in os_name or 'windows' in os_name or 'tlinux' in os_name:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
elif 'debian' in os_name or 'ubuntu' in os_name:
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
print(os.environ['PYOPENGL_PLATFORM'])

import pyrender
from pyrender.constants import RenderFlags
from pyrender.constants import DEFAULT_Z_NEAR
# autopep8: on


class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear, zfar=zfar, name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class Renderer:
    def __init__(self, smpl_faces, resolution=(224, 224), orig_img=False, wireframe=False, use_gpu=True):
        """
        resolution is: h, w
        """
        self.name = 'pyrender'
        self.resolution = resolution
        self.faces = smpl_faces
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[1],
            viewport_height=self.resolution[0],
            point_size=1.0,
        )
        # set the scene
        self.scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        # light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)
        print('render initiated.')

    def render(
        self,
        img,
        verts,
        cam,
        angle=None,
        axis=None,
        mesh_filename=None,
        color=[1.0, 1.0, 0.9],
        rotate=False,
    ):
        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)

        Rx = trimesh.transformations.rotation_matrix(
            math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)
        
        if rotate:
            rot = trimesh.transformations.rotation_matrix(np.radians(60), [0, 1, 0])
            mesh.apply_transform(rot)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(
                math.radians(angle), axis)
            mesh.apply_transform(R)

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        sx, sy, tx, ty = cam

        camera = WeakPerspectiveCamera(
            scale=[sx, sy], translation=[tx, ty], zfar=1000.0
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode="OPAQUE",
            smooth=True,
            wireframe=True,
            roughnessFactor=1.0,
            emissiveFactor=(0.1, 0.1, 0.1),
            baseColorFactor=(color[0], color[1], color[2], 1.0),
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        mesh_node = self.scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA
        rgb, _ = self.renderer.render(self.scene, flags=render_flags)

        if rgb.shape[-1] == 4:
            valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
            output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
            image = output_img.astype(np.uint8)
        else:
            # rgb could be 3 channel output
            valid_mask = (rgb > 0)
            output_img = rgb * valid_mask + (1 - valid_mask) * img
            image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)
        return image
