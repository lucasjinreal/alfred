from pyrender import Mesh, Scene, Viewer, RenderFlags
from pyrender.constants import DEFAULT_Z_NEAR
from io import BytesIO
import numpy as np
import trimesh
import requests
import pyrender
import cv2


class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
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


duck = trimesh.load('examples/dolphine.obj')
duckmesh = Mesh.from_trimesh(duck)
scene = Scene(ambient_light=np.array(
    [1.0, 1.0, 1.0, 1.0]), bg_color=[0.0, 0.0, 0.0, 0.0],)
scene.add(duckmesh)
# Viewer(scene)

renderer = pyrender.OffscreenRenderer(
    viewport_width=512,
    viewport_height=512,
    point_size=1.0
)
camera_pose = np.eye(4)
sx, sy, tx, ty = 1, 1, 0, 0
camera = WeakPerspectiveCamera(
    scale=[sx, sy],
    translation=[tx, ty],
    zfar=1000.
)
cam_node = scene.add(camera, pose=camera_pose)
print(RenderFlags.RGBA)
rgb, _ = renderer.render(scene, flags=RenderFlags.RGBA)
print(rgb.shape)
cv2.imshow('aa', rgb)
cv2.waitKey(0)
