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


obj_f = "examples/dolphine.obj"
ori_img = cv2.imread("examples/data/000000.png")
fuze_trimesh = trimesh.load(obj_f, force="mesh")
verts = fuze_trimesh.vertices
faces = fuze_trimesh.faces

renderer = Renderer(smpl_faces=faces, resolution=ori_img.shape[:-1])
mesh_color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
img = renderer.render(
    None,
    verts,
    # cam=orig_cam,
    color=mesh_color,
    mesh_filename=None,
)
print(img)
cv2.imwrite("a.png", img)
print("saved?")

