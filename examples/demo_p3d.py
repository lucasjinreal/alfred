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
import os
try:
    from alfred.vis.renders.render_p3d import Renderer
except ImportError:
    from alfred.vis.renders.render_prd import Renderer
import cv2
import numpy as np

trg_obj = os.path.join("examples/dolphine.obj")
# We read the target 3D model using load_obj
verts, faces, aux = load_obj(trg_obj)


renderer = Renderer()

result = renderer(verts, faces).cpu().numpy()
cv2.imwrite("test.png", (result * 255).astype(np.uint8))

