import sys
import open3d as o3d
from alfred import print_shape
import numpy as np


a = sys.argv[1]

mesh = o3d.io.read_triangle_mesh(a, print_progress=True)
mesh.compute_vertex_normals()

print(np.asarray(mesh.vertices).shape)
print(np.asarray(mesh.triangles).shape)

o3d.visualization.draw_geometries([mesh])