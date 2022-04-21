import sys
import open3d as o3d

a = sys.argv[1]

mesh = o3d.io.read_triangle_mesh(a, print_progress=True)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])