import copy
import os
import open3d as o3d
import numpy as np
import numpy as np
import cv2
import numpy as np
from tqdm import tqdm
from os.path import join
from alfred import logger


def calRot(axis, direc):
    direc = direc / np.linalg.norm(direc)
    axis = axis / np.linalg.norm(axis)
    rotdir = np.cross(axis, direc)
    rotdir = rotdir / np.linalg.norm(rotdir)
    rotdir = rotdir * np.arccos(np.dot(direc, axis))
    rotmat, _ = cv2.Rodrigues(rotdir)
    return rotmat


def create_ground_(
    center=[0, 0, 0],
    xdir=[1, 0, 0],
    ydir=[0, 1, 0],  # 位置
    step=1,
    xrange=10,
    yrange=10,  # 尺寸
    white=[1.0, 1.0, 1.0],
    black=[0.0, 0.0, 0.0],  # 颜色
    two_sides=True,
):
    if isinstance(center, list):
        center = np.array(center)
        xdir = np.array(xdir)
        ydir = np.array(ydir)
    logger.info("[Vis Info] {}, x: {}, y: {}".format(center, xdir, ydir))
    xdir = xdir * step
    ydir = ydir * step
    vertls, trils, colls = [], [], []
    cnt = 0
    min_x = -xrange if two_sides else 0
    min_y = -yrange if two_sides else 0
    for i in range(min_x, xrange):
        for j in range(min_y, yrange):
            point0 = center + i * xdir + j * ydir
            point1 = center + (i + 1) * xdir + j * ydir
            point2 = center + (i + 1) * xdir + (j + 1) * ydir
            point3 = center + (i) * xdir + (j + 1) * ydir
            if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                col = white
            else:
                col = black
            vert = np.stack([point0, point1, point2, point3])
            col = np.stack([col for _ in range(vert.shape[0])])
            tri = np.array([[2, 3, 0], [0, 1, 2]]) + vert.shape[0] * cnt
            cnt += 1

            # vert = vert.astype(np.float32)
            # vert[:, 0] /= xrange * 1.0
            # vert[:, 1] /= yrange * 1.0
            vertls.append(vert)
            trils.append(tri)
            colls.append(col)
    vertls = np.vstack(vertls)
    trils = np.vstack(trils)
    colls = np.vstack(colls)
    res = {"vertices": vertls, "faces": trils, "colors": colls, "name": "ground"}
    # print(res)
    return res


def get_rotation_from_two_directions(direc0, direc1):
    direc0 = direc0 / np.linalg.norm(direc0)
    direc1 = direc1 / np.linalg.norm(direc1)
    rotdir = np.cross(direc0, direc1)
    if np.linalg.norm(rotdir) < 1e-2:
        return np.eye(3)
    rotdir = rotdir / np.linalg.norm(rotdir)
    rotdir = rotdir * np.arccos(np.dot(direc0, direc1))
    rotmat, _ = cv2.Rodrigues(rotdir)
    return rotmat


PLANE_VERTICES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
)
PLANE_FACES = np.array(
    [
        [4, 7, 5],
        [4, 6, 7],
        [0, 2, 4],
        [2, 6, 4],
        [0, 1, 2],
        [1, 3, 2],
        [1, 5, 7],
        [1, 7, 3],
        [2, 3, 7],
        [2, 7, 6],
        [0, 4, 1],
        [1, 4, 5],
    ],
    dtype=np.int32,
)


current_dir = os.path.dirname(os.path.realpath(__file__))

Vector3dVector = o3d.utility.Vector3dVector
Vector3iVector = o3d.utility.Vector3iVector
Vector2iVector = o3d.utility.Vector2iVector
TriangleMesh = o3d.geometry.TriangleMesh
load_mesh = o3d.io.read_triangle_mesh
vis = o3d.visualization.draw_geometries


def create_mesh(vertices, faces, colors=None, **kwargs):
    mesh = TriangleMesh()
    mesh.vertices = Vector3dVector(vertices)
    mesh.triangles = Vector3iVector(faces)
    if colors is not None:
        mesh.vertex_colors = Vector3dVector(colors)
    else:
        mesh.paint_uniform_color([1.0, 0.8, 0.8])
    mesh.compute_vertex_normals()
    return mesh


def create_ground(**kwargs):
    ground = create_ground_(**kwargs)
    return create_mesh(**ground)


def create_coord(camera=[0, 0, 0], radius=1, scale=1):
    # camera_frame = TriangleMesh.create_coordinate_frame(size=radius, origin=camera)
    camera_frame = TriangleMesh.create_coordinate_frame(size=radius, origin=camera)
    # camera_frame = TriangleMesh.create_coordinate_frame()
    # camera_frame = o3d.geometry.create_mesh_coordinate_frame(size=0.1, origin=camera)

    
    T = np.eye(4)
    T[2, 2] = -1 
    # camera_frame.transform(T)
    camera_frame.scale(scale, np.zeros([3, 1]))
    return camera_frame

    fuck_frame = copy.deepcopy(camera_frame)
    fuck_frame.scale(0.01, fuck_frame.get_center())
    # return fuck_frame
    pass
