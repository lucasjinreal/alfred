import cv2
import numpy as np
from transformations import (
    reflection_from_matrix,
    identity_matrix,
    reflection_matrix,
    scale_from_matrix,
    scale_matrix,
    rotation_matrix,
)


class Camera:
    def __init__(self, R, t, K, dist=None, name=""):
        self.R = R.copy()
        self.t = t.copy()
        self.K = K.copy()
        self.dist = dist

        self.name = name

    def update_after_crop(self, bbox):
        left, upper, right, lower = bbox

        cx, cy = self.K[0, 2], self.K[1, 2]

        new_cx = cx - left
        new_cy = cy - upper

        self.K[0, 2], self.K[1, 2] = new_cx, new_cy

    def update_after_resize(self, image_shape, new_image_shape):
        height, width = image_shape
        new_width, new_height = new_image_shape

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        new_fx = fx * (new_width / width)
        new_fy = fy * (new_height / height)
        new_cx = cx * (new_width / width)
        new_cy = cy * (new_height / height)

        self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2] = (
            new_fx,
            new_fy,
            new_cx,
            new_cy,
        )

    @property
    def projection(self):
        return self.K.dot(self.extrinsics)

    @property
    def extrinsics(self):
        return np.hstack([self.R, self.t])


def generate_grid_mesh(start, end, step=1.0):
    num_point_per_line = int((end - start) // step + 1)
    its = np.linspace(start, end, num_point_per_line)
    line = []
    color = []
    common_line_color = [192, 192, 192]
    for i in range(num_point_per_line):
        line.append([its[0], its[i], 0, its[-1], its[i], 0])
        if its[i] == 0:
            color.append([0, 255, 0])
        else:
            color.append(common_line_color)

    for i in range(num_point_per_line):
        line.append([its[i], its[-1], 0, its[i], its[0], 0])
        if its[i] == 0:
            color.append([0, 0, 255])
        else:
            color.append(common_line_color)

    return np.array(line, dtype=np.float32), np.array(color, dtype=np.uint8)


def euclidean_to_homogeneous(points):
    if isinstance(points, np.ndarray):
        return np.hstack([points, np.ones((len(points), 1))])
    else:
        raise TypeError("Works only with numpy arrays")


def homogeneous_to_euclidean(points):
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    else:
        raise TypeError("Works only with numpy arrays")


def projection_to_2d_plane(vertices, projection_matrix, view_matrix=None, scale=None):
    if view_matrix is not None:
        vertices = (
            homogeneous_to_euclidean(
                (euclidean_to_homogeneous(vertices) @ view_matrix.T)
                @ projection_matrix.T
            )[:, :2]
        ) * scale

        vertices[:, 1] = scale - vertices[:, 1]
        vertices[:, 0] = vertices[:, 0] + scale
    else:
        vertices = euclidean_to_homogeneous(vertices) @ projection_matrix.T
        vertices = homogeneous_to_euclidean(vertices)
    return vertices.astype(np.int32)


def look_at(eye, center, up):
    f = unit_vector(center - eye)
    u = unit_vector(up)
    s = unit_vector(np.cross(f, u))
    u = np.cross(s, f)

    result = identity_matrix()
    result[:3, 0] = s
    result[:3, 1] = u
    result[:3, 2] = -f
    result[3, 0] = -np.dot(s, eye)
    result[3, 1] = -np.dot(u, eye)
    result[3, 2] = np.dot(f, eye)
    return result.T


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def update_camera_vectors():
    global front
    front_temp = np.zeros((3,))
    front_temp[0] = np.cos(np.radians(yaw)) * np.cos(np.radians(pitch))
    front_temp[1] = np.sin(np.radians(pitch))
    front_temp[2] = np.sin(np.radians(yaw)) * np.cos(np.radians(pitch))
    front = unit_vector(front_temp)
    global right
    right = unit_vector(np.cross(front, world_up))


camera_vertices = np.array(
    [
        [0, 0, 0],
        [-1, -1, 2],
        [0, 0, 0],
        [1, 1, 2],
        [0, 0, 0],
        [1, -1, 2],
        [0, 0, 0],
        [-1, 1, 2],
        [-1, 1, 2],
        [-1, -1, 2],
        [-1, -1, 2],
        [1, -1, 2],
        [1, -1, 2],
        [1, 1, 2],
        [1, 1, 2],
        [-1, 1, 2],
    ],
    dtype=np.float32,
)

human36m_connectivity_dict = [
    (0, 1),
    (1, 2),
    (2, 6),
    (5, 4),
    (4, 3),
    (3, 6),
    (6, 7),
    (7, 8),
    (8, 16),
    (9, 16),
    (8, 12),
    (11, 12),
    (10, 11),
    (8, 13),
    (13, 14),
    (14, 15),
]

multiview_data = np.load(
    "D:/Downloads/3D_sence_multiview.npy", allow_pickle=True
).tolist()
subject_name, camera_name, action_name, camera_configs, labels = (
    multiview_data["subject_names"],
    multiview_data["camera_names"],
    multiview_data["action_names"],
    multiview_data["cameras"],
    multiview_data["table"],
)
print(subject_name, camera_name)

camera_name = [str(i) for i, c in enumerate(camera_name)]

# subject_name ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
# action_name ['Directions-1', 'Directions-2', 'Discussion-1', 'Discussion-2', 'Eating-1', 'Eating-2', 'Greeting-1', 'Greeting-2', 'Phoning-1', 'Phoning-2', 'Posing-1', 'Posing-2', 'Purchases-1', 'Purchases-2', 'Sitting-1', 'Sitting-2', 'SittingDown-1', 'SittingDown-2', 'Smoking-1', 'Smoking-2', 'TakingPhoto-1', 'TakingPhoto-2', 'Waiting-1', 'Waiting-2', 'Walking-1', 'Walking-2', 'WalkingDog-1', 'WalkingDog-2', 'WalkingTogether-1', 'WalkingTogether-2']


specific_subject = "S9"
specific_action = "WalkingDog-2"
mask_subject = labels["subject_idx"] == subject_name.index(specific_subject)
actions = [action_name.index(specific_action)]
mask_actions = np.isin(labels["action_idx"], actions)
mask_subject = mask_subject & mask_actions
indices = []
indices.append(np.nonzero(mask_subject)[0])
specific_label = labels[np.concatenate(indices)]
specific_3d_skeleton = specific_label["keypoints"]

specific_camera_config = camera_configs[subject_name.index(specific_subject)]
specific_camera_config = [
    Camera(
        specific_camera_config["R"][i],
        specific_camera_config["t"][i],
        specific_camera_config["K"][i],
    )
    for i in range(len(camera_name))
]

# first person setup
yaw = -125
pitch = -15
world_up = np.array([0.0, 1.0, 0.0])
position = np.array([5000, 2500, 7557])
front = np.array([0.0, 0.0, -1.0])
right = np.array([0.0, 0.0, 0.0])

grid_vertices, grid_color = generate_grid_mesh(-4, 4, step=1)
grid_vertices = grid_vertices.reshape(-1, 3)

rorate_x_90 = rotation_matrix(np.radians(-90), (1, 0, 0))

frame_size = 900
original_video_frame_size = 1000
frame = np.zeros([frame_size, frame_size])

for i in range(len(camera_name)):
    specific_camera_config[i].update_after_resize(
        (original_video_frame_size,) * 2, (frame_size,) * 2
    )

update_camera_vectors()
view_matrix = look_at(position, position + front, world_up)
print(view_matrix)

projection_matrix = np.array(
    [[2.41421, 0, 0, 0], [0, 2.41421, 0, 0], [0, 0, -1, -0.2], [0, 0, -1, 0]],
    dtype=np.float32,
)

o_view_matrix = view_matrix.copy()
o_projection_matrix = projection_matrix.copy()

total_frame = specific_3d_skeleton.shape[0]
frame_index = 0

view_camera_index = -1
while True:
    if frame_index == total_frame:
        frame_index = 0
    frame = np.zeros([frame_size, frame_size, 3])
    if view_camera_index >= 0:
        view_matrix = None
        projection_matrix = specific_camera_config[view_camera_index].projection
    else:
        view_matrix = o_view_matrix
        projection_matrix = o_projection_matrix
    print(projection_matrix)

    grid_vertices_project = grid_vertices @ (
        np.eye(3) if view_matrix is None else rorate_x_90[:3, :3].T
    )
    print(grid_vertices_project)
    grid_vertices_project = grid_vertices_project @ scale_matrix(650)[:3, :3].T
    grid_vertices_project = projection_to_2d_plane(
        grid_vertices_project, projection_matrix, view_matrix, int(frame_size / 2)
    ).reshape(-1, 4)
    print(grid_vertices_project)
    # draw line
    for index, line in enumerate(grid_vertices_project):
        cv2.line(
            frame, (line[0], line[1]), (line[2], line[3]), grid_color[index].tolist()
        )

    # draw camera
    for camera_index, conf in enumerate(specific_camera_config):
        if view_camera_index == camera_index:
            continue
        m_rt = identity_matrix()
        r = np.array(conf.R, dtype=np.float32).T
        m_rt[:-1, -1] = -r @ np.array(conf.t, dtype=np.float32).squeeze()
        m_rt[:-1, :-1] = r

        m_s = identity_matrix() * 200
        m_s[3, 3] = 1

        camera_vertices_convert = homogeneous_to_euclidean(
            euclidean_to_homogeneous(camera_vertices)
            @ ((np.eye(4) if view_matrix is None else rorate_x_90) @ m_rt @ m_s).T
        )

        camera_vertices_convert = projection_to_2d_plane(
            camera_vertices_convert, projection_matrix, view_matrix, int(frame_size / 2)
        )
        camera_vertices_convert = camera_vertices_convert.reshape(-1, 4)
        for index, line in enumerate(camera_vertices_convert):
            cv2.line(
                frame,
                (line[0], line[1]),
                (line[2], line[3]),
                (0, 153, 255),
                thickness=1,
            )
        cv2.putText(
            frame,
            camera_name[camera_index],
            (camera_vertices_convert[1, 0], camera_vertices_convert[1, 1] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
        )

    specific_3d_skeleton_project = specific_3d_skeleton[frame_index].reshape(-1, 3)
    specific_3d_skeleton_project = (
        specific_3d_skeleton_project
        @ (np.eye(3) if view_matrix is None else rorate_x_90[:3, :3]).T
    )
    specific_3d_skeleton_project = (
        specific_3d_skeleton_project @ np.eye(3, dtype=np.float32) * 1
    )
    specific_3d_skeleton_project = projection_to_2d_plane(
        specific_3d_skeleton_project,
        projection_matrix,
        view_matrix,
        int(frame_size / 2),
    ).reshape(17, 2)

    print(specific_3d_skeleton_project)
    for c in human36m_connectivity_dict:
        cv2.line(
            frame,
            (*specific_3d_skeleton_project[c[0]],),
            (*specific_3d_skeleton_project[c[1]],),
            (100, 155, 255),
            thickness=2,
        )
        cv2.circle(frame, (*specific_3d_skeleton_project[c[0]],), 3, (0, 0, 255), -1)
        cv2.circle(frame, (*specific_3d_skeleton_project[c[1]],), 3, (0, 0, 255), -1)

    frame_index += 1
    print(frame_index)
    cv2.imshow(specific_action, frame)
    if cv2.waitKey(6) & 0xFF == ord("1"):
        view_camera_index += 1
        if view_camera_index == 4:
            view_camera_index = -1

    if cv2.waitKey(2) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
