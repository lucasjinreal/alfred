"""
this file contains some fusion process utils

such as:

get a set of point cloud of a particular angle of an image
we said crop point cloud to that image

"""
import numpy as np


class FrameCalibrationData(object):
    """
    3x4    p0-p3      Camera P matrix. Contains extrinsic
                          and intrinsic parameters.

    3x3    r0_rect    Rectification matrix, required to transform points
                      from velodyne to camera coordinate frame.

    3x4    tr_velodyne_to_cam    Used to transform from velodyne to cam
                                 coordinate frame according to:
                                 Point_Camera = P_cam * R0_rect *
                                                Tr_velo_to_cam *
                                                Point_Velodyne.
    """
    def __init__(self):
        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.r0_rect = []
        self.tr_velodyne_to_cam = []

    def __str__(self):
        return 'p0: {}\np1: {}\np2: {}\np3: {}\nr0_rect: {}\ntr_veodyne_to_cam: {}\n'.format(
            self.p0, self.p1, self.p2, self.p3, self.r0_rect, self.tr_velodyne_to_cam
        )


class StereoCalibrationData(object):
    """
    Stereo Calibration Holder
    1    baseline    distance between the two camera centers.
    1    f    focal length.
    3x3    k    intrinsic calibration matrix.
    3x4    p    camera matrix.
    1    center_u    camera origin u coordinate.
    1    center_v    camera origin v coordinate.
    """
    def __init__(self):
        self.baseline = 0.0
        self.f = 0.0
        self.k = []
        self.p = []
        self.center_u = 0.0
        self.center_v = 0.0


def lidar_to_cam_frame(xyz_lidar, frame_calib):
    """
    convert fusion to camera frame
    :param xyz_lidar:
    :param frame_calib:
    :return:
    """
    # Pad the r0_rect matrix to a 4x4
    r0_rect_mat = frame_calib.r0_rect
    r0_rect_mat = np.pad(r0_rect_mat, ((0, 1), (0, 1)),
                         'constant', constant_values=0)
    r0_rect_mat[3, 3] = 1

    # Pad the tr_vel_to_cam matrix to a 4x4
    tf_mat = frame_calib.tr_velodyne_to_cam
    tf_mat = np.pad(tf_mat, ((0, 1), (0, 0)),
                    'constant', constant_values=0)
    tf_mat[3, 3] = 1

    # Pad the pointcloud with 1's for the transformation matrix multiplication
    one_pad = np.ones(xyz_lidar.shape[0]).reshape(-1, 1)
    xyz_lidar = np.append(xyz_lidar, one_pad, axis=1)

    # p_cam = P2 * R0_rect * Tr_velo_to_cam * p_velo
    rectified = np.dot(r0_rect_mat, tf_mat)
    ret_xyz = np.dot(rectified, xyz_lidar.T)

    # Change to N x 3 array for consistency.
    return ret_xyz[0:3].T


def get_lidar_point_cloud(frame_calib, lidar_points, im_size=None, min_intensity=None):
    """ Calculates the fusion point cloud, and optionally returns only the
    points that are projected to the image.

    :param img_idx: image index
    :param calib_dir: directory with calibration files
    :param velo_dir: directory with velodyne files
    :param im_size: (optional) 2 x 1 list containing the size of the image
                      to filter the point cloud [w, h]
    :param min_intensity: (optional) minimum intensity required to keep a point

    :return: (3, N) point_cloud in the form [[x,...][y,...][z,...]]
    """
    if not isinstance(frame_calib, FrameCalibrationData):
        raise ValueError('frame_calib must be an FrameCalibrationData instance, construct one.')
    # lidar points should be [n, 4] which is [x, y, z, intensity]
    assert lidar_points.shape[1] == 4, '{get_lidar_point_cloud} ' \
                                       'lidar points should be [n, 4] which is [x, y, z, intensity]'

    pts = lidar_to_cam_frame(lidar_points[:, :-1], frame_calib)

    # print('pts after lidar to cam frame: ', pts)
    # print('frame_calib: {}'.format(frame_calib))
    # print('img size: ', im_size)

    # The given image is assumed to be a 2D image
    if not im_size:
        point_cloud = pts.T
        return point_cloud

    else:
        # Only keep points in front of camera (positive z)
        pts = pts[pts[:, 2] > 0]
        point_cloud = pts.T
        # print('im_size not none, now is: ',)
        # print('pts: ', pts)
        # print('point_cloud: ', point_cloud)

        # Project to image frame
        point_in_im = project_to_image(point_cloud, p=frame_calib.p2).T
        # print('point in im: ', point_in_im)

        # Filter based on the given image size
        image_filter = (point_in_im[:, 0] > 0) & \
                       (point_in_im[:, 0] < im_size[0]) & \
                       (point_in_im[:, 1] > 0) & \
                       (point_in_im[:, 1] < im_size[1])

    if not min_intensity:
        final_pts = pts[image_filter].T
        # print('final from get_lidar_point_cloud: ', final_pts)
        return final_pts

    else:
        intensity_filter = lidar_points[:, -1] > min_intensity
        point_filter = np.logical_and(image_filter, intensity_filter)
        return pts[point_filter].T


def project_to_image(point_cloud, p):
    """ Projects a 3D point cloud to 2D points for plotting

    :param point_cloud: 3D point cloud (3, N)
    :param p: Camera matrix (3, 4)

    :return: pts_2d: the image coordinates of the 3D points in the shape (2, N)
    """

    pts_2d = np.dot(p, np.append(point_cloud,
                                 np.ones((1, point_cloud.shape[1])),
                                 axis=0))

    pts_2d[0, :] = pts_2d[0, :] / pts_2d[2, :]
    pts_2d[1, :] = pts_2d[1, :] / pts_2d[2, :]
    pts_2d = np.delete(pts_2d, 2, 0)
    return pts_2d


def compute_box_corners_3d(t_xyz, lwh, ry):
    """
    Compute 3d box corner based on center and l, w, h, ry
    :param t_xyz:
    :param lwh
    :param ry:
    :return:
    """
    # Compute rotational matrix
    rot = np.array([[+np.cos(ry), 0, +np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, +np.cos(ry)]])
    l = lwh[0]
    w = lwh[1]
    h = lwh[2]
    # 3D BB corners
    x_corners = np.array(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

    corners_3d = np.dot(rot, np.array([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + t_xyz[0]
    corners_3d[1, :] = corners_3d[1, :] + t_xyz[1]
    corners_3d[2, :] = corners_3d[2, :] + t_xyz[2]
    return corners_3d


def project_box3d_to_image(corners_3d, p):
    """Computes the 3D bounding box projected onto
    image space.

    Keyword arguments:
    obj -- object file to draw bounding box
    p -- transform matrix

    Returns:
        corners : numpy array of corner points projected
        onto image space.
        face_idx: numpy array of 3D bounding box face
    """
    # index for 3d bounding box face
    # it is converted to 4x4 matrix
    face_idx = np.array([0, 1, 5, 4,  # front face
                         1, 2, 6, 5,  # left face
                         2, 3, 7, 6,  # back face
                         3, 0, 4, 7]).reshape((4, 4))  # right face
    return project_to_image(corners_3d, p), face_idx


# ------------------- Call this convert twhl format to corners
def convert_twhl_to_corners(tlwh_box, calib_p):
    t = tlwh_box[0: 3]
    lwh = tlwh_box[3: 6]
    ry = tlwh_box[6]
    corners3d = compute_box_corners_3d(t_xyz=t, lwh=lwh, ry=ry)
    corners, face_idx = project_box3d_to_image(corners3d, calib_p)
    return corners, face_idx
