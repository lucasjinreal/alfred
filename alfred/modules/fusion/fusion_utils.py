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

    pts = lidar_to_cam_frame(lidar_points, frame_calib)

    # The given image is assumed to be a 2D image
    if not im_size:
        point_cloud = pts.T
        return point_cloud

    else:
        # Only keep points in front of camera (positive z)
        pts = pts[pts[:, 2] > 0]
        point_cloud = pts.T

        # Project to image frame
        point_in_im = calib_utils.project_to_image(point_cloud, p=frame_calib.p2).T

        # Filter based on the given image size
        image_filter = (point_in_im[:, 0] > 0) & \
                       (point_in_im[:, 0] < im_size[0]) & \
                       (point_in_im[:, 1] > 0) & \
                       (point_in_im[:, 1] < im_size[1])

    if not min_intensity:
        return pts[image_filter].T

    else:
        intensity_filter = i > min_intensity
        point_filter = np.logical_and(image_filter, intensity_filter)
        return pts[point_filter].T