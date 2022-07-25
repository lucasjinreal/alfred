from .pose_dataset_info import DatasetInfo
from .pose_dataset_info import get_dataset_info_by_name
import cv2
import numpy as np
import math
from alfred.vis.image.common import colors

BUILTIN_JOINTS_17_COCO = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]

BUILTIN_JOINTS_26_HALPE = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # Head
    (5, 18),
    (6, 18),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # Body
    (17, 18),
    (18, 19),
    (19, 11),
    (19, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
    (20, 24),
    (21, 25),
    (23, 25),
    (22, 24),
    (15, 24),
    (16, 25),
]

BUILTIN_JOINT_28_COMBINED = []


def vis_pose_by_joints(
    img,
    poses,
    joints,
    color=colors(np.random.randint(30)),
    point_color=[252, 219, 3],
    point_size=6,
    line_width=2,
    face_connect=False,
    aa=False,
):
    if len(poses.shape) == 2:
        poses = np.array([poses])
        N, num_kps, _ = poses.shape
    else:
        N, num_kps, _ = poses.shape
    assert isinstance(joints, np.ndarray), "joints must be numpy array!"
    if len(joints.shape) == 3:
        joints = joints[0]

    for pose in poses:
        # for part_id in range(num_kps - 1):
        for part_id in range(len(joints)):
            # if num_kps is 18, then 0 ... 17
            kpt_a_id = joints[part_id][0] - 1
            global_kpt_a = pose[kpt_a_id].tolist()
            x_a, y_a, _ = global_kpt_a
            if x_a > 0 and y_a > 0:
                cv2.circle(img, (int(x_a), int(y_a)), 3, color, -1, cv2.LINE_AA)

            kpt_b_id = joints[part_id][1] - 1
            global_kpt_b = pose[kpt_b_id].tolist()
            x_b, y_b, _ = global_kpt_b
            if x_b > 0 and y_b > 0:
                x_b, y_b, _ = global_kpt_b
                if (
                    kpt_a_id not in [0, 1, 2, 3, 4]
                    and kpt_b_id not in [0, 1, 2, 3, 4]
                    and not face_connect
                ):
                    cv2.circle(
                        img,
                        (int(x_b), int(y_b)),
                        point_size + 1,
                        point_color,
                        -1,
                        cv2.LINE_AA if aa else cv2.LINE_8,
                    )
                else:
                    cv2.circle(
                        img,
                        (int(x_b), int(y_b)),
                        point_size,
                        point_color,
                        -1,
                        cv2.LINE_AA if aa else cv2.LINE_8,
                    )
            if x_a > 0 and y_a > 0 and x_b > 0 and y_b > 0:
                if face_connect:
                    cv2.line(
                        img,
                        (int(x_a), int(y_a)),
                        (int(x_b), int(y_b)),
                        color,
                        line_width,
                        cv2.LINE_AA if aa else cv2.LINE_8,
                    )
                else:
                    if kpt_a_id not in [0, 1, 2, 3, 4] and kpt_b_id not in [
                        0,
                        1,
                        2,
                        3,
                        4,
                    ]:
                        cv2.line(
                            img,
                            (int(x_a), int(y_a)),
                            (int(x_b), int(y_b)),
                            color,
                            line_width,
                            cv2.LINE_AA if aa else cv2.LINE_8,
                        )
    return img


def vis_pose_coco_17(
    img,
    poses,
    color=colors(np.random.randint(30)),
    point_color=[252, 219, 3],
    point_size=6,
):
    joints = BUILTIN_JOINTS_17_COCO
    assert isinstance(poses, np.ndarray), "poses must be numpy array!"
    N, num_kps, last_dim = poses.shape
    for pose in poses:
        # for part_id in range(num_kps - 1):
        for part_id in range(len(joints)):
            # if num_kps is 18, then 0 ... 17
            kpt_a_id = joints[part_id][0] - 1
            global_kpt_a = pose[kpt_a_id].tolist()
            x_a, y_a, _ = global_kpt_a
            if x_a >= 0 and y_a >= 0:
                cv2.circle(img, (int(x_a), int(y_a)), 3, color, -1, cv2.LINE_AA)

            kpt_b_id = joints[part_id][1] - 1
            global_kpt_b = pose[kpt_b_id].tolist()
            x_b, y_b, _ = global_kpt_b
            if x_b >= 0 and y_b >= 0:
                x_b, y_b, _ = global_kpt_b
                cv2.circle(
                    img, (int(x_b), int(y_b)), point_size, point_color, -1, cv2.LINE_AA
                )
            if (
                global_kpt_a[0] >= 0
                and global_kpt_a[1] >= 0
                and global_kpt_b[0] >= 0
                and global_kpt_b[1] >= 0
            ):
                cv2.line(
                    img,
                    (int(x_a), int(y_a)),
                    (int(x_b), int(y_b)),
                    color,
                    2,
                    cv2.LINE_AA,
                )
    return img


def vis_pose_result(
    img,
    pose_result,
    radius=4,
    thickness=1,
    kpt_score_thr=0.3,
    dataset="TopDownCocoDataset",
    dataset_info=None,
    show=False,
    out_file=None,
):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str | np.ndarray): Image filename or loaded image.
        pose_result (list[dict]): The results to draw over `img`, [N, 17, 3] for coco body
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple()]): Default None.
        show (bool):  Whether to show the image. Default True.
        out_file (str|None): The filename of the output visualization image.
    """
    if isinstance(pose_result, np.ndarray):
        if len(pose_result.shape) < 3:
            # if shape is 2, expand to 3
            pose_result = np.expand_dims(pose_result, axis=0)

    # get dataset info
    if dataset_info is None:
        dataset_info = get_dataset_info_by_name(dataset)
        assert (
            dataset_info is not None
        ), "{} dataset not supported built in, you can specific dataset_info manually.".format(
            dataset
        )
    else:
        dataset_info = DatasetInfo(dataset_info)

    skeleton = dataset_info.skeleton
    pose_link_color = dataset_info.pose_link_color
    pose_kpt_color = dataset_info.pose_kpt_color
    img = imshow_keypoints(
        img,
        pose_result,
        skeleton=skeleton,
        kpt_score_thr=kpt_score_thr,
        pose_kpt_color=pose_kpt_color,
        pose_link_color=pose_link_color,
        radius=radius,
        thickness=thickness,
        show_keypoint_weight=False,
    )
    if show:
        cv2.imshow("pose result", img)
        cv2.waitKey(0)
    if out_file is not None:
        cv2.imwrite(out_file, img)
    return img


def imshow_keypoints(
    img,
    pose_result,
    skeleton=None,
    kpt_score_thr=0.3,
    pose_kpt_color=None,
    pose_link_color=None,
    radius=4,
    thickness=1,
    show_keypoint_weight=False,
):
    """Draw keypoints and links on an image.

    Args:
            img (str or Tensor): The image to draw poses on. If an image array
                is given, id will be modified in-place.
            pose_result (list[kpts]): The poses to draw. Each element kpts is
                a set of K keypoints as an Kx3 numpy.ndarray, where each
                keypoint is represented as x, y, score.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints. If None,
                the keypoint will not be drawn.
            pose_link_color (np.array[Mx3]): Color of M links. If None, the
                links will not be drawn.
            thickness (int): Thickness of lines.
    """
    img_h, img_w, _ = img.shape

    for kpts in pose_result:
        kpts = np.array(kpts, copy=False)

        # draw each point on image
        if pose_kpt_color is not None:
            if len(pose_kpt_color) != len(kpts):
                c = colors(np.random.randint(50))
                pose_kpt_color = [c for _ in range(len(kpts))]
            # assert len(pose_kpt_color) == len(
            # kpts), 'pose_kpt_color: {} not equal kpts: {}'.format(len(pose_kpt_color), len(kpts))
            for kid, kpt in enumerate(kpts):
                if len(kpt) > 2:
                    x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
                    if kpt_score > kpt_score_thr:
                        if show_keypoint_weight:
                            img_copy = img.copy()
                            r, g, b = pose_kpt_color[kid]
                            cv2.circle(
                                img_copy,
                                (int(x_coord), int(y_coord)),
                                radius,
                                (int(r), int(g), int(b)),
                                -1,
                                cv2.LINE_AA,
                            )
                            transparency = max(0, min(1, kpt_score))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img,
                            )
                        else:
                            r, g, b = pose_kpt_color[kid]
                            cv2.circle(
                                img,
                                (int(x_coord), int(y_coord)),
                                radius,
                                (int(r), int(g), int(b)),
                                -1,
                                cv2.LINE_AA,
                            )
                else:
                    x_coord, y_coord = int(kpt[0]), int(kpt[1])
                    if show_keypoint_weight:
                        img_copy = img.copy()
                        r, g, b = pose_kpt_color[kid]
                        cv2.circle(
                            img_copy,
                            (int(x_coord), int(y_coord)),
                            radius,
                            (int(r), int(g), int(b)),
                            -1,
                            cv2.LINE_AA,
                        )
                        transparency = max(0, min(1, kpt_score))
                        cv2.addWeighted(
                            img_copy, transparency, img, 1 - transparency, 0, dst=img
                        )
                    else:
                        r, g, b = pose_kpt_color[kid]
                        cv2.circle(
                            img,
                            (int(x_coord), int(y_coord)),
                            radius,
                            (int(r), int(g), int(b)),
                            -1,
                            cv2.LINE_AA,
                        )

        # draw links
        if skeleton is not None and pose_link_color is not None:
            assert len(pose_link_color) == len(skeleton)
            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
                if len(kpts[sk[0]]) > 2:
                    if (
                        pos1[0] > 0
                        and pos1[0] < img_w
                        and pos1[1] > 0
                        and pos1[1] < img_h
                        and pos2[0] > 0
                        and pos2[0] < img_w
                        and pos2[1] > 0
                        and pos2[1] < img_h
                        and kpts[sk[0], 2] > kpt_score_thr
                        and kpts[sk[1], 2] > kpt_score_thr
                    ):
                        r, g, b = pose_link_color[sk_id]
                        if show_keypoint_weight:
                            img_copy = img.copy()
                            X = (pos1[0], pos2[0])
                            Y = (pos1[1], pos2[1])
                            mX = np.mean(X)
                            mY = np.mean(Y)
                            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                            stickwidth = 2
                            polygon = cv2.ellipse2Poly(
                                (int(mX), int(mY)),
                                (int(length / 2), int(stickwidth)),
                                int(angle),
                                0,
                                360,
                                1,
                            )
                            cv2.fillConvexPoly(
                                img_copy, polygon, (int(r), int(g), int(b))
                            )
                            transparency = max(
                                0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2]))
                            )
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img,
                            )
                        else:
                            cv2.line(
                                img,
                                pos1,
                                pos2,
                                (int(r), int(g), int(b)),
                                thickness=thickness,
                                lineType=cv2.LINE_AA,
                            )
                else:
                    if (
                        pos1[0] > 0
                        and pos1[0] < img_w
                        and pos1[1] > 0
                        and pos1[1] < img_h
                        and pos2[0] > 0
                        and pos2[0] < img_w
                        and pos2[1] > 0
                        and pos2[1] < img_h
                    ):
                        r, g, b = pose_link_color[sk_id]
                        if show_keypoint_weight:
                            img_copy = img.copy()
                            X = (pos1[0], pos2[0])
                            Y = (pos1[1], pos2[1])
                            mX = np.mean(X)
                            mY = np.mean(Y)
                            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                            stickwidth = 2
                            polygon = cv2.ellipse2Poly(
                                (int(mX), int(mY)),
                                (int(length / 2), int(stickwidth)),
                                int(angle),
                                0,
                                360,
                                1,
                            )
                            cv2.fillConvexPoly(
                                img_copy, polygon, (int(r), int(g), int(b))
                            )
                            transparency = max(
                                0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2]))
                            )
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img,
                            )
                        else:
                            cv2.line(
                                img,
                                pos1,
                                pos2,
                                (int(r), int(g), int(b)),
                                thickness=thickness,
                                lineType=cv2.LINE_AA,
                            )

    return img
