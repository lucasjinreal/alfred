from .pose_dataset_info import DatasetInfo
from .pose_dataset_info import get_dataset_info_by_name
import cv2
import numpy as np
import math


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

    # get dataset info
    if dataset_info is None:
        dataset_info = get_dataset_info_by_name(dataset)
        assert dataset_info is None, '{} dataset not supported built in, you can specific dataset_info manually.'.format(dataset)
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
            assert len(pose_kpt_color) == len(kpts)
            for kid, kpt in enumerate(kpts):
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
                        cv2.fillConvexPoly(img_copy, polygon, (int(r), int(g), int(b)))
                        transparency = max(
                            0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2]))
                        )
                        cv2.addWeighted(
                            img_copy, transparency, img, 1 - transparency, 0, dst=img
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
