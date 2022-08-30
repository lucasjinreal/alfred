import numpy as np
import cv2

try:
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
except ImportError:
    plt = None

color_hand_joints = [
    [1.0, 0.0, 0.0],
    [0.0, 0.4, 0.0],
    [0.0, 0.6, 0.0],
    [0.0, 0.8, 0.0],
    [0.0, 1.0, 0.0],  # thumb
    [0.0, 0.0, 0.6],
    [0.0, 0.0, 1.0],
    [0.2, 0.2, 1.0],
    [0.4, 0.4, 1.0],  # index
    [0.0, 0.4, 0.4],
    [0.0, 0.6, 0.6],
    [0.0, 0.8, 0.8],
    [0.0, 1.0, 1.0],  # middle
    [0.4, 0.4, 0.0],
    [0.6, 0.6, 0.0],
    [0.8, 0.8, 0.0],
    [1.0, 1.0, 0.0],  # ring
    [0.4, 0.0, 0.4],
    [0.6, 0.0, 0.6],
    [0.8, 0.0, 0.8],
    [1.0, 0.0, 1.0],
]  # little


def vis_hand_pose(image, pose_uv):
    """
    :param image: H x W x 3
    :param pose_uv: 21 x 2
    wrist,
    thumb_mcp, thumb_pip, thumb_dip, thumb_tip
    index_mcp, index_pip, index_dip, index_tip,
    middle_mcp, middle_pip, middle_dip, middle_tip,
    ring_mcp, ring_pip, ring_dip, ring_tip,
    little_mcp, little_pip, little_dip, little_tip
    :return:
    """
    assert pose_uv.shape[0] == 21
    skeleton_overlay = image.copy()
    marker_sz = 5
    line_wd = 3
    root_ind = 0

    for joint_ind in range(pose_uv.shape[0]):
        joint = pose_uv[joint_ind, 0].astype("int32"), pose_uv[joint_ind, 1].astype(
            "int32"
        )
        cv2.circle(
            skeleton_overlay,
            joint,
            radius=marker_sz,
            color=color_hand_joints[joint_ind] * np.array(255),
            thickness=-1,
            lineType=cv2.CV_AA if cv2.__version__.startswith("2") else cv2.LINE_AA,
        )
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            root_joint = pose_uv[root_ind, 0].astype("int32"), pose_uv[
                root_ind, 1
            ].astype("int32")
            cv2.line(
                skeleton_overlay,
                root_joint,
                joint,
                color=color_hand_joints[joint_ind] * np.array(255),
                thickness=int(line_wd),
                lineType=cv2.CV_AA if cv2.__version__.startswith("2") else cv2.LINE_AA,
            )
        else:
            joint_2 = pose_uv[joint_ind - 1, 0].astype("int32"), pose_uv[
                joint_ind - 1, 1
            ].astype("int32")
            cv2.line(
                skeleton_overlay,
                joint_2,
                joint,
                color=color_hand_joints[joint_ind] * np.array(255),
                thickness=int(line_wd),
                lineType=cv2.CV_AA if cv2.__version__.startswith("2") else cv2.LINE_AA,
            )
    return skeleton_overlay


"""
Drawing 3d mesh using matplotlib
"""


def fig2data(fig):
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


camera_shape = [
    [
        [-0.05, 0.05, 0.05, -0.05, -0.05],
        [-0.05, -0.05, 0.05, 0.05, -0.05],
        [0, 0, 0, 0, 0],
    ],
    [[0.05, 0], [0.05, 0], [0, -0.1]],
    [[0.05, 0], [-0.05, 0], [0, -0.1]],
    [[-0.05, 0], [-0.05, 0], [0, -0.1]],
    [[-0.05, 0], [0.05, 0], [0, -0.1]],
]
camera_color = (0, 0, 200 / 255)


def vis_3d_mesh_matplot_3d_coordinates(mesh_xyz, image_size, face):
    """
    :param mesh_xyz: 778 x 3
    :param image_size: H, W
    :param face: 1538 x 3
    :return:
    """
    fig = plt.figure()
    fig.set_size_inches(
        float(image_size[1]) / fig.dpi, float(image_size[0]) / fig.dpi, forward=True
    )

    ax = plt.subplot(111, projection="3d")

    for i, shape in enumerate(camera_shape):
        ax.plot(
            shape[0],
            shape[1],
            shape[2],
            color=camera_color,
            linestyle=(":", "-")[i == 0],
        )

    triang = mtri.Triangulation(mesh_xyz[:, 0], mesh_xyz[:, 1], triangles=face)
    ax.plot_trisurf(triang, mesh_xyz[:, 2], color=(145 / 255, 181 / 255, 255 / 255))

    ax.axis("auto")
    x_lim = [-0.1, 0.1, 0.02]
    y_lim = [-0.1, 0.12, 0.02]
    z_lim = [0.0, 0.8, 0.1]
    x_ticks = np.arange(x_lim[0], x_lim[1], step=x_lim[2])
    y_ticks = np.arange(y_lim[0], y_lim[1], step=y_lim[2])
    z_ticks = np.arange(z_lim[0], z_lim[1], step=z_lim[2])
    plt.xticks(
        x_ticks, [x_lim[0], "", "", "", "", 0, "", "", "", x_lim[1]], fontsize=14
    )
    plt.yticks(
        y_ticks, [y_lim[0], "", "", "", "", 0, "", "", "", -y_lim[0], ""], fontsize=14
    )
    ax.set_zticks(z_ticks)
    z_ticks = [""] * (z_ticks.shape[0])
    z_ticks[4] = 0.4
    ax.set_zticklabels(z_ticks, fontsize=14)
    ax.view_init(elev=140, azim=80)
    plt.subplots_adjust(left=-0.06, right=1, top=0.95, bottom=-0.06, wspace=0, hspace=0)
    plt.subplots_adjust(
        left=-0.06, right=0.98, top=0.93, bottom=-0.07, wspace=0, hspace=0
    )

    ret = fig2data(fig)
    plt.close(fig)
    return ret


def vis_3d_mesh_on_img(image, cam_param, mesh_xyz, face):
    """
    :param image: H x W x 3
    :param cam_param: 1 x 3 x 3
    :param mesh_xyz: 778 x 3
    :param face: 1538 x 3 x 2
    :return:
    """
    vertex2uv = np.matmul(cam_param, mesh_xyz.T).T
    vertex2uv = (vertex2uv / vertex2uv[:, 2:3])[:, :2].astype(np.int)

    fig = plt.figure()
    fig.set_size_inches(
        float(image.shape[1]) / fig.dpi, float(image.shape[0]) / fig.dpi, forward=True
    )
    plt.imshow(image)
    plt.axis("off")
    if face is None:
        plt.plot(vertex2uv[:, 0], vertex2uv[:, 1], "o", color="green", markersize=1)
    else:
        plt.triplot(vertex2uv[:, 0], vertex2uv[:, 1], face, lw=0.5, color="orange")
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0, wspace=0, hspace=0)
    ret = fig2data(fig)
    plt.close(fig)
    return ret
