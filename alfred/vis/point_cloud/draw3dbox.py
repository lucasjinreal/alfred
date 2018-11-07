"""
draw 3d box directly on point cloud

it's just a wrapper in python calling
C++ source codes

"""
import numpy as np
from mayavi.mlab import points3d, figure
from mayavi import mlab


def show_3d_point(points):
    """
    show a (n, 3) array of point cloud
    :param points:
    :return:
    """
    if points.shape[1] != 3:
        points = np.asarray(points).T
    print('show shape: ', points.shape)
    fig = figure(bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), size=(1400, 800))
    points3d(points[:, 0], points[:, 1], points[:, 2], mode='sphere', colormap='gnuplot', scale_factor=0.05, figure=fig)

    # mlab.view(75.0, 140.0, [30., 30., 30.])
    mlab.roll(360)
    mlab.move(3, -1, 3.2)
    print(mlab.view())
    mlab.show()