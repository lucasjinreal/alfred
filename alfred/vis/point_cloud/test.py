import numpy as np
from open3d import *


import pcl
import pcl.visualization


def vis_pcl_python():
    ## Visualization
    visual = pcl.pcl_visualization.CloudViewing()
    visual.ShowColorACloud(cloud, b'cloud')

    v = True
    while v:
        v = not (visual.WasStopped())