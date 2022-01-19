

from alfred.vis.mesh3d.o3dsocket import VisOpen3DSocket
from alfred.vis.mesh3d.o3d_visconfig import Config, get_default_visconfig


def main():
    cfg = get_default_visconfig()
    # for vis in different format, support 25 keypoints, 24 keypoints etc.
    cfg.body_model.args.body_type = 'smpl'
    # cfg.body_model.args.body_type = 'body25'
    server = VisOpen3DSocket(cfg=cfg)
    while True:
        server.update()


if __name__ == "__main__":
    main()
