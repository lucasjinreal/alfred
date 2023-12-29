from alfred.vis.mesh3d.o3dsocket import VisOpen3DSocket
from alfred.vis.mesh3d.o3d_visconfig import Config, get_default_visconfig, CONFIG
import keyboard


def main():
    cfg = get_default_visconfig()
    # cfg.body_model.args.body_type = 'smpl'
    # cfg.body_model.args.body_type = 'body25'
    cfg.body_model.args.body_type = 'h36m'

    server = VisOpen3DSocket(cfg=cfg)
    while True:
        server.update()
        if keyboard.is_pressed("q"):
            server.close()
            break


if __name__ == "__main__":
    main()
