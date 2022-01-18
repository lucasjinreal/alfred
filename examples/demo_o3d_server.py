

from alfred.vis.mesh3d.o3dsocket import VisOpen3DSocket


def main():
    server = VisOpen3DSocket()
    while True:
        server.update()


if __name__ == "__main__":
    main()
