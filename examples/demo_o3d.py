import socket
import time
from alfred.vis.mesh3d.utils import BaseSocketClient
import os
import numpy as np
import json


def send_body25(client):
    crt_d = os.path.dirname(__file__)
    aa = json.load(open(os.path.join(crt_d, 'data/keyp3d.json')))
    for d in aa:
        pid = d['id'] if 'id' in d.keys() else d['personID']
        pose3d = np.array(d['keypoints3d'], dtype=np.float32)
        if pose3d.shape[0] > 25:
            pose3d[25, :] = pose3d[7, :]
            pose3d[46, :] = pose3d[4, :]
        if pose3d.shape[1] == 3:
            pose3d = np.hstack([pose3d, np.ones((pose3d.shape[0], 1))])
        a = [{
            'id': pid,
            'keypoints3d': pose3d
        }]
        client.send(a)
        time.sleep(0.05)


def send_smpl24(client):
    crt_d = os.path.dirname(__file__)
    aa = json.load(open('/media/jintian/samsung/source/ai/swarm/toolchains/mmkd/a.json'))
    for d in aa:
        pid = d['id'] if 'id' in d.keys() else d['personID']
        pose3d = np.array(d['keypoints3d'], dtype=np.float32)
        a = [{
            'id': pid,
            'keypoints3d': pose3d
        }]
        client.send(a)
        time.sleep(0.05)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9999)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--smpl', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.host == 'auto':
        args.host = socket.gethostname()
    client = BaseSocketClient(args.host, args.port)
    if args.smpl:
        send_smpl24(client)
    else:
        send_body25(client)
