import socket
import time
from alfred.vis.mesh3d.utils import BaseSocketClient
import os
import numpy as np
import json
from os.path import join
from glob import glob
from tqdm import tqdm


def read_json(path):
    assert os.path.exists(path), path
    with open(path) as f:
        data = json.load(f)
    return data


def read_keypoints3d(filename):
    data = read_json(filename)
    res_ = []
    for d in data:
        pid = d['id'] if 'id' in d.keys() else d['personID']
        pose3d = np.array(d['keypoints3d'], dtype=np.float32)
        if pose3d.shape[0] > 25:
            # 对于有手的情况，把手的根节点赋值成body25上的点
            pose3d[25, :] = pose3d[7, :]
            pose3d[46, :] = pose3d[4, :]
        if pose3d.shape[1] == 3:
            pose3d = np.hstack([pose3d, np.ones((pose3d.shape[0], 1))])
        res_.append({
            'id': pid,
            'keypoints3d': pose3d
        })
    return res_


def send_dir(client, path, step):

    results = sorted(glob(join(path, '*.json')))
    for result in tqdm(results[::step]):
        if args.smpl:
            data = read_smpl(result)
            client.send_smpl(data)
        else:
            data = read_keypoints3d(result)
            client.send(data)
        time.sleep(0.005)


def send_rand(client):
    N_person = 3
    datas = []
    for i in range(N_person):
        transl = (np.random.rand(1, 3) - 0.5) * 3
        kpts = np.random.rand(25, 4)
        kpts[:, :3] += transl
        data = {
            'id': i,
            'keypoints3d': kpts
        }
        datas.append(data)
    for _ in range(1000000):
        for i in range(N_person):
            move = (np.random.rand(1, 3) - 0.5) * 0.1
            datas[i]['keypoints3d'][:, :3] += move
        client.send(datas)
        time.sleep(0.5)
    client.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9999)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--smpl', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.host == 'auto':
        args.host = socket.gethostname()
    client = BaseSocketClient(args.host, args.port)

    if args.path is not None and os.path.isdir(args.path):
        send_dir(client, args.path, step=args.step)
    else:
        send_rand(client)
