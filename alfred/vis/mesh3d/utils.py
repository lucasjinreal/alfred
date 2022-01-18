import time
import tabulate
import socket
import time
from threading import Thread
from queue import Queue
import cv2
import numpy as np
import os


def log(x):
    from datetime import datetime
    time_now = datetime.now().strftime("%m-%d-%H:%M:%S.%f ")
    print(time_now + x)


def myarray2string(array, separator=', ', fmt='%.3f', indent=8):
    assert len(array.shape) == 2, 'Only support MxN matrix, {}'.format(
        array.shape)
    blank = ' ' * indent
    res = ['[']
    for i in range(array.shape[0]):
        res.append(
            blank + '  ' + '[{}]'.format(separator.join([fmt % (d) for d in array[i]])))
        if i != array.shape[0] - 1:
            res[-1] += ', '
    res.append(blank + ']')
    return '\r\n'.join(res)


def write_common_results(dumpname=None, results=[], keys=[], fmt='%2.3f'):
    format_out = {'float_kind': lambda x: fmt % x}
    out_text = []
    out_text.append('[\n')
    for idata, data in enumerate(results):
        out_text.append('    {\n')
        output = {}
        output['id'] = data['id']
        for key in keys:
            if key not in data.keys():
                continue
            # BUG: This function will failed if the rows of the data[key] is too large
            # output[key] = np.array2string(data[key], max_line_width=1000, separator=', ', formatter=format_out)
            output[key] = myarray2string(data[key], separator=', ', fmt=fmt)
        for key in output.keys():
            out_text.append('        \"{}\": {}'.format(key, output[key]))
            if key != keys[-1]:
                out_text.append(',\n')
            else:
                out_text.append('\n')
        out_text.append('    }')
        if idata != len(results) - 1:
            out_text.append(',\n')
        else:
            out_text.append('\n')
    out_text.append(']\n')
    if dumpname is not None:
        os.makedirs(os.path.dirname(dumpname), exist_ok=True)
        with open(dumpname, 'w') as f:
            f.writelines(out_text)
    else:
        return ''.join(out_text)


def encode_detect(data):
    res = write_common_results(None, data, ['keypoints3d'])
    res = res.replace('\r', '').replace('\n', '').replace(' ', '')
    return res.encode('ascii')


def encode_smpl(data):
    res = write_common_results(
        None, data, ['poses', 'shapes', 'expression', 'Rh', 'Th'])
    res = res.replace('\r', '').replace('\n', '').replace(' ', '')
    return res.encode('ascii')


def encode_image(image):
    fourcc = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, img_encode = cv2.imencode('.jpg', image, fourcc)
    data = np.array(img_encode)
    stringData = data.tostring()
    return stringData


class BaseSocketClient:
    def __init__(self, host, port) -> None:
        if host == 'auto':
            host = socket.gethostname()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        self.s = s

    def send(self, data):
        val = encode_detect(data)
        self.s.send(bytes('{}\n'.format(len(val)), 'ascii'))
        self.s.sendall(val)

    def send_smpl(self, data):
        val = encode_smpl(data)
        self.s.send(bytes('{}\n'.format(len(val)), 'ascii'))
        self.s.sendall(val)

    def close(self):
        self.s.close()


class BaseSocket:
    def __init__(self, host, port, debug=False) -> None:
        # 创建 socket 对象
        print('[Info] server start')
        serversocket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind((host, port))
        serversocket.listen(1)
        self.serversocket = serversocket
        self.queue = Queue()
        self.t = Thread(target=self.run)
        self.t.start()
        self.debug = debug
        self.disconnect = False

    @staticmethod
    def recvLine(sock):
        flag = True
        result = b''
        while not result.endswith(b'\n'):
            res = sock.recv(1)
            if not res:
                flag = False
                break
            result += res
        return flag, result.strip().decode('ascii')

    @staticmethod
    def recvAll(sock, l):
        l = int(l)
        result = b''
        while (len(result) < l):
            t = sock.recv(l - len(result))
            result += t
        return result.decode('ascii')

    def run(self):
        while True:
            clientsocket, addr = self.serversocket.accept()
            print("[Info] Connect: %s" % str(addr))
            self.disconnect = False
            while True:
                flag, l = self.recvLine(clientsocket)
                if not flag:
                    print("[Info] Disonnect: %s" % str(addr))
                    self.disconnect = True
                    break
                data = self.recvAll(clientsocket, l)
                if self.debug:
                    log('[Info] Recv data')
                self.queue.put(data)
            clientsocket.close()

    def update(self):
        time.sleep(1)
        while not self.queue.empty():
            log('update')
            data = self.queue.get()
            self.main(data)

    def main(self, datas):
        print(datas)

    def __del__(self):
        self.serversocket.close()
        self.t.join()


class Timer:
    records = {}
    tmp = None

    @classmethod
    def tic(cls):
        cls.tmp = time.time()

    @classmethod
    def toc(cls):
        res = (time.time() - cls.tmp) * 1000
        cls.tmp = None
        return res

    @classmethod
    def report(cls):
        header = ['', 'Time(ms)']
        contents = []
        for key, val in cls.records.items():
            contents.append(
                ['{:20s}'.format(key), '{:.2f}'.format(sum(val)/len(val))])
        print(tabulate.tabulate(contents, header, tablefmt='fancy_grid'))

    def __init__(self, name, silent=False):
        self.name = name
        self.silent = silent
        if name not in Timer.records.keys():
            Timer.records[name] = []

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        end = time.time()
        Timer.records[self.name].append((end-self.start)*1000)
        if not self.silent:
            t = (end - self.start)*1000
            if t > 1000:
                print('-> [{:20s}]: {:5.1f}s'.format(self.name, t/1000))
            elif t > 1e3*60*60:
                print('-> [{:20s}]: {:5.1f}min'.format(self.name, t/1e3/60))
            else:
                print('-> [{:20s}]: {:5.1f}ms'.format(self.name,
                      (end-self.start)*1000))


'''
Color utils
'''


def generate_colorbar(N=20, cmap='jet'):
    bar = ((np.arange(N)/(N-1))*255).astype(np.uint8).reshape(-1, 1)
    colorbar = cv2.applyColorMap(bar, cv2.COLORMAP_JET).squeeze()
    if False:
        colorbar = np.clip(colorbar + 64, 0, 255)
    import random
    random.seed(666)
    index = [i for i in range(N)]
    random.shuffle(index)
    rgb = colorbar[index, :]
    rgb = rgb.tolist()
    return rgb


colors_bar_rgb = generate_colorbar(cmap='hsv')

colors_table = {
    'b': [0.65098039, 0.74117647, 0.85882353],
    '_pink': [.9, .7, .7],
    '_mint': [166/255.,  229/255.,  204/255.],
    '_mint2': [202/255.,  229/255.,  223/255.],
    '_green': [153/255.,  216/255.,  201/255.],
    '_green2': [171/255.,  221/255.,  164/255.],
    'r': [251/255.,  128/255.,  114/255.],
    '_orange': [253/255.,  174/255.,  97/255.],
    'y': [250/255.,  230/255.,  154/255.],
    '_r': [255/255, 0, 0],
    'g': [0, 255/255, 0],
    '_b': [0, 0, 255/255],
    'k': [0, 0, 0],
    '_y': [255/255, 255/255, 0],
    'purple': [128/255, 0, 128/255],
    'smap_b': [51/255, 153/255, 255/255],
    'smap_r': [255/255, 51/255, 153/255],
    'smap_b': [51/255, 255/255, 153/255],
}


def get_rgb(index):
    if isinstance(index, int):
        if index == -1:
            return (255, 255, 255)
        if index < -1:
            return (0, 0, 0)
        col = colors_bar_rgb[index % len(colors_bar_rgb)]
    else:
        col = colors_table.get(index, (1, 0, 0))
        col = tuple([int(c*255) for c in col[::-1]])
    return col


def get_rgb_01(index):
    col = get_rgb(index)
    return [i*1./255 for i in col[:3]]


class BaseCrit:
    def __init__(self, min_conf, min_joints=3) -> None:
        self.min_conf = min_conf
        self.min_joints = min_joints
        self.name = self.__class__.__name__

    def __call__(self, keypoints3d, **kwargs):
        # keypoints3d: (N, 4)
        conf = keypoints3d[..., -1]
        conf[conf < self.min_conf] = 0
        idx = keypoints3d[..., -1] > self.min_conf
        return len(idx) > self.min_joints


class CritRange(BaseCrit):
    def __init__(self, minr, maxr, rate_inlier, min_conf) -> None:
        super().__init__(min_conf)
        self.min = minr
        self.max = maxr
        self.rate = rate_inlier

    def __call__(self, keypoints3d, **kwargs):
        idx = keypoints3d[..., -1] > self.min_conf
        k3d = keypoints3d[idx, :3]
        crit = (k3d[:, 0] > self.min[0]) & (k3d[:, 0] < self.max[0]) &\
            (k3d[:, 1] > self.min[1]) & (k3d[:, 1] < self.max[1]) &\
            (k3d[:, 2] > self.min[2]) & (k3d[:, 2] < self.max[2])
        self.log = '{}: {}'.format(self.name, k3d)
        return crit.sum()/crit.shape[0] > self.rate
