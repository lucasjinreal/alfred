import socket
import numpy as np


class LocalSocketExchanger:
    """
    send and receive data between 2 program
    """

    def __init__(self, ip="127.0.0.1", port=5005, is_server=True) -> None:
        self.MAX_BUFFER_SIZE = 20480

        self.ip = ip
        self.port = port
        self.did_received_data_func = None
        self.is_server = is_server
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._connect()

    def _connect(self):
        if self.is_server:
            self.socket.bind((self.ip, self.port))
            self.socket.listen(3)
            self.conn, addr = self.socket.accept()
            print('Server is ready. ', addr)
        else:
            self.socket.connect((self.ip, self.port))

    def send(self, data):
        """
        send any type of data, it might be List, NDArray, etc
        """
        if not self.is_server:
            self.socket.send(data)
        else:
            print('server can not send data for now.')

    def did_received_data(self, func):
        self.did_received_data_func = func

    def receive(self):
        data = self.conn.recv(self.MAX_BUFFER_SIZE)
        a = np.frombuffer(data, dtype=np.float32)
        if len(a) > 0:
            # print("receive data: ", a, a.shape)
            if self.did_received_data_func:
                self.did_received_data_func(a)

    def listen_and_loop(self):
        while True:
            self.receive()
