import json
import socket


class UdpPublisher(object):
    def __init__(self, port):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.port = port

    def send(self, data: dict):
        msg = json.dumps(data)
        ip_port = ('127.0.0.1', self.port)
        self.client.sendto(msg.encode('utf-8'), ip_port)


udp_pub = UdpPublisher(9870)

if __name__ == '__main__':
    import time
    import math
    for i in range(1000):
        udp_pub.send({'data': math.sin(i / 100)})
        time.sleep(0.01)
