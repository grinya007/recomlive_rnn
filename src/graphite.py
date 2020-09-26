import socket, os, time

class Graphite(object):
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            self = super(Graphite, cls).__new__(cls)
            host = os.getenv('CARBON_HOST', 'carbon')
            port = int(os.getenv('CARBON_PORT', 2003))
            self.addr = (host, port)
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            cls._instance = self
        return cls._instance

    def send(self, metric, value):
        msg = '{} {} {:.0f}'.format(metric, value, time.time())
        return self.sock.sendto(bytes(msg, 'ascii'), self.addr)

