from threading import Thread
from queue import Queue

import os, time, sys, logging, socket, signal

def lazyprop(fn):
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazyprop

class Server(object):
    def __init__(
            self,
            dispatcher,
            periodic    = None,
            period      = None,
            host        = '0.0.0.0',
            port        = 25000,
            logfile     = 'var/log/{}.log'.format(os.path.basename(sys.argv[0])),
            pidfile     = 'var/run/{}.pid'.format(os.path.basename(sys.argv[0])),
            queue_limit = 10000
        ):
        self.dispatcher         = dispatcher
        self.periodic           = periodic
        self.period             = period
        self.host               = host
        self.port               = port
        self.logfile            = logfile
        self.pidfile            = pidfile
        self.queue_limit        = queue_limit
        self.running            = False
        self.chkdir(logfile)
        self.chkdir(pidfile)

    @lazyprop
    def daemon(self):
        return Daemon(self.pidfile, self.onstart, self.onstop)

    @lazyprop
    def socket(self):
        return Socket(self.host, self.port, self.dispatch)

    @lazyprop
    def queue(self):
        return Queue()

    @lazyprop
    def dispatcher_thread(self):
        return Thread(target = self._dispatcher)

    @lazyprop
    def periodic_thread(self):
        return Thread(target = self._periodic)


    def chkdir(self, sfile):
        sdir = os.path.dirname(sfile)
        if not os.path.isdir(sdir):
            os.makedirs(sdir)

    def command(self, cmd):
        return self.daemon.command(cmd)

    def onstart(self):
        self._init_logger()
        self.running = True
        print('Started!!1')

        self.dispatcher_thread.start()
        if self.periodic:
            self.periodic_thread.start()

        self.socket.listen()

    def onstop(self):
        self.running = False
        self.queue.put(('__stop__', None))
        self.queue.join()
        self.dispatcher_thread.join()

        if self.periodic:
            self.periodic_thread.join()

        print('Stopped =(')

    def dispatch(self, data, response):
        if self.queue.qsize() >= self.queue_limit:
            print('The queue is full')
        else:
            self.queue.put((data, response))

    def _dispatcher(self):
        while True:
            data, response = self.queue.get()
            
            if data == '__stop__':
                self.queue.task_done()
                break

            try:
                self.dispatcher(self, data, response)
            except Exception as e:
                print('Error in dispatcher:', e)

            self.queue.task_done()

    def _periodic(self):
        while self.running:
            time.sleep(self.period)
            self.periodic(self)

    def _init_logger(self):
        logging.basicConfig(
            level = logging.DEBUG,
            format = '%(asctime)s %(name)s %(message)s',
            filename = self.logfile,
            filemode = 'a'
        )
        sys.stdout = StreamToLogger(logging.getLogger('STDOUT'))
        sys.stderr = StreamToLogger(logging.getLogger('STDERR'))


class Daemon(object):
    def __init__(self, pidfile, onstart, onstop):
        self.pidfile = pidfile
        self.onstart = onstart
        self.onstop  = onstop

    def command(self, cmd):
        method_name = '_' + str(cmd)
        method = getattr(self, method_name, self._unknown_command)
        return method()

    def _start(self):
        if self._is_running():
            raise Exception('Already running')
        pid = os.fork()
        if pid:
            os.waitpid(pid, 0)
            os._exit(0)
        os.umask(0)
        os.setsid()
        pid = os.fork()
        if pid:
            os._exit(0)
        self._write_pidfile()
        signal.signal(signal.SIGTERM, self._onterm)
        self.onstart()

    def _startindocker(self):
        signal.signal(signal.SIGTERM, self._onterm)
        self.onstart()

    def _onterm(self, signum, frame):
        self.onstop()
        sys.exit(0)

    def _stop(self):
        pid = self._read_pidfile()
        if pid != None and self._is_running(pid):
            os.kill(pid, signal.SIGTERM)
            while self._is_running(pid):
                time.sleep(0.1)
        else:
            raise Exception('Not running')

    def _restart(self):
        try:
            self._stop()
        except Exception:
            pass
        finally:
            self._start()

    def _is_running(self, pid = None):
        running = False
        if pid == None:
            pid = self._read_pidfile()
        if pid != None and os.path.isdir('/proc/{}'.format(pid)):
            running = True
        return running

    def _unknown_command(self):
        raise ValueError('Unknown command called')

    def _read_pidfile(self):
        pid = None
        if os.path.isfile(self.pidfile):
            fh = open(self.pidfile)
            spid = fh.read().strip()
            if spid.isdigit():
                pid = int(spid)
        return pid

    def _write_pidfile(self):
        open(self.pidfile, 'w').write(str(os.getpid()))


class Socket(object):
    def __init__(self, host, port, dispatch, max_size = 65535):
        self.host = host
        self.port = port
        self.max_size = max_size
        self.dispatch = dispatch

    def listen(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        while True:
            data, address = self.sock.recvfrom(self.max_size)
            response = lambda rdata: self.sock.sendto(rdata, address)
            try:
                self.dispatch(data, response)
            except Exception as e:
                print('Error in dispatch:', e)


class StreamToLogger(object):
   def __init__(self, logger, log_level=logging.INFO):
      self.logger = logger
      self.log_level = log_level

   def flush(self):
       pass

   def write(self, buf):
      for line in buf.rstrip().splitlines():
         self.logger.log(self.log_level, line.rstrip())

