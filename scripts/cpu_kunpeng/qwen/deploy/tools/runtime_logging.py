"""Bounded byte logs for service processes and the shell driver's console output."""
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
if __package__:
    from .instances import write_json
else:
    from instances import write_json


class RollingLog:
    def __init__(self, path, max_bytes=None, backups=None, env=None):
        env = os.environ if env is None else env
        self.path = Path(path)
        self.limit = int(env.get('LOG_MAX_MIB', '16')) * 1024 * 1024 if max_bytes is None else max_bytes
        self.backups = int(env.get('LOG_BACKUPS', '2')) if backups is None else backups
        if self.limit < 1 or not 0 <= self.backups <= 20:
            raise ValueError('日志大小须为正数，QWEN_LOG_BACKUPS须在0..20之间')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('ab', buffering=0)
        self.size = self.stream.tell()

    def rotate(self):
        self.stream.close()
        if self.backups:
            self.path.with_name(self.path.name + '.' + str(self.backups)).unlink(missing_ok=True)
            for index in range(self.backups - 1, 0, -1):
                source = self.path.with_name(self.path.name + '.' + str(index))
                if source.exists():
                    source.replace(self.path.with_name(self.path.name + '.' + str(index + 1)))
            self.path.replace(self.path.with_name(self.path.name + '.1'))
        else:
            self.path.unlink(missing_ok=True)
        self.stream = self.path.open('wb', buffering=0)
        self.size = 0

    def write(self, data):
        while data:
            if self.size >= self.limit:
                self.rotate()
            count = min(len(data), self.limit - self.size)
            self.stream.write(data[:count])
            self.size += count
            data = data[count:]

    def write_record(self, data):
        if len(data) > self.limit:
            raise ValueError('单条日志记录超过轮转大小')
        if self.size + len(data) > self.limit:
            self.rotate()
        self.write(data)

    def close(self):
        self.stream.close()


def spawn(argv, path, **kwargs):
    """Drain a child's merged output in fixed-size chunks, retaining its real PID."""
    log = RollingLog(path, env=kwargs.get('env'))
    try:
        child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)
    except BaseException:
        log.close()
        raise
    child.log_error = None

    def drain():
        try:
            while data := os.read(child.stdout.fileno(), 65536):
                if child.log_error is None:
                    try:
                        log.write(data)
                    except OSError as error:
                        # Continue draining until the owner can stop the child.
                        child.log_error = str(error)
        except OSError as error:
            child.log_error = str(error)
        finally:
            child.stdout.close()
            log.close()

    child.log_thread = threading.Thread(target=drain, name='log-' + Path(path).name, daemon=True)
    child.log_thread.start()
    return child


def check(child):
    if child.log_error:
        raise RuntimeError('子进程日志写入失败：' + child.log_error)


def finish(child):
    # A descendant may retain the inherited descriptor after its parent exits.
    child.log_thread.join(timeout=5)
    if child.log_thread.is_alive():
        child.log_error = '子进程日志管道仍未关闭，请检查残留进程'
    return child.log_error


def tee(path):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    log = RollingLog(path)
    console = True
    try:
        while data := os.read(sys.stdin.fileno(), 65536):
            log.write(data)
            if console:
                try:
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                except BrokenPipeError:
                    console = False
    finally:
        log.close()
        write_json(str(path) + '.done.json', {'completed': True, 'finished_epoch': time.time(),
                                            'hostname': socket.gethostname(), 'log': str(path)})


if __name__ == '__main__':
    tee(sys.argv[1])
