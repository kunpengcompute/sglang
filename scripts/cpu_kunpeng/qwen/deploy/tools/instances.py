"""Per-host TP16 ownership and explicit instance selection on shared storage."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import socket
import time
import uuid


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def host_key():
    return re.sub(r'[^A-Za-z0-9_.-]', '_', socket.gethostname())


def current_path(root, port):
    return Path(root) / 'current' / (host_key() + '_' + str(port) + '.json')


def select(root, port, instance_id=None):
    """Never fall back to another node's LAST_RUN or latest directory."""
    root = Path(root).resolve()
    if instance_id is None:
        pointer = current_path(root, port)
        if not pointer.is_file():
            raise RuntimeError('当前节点/端口没有实例记录；可用 --instance 实例ID 查询其他节点的记录')
        instance_id = json.loads(pointer.read_text())['instance_id']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', instance_id):
        raise ValueError('实例ID只能是instances下的直接目录名，不能包含路径')
    run = (root / instance_id).resolve()
    if run.parent != root or not (run / 'STATE.json').is_file():
        raise RuntimeError('找不到实例状态：' + instance_id)
    return run


def require_local_http(state):
    if state['hostname'] != socket.gethostname():
        raise RuntimeError('request/bench请在实例所在节点执行；跨节点可用status/stop --instance，或HTTP客户端访问目标节点地址')


@contextmanager
def own(root):
    """One TP16/full-node resource owner per host in this deployment.

    flock is released on process exit; leave the lock file in place to avoid
    unlink/open races. Independent hosts never lock each other out.
    """
    locks = Path(root) / '.locks'
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / (host_key() + '.lock')).open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('当前节点已有本deployment的TP16管理进程，不能同时申请整节点资源') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def create(root, deploy, port, prefix=''):
    identity = (prefix + time.strftime('%Y%m%d_%H%M%S') + '_' + host_key()
                + '_' + str(port) + '_' + str(os.getpid()) + '_' + uuid.uuid4().hex[:8])
    run = Path(root) / identity
    logs = Path(deploy) / 'logs/instances' / identity
    run.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    return run, logs


def publish(root, port, run):
    write_json(current_path(root, port), {'instance_id': Path(run).name,
               'hostname': socket.gethostname(), 'http_port': int(port)})
