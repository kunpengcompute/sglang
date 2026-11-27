"""Configure TP16 PP stages over SSH, then join logical instances through the router."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

from instances import write_json
import runtime_logging

TOOL = Path(__file__).resolve().parents[1]
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
TERMINAL = {'STOPPED', 'FAILED'}


def read(path):
    # Opening revalidates remote renames; stat may retain a negative lookup.
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


def http(url, body=None, timeout=5):
    request = urllib.request.Request(url, None if body is None else json.dumps(body).encode(),
                                    {'Content-Type': 'application/json'})
    with HTTP.open(request, timeout=timeout) as response:
        data = response.read()
    return json.loads(data) if body is not None else data


def base_environment(config):
    # Config is the user's existing trusted shell config, not a new parser/format.
    code = 'import os,json; print(json.dumps(dict(os.environ)))'
    env = {k: v for k, v in os.environ.items() if k not in ('LD_PRELOAD', 'PYTHONPATH', 'PYTHONHOME')}
    out = subprocess.check_output(['bash', '-c', 'set -a; source "$1"; exec "$2" -B -S -c "$3"',
                                   '_', str(config), sys.executable, code], env=env, text=True)
    return json.loads(out.splitlines()[-1])


def role_runtime_overrides(role):
    # A narrow list of typed configuration fields, never an arbitrary env map.
    prefix = 'PD_' + role.upper() + '_'
    fields = {
        'SWAP_KV_IN': 'SGLANG_KUNPENG_SWAP_KV_IN',
        'QWEN_DRAFT_MIRROR_REUSE': 'SGLANG_KUNPENG_QWEN_DRAFT_MIRROR_REUSE',
        'QWEN_DRAFT_RESIDENCY_V2': 'SGLANG_KUNPENG_QWEN_DRAFT_RESIDENCY_V2',
        'PREFILL_WORKSPACE_HBM': 'SGLANG_KUNPENG_QWEN_PREFILL_WORKSPACE_HBM',
        'GQA_DOUBLE_PAIR': 'SGLANG_KUNPENG_QWEN_GQA_DOUBLE_PAIR',
        'IS_PREFILL': 'IS_PREFILL',
    }
    integers = {
        'MAX_RUNNING_REQUESTS': 'MAX_RUNNING_REQUESTS',
        'MAX_TOTAL_TOKENS': 'MAX_TOTAL_TOKENS',
        'MAX_SEQ_NUM': 'SGLANG_KUNPENG_MAX_SEQ_NUM',
        'SHARED_HBW_SHM_MIB': 'SHM_MIB',
        'GRAPH_MAX_REQUESTS': 'SGLANG_KUNPENG_QWEN_DECODE_MAX_REQUESTS',
        'GRAPH_MAX_TOKENS': 'SGLANG_KUNPENG_QWEN_DECODE_MAX_TOKENS',
    }
    fields.update(integers)
    values = {}
    for field, target in fields.items():
        value = os.environ.get(prefix + field, '').strip()
        if not value:
            continue  # Unspecified fields preserve the existing base config.
        if field in integers:
            if not re.fullmatch(r'[0-9]+', value) or int(value) < 1:
                raise ValueError(prefix + field + '必须为正整数')
            value = str(int(value))
        elif value not in ('0', '1'):
            raise ValueError(prefix + field + '必须为0或1；留空继承base')
        if field == 'IS_PREFILL' and value != ('1' if role == 'prefill' else '0'):
            raise ValueError(prefix + field + '与角色不匹配')
        values[target] = value
    if role == 'prefill' and values.get('SGLANG_KUNPENG_SWAP_KV_IN') == '1':
        raise ValueError('此Qwen配方仅在Decode启用GQA换入；Prefill保持SWAP_KV_IN=0')
    if role == 'prefill' and values.get('SGLANG_KUNPENG_QWEN_DRAFT_MIRROR_REUSE') == '1':
        raise ValueError('Qwen draft mirror reuse仅用于Decode；Prefill保持0')
    if role == 'prefill' and values.get('SGLANG_KUNPENG_QWEN_DRAFT_RESIDENCY_V2') == '1':
        raise ValueError('Qwen draft residency v2仅用于Decode；Prefill保持0')
    shm = values.get('SHM_MIB')
    if shm is not None:
        if int(shm) >= 4000:
            raise ValueError(prefix + 'SHARED_HBW_SHM_MIB必须小于4000MiB总预算')
    return values


def configuration():
    p, d = int(os.environ['PD_PREFILL_NODES']), int(os.environ['PD_DECODE_NODES'])
    hosts = os.environ.get('PD_NODE_HOSTS', '').split()
    if not hosts and os.environ.get('PD_NODE_FILE'):
        hosts = [line.split('#', 1)[0].strip() for line in Path(os.environ['PD_NODE_FILE']).read_text().splitlines()]
        hosts = [line for line in hosts if line]
    advertised = os.environ.get('PD_ADVERTISE_HOSTS', '').split() or hosts
    if min(p, d) < 1 or len(hosts) != p + d or len(advertised) != len(hosts):
        raise ValueError('要求至少1P1D，节点列表及通信地址数须等于prefill数量+decode数量')
    router = os.environ.get('PD_ROUTER_NODE') or hosts[0]
    for host in [*hosts, *advertised, router]:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', host) or host in ('0.0.0.0', 'localhost') or host.startswith('127.'):
            raise ValueError('节点/通信地址须为可路由IPv4或主机名：' + host)
    if len(set(hosts)) != len(hosts) or len(set(advertised)) != len(advertised):
        raise ValueError('本版每节点16个本地rank，不允许重复节点或通信地址')
    base = Path(os.environ['PD_BASE_CONFIG']).expanduser().resolve()
    if not base.is_file():
        raise FileNotFoundError(base)
    ports = {name: int(os.environ.get(key, default)) for name, key, default in [
        ('service_port', 'SERVER_PORT', '31086'), ('dist_port', 'DIST_PORT', '29986'),
        ('bootstrap_port', 'DISAGGREGATION_BOOTSTRAP_PORT', '9001'), ('router_port', 'PD_ROUTER_PORT', '30000'),
        ('metrics_port', 'PD_ROUTER_METRICS_PORT', '30010')]}
    if any(not 1 <= value <= 65535 for value in ports.values()) or ports['service_port'] > 55535:
        raise ValueError('端口无效；服务HTTP端口须<=55535（上游还派生HTTP+10000）')
    local_ports = [ports[k] for k in ('service_port', 'dist_port', 'bootstrap_port')]
    local_ports.append(ports['service_port'] + 10000)
    if len(set(local_ports)) != len(local_ports) or ports['router_port'] == ports['metrics_port']:
        raise ValueError('同节点服务/通信/派生端口发生冲突')
    if router in hosts and any(ports[k] in local_ports for k in ('router_port', 'metrics_port')):
        raise ValueError('router与计算服务端口冲突')
    user = os.environ.get('PD_SSH_USER', '')
    if user and not re.fullmatch(r'[A-Za-z0-9_.-]+', user):
        raise ValueError('SSH用户名无效')
    layouts = {}
    for role, count in (('prefill', p), ('decode', d)):
        prefix = 'PD_' + role.upper() + '_'
        tp = int(os.environ.get(prefix + 'TP_SIZE', '16'))
        pp = int(os.environ.get(prefix + 'PP_SIZE', '1'))
        nnodes = int(os.environ.get(prefix + 'NNODES', str(pp)))
        partition = os.environ.get(prefix + 'LAYER_PARTITION', '').strip()
        # Each physical node owns its 16 rank-private NUMA images and one stage.
        if tp != 16 or pp < 1 or nnodes != pp or count % nnodes:
            raise ValueError(prefix + '要求TP16、NNODES=PP_SIZE且物理节点数为NNODES的倍数')
        if partition:
            parts = [int(x) for x in partition.split(',')]
            if len(parts) != pp or min(parts) < 1:
                raise ValueError(prefix + 'LAYER_PARTITION必须为每stage的正整数层数')
        layouts[role] = dict(tp_size=tp, pp_size=pp, nnodes=nnodes, layer_partition=partition,
                             runtime_overrides=role_runtime_overrides(role))
    nodes = []
    for i, (host, addr) in enumerate(zip(hosts, advertised)):
        role, index = ('prefill', i) if i < p else ('decode', i - p)
        layout = layouts[role]
        node_rank = index % layout['nnodes']
        first = (0 if role == 'prefill' else p) + index - node_rank
        runtime_overrides = dict(layout['runtime_overrides'])
        nodes.append({'id': role + str(index), 'role': role, 'host': host, 'address': addr,
                      'instance_id': role + str(index // layout['nnodes']),
                      'node_rank': node_rank, 'pp_rank': node_rank,
                      'http_endpoint': node_rank == 0,
                      'dist_init_addr': advertised[first] + ':' + str(ports['dist_port']),
                      'global_rank_range': [node_rank * 16, (node_rank + 1) * 16 - 1],
                      'url': 'http://' + addr + ':' + str(ports['service_port']), **layout,
                      'runtime_overrides': runtime_overrides})
    router_address = advertised[hosts.index(router)] if router in hosts else router
    if min(int(os.environ.get('PD_START_TIMEOUT', '1800')), int(os.environ.get('PD_STOP_TIMEOUT', '600'))) < 1:
        raise ValueError('启动和停止超时必须为正整数秒')
    return dict(nodes=nodes, base_config=str(base), tool=str(TOOL), ssh_user=user,
                ssh_options=shlex.split(os.environ.get('PD_SSH_OPTIONS', '-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10')),
                control_python=os.environ.get('PD_CONTROL_PYTHON', '/usr/bin/python3'),
                cluster_dir=str(Path(os.environ['PD_CLUSTER_DIR']).expanduser().resolve()),
                router_host=router, router_url='http://' + router_address + ':' + str(ports['router_port']),
                router_cpus=os.environ.get('PD_ROUTER_CPUS', '586,607'),
                ib_device=os.environ.get('PD_IB_DEVICE', ''), interface=os.environ.get('PD_NET_INTERFACE', ''),
                start_timeout=int(os.environ.get('PD_START_TIMEOUT', '1800')),
                stop_timeout=int(os.environ.get('PD_STOP_TIMEOUT', '600')), **ports)


def render(plan):
    root = Path(plan['cluster_dir'])
    run = root / 'runs' / (time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:10])
    run.mkdir(parents=True)
    plan = dict(plan, run=str(run), token=uuid.uuid4().hex)
    for node in plan['nodes']:
        folder = run / node['id']
        folder.mkdir()
        node.update(config=str(folder / 'config.sh'), state_file=str(folder / 'STATE.json'))
        # Re-enter the same run.sh through the local node lifecycle. DISAGGREGATION_MODE
        # still selects the real prefill/decode server protocol in start.py.
        overrides = dict(DEPLOY_MODE='single',
            SERVER_HOST=node['address'], SERVER_PORT=plan['service_port'],
            DIST_PORT=plan['dist_port'], DISAGGREGATION_MODE=node['role'],
            TP_SIZE=node['tp_size'], PP_SIZE=node['pp_size'],
            NNODES=node['nnodes'], NODE_RANK=node['node_rank'],
            DIST_INIT_ADDR=node['dist_init_addr'],
            SGLANG_KUNPENG_PP_LAYOUT='node_block',
            DISAGGREGATION_BOOTSTRAP_PORT=plan['bootstrap_port'], DISAGGREGATION_IB_DEVICE=plan['ib_device'],
            CLUSTER_STATE_FILE=node['state_file'], CLUSTER_TOKEN=plan['token'],
            PD_START_TIMEOUT=plan['start_timeout'],
            CPU_REMAP_SAME_NUMA=0, SGLANG_ENABLE_TOKENIZER_SEPERATE=0, SGLANG_DISAGG_STAGING_BUFFER=0)
        overrides.update(node.get('runtime_overrides', {}))
        if node['pp_size'] > 1:
            overrides['WORLD_SIZE'] = node['tp_size'] * node['pp_size']
            if node['role'] == 'prefill':
                overrides['SGLANG_KUNPENG_RDMA_PP_COMM'] = 1
        if node['layer_partition']:
            overrides['SGLANG_PP_LAYER_PARTITION'] = node['layer_partition']
        if plan['interface']:
            overrides['GLOO_SOCKET_IFNAME'] = plan['interface']
        config = 'source ' + shlex.quote(plan['base_config']) + '\n'
        config += 'unset SGLANG_PP_LAYER_PARTITION\n'
        config += ''.join('export ' + key + '=' + shlex.quote(str(value)) + '\n' for key, value in overrides.items())
        Path(node['config']).write_text(config)
        Path(node['config']).chmod(0o600)
    write_json(run / 'PLAN.json', plan)
    return plan


def ssh(plan, host, command):
    target = (plan['ssh_user'] + '@' if plan['ssh_user'] else '') + host
    # SSH sessions may inherit only the 16 spare CPUs. The user must already
    # own the full node; restore the TP16 manager's 608-CPU range before probing.
    command = ['/usr/bin/taskset', '-c', '0-607', *command]
    return ['ssh', '-T', *plan['ssh_options'], target, shlex.join(list(map(str, command)))]


def node_command(plan, host, action, node_id=None):
    command = [plan['control_python'], '-u', '-B', '-S', str(TOOL / 'tools/pd_cluster.py'),
               'node-' + action, '--plan', str(Path(plan['run']) / 'PLAN.json')]
    if node_id:
        command += ['--node', node_id]
    return ssh(plan, host, command)


def probe(plan, node):
    log = Path(plan['run']) / node['id'] / 'probe.log'
    result = subprocess.run(node_command(plan, node['host'], 'probe', node['id']),
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=plan['start_timeout'])
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(node['id'] + '环境检查失败：' + str(log))
    return json.loads(result.stdout.splitlines()[-1])


def node_probe(plan, node):
    config = Path(node['config'])
    env = base_environment(config)
    deploy = Path(env['DEPLOY_DIR'])
    for path in (deploy / 'release/RUNTIME_LOCK.json', deploy / 'PACKAGE.json', deploy / 'numa_images/images.json'):
        if not path.is_file():
            raise FileNotFoundError('先完成build/pack，所有节点须可见：' + str(path))
    if len(os.sched_getaffinity(0)) < 608:
        raise RuntimeError('本版要求已分配完整608 CPU的920F节点')
    if plan['ib_device'] and not (Path('/sys/class/infiniband') / plan['ib_device']).exists():
        raise RuntimeError('本节点没有RDMA设备：' + plan['ib_device'])
    if plan['interface'] and not (Path('/sys/class/net') / plan['interface']).exists():
        raise RuntimeError('本节点没有配置的网卡：' + plan['interface'])
    with (Path(node['state_file']).parent / 'verify.log').open('w') as stream:
        result = subprocess.run(['bash', str(TOOL / 'run.sh'), 'verify', str(config)],
                                stdout=stream, stderr=subprocess.STDOUT, timeout=plan['start_timeout'])
    if result.returncode:
        raise RuntimeError('run.sh verify失败，见节点verify.log')
    return dict(hostname=socket.gethostname(), deploy=str(deploy), role=node['role'])


def router_command(plan, env):
    command = [env['PYTHON'], '-m', 'sglang_router.launch_router', '--pd-disaggregation']
    for node in plan['nodes']:
        if not node['http_endpoint']:
            continue
        command += ['--' + node['role'], node['url']]
        if node['role'] == 'prefill':
            command += [str(plan['bootstrap_port'])]
    command += ['--policy', 'round_robin', '--decode-policy', 'round_robin', '--host', '0.0.0.0',
                '--port', str(plan['router_port']), '--prometheus-port', str(plan['metrics_port']),
                '--prometheus-host', '127.0.0.1', '--worker-startup-timeout-secs', str(plan['start_timeout']),
                '--request-timeout-secs', '600', '--disable-retries']
    command += ['--log-level', env.get('LOG_LEVEL', 'warning').replace('warning', 'warn')]
    if plan['router_cpus']:
        command = ['taskset', '-c', plan['router_cpus'], *command]
    return command


def router_environment(plan):
    env = base_environment(plan['base_config'])
    prefix = Path(env['PYTHON']).parent.parent
    clean = {k: v for k, v in env.items() if k not in ('LD_PRELOAD', 'PYTHONPATH', 'PYTHONHOME')}
    clean.update(LD_LIBRARY_PATH=':'.join(p for p in [env.get('COMPILER_RUNTIME_LIB', ''),
                 str(prefix / 'lib'), env.get('EXTRA_LIBRARY_PATH', '')] if p),
                 NO_PROXY='*', no_proxy='*', RAYON_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    return env, clean


def node_router_check(plan):
    env, clean = router_environment(plan)
    result = subprocess.run([env['PYTHON'], '-m', 'sglang_router.launch_router', '--help'],
                            env=clean, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    if result.returncode or not all(flag in result.stdout for flag in ('--pd-disaggregation', '--prefill', '--decode')):
        raise RuntimeError('目标环境的sglang_router不支持本版PD入口：\n' + result.stdout[-3000:])
    if plan['router_cpus']:
        subprocess.run(['taskset', '-c', plan['router_cpus'], '/usr/bin/true'], check=True)
    return {'router_python': env['PYTHON'], 'hostname': socket.gethostname(), 'pd_flags_present': True}


def node_router(plan):
    run = Path(plan['run'])
    env, clean = router_environment(plan)
    command = router_command(plan, env)
    child = None
    try:
        # Router binds its configured addresses and reports the actual error.
        write_json(run / 'ROUTER_COMMAND.json', {'argv': command, 'host': socket.gethostname()})
        child = runtime_logging.spawn(command, run / 'router.log', env=clean, start_new_session=True)
        write_json(run / 'ROUTER.json', {'phase': 'STARTING', 'token': plan['token'], 'pid': child.pid})
        while not (run / 'STOP').exists():
            runtime_logging.check(child)
            if child.poll() is not None:
                raise RuntimeError('router退出，见router.log；exit=' + str(child.returncode))
            time.sleep(1)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        log_error = runtime_logging.finish(child) if child is not None else None
        write_json(run / 'ROUTER.json', {'phase': 'STOPPED', 'token': plan['token'],
                                      'cleanup': 'confirmed', 'child_started': child is not None,
                                      'log_error': log_error})


def owned_nodes(plan):
    rows = []
    for node in plan['nodes']:
        state = read(node['state_file'])
        if state and state.get('cluster_token') == plan['token']:
            rows.append((node, state))
    return rows


def node_status(node):
    state = read(node['state_file'])
    if state is None:
        return None
    result = {key: state.get(key) for key in
              ('phase', 'hostname', 'pd_mode', 'mtp', 'run', 'logs', 'url', 'topology', 'cleanup', 'error')}
    result['frequency_ok'] = state.get('frequency', {}).get('performance_condition_met')
    result['resources_status'] = state.get('resources', {}).get('status')
    return result


def stop_owned(plan):
    run = Path(plan['run'])
    (run / 'STOP').touch()
    deadline = time.monotonic() + plan['stop_timeout']
    while True:
        rows = owned_nodes(plan)
        # A startup may publish its state after STOP was requested.
        for _, state in rows:
            if state['phase'] not in TERMINAL:
                (Path(state['run']) / 'STOP').touch()
        missing = set(plan.get('launched', [])) - {n['id'] for n, _ in rows} - {'router'}
        pending = sorted(missing) + [node['id'] for node, state in rows
                   if state['phase'] not in TERMINAL or
                   (state.get('resources_acquired') is not False and not state.get('cleanup'))]
        router = read(run / 'ROUTER.json')
        if ('router' in plan.get('launched', []) and not router) or (router and router.get('phase') != 'STOPPED'):
            pending.append('router')
        if not pending:
            break
        if time.monotonic() > deadline:
            break
        time.sleep(1)
    unconfirmed = [node['id'] for node, state in rows
                   if state.get('resources_acquired') is not False and
                   (state.get('cleanup') != 'confirmed' or state.get('resources', {}).get('status') != 'confirmed')]
    unconfirmed += sorted(missing)
    return {'pending': pending, 'unconfirmed': unconfirmed, 'owned_instances': len(rows),
            'confirmed': not pending and not unconfirmed}


def wait_for(plan, predicate, children):
    deadline = time.monotonic() + plan['start_timeout']
    while True:
        if (Path(plan['run']) / 'STOP').exists():
            raise KeyboardInterrupt()
        for name, child in children.items():
            runtime_logging.check(child)
            if child.poll() is not None:
                raise RuntimeError(name + '启动进程退出；见对应ssh日志')
        states = {node['id']: read(node['state_file']) for node in plan['nodes']}
        if any(value and value.get('cluster_token') != plan['token'] for value in states.values()):
            raise RuntimeError('节点状态不属于本次集群')
        failed = [name for name, value in states.items() if value and value['phase'] in TERMINAL]
        if failed:
            raise RuntimeError('节点服务提前结束：' + ','.join(failed))
        if predicate(states):
            return states
        if time.monotonic() > deadline:
            raise TimeoutError('集群启动超时，见各节点STATE.json及ssh日志')
        time.sleep(1)


def preflight_cluster(plan):
    run = Path(plan['run'])
    with ThreadPoolExecutor(max_workers=len(plan['nodes'])) as pool:
        facts = list(pool.map(lambda node: probe(plan, node), plan['nodes']))
    if len({row['hostname'] for row in facts}) != len(facts):
        raise RuntimeError('不同SSH名称实际指向同一节点')
    write_json(run / 'PREFLIGHT.json', facts)
    checked = subprocess.run(node_command(plan, plan['router_host'], 'router-check'),
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, timeout=120)
    (run / 'router_check.log').write_text(checked.stdout)
    if checked.returncode:
        raise RuntimeError('router环境检查失败：见router_check.log')
    return facts


def start(plan):
    run = Path(plan['run'])
    children = {}
    state = {'phase': 'PREFLIGHT', 'run': str(run), 'token': plan['token'], 'controller_pid': os.getpid()}
    write_json(Path(plan['cluster_dir']) / 'CURRENT.json', {'run': str(run)})
    def save():
        write_json(run / 'STATE.json', state)
    save()
    def spawn(name, argv):
        children[name] = runtime_logging.spawn(argv, run / (name + '_ssh.log'), stdin=subprocess.DEVNULL)
        plan.setdefault('launched', []).append(name)
        write_json(run / 'PLAN.json', plan)
    try:
        preflight_cluster(plan)
        state['phase'] = 'STARTING_SERVICES'; save()
        for node in plan['nodes']:
            spawn(node['id'], ssh(plan, node['host'], ['bash', str(TOOL / 'run.sh'), 'start', node['config']]))
        wait_for(plan, lambda rows: all(rows[n['id']] and rows[n['id']]['phase'] ==
                 ('HTTP_READY' if n['http_endpoint'] else 'STAGE_READY') for n in plan['nodes']), children)
        state['phase'] = 'STARTING_ROUTER'; save()
        spawn('router', node_command(plan, plan['router_host'], 'router'))
        def router_ready(_):
            try:
                http(plan['router_url'] + '/health')
                return True
            except OSError:
                return False
        wait_for(plan, router_ready, children)
        state['phase'] = 'PD_WARMUP'; save()
        warmups = []
        count = max(sum(n['role'] == role and n['http_endpoint'] for n in plan['nodes'])
                    for role in ('prefill', 'decode'))
        for _ in range(count):
            response = http(plan['router_url'] + '/generate', {'text': '请用一句话介绍你自己。',
                'sampling_params': {'temperature': 0, 'max_new_tokens': 16, 'ignore_eos': True}, 'stream': False}, timeout=600)
            if response.get('error') or response.get('meta_info', {}).get('completion_tokens', 0) <= 0:
                raise RuntimeError('真实PD首请求失败：' + str(response))
            warmups.append(response)
        write_json(run / 'WARMUP.json', warmups)
        # Each physical node samples local frequencies under real router work.
        # Admit serially so these internal warmups preserve client concurrency 1.
        for node in plan['nodes']:
            value = read(node['state_file'])
            write_json(Path(value['run']) / 'PD_ADMIT.json', {'cluster_token': plan['token'], 'router_url': plan['router_url']})
            wait_for(plan, lambda rows: rows[node['id']] and rows[node['id']]['phase'] == 'READY', children)
        wait_for(plan, lambda rows: all(v and v['phase'] == 'READY' for v in rows.values()), children)
        state.update(phase='READY', url=plan['router_url']); save()
        print('READY：' + plan['router_url'] + '；状态目录：' + str(run), flush=True)
        while not (run / 'STOP').exists():
            for child in children.values():
                runtime_logging.check(child)
            if any(child.poll() is not None for child in children.values()):
                raise RuntimeError('运行中的节点或router退出')
            failed = [n['id'] for n, value in owned_nodes(plan) if value['phase'] in TERMINAL]
            if failed:
                raise RuntimeError('运行中的stage失败或停止：' + ','.join(failed))
            time.sleep(1)
        state['phase'] = 'STOPPED'
    except KeyboardInterrupt:
        state['phase'] = 'STOPPED'
    except BaseException as error:
        state.update(phase='FAILED', error=repr(error))
        raise
    finally:
        state['cleanup'] = stop_owned(plan)
        for child in children.values():
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                state['cleanup']['confirmed'] = False
        if not state['cleanup']['confirmed']:
            state['phase'] = 'NEEDS_ATTENTION'
        save()
        errors = [error for child in children.values() if (error := runtime_logging.finish(child))]
        if errors:
            state['log_errors'] = errors
            save()


@contextmanager
def control_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.controller.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('当前集群已有控制进程；请先status/stop') from None
        yield


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['plan', 'check', 'start', 'status', 'stop', 'node-probe', 'node-router', 'node-router-check'])
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--node')
    args = parser.parse_args()
    def interrupted(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if args.action.startswith('node-'):
        plan = read(args.plan)
        if args.action == 'node-probe':
            node = next(n for n in plan['nodes'] if n['id'] == args.node)
            print(json.dumps(node_probe(plan, node)))
        elif args.action == 'node-router-check':
            print(json.dumps(node_router_check(plan)))
        else:
            node_router(plan)
        return
    root = Path(os.environ['PD_CLUSTER_DIR']).expanduser().resolve()
    if args.action in ('status', 'stop'):
        current = read(root / 'CURRENT.json')
        if not current:
            raise RuntimeError('没有本集群的运行记录')
        run = Path(current['run']); plan = read(run / 'PLAN.json')
        if args.action == 'stop':
            result = stop_owned(plan)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if not result['confirmed']:
                raise SystemExit(1)
        else:
            print(json.dumps({'cluster': read(run / 'STATE.json'),
                              'nodes': {n['id']: node_status(n) for n in plan['nodes']},
                              'router': read(run / 'ROUTER.json')}, ensure_ascii=False, indent=2))
        return
    plan = configuration()
    if args.action == 'plan':
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return
    with control_lock(root):
        current = read(root / 'CURRENT.json')
        prior = read(Path(current['run']) / 'STATE.json') if current else None
        if prior and (prior['phase'] not in TERMINAL or not prior.get('cleanup', {}).get('confirmed')):
            raise RuntimeError('上次集群尚未确认收尾，请先stop并检查状态')
        plan = render(plan)
        if args.action == 'check':
            facts = preflight_cluster(plan)
            print(json.dumps(facts, ensure_ascii=False, indent=2))
        else:
            start(plan)


if __name__ == '__main__':
    main()
