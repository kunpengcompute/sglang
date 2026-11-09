"""Standalone TP16 lifecycle; deployment resources are managed outside SGLang."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from tools.check_dependencies import check as check_dependencies
from tools.runtime_contract import environment, snapshot, verify, verify_payload, sha256
from tools.resources import Resources, budget, prepare_compute, save as save_json, warning
from tools import runtime_frequency, instances

kit = Path(os.environ['QWEN_KIT'])
payload = Path(os.environ.get('QWEN_PAYLOAD_ROOT', str(kit)))
root = Path(os.environ['WORK_ROOT']).expanduser()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http(url, body=None):
    request = urllib.request.Request(url, None if body is None else json.dumps(body).encode(),
                                    {'Content-Type': 'application/json'})
    with opener.open(request, timeout=180 if body is not None else 2) as response:
        return response.read()


def request(url, tokens=32):
    return json.loads(http(url+'/generate', {'text': '请用一句话介绍你自己。',
        'sampling_params': {'temperature': 0, 'max_new_tokens': tokens,
                            'ignore_eos': True}, 'stream': False}))


def checked_warmup(url):
    response = request(url, tokens=64)
    if 'error' in response or response.get('meta_info', {}).get('completion_tokens', 0) <= 0:
        raise ValueError('换核后的预热请求失败：' + str(response))
    return response


def mtp_settings():
    enabled = os.environ['SGLANG_ENABLE_MTP']
    if enabled not in ('0', '1'):
        raise ValueError('SGLANG_ENABLE_MTP只能为0或1')
    steps = int(os.environ.get('SGLANG_SPECULATIVE_NUM_STEPS', '3')) if enabled == '1' else 0
    if enabled == '1' and steps < 1:
        raise ValueError('开启MTP时SGLANG_SPECULATIVE_NUM_STEPS必须为正整数')
    return {'mtp': int(enabled), 'mtp_steps': steps,
            'speculative_algorithm': 'EAGLE' if steps else None,
            'speculative_eagle_topk': 1 if steps else None,
            'speculative_num_draft_tokens': steps + 1 if steps else None}


def check(env, current, capacity=True):
    assert os.uname().machine == 'aarch64', '此配方用于 920F aarch64'
    mtp_settings()
    for n in range(16):
        p = Path(f'/sys/devices/system/node/node{n}/cpulist')
        assert p.read_text().strip() == f'{n*38}-{n*38+37}', 'NUMA/CPU编号与此配方不同，需要适配核表'
    if os.environ['SGLANG_ENABLE_HBW_POOL'] == '1':
        plan = budget()
        if capacity:
            for n in range(16, 32):
                node = Path(f'/sys/devices/system/node/node{n}')
                free = int((node/'hugepages/hugepages-2048kB/free_hugepages').read_text())
                fields = [line.split() for line in (node/'meminfo').read_text().splitlines()]
                kib = next(int(x[-2]) for x in fields if 'MemFree:' in x)
                assert (free >= plan['required_pages'] and kib >= plan['shm_mib']*1024
                        and kib+free*2048 >= plan['total_mib']*1024), f'HBM node{n}大页或空闲容量不足'
        print('每rank HBM预算：' + json.dumps(plan, ensure_ascii=False), flush=True)
    return dict(current['imports'], common_sha256=current['common_sha256'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['deps', 'verify', 'check', 'start', 'status', 'stop', 'request', 'bench'])
    parser.add_argument('--instance', help='instances下的完整实例目录名；省略则定位当前节点和配置端口')
    args = parser.parse_args()
    action = args.action
    if action in ('status', 'stop', 'request', 'bench'):
        r = instances.select(root, os.environ['QWEN_HTTP_PORT'], args.instance)
        state = json.loads((r/'STATE.json').read_text())
        if action == 'stop':
            (r/'STOP').touch()
            print('已请求停止本次服务；用 status 查看退出与清理结果。')
        elif action == 'status':
            print(json.dumps(state, ensure_ascii=False, indent=2))
        elif action == 'bench':
            instances.require_local_http(state)
            assert state['phase'] == 'READY', '先启动服务并等待 READY'
            subprocess.run([os.environ['CLIENT_PYTHON'], '-u', '-B', '-S',
                            kit/'tools/benchmark.py', '--run', r,
                            '--input-tokens', os.environ['QWEN_TEST_INPUT_TOKENS'],
                            '--output-tokens', os.environ['QWEN_TEST_OUTPUT_TOKENS'],
                            '--concurrency', os.environ['QWEN_TEST_CONCURRENCY'],
                            '--warmup-tokens', os.environ['QWEN_TEST_WARMUP_TOKENS']], check=True)
        else:
            instances.require_local_http(state)
            assert state['phase'] == 'READY' , '服务尚未 READY；先查看 status 和实例logs目录'
            print(json.dumps(request(state['url']), ensure_ascii=False, indent=2))
        return
    if args.instance:
        parser.error('--instance仅用于status/stop/request/bench')
    if action == 'start':
        # Acquire ownership before checking mutable resources or writing pointers.
        with instances.own(root):
            r, logs = instances.create(root, os.environ['DEPLOY_DIR'], os.environ['QWEN_HTTP_PORT'])
            state = {'phase': 'PREFLIGHT', 'run': str(r), 'logs': str(logs),
                     'hostname': socket.gethostname(), 'http_port': int(os.environ['QWEN_HTTP_PORT']),
                     'owner_pid': os.getpid(), **mtp_settings()}
            instances.write_json(r/'STATE.json', state)
            instances.publish(root, os.environ['QWEN_HTTP_PORT'], r)
            try:
                start(r, logs)
            except BaseException as error:
                state = json.loads((r/'STATE.json').read_text())
                if state['phase'] == 'PREFLIGHT':
                    state.update(phase='FAILED', error=repr(error), resources_acquired=False)
                    instances.write_json(r/'STATE.json', state)
                raise
    else:
        r, _ = instances.create(root/'checks', os.environ['DEPLOY_DIR'], os.environ['QWEN_HTTP_PORT'], prefix=action+'_')
        preflight(action, r)


def preflight(action, r):
    env = environment(dependencies_only=action in ('deps', 'verify'))
    dependencies = check_dependencies(env, r/'DEPENDENCIES.json')
    if action == 'deps':
        return env, dependencies, None, None
    if not (payload/'RUNTIME_LOCK.json').is_file():
        raise RuntimeError('先执行pack，生成DEPLOY_DIR/release和numa_images')
    verify_payload(payload)
    current = snapshot(env, dependencies)
    verify(payload, current, r/'RUNTIME_CHECK.json')
    if action == 'verify':
        print('部署文件及目标机运行库匹配；未启动模型、未申请HBM。')
        return env, dependencies, current, None
    probe = check(env, current, capacity=action == 'check')
    if action == 'check':
        print(json.dumps(probe, ensure_ascii=False, indent=2))
    return env, dependencies, current, probe


def validate_images(images):
    package = json.loads((Path(os.environ['DEPLOY_DIR'])/'PACKAGE.json').read_text())
    if package['phase'] != 'COMPLETE':
        raise RuntimeError('pack未完成，不启动未准备完整的副本')
    info = json.loads((images/'images.json').read_text())
    # hostname documents where files were prepared; it is not software identity.
    if info['runtime_lock_sha256'] != sha256(payload/'RUNTIME_LOCK.json'):
        raise RuntimeError('NUMA副本与发布版本不同，请使用匹配版本')
    if Path(info['python_prefix']).resolve() != Path(os.environ['PYTHON']).resolve().parent.parent:
        raise RuntimeError('NUMA副本引用的Python环境路径已变化，需重新准备副本')
    expected = {'ranks': 16, 'cpus_per_ddr_numa': 38,
                'ddr_nodes': list(range(16)), 'hbm_nodes': list(range(16, 32))}
    if info.get('topology', expected) != expected:
        raise RuntimeError('NUMA副本拓扑配方不匹配')
    if len(info['images']) != 16 or any(Path(name).resolve() != (images/f'rank{i}').resolve()
                                        for i, name in enumerate(info['images'])):
        raise RuntimeError('NUMA副本目录已移动或不完整，不能引用其他部署的绝对路径')


def start(r, logs):
    env, dependencies, current, probe = preflight('start', r)
    for port in (int(os.environ['QWEN_HTTP_PORT']), int(os.environ['QWEN_DIST_PORT'])):
        with socket.socket() as probe_port:
            probe_port.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe_port.bind(('0.0.0.0', port))
    images = Path(os.environ['DEPLOY_DIR'])/'numa_images'
    validate_images(images)
    os.sched_setaffinity(0, range(608))
    assert set(range(608)) <= os.sched_getaffinity(0), '当前进程不能使用所需608个CPU'
    (r/'config.sh').write_bytes(Path(os.environ['QWEN_CONFIG_PATH']).read_bytes())
    (r/'IMPORT_PATHS.json').write_text(json.dumps(probe, indent=2)+'\n')
    profile = {k: v for k, v in os.environ.items() if k.startswith(('SGLANG_', 'KUTACC_', 'KUPL_'))
               or k in ('EP_SIZE','PP_SIZE','WORLD_SIZE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE',
                        'KMP_AFFINITY','MV2_COMM_WORLD_LOCAL_SIZE','OMP_NUM_THREADS','OMP_PROC_BIND',
                        'PYTHONNOUSERSITE','TORCH_COMPILE_DISABLE','TORCH_USE_KUPL',
                        'SPECULATIVE_DRAFT_MODEL_PATH')}
    (r/'environment.json').write_text(json.dumps(profile, indent=2)+'\n')
    state = {'phase': 'RESOURCES', 'run': str(r), 'http_port': int(os.environ['QWEN_HTTP_PORT']),
             'tools_path': str(kit), 'tools_revision': json.loads((kit/'VERSION.json').read_text())['tool_revision'],
             'payload_root': str(payload), **mtp_settings(), 'hostname': socket.gethostname(), 'owner_pid': os.getpid(), 'logs': str(logs), 'url': 'http://127.0.0.1:'+os.environ['QWEN_HTTP_PORT']}
    service = None
    sampler = None
    resources = Resources(r, logs)

    def save():
        instances.write_json(r/'STATE.json', state)

    def interrupted(*_):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save()
    print('运行目录：'+str(r), flush=True)
    clean = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_LIBRARY_PATH')}
    try:
        resources.prepare()
        check(env, current)
        if (r/'STOP').exists():
            raise KeyboardInterrupt()
        copied = [images/f'rank{i}/packages/sgl_kernel'/Path(probe['common_ops']).name for i in range(16)]
        assert len({p.stat().st_ino for p in copied}) == 16
        # Do not read all private DSOs in this unbound control process: on a
        # deferred-copy filesystem that can become their first page-cache touch.
        for rank, native in enumerate(copied):
            receipt = json.loads((images/f'rank{rank}/common_ops_receipt.json').read_text())
            assert receipt['rank'] == rank and receipt['inode'] == native.stat().st_ino
            assert receipt['sha256'] == probe['common_sha256']
            assert receipt['copy_hash_cpus'] == [rank * 38]
        argv = [os.environ['CLIENT_PYTHON'], '-u', kit/'tools/launch.py',
            '--source-root', os.environ['SGLANG_SOURCE'], '--model', os.environ['MODEL'],
            '--images', images, '--run', r/'service', '--log-dir', logs, '--environment', r/'environment.json',
            '--tiling', os.environ['TILING_FILE'], '--chat-template', os.environ['CHAT_TEMPLATE']]
        for option, key in [('host','QWEN_HOST'),('port','QWEN_HTTP_PORT'),('dist-port','QWEN_DIST_PORT'),
            ('context-length','QWEN_CONTEXT_LENGTH'),('max-total-tokens','QWEN_MAX_TOTAL_TOKENS'),
            ('max-running-requests','QWEN_MAX_RUNNING_REQUESTS'),('random-seed','QWEN_SEED')]:
            argv += ['--'+option, os.environ[key]]
        with (logs/'launcher.log').open('w') as log:
            service = subprocess.Popen(list(map(str, argv)), env=clean, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        state['phase'] = 'STARTING'
        save()
        deadline = time.monotonic()+1800
        while True:
            if (r/'STOP').exists():
                raise KeyboardInterrupt()
            if service.poll() is not None:
                raise RuntimeError('服务退出；查看logs中的rank0.log和launcher.log')
            try:
                http(state['url']+'/health')
                break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('等待 health 超时')
            time.sleep(3)
        state['phase'] = 'FIRST_REQUEST'
        save()
        response = request(state['url'], tokens=16)
        (r/'FIRST_REQUEST.json').write_text(json.dumps(response, ensure_ascii=False, indent=2)+'\n')
        assert 'error' not in response and response.get('meta_info', {}).get('completion_tokens', 0) > 0, response
        state['phase'] = 'FREQUENCY_CHECK'
        save()
        sampler, state['frequency'] = prepare_compute(r, logs, lambda: checked_warmup(state['url']))
        state['phase'] = 'READY'
        save()
        print('READY；频率状态=' + str(state['frequency']) + '；另开终端执行 bash run.sh request 或 bench。Ctrl-C 或 bash run.sh stop 停止。', flush=True)
        frequency_warned = False
        while not (r/'STOP').exists():
            if sampler is not None:
                condition = (False if sampler['out_of_range_reads'] else
                             None if sampler['missing_reads'] or sampler['error'] or not sampler['sample_rounds'] else True)
                state['frequency'] = {'performance_condition_met': condition,
                    'out_of_range_reads': sampler['out_of_range_reads'], 'missing_reads': sampler['missing_reads'],
                    'range_khz': sampler['range_khz'], 'sample_rounds': sampler['sample_rounds']}
                save_json(r/'frequency_summary.json', state['frequency'])
                save()
                if condition is not True and not frequency_warned:
                    warning('运行期计算核频率不满足要求或读数缺失，继续服务，详情见frequency_summary.json')
                    frequency_warned = True
            if service.poll() is not None:
                raise RuntimeError('运行中的服务退出，查看 rank 日志')
            time.sleep(1)
        state['phase'] = 'STOPPED'
    except KeyboardInterrupt:
        state['phase'] = 'STOPPED'
    except BaseException as error:
        state.update(phase='FAILED', error=repr(error))
        raise
    finally:
        if sampler is not None:
            try:
                state['frequency'] = runtime_frequency.stop(sampler, r/'frequency_summary.json')
            except Exception as error:
                state['frequency'] = {'performance_condition_met': None, 'error': repr(error)}
        try:
            if service is not None and service.poll() is None:
                os.killpg(service.pid, signal.SIGTERM)
                service.wait(timeout=75)
            subprocess.run([os.environ['CLIENT_PYTHON'], '-B', '-S',
                            kit/'tools/cleanup_owned_shm.py', r], env=clean, check=True)
            state['cleanup'] = 'confirmed'
        except BaseException as error:
            state.update(cleanup='needs_attention', cleanup_error=repr(error))
        state['resources'] = resources.restore(state.get('cleanup') == 'confirmed')
        save()


if __name__ == '__main__':
    main()
