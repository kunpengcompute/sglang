"""Deployment resource lifecycle; numerical code and graph execution are untouched."""
import fcntl
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
from tools import compute_frequency as cores
from tools import runtime_frequency

SYS = Path('/sys/devices/system')
PAGE_KIB = (64, 2048, 32768, 1048576)


def save(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def budget():
    total = int(os.environ['SGLANG_KUNPENG_WEIGHTS_HBW_POOL_SIZE_MB'])
    sizes = [int(os.environ['SGLANG_KUNPENG_' + phase + '_SHM_SIZE_MB'])
             for phase in ('PREFILL', 'DECODE')]
    # This launcher uses one role/window; equal sizes also keep either role valid.
    if total != 4000 or sizes[0] != sizes[1] or not 0 < sizes[0] < total:
        raise ValueError('当前split配方要求总预算4000MiB、prefill/decode SHM相等且小于总预算')
    private = total - sizes[0]
    configured = os.environ.get('HBM_HUGEPAGE_MB_PER_NODE', 'auto')
    pages = math.ceil(private / 2) + 2 if configured == 'auto' else math.ceil(float(configured) / 2)
    if pages < math.ceil(private / 2):
        raise ValueError('配置的大页容量不足以覆盖私有HBW池')
    return dict(total_mib=total, shm_mib=sizes[0], private_mib=private,
                page_kib=2048, required_pages=math.ceil(private / 2), reserved_pages=pages)


def page_snapshot():
    return {str(p): int(p.read_text()) for p in
            sorted((SYS / 'node').glob('node*/hugepages/*/nr_hugepages'))}


def uniform_page_args(snapshot):
    args = []
    for nodes in (range(16), range(16, 32)):
        for size in PAGE_KIB:
            values = {snapshot.get(str(SYS / f'node/node{n}/hugepages/hugepages-{size}kB/nr_hugepages'), 0)
                      for n in nodes}
            if len(values) != 1:
                raise RuntimeError('现有大页配置不均匀，旧hugepage.sh不能无损保留；请使用check模式由管理员准备')
            args.append(str(values.pop()))
    return args


def call_helper(name, args, log):
    prefix = Path(os.environ['PERF_TUNE'])
    with Path(log).open('a') as stream:
        result = subprocess.run([str(prefix / 'root_proxy'), name, *map(str, args)],
                                cwd=prefix, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=180)
    return result.returncode


def apply_frequency(targets, tolerance, log, receipt):
    """Retry only out-of-range owned CPUs; retain helper codes and measured values."""
    pending = dict(targets)
    attempts = []
    readings = {}
    for attempt in range(3):
        groups = {}
        for cpu, khz in pending.items():
            groups.setdefault(khz, []).append(cpu)
        codes = {str(khz): call_helper('set_cpu_frequency.sh',
                    [','.join(map(str, cpus)), f'{khz/1e6:g}GHz'], log)
                 for khz, cpus in groups.items()}
        time.sleep(1)
        readings = {cpu: cores.read_khz(cpu) for cpu in targets}
        pending = {cpu: khz for cpu, khz in targets.items()
                   if abs(readings[cpu] - khz) > tolerance}
        attempts.append({'attempt': attempt, 'helper_returncodes': codes,
                         'readings_khz': readings, 'out_of_range': list(pending)})
        save(receipt, {'attempts': attempts, 'confirmed': not pending})
        if not pending:
            break
    return readings, list(pending)


class WarmupFailed(Exception):
    pass


def warning(message):
    print('警告：' + message + '；服务将继续启动/运行。', flush=True)


class Resources:
    def __init__(self, run, logs):
        self.run, self.logs = Path(run), Path(logs)
        self.before = {'hostname': socket.gethostname(), 'budget': budget(),
                       'pages': page_snapshot(), 'frequency': {}}
        self.pages_after = None
        self.frequency_changed = []
        self.frequency_restore = {}
        self.lock = None

    def prepare(self):
        # Serialize resource setters from this toolkit on the same host/user.
        for process in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                args = process.read_bytes().decode(errors='replace').split('\0')
            except OSError:
                continue
            if any(Path(arg).name == 'rank_entry.py' or arg == 'sglang.launch_server' for arg in args if arg):
                raise RuntimeError('节点上已有模型服务，本TP16部署不修改其机器资源')
        self.lock = open(f'/tmp/qwen920f-resources-{os.getuid()}.lock', 'a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(self.run / 'resources_before.json', self.before)
        mode = os.environ.get('HBM_HUGEPAGE_MODE', 'ensure')
        if mode not in ('check', 'ensure'):
            raise ValueError('HBM_HUGEPAGE_MODE应为check或ensure')
        required = self.before['budget']['required_pages']
        paths = [SYS / f'node/node{n}/hugepages/hugepages-2048kB/free_hugepages'
                 for n in range(16, 32)]
        if mode == 'ensure' and any(int(p.read_text()) < required for p in paths):
            args = uniform_page_args(self.before['pages'])
            # Never shrink prior reservations; preserve DDR and other page sizes.
            args[5] = str(max(int(args[5]), self.before['budget']['reserved_pages']))
            try:
                rc = call_helper('hugepage.sh', args, self.logs / 'hugepages.log')
            finally:
                self.pages_after = page_snapshot()
                save(self.run / 'resources_applied.json', {'pages': self.pages_after})
            if rc or any(int(p.read_text()) < required for p in paths):
                raise RuntimeError('大页准备未满足实际需求，详见hugepages.log')
        self.prepare_frequency()

    def prepare_frequency(self):
        mode = os.environ.get('CPU_FREQ_MODE', 'keep')
        if mode not in ('keep', 'apply'):
            raise ValueError('CPU_FREQ_MODE应为keep或apply')
        if mode == 'keep':
            return
        candidates = [r * 38 + off for r in range(16) for off in
                      (*cores.PREFERRED_WORKER_OFFSETS, cores.CALLER_OFFSET,
                       *cores.REPLACEMENT_OFFSETS)]
        try:
            allowed = os.sched_getaffinity(0)
            restore = os.environ.get('CPU_FREQ_RESTORE_MHZ', '').strip()
            for cpu in candidates:
                directory = SYS / f'cpu/cpu{cpu}/cpufreq'
                row = {field: (directory / field).read_text().strip() for field in
                       ('scaling_governor', 'scaling_min_freq', 'scaling_max_freq', 'scaling_setspeed')}
                related = cores.parse_cpu_list((directory / 'related_cpus').read_text().replace(' ', ','))
                if not related <= allowed or row['scaling_governor'] != 'userspace':
                    raise RuntimeError('调频共享域超出当前允许CPU，或不是userspace governor')
                # Some 920F drivers expose a measured value through scaling_setspeed.
                # Do not invent the prior nominal setting by rounding that value.
                available = (directory / 'scaling_available_frequencies')
                frequencies = available.read_text().split() if available.exists() else []
                if restore:
                    previous = int(round(float(restore) * 1000))
                elif row['scaling_setspeed'] in frequencies:
                    previous = int(row['scaling_setspeed'])
                else:
                    raise RuntimeError('无法确定原标称频率，请配置CPU_FREQ_RESTORE_MHZ；本次不主动调频')
                if not int(row['scaling_min_freq']) <= previous <= int(row['scaling_max_freq']):
                    raise ValueError('恢复频率超出当前策略范围')
                self.before['frequency'][str(cpu)] = row
                self.frequency_restore[cpu] = previous
            save(self.run / 'resources_before.json', self.before)
            target = float(os.environ['CPU_FREQ_TARGET_MHZ'])
            # Record before invoking: the helper may change hardware and then return nonzero.
            self.frequency_changed = candidates
            save(self.run / 'frequency_restore.json', self.frequency_restore)
            _, bad = apply_frequency(dict.fromkeys(candidates, round(target * 1000)),
                        round(float(os.environ.get('CPU_FREQ_TOLERANCE_MHZ', '20')) * 1000),
                        self.logs / 'frequency_setup.log', self.run / 'frequency_setup_attempts.json')
            if bad:
                warning('有限重试后仍有计算候选核不达目标频率：' + str(bad))
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            warning('调频设置未完成：' + str(error))
            save(self.run / 'frequency_setup_warning.json', {'warning': str(error)})

    def restore(self, service_clean):
        report = {'service_cleanup_confirmed': service_clean, 'errors': [], 'pages_restored': None,
                  'frequency_restore_requested': bool(self.frequency_changed)}
        try:
            if not service_clean:
                raise RuntimeError('服务资源尚未确认释放，保留机器设置并记录待恢复项')
            if self.frequency_changed:
                # Compare policies and measured frequency with the requested tolerance.
                tolerance = round(float(os.environ.get('CPU_FREQ_TOLERANCE_MHZ', '20')) * 1000)
                readings, bad = apply_frequency(self.frequency_restore, tolerance,
                        self.logs / 'frequency_restore.log', self.run / 'frequency_restore_attempts.json')
                report['errors'].extend(f'CPU{cpu}恢复读数未满足容差' for cpu in bad)
                for cpu in self.frequency_changed:
                    directory = SYS / f'cpu/cpu{cpu}/cpufreq'
                    for field in ('scaling_governor', 'scaling_min_freq', 'scaling_max_freq'):
                        if (directory / field).read_text().strip() != self.before['frequency'][str(cpu)][field]:
                            report['errors'].append(f'CPU{cpu}策略字段{field}已变化')
                report['frequency_readings_khz'] = readings
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            report['errors'].append(str(error))
        # Hugepage restoration is independent of a frequency helper failure.
        if self.pages_after is not None and service_clean:
            try:
                if page_snapshot() != self.pages_after:
                    raise RuntimeError('大页预留已被其他操作修改，不覆盖当前设置')
                rc = call_helper('hugepage.sh', uniform_page_args(self.before['pages']),
                                 self.logs / 'hugepages_restore.log')
                report['pages_restored'] = page_snapshot() == self.before['pages']
                if rc or not report['pages_restored']:
                    report['errors'].append('大页恢复未确认成功')
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                report['errors'].append(str(error))
        report['status'] = 'needs_attention' if report['errors'] else 'confirmed'
        save(self.run / 'resources_after.json', report)
        if self.lock:
            self.lock.close()
        return report


def frequency_window():
    target = float(os.environ.get('CPU_FREQ_TARGET_MHZ', '2000'))
    tolerance = float(os.environ.get('CPU_FREQ_TOLERANCE_MHZ', '20'))
    if target <= 0 or not 0 <= tolerance < target:
        raise ValueError('频率目标或容差无效')
    return round((target - tolerance) * 1000), round((target + tolerance) * 1000)


def prepare_compute(run, logs, warmup):
    """Reuse legacy planning/apply; frequency failure never rejects a service."""
    low, high = frequency_window()
    try:
        # Read frequencies while an actual request is running, then wait for it.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(warmup)
            try:
                rows, source = cores.sample_live(10, 0.2)
            finally:
                try:
                    pending.result()
                except Exception as error:
                    raise WarmupFailed('选频负载请求失败') from error
        source['hostname'] = socket.gethostname()
        try:
            plan = cores.make_plan(rows, source, low, high)
            save(run / 'compute_plan.json', plan)
            if os.environ.get('CPU_REMAP_SAME_NUMA', '1') == '1':
                receipt = cores.apply_plan(plan, run / 'service', False)
                save(run / 'compute_remap.json', receipt)
            elif plan['replacement_count']:
                raise RuntimeError('已关闭换核，保留原核表')
        except RuntimeError as error:
            warning('未完成同NUMA换核：' + str(error))
            # Observe the original binding, not an unapplied candidate map.
            plan = cores.make_plan({cpu: {'min_khz': low, 'max_khz': high,
                                         'sample_count': 1} for cpu in range(608)},
                                   {'hostname': socket.gethostname()}, low, high)
            plan['frequency_evidence'] = {'status': 'ORIGINAL_BINDING_FALLBACK', 'reason': str(error)}
            save(run / 'compute_plan.json', plan)
        # Confirm actual caller/worker placement before labeling 512 CPUs as compute.
        for rank, pid in cores.load_rank_pids(run / 'service').items():
            actual = cores.task_rows(pid)
            row = plan['ranks'][str(rank)]
            if cores.task_affinity(pid, pid) != {row['caller_cpu']} or any(
                    len(cores.find_singleton(actual, cpu)) != 1 for cpu in row['selected_worker_cpus']):
                raise RuntimeError('实际worker绑定与核表不匹配，频率条件未知')
        admission = runtime_frequency.start(logs / 'frequency_admission.jsonl', run / 'compute_plan.json')
        try:
            try:
                warmup()
            except Exception as error:
                raise WarmupFailed('换核后预热请求失败') from error
        finally:
            report = runtime_frequency.stop(admission, run / 'frequency_admission.json')
        if report['performance_condition_met'] is not True:
            warning('频率不满足要求或读数不完整，详见frequency_admission.json')
        sampler = runtime_frequency.start(logs / 'frequency.jsonl', run / 'compute_plan.json')
        return sampler, {'status': report['status'], 'performance_condition_met': report['performance_condition_met'],
                         'range_khz': [low, high]}
    except (OSError, ValueError, RuntimeError) as error:
        warning('频率验收未完成：' + str(error))
        report = {'status': 'UNKNOWN', 'performance_condition_met': None, 'error': str(error),
                  'range_khz': [low, high]}
        save(run / 'frequency_summary.json', report)
        return None, report
