"""Prune completed deployment logs; retain state, resource and build receipts."""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import re
import time


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def log_files(folder):
    if folder.is_symlink() or not folder.is_dir():
        return []
    return [path for path in folder.rglob('*') if path.is_file()
            and not path.is_symlink() and not any(p.is_symlink() for p in path.parents)
            and re.fullmatch(r'.+\.(?:log|jsonl)(?:\.\d+)?', path.name)]


def candidates(deploy):
    current = {read(path).get('instance_id')
               for path in (deploy / 'instances/current').glob('*.json')}
    for run in (deploy / 'instances').glob('*'):
        if not run.is_dir() or run.is_symlink() or run.name in current:
            continue
        state_path = run / 'STATE.json'
        state = read(state_path)
        safe = state.get('resources_acquired') is False or (
            state.get('cleanup') == 'confirmed'
            and state.get('resources', {}).get('status') == 'confirmed')
        if state.get('phase') in ('STOPPED', 'FAILED') and safe:
            yield 'node:' + state.get('hostname', ''), state_path, log_files(deploy / 'logs/instances' / run.name)

    cluster = Path(os.environ.get('PD_CLUSTER_DIR', str(deploy / 'cluster'))).resolve()
    if not cluster.is_relative_to(deploy):
        return  # A custom external cluster directory is outside this cleanup scope.
    current = read(cluster / 'CURRENT.json').get('run')
    for run in (cluster / 'runs').glob('*'):
        if not run.is_dir() or run.is_symlink() or str(run.resolve()) == current:
            continue
        state_path = run / 'STATE.json'
        state = read(state_path)
        if (state.get('phase') in ('STOPPED', 'FAILED')
                and state.get('cleanup', {}).get('confirmed') is True):
            yield 'cluster', state_path, log_files(run)

    for receipt in (deploy / 'logs/driver').glob('*.log.done.json'):
        if receipt.is_symlink():
            continue
        state = read(receipt)
        if state.get('completed') is True:
            log = receipt.with_name(receipt.name.removesuffix('.done.json'))
            files = [path for path in log.parent.glob(log.name + '*')
                     if path.is_file() and not path.is_symlink()
                     and (path == log or path.name.removeprefix(log.name + '.').isdigit())]
            yield 'driver:' + state.get('hostname', ''), receipt, files


def prune(deploy, keep_days, keep_runs, apply=False):
    groups = defaultdict(list)
    for group, record, files in candidates(deploy):
        if files:
            groups[group].append((record.stat().st_mtime, record, files))
    cutoff = time.time() - keep_days * 86400
    selected = []
    for rows in groups.values():
        for index, (modified, record, files) in enumerate(sorted(rows, key=lambda r: r[0], reverse=True)):
            if index < keep_runs and modified >= cutoff:
                continue
            selected.append({'record': str(record), 'files': [str(p) for p in files],
                             'bytes': sum(p.stat().st_size for p in files)})
            if apply:
                for path in files:
                    path.unlink(missing_ok=True)
    return {'apply': apply, 'keep_days': keep_days, 'keep_runs': keep_runs,
            'bytes': sum(row['bytes'] for row in selected), 'selected': selected,
            'scope': 'completed logs only; current/active/unconfirmed and all JSON state/build receipts retained'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Actually delete selected completed log files')
    args = parser.parse_args()
    deploy = Path(os.environ['DEPLOY_DIR']).expanduser().resolve()
    days = int(os.environ.get('LOG_KEEP_DAYS', '7'))
    runs = int(os.environ.get('LOG_KEEP_RUNS', '5'))
    if min(days, runs) < 1:
        parser.error('LOG_KEEP_DAYS and LOG_KEEP_RUNS must be positive')
    if not deploy.is_dir():
        parser.error('DEPLOY_DIR does not exist')
    with (deploy / '.prune-logs.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(json.dumps(prune(deploy, days, runs, args.apply), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
