"""Remove only inactive KuPL windows named by this run's recorded scheduler PIDs."""
import json
import os
from pathlib import Path
import re
import time

root = Path(__import__("sys").argv[1])
owned = set()
for service in root.glob('service'):
    if service.is_dir():
        for record in service.glob('rank*.json'):
            owned.add(int(json.loads(record.read_text())['pid']))
live = []
for pid in owned:
    stat = Path('/proc') / str(pid) / 'stat'
    if stat.exists() and stat.read_text().split()[2] != 'Z':
        live.append(pid)
if live:
    raise RuntimeError(f'Recorded scheduler PIDs still active or reused: {live}')

candidates = []
for path in Path('/dev/shm').glob('shm_mmap_*'):
    match = re.fullmatch(r'shm_mmap_[0-9a-f]+_\d+_(\d+)', path.name)
    stat = path.stat()
    if match and int(match[1]) in owned and stat.st_uid == os.getuid():
        candidates.append((path, stat.st_ino, stat.st_size))

# Keep an exact record, including inode, before unlinking any owned stale name.
report = {'epoch': time.time(), 'hostname': os.uname().nodename,
          'job_id': os.environ.get('BATCH_JOB_ID'), 'recorded_schedulers': sorted(owned),
          'candidates': [{'path': str(p), 'inode': inode, 'size': size}
                         for p, inode, size in candidates], 'removed': []}
output = root / 'cleanup_owned_shm.json'
output.write_text(json.dumps(report, indent=2) + '\n')
for path, inode, size in candidates:
    if path.stat().st_ino != inode:
        raise RuntimeError(f'Inode changed before cleanup: {path}')
    path.unlink()
    report['removed'].append(str(path))
report['remaining'] = [str(p) for p, _, _ in candidates if p.exists()]
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'owned_windows_removed': len(report['removed']),
                  'remaining': report['remaining']}))
