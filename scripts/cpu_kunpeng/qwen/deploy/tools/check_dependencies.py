"""Check trusted local ELF dependencies without importing Torch or sgl_kernel."""
import importlib.machinery
import json
from pathlib import Path
import re
import shutil
import subprocess


def inspect(env):
    # Never preload KuTACC into the diagnostic interpreter/ldd: that would fail
    # before we can report the whole set of missing dependencies.
    clean = dict(env)
    clean.pop('LD_PRELOAD', None)
    report = {'ok': False, 'library_path': clean.get('LD_LIBRARY_PATH', ''),
              'missing_commands': [], 'errors': [], 'roots': [], 'resolved': {}}
    for command in ('ldd', 'numactl', 'taskset'):
        if not shutil.which(command, path=clean.get('PATH')):
            report['missing_commands'].append(command)
    python = Path(env['PYTHON'])
    prefix = python.parent.parent
    # This recipe targets Python 3.12; ignore other installations/compat aliases.
    sites = [prefix / 'lib/python3.12/site-packages']
    search = [p for p in clean.get('PYTHONPATH', '').split(':') if p]
    search += [str(p) for p in sites]
    roots = {python}
    for key, name in [('KUTACC_PREFIX', 'libkutacc.so'), ('KUPL_PREFIX', 'libkupl.so')]:
        roots.add(Path(env[key]) / 'lib' / name)
    # PathFinder only locates packages; it does not execute their __init__.py.
    for name in ('torch', 'sgl_kernel'):
        spec = importlib.machinery.PathFinder.find_spec(name, search)
        if not spec or not spec.submodule_search_locations:
            report['errors'].append('找不到包目录：' + name)
            continue
        directory = Path(next(iter(spec.submodule_search_locations)))
        roots.update(p for p in directory.rglob('*')
                     if p.is_file() and (p.name.endswith('.so') or '.so.' in p.name))
    seen = set()
    for root in sorted(roots):
        path = root.resolve()
        if path in seen:
            continue
        seen.add(path)
        row = {'path': str(root), 'missing': [], 'errors': []}
        report['roots'].append(row)
        if not root.is_file():
            row['errors'].append('文件不存在或软链接失效')
            continue
        if 'ldd' in report['missing_commands']:
            continue
        try:
            result = subprocess.run(['ldd', str(root)], env=clean, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
            output = result.stdout
            row['missing'] = sorted(set(re.findall(r'^\s*(\S+)\s+=>\s+not found', output, re.M)))
            # ABI/version mismatches also fail, even if all filenames exist.
            row['errors'] += [line.strip() for line in output.splitlines()
                              if ('not found' in line and '=> not found' not in line)
                              or 'error while loading' in line]
            if result.returncode and not (row['missing'] or row['errors']):
                row['errors'].append(output.strip() or 'ldd 返回 ' + str(result.returncode))
            for name, resolved in re.findall(r'^\s*(\S+)\s+=>\s+(/\S+)', output, re.M):
                report['resolved'].setdefault(name, [])
                if resolved not in report['resolved'][name]:
                    report['resolved'][name].append(resolved)
        except (OSError, subprocess.TimeoutExpired) as error:
            row['errors'].append(str(error))
    report['missing_libraries'] = sorted({name for row in report['roots'] for name in row['missing']})
    report['ok'] = not (report['missing_commands'] or report['errors'] or
                        any(row['missing'] or row['errors'] for row in report['roots']))
    return report


def check(env, destination):
    report = inspect(env)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('动态库检查报告：' + str(destination), flush=True)
    if not report['ok']:
        for row in report['roots']:
            if row['missing'] or row['errors']:
                print(row['path'] + ': ' + '; '.join(row['missing'] + row['errors']), flush=True)
        for error in report['errors']:
            print(error, flush=True)
        if report['missing_commands']:
            print('缺少命令：' + ', '.join(report['missing_commands']), flush=True)
        raise RuntimeError('依赖未齐全：见报告全部缺失项；配置 COMPILER_RUNTIME_LIB / '
                           'EXTRA_LIBRARY_PATH，或安装与当前二进制匹配的运行库。')
    print('原生动态库依赖检查通过；版本/算子接口还需后续实际导入检查。', flush=True)
    return report
