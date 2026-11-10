"""One runtime environment and an external-library contract for build/run."""
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def environment(dependencies_only=False):
    required = ['PYTHON', 'KUTACC_PREFIX', 'KUPL_PREFIX']
    if not dependencies_only:
        required += ['SGLANG_SOURCE', 'MODEL']
    for key in required:
        if not os.environ.get(key) or '请填写' in os.environ[key]:
            raise ValueError('先填写配置：' + key)
    paths = [Path(os.environ['PYTHON']),
             Path(os.environ['KUTACC_PREFIX']) / 'lib/libkutacc.so',
             Path(os.environ['KUPL_PREFIX']) / 'lib/libkupl.so']
    if not dependencies_only:
        paths += [Path(os.environ['SGLANG_SOURCE']) / 'python/sglang',
                  Path(os.environ['MODEL']) / 'config.json',
                  Path(os.environ['TILING_FILE']), Path(os.environ['CHAT_TEMPLATE'])]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    vendor = Path(os.environ['PYTHON']).parent.parent
    kit = Path(os.environ.get('QWEN_PAYLOAD_ROOT', os.environ['QWEN_KIT']))
    if (kit / 'RUNTIME_LOCK.json').is_file():
        for key, relative in [('SGLANG_SOURCE', 'app/sglang'), ('KERNEL_PACKAGE_ROOT', 'app/kernel')]:
            if Path(os.environ[key]).resolve() != (kit / relative).resolve():
                raise ValueError('部署包必须使用包内应用：' + key)
    env = dict(os.environ)
    env['PYTHONPATH'] = ':'.join(p for p in [os.environ['SGLANG_SOURCE'] + '/python',
                                           os.environ.get('KERNEL_PACKAGE_ROOT', '')] if p)
    env['LD_LIBRARY_PATH'] = ':'.join(p for p in [os.environ['KUTACC_PREFIX'] + '/lib',
        os.environ['KUPL_PREFIX'] + '/lib', str(vendor / 'lib'),
        str(vendor / 'lib/python3.12/site-packages/torch/lib'),
        os.environ.get('COMPILER_RUNTIME_LIB', ''),
        os.environ.get('EXTRA_LIBRARY_PATH', ''), '/usr/lib64', '/lib64'] if p)
    env['LD_PRELOAD'] = os.environ['KUTACC_PREFIX'] + '/lib/libkutacc.so'
    return env


def snapshot(env, dependencies, load_modules=True):
    """Probe the selected interpreter, including actual loaded library paths."""
    code = '''import sys,sysconfig,platform,json,importlib.metadata as md,site,importlib.machinery as machinery
from pathlib import Path
sys.path.extend(p for p in site.getsitepackages() if p not in sys.path)
names=['torch','transformers','tokenizers','safetensors','numpy','fastapi','uvicorn','pyzmq','msgspec']
versions={}
for name in names:
    try: versions[name]=md.version(name)
    except md.PackageNotFoundError: versions[name]=None
imports={}
for name in ('torch','sgl_kernel','sglang'):
    spec=machinery.PathFinder.find_spec(name,sys.path)
    if not spec or not spec.origin: raise RuntimeError('Cannot locate '+name)
    imports[name]=spec.origin
common=list(Path(imports['sgl_kernel']).parent.glob('common_ops*.so'))
if len(common)!=1: raise RuntimeError('Expected one common_ops DSO')
imports['common_ops']=str(common[0])
loaded=[]
if LOAD_MODULES:
    import torch,sgl_kernel,sglang
    from sgl_kernel import common_ops
    imports=dict(sglang=sglang.__file__,sgl_kernel=sgl_kernel.__file__,common_ops=common_ops.__file__,torch=torch.__file__)
    loaded=sorted({line.split()[-1] for line in open('/proc/self/maps') if '/' in line.split()[-1]})
print(json.dumps(dict(machine=platform.machine(),python=list(sys.version_info[:2]),
soabi=sysconfig.get_config_var('SOABI'),packages=versions,loaded=loaded,
imports=imports)))'''.replace('LOAD_MODULES', repr(load_modules))
    probe_env = dict(env)
    if not load_modules:
        probe_env.pop('LD_PRELOAD', None)
    result = subprocess.run([env['PYTHON'], '-B', *([] if load_modules else ['-S']), '-c', code], env=probe_env,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if result.returncode:
        raise RuntimeError('运行环境探测失败，进程返回 ' + str(result.returncode) + '：\n'
                           + result.stdout[-2000:] + result.stderr[-6000:])
    info = json.loads(result.stdout.splitlines()[-1])
    if info['machine'] != 'aarch64' or info['python'] != [3, 12]:
        raise RuntimeError('此部署配方要求 aarch64 / Python 3.12')
    expected = Path(env['SGLANG_SOURCE']).resolve() / 'python'
    if expected not in Path(info['imports']['sglang']).resolve().parents:
        raise RuntimeError('实际导入SGLang不是配置的应用')
    kernel_root = env.get('KERNEL_PACKAGE_ROOT')
    if kernel_root and Path(kernel_root).resolve() not in Path(info['imports']['sgl_kernel']).resolve().parents:
        raise RuntimeError('实际导入sgl_kernel不是配置的应用算子包')
    native = {}
    torch_dir = Path(info['imports']['torch']).resolve().parent
    for row in dependencies['roots']:
        path = Path(row['path']).resolve()
        if torch_dir in path.parents:
            native['torch/' + str(path.relative_to(torch_dir))] = sha256(path)
    # Lock non-system numerical/compiler dependencies and what was really loaded.
    prefixes = ('libkupl.so', 'libkutacc.so', 'libomp.so', 'libopenblas.so',
                'libflang', 'libbsmath', 'libtorch', 'libc10.so',
                'libstdc++.so', 'libgcc_s.so')
    selected = {}
    for name, paths in dependencies['resolved'].items():
        if name.startswith(prefixes):
            for path in paths:
                selected.setdefault(name, set()).add(path)
    locations = {}
    for name, paths in sorted(selected.items()):
        hashes = {sha256(path) for path in paths}
        if len(hashes) != 1:
            raise RuntimeError('同名库解析到不同内容：' + name + ': ' + str(sorted(paths)))
        native[name] = hashes.pop()
        locations[name] = sorted(paths)
    for path in info['loaded']:
        if Path(path).name.startswith(prefixes) and sha256(path) not in native.values():
            raise RuntimeError('实际加载库不在已解析依赖中：' + path)
    return {'identity': {key: info[key] for key in ('machine', 'python', 'soabi', 'packages')}
                        | {'native': native},
            'locations': locations, 'imports': info['imports'],
            'common_sha256': sha256(info['imports']['common_ops']),
            'native_import_performed': load_modules}


def differences(expected, actual, prefix=''):
    """Path-independent comparison; exact builds avoid same-SONAME ambiguity."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        rows = []
        for key in sorted(expected.keys() | actual.keys()):
            label = prefix + '/' + key
            if key not in expected or key not in actual:
                rows.append(label + ': 缺失或新增')
            else:
                rows += differences(expected[key], actual[key], label)
        return rows
    return [] if expected == actual else [prefix + ': 构建/版本不一致']


def verify(kit, current, destination):
    lock = json.loads((kit / 'RUNTIME_LOCK.json').read_text())
    errors = differences(lock['identity'], current['identity'])
    if current['common_sha256'] != lock['common_sha256']:
        errors.append('sgl_kernel common_ops 与打包版本不同')
    report = {'ok': not errors, 'mismatches': errors, 'actual': current}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if errors:
        raise RuntimeError('目标环境与构建锁不匹配；见 ' + str(destination) + '\n' + '\n'.join(errors))


def verify_payload(kit):
    manifest = json.loads((kit / 'PAYLOAD_MANIFEST.json').read_text())
    for name, digest in manifest.items():
        path = kit / name
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise RuntimeError('部署文件变更或缺失：' + name + '；请重新打包，勿混用旧文件')
    actual = {str(p.relative_to(kit / 'app')) for p in (kit / 'app').rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    expected = {name[4:] for name in manifest if name.startswith('app/')}
    if actual != expected:
        raise RuntimeError('app 中存在未登记或缺失的文件，请重新解压部署包')
