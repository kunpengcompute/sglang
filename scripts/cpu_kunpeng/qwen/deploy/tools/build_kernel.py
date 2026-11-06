"""Build this checkout's CPU extensions into an empty, external deployment."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = Path(os.environ['SGLANG_SOURCE']).resolve()
    deploy = Path(os.environ['DEPLOY_DIR']).resolve()
    build = deploy / 'build/sgl-kernel'
    kernel = deploy / 'build/kernel'
    if build.exists() or kernel.exists():
        raise FileExistsError('Use a fresh DEPLOY_DIR; existing build/kernel is not reused')
    for key in ('PYTHON', 'CC', 'CXX'):
        if not Path(os.environ[key]).is_file():
            raise FileNotFoundError(key + ': ' + os.environ[key])
    for key in ('KUTACC_PREFIX', 'KUPL_PREFIX'):
        prefix = Path(os.environ[key])
        if not (prefix / 'include').is_dir() or not (prefix / 'lib').is_dir():
            raise FileNotFoundError('SDK include/lib: ' + str(prefix))
    env = dict(os.environ)
    for key in ('LD_PRELOAD', 'PYTHONHOME', 'PYTHONPATH'):
        env.pop(key, None)
    python = Path(env['PYTHON'])
    vendor = python.parent.parent
    env['CONDA_PREFIX'] = str(vendor)
    env['PATH'] = ':'.join((str(python.parent), str(Path(env['CXX']).parent), env['PATH']))
    env['KUTACC_LIB'] = env['KUTACC_PREFIX'] + '/lib'
    env['KUTACC_INCLUDE'] = env['KUTACC_PREFIX'] + '/include'
    env['CPATH'] = env['KUPL_PREFIX'] + '/include'
    env['LIBRARY_PATH'] = env['KUPL_PREFIX'] + '/lib'
    libraries = [env['KUTACC_LIB'], env['KUPL_PREFIX'] + '/lib', str(vendor / 'lib'),
                 str(vendor / 'lib/python3.12/site-packages/torch/lib'),
                 env.get('COMPILER_RUNTIME_LIB', ''), env.get('EXTRA_LIBRARY_PATH', '')]
    env['LD_LIBRARY_PATH'] = ':'.join(p for p in libraries if p)
    deploy.mkdir(parents=True, exist_ok=True)
    logs = deploy / 'logs/build'
    logs.mkdir(parents=True, exist_ok=True)
    record = {'phase': 'BUILDING', 'started_epoch': time.time(), 'source': str(source),
              'build': str(build), 'kernel_root': str(kernel), 'fresh_build': True,
              'prebuilt_kernel_input': False, 'commands': [], 'libraries': libraries}
    report = deploy / 'KERNEL_BUILD.json'

    def save():
        report.write_text(json.dumps(record, indent=2) + '\n')

    def call(argv, log):
        record['commands'].append([str(x) for x in argv])
        save()
        with (logs / log).open('w') as stream:
            subprocess.run(list(map(str, argv)), env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)

    try:
        record['source_commit'] = subprocess.check_output(
            ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        record['source_dirty_status'] = subprocess.check_output(
            ['git', '-C', str(source), 'status', '--porcelain'], text=True)
        record['dependencies'] = {
            key: {'prefix': env[key], 'sha256': digest(Path(env[key]) / 'lib' / library)}
            for key, library in [('KUTACC_PREFIX', 'libkutacc.so'),
                                 ('KUPL_PREFIX', 'libkupl.so')]}
        call(['cmake', '-S', source / 'sgl-kernel/csrc/cpu', '-B', build,
              '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
              '-DCMAKE_C_COMPILER=' + env['CC'], '-DCMAKE_CXX_COMPILER=' + env['CXX'],
              '-DPython_EXECUTABLE=' + str(python), '-DCMAKE_INSTALL_PREFIX=' + str(kernel),
              '-DCMAKE_INSTALL_RPATH=' + ';'.join(p for p in libraries if p)], 'configure.log')
        call(['cmake', '--build', build, '--parallel', env.get('BUILD_JOBS', '16')], 'build.log')
        call(['cmake', '--install', build], 'install.log')
        # Python wrappers come from this checkout; no precompiled source-tree DSO is copied.
        shutil.copytree(source / 'sgl-kernel/python/sgl_kernel', kernel / 'sgl_kernel',
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('*.so', '*.so.*', '*.pyc', '__pycache__'))
        native = list((kernel / 'sgl_kernel').glob('*.so'))
        if not any(p.name.startswith('common_ops') for p in native):
            raise RuntimeError('CMake did not install common_ops')
        record.update(phase='COMPLETE', finished_epoch=time.time(),
                      native_sha256={p.name: digest(p) for p in native},
                      compiled_translation_units=len(json.loads((build / 'compile_commands.json').read_text())))
    except BaseException as error:
        record.update(phase='FAILED', error=repr(error))
        raise
    finally:
        save()
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
