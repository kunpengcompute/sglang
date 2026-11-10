"""Freeze this application and prepare reusable rank images; retain no archive."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.check_dependencies import check as check_dependencies
from tools.runtime_contract import environment, sha256, snapshot, verify_payload


def copy_tree(source, target):
    shutil.copytree(source, target, symlinks=False,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store', '.git'))


def build(kit, deploy):
    if (kit / 'RUNTIME_LOCK.json').exists():
        raise RuntimeError('这是已冻结的部署包；更新代码请从构建工具重新打包')
    deploy.mkdir(parents=True, exist_ok=True)
    if (deploy / 'release').exists() or (deploy / 'numa_images').exists():
        raise FileExistsError('使用新的DEPLOY_DIR，不覆盖release或NUMA副本')
    env = environment(dependencies_only=True)
    receipt = Path(env.get('KERNEL_BUILD_RECEIPT') or deploy / 'KERNEL_BUILD.json')
    kernel_build = json.loads(receipt.read_text())
    if kernel_build['phase'] != 'COMPLETE' or not kernel_build['fresh_build']:
        raise RuntimeError('先从源码完成 build.sh')
    if Path(kernel_build['kernel_root']).resolve() != Path(env['KERNEL_PACKAGE_ROOT']).resolve():
        raise RuntimeError('算子输入不是本次源码构建输出')
    if kernel_build['source_commit'] != subprocess.check_output(
            ['git', '-C', env['SGLANG_SOURCE'], 'rev-parse', 'HEAD'], text=True).strip():
        raise RuntimeError('构建记录与当前源码commit不匹配，不复用旧kernel')
    subprocess.run(['git', '-C', env['SGLANG_SOURCE'], 'diff', '--quiet', 'HEAD', '--',
                    'python', 'sgl-kernel'], check=True)
    for key, library in [('KUTACC_PREFIX', 'libkutacc.so'), ('KUPL_PREFIX', 'libkupl.so')]:
        if sha256(Path(env[key])/'lib'/library) != kernel_build['dependencies'][key]['sha256']:
            raise RuntimeError('SDK与原构建记录不匹配：' + key)
    report = check_dependencies(env, deploy / 'PACKAGE_DEPENDENCIES.json')
    current = snapshot(env, report, load_modules=False)
    source = Path(env['SGLANG_SOURCE']).resolve()
    kernel = Path(current['imports']['sgl_kernel']).resolve().parent
    try:
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    except subprocess.CalledProcessError:
        commit, dirty = None, 'source has no Git metadata; payload hashes are authoritative'
    with tempfile.TemporaryDirectory(prefix='.qwen-pack-', dir=deploy) as tmp:
        stage = Path(tmp) / 'qwen_920f_deploy'
        stage.mkdir()
        for name in ('run.sh', 'start.py', 'config.example.sh', 'README.md', 'VERSION.json'):
            shutil.copy2(kit / name, stage / name)
        shutil.copy2(env['QWEN_CONFIG_PATH'], stage / 'config.sh')
        copy_tree(kit / 'tools', stage / 'tools')
        shutil.copy2(receipt, stage / 'KERNEL_BUILD.json')
        (stage / 'app/sglang/python').mkdir(parents=True)
        copy_tree(source / 'python/sglang', stage / 'app/sglang/python/sglang')
        for name in ('LICENSE', 'NOTICE'):
            if (source / name).is_file():
                shutil.copy2(source / name, stage / 'app/sglang' / name)
        copy_tree(kernel, stage / 'app/kernel/sgl_kernel')
        # Do not accidentally distribute installed vendor runtimes inside a kernel package.
        banned = ('libkupl', 'libkutacc', 'libomp', 'libtorch', 'libc10', 'libopenblas',
                  'libflang', 'libbsmath', 'libstdc++', 'libgcc_s')
        for path in (stage / 'app').rglob('*'):
            if path.is_file() and path.name.startswith(banned) and '.so' in path.name:
                raise RuntimeError('应用目录含外部运行库，请清理安装包再构建：' + str(path))
        copied_common = stage / 'app/kernel/sgl_kernel' / Path(current['imports']['common_ops']).name
        if kernel_build['native_sha256'].get(copied_common.name) != current['common_sha256']:
            raise RuntimeError('common_ops不是本轮编译生成的文件')
        if sha256(copied_common) != current['common_sha256']:
            raise RuntimeError('构建中算子库发生变化，请停止更新后重试')
        actual_native = {p.name: sha256(p) for p in (stage/'app/kernel/sgl_kernel').glob('*.so')}
        if actual_native != kernel_build['native_sha256']:
            raise RuntimeError('原生库集合或哈希与构建记录不同')
        lock = {'schema': 1, 'policy': 'exact-vendor-builds-external-paths-may-change',
                'identity': current['identity'], 'common_sha256': current['common_sha256']}
        (stage / 'RUNTIME_LOCK.json').write_text(json.dumps(lock, indent=2) + '\n')
        provenance = {'source_commit': commit, 'source_dirty_status': dirty,
                      'build_imports': current['imports'], 'build_library_paths': current['locations'],
                      'external_libraries_in_release': False, 'model_weights_in_release': False,
                      'numa_copies_created_on_target': 16, 'new_model_service_validation': False,
                      'kernel_source_build': kernel_build}
        (stage / 'BUILD_INFO.json').write_text(json.dumps(provenance, indent=2) + '\n')
        version = json.loads((stage / 'VERSION.json').read_text())
        version.update(source_commit=commit, reference_common_ops_sha256=current['common_sha256'],
                       launcher_revision=version['launcher_revision'],
                       current_validation='packaging_and_runtime_contract_only_no_new_model_test')
        (stage / 'VERSION.json').write_text(json.dumps(version, indent=2) + '\n')
        # The external config remains editable and outside payload integrity checks.
        config = (stage / 'config.sh').read_text()
        config = config.replace('export SGLANG_SOURCE="/请填写/sglang"',
                                'export SGLANG_SOURCE="${QWEN_KIT}/app/sglang"')
        config = config.replace('export KERNEL_PACKAGE_ROOT=""',
                                'export KERNEL_PACKAGE_ROOT="${QWEN_KIT}/app/kernel"')
        (stage / 'config.sh').write_text(config)
        manifest = {str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob('*'))
                    if p.is_file() and p != stage / 'config.sh'}
        (stage / 'PAYLOAD_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
        verify_payload(stage)
        release = deploy / 'release'
        stage.rename(release)
    # NUMA images reference the final frozen app, never a temporary staging path.
    image_env = dict(env, QWEN_KIT=str(release),
                     SGLANG_SOURCE=str(release / 'app/sglang'),
                     KERNEL_PACKAGE_ROOT=str(release / 'app/kernel'))
    image_env['PYTHONPATH'] = str(release / 'app/sglang/python') + ':' + str(release / 'app/kernel')
    logs = deploy / 'logs/pack'
    logs.mkdir(parents=True, exist_ok=True)
    result = {'phase': 'PREPARING_IMAGES', 'release': str(release),
              'source_commit': commit, 'application_files': len(manifest),
              'source_build_receipt': str(receipt), 'native_recompiled': False}
    receipt_path = deploy / 'PACKAGE.json'
    receipt_path.write_text(json.dumps(result, indent=2) + '\n')
    try:
        with (logs / 'numa_images.log').open('w') as log:
            subprocess.run([env['CLIENT_PYTHON'], '-u', '-B', '-S',
                release / 'tools/prepare_numa_images.py', '--python', env['PYTHON'],
                '--kutacc-prefix', env['KUTACC_PREFIX'], '--kupl-prefix', env['KUPL_PREFIX'],
                '--runtime-lock', release / 'RUNTIME_LOCK.json',
                '--output', deploy / 'numa_images'], env=image_env, stdout=log,
                stderr=subprocess.STDOUT, check=True)
        result['phase'] = 'COMPLETE'
    except BaseException as error:
        result.update(phase='FAILED', error=repr(error))
        raise
    finally:
        receipt_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    build(Path(os.environ['QWEN_KIT']), Path(os.environ['DEPLOY_DIR']).expanduser().resolve())
