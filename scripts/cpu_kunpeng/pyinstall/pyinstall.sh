# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/bin/bash
set -e

# ===================== 路径定义 =====================
PYTHON_VERSION=3.12
source ../env.sh native

echo "[pyinstall] SGLANG_PATH: $SGLANG_PATH"
echo "[pyinstall] CONDA_ENV_PATH: $CONDA_ENV_PATH"
echo "[pyinstall] HPCKIT_PATH: $HPCKIT_PATH"
echo "[pyinstall] KUTACC_PATH: $KUTACC_PATH"

SGLANG_SRC=$SGLANG_PATH/python
SGL_KERNEL_SRC=$SGLANG_PATH/sgl-kernel
BISHENG_LIB=$HPCKIT_PATH/latest/compiler/bisheng/lib
KUPL_LIB=$HPCKIT_PATH/latest/kupl/bisheng/release/lib
KUTACC_LIB=${KUTACC_PATH}/install/lib
SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")

rm -rf $PYINSTALL_PATH/dist

# ===================== PyInstaller 打包 =====================
if [ ! -f sglang_server.spec ]; then
    echo "[pyinstall] generate spec file..."
    pyi-makespec \
      --name sglang_server \
      --onedir \
      --noconsole \
      --add-binary "$CONDA_ENV_PATH/lib/libpython$PYTHON_VERSION.so.1.0:." \
      --add-binary "$KUPL_LIB/libkupl.so.1:." \
      --add-binary "$KUTACC_LIB/libkutacc.so.25.1.RC1:." \
      --add-binary "$SITE_PACKAGES/torch/lib/*.so:torch/lib" \
      --add-binary "$BISHENG_LIB/libomp.so:." \
      --add-binary "$CONDA_ENV_PATH/lib/libstdc++.so.6:." \
      --add-binary "$CONDA_ENV_PATH/lib/libgcc_s.so.1:." \
      --add-binary "/usr/lib64/libdl.so.2:." \
      --add-binary "/usr/lib64/libpthread.so.0:." \
      --add-binary "/usr/lib64/libc.so.6:." \
      --add-binary "/usr/lib64/libutil.so.1:." \
      --add-binary "/usr/lib64/libm.so.6:." \
      --add-binary "/usr/lib64/librt.so.1:." \
      --add-binary "/usr/lib64/libmemkind.so:." \
      --add-binary "/usr/lib64/libhwloc.so.15:." \
      --add-binary "/usr/lib64/libfribidi.so.0:." \
      --add-binary "/usr/lib64/libresolv.so.2:." \
      --add-binary "/usr/lib64/libcrypt.so.1:." \
      --add-data "$SGLANG_SRC/sglang:sglang" \
      --add-data "$SITE_PACKAGES/sgl_kernel:sgl_kernel" \
      --hidden-import torch \
      --hidden-import torchvision \
      --hidden-import triton \
      --hidden-import sglang \
      --hidden-import sgl_kernel \
      --hidden-import pybase64 \
      --hidden-import zmq \
      --hidden-import zmq.asyncio \
      --hidden-import fastapi \
      --hidden-import fastapi.middleware.cors \
      --hidden-import starlette \
      --hidden-import starlette.middleware.cors \
      --hidden-import uvicorn \
      --hidden-import setproctitle \
      --hidden-import openai \
      --hidden-import vllm \
      --hidden-import vllm.logging_utils \
      --hidden-import atomics \
      --hidden-import distro \
      --hidden-import partial_json_parser \
      --hidden-import transformers \
      --hidden-import transformers.models.ernie4_5 \
      --hidden-import transformers.models.ernie4_5_moe \
      --hidden-import msgspec \
      --collect-all vllm \
      --collect-all torch \
      --collect-binaries torch \
      $SGLANG_SRC/sglang/launch_server.py
fi

# ===================== 自动修改 spec 文件 =====================
echo "[pyinstall] modify spec file, move sglang/sgl_kernel out of PYZ..."
python - <<EOF
with open('sglang_server.spec', 'r') as f:
    content = f.read()

filter_code = "a.pure = [m for m in a.pure if not m[0].startswith('sglang') and not m[0].startswith('sgl_kernel')]\n"
if filter_code in content:
    print("Spec already contains filter code, no need to modify")
else:
    # 匹配 pyz = PYZ(a.pure) 并替换
    import re
    new_content, count = re.subn(
        r'pyz = PYZ\(a\.pure.*?\)',
        filter_code + 'pyz = PYZ(a.pure)',
        content
    )
    if count == 1:
        with open('sglang_server.spec', 'w') as f:
            f.write(new_content)
        print("Spec modified successfully")
    else:
        print("Warning: not found expected pyz = PYZ(...) line, please modify spec file manually")
EOF

# ===================== 用修改后的 spec 打包 =====================
echo "[pyinstall] start build..."
pyinstaller sglang_server.spec --distpath ./dist --workpath ./build --noconfirm

# echo "[pyinstall] numa duplication ..."
bash numa_duplication.sh

echo "=================================================================="
echo "complete! output dir: dist/sglang_server"
echo "=================================================================="