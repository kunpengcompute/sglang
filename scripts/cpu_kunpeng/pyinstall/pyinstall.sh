#!/bin/bash
set -e

# ===================== 环境准备 =====================
ENV_NAME=sgl-ljp-0516
PYTHON_VERSION=3.12

# ===================== 路径定义 =====================
KUTACC_PATH=/root/pacific_ext/huawei/ljp/kutacc/kutacc-630
HPCKIT_PATH=/root/pacific_ext/chenyi/HPCKit_26.0.RC1/HPCKit
SGLANG_PATH=/root/pacific_ext/huawei/ljp/sglang/sglang-0.5.11-open
TRITON_SRC=/root/pacific_ext/huawei/drs/triton/python/triton

CONDA_ENV=/root/pacific_ext/huawei/z00515076/anaconda3/envs/$ENV_NAME
SGLANG_SRC=$SGLANG_PATH/python
SGL_KERNEL_SRC=$SGLANG_PATH/sgl-kernel
BISHENG_LIB=$HPCKIT_PATH/latest/compiler/bisheng/lib
KUPL_LIB=$HPCKIT_PATH/latest/kupl/bisheng/release/lib
KUTACC_LIB=${KUTACC_PATH}/install/lib
SITE_PACKAGES=$CONDA_ENV/lib/python$PYTHON_VERSION/site-packages

# ===================== 标准库检查 =====================
echo "检查 Python 标准库完整性..."
python -c "import encodings, re, enum, functools, collections, keyword; print('标准库可用')" || {
    echo "标准库损坏，请执行: conda install --force-reinstall python=$PYTHON_VERSION -n $ENV_NAME"
    exit 1
}

# ===================== PyInstaller 打包 =====================
pyinstaller \
  --name sglang_server \
  --onedir \
  --noconsole \
  --add-binary "$CONDA_ENV/lib/libpython$PYTHON_VERSION.so.1.0:." \
  --add-binary "$BISHENG_LIB/libomp.so:." \
  --add-binary "$KUPL_LIB/libkupl.so.1:." \
  --add-binary "$KUTACC_LIB/libkutacc.so.25.1.RC1:." \
  --add-binary "$SITE_PACKAGES/torch/lib/*.so:torch/lib" \
  --add-data "$SGLANG_SRC/sglang:sglang" \
  --add-data "$TRITON_SRC:triton" \
  --add-data "$SITE_PACKAGES/sgl_kernel:sgl_kernel" \
  --add-data "$SITE_PACKAGES/atomics:atomics" \
  --add-data "$CONDA_ENV/include/python$PYTHON_VERSION:include/python$PYTHON_VERSION" \
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
  --hidden-import transformer \
  --hidden-import transformers.models.ernie4_5 \
  --hidden-import transformers.models.ernie4_5_moe \
  --hidden-import msgspec \
  --collect-all vllm \
  --collect-all torch \
  --collect-binaries torch \
  $SGLANG_SRC/sglang/launch_server.py

echo "PyInstaller 阶段完成"

# ===================== 强制注入完整标准库 =====================
# 原因：部分环境下不会自动打包标准库，直接把 Conda 环境的标准库复制到打包结果中
STD_LIB_SRC="$CONDA_ENV/lib/python$PYTHON_VERSION"
TARGET_DIR="dist/sglang_server/_internal"

echo "正在将标准库复制到 dist/sglang_server/_internal ..."
rsync -av --exclude='site-packages' "$STD_LIB_SRC/" "$TARGET_DIR/"

echo "正在复制triton ..."
rm -rf ./dist/sglang_server/_internal/triton
cp -rf /root/pacific_ext/huawei/z00515076/anaconda3/envs/sgl-ljp-0516/lib/python3.12/site-packages/triton ./dist/sglang_server/_internal/

echo "正在复制其他库 ..."
\cp -f /root/pacific_ext/huawei/z00515076/anaconda3/envs/sgl-ljp-0516/lib/libstdc++.so.6* ./dist/sglang_server/_internal/
\cp -f /root/pacific_ext/huawei/z00515076/anaconda3/envs/sgl-ljp-0516/lib/libgcc_s.so.1* ./dist/sglang_server/_internal/
\cp -f /root/pacific_ext/huawei/z00515076/anaconda3/envs/sgl-ljp-0516/lib/libomp.so* ./dist/sglang_server/_internal/
\cp -f /root/pacific_ext/huawei/z00515076/anaconda3/envs/sgl-ljp-0516/lib/libpython3.12.so.1.0 ./dist/sglang_server/_internal/
chmod +x ./dist/sglang_server/_internal/*.so*

echo "=================================================================="
echo "打包完成！输出目录: dist/sglang_server"
echo "可直接执行: ./dist/sglang_server/sglang_server"
echo "=================================================================="