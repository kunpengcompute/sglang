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

# ===================== Path definitions =====================
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
KUTACC_LIB=${KUTACC_PATH}/lib
SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")

# ===================== kuccl binary dependencies =====================
# kuccl_pg.py fallback: _kuccl_dir/install/{hucx,xucg}/
# .so files must be placed under kuccl/install/{hucx,xucg}/ to match fallback path
KUCCL_FLAGS=()
if [[ "${SGLANG_ENABLE_KUCCL:-0}" == "1" ]]; then
    echo "[pyinstall] SGLANG_ENABLE_KUCCL=1, adding kuccl binaries..."
    KUCCL_FLAGS=(
      --add-binary "/usr/lib64/libnuma.so:."
      --add-binary "/usr/lib64/librdmacm.so.1:."
      --add-binary "/usr/lib64/libsdma_dk.so:."
      # UCX direct deps -> kuccl/install/hucx/lib/
      --add-binary "$HUCX_DIR/lib/libucs.so:kuccl/install/hucx/lib"
      --add-binary "$HUCX_DIR/lib/libucm.so.0:kuccl/install/hucx/lib"
      --add-binary "$HUCX_DIR/lib/libucp.so:kuccl/install/hucx/lib"
      --add-binary "$HUCX_DIR/lib/libuct.so.0:kuccl/install/hucx/lib"
      # UCX transport plugins -> kuccl/install/hucx/lib/ucx/
      --add-binary "$HUCX_DIR/lib/ucx/libuct_ib.so:kuccl/install/hucx/lib/ucx"
      --add-binary "$HUCX_DIR/lib/ucx/libuct_rdmacm.so:kuccl/install/hucx/lib/ucx"
      --add-binary "$HUCX_DIR/lib/ucx/libuct_cma.so:kuccl/install/hucx/lib/ucx"
      --add-binary "$HUCX_DIR/lib/ucx/libuct_sdma.so:kuccl/install/hucx/lib/ucx"
      # UCG direct dep -> kuccl/install/xucg/lib/
      --add-binary "$XUCG_DIR/lib/libucg.so:kuccl/install/xucg/lib"
      # UCG plan plugins -> kuccl/install/xucg/lib/planc/
      --add-binary "$XUCG_DIR/lib/planc/libucg_planc_ucx.so:kuccl/install/xucg/lib/planc"
      --add-binary "$XUCG_DIR/lib/planc/libucg_planc_stars.so:kuccl/install/xucg/lib/planc"
      --add-binary "$XUCG_DIR/lib/planc/libucg_planm_ucx_hicoll.so:kuccl/install/xucg/lib/planc"
    )
fi

# ===================== PyInstaller flags =====================
PYI_FLAGS=(
  --name sglang_server
  --onedir
  --noconsole
  --add-binary "$CONDA_ENV_PATH/lib/libpython$PYTHON_VERSION.so.1.0:."
  --add-binary "$KUPL_LIB/libkupl.so.1:."
  --add-binary "$KUTACC_LIB/libkutacc.so.25.1.RC1:."
  --add-binary "$KUCCL_PATH/kuccl_backend_pg*.so:kuccl/"
  --add-data "$KUCCL_PATH/kuccl_pg.py:kuccl/"
  --add-binary "$SITE_PACKAGES/torch/lib/*.so:torch/lib"
  --add-binary "$BISHENG_LIB/libomp.so:."
  --add-binary "$CONDA_ENV_PATH/lib/libstdc++.so.6:."
  --add-binary "$CONDA_ENV_PATH/lib/libgcc_s.so.1:."
  --add-binary "/usr/lib64/libdl.so.2:."
  --add-binary "/usr/lib64/libpthread.so.0:."
  --add-binary "/usr/lib64/libc.so.6:."
  --add-binary "/usr/lib64/libutil.so.1:."
  --add-binary "/usr/lib64/libm.so.6:."
  --add-binary "/usr/lib64/librt.so.1:."
  --add-binary "/usr/lib64/libmemkind.so:."
  --add-binary "/usr/lib64/libhwloc.so.15:."
  --add-binary "/usr/lib64/libfribidi.so.0:."
  --add-binary "/usr/lib64/libresolv.so.2:."
  --add-binary "/usr/lib64/libcrypt.so.1:."
  "${KUCCL_FLAGS[@]}"
  --add-data "$SGLANG_SRC/sglang:sglang"
  --add-data "$SITE_PACKAGES/sgl_kernel:sgl_kernel"
  --hidden-import torch
  --hidden-import torchvision
  --hidden-import triton
  --hidden-import sglang
  --hidden-import sgl_kernel
  --hidden-import kuccl_pg
  --hidden-import kuccl_backend_pg
  --hidden-import pybase64
  --hidden-import zmq
  --hidden-import zmq.asyncio
  --hidden-import fastapi
  --hidden-import fastapi.middleware.cors
  --hidden-import starlette
  --hidden-import starlette.middleware.cors
  --hidden-import uvicorn
  --hidden-import setproctitle
  --hidden-import openai
  --hidden-import vllm
  --hidden-import vllm.logging_utils
  --hidden-import atomics
  --hidden-import distro
  --hidden-import partial_json_parser
  --hidden-import transformers
  --hidden-import transformers.models.ernie4_5
  --hidden-import transformers.models.ernie4_5_moe
  --hidden-import msgspec
  --collect-all vllm
  --collect-all torch
  --collect-binaries torch
  $SGLANG_SRC/sglang/launch_server.py
)

# ===================== Pre-check all add-binary/add-data sources =====================
# Report every missing source path up front instead of letting pyinstaller fail
# on the first one. Run with "check" (or CHECK_ONLY=1) to only validate.
precheck_sources() {
    echo "[precheck] resolved paths:"
    local v
    for v in SGLANG_PATH SGLANG_SRC SITE_PACKAGES CONDA_ENV_PATH HPCKIT_PATH \
             BISHENG_LIB KUPL_LIB KUTACC_LIB KUCCL_PATH HUCX_DIR XUCG_DIR; do
        printf '  %-16s = %s\n' "$v" "${!v}"
    done
    echo

    local expect=0 arg src dest m total=0 missing=0
    for arg in "$@"; do
        if [[ "$expect" == "1" ]]; then
            expect=0
            src="${arg%%:*}"
            dest="${arg#*:}"
            total=$((total + 1))
            for m in $src; do
                if [[ -e "$m" ]]; then
                    printf '  [OK]      %s  ->  %s\n' "$m" "$dest"
                else
                    printf '  [MISSING] %s  ->  %s\n' "$m" "$dest"
                    missing=$((missing + 1))
                fi
            done
            continue
        fi
        case "$arg" in
            --add-binary|--add-data) expect=1 ;;
        esac
    done

    if [[ ! -e "$SGLANG_SRC/sglang/launch_server.py" ]]; then
        printf '  [MISSING] %s\n' "$SGLANG_SRC/sglang/launch_server.py"
        missing=$((missing + 1))
    fi

    if [[ "${SGLANG_ENABLE_KUCCL:-0}" == "1" ]]; then
        echo
        echo "[precheck] kuccl/sdma location probes:"
        for p in "$KUCCL_PATH/install" "$KUCCL_PATH/install/hucx/lib" \
                 "$KUCCL_PATH/hucx/lib" "$HUCX_DIR/lib" "$HUCX_DIR/lib/ucx" \
                 "$XUCG_DIR/lib" "$XUCG_DIR/lib/planc"; do
            if [[ -e "$p" ]]; then printf '  [DIR ] %s\n' "$p"; else printf '  [----] %s\n' "$p"; fi
        done
        echo "[precheck] find libsdma_dk.so / libuct_sdma.so under KUCCL_PATH + HPCKIT_PATH + /usr/lib64:"
        find "$KUCCL_PATH" "$HPCKIT_PATH" /usr/lib64 \
             \( -name 'libsdma_dk.so' -o -name 'libuct_sdma.so' \) 2>/dev/null | head -20
    fi

    echo
    printf '[precheck] entries checked: %d, missing: %d\n' "$total" "$missing"
    return "$missing"
}

_MISSING=0
precheck_sources "${PYI_FLAGS[@]}" || _MISSING=$?

if [[ "${1:-}" == "check" || "${CHECK_ONLY:-0}" == "1" ]]; then
    echo "[precheck] check-only mode, exit before spec/build."
    exit 0
fi
if [[ "$_MISSING" -ne 0 ]]; then
    echo "[precheck] ERROR: $_MISSING source path(s) missing, abort before pyinstaller." >&2
    exit 1
fi

# ===================== PyInstaller spec =====================
echo "[pyinstall] generate spec file..."
rm -f sglang_server.spec
pyi-makespec "${PYI_FLAGS[@]}"

# ===================== Auto-modify spec file =====================
echo "[pyinstall] modify spec file, move sglang/sgl_kernel/kuccl_pg out of PYZ..."
python - <<EOF
with open('sglang_server.spec', 'r') as f:
    content = f.read()

filter_code = "a.pure = [m for m in a.pure if not m[0].startswith('sglang') and not m[0].startswith('sgl_kernel') and not m[0].startswith('kuccl_pg')]\n"
if filter_code in content:
    print("Spec already contains filter code, no need to modify")
else:
    # Match pyz = PYZ(a.pure) and replace
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

# ===================== Build with modified spec =====================
rm -rf $PYINSTALL_PATH/dist
echo "[pyinstall] start build..."
pyinstaller sglang_server.spec --distpath ./dist --workpath ./build --noconfirm

# echo "[pyinstall] numa duplication ..."
bash numa_duplication.sh

echo "=================================================================="
echo "complete! output dir: dist/sglang_server"
echo "=================================================================="