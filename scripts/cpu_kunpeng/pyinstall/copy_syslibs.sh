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

# ------------------------------------------------------------
# Direct NEEDED dependencies (found via pyinstall _internal/ rpath)
# ------------------------------------------------------------
LIBS=(
    # Original system libs
    "/usr/lib64/libdl.so.2"
    "/usr/lib64/libpthread.so.0"
    "/usr/lib64/libc.so.6"
    "/usr/lib64/libutil.so.1"
    "/usr/lib64/libm.so.6"
    "/usr/lib64/librt.so.1"
    "/usr/lib64/libmemkind.so"
    "/usr/lib64/libhwloc.so.15"
    "/usr/lib64/libfribidi.so.0"
    "/usr/lib64/libresolv.so.2"
    "/usr/lib64/libcrypt.so.1"
    # kuccl system deps (needed by kuccl_backend_pg.so via rpath)
    "/usr/lib64/libnuma.so"
    "/usr/lib64/librdmacm.so.1"
)

DIST_DIR="$PYINSTALL_PATH/dist"

echo "[copy_syslibs] Copying system libraries to all tp directories..."

for i in $(seq 0 15); do
    (
        TARGET="$DIST_DIR/sglang_server_tp$i/_internal"
        if [ ! -d "$TARGET" ]; then
            echo "[copy_syslibs] WARNING: $TARGET does not exist, skipping tp$i"
            exit 0
        fi

        # Direct NEEDED dependencies -> _internal/
        for lib in "${LIBS[@]}"; do
            cp -f "$lib" "$TARGET/"
        done

        if [[ "${SGLANG_ENABLE_KUCCL:-0}" == "1" ]]; then
            HUCX_DIR="${HUCX_DIR:-$KUCCL_PATH/hucx}"
            XUCG_DIR="${XUCG_DIR:-$KUCCL_PATH/xucg}"

            # Copy UCX/UCG libs to kuccl/install/ to match kuccl_pg.py fallback path:
            #   _kuccl_dir = dirname(kuccl_pg.py) = _internal/kuccl/
            #   fallback HUCX_DIR = _kuccl_dir/install/hucx
            #   fallback XUCG_DIR = _kuccl_dir/install/xucg
            KUCCL_INSTALL="$TARGET/kuccl/install"

            # UCX direct deps + plugins -> kuccl/install/hucx/lib/
            mkdir -p "$KUCCL_INSTALL/hucx/lib/ucx"
            cp -f "$HUCX_DIR/lib/libucs.so"      "$KUCCL_INSTALL/hucx/lib/"
            cp -f "$HUCX_DIR/lib/libucm.so.0"    "$KUCCL_INSTALL/hucx/lib/"
            cp -f "$HUCX_DIR/lib/libucp.so"      "$KUCCL_INSTALL/hucx/lib/"
            cp -f "$HUCX_DIR/lib/libuct.so.0"    "$KUCCL_INSTALL/hucx/lib/"
            cp -f "$HUCX_DIR/lib/ucx/libuct_ib.so"     "$KUCCL_INSTALL/hucx/lib/ucx/"
            cp -f "$HUCX_DIR/lib/ucx/libuct_rdmacm.so" "$KUCCL_INSTALL/hucx/lib/ucx/"
            cp -f "$HUCX_DIR/lib/ucx/libuct_cma.so"    "$KUCCL_INSTALL/hucx/lib/ucx/"
            cp -f "$HUCX_DIR/lib/ucx/libuct_sdma.so"   "$KUCCL_INSTALL/hucx/lib/ucx/"

            # UCG direct dep + plugins -> kuccl/install/xucg/lib/
            mkdir -p "$KUCCL_INSTALL/xucg/lib/planc"
            cp -f "$XUCG_DIR/lib/libucg.so" "$KUCCL_INSTALL/xucg/lib/"
            cp -f "$XUCG_DIR/lib/planc/libucg_planc_ucx.so"        "$KUCCL_INSTALL/xucg/lib/planc/"
            cp -f "$XUCG_DIR/lib/planc/libucg_planc_stars.so"      "$KUCCL_INSTALL/xucg/lib/planc/"
            cp -f "$XUCG_DIR/lib/planc/libucg_planm_ucx_hicoll.so" "$KUCCL_INSTALL/xucg/lib/planc/"

            # libsdma_dk.so: direct dep of libuct_sdma.so plugin -> _internal/
            if [ -f "$HUCX_DIR/lib/libsdma_dk.so" ]; then
                cp -f "$HUCX_DIR/lib/libsdma_dk.so" "$TARGET/"
            fi
        fi

        echo "[copy_syslibs] tp$i done"
    ) &
done
wait

echo "[copy_syslibs] All done."
