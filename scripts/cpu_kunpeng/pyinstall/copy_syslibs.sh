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

            if [ -d "$TARGET/kuccl" ]; then
                rm -rf "$TARGET/kuccl"
            fi

            mkdir -p "$TARGET/kuccl/install"

            SOURCE_HPCKIT_DIR="${HPCKIT_PATH}/26.1.RC1/hmpi/bisheng/release"

            if [ -d "$SOURCE_HPCKIT_DIR" ]; then
                cp -a "$SOURCE_HPCKIT_DIR"/. "$TARGET/kuccl/install/"
                echo "[copy_syslibs] KUCCL libs copied from HPCKIT to tp$i"
            else
                echo "[copy_syslibs] WARNING: HPCKIT source directory $SOURCE_HPCKIT_DIR does not exist, skipping KUCCL copy for tp$i"
            fi
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
