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
SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")

UPDATE_SGLANG=false
UPDATE_KERNEL=false
UPDATE_TORCH=false
# KUTACC/KUPL .so: only 2 files, cheap to copy, force always
UPDATE_KUTACC=true
UPDATE_KUPL=true
# KUCCL .so: only update when SGLANG_ENABLE_KUCCL=1
if [ "${SGLANG_ENABLE_KUCCL:-0}" = "1" ]; then
    UPDATE_KUCCL=true
else
    UPDATE_KUCCL=false
fi

case "$1" in
    sglang)
        UPDATE_SGLANG=true
        ;;
    kernel)
        UPDATE_KERNEL=true
        ;;
    torch)
        UPDATE_TORCH=true
        ;;
    ""|all)
        UPDATE_SGLANG=true
        UPDATE_KERNEL=true
        UPDATE_TORCH=true
        ;;
    *)
        echo "Usage: $0 [sglang|kernel|torch|all] (default: all)"
        exit 1
        ;;
esac

MARKER_FILE="$PYINSTALL_PATH/dist/.updated_marker"

mkdir -p "$PYINSTALL_PATH/dist"

# Resolve kuccl backend .so and .py paths from KUCCL_PATH (only if enabled)
if [ "${UPDATE_KUCCL}" != "false" ]; then
    KUCCL_SO=$(ls "$KUCCL_PATH"/kuccl_backend_pg*.so 2>/dev/null | head -n1)
    KUCCL_PY="$KUCCL_PATH/kuccl_pg.py"
    if [ -z "$KUCCL_SO" ] || [ ! -f "$KUCCL_PY" ]; then
        echo "[update] WARNING: kuccl_backend_pg*.so or kuccl_pg.py not found in $KUCCL_PATH, kuccl update will be skipped"
        UPDATE_KUCCL=false
    fi
fi

# Returns 0 if update is needed, 1 otherwise. Sets the N*_DIRTY flags so the
# copy loop below only copies components that were actually changed.
NSGLANG=false
NKERNEL=false
NKUTACC=false
NKUPL=false
NKUCCL=false
NTORCH=false

check_if_update_needed() {
    if [ ! -f "$MARKER_FILE" ]; then
        # No marker yet: every selected component is dirty.
        NSGLANG="${UPDATE_SGLANG}"
        NKERNEL="${UPDATE_KERNEL}"
        NKUTACC="${UPDATE_KUTACC}"
        NKUPL="${UPDATE_KUPL}"
        NKUCCL="${UPDATE_KUCCL}"
        NTORCH="${UPDATE_TORCH}"
        return 0
    fi
    local needed=1
    if [ "${UPDATE_SGLANG}" = "true" ] && [ -n "$(find "$SGLANG_PATH/python/sglang" -newer "$MARKER_FILE" -print -quit 2>/dev/null)" ]; then
        NSGLANG=true; needed=0
    fi
    if [ "${UPDATE_KERNEL}" = "true" ] && [ -n "$(find "$SITE_PACKAGES/sgl_kernel" -newer "$MARKER_FILE" -print -quit 2>/dev/null)" ]; then
        NKERNEL=true; needed=0
    fi
    if [ "${UPDATE_KUTACC}" = "true" ]; then
        NKUTACC=true; needed=0
    fi
    if [ "${UPDATE_KUPL}" = "true" ]; then
        NKUPL=true; needed=0
    fi
    if [ "${UPDATE_KUCCL}" = "true" ] && [ -n "$KUCCL_SO" ]; then
        NKUCCL=true; needed=0
    fi
    if [ "${UPDATE_TORCH}" = "true" ] && [ -n "$(find "$SITE_PACKAGES/torch" -newer "$MARKER_FILE" -print -quit 2>/dev/null)" ]; then
        NTORCH=true; needed=0
    fi
    return $needed
}

if check_if_update_needed; then :; else
    echo "[update] No changed sources since last update, skipping."
    exit 0
fi

# Atomic directory swap: cp to temp, then rename old+new.
# Safe for concurrent execution (prefill & decode launching together).
# Each process uses unique temp/trash names via $$, so no conflicts.
swap_dir() {
    local src="$1" dest="$2"
    local tmp="${dest}.new.$$"
    local old="${dest}.old.$$"
    cp -rf "$src" "$tmp"
    mv "$dest" "$old" 2>/dev/null   # may fail if dest doesn't exist yet
    if mv "$tmp" "$dest" 2>/dev/null; then
        rm -rf "$old" 2>/dev/null &
    else
        # dest already replaced by concurrent process; clean up our temp
        rm -rf "$tmp" "$old" 2>/dev/null
    fi
}

swap_file() {
    local src="$1" dest="$2"
    cp -f "$src" "${dest}.tmp.$$"
    mv -f "${dest}.tmp.$$" "$dest"
}

echo "[update] Updating NUMA copies..."

# Resolve torch dist-info dir name once (e.g. torch-2.9.0.dist-info) and
# pre-clean stale dist-info dirs across all NUMA copies.
TORCH_DIST_INFO_NAME=""
if [ "${NTORCH}" = "true" ]; then
    TORCH_DIST_INFO_NAME=$(ls -d "$SITE_PACKAGES"/torch-*.dist-info 2>/dev/null | head -n1 | xargs -r basename)
    if [ -z "$TORCH_DIST_INFO_NAME" ]; then
        echo "[update] WARNING: no torch-*.dist-info found in $SITE_PACKAGES, torch update may be incomplete"
    fi
    echo "[update] Pre-cleaning stale torch-*.dist-info dirs..."
    for i in $(seq 0 15); do
        rm -rf "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal"/torch-*.dist-info 2>/dev/null
    done
fi
export TORCH_DIST_INFO_NAME

for i in $(seq 0 15); do
    (
        if [ "${NSGLANG}" = "true" ]; then
            swap_dir "$SGLANG_PATH/python/sglang" "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/sglang"
        fi
        if [ "${NKERNEL}" = "true" ]; then
            swap_dir "$SITE_PACKAGES/sgl_kernel" "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/sgl_kernel"
        fi
        if [ "${NKUTACC}" = "true" ]; then
            swap_file "$KUTACC_PATH/lib/libkutacc.so.25.1.RC1" "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/libkutacc.so.25.1.RC1"
        fi
        if [ "${NKUPL}" = "true" ]; then
            swap_file "$KUPL_PATH/lib/libkupl.so.1" "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/libkupl.so.1"
        fi
        if [ "${NKUCCL}" = "true" ]; then
            mkdir -p "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/kuccl"
            swap_file "$KUCCL_SO" "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/kuccl/$(basename "$KUCCL_SO")"
            swap_file "$KUCCL_PY" "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/kuccl/kuccl_pg.py"
        fi
        if [ "${NTORCH}" = "true" ]; then
            swap_dir "$SITE_PACKAGES/torch" "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/torch"
            if [ -n "$TORCH_DIST_INFO_NAME" ]; then
                swap_dir "$SITE_PACKAGES/$TORCH_DIST_INFO_NAME" \
                         "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/$TORCH_DIST_INFO_NAME"
            fi
        fi
    ) &
done
wait

touch "$MARKER_FILE"
echo "[update] Update complete."