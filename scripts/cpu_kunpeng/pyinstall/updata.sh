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
UPDATE_KUTACC=true
UPDATE_KUPL=true

case "$1" in
    sglang)
        UPDATE_SGLANG=true
        ;;
    kernel)
        UPDATE_KERNEL=true
        ;;
    ""|all)
        UPDATE_SGLANG=true
        UPDATE_KERNEL=true
        ;;
    *)
        echo "Usage: $0 [sglang|kernel|kutacc|kupl|all] (default: all)"
        exit 1
        ;;
esac

MARKER_FILE="$PYINSTALL_PATH/dist/.updated_marker"

mkdir -p "$PYINSTALL_PATH/dist"

# Returns 0 if update is needed, 1 otherwise.
check_if_update_needed() {
    if [ ! -f "$MARKER_FILE" ]; then
        return 0
    fi
    if [ "${UPDATE_SGLANG}" = "true" ]; then
        if [ -n "$(find "$SGLANG_PATH/python/sglang" -newer "$MARKER_FILE" -print -quit 2>/dev/null)" ]; then
            return 0
        fi
    fi
    if [ "${UPDATE_KERNEL}" = "true" ]; then
        if [ -n "$(find "$SITE_PACKAGES/sgl_kernel" -newer "$MARKER_FILE" -print -quit 2>/dev/null)" ]; then
            return 0
        fi
    fi
    if [ "${UPDATE_KUTACC}" = "true" ]; then
        if [ "$KUTACC_PATH/lib/libkutacc.so.25.1.RC1" -nt "$MARKER_FILE" ]; then
            return 0
        fi
    fi
    if [ "${UPDATE_KUPL}" = "true" ]; then
        if [ "$KUPL_PATH/lib/libkupl.so.1" -nt "$MARKER_FILE" ]; then
            return 0
        fi
    fi
    return 1
}

if check_if_update_needed; then :; else
    echo "[updata] All sources unchanged since last update, skipping."
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

echo "[updata] Updating NUMA copies..."
for i in $(seq 0 15); do
    (
        if [ "${UPDATE_SGLANG}" = "true" ]; then
            swap_dir "$SGLANG_PATH/python/sglang" "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/sglang"
        fi
        if [ "${UPDATE_KERNEL}" = "true" ]; then
            swap_dir "$SITE_PACKAGES/sgl_kernel" "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/sgl_kernel"
        fi
        if [ "${UPDATE_KUTACC}" = "true" ]; then
            swap_file "$KUTACC_PATH/lib/libkutacc.so.25.1.RC1" "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/libkutacc.so.25.1.RC1"
        fi
        if [ "${UPDATE_KUPL}" = "true" ]; then
            swap_file "$KUPL_PATH/lib/libkupl.so.1" "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/libkupl.so.1"
        fi
    ) &
done
wait

touch "$MARKER_FILE"
echo "[updata] Update complete."