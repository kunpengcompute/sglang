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
UPDATE_KUTACC=false

case "$1" in
    sglang)
        UPDATE_SGLANG=true
        ;;
    kernel)
        UPDATE_KERNEL=true
        ;;
    kutacc)
        UPDATE_KUTACC=true
        ;;
    ""|all)
        UPDATE_SGLANG=true
        UPDATE_KERNEL=true
        UPDATE_KUTACC=true
        ;;
    *)
        echo "Usage: $0 [sglang|kernel|kutacc|all] (default: all)"
        exit 1
        ;;
esac

MARKER_FILE="$PYINSTALL_PATH/dist/.updated_marker"

mkdir -p "$PYINSTALL_PATH/dist"

# Skip if no source files are newer than the marker
if [ -f "$MARKER_FILE" ]; then
    need_update=false
    if [ "${UPDATE_SGLANG}" = "true" ]; then
        if [ -n "$(find "$SGLANG_PATH/python/sglang" -newer "$MARKER_FILE" -print -quit 2>/dev/null)" ]; then
            need_update=true
        fi
    fi
    if [ "${UPDATE_KERNEL}" = "true" ]; then
        if [ -n "$(find "$SITE_PACKAGES/sgl_kernel" -newer "$MARKER_FILE" -print -quit 2>/dev/null)" ]; then
            need_update=true
        fi
    fi
    if [ "${UPDATE_KUTACC}" = "true" ]; then
        if [ "$KUTACC_PATH/install/lib/libkutacc.so.25.1.RC1" -nt "$MARKER_FILE" ]; then
            need_update=true
        fi
    fi

    if [ "$need_update" = "false" ]; then
        echo "[updata] All sources unchanged since last update, skipping."
        exit 0
    fi
fi

echo "[updata] Updating NUMA copies..."
for i in $(seq 0 15); do
    (
        if [ "${UPDATE_SGLANG}" = "true" ]; then
            rm -rf "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/sglang"
            cp -rf "$SGLANG_PATH/python/sglang" "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/sglang"
        fi
        if [ "${UPDATE_KERNEL}" = "true" ]; then
            rm -rf "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/sgl_kernel"
            cp -rf "$SITE_PACKAGES/sgl_kernel" "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/sgl_kernel"
        fi
        if [ "${UPDATE_KUTACC}" = "true" ]; then
            rm -f "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/libkutacc.so.25.1.RC1"
            cp -f "$KUTACC_PATH/install/lib/libkutacc.so.25.1.RC1" "$PYINSTALL_PATH/dist/sglang_server_tp$i/_internal/"
        fi
    ) &
done
wait

touch "$MARKER_FILE"
echo "[updata] Update complete."