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
# launch_store.sh - Launch the mooncake store service on MOONCAKE_STORE_NODE.
# Holds the shared L3 segment while the SGLang nodes contribute nothing.
# Invoked by: ./launch.sh store (and by ./launch.sh all after the master).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

SKIP_CONDA=1 source "$SCRIPT_DIR/env.sh" store || exit 1

sh "$SCRIPT_DIR/stop.sh" store

mkdir -p "$LOG_DIR"
ln -sfn "$LOG_TIME" "$LOG_BASE_DIR/$LOG_DATE/store/latest"
mkdir -p "$LOG_BASE_DIR/latest"
ln -sfn "$LOG_BASE_DIR/$LOG_DATE/store/$LOG_TIME" "$LOG_BASE_DIR/latest/store"

echo "[$(date +%T)] Launching mooncake store service on $MOONCAKE_STORE_NODE:$MOONCAKE_STORE_PORT (segment $MOONCAKE_GLOBAL_SEGMENT_SIZE, master $MOONCAKE_MASTER)"

ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
    "root@$MOONCAKE_STORE_NODE" \
    "cd \"$PWD\" && mkdir -p \"$LOG_DIR\" && \
     nohup sh ./runtime/server_store.sh \"$LOG_DIR\" \
       </dev/null >\"$LOG_DIR/store_service.log\" 2>&1 &" \
    >"$LOG_DIR/ssh_store.log" 2>&1 &

echo "[$(date +%T)] Waiting for store service on port $MOONCAKE_STORE_PORT ..."
_ready=0
for _ in $(seq 1 60); do
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
        "root@$MOONCAKE_STORE_NODE" \
        "ss -lnt | grep -q ':$MOONCAKE_STORE_PORT '"; then
        _ready=1
        break
    fi
    sleep 1
done
if [[ "$_ready" == "1" ]]; then
    echo "[$(date +%T)] Mooncake store service ready on $MOONCAKE_STORE_NODE:$MOONCAKE_STORE_PORT"
else
    echo "WARNING: mooncake store service did not listen on port $MOONCAKE_STORE_PORT within 60s;" >&2
    echo "         check $LOG_DIR/store_service.log on $MOONCAKE_STORE_NODE" >&2
fi