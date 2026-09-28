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
# launch_mooncake.sh - Launch the mooncake store master on MOONCAKE_MASTER_NODE.
# Backs the HiCache L3 tier when KUNPENG_HICACHE_BACKEND=mooncake; must be up
# before the prefill servers register their L2 host pools with it.
# Invoked by: ./launch.sh mooncake (and by ./launch.sh all before prefill).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# No conda needed: mooncake_master is a native binary, so keep this control-side
# shell out of the python env (env_mooncake.sh still validates the binary path).
SKIP_CONDA=1 source "$SCRIPT_DIR/env.sh" mooncake || exit 1

sh "$SCRIPT_DIR/stop.sh" mooncake

mkdir -p "$LOG_DIR"
# Refresh "latest" symlink to this run's time dir (e.g. latest -> 214700)
ln -sfn "$LOG_TIME" "$LOG_BASE_DIR/$LOG_DATE/mooncake/latest"
mkdir -p "$LOG_BASE_DIR/latest"
ln -sfn "$LOG_BASE_DIR/$LOG_DATE/mooncake/$LOG_TIME" "$LOG_BASE_DIR/latest/mooncake"

echo "[$(date +%T)] Launching mooncake master on $MOONCAKE_MASTER_NODE ($MOONCAKE_MASTER)"
ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
    "root@$MOONCAKE_MASTER_NODE" \
    "cd \"$PWD\" && mkdir -p \"$LOG_DIR\" && \
     nohup \"$MOONCAKE_MASTER_BIN\" \
       --enable_http_metadata_server=true \
       --http_metadata_server_port=$MOONCAKE_METADATA_PORT \
       --eviction_high_watermark_ratio=$MOONCAKE_EVICTION_WATERMARK \
       </dev/null >\"$LOG_DIR/mooncake_master.log\" 2>&1 &" \
    >"$LOG_DIR/ssh_mooncake.log" 2>&1 &

# Wait for the master to listen (up to 60s). A slow master only warns: the
# prefill servers retry their own registration.
echo "[$(date +%T)] Waiting for $MOONCAKE_MASTER ..."
_ready=0
for _ in $(seq 1 60); do
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
        "root@$MOONCAKE_MASTER_NODE" \
        "ss -lnt | grep -q ':$MOONCAKE_MASTER_PORT '"; then
        _ready=1
        break
    fi
    sleep 1
done
if [[ "$_ready" == "1" ]]; then
    echo "[$(date +%T)] Mooncake master ready on $MOONCAKE_MASTER"
else
    echo "WARNING: mooncake master did not listen on $MOONCAKE_MASTER within 60s;" >&2
    echo "         check $LOG_DIR/mooncake_master.log on $MOONCAKE_MASTER_NODE" >&2
fi
