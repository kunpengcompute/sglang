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
# Usage: ./launch.sh [prefill|decode|native]  (default: native)


if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [prefill|decode|native]" >&2
    exit 1
fi

ROLE="${1:-native}"
if [[ "$ROLE" != "prefill" && "$ROLE" != "decode" && "$ROLE" != "native" ]]; then
    echo "Error: ROLE must be 'prefill' or 'decode' or 'native'" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source config for the specified role
# Exports NODE_IPS_LIST, CONDA_ACTIVATE_CMD, WORLD_SIZE, etc.
source ./env.sh "$ROLE"

mkdir -p "$LOG_DIR"

sh stop.sh "$ROLE"

# Convert space-separated IP list to array
IFS=' ' read -ra NODES <<< "$NODE_IPS_LIST"
WORLD_SIZE=${#NODES[@]}

if [[ "$SGLANG_ENABLE_NUMA_DUPLICATION" == "1" ]]; then
    echo "Update binary sglang..."
    sh ./pyinstall/updata.sh
fi

echo "Launching $ROLE on $WORLD_SIZE node(s)"

for i in "${!NODES[@]}"; do
    node_ip="${NODES[i]}"
    dp_rank="$i"
    echo "[$(date +%T)] Starting dp_rank $dp_rank on $node_ip"
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
        "root@$node_ip" \
        "cd \"$PWD\" && sh ./server.sh \"$ROLE\" \"$dp_rank\" \"$LOG_DIR\"" &
done

echo "All $ROLE nodes launched."
echo "Logs: $LOG_DIR"

if [[ "$SGLANG_ENABLE_BINARY_LAUNCH" == "1" ]]; then
    log_file="$LOG_DIR/0_0_${NODES[0]}.log"
else
    log_file="$LOG_DIR/0_${NODES[0]}.log"
fi

while [ ! -f "$log_file" ]; do
    sleep 1
done
tail -f "$log_file"
