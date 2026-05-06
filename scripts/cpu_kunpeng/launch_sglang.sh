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
# Usage: ./launch_sglang.sh [prefill|decode]

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 [prefill|decode]" >&2
    exit 1
fi

ROLE="$1"
if [[ "$ROLE" != "prefill" && "$ROLE" != "decode" ]]; then
    echo "Error: ROLE must be 'prefill' or 'decode'" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source config for the specified role
# Exports NODE_IPS_LIST, CONDA_ACTIVATE_CMD, WORLD_SIZE, etc.
source env.sh "$ROLE"

mkdir -p "$LOG_DIR"

# Convert space-separated IP list to array
IFS=' ' read -ra NODES <<< "$NODE_IPS_LIST"
WORLD_SIZE=${#NODES[@]}

echo "Launching $ROLE on $WORLD_SIZE node(s)"

for i in "${!NODES[@]}"; do
    node_ip="${NODES[i]}"
    rank="$i"
    echo "[$(date +%T)] Starting rank $rank on $node_ip"
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
        "root@$node_ip" \
        "cd \"$PWD\" && bash ./run.sh \"$ROLE\" \"$rank\"" > "$LOG_DIR/${rank}_${node_ip}.log" 2>&1 &
done

echo "All $ROLE nodes launched. Logs: $LOG_DIR"