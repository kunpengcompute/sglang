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
# launch_cluster.sh - Cluster launch for prefill/decode/native roles.
# Invoked by: ./launch.sh <prefill|decode|native>
# Or directly: bash runtime/launch_cluster.sh <prefill|decode|native>

show_usage() {
    echo "Usage: $0 [prefill|decode|native]" >&2
    echo "  prefill  - Launch prefill server (PD disaggregation, prefill side)" >&2
    echo "  decode   - Launch decode server (PD disaggregation, decode side)" >&2
    echo "  native   - Launch without PD disaggregation" >&2
}

# Parse args: ROLE (positional)
# Optional: SKIP_LOG=1 before running to skip tail -f (exit after launching)
ROLE=""
for arg in "$@"; do
    case "$arg" in
        prefill|decode|native)
            ROLE="$arg"
            ;;
        *)
            echo "Error: Unknown argument '$arg'" >&2
            show_usage
            exit 1
            ;;
    esac
done

# Default role if none given
ROLE="${ROLE:-native}"

# Re-validate role
VALID_ROLES=("prefill" "decode" "native")
if [[ ! " ${VALID_ROLES[*]} " =~ " ${ROLE} " ]]; then
    echo "Error: Invalid role '$ROLE'. Must be one of: ${VALID_ROLES[*]}" >&2
    show_usage
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Source config for the specified role
# Exports NODE_IPS_LIST, CONDA_ACTIVATE_CMD, WORLD_SIZE, etc.
source "$SCRIPT_DIR/env.sh" "$ROLE"

mkdir -p "$LOG_DIR"
# Refresh "latest" symlink to this run's time dir (e.g. latest -> 214700)
ln -sfn "$LOG_TIME" "$LOG_BASE_DIR/$LOG_DATE/$ROLE/latest"

sh "$SCRIPT_DIR/stop.sh" "$ROLE"

# Convert space-separated IP list to array
IFS=' ' read -ra NODES <<< "$NODE_IPS_LIST"
WORLD_SIZE=${#NODES[@]}

echo "Launching $ROLE on $WORLD_SIZE node(s)"

for i in "${!NODES[@]}"; do
    node_ip="${NODES[i]}"
    node_rank="$i"
    echo "[$(date +%T)] Starting node_rank $node_rank ($node_ip)"
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
        "root@$node_ip" \
        "cd \"$PWD\" && sh ./server.sh \"$ROLE\" \"$node_rank\" \"$LOG_DIR\"" \
        >"$LOG_DIR/ssh_${ROLE}_rank${node_rank}.log" 2>&1 &
done

echo "All $ROLE nodes launched."

if [[ "$SGLANG_ENABLE_BINARY_LAUNCH" == "1" ]]; then
    # Binary launch names each log by its true rank (see server.sh):
    # pp{pp}_dp{dp}_tp{tp}_{ip}.log. Rank 0 = node 0, rank-in-node 0 = pp0/dp0/tp0.
    rank0_log_file="$LOG_DIR/pp0_dp0_tp0_${NODES[0]}.log"
else
    rank0_log_file="$LOG_DIR/0_${NODES[0]}.log"
fi

echo "Log file of rank_0: $rank0_log_file"

# Skip tail -f if SKIP_LOG=1 was set
if [[ "${SKIP_LOG:-0}" == "1" ]]; then
    exit 0
fi

while [ ! -f "$rank0_log_file" ]; do
    sleep 1
done
tail -f "$rank0_log_file"
