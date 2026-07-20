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

show_usage() {
    echo "Usage: $0 [prefill|decode|native|router|all] [--no-log]" >&2
    echo "  prefill  - Launch prefill server (PD disaggregation, prefill side)" >&2
    echo "  decode   - Launch decode server (PD disaggregation, decode side)" >&2
    echo "  native   - Launch without PD disaggregation" >&2
    echo "  router   - Launch router server (route requests to prefill/decode)" >&2
    echo "  all      - Launch prefill, decode, and router sequentially" >&2
    echo "  --no-log - Do not tail logs (exit after launching)" >&2
}

# Parse args: ROLE (positional) + optional --no-log flag
ROLE=""
SHOW_LOG=1
for arg in "$@"; do
    case "$arg" in
        --no-log)
            SHOW_LOG=0
            ;;
        prefill|decode|native|router|all)
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
VALID_ROLES=("prefill" "decode" "native" "router" "all")
if [[ ! " ${VALID_ROLES[*]} " =~ " ${ROLE} " ]]; then
    echo "Error: Invalid role '$ROLE'. Must be one of: ${VALID_ROLES[*]}" >&2
    show_usage
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# all mode: launch prefill, decode, and router via background launch.sh calls
if [[ "$ROLE" == "all" ]]; then
    echo "[$(date +%T)] ===== Launching all roles (prefill + decode + router) in background ====="

    bash ./stop.sh router
    bash ./launch.sh prefill --no-log
    bash ./launch.sh decode --no-log

    source ./env.sh native
    # Wait for prefill and decode HTTP servers to be ready (up to 20 minutes total)
    endpoints=("${PREFILL_MASTER_ADDR}:30000" "${DECODE_MASTER_ADDR}:30000")
    echo "[$(date +%T)] Waiting for prefill and decode servers to be ready (up to 20 minutes)..."
    for i in $(seq 1 600); do
        curl -sf --max-time 2 "http://${endpoints[0]}/health" >/dev/null 2>&1 &&
            curl -sf --max-time 2 "http://${endpoints[1]}/health" >/dev/null 2>&1 && break
        sleep 2
    done
    for ep in "${endpoints[@]}"; do
        curl -sf --max-time 2 "http://$ep/health" >/dev/null 2>&1 || {
            echo "ERROR: HTTP server at $ep failed to start within 30 minutes"
            exit 1
        }
    done
    echo "[$(date +%T)] ===== Prefill and decode servers launched (running in background) ====="

    bash ./launch.sh router

    exit 0
fi

# Source config for the specified role
# Exports NODE_IPS_LIST, CONDA_ACTIVATE_CMD, WORLD_SIZE, etc.
source ./env.sh "$ROLE"

mkdir -p "$LOG_DIR"

sh stop.sh "$ROLE"

# Convert space-separated IP list to array
IFS=' ' read -ra NODES <<< "$NODE_IPS_LIST"
WORLD_SIZE=${#NODES[@]}

if [[ "$SGLANG_ENABLE_NUMA_DUPLICATION" == "1" && "$ROLE" != "router" ]]; then
    echo "Update binary sglang..."
    bash ./pyinstall/updata.sh
fi

echo "Launching $ROLE on $WORLD_SIZE node(s)"

for i in "${!NODES[@]}"; do
    node_ip="${NODES[i]}"
    dp_rank="$i"
    echo "[$(date +%T)] Starting dp_rank $dp_rank on $node_ip"
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
        "root@$node_ip" \
        "cd \"$PWD\" && sh ./server.sh \"$ROLE\" \"$dp_rank\" \"$LOG_DIR\"" \
        >"$LOG_DIR/ssh_${ROLE}_rank${dp_rank}.log" 2>&1 &
done

echo "All $ROLE nodes launched."

if [[ "$ROLE" == "router" ]]; then
    rank0_log_file="$LOG_DIR/router_${NODES[0]}.log"
elif [[ "$SGLANG_ENABLE_BINARY_LAUNCH" == "1" ]]; then
    rank0_log_file="$LOG_DIR/0_0_${NODES[0]}.log"
else
    rank0_log_file="$LOG_DIR/0_${NODES[0]}.log"
fi

echo "Log file of rank_0: $rank0_log_file"

# Skip tail -f if --no-log was given
if [[ "$SHOW_LOG" -eq 0 ]]; then
    exit 0
fi

while [ ! -f "$rank0_log_file" ]; do
    sleep 1
done
tail -f "$rank0_log_file"
