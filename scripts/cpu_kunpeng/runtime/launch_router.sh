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
# launch_router.sh - Launch router node (single-node, gateway + tokenizer HTTP).
# Invoked by: ./launch.sh router

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/env.sh" router

# Wait for ALL gateway backends to be ready (up to 20 minutes):
# always the prefill/decode masters, plus the tokenizer HTTP servers
# (30001/30002) in tokenizer-separate mode (tokenizer starts in parallel
# with the compute servers, so its ports may not be up yet either).
endpoints=(
    "prefill|${PREFILL_MASTER_ADDR}:30000"
    "decode|${DECODE_MASTER_ADDR}:30000"
)
if [[ "$SGLANG_ENABLE_TOKENIZER_SEPERATE" == "1" ]]; then
    endpoints+=(
        "tokenizer-prefill|${ROUTER_IP}:30001"
        "tokenizer-decode|${ROUTER_IP}:30002"
    )
fi
echo "[$(date +%T)] Waiting for backends to be ready (up to 20 minutes):"
for _ep in "${endpoints[@]}"; do echo "  ${_ep%%|*}  ${_ep#*|}"; done
ready=()
for _j in "${endpoints[@]}"; do ready+=(0); done
for i in $(seq 1 12000); do
    for j in "${!endpoints[@]}"; do
        _role="${endpoints[$j]%%|*}"
        _addr="${endpoints[$j]#*|}"
        if [[ "${ready[$j]}" -eq 0 ]] &&
            curl -sf --noproxy "*" --max-time 2 "http://${_addr}/health" >/dev/null 2>&1; then
            ready[$j]=1
            echo "[$(date +%T)] ${_role} ${_addr} ready"
        fi
    done
    ready_all=1
    for v in "${ready[@]}"; do [[ "$v" -eq 1 ]] || { ready_all=0; break; }; done
    [[ "$ready_all" -eq 1 ]] && break
    sleep 0.1
done
for j in "${!endpoints[@]}"; do
    if [[ "${ready[$j]}" -eq 0 ]]; then
        echo "ERROR: HTTP server at ${endpoints[$j]%%|*} (${endpoints[$j]#*|}) failed to start within 20 minutes"
        exit 1
    fi
done
echo "[$(date +%T)] ===== Prefill and decode servers ready ====="

mkdir -p "$LOG_DIR"
# Refresh "latest" symlink to this run's time dir (e.g. latest -> 214700)
ln -sfn "$LOG_TIME" "$LOG_BASE_DIR/$LOG_DATE/router/latest"
sh "$SCRIPT_DIR/stop.sh" router

IFS=' ' read -ra NODES <<< "$NODE_IPS_LIST"
WORLD_SIZE=${#NODES[@]}

echo "Launching router on $WORLD_SIZE node(s)"

for i in "${!NODES[@]}"; do
    node_ip="${NODES[i]}"
    node_rank="$i"
    echo "[$(date +%T)] Starting node_rank $node_rank ($node_ip)"
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
        "root@$node_ip" \
        "cd \"$PWD\" && sh ./server.sh router \"$node_rank\" \"$LOG_DIR\"" \
        >"$LOG_DIR/ssh_router_rank${node_rank}.log" 2>&1 &
done

echo "All router nodes launched."

rank0_log_file="$LOG_DIR/router_${NODES[0]}.log"
echo "Log file of rank_0: $rank0_log_file"

if [[ "${SKIP_LOG:-0}" == "1" ]]; then
    exit 0
fi

while [ ! -f "$rank0_log_file" ]; do
    sleep 1
done
tail -f "$rank0_log_file"
