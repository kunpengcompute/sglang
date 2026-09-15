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
# launch_tokenizer.sh - Launch tokenizer HTTP servers (prefill:30001 + decode:30002)
# on the router node. Invoked by: ./launch.sh tokenizer

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/env.sh" tokenizer

# Note: safe to start in parallel with the prefill/decode servers. The
# http-only process never runs torch.distributed init (dist-init-addr is
# unused), its ZMQ PUSH to the remote scheduler lazily queues until the
# compute side binds, and the bootstrap server (port 9001) only listens
# for inbound registrations.

mkdir -p "$LOG_DIR"
# Refresh "latest" symlink to this run's time dir (e.g. latest -> 214700)
ln -sfn "$LOG_TIME" "$LOG_BASE_DIR/$LOG_DATE/tokenizer/latest"
sh "$SCRIPT_DIR/stop.sh" tokenizer

IFS=' ' read -ra NODES <<< "$NODE_IPS_LIST"
WORLD_SIZE=${#NODES[@]}

echo "Launching tokenizer on $WORLD_SIZE node(s)"

for i in "${!NODES[@]}"; do
    node_ip="${NODES[i]}"
    node_rank="$i"
    echo "[$(date +%T)] Starting node_rank $node_rank ($node_ip)"
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
        "root@$node_ip" \
        "cd \"$PWD\" && sh ./server.sh tokenizer \"$node_rank\" \"$LOG_DIR\"" \
        >"$LOG_DIR/ssh_tokenizer_rank${node_rank}.log" 2>&1 &
done

echo "All tokenizer nodes launched. The remote server.sh polls until ports 30001/30002 are ready."

if [[ "${SKIP_LOG:-0}" == "1" ]]; then
    exit 0
fi

rank0_log_file="$LOG_DIR/tokenizer_prefill_http.log"
echo "Log file of rank_0: $rank0_log_file"

while [ ! -f "$rank0_log_file" ]; do
    sleep 1
done
tail -f "$rank0_log_file"
