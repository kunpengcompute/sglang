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
# stop_server.sh - Kill sglang on the role's cluster nodes (all of them,
# in parallel via SSH), then clean up zombies / shm / hugepages.
# Invoked by: ./stop.sh server [prefill|decode|native] [instance]
# Usage: bash runtime/stop_server.sh [prefill|decode|native] [instance]

ROLE="${1:-native}"
INSTANCE="${2:-}"
if [[ "$ROLE" != "prefill" && "$ROLE" != "decode" && "$ROLE" != "native" ]]; then
    echo "Usage: $0 [prefill|decode|native] [instance]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Source config for the specified role (+ optional instance)
# Exports NODE_IPS_LIST, CONDA_ACTIVATE_CMD, WORLD_SIZE, etc.
SKIP_CONDA=1 source ./env.sh "$ROLE" "$INSTANCE"

# Convert space-separated IP list to array
IFS=' ' read -ra NODES <<< "$NODE_IPS_LIST"
WORLD_SIZE=${#NODES[@]}

echo "Killing $ROLE on $WORLD_SIZE node(s)"

for i in "${!NODES[@]}"; do
    node="${NODES[i]}"

    ssh "$node" '
        MAIN_PIDS=$(ps aux | grep sglang | grep -v grep | awk "{print \$2}")

        if [ -n "$MAIN_PIDS" ]; then
            echo "Found SGLang processes on '"$node"'"

            for pid in $MAIN_PIDS; do
                kill -15 $pid 2>/dev/null
            done

            sleep 15

            REMAINING=$(ps aux | grep sglang | grep -v grep | awk "{print \$2}")
            if [ -n "$REMAINING" ]; then
                echo "Processes still running on '"$node"'. Sending SIGKILL..."
                for pid in $REMAINING; do
                    kill -9 $pid 2>/dev/null
                done
            else
                echo "All processes on '"$node"' terminated gracefully."
            fi
        else
            echo "No SGLang processes found on '"$node"'"
        fi

        ZOMBIES=$(ps aux | awk '\''$8 ~ /^Z/ && $11 ~ /sglang/ {print $2}'\'')
        if [ -n "$ZOMBIES" ]; then
            echo "Found zombie processes on '$node'"
            for zpid in $ZOMBIES; do
                parent_pid=$(ps -o ppid= -p $zpid 2>/dev/null | xargs)
                if [ -n "$parent_pid" ] && [ "$parent_pid" != "1" ]; then
                    kill -9 $parent_pid 2>/dev/null
                fi
            done
        fi

        rm -rf /dev/shm/shm_mmap_*
        for i in $(seq 0 31); do
            echo 0 > /sys/devices/system/node/node${i}/hugepages/hugepages-2048kB/nr_hugepages
        done

        # Drop caches if configured
        if [ "'"${DROP_CACHES:-0}"'" = "1" ]; then
            echo "Dropping caches on '"$node"'..."
            echo 3 > /proc/sys/vm/drop_caches
        fi
    ' &
done

wait
echo "All $ROLE nodes processed."
