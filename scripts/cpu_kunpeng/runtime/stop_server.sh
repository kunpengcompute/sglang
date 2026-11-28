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
# in parallel via SSH), then clean up zombies / shm / hugepages. All entries
# under /dev/shm are removed outright, so no stale segment name matching.
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

# Fail fast instead of hanging when a node is unreachable: bound the TCP
# connect and abort on a dead/unresponsive session (BatchMode also avoids
# hanging on an interactive password prompt).
SSH_OPTS=(
    -o BatchMode=yes
    -o ConnectTimeout=10
    -o ServerAliveInterval=5
    -o ServerAliveCountMax=3
)

pids=()
nodes=()
for i in "${!NODES[@]}"; do
    node="${NODES[i]}"

    ssh "${SSH_OPTS[@]}" "$(ssh_target "$node")" '
        PAT="sglang[.]launch_server|sglang[_]server|sglang[:]:"
        MAIN_PIDS=$(ps aux | grep -E "$PAT" | grep -v grep | awk "{print \$2}")

        if [ -n "$MAIN_PIDS" ]; then
            echo "Found SGLang processes on '"$node"'"

            for pid in $MAIN_PIDS; do
                kill -15 $pid 2>/dev/null
            done

            sleep 15

            REMAINING=$(ps aux | grep -E "$PAT" | grep -v grep | awk "{print \$2}")
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

        # Wipe the whole POSIX shm namespace instead of matching individual
        # segment names (shm_mmap_*, sgk_hicache_shard_*, ...). Under a non-root
        # deployment root-owned leftovers go through the site helper; otherwise
        # fall back to removing segments owned by the current user.
        if [ "'"${SGLANG_ENABLE_NOROOT_USER:-0}"'" = "1" ]; then
            "'"${SGLANG_KUNPENG_NOROOT_PROXY_PATH}"'/root_proxy" sgl_clean_shm.sh 2>/dev/null \
                || rm -rf /dev/shm/* 2>/dev/null || true
        else
            rm -rf /dev/shm/*
        fi

        for i in $(seq 0 31); do
            if [ "'"${SGLANG_ENABLE_NOROOT_USER:-0}"'" = "1" ]; then
                "'"${SGLANG_KUNPENG_NOROOT_PROXY_PATH}"'/root_proxy" hugepage_2m.sh 0 ${i}
            else
                echo 0 > /sys/devices/system/node/node${i}/hugepages/hugepages-2048kB/nr_hugepages
            fi
        done

        # Drop caches if configured
        if [ "'"${DROP_CACHES:-0}"'" = "1" ]; then
            echo "Dropping caches on '"$node"'..."
            if [ "'"${SGLANG_ENABLE_NOROOT_USER:-0}"'" = "1" ]; then
                "'"${SGLANG_KUNPENG_NOROOT_PROXY_PATH}"'/root_proxy" sgl_drop_cache.sh
            else
                echo 3 > /proc/sys/vm/drop_caches
            fi
        fi
    ' &
    pids+=("$!")
    nodes+=("$node")
done

failed_nodes=()
for i in "${!pids[@]}"; do
    wait "${pids[i]}"
    rc=$?
    # 255 = ssh itself failed (connect timeout, keepalive timeout, auth).
    if [[ $rc -eq 255 ]]; then
        failed_nodes+=("${nodes[i]}")
    fi
done

if [[ ${#failed_nodes[@]} -gt 0 ]]; then
    echo "ERROR: SSH connection failed or timed out on node(s): ${failed_nodes[*]}" >&2
    echo "ERROR: cannot stop $ROLE on ${#failed_nodes[@]} of $WORLD_SIZE node(s)" >&2
    exit 1
fi

echo "All $ROLE nodes processed."
