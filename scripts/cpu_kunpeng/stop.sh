#!/bin/bash
# Usage: ./stop.sh [prefill|decode|native]


if [[ $# -ne 1 ]]; then
    echo "Usage: $0 [prefill|decode|native]" >&2
    exit 1
fi

ROLE="$1"
if [[ "$ROLE" != "prefill" && "$ROLE" != "decode" && "$ROLE" != "native" ]]; then
    echo "Error: ROLE must be 'prefill' or 'decode' or 'native'" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source config for the specified role
# Exports NODE_IPS_LIST, CONDA_ACTIVATE_CMD, WORLD_SIZE, etc.
source ./env.sh "$ROLE"

# Convert space-separated IP list to array
IFS=' ' read -ra NODES <<< "$NODE_IPS_LIST"
WORLD_SIZE=${#NODES[@]}

echo "Killing $ROLE on $WORLD_SIZE node(s)"

for i in "${!NODES[@]}"; do
    node="${NODES[i]}"

    ssh "$node" '
        MAIN_PIDS=$(ps aux | grep sglang | grep -v grep | awk "{print \$2}")

        if [ -n "$MAIN_PIDS" ]; then
            echo "Found SGLang processes on '"$node"': $MAIN_PIDS"

            for pid in $MAIN_PIDS; do
                echo "Sending SIGTERM to $pid on '"$node"'"
                kill -15 $pid 2>/dev/null
            done

            echo "Waiting 10 seconds for shutdown..."
            sleep 10

            REMAINING=$(ps aux | grep sglang | grep -v grep | awk "{print \$2}")
            if [ -n "$REMAINING" ]; then
                echo "Processes still running on '"$node"': $REMAINING"
                echo "Sending SIGKILL..."
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
            echo "Found zombie processes on '$node': $ZOMBIES"
            for zpid in $ZOMBIES; do
                parent_pid=$(ps -o ppid= -p $zpid 2>/dev/null | xargs)
                if [ -n "$parent_pid" ] && [ "$parent_pid" != "1" ]; then
                    echo "Killing parent process $parent_pid of zombie $zpid on '$node'"
                    kill -9 $parent_pid 2>/dev/null
                fi
            done
        fi
    ' &
done

wait
echo "All $ROLE nodes processed."
