#!/bin/bash
# Usage: ./launch.sh [prefill|decode|native]

# set -euo pipefail

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
source env.sh "$ROLE"

mkdir -p "$LOG_DIR"

bash stop.sh "$ROLE"

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
        "cd \"$PWD\" && bash ./server.sh \"$ROLE\" \"$rank\"" > "$LOG_DIR/${rank}_${node_ip}.log" 2>&1 &
done

echo "All $ROLE nodes launched."
echo "Logs: $LOG_DIR"

sleep 10

tail -f "$LOG_DIR/0_${NODES[0]}.log"
