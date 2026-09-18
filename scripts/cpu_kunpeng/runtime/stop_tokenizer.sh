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
# stop_tokenizer.sh - Kill the tokenizer HTTP server parent(s)
# (python -m sglang.launch_server ... --port <tok_port>) and all their
# child workers (sglang::detokenizer / sglang::tokenizer_worker) on the
# router node, so no orphan children survive a kill -9 fallback.
# Invoked by: ./stop.sh tokenizer [prefill|decode|all]
# Usage: bash runtime/stop_tokenizer.sh [prefill|decode|all]
#   Each argument may carry an instance suffix (e.g. decode_128p); "all"
#   resolves one port per INSTANCES entry (same parsing as launch.sh all).

TOK_ENTRY="${1:-all}"
# Accept "<side>[_<instance>]" (e.g. "decode_128p"): the instance selects
# the side's instance env when sourcing; the kill pattern matches the
# side's per-instance tokenizer HTTP port (<side>_TOK_PORT).
if [[ "$TOK_ENTRY" == *_* ]]; then
    TOK_SIDE="${TOK_ENTRY%%_*}"
else
    TOK_SIDE="$TOK_ENTRY"
fi
if [[ "$TOK_SIDE" != "prefill" && "$TOK_SIDE" != "decode" && "$TOK_SIDE" != "all" ]]; then
    echo "Usage: $0 [prefill|decode|all] (side may carry an instance suffix, e.g. decode_128p)" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Source config to get ROUTER_IP. Sides load only their own role env
# (env.sh tokenizer "<side>[_<instance>]"); "all" iterates the INSTANCES
# entries so every instance's tokenizer port is covered.
ports=()
if [[ "$TOK_SIDE" == "all" ]]; then
    SKIP_CONDA=1 source ./env.sh none
    IFS=',' read -ra _INST_LIST <<< "$INSTANCES"
    for _entry in "${_INST_LIST[@]}"; do
        _entry="${_entry//[[:space:]]/}"
        [[ -z "$_entry" ]] && continue
        _side="${_entry%%_*}"
        [[ "$_side" == "prefill" || "$_side" == "decode" ]] || continue
        # Unset the derived vars so each entry re-derives its own values
        # (sequential sourcing in this shell would otherwise keep the
        # first entry's ports).
        unset PREFILL_TOK_PORT PREFILL_BOOTSTRAP_PORT PREFILL_NUMA_BASE
        unset DECODE_TOK_PORT DECODE_BOOTSTRAP_PORT DECODE_NUMA_BASE
        SKIP_CONDA=1 source ./env.sh tokenizer "$_entry" >/dev/null 2>&1
        if [[ "$_side" == "prefill" ]]; then
            ports+=("${PREFILL_TOK_PORT:-30001}")
        else
            ports+=("${DECODE_TOK_PORT:-30002}")
        fi
    done
    # Fallback: INSTANCES empty/unparsed -> both default ports.
    [[ ${#ports[@]} -eq 0 ]] && ports=(30001 30002)
else
    SKIP_CONDA=1 source ./env.sh tokenizer "$TOK_ENTRY"
    if [[ "$TOK_SIDE" == "prefill" ]]; then
        ports+=("${PREFILL_TOK_PORT:-30001}")
    else
        ports+=("${DECODE_TOK_PORT:-30002}")
    fi
fi
# Dedupe (default + instance entries may resolve to the same port)
ports=($(printf '%s\n' "${ports[@]}" | sort -u))

TOK_PAT="sglang[.]launch_server.*--port ($(IFS='|'; echo "${ports[*]}"))"

echo "Killing tokenizer ($TOK_SIDE) HTTP server(s) on $ROUTER_IP (port(s): ${ports[*]})"
ssh "root@$ROUTER_IP" "
    MAIN_PIDS=\$(ps aux | grep -E '$TOK_PAT' | grep -v grep | awk '{print \$2}')
    echo \"Main tokenizer PID(s): \$MAIN_PIDS\"
    if [ -n \"\$MAIN_PIDS\" ]; then
        # Gather each parent PID together with all of its child workers
        ALL_PIDS=''
        for pid in \$MAIN_PIDS; do
            ALL_PIDS=\"\$ALL_PIDS \$pid \$(ps -o pid= --ppid \$pid 2>/dev/null)\"
        done
        echo \"Killing process(es):\$ALL_PIDS\"
        kill -15 \$ALL_PIDS 2>/dev/null
        sleep 5
        REMAINING=''
        for pid in \$ALL_PIDS; do
            kill -0 \$pid 2>/dev/null && REMAINING=\"\$REMAINING \$pid\"
        done
        if [ -n \"\$REMAINING\" ]; then
            echo \"Force killing:\$REMAINING\"
            kill -9 \$REMAINING 2>/dev/null
        fi
        echo 'Tokenizer stopped.'
    else
        echo 'No tokenizer process found.'
    fi
"
