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

# Usage: spy.sh [decode|prefill|all]   (default: decode)
#
# Sources env.sh with <role> and SKIP_CONDA=1 to obtain the correct
# SGLANG_PATH / CONDA_ENV_PATH / NODE_IPS from the cluster config.
# Requires bash (env.sh uses bash syntax); re-exec under bash if it
# was invoked as "sh spy.sh".

if [ -z "$BASH_VERSION" ]; then
    exec /bin/bash "$0" "$@"
fi

ROLE="${1:-decode}"
case "$ROLE" in
    decode|prefill|all) ;;
    *)
        echo "ERROR: unknown role '$ROLE'. Usage: spy.sh [decode|prefill|all] (default decode)" >&2
        ROLE=decode
        ;;
esac

# Locate env.sh from this script's own directory (same cpu_kunpeng dir).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SH="$SCRIPT_DIR/env.sh"
if [[ ! -f "$ENV_SH" ]]; then
    echo "ERROR: env.sh not found at $ENV_SH" >&2
    exit 1
fi

# ------------------------------------------------------------------
# Source env.sh for the required role(s) with SKIP_CONDA=1. For "all" we
# need both decode and prefill node sets, so source twice (save the
# NODE_IPS array each time before it gets overwritten).
# ------------------------------------------------------------------
DECODE_IPS=()
PREFILL_IPS=()
case "$ROLE" in
    decode|all)
        SKIP_CONDA=1 source "$ENV_SH" decode
        DECODE_IPS=("${NODE_IPS[@]}")
        ;;
esac
case "$ROLE" in
    prefill|all)
        SKIP_CONDA=1 source "$ENV_SH" prefill
        PREFILL_IPS=("${NODE_IPS[@]}")
        ;;
esac

if [[ "$ROLE" == "all" ]]; then
    declare -A __seen
    NODE_IPS=()
    for ip in "${DECODE_IPS[@]}" "${PREFILL_IPS[@]}"; do
        [[ -n "${__seen[$ip]:-}" ]] && continue
        __seen[$ip]=1
        NODE_IPS+=("$ip")
    done
elif [[ "$ROLE" == "decode" ]]; then
    NODE_IPS=("${DECODE_IPS[@]}")
else
    NODE_IPS=("${PREFILL_IPS[@]}")
fi

echo "ROLE          = $ROLE"
echo "SGLANG_PATH   = $SGLANG_PATH"
echo "CONDA_ENV_PATH= $CONDA_ENV_PATH"
echo "NODE_IPS (${#NODE_IPS[@]}) = ${NODE_IPS[*]}"
PYSPY="$CONDA_ENV_PATH/bin/py-spy"

# ------------------------------------------------------------
# Dump (each run goes into a timestamped subfolder; keep history)
# ------------------------------------------------------------
DUMP_ROOT="$SGLANG_PATH/logs/dump_file"
mkdir -p "$DUMP_ROOT"
STAMP="$(date +%m%d_%H%M)"
DUMP_DIR="$DUMP_ROOT/$STAMP"
mkdir -p "$DUMP_DIR"
echo "Dump files -> $DUMP_DIR"

for ip in "${NODE_IPS[@]}"; do
    {
        echo "Processing $ip ..."
        ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@"$ip" "
            ps -eo pid,args | grep 'sglang::scheduler_' | grep -v grep | awk '{print \$1}' | \
            while read pid; do
                $PYSPY dump --native -p \"\$pid\"
            done
        " > "$DUMP_DIR/$ip" 2>&1
    } &
done

wait
echo "All tasks completed."

# ========== summary generation ==========
SUMMARY="$DUMP_DIR/summary_$(date +%Y%m%d_%H%M%S).txt"
> "$SUMMARY"

for ip in "${NODE_IPS[@]}"; do
    file="$DUMP_DIR/$ip"
    echo "=== Node $ip ===" >> "$SUMMARY"
    if [ -s "$file" ]; then
        awk '
        BEGIN {
            RS = "Process "; ORS = "";
            # sort DP groups and TP ranks numerically (not lexicographic)
            PROCINFO["sorted_in"] = "@ind_num_asc";
        }
        # Keep only the main-thread stack: in py-spy output the main thread is
        # the one whose thread id equals the process pid, i.e. "Thread <pid>".
        function filter_main_thread(rec, pid,   i, start, n, lines, out, m) {
            n = split(rec, lines, "\n");
            out = lines[1];                       # process header "Process <pid>: <cmd>"
            start = 0;
            for (i = 1; i <= n; i++) {
                if (lines[i] ~ /^Thread [0-9]+/ &&
                    match(lines[i], /^Thread ([0-9]+)/, m) && m[1] == pid) {
                    start = i;                    # the main-thread header
                    break;
                }
            }
            if (start == 0) {
                return out;                       # no main thread found -> header only
            }
            for (i = start; i <= n; i++) {
                if (i > start && lines[i] ~ /^Thread [0-9]+/) break;
                out = out "\n" lines[i];
            }
            return out;
        }
        NR > 1 {
            rec = "Process " $0;
            # DP / TP ranks parsed from the process title, e.g. "sglang::scheduler_DP0_TP1".
            # Within each node we want stacks ordered by DP then TP.
            if (match($0, /sglang::scheduler_DP([0-9]+)/, a)) dp = a[1] + 0; else dp = 999;
            if (match($0, /_TP([0-9]+)/, t)) tp = t[1] + 0; else tp = 0;

            match(rec, /^Process ([0-9]+):/, hdr);
            pid = hdr[1];
            block = filter_main_thread(rec, pid);
            if ((dp in data) && (tp in data[dp])) {
                data[dp][tp] = data[dp][tp] "\n" block;
            } else {
                data[dp][tp] = block;
            }
        }
        END {
            for (dp in data) {
                for (tp in data[dp]) {
                    printf "%s\n", data[dp][tp];
                }
            }
        }
        ' "$file" >> "$SUMMARY"
    else
        echo "  (empty or error)" >> "$SUMMARY"
    fi
done

echo "Summary created: $SUMMARY"