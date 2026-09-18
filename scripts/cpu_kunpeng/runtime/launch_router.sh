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
# launch_router.sh - Launch the gateway (sgl-model-gateway) on the router node.
# Waits for all backends, then SSHes to ROUTER_IP to run runtime/server_router.sh.
# Invoked by: ./launch.sh router

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# No conda needed here: this control-side script only curls/ssh's; the
# gateway binary itself is launched by runtime/server_router.sh (Rust, no
# python env). SKIP_CONDA=1 skips the activation on this node.
SKIP_CONDA=1 source "$SCRIPT_DIR/env.sh" router

sh "$SCRIPT_DIR/stop.sh" router

# Wait for ALL gateway backends to be ready (up to 20 minutes): one master
# per INSTANCES entry (e.g. "prefill,decode_128p" -> prefill master +
# decode 128p master; each entry's role env is sourced in a subshell to
# read its real master address), plus each entry's tokenizer HTTP server
# (per-instance <side>_TOK_PORT) in tokenizer-separate mode (tokenizer
# starts in parallel with the compute servers, so its ports may not be up
# yet either).
endpoints=()
for _entry in ${INSTANCES//,/ }; do
    _entry="${_entry//[[:space:]]/}"
    [[ -z "$_entry" ]] && continue
    _role="${_entry%%_*}"
    _inst=""
    [[ "$_entry" == *_* ]] && _inst="${_entry#*_}"
    case "$_role" in
        prefill)
            _addr="$(SKIP_CONDA=1 bash -c "
                source '$SCRIPT_DIR/env.sh' prefill '$_inst' >/dev/null 2>&1
                echo \"\$PREFILL_MASTER_ADDR\"")"
            endpoints+=("prefill${_inst:+-$_inst}|${_addr}:30000")
            if [[ "$SGLANG_ENABLE_TOKENIZER_SEPERATE" == "1" ]]; then
                _tok_port="$(SKIP_CONDA=1 bash -c "
                    unset PREFILL_TOK_PORT PREFILL_BOOTSTRAP_PORT PREFILL_NUMA_BASE
                    source '$SCRIPT_DIR/env.sh' prefill '$_inst' >/dev/null 2>&1
                    echo \"\${PREFILL_TOK_PORT:-30001}\"")"
                endpoints+=("tokenizer-prefill${_inst:+-$_inst}|${ROUTER_IP}:${_tok_port}")
            fi
            ;;
        decode)
            _addr="$(SKIP_CONDA=1 bash -c "
                source '$SCRIPT_DIR/env.sh' decode '$_inst' >/dev/null 2>&1
                echo \"\$DECODE_MASTER_ADDR\"")"
            endpoints+=("decode${_inst:+-$_inst}|${_addr}:30000")
            if [[ "$SGLANG_ENABLE_TOKENIZER_SEPERATE" == "1" ]]; then
                _tok_port="$(SKIP_CONDA=1 bash -c "
                    unset DECODE_TOK_PORT DECODE_BOOTSTRAP_PORT DECODE_NUMA_BASE
                    source '$SCRIPT_DIR/env.sh' decode '$_inst' >/dev/null 2>&1
                    echo \"\${DECODE_TOK_PORT:-30002}\"")"
                endpoints+=("tokenizer-decode${_inst:+-$_inst}|${ROUTER_IP}:${_tok_port}")
            fi
            ;;
    esac
done
# Fallback: INSTANCES empty/unparsed -> resolve one backend per side via
# the default (no-instance) role env in subshells.
if [[ ${#endpoints[@]} -eq 0 ]]; then
    _pf_addr="$(SKIP_CONDA=1 bash -c "
        source '$SCRIPT_DIR/env.sh' prefill >/dev/null 2>&1
        echo \"\$PREFILL_MASTER_ADDR\"")"
    _de_addr="$(SKIP_CONDA=1 bash -c "
        source '$SCRIPT_DIR/env.sh' decode >/dev/null 2>&1
        echo \"\$DECODE_MASTER_ADDR\"")"
    endpoints=(
        "prefill|${_pf_addr}:30000"
        "decode|${_de_addr}:30000"
    )
    if [[ "$SGLANG_ENABLE_TOKENIZER_SEPERATE" == "1" ]]; then
        endpoints+=(
            "tokenizer-prefill|${ROUTER_IP}:30001"
            "tokenizer-decode|${ROUTER_IP}:30002"
        )
    fi
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

# Refresh unified "$LOG_BASE_DIR/latest/router" -> this run's time dir
mkdir -p "$LOG_BASE_DIR/latest"
ln -sfn "$LOG_BASE_DIR/$LOG_DATE/router/$LOG_TIME" "$LOG_BASE_DIR/latest/router"


echo "[$(date +%T)] Launching gateway on $ROUTER_IP"
ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
    "root@$ROUTER_IP" \
    "cd \"$PWD\" && bash runtime/server_router.sh \"$LOG_DIR\"" \
    >"$LOG_DIR/ssh_router.log" 2>&1 &

echo "Gateway launched."

rank0_log_file="$LOG_DIR/router_${ROUTER_IP}.log"
echo "Log file of rank_0: $rank0_log_file"

if [[ "${SKIP_LOG:-0}" == "1" ]]; then
    exit 0
fi

while [ ! -f "$rank0_log_file" ]; do
    sleep 1
done
tail -f "$rank0_log_file"
