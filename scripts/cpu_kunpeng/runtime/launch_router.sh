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

mkdir -p "$LOG_DIR"

# Resolve a role's master address in its own env.sh subshell (the role /
# instance env may override the base value). The subshell's stderr is kept in
# <log>/router_env_<role>[_<inst>].log instead of being dropped: a broken or
# incomplete role env then fails here with a readable error, instead of
# surfacing 20 minutes later as ":30000 failed to start".
_resolve_master_addr() {
    local _role="$1" _inst="${2:-}" _var _log _addr
    case "$_role" in
        prefill) _var="PREFILL_MASTER_ADDR" ;;
        decode)  _var="DECODE_MASTER_ADDR" ;;
    esac
    _log="$LOG_DIR/router_env_${_role}${_inst:+_$_inst}.log"
    _addr="$(SKIP_CONDA=1 bash -c "
        source '$SCRIPT_DIR/env.sh' '$_role' '$_inst' >/dev/null
        echo \"\$$_var\"" 2>"$_log")"
    _addr="${_addr//[[:space:]]/}"
    if [[ -z "$_addr" ]]; then
        echo "ERROR: ${_role} master address not resolved (entry '${_role}${_inst:+_$_inst}', \$${_var} is empty); see $_log" >&2
        return 1
    fi
    echo "$_addr"
}

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
            _addr="$(_resolve_master_addr prefill "$_inst")" || exit 1
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
            _addr="$(_resolve_master_addr decode "$_inst")" || exit 1
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
    _pf_addr="$(_resolve_master_addr prefill '')" || exit 1
    _de_addr="$(_resolve_master_addr decode '')" || exit 1
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
probe_interval=()
probe_next_at=()
probe_code=()
for _j in "${!endpoints[@]}"; do
    ready+=("0"); probe_interval+=("1"); probe_next_at+=("0"); probe_code+=("")
done
_deadline=$(( $(date +%s) + 1200 ))
while (( $(date +%s) < _deadline )); do
    _now=$(date +%s)
    for j in "${!endpoints[@]}"; do
        [[ "${ready[$j]}" -eq 1 ]] && continue
        (( _now < ${probe_next_at[j]} )) && continue
        _role="${endpoints[$j]%%|*}"
        _addr="${endpoints[$j]#*|}"
        case "$_role" in
            tokenizer-*) _path="/v1/models" ;;
            *)           _path="/health" ;;
        esac
        probe_code[$j]="$(curl -s -o /dev/null -w '%{http_code}' --noproxy "*" --max-time 2 \
            "http://${_addr}${_path}" 2>/dev/null)"
        if [[ "${probe_code[$j]}" == "200" ]]; then
            ready[$j]=1
            echo "[$(date +%T)] ${_role} ${_addr} ready (${_path})"
        else
            probe_next_at[$j]=$(( _now + probe_interval[j] ))
            (( probe_interval[j] = probe_interval[j] < 10 ? probe_interval[j] * 2 : 10 ))
        fi
    done
    ready_all=1
    for v in "${ready[@]}"; do [[ "$v" -eq 1 ]] || { ready_all=0; break; }; done
    [[ "$ready_all" -eq 1 ]] && break
    sleep 0.5
done
for j in "${!endpoints[@]}"; do
    if [[ "${ready[$j]}" -eq 0 ]]; then
        echo "ERROR: HTTP server at ${endpoints[$j]%%|*} (${endpoints[$j]#*|}) failed to start within 20 minutes (last probe: ${probe_code[$j]:-no-response})"
        exit 1
    fi
done
echo "[$(date +%T)] ===== Prefill and decode servers ready ====="

# "<base>/latest/router" -> this run's router dir
mkdir -p "$LOG_DIR"
mkdir -p "$LOG_BASE_DIR/latest"
ln -sfn "$LOG_DIR" "$LOG_BASE_DIR/latest/router"


echo "[$(date +%T)] Launching gateway on $ROUTER_IP"
ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
    "root@$ROUTER_IP" \
    "cd \"$PWD\" && bash runtime/server_router.sh \"$LOG_DIR\"" \
    >"$LOG_DIR/ssh_router.log" 2>&1 &

echo "Gateway launched."

rank0_log_file="$LOG_DIR/router_${ROUTER_IP}.log"
echo "Log file of rank_0: $rank0_log_file"

# ---- End-to-end smoke check through the gateway -------------------------
# Gate on a REAL completion instead of the gateway's /readiness: in PD mode
# /readiness lags behind the true worker state (and the gateway's own health
# interval is 10000s), so a successful 1-token completion is the only signal
# that prefill -> KV transfer -> decode -> detokenizer all work.
# The model id is re-read from /v1/models on every attempt until it appears
# (it is proxied to the first prefill worker). SKIP_E2E_SMOKE=1 skips this.
if [[ "${SKIP_E2E_SMOKE:-0}" != "1" ]]; then
    _gw="http://${ROUTER_IP}:30000"
    _smoke_resp="$(mktemp -t sglang_smoke.XXXXXX 2>/dev/null || echo "/tmp/sglang_smoke.$$")"
    _model=""
    _attempt=0
    _smoke_deadline=$(( $(date +%s) + 1200 ))
    echo "[$(date +%T)] Waiting for gateway e2e (real completion every 1s, up to 20 minutes)"

    while :; do
        _attempt=$(( _attempt + 1 ))

        if [[ -z "$_model" ]]; then
            _model="$(curl -s --noproxy "*" --max-time 2 "$_gw/v1/models" 2>/dev/null \
                | grep -o '"id"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"\([^"]*\)"$/\1/')"
        fi

        if [[ -n "$_model" ]]; then
            _smoke_code="$(curl -s -o "$_smoke_resp" -w '%{http_code}' --noproxy "*" --max-time 15 \
                -H 'Content-Type: application/json' \
                -d "{\"model\":\"${_model}\",\"prompt\":\"1+1=\",\"max_tokens\":1,\"temperature\":0}" \
                "$_gw/v1/completions" 2>/dev/null)"
        else
            _smoke_code=""
        fi

        if [[ "$_smoke_code" == "200" ]]; then
            echo "[$(date +%T)] ===== Router started successfully: gateway -> prefill -> decode verified end-to-end (attempt=$_attempt) ====="
            rm -f "$_smoke_resp"
            break
        fi

        if (( $(date +%s) >= _smoke_deadline )); then
            echo "ERROR: end-to-end smoke not ready after 20 minutes (last http=${_smoke_code:-no-response}); deployment is not usable" >&2
            rm -f "$_smoke_resp"
            exit 1
        fi
        sleep 1
    done
fi

if [[ "${SKIP_LOG:-0}" == "1" ]]; then
    exit 0
fi

while [ ! -f "$rank0_log_file" ]; do
    sleep 1
done
# Replay the last lines (worker registration / tokenizer info) so the console
# shows the gateway is up, then follow new lines.
tail -n 5 -f "$rank0_log_file"
