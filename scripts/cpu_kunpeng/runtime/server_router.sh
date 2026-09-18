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
# server_router.sh - Single-node gateway (sgl-model-gateway) launcher.
# Runs ON the router node. Invoked by runtime/launch_router.sh via SSH.
# Usage: bash runtime/server_router.sh <log_path>

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <log_path>" >&2
    exit 1
fi

LOG_PATH="$1"
IP="$(ifconfig enp26s0f0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}')"

# No conda activation needed: the gateway is a Rust binary; build its path
# directly from the conda env. SKIP_CONDA=1 skips the (slow) activation.
SKIP_CONDA=1 source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh" router

# Record this node's invocation and full environment for debugging
{
    echo "[$(date +%T)] invocation: $0 $*"
    echo "---- environment ----"
    env | sort
} > "$LOG_PATH/env_router.log"

export GATEWAY_CPUS="${GATEWAY_CPUS:-570-590}"
SPECIFIC_ARGS=(
    --model-path "$MODEL_PATH"
    --pd-disaggregation
    --policy cache_aware
    --health-check-interval-secs 10000
    --queue-timeout-secs 10000
    --request-timeout-secs 10000
    --health-check-timeout-secs 10000
    --host "$IP"
)
# Backend URLs per INSTANCES entry. In tokenizer-separate mode each entry
# fronts its side's tokenizer HTTP server on the router node (per-instance
# port <side>_TOK_PORT from the entry's env; prefill entries also carry
# that instance's bootstrap port <side>_BOOTSTRAP_PORT — the gateway hands
# it to the decode side per request). Otherwise every instance's master
# :30000 is a backend directly (one --prefill/--decode pair per entry; the
# gateway accepts repeated flags to build its worker set).
_ENV_SH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
for _entry in ${INSTANCES//,/ }; do
    _entry="${_entry//[[:space:]]/}"
    [[ -z "$_entry" ]] && continue
    _role="${_entry%%_*}"
    _inst=""
    [[ "$_entry" == *_* ]] && _inst="${_entry#*_}"
    case "$_role" in
        prefill)
            if [[ "$SGLANG_ENABLE_TOKENIZER_SEPERATE" == "1" ]]; then
                _ports="$(SKIP_CONDA=1 bash -c "
                    unset PREFILL_TOK_PORT PREFILL_BOOTSTRAP_PORT PREFILL_NUMA_BASE
                    source '$_ENV_SH' prefill '$_inst' >/dev/null 2>&1
                    echo \"\${PREFILL_TOK_PORT:-30001} \${PREFILL_BOOTSTRAP_PORT:-9001}\"")"
                SPECIFIC_ARGS+=(--prefill "http://${ROUTER_IP}:${_ports%% *}" "${_ports##* }")
            else
                _addr="$(SKIP_CONDA=1 bash -c "
                    source '$_ENV_SH' prefill '$_inst' >/dev/null 2>&1
                    echo \"\$PREFILL_MASTER_ADDR\"")"
                SPECIFIC_ARGS+=(--prefill "http://${_addr}:30000" 9001)
            fi
            ;;
        decode)
            if [[ "$SGLANG_ENABLE_TOKENIZER_SEPERATE" == "1" ]]; then
                _tok_port="$(SKIP_CONDA=1 bash -c "
                    unset DECODE_TOK_PORT DECODE_BOOTSTRAP_PORT DECODE_NUMA_BASE
                    source '$_ENV_SH' decode '$_inst' >/dev/null 2>&1
                    echo \"\${DECODE_TOK_PORT:-30002}\"")"
                SPECIFIC_ARGS+=(--decode "http://${ROUTER_IP}:${_tok_port}")
            else
                _addr="$(SKIP_CONDA=1 bash -c "
                    source '$_ENV_SH' decode '$_inst' >/dev/null 2>&1
                    echo \"\$DECODE_MASTER_ADDR\"")"
                SPECIFIC_ARGS+=(--decode "http://${_addr}:30000")
            fi
            ;;
    esac
done
# Fallback: INSTANCES empty/unparsed -> single backend per side resolved
# from the default (no-instance) role env in subshells.
if [[ ${#SPECIFIC_ARGS[@]} -eq 15 ]]; then
    if [[ "$SGLANG_ENABLE_TOKENIZER_SEPERATE" == "1" ]]; then
        _router_prefill_url="http://${ROUTER_IP}:30001"
        _router_decode_url="http://${ROUTER_IP}:30002"
    else
        _pf_addr="$(SKIP_CONDA=1 bash -c "
            source '$_ENV_SH' prefill >/dev/null 2>&1
            echo \"\$PREFILL_MASTER_ADDR\"")"
        _de_addr="$(SKIP_CONDA=1 bash -c "
            source '$_ENV_SH' decode >/dev/null 2>&1
            echo \"\$DECODE_MASTER_ADDR\"")"
        _router_prefill_url="http://${_pf_addr}:30000"
        _router_decode_url="http://${_de_addr}:30000"
    fi
    SPECIFIC_ARGS+=(--prefill "$_router_prefill_url" 9001 --decode "$_router_decode_url")
fi

# Optional prefill routing policy override (PD mode), e.g. bucket policy
# for multi-prefill-instance deployments (see env_base.sh). Applied after
# the backend loop so the fallback arg-count check above stays intact.
if [[ -n "${ROUTER_PREFILL_POLICY:-}" ]]; then
    SPECIFIC_ARGS+=(
        --prefill-policy "$ROUTER_PREFILL_POLICY"
        --balance-abs-threshold "${ROUTER_BALANCE_ABS_THRESHOLD:-64}"
        --balance-rel-threshold "${ROUTER_BALANCE_REL_THRESHOLD:-1.5}"
        --bucket-adjust-interval-secs "${ROUTER_BUCKET_ADJUST_INTERVAL_SECS:-5}"
    )
    # Grouped mode: short requests (< LENGTH_THRESHOLD chars) -> first
    # SHORT_COUNT prefill backends by ascending URL, long ones -> the rest.
    if [[ "${ROUTER_PREFILL_SHORT_COUNT:-0}" != "0" ]]; then
        SPECIFIC_ARGS+=(
            --prefill-short-count "$ROUTER_PREFILL_SHORT_COUNT"
            --prefill-length-threshold "${ROUTER_PREFILL_LENGTH_THRESHOLD:-4096}"
        )
    fi
fi

# Launch sgl-model-gateway (Rust router) ----
echo "Launching PD disaggregation router..."
# Prefer an explicit MODEL_GATEWAY_BIN (e.g. from .user_env.sh);
# otherwise resolve the binary directly inside the conda env
# (no activation required for a Rust binary).
GATEWAY_BIN="${MODEL_GATEWAY_BIN:-${CONDA_ENV_PATH}/bin/sgl-model-gateway}"
if [[ ! -x "$GATEWAY_BIN" ]]; then
    echo "ERROR: sgl-model-gateway not found at '$GATEWAY_BIN' (MODEL_GATEWAY_BIN=${MODEL_GATEWAY_BIN:-<unset>})" >&2
    exit 1
fi
# Pin the gateway to a dedicated core range so it does not collide with
# the tokenizer/detokenizer workers (see the NUMA plan in server.sh).
_router_log="$LOG_PATH/router_$IP.log"
echo "[$(date +%T)] command: LD_PRELOAD=$LIBPTHREAD_HOOK_PATH taskset -c $GATEWAY_CPUS $GATEWAY_BIN ${SPECIFIC_ARGS[*]}" > "$_router_log"
LD_PRELOAD="$LIBPTHREAD_HOOK_PATH" \
taskset -c "$GATEWAY_CPUS" "$GATEWAY_BIN" "${SPECIFIC_ARGS[@]}" \
    >> "$_router_log" 2>&1 &

echo "Gateway launched, pid=$!"
