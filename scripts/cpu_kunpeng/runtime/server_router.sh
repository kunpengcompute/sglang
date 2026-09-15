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
# Backend URLs per INSTANCES entry. In tokenizer-separate mode each side
# has a single fixed tokenizer HTTP port (30001/30002) that fronts the
# side's schedulers. Otherwise every instance's master :30000 is a
# backend directly (one --prefill/--decode pair per entry; the gateway
# accepts repeated flags to build its worker set).
for _entry in ${INSTANCES//,/ }; do
    _entry="${_entry//[[:space:]]/}"
    [[ -z "$_entry" ]] && continue
    _role="${_entry%%_*}"
    _inst=""
    [[ "$_entry" == *_* ]] && _inst="${_entry#*_}"
    case "$_role" in
        prefill)
            if [[ "$SGLANG_ENABLE_TOKENIZER_SEPERATE" == "1" ]]; then
                SPECIFIC_ARGS+=(--prefill "http://${ROUTER_IP}:30001" 9001)
            else
                _addr="$(SKIP_CONDA=1 bash -c "
                    source '$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh' prefill '$_inst' >/dev/null 2>&1
                    echo \"\$PREFILL_MASTER_ADDR\"")"
                SPECIFIC_ARGS+=(--prefill "http://${_addr}:30000" 9001)
            fi
            ;;
        decode)
            if [[ "$SGLANG_ENABLE_TOKENIZER_SEPERATE" == "1" ]]; then
                SPECIFIC_ARGS+=(--decode "http://${ROUTER_IP}:30002")
            else
                _addr="$(SKIP_CONDA=1 bash -c "
                    source '$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh' decode '$_inst' >/dev/null 2>&1
                    echo \"\$DECODE_MASTER_ADDR\"")"
                SPECIFIC_ARGS+=(--decode "http://${_addr}:30000")
            fi
            ;;
    esac
done
# Fallback: INSTANCES empty/unparsed -> single backend per side from the
# current role env (the 15 fixed args above were not extended).
if [[ ${#SPECIFIC_ARGS[@]} -eq 15 ]]; then
    if [[ "$SGLANG_ENABLE_TOKENIZER_SEPERATE" == "1" ]]; then
        _router_prefill_url="http://${ROUTER_IP}:30001"
        _router_decode_url="http://${ROUTER_IP}:30002"
    else
        _router_prefill_url="http://${PREFILL_MASTER_ADDR}:30000"
        _router_decode_url="http://${DECODE_MASTER_ADDR}:30000"
    fi
    SPECIFIC_ARGS+=(--prefill "$_router_prefill_url" 9001 --decode "$_router_decode_url")
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
LD_PRELOAD="$LIBPTHREAD_HOOK_PATH" \
taskset -c "$GATEWAY_CPUS" "$GATEWAY_BIN" "${SPECIFIC_ARGS[@]}" \
    > "$LOG_PATH/router_$IP.log" 2>&1 &

echo "Gateway launched, pid=$!"
