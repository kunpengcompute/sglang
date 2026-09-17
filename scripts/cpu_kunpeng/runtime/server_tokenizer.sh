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
# server_tokenizer.sh - Single-node tokenizer HTTP server launcher.
# Runs ON the router node. Invoked by runtime/launch_tokenizer.sh via SSH.
# Starts the HTTP server for the given side (prefill: port 30001 / decode:
# port 30002), then polls until it is ready.
# Usage: bash runtime/server_tokenizer.sh <prefill|decode> <log_path> [instance]
#   instance (e.g. "128p") selects the per-instance env file so the
#   tokenizer points at that instance's cluster (master addr, DP size).

if [[ $# -lt 2 || $# -gt 3 || ( "$1" != "prefill" && "$1" != "decode" ) ]]; then
    echo "Usage: $0 <prefill|decode> <log_path> [instance]" >&2
    exit 1
fi

TOK_SIDE="$1"
LOG_PATH="$2"
TOK_INSTANCE="${3:-}"

# env.sh's 2nd arg for tokenizer is "<side>[_<instance>]": only that
# side's role env is loaded (with the instance file if given).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh" tokenizer "${TOK_SIDE}${TOK_INSTANCE:+_$TOK_INSTANCE}"

# ================= Router-node CPU / NUMA binding plan =================
# each NUMA = 38 cores, the last core is isolated, workers take
# 18-core halves). 16 NUMAs total (0-15). Each role (tokenizer-separate
# HTTP server) takes a contiguous 4-NUMA block:
#   base   : parent [0..17]      + bootstrap server [18..35]
#   base+1 : worker 0 [0..17]    + worker 1 [18..35]
#   base+2 : worker 2 [0..17]    + worker 3 [18..35]
#   base+3 : detokenizer (whole NUMA)
# i.e. prefill = NUMA 0-3, second prefill = NUMA 4-7, decode = NUMA 8-11;
# the decode role runs no bootstrap server (prefill-only), so its base
# upper half stays spare. NUMA 15 is left for the gateway.
export PREFILL_NUMA_BASE="${PREFILL_NUMA_BASE:-0}"
export DECODE_NUMA_BASE="${DECODE_NUMA_BASE:-8}"
export PREFILL_BOOTSTRAP_CPU="${PREFILL_BOOTSTRAP_CPU:-$((PREFILL_NUMA_BASE * 38 + 18))-$((PREFILL_NUMA_BASE * 38 + 35))}"

# Common args for tokenizer-side HTTP server
HTTP_COMMON_ARGS=(
    --model "$MODEL_PATH"
    --device cpu --trust-remote-code
    --host "$ROUTER_IP"
    --disaggregation-bootstrap-port 9001
    --nnodes 1 --node-rank 0 --dist-timeout 600
    --tp-size 1
    --max-total-tokens 64
    --tokenizer-worker-num "$TOKENIZER_WORKER_NUM"
    --skip-server-warmup
    --enable-dynamic-batch-tokenizer
    --batch-notify-size "$SGLANG_KUNPENG_MAX_SEQ_NUM"
    --tokenizer-backend "${SGLANG_TOKENIZER_BACKEND:-huggingface}"
)

# Launch the HTTP server for the requested side (tokenizer side)
if [[ "$TOK_SIDE" == "prefill" ]]; then
    TOK_PORT=30001
    TOK_DP_SIZE="$PREFILL_DP_SIZE"
    TOK_DIST_ADDR="$PREFILL_MASTER_ADDR:$PREFILL_MASTER_PORT"
    TOK_NUMA_BASE="$PREFILL_NUMA_BASE"
    export SGLANG_KUNPENG_BOOTSTRAP_SERVER_CPU="$PREFILL_BOOTSTRAP_CPU"
else
    TOK_PORT=30002
    TOK_DP_SIZE="$DECODE_DP_SIZE"
    TOK_DIST_ADDR="$DECODE_MASTER_ADDR:$DECODE_MASTER_PORT"
    TOK_NUMA_BASE="$DECODE_NUMA_BASE"
fi

echo "Launching $TOK_SIDE HTTP server..."
SGLANG_KUNPENG_TOKENIZER_BASE_NUMA="$TOK_NUMA_BASE" \
LD_PRELOAD="$LIBPTHREAD_HOOK_PATH" \
python -m sglang.launch_server \
    "${HTTP_COMMON_ARGS[@]}" \
    --dp-size "$TOK_DP_SIZE" \
    --port "$TOK_PORT" \
    --dist-init-addr "$TOK_DIST_ADDR" \
    --disaggregation-mode "$TOK_SIDE" \
> "$LOG_PATH/tokenizer_${TOK_SIDE}_http.log" 2>&1 &

# Poll until the HTTP server is ready (up to 30 minutes)
echo "Waiting for HTTP server on port $TOK_PORT to be ready..."
ready=0
for i in $(seq 1 18000); do  # up to 30 minutes
    if curl -sf --noproxy "*" --max-time 2 "http://${ROUTER_IP}:${TOK_PORT}/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 0.1
done
[[ "$ready" -eq 1 ]] || {
    echo "ERROR: HTTP server on port $TOK_PORT not ready within 30 minutes" >&2
    exit 1
}
echo "HTTP server on port $TOK_PORT ready"
