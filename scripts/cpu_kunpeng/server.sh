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
# server.sh - Single node execution for SGLang (prefill or decode).
# Usage: ./server.sh <role> <dp_rank> <log_path>

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <role> <dp_rank> <log_path>" >&2
    exit 1
fi

ROLE="$1"
DP_RANK="$2"
LOG_PATH="$3"
IP="$(ifconfig enp26s0f0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}')"

# Source environment config (exports CONDA_ACTIVATE_CMD, PYTHON_SCRIPT, etc.)
source ./env.sh "$ROLE"

# Activate conda
eval "$CONDA_ACTIVATE_CMD"

# Base arguments common to both roles
BASE_ARGS=(
    --model "$MODEL_PATH"
    --device cpu
    --trust-remote-code
    --attention-backend kunpeng_cpu
    --moe-a2a-backend kunpeng_cpu
    --host "$IP"
    --dist-init-addr "$MASTER_ADDR:$MASTER_PORT"
    --nnodes "$WORLD_SIZE"
    --node-rank "$DP_RANK"
    --dist-timeout 600
    --enable-dp-attention
    --dp-size "$WORLD_SIZE"
    --tp-size "$TP_SIZE"
    --ep-size "$EP_SIZE"
    --page-size 64
    --mem-fraction-static 0.88
    --chunked-prefill-size -1
    --skip-server-warmup
    --disable-custom-all-reduce
    --disable-radix-cache
    --disable-overlap-schedule
    --enable-dp-lm-head
    --enable-dp-mlp
    --quantization w8a8_int8
    ${LOAD_FORMAT:+--load-format "$LOAD_FORMAT"}
)

# Role-specific arguments
case "$ROLE" in
    prefill)
        SPECIFIC_ARGS=(
            --disaggregation-mode prefill
            --disaggregation-bootstrap-port 9001
            --max-prefill-tokens 4096
            --max-total-tokens 18496
            --prefill-max-requests 4
            --load-balance-method follow_bootstrap_room
            --enable-dynamic-batch-tokenizer
        )
        ;;
    decode)
        SPECIFIC_ARGS=(
            --disaggregation-mode decode
            --max-total-tokens 139328
            --load-balance-method follow_bootstrap_room
            --decode-log-interval 1
            --num-reserved-decode-tokens 256
        )
        ;;
    native)
        SPECIFIC_ARGS=(
            --disaggregation-bootstrap-port 9001
            --max-prefill-tokens 4096
            --max-total-tokens 18496
            --load-balance-method round_robin
        )
        ;;
    *)
        echo "Error: unknown role '$ROLE'" >&2
        exit 1
        ;;
esac

# Combine and execute
if [[ "$SGLANG_ENABLE_BINARY_LAUNCH" == "1" ]]; then
    echo "Update binary sglang..."
    sh ./pyinstall/updata.sh
    echo "Launch binary server..."
    for ((ATTN_TP_RANK=0; ATTN_TP_RANK < (TP_SIZE / WORLD_SIZE); ATTN_TP_RANK++)); do
        taskset -c $((ATTN_TP_RANK * 38)) \
        $PYINSTALL_PATH/dist/sglang_server_tp${ATTN_TP_RANK}/sglang_server "${BASE_ARGS[@]}" "${SPECIFIC_ARGS[@]}" \
          --tp-rank-in-node ${ATTN_TP_RANK} \
          --port $((30000 + ATTN_TP_RANK)) \
          > "${LOG_PATH}/${DP_RANK}_${ATTN_TP_RANK}_$IP.log" 2>&1 &
    done
elif [[ "$SGLANG_ENABLE_BINARY_LAUNCH" == "2" ]]; then
    for ((ATTN_TP_RANK=0; ATTN_TP_RANK < (TP_SIZE / WORLD_SIZE); ATTN_TP_RANK++)); do
        taskset -c $((ATTN_TP_RANK * 38)) \
        python -m sglang.launch_server "${BASE_ARGS[@]}" "${SPECIFIC_ARGS[@]}" \
          --tp-rank-in-node ${ATTN_TP_RANK} \
          --port $((30000 + ATTN_TP_RANK)) \
          > "${LOG_PATH}/${DP_RANK}_${ATTN_TP_RANK}_$IP.log" 2>&1 &
    done
else
    python -m sglang.launch_server "${BASE_ARGS[@]}" "${SPECIFIC_ARGS[@]}" \
      --port 30000 \
      > "$LOG_PATH/${DP_RANK}_$IP.log" 2>&1
fi


