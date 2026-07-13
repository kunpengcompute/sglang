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
    --dp-size "$((WORLD_SIZE / PP_SIZE))"
    --tp-size "$TP_SIZE"
    --ep-size "$EP_SIZE"
    --pp-size "$PP_SIZE"
    --page-size 64
    --mem-fraction-static 0.88
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --skip-server-warmup
    --disable-custom-all-reduce
    --disable-radix-cache
    --disable-overlap-schedule
    --enable-dp-lm-head
    --enable-dp-mlp
    --quantization w8a8_int8
    --speculative-draft-model-path "$MODEL_PATH"
    --speculative-algorithm NEXTN
    --speculative-num-steps 1
    --speculative-eagle-topk 1
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
            --max-prefill-tokens 4096   # long prompt -> 131072
            --max-total-tokens 18496 # long prompt -> 131072
            # --context-length 131072 # long prompt
            --load-balance-method round_robin
        )
        ;;
    router)
        if [[ "$SGLANG_ENABLE_TOKENIZER_SEPERATE" == "1" ]]; then
            _router_prefill_url="http://${ROUTER_IP}:30001"
            _router_decode_url="http://${ROUTER_IP}:30002"
        else
            _router_prefill_url="http://${PREFILL_MASTER_ADDR}:30000"
            _router_decode_url="http://${DECODE_MASTER_ADDR}:30000"
        fi
        SPECIFIC_ARGS=(
            --model-path "$MODEL_PATH"
            --pd-disaggregation
            --prefill "$_router_prefill_url" 9001
            --decode "$_router_decode_url"
            --policy cache_aware
            --prefill-policy cache_aware
            --health-check-interval-secs 10000
            --queue-timeout-secs 10000
            --request-timeout-secs 10000
            --health-check-timeout-secs 10000
            --host "$IP"
        )
        ;;
    *)
        echo "Error: unknown role '$ROLE'" >&2
        exit 1
        ;;
esac

# Combine and execute
if [[ "$ROLE" == "router" ]]; then
    if [[ "$SGLANG_ENABLE_TOKENIZER_SEPERATE" == "1" ]]; then
        # 启动 prefill HTTP server（tokenizer 侧）
        echo "Launching prefill HTTP server..."
        HTTP_PREFILL_ARGS=(
            --model "$MODEL_PATH"
            --device cpu --trust-remote-code
            --host "$ROUTER_IP" --port 30001
            --dist-init-addr "$PREFILL_MASTER_ADDR:$PREFILL_MASTER_PORT"
            --disaggregation-mode prefill
            --disaggregation-bootstrap-port 9001
            --nnodes 1 --node-rank 0 --dist-timeout 600
            --enable-dp-attention --dp 1 --tp-size 1
            --max-total-tokens 64
            --skip-server-warmup
        )
        LD_PRELOAD="/path/to/libpthread_hook.so" \
        python -m sglang.launch_server "${HTTP_PREFILL_ARGS[@]}" \
            > "$LOG_PATH/router_prefill_http.log" 2>&1 &
        PREFILL_HTTP_PID=$!

        # 启动 decode HTTP server（tokenizer 侧）
        echo "Launching decode HTTP server..."
        HTTP_DECODE_ARGS=(
            --model "$MODEL_PATH"
            --device cpu --trust-remote-code
            --host "$ROUTER_IP" --port 30002
            --dist-init-addr "$DECODE_MASTER_ADDR:$DECODE_MASTER_PORT"
            --disaggregation-mode decode
            --disaggregation-bootstrap-port 9001
            --nnodes 1 --node-rank 0 --dist-timeout 600
            --enable-dp-attention --dp 1 --tp-size 1
            --max-total-tokens 64
            --skip-server-warmup
        )
        LD_PRELOAD="/path/to/libpthread_hook.so" \
        python -m sglang.launch_server "${HTTP_DECODE_ARGS[@]}" \
            > "$LOG_PATH/router_decode_http.log" 2>&1 &
        DECODE_HTTP_PID=$!

        # 轮询等待两个 HTTP server 就绪
        for port in 30001 30002; do
            echo "Waiting for HTTP server on port $port..."
            _ready=0
            for i in $(seq 1 60); do
                if ss -tlnp 2>/dev/null | grep -q ":$port "; then
                    echo "HTTP server on port $port ready"
                    _ready=1
                    break
                fi
                sleep 2
            done
            if [[ "$_ready" -eq 0 ]]; then
                echo "ERROR: HTTP server on port $port failed to start within 120 seconds"
                exit 1
            fi
        done
    fi
    # 启动 sglang_router ----
    echo "Launching PD disaggregation router..."
        python -m sglang_router.launch_router "${SPECIFIC_ARGS[@]}" \
        > "$LOG_PATH/router_$IP.log" 2>&1 &

    exit 0
fi

# Build IB device args based on role.
IB_DEVICE_ALL="roceroh0,roceroh1,roceroh2,roceroh3,roceroh4,roceroh5,roceroh6,roceroh7"

if [[ "$SGLANG_ENABLE_BINARY_LAUNCH" == "1" ]]; then
    echo "Launch binary server..."
    for ((ATTN_TP_RANK=0; ATTN_TP_RANK < (TP_SIZE * PP_SIZE / WORLD_SIZE); ATTN_TP_RANK++)); do
        if [[ "$SGLANG_ENABLE_NUMA_DUPLICATION" == "1" ]]; then
            SERVER_BIN="$PYINSTALL_PATH/dist/sglang_server_tp${ATTN_TP_RANK}/sglang_server"
        else
            SERVER_BIN="python -m sglang.launch_server"
        fi

        ON_PACKAGE_MEMORY_NODE=$((ATTN_TP_RANK +16))
        echo 0 > /sys/devices/system/node/node${ATTN_TP_RANK}/hugepages/hugepages-2048kB/nr_hugepages
        echo 2020 > /sys/devices/system/node/node${ON_PACKAGE_MEMORY_NODE}/hugepages/hugepages-2048kB/nr_hugepages

        # Per-rank IB device: map ATTN_TP_RANK to an index in IB_DEVICE_ALL.
        # With N configured devices, each device serves (attn_tp_size / N) ranks.
        if [[ "$ROLE" == "prefill" || "$ROLE" == "decode" ]]; then
            IFS=',' read -ra _IB_DEVS <<< "$IB_DEVICE_ALL"
            _IB_COUNT=${#_IB_DEVS[@]}
            _IB_IDX=$((ATTN_TP_RANK * _IB_COUNT / (TP_SIZE * PP_SIZE / WORLD_SIZE)))
            IB_ARGS=(--disaggregation-ib-device "${_IB_DEVS[$_IB_IDX]}")
        else
            IB_ARGS=()
        fi

        taskset -c $((ATTN_TP_RANK * 38 + 20)) \
        $SERVER_BIN "${BASE_ARGS[@]}" "${SPECIFIC_ARGS[@]}" "${IB_ARGS[@]}" \
          --tp-rank-in-node ${ATTN_TP_RANK} \
          --port $((30000 + ATTN_TP_RANK)) \
          > "${LOG_PATH}/${DP_RANK}_${ATTN_TP_RANK}_$IP.log" 2>&1 &
    done
else
    # Non-binary launch: sglang forks workers internally, so pass all devices
    # as comma-separated string (per-rank JSON not supported by _validate_ib_devices).
    if [[ "$ROLE" == "prefill" || "$ROLE" == "decode" ]]; then
        IB_ARGS=(--disaggregation-ib-device "$IB_DEVICE_ALL")
    else
        IB_ARGS=()
    fi

    python -m sglang.launch_server "${BASE_ARGS[@]}" "${SPECIFIC_ARGS[@]}" "${IB_ARGS[@]}" \
      --port 30000 \
      > "$LOG_PATH/${DP_RANK}_$IP.log" 2>&1
fi


