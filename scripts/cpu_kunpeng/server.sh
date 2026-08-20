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
INSTANCE="$4"
BUCKET="$5"
IP="$(ifconfig enp26s0f0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}')"

# Source environment config (exports CONDA_ACTIVATE_CMD, PYTHON_SCRIPT, etc.)
source ./env.sh "$ROLE" "$INSTANCE" "$BUCKET"

rmmod sdma_dae 2>/dev/null || true
insmod "$SDMA_KO_PATH" safe_mode=0 share_chns=160

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
    --dp-size "$DP_SIZE"
    --tp-size "$TP_SIZE"
    --ep-size "$EP_SIZE"
    --pp-size "$PP_SIZE"
    --page-size 64
    --mem-fraction-static 0.88
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --skip-server-warmup
    --disable-custom-all-reduce
    --disable-radix-cache
    --enable-dp-attention
    --enable-dp-lm-head
    --enable-dp-mlp
    --enable-dp-attention-local-control-broadcast
    --quantization w8a8_int8
    ${LOAD_FORMAT:+--load-format "$LOAD_FORMAT"}
    --chat-template  "$SGLANG_PATH/examples/chat_template/tool_chat_template_deepseekr1.jinja"
    --tool-call-parser deepseekv3
    --reasoning-parser deepseek-r1
    --stream-interval "$STREAM_INTERVAL"
)

# Add redundant experts only when enabled (REDUNDANT_EXPERTS > 0)
if [[ "${REDUNDANT_EXPERTS:-0}" -gt 0 ]]; then
    BASE_ARGS+=(--ep-num-redundant-experts "$REDUNDANT_EXPERTS")
fi

# Redundant experts require a dispatch algorithm; default to static
# when EP_DISPATCH_ALGORITHM is not set.
if [[ -n "${EP_DISPATCH_ALGORITHM:-}" ]] || [[ "${REDUNDANT_EXPERTS:-0}" -gt 0 ]]; then
    BASE_ARGS+=(--ep-dispatch-algorithm "${EP_DISPATCH_ALGORITHM:-static}")
fi

# Pass expert-location mapping file (JSON/PT) to --init-expert-location when set
if [[ -n "${INIT_EXPERT_LOCATION:-}" ]]; then
    BASE_ARGS+=(--init-expert-location "$INIT_EXPERT_LOCATION")
fi

# Disable overlap schedule
if [[ "$SGLANG_ENABLE_OVERLAP" == "0" ]]; then
    BASE_ARGS+=(
        --disable-overlap-schedule
    )
fi

if [[ "$SGLANG_ENABLE_MTP" == "1" ]]; then
    BASE_ARGS+=(
        --speculative-algorithm NEXTN
        --speculative-num-steps 1
        --speculative-eagle-topk 1
    )
    # Use explicit draft model path if set; otherwise omit the argument
    # (DeepSeek MTP falls back to MODEL_PATH automatically).
    if [[ -n "$SPECULATIVE_DRAFT_MODEL_PATH" ]]; then
        BASE_ARGS+=(--speculative-draft-model-path "$SPECULATIVE_DRAFT_MODEL_PATH")
    fi
fi

# Role-specific arguments
case "$ROLE" in
    prefill)
        SPECIFIC_ARGS=(
            --disaggregation-mode prefill
            --disaggregation-bootstrap-port 9001
            --max-prefill-tokens $((SGLANG_KUNPENG_MAX_SEQ_NUM * SGLANG_KUNPENG_MAX_CUR_LEN))
            --max-total-tokens 139328
            --prefill-max-requests "$SGLANG_KUNPENG_MAX_SEQ_NUM"
            --load-balance-method round_robin
            --enable-dynamic-batch-tokenizer
        )
        ;;
    decode)
        SPECIFIC_ARGS=(
            --disaggregation-mode decode
            --max-total-tokens 139328
            --load-balance-method round_robin
            --decode-log-interval 2
            --num-reserved-decode-tokens 256
            --max-running-requests  $((SGLANG_KUNPENG_MAX_SEQ_NUM * DP_SIZE))
        )
        ;;
    native)
        SPECIFIC_ARGS=(
            --disaggregation-bootstrap-port 9001
            --prefill-max-requests "$SGLANG_KUNPENG_MAX_SEQ_NUM"
            --max-prefill-tokens $((SGLANG_KUNPENG_MAX_SEQ_NUM * SGLANG_KUNPENG_MAX_CUR_LEN))   # long prompt -> 131072
            --max-total-tokens 18496    # long prompt -> 131072
            # --context-length 131072   # long prompt
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
            --health-check-interval-secs 10000
            --queue-timeout-secs 10000
            --request-timeout-secs 10000
            --health-check-timeout-secs 10000
            --host "$IP"
        )
        if [[ "$PREFILL_BUCKET" == "1" ]]; then
            SPECIFIC_ARGS+=(
                --prefill "${PREFILL_LONG_PROMPT_MASTER_ADDR:+http://$PREFILL_LONG_PROMPT_MASTER_ADDR:30000}" 9001
                --prefill-policy bucket
                --balance-abs-threshold 64
                --balance-rel-threshold 1.5
                --bucket-adjust-interval-secs 5
            )
        fi
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
        # Launch prefill HTTP server (tokenizer side)
        echo "Launching prefill HTTP server..."
        LD_PRELOAD="$LIBPTHREAD_HOOK_PATH" \
        python -m sglang.launch_server \
            "${HTTP_COMMON_ARGS[@]}" \
            --dp-size "$PREFILL_DP_SIZE" \
            --port 30001 \
            --dist-init-addr "$PREFILL_MASTER_ADDR:$PREFILL_MASTER_PORT" \
            --disaggregation-mode prefill \
        > "$LOG_PATH/router_prefill_http.log" 2>&1 &

        # Launch decode HTTP server (tokenizer side)
        echo "Launching decode HTTP server..."
        LD_PRELOAD="$LIBPTHREAD_HOOK_PATH" \
        python -m sglang.launch_server \
            "${HTTP_COMMON_ARGS[@]}" \
            --dp-size "$DECODE_DP_SIZE" \
            --port 30002 \
            --dist-init-addr "$DECODE_MASTER_ADDR:$DECODE_MASTER_PORT" \
            --disaggregation-mode decode \
        > "$LOG_PATH/router_decode_http.log" 2>&1 &

        # Poll until both HTTP servers are ready (up to 2 minutes each)
        for port in 30001 30002; do
            echo "Waiting for HTTP server on port $port to be ready..."
            ready=0
            for i in $(seq 1 900); do  # up to 30 minutes
                if curl -sf --max-time 2 "http://${ROUTER_IP}:${port}/health" >/dev/null 2>&1; then
                    ready=1
                    break
                fi
                sleep 2
            done
            [[ "$ready" -eq 1 ]] || {
                echo "ERROR: HTTP server on port $port not ready within 30 minutes" >&2
                exit 1
            }
            echo "HTTP server on port $port ready"
        done
    fi
    
    # Launch sgl-model-gateway (Rust router) ----
    echo "Launching PD disaggregation router..."
    # Prefer the absolute path from .user_env.sh (MODEL_GATEWAY_BIN);
    # fall back to the bare command name if the variable is not set.
    GATEWAY_BIN="${MODEL_GATEWAY_BIN:-sgl-model-gateway}"
    if [[ ! -x "$GATEWAY_BIN" ]] && ! command -v "$GATEWAY_BIN" >/dev/null 2>&1; then
        echo "ERROR: sgl-model-gateway not found (MODEL_GATEWAY_BIN=${MODEL_GATEWAY_BIN:-<unset>})" >&2
        exit 1
    fi
    # Bind gateway to a dedicated core (NUMA 10, first core) so it does not
    # collide with the tokenizer workers (decode tokenizer now owns NUMA 4-7,
    # i.e. cores 152-303; detokenizer owns NUMA 8-9, i.e. 304-379).
    taskset -c 380-416 "$GATEWAY_BIN" "${SPECIFIC_ARGS[@]}" \
        > "$LOG_PATH/router_$IP.log" 2>&1 &

    exit 0
fi

# Build IB device args based on role.
IB_DEVICE_ALL="roceroh0,roceroh1,roceroh2,roceroh3,roceroh4,roceroh5,roceroh6,roceroh7"

if [[ "$SGLANG_ENABLE_BINARY_LAUNCH" == "1" ]]; then
    for ((RANK_IN_NODE=0; RANK_IN_NODE < (TP_SIZE * PP_SIZE / WORLD_SIZE); RANK_IN_NODE++)); do
        if [[ "$SGLANG_ENABLE_NUMA_DUPLICATION" == "1" ]]; then
            SERVER_BIN="$PYINSTALL_PATH/dist/sglang_server_tp${RANK_IN_NODE}/sglang_server"
            # Point kuccl runtime plugin paths to this rank's NUMA-local copy.
            # kuccl_pg.py fallback: _internal/kuccl/install/{hucx,xucg}/
            if [[ "${SGLANG_ENABLE_KUCCL:-0}" == "0" ]]; then
                KUCCL_INSTALL="$PYINSTALL_PATH/dist/sglang_server_tp${RANK_IN_NODE}/_internal/kuccl/install"
                unset HUCX_DIR
                unset XUCG_DIR
                export LD_LIBRARY_PATH="$KUCCL_INSTALL:${LD_LIBRARY_PATH}"
            fi
        else
            SERVER_BIN="python -m sglang.launch_server"
            if [[ "${SGLANG_ENABLE_KUCCL:-0}" == "1" ]]; then
                export PYTHONPATH="$KUCCL_PATH:${PYTHONPATH}"
            fi
        fi

        ON_PACKAGE_MEMORY_NODE=$((RANK_IN_NODE +16))
        echo 0 > /sys/devices/system/node/node${RANK_IN_NODE}/hugepages/hugepages-2048kB/nr_hugepages
        echo 2020 > /sys/devices/system/node/node${ON_PACKAGE_MEMORY_NODE}/hugepages/hugepages-2048kB/nr_hugepages

        # Per-rank IB device: map RANK_IN_NODE to an index in IB_DEVICE_ALL.
        if [[ "$ROLE" == "prefill" || "$ROLE" == "decode" ]]; then
            IFS=',' read -ra _IB_DEVS <<< "$IB_DEVICE_ALL"
            _IB_COUNT=${#_IB_DEVS[@]}
            _ATTN_TP_SIZE=$((TP_SIZE / DP_SIZE))
            if [[ "$_ATTN_TP_SIZE" == "8" ]]; then
                # attn_tp=8 (dp=32): 1 rank maps to 1 NIC, aligned with prefill attn_rank to avoid cross-subnet RDMA
                _IB_IDX=$((RANK_IN_NODE % _IB_COUNT))
            else
                # attn_tp=16 (dp=16): 2 ranks share 1 NIC (original formula)
                _IB_IDX=$((RANK_IN_NODE * _IB_COUNT / (TP_SIZE * PP_SIZE / WORLD_SIZE)))
            fi
            IB_ARGS=(--disaggregation-ib-device "${_IB_DEVS[$_IB_IDX]}")
        else
            IB_ARGS=()
        fi

        taskset -c $((RANK_IN_NODE * 38 + 20)) \
        $SERVER_BIN "${BASE_ARGS[@]}" "${SPECIFIC_ARGS[@]}" "${IB_ARGS[@]}" \
          --tp-rank-in-node ${RANK_IN_NODE} \
          --port $((30000 + RANK_IN_NODE)) \
          > "${LOG_PATH}/${DP_RANK}_${RANK_IN_NODE}_$IP.log" 2>&1 &
    done
else
    # Non-binary launch: sglang forks workers internally, so pass all devices
    # as comma-separated string (per-rank JSON not supported by _validate_ib_devices).
    if [[ "$ROLE" == "prefill" || "$ROLE" == "decode" ]]; then
        IB_ARGS=(--disaggregation-ib-device "$IB_DEVICE_ALL")
    else
        IB_ARGS=()
    fi

    if [[ "${SGLANG_ENABLE_KUCCL:-0}" == "1" ]]; then
        export PYTHONPATH="$KUCCL_PATH:${PYTHONPATH}"
    fi
    python -m sglang.launch_server "${BASE_ARGS[@]}" "${SPECIFIC_ARGS[@]}" "${IB_ARGS[@]}" \
      --port 30000 \
      > "$LOG_PATH/${DP_RANK}_$IP.log" 2>&1 &
fi
