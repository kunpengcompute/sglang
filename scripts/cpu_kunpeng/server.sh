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
# server.sh - Single node execution for SGLang (prefill/decode/native).
# Router (gateway) is launched by runtime/server_router.sh instead.
# Tokenizer HTTP servers are launched by runtime/server_tokenizer.sh instead.
# Usage: ./server.sh <role> <dp_rank> <log_path> [instance]

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <role> <dp_rank> <log_path> [instance]" >&2
    exit 1
fi

ROLE="$1"
# NOTE: $2 is the NODE index inside the role's NODE_IPS_LIST (0..WORLD_SIZE-1),
# NOT the data-parallel rank. The true dp/pp/tp identity of each launched
# process is derived below from (node index, rank-in-node).
NODE_RANK="$2"
LOG_PATH="$3"
INSTANCE="${4:-}"
IP="$(ifconfig enp26s0f0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}')"

# Source environment config (exports CONDA_ACTIVATE_CMD, PYTHON_SCRIPT, etc.)
source ./env.sh "$ROLE" "$INSTANCE" || exit 1

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
    --node-rank "$NODE_RANK"
    --dist-timeout 600
    --dp-size "$DP_SIZE"
    --tp-size "$TP_SIZE"
    --ep-size "$EP_SIZE"
    --pp-size "$PP_SIZE"
    --page-size 64
    --mem-fraction-static 0.88
    --chunked-prefill-size $((CHUNKED_PREFILL_SIZE_PER_DP * DP_SIZE))
    --skip-server-warmup
    --disable-custom-all-reduce
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

# Disable radix cache
if [[ "$SGLANG_DISABLE_RADIX_CACHE" == "1" ]]; then
    BASE_ARGS+=(--disable-radix-cache)
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
        --speculative-num-steps "${SGLANG_SPECULATIVE_NUM_STEPS:-2}"
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
            --max-prefill-tokens $((SGLANG_KUNPENG_MAX_SEQ_NUM * SGLANG_KUNPENG_MAX_CUR_LEN))
            --max-total-tokens 180000
            --prefill-max-requests "$SGLANG_KUNPENG_MAX_SEQ_NUM"
            --max-running-requests $((8 * SGLANG_KUNPENG_MAX_SEQ_NUM * DP_SIZE))
            --load-balance-method round_robin
            --enable-dynamic-batch-tokenizer
            --disaggregation-bootstrap-port 9001
        )
        ;;
    decode)
        SPECIFIC_ARGS=(
            --disaggregation-mode decode
            --max-total-tokens 180000
            --load-balance-method round_robin
            --decode-log-interval 1
            --num-reserved-decode-tokens 1024
            --max-running-requests $((4 * SGLANG_KUNPENG_MAX_SEQ_NUM * DP_SIZE))
        )
        if [[ "${DECODE_FAKE_TRANSFER:-0}" == "1" ]]; then
            SPECIFIC_ARGS+=(--disaggregation-transfer-backend fake)
        fi
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
    tokenizer)
        # Tokenizer HTTP servers now live in runtime/server_tokenizer.sh,
        # invoked by runtime/launch_tokenizer.sh. server.sh no longer handles it.
        echo "Error: role 'tokenizer' is handled by runtime/server_tokenizer.sh" >&2
        exit 1
        ;;
    router)
        # Router (gateway) launch now lives in runtime/server_router.sh,
        # invoked by runtime/launch_router.sh. server.sh no longer handles it.
        echo "Error: role 'router' is handled by runtime/server_router.sh" >&2
        exit 1
        ;;
    *)
        echo "Error: unknown role '$ROLE'" >&2
        exit 1
        ;;
esac

# Build IB device args based on role.
IB_DEVICE_ALL="roceroh0,roceroh1,roceroh2,roceroh3,roceroh4,roceroh5,roceroh6,roceroh7"

if [[ "$SGLANG_ENABLE_BINARY_LAUNCH" == "1" ]]; then
    # ── Log naming: translate (node index, rank-in-node) into the true
    # (dp, pp, attn-tp) identity so every log file is named after its real rank.
    # Global ranks are node-major: g = NODE_RANK * LOCAL_WORLD_SIZE + tp_rank_in_node.
    #
    # Two PP layouts:
    # - Legacy node-block (SGLANG_KUNPENG_PP_LAYOUT=0): PP stages are
    #   contiguous chunks of TP_SIZE ranks (pp = g / TP_SIZE); inside one pp stage
    #   each DP group owns ATTENTION_TP_SIZE consecutive ranks
    #   (dp = (g % TP_SIZE) / ATTENTION_TP_SIZE, tp = (g % TP_SIZE) % ATTENTION_TP_SIZE).
    #   E.g. TP=256 DP=32 PP=2 on 32 nodes: node 0..15 = pp0, node 16..31 = pp1.
    # - In-node interleave (SGLANG_KUNPENG_PP_LAYOUT=1): every node hosts
    #   all PP stages; rin 0..(LRPS-1) -> PP0, next LRPS ranks -> PP1
    #   (LRPS = LOCAL_WORLD_SIZE / PP_SIZE).  Stage-local tp =
    #   NODE_RANK*LRPS + (rin % LRPS); dp = stage_tp / ATTENTION_TP_SIZE,
    #   tp = stage_tp % ATTENTION_TP_SIZE.  E.g. TP=256 DP=32 PP=2 on 32 nodes:
    #   LOCAL_WORLD_SIZE=16, LRPS=8, ATTENTION_TP_SIZE=8 -> dp = NODE_RANK.
    LOCAL_WORLD_SIZE=$((TP_SIZE * PP_SIZE / WORLD_SIZE))
    ATTENTION_TP_SIZE=$((TP_SIZE / DP_SIZE))
    # In-node interleave flag: SGLANG_KUNPENG_PP_LAYOUT=1 (or "interleave").
    _PP_INTERLEAVE=0
    if [[ "${SGLANG_KUNPENG_PP_LAYOUT:-0}" == "1" || "${SGLANG_KUNPENG_PP_LAYOUT:-0}" == "interleave" ]]; then _PP_INTERLEAVE=1; fi
    for ((RANK_IN_NODE=0; RANK_IN_NODE < (TP_SIZE * PP_SIZE / WORLD_SIZE); RANK_IN_NODE++)); do
        if [[ "$SGLANG_ENABLE_NUMA_DUPLICATION" == "1" ]]; then
            SERVER_BIN="$PYINSTALL_PATH/dist/sglang_server_tp${RANK_IN_NODE}/sglang_server"
            # Point kuccl runtime plugin paths to this rank's NUMA-local copy.
            # kuccl_pg.py fallback: _internal/kuccl/install/{hucx,xucg}/
            if [[ "${SGLANG_ENABLE_KUCCL:-0}" == "1" ]]; then
                KUCCL_LOCAL_DIR="$PYINSTALL_PATH/dist/sglang_server_tp${RANK_IN_NODE}/_internal/kuccl"
                KUCCL_INSTALL="$KUCCL_LOCAL_DIR/install"

                export HUCX_DIR="$KUCCL_INSTALL/hucx"
                export XUCG_DIR="$KUCCL_INSTALL/xucg"

                export UCX_MODULE_DIR="${HUCX_DIR}/lib/ucx"

                export UCG_PLANC=ucx
                export UCG_PLANC_PATH="${XUCG_DIR}/lib/planc"
            
                export LD_LIBRARY_PATH="${HUCX_DIR}/lib:${XUCG_DIR}/lib:${XUCG_DIR}/lib/planc:${LD_LIBRARY_PATH:-}"
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

        # True rank identity of this process (see the mapping above).
        if [[ ${_PP_INTERLEAVE} == 1 && ${PP_SIZE} -gt 1 ]]; then
            _RIN_PER_STAGE=$((LOCAL_WORLD_SIZE / PP_SIZE))
            PP_RANK=$((RANK_IN_NODE / _RIN_PER_STAGE))
            _RIN_IN_STAGE=$((RANK_IN_NODE % _RIN_PER_STAGE))
            _STAGE_TP=$((NODE_RANK * _RIN_PER_STAGE + _RIN_IN_STAGE))
            DP_RANK_ACTUAL=$((_STAGE_TP / ATTENTION_TP_SIZE))
            TP_RANK_ACTUAL=$((_STAGE_TP % ATTENTION_TP_SIZE))
        else
            GLOBAL_RANK=$((NODE_RANK * LOCAL_WORLD_SIZE + RANK_IN_NODE))
            PP_RANK=$((GLOBAL_RANK / TP_SIZE))
            _IN_PP=$((GLOBAL_RANK % TP_SIZE))
            DP_RANK_ACTUAL=$((_IN_PP / ATTENTION_TP_SIZE))
            TP_RANK_ACTUAL=$((_IN_PP % ATTENTION_TP_SIZE))
        fi

        taskset -c $((RANK_IN_NODE * 38 + 20)) \
        $SERVER_BIN "${BASE_ARGS[@]}" "${SPECIFIC_ARGS[@]}" "${IB_ARGS[@]}" \
          --tp-rank-in-node ${RANK_IN_NODE} \
          --port $((30000 + RANK_IN_NODE)) \
          > "${LOG_PATH}/pp${PP_RANK}_dp${DP_RANK_ACTUAL}_tp${TP_RANK_ACTUAL}_$IP.log" 2>&1 &
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
      > "$LOG_PATH/${NODE_RANK}_$IP.log" 2>&1 &
fi
