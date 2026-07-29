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
# Usage: source env.sh [prefill|decode|native|router]

# ------------------------------------------------------------
# Helper: expand IP range notation "base_ip | ranges"
# Example: "192.168.1. | 1-3,5" -> "192.168.1.1 192.168.1.2 192.168.1.3 192.168.1.5"
# ------------------------------------------------------------
expand_ip_range() {
    local spec="$1"
    local ips=()

    # Collect all IP-base prefixes (e.g. "10.36.182.") in order
    local bases=() temp="$spec"
    while [[ "$temp" =~ ([0-9]+\.[0-9]+\.[0-9]+\.) ]]; do
        bases+=("${BASH_REMATCH[1]}")
        temp="${temp#*"${BASH_REMATCH[1]}"}"
    done

    for ((idx=0; idx<${#bases[@]}; idx++)); do
        local base="${bases[idx]}"

        # Extract substring between this base and the next base (or end of string)
        local sub="${spec#*"$base"}"
        sub="${sub#"${sub%%[![:space:]]*}"}"  # trim leading spaces
        sub="${sub#|}"
        sub="${sub#"${sub%%[![:space:]]*}"}"  # trim leading spaces

        if ((idx+1 < ${#bases[@]})); then
            local next_base="${bases[idx+1]}"
            sub="${sub%%"$next_base"*}"
        fi

        # Trim trailing spaces / commas
        sub="${sub%"${sub##*[![:space:]]}"}"

        IFS=',' read -ra parts <<< "$sub"
        for part in "${parts[@]}"; do
            part="${part// /}"
            [[ -z "$part" ]] && continue
            if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            # range: start-end
                for ((i=${BASH_REMATCH[1]}; i<=${BASH_REMATCH[2]}; i++)); do
                    ips+=("${base}${i}")
                done
            elif [[ "$part" =~ ^[0-9]+$ ]]; then
            # single number
                ips+=("${base}${part}")
            else
                echo "Error: invalid IP range part '$part'" >&2
                return 1
            fi
        done
    done

    echo "${ips[@]}"
}

# ------------------------------------------------------------
# Configuration variables (edit these as needed)
# ------------------------------------------------------------
# IP range, Master address/port for prefill nodes
PREFILL_IP_SPEC="xxx.xxx.xxx. | 1-16"
PREFILL_MASTER_ADDR="xxx.xxx.xxx.1"
PREFILL_MASTER_PORT="5000"

# Bucket policy switch: 1=enable second prefill instance + bucket policy, 0=disable
export PREFILL_BUCKET=0

# Prefill long prompt instance
PREFILL_LONG_PROMPT_IP_SPEC="xxx.xxx.xxx. | 17-32"
PREFILL_LONG_PROMPT_MASTER_ADDR="xxx.xxx.xxx.17"
PREFILL_LONG_PROMPT_MASTER_PORT="5020"

DECODE_IP_SPEC="xxx.xxx.xxx. | 17-32"
DECODE_MASTER_ADDR="xxx.xxx.xxx.17"
DECODE_MASTER_PORT="5010"

NATIVE_IP_SPEC="xxx.xxx.xxx. | 17-32"
NATIVE_MASTER_ADDR="xxx.xxx.xxx.1"
NATIVE_MASTER_PORT="5010"

# Router node IP (single IP for PD disaggregation router)
export ROUTER_IP="xxx.xxx.xxx.1"

# Paths
LOG_BASE_DIR="/path-to-logs"
CONDA_ENV_NAME="my_env"
CONDA_BASE_PATH="/path-to-conda"
MODEL_PATH="/path-to-deepseek-r1-channel-int8"
SPECULATIVE_DRAFT_MODEL_PATH=""

export HPCKIT_PATH="/path-to-HPCKit"
export OpenBLAS_PATH="/path-to-OpenBLAS"
export KUPL_PATH="/path-to-KUPL"
export KUTACC_PATH="/path-to-KUTACC"
export SGLANG_PATH="/path-to-SGLang"
export CONDA_ENV_PATH="$CONDA_BASE_PATH/envs/$CONDA_ENV_NAME"
export PYINSTALL_PATH="$SGLANG_PATH/scripts/cpu_kunpeng/pyinstall"
# Required when SGLANG_ENABLE_TOKENIZER_SEPERATE=1 (used by tokenizer-side HTTP servers)
export LIBPTHREAD_HOOK_PATH="/path/to/libpthread_hook.so"
export GEMM_TILING_PLAN_FILE="$SGLANG_PATH/scripts/cpu_kunpeng/configs"

# Kunpeng SDMA driver
export SDMA_KO_PATH="/path-to-sdma-ko"

# TP/EP size
export TP_SIZE=256
export EP_SIZE=${TP_SIZE}

# PP size and chunked prefill size can be configured independently
export CHUNKED_PREFILL_SIZE=-1  # must be divisible by page_size * dp_size
export PP_SIZE=1 # >1 enable pp  eg: 2
export PREFILL_PP_SIZE=1
export PREFILL_LONG_PROMPT_PP_SIZE=2
export DECODE_PP_SIZE=1

# Optional second argument: prefill instance name (e.g. "long_prompt")
if [[ -n "$2" ]]; then
    export PREFILL_INSTANCE="$2"
fi

# Optional third argument: enable prefill bucket policy (e.g. "prefill_bucket")
if [[ "$3" == "prefill_bucket" ]]; then
    export PREFILL_BUCKET=1
fi

# Communication
export GLOO_SOCKET_IFNAME=enp26s0f0
export MV2_COMM_WORLD_LOCAL_SIZE=16

# Thread
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false
export TORCH_USE_KUPL=1
export KUPL_EXECUTOR_BACKEND=pthread
export KUPL_EXECUTOR_COUNT=32
export TORCH_COMPILE_DISABLE=1
export SGLANG_ENABLE_TORCH_COMPILE=0
export SGLANG_FORWARD_ASYNC=0

# SGLang
export SGLANG_LOG_MS=1
export SGLANG_USE_CPU_ENGINE=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export SGLANG_WARMUP_TIMEOUT=1600
# Force query prefill DP rank, when disaggregation and curl -d is needed
export SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK=0

# Kunpeng CPU
export SGLANG_USE_CPU_920F=1
export SGLANG_ENABLE_TOKENIZER_SEPERATE=0  # Only supports PD disaggregation mode
export SGLANG_KUNPENG_PROFILE=0
export SGLANG_ENABLE_BINARY_LAUNCH=1
export SGLANG_ENABLE_NUMA_DUPLICATION=1
export SGLANG_KUNPENG_DISABLE_MLA_ALL2ALL=0
export SGLANG_KUNPENG_RDMA_ALLGATHER=0  # requires kutacc built from https://gitcode.com/zhengzhong722/kutacc/tree/br_sglang
export SGLANG_ENABLE_MTP=0
# Kunpeng SHM pool
export SGLANG_KUNPENG_PREFILL_SHM_SIZE_MB=476
export SGLANG_KUNPENG_DECODE_SHM_SIZE_MB=50
# Kunpeng HBW pool
export SGLANG_ENABLE_HBW_POOL=1
export SGLANG_KUNPENG_WEIGTHS_HBW_POOL_SIZE_MB=4000
export SGLANG_KUNPENG_SWAP_EXPERT=0
export SGLANG_KUNPENG_SWAP_KV=0
export SGLANG_KUNPENG_SWAP_KV_BLOCKWISE=0
# Kunpeng SDMA parameters
export SGLANG_KUNPENG_SDMA_MAX_EVENTS=10
export SGLANG_KUNPENG_SDMA_THRESHOLD=5
# Kunpeng graph capture
export SGLANG_ENABLE_GRAPH_CAPTURE=0
export SGLANG_ENABLE_GRAPH_PROFILE=0
# Load format (e.g. "kunpeng_state", leave empty for default)
export LOAD_FORMAT=""
# PD disaggregation mode
export IS_PREFILL="1"
# Other options
export DROP_CACHES=0

# ------------------------------------------------------------
# load local config
# ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_ENV_ROLE="${1:-native}"
if [[ -f "$SCRIPT_DIR/.user_env.sh" ]]; then
    source "$SCRIPT_DIR/.user_env.sh" "$USER_ENV_ROLE"
fi

# ------------------------------------------------------------
# Function: prefill_config
# ------------------------------------------------------------
prefill_config() {
    local instance="${PREFILL_INSTANCE:-1}"
    if [[ "$instance" == "long_prompt" ]]; then
        NODE_IPS=($(expand_ip_range "$PREFILL_LONG_PROMPT_IP_SPEC"))
        export MASTER_ADDR="$PREFILL_LONG_PROMPT_MASTER_ADDR"
        export MASTER_PORT="$PREFILL_LONG_PROMPT_MASTER_PORT"
        export PP_SIZE=$PREFILL_LONG_PROMPT_PP_SIZE
    else
        NODE_IPS=($(expand_ip_range "$PREFILL_IP_SPEC"))
        export MASTER_ADDR="$PREFILL_MASTER_ADDR"
        export MASTER_PORT="$PREFILL_MASTER_PORT"
        export PP_SIZE=$PREFILL_PP_SIZE
    fi
    export WORLD_SIZE=${#NODE_IPS[@]}
    export NODE_IPS_LIST="${NODE_IPS[*]}"
    export ROLE="prefill"
    export IS_PREFILL="1"
    export SGLANG_SKIP_HTTP=1
}

# ------------------------------------------------------------
# Function: decode_config
# ------------------------------------------------------------
decode_config() {
    NODE_IPS=($(expand_ip_range "$DECODE_IP_SPEC"))
    export WORLD_SIZE=${#NODE_IPS[@]}
    export MASTER_ADDR="$DECODE_MASTER_ADDR"
    export MASTER_PORT="$DECODE_MASTER_PORT"
    export NODE_IPS_LIST="${NODE_IPS[*]}"
    export ROLE="decode"
    export IS_PREFILL="0"
    export SGLANG_SKIP_HTTP=1
    export PP_SIZE=$DECODE_PP_SIZE
}

# ------------------------------------------------------------
# Function: native_config
# ------------------------------------------------------------
native_config() {
    NODE_IPS=($(expand_ip_range "$NATIVE_IP_SPEC"))
    export WORLD_SIZE=${#NODE_IPS[@]}
    export MASTER_ADDR="$NATIVE_MASTER_ADDR"
    export MASTER_PORT="$NATIVE_MASTER_PORT"
    export NODE_IPS_LIST="${NODE_IPS[*]}"
    export ROLE="native"
}

# ------------------------------------------------------------
# Function: router_config
# ------------------------------------------------------------
router_config() {
    export ROLE="router"
    export NODE_IPS_LIST="$ROUTER_IP"
    export IS_PREFILL="0"
    export SGLANG_LAUNCH_HTTP_ONLY=1
}

# ------------------------------------------------------------
# Main: dispatch based on command-line argument
# ------------------------------------------------------------
ACTION="${1:-native}"
shift

case "$ACTION" in
    prefill)
        prefill_config
        ;;
    decode)
        decode_config
        ;;
    native)
        native_config
        ;;
    router)
        router_config
        ;;
    *)
        echo "Usage: source env.sh [prefill|decode|native|router]" >&2
        return 1
        ;;
esac

source "${SCRIPT_DIR}/.time_env.sh"
export LOG_DIR="${LOG_BASE_DIR}/${LOG_DATE}/$ROLE/${LOG_TIME}"
export SGLANG_TORCH_PROFILER_DIR="${LOG_DIR}/torch_profiler"

if [[ "$IS_PREFILL" == "1" ]]; then
    export SGLANG_KUNPENG_MAX_SEQ_NUM=4
    export SGLANG_KUNPENG_MAX_CUR_LEN=1024
else
    export SGLANG_KUNPENG_MAX_SEQ_NUM=128
    export SGLANG_KUNPENG_MAX_CUR_LEN=1
fi

if [[ "$SGLANG_ENABLE_NUMA_DUPLICATION" != "1" ]] || [[ "$ROLE" == "router" ]]; then
    source ${HPCKIT_PATH}/latest/compiler/bisheng/env/setvars.sh

    export LD_LIBRARY_PATH=${OpenBLAS_PATH}/lib:${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=/usr/lib64/libibverbs:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${KUPL_PATH}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${KUTACC_PATH}/install/lib:$LD_LIBRARY_PATH
fi

export KUTACC_LIB=${KUTACC_PATH}/install/lib
export KUTACC_INCLUDE=${KUTACC_PATH}/install/include

export CPATH=${KUPL_PATH}/include:$CPATH
export INCLUDE=${KUPL_PATH}/include:$INCLUDE
export LIBRARY_PATH=${KUPL_PATH}/lib:$LIBRARY_PATH

export CONDA_ACTIVATE_CMD="eval \"\$($CONDA_BASE_PATH/bin/conda shell.bash hook)\" && conda activate $CONDA_ENV_NAME"
eval "$CONDA_ACTIVATE_CMD"