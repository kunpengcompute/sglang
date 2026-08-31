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
# Usage: source env.sh [prefill|decode|native|router|build]

# ------------------------------------------------------------
# Helper: expand IP range notation "base_ip | ranges"
# Example: "192.168.1. | 1-3,5" -> "192.168.1.1 192.168.1.2 192.168.1.3 192.168.1.5"
# If a second argument (IP_FILE) is given and the file exists,
# read one IP per line instead (empty lines and '#' comments are skipped).
# ------------------------------------------------------------
expand_ip_range() {
    local spec="$1"
    local ip_file="$2"
    local ips=()

    # If an IP file is provided, read one IP per line
    if [[ -n "$ip_file" ]] && [[ -f "$ip_file" ]]; then
        while IFS= read -r line || [[ -n "$line" ]]; do
            line="${line//[$'\r' ]/}"  # strip CR and spaces
            [[ -z "$line" ]] && continue
            [[ "$line" == \#* ]] && continue
            ips+=("$line")
        done < "$ip_file"
        echo "${ips[@]}"
        return 0
    fi

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
# IP_FILE: file with one IP per line (e.g. "192.168.1.1" per line)
# IP_SPEC: alternative range notation, e.g. "192.168.1. | 1-3,5" (used only when IP_FILE is empty)
PREFILL_IP_SPEC=""
PREFILL_IP_FILE=""
PREFILL_MASTER_ADDR="xxx.xxx.xxx.1"
PREFILL_MASTER_PORT="5000"

# Bucket policy switch: 1=enable second prefill instance + bucket policy, 0=disable
export PREFILL_BUCKET=0

# Prefill long prompt instance
PREFILL_LONG_PROMPT_IP_SPEC=""
PREFILL_LONG_PROMPT_IP_FILE=""
PREFILL_LONG_PROMPT_MASTER_ADDR="xxx.xxx.xxx.17"
PREFILL_LONG_PROMPT_MASTER_PORT="5020"

DECODE_IP_SPEC=""
DECODE_IP_FILE=""
DECODE_MASTER_ADDR="xxx.xxx.xxx.17"
DECODE_MASTER_PORT="5010"

NATIVE_IP_SPEC=""
NATIVE_IP_FILE=""
NATIVE_MASTER_ADDR="xxx.xxx.xxx.1"
NATIVE_MASTER_PORT="5010"

# Router node IP (single IP for PD disaggregation router)
export ROUTER_IP="xxx.xxx.xxx.1"

# Paths
LOG_BASE_DIR="/path-to-logs"
CONDA_ENV_NAME="my_env"
CONDA_BASE_PATH="/path-to-conda"
MODEL_PATH="/path-to-deepseek-r1-channel-int8"
MODEL_PATH_PREFILL=$MODEL_PATH
MODEL_PATH_DECODE=$MODEL_PATH
SPECULATIVE_DRAFT_MODEL_PATH="/path-to-deepseek-r1-channel-int8_mtp"
SPECULATIVE_DRAFT_MODEL_PATH_PREFILL=$SPECULATIVE_DRAFT_MODEL_PATH
SPECULATIVE_DRAFT_MODEL_PATH_DECODE=$SPECULATIVE_DRAFT_MODEL_PATH

export HPCKIT_PATH="/path-to-HPCKit"
export OpenBLAS_PATH="/path-to-OpenBLAS"
export KUPL_PATH="/path-to-KUPL"
export KUTACC_PATH="/path-to-KUTACC"
export SGLANG_PATH="/path-to-SGLang"
export DATASET_PATH="/path-to-dataset"
export CONDA_ENV_PATH="$CONDA_BASE_PATH/envs/$CONDA_ENV_NAME"
export PYINSTALL_PATH="$SGLANG_PATH/scripts/cpu_kunpeng/pyinstall"
# Required when SGLANG_ENABLE_TOKENIZER_SEPERATE=1 (used by tokenizer-side HTTP servers)
export LIBPTHREAD_HOOK_PATH="/path/to/libpthread_hook.so"
export GEMM_TILING_PLAN_FILE="$SGLANG_PATH/scripts/cpu_kunpeng/configs/dsv3_32_tiling.csv"
# Kunpeng SDMA driver
export SDMA_KO_PATH="/path-to-sdma-ko"
# Required when SGLANG_ENABLE_KUCCL=1
export KUCCL_PATH="/path-to-KUCCL"

# Native TP/EP size
export TP_SIZE=256
export DP_SIZE=16
export EP_SIZE=${TP_SIZE}
export PP_SIZE=1  # >1 enable pp  eg: 2
export REDUNDANT_EXPERTS=0
export INIT_EXPERT_LOCATION=""
export EP_DISPATCH_ALGORITHM=""  # e.g. static, dynamic
# Dynamic redundant-expert remap shuffle mode: 0 = round-robin, 1 = random
export SGLANG_KUNPENG_MOE_SHUFFLE_MODE=0

# Prefill TP/EP/PP size
export PREFILL_TP_SIZE=${TP_SIZE}
export PREFILL_DP_SIZE=${DP_SIZE}
export PREFILL_EP_SIZE=${PREFILL_TP_SIZE}
export PREFILL_PP_SIZE=${PP_SIZE}
export PREFILL_REDUNDANT_EXPERTS=0
export PREFILL_INIT_EXPERT_LOCATION=""
export PREFILL_EP_DISPATCH_ALGORITHM=""
export PREFILL_SGLANG_KUNPENG_MOE_SHUFFLE_MODE=0

# Decode TP/EP/PP size
export DECODE_TP_SIZE=${TP_SIZE}
export DECODE_DP_SIZE=${DP_SIZE}
export DECODE_EP_SIZE=${DECODE_TP_SIZE}
export DECODE_PP_SIZE=${PP_SIZE}
export DECODE_REDUNDANT_EXPERTS=0
export DECODE_INIT_EXPERT_LOCATION=""
export DECODE_EP_DISPATCH_ALGORITHM=""
export DECODE_SGLANG_KUNPENG_MOE_SHUFFLE_MODE=0

# PP size and chunked prefill size can be configured independently
export CHUNKED_PREFILL_SIZE=65536  # must be divisible by page_size * dp_size
export PREFILL_LONG_PROMPT_PP_SIZE=2

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
export RAYON_NUM_THREADS=1
export KUPL_EXECUTOR_BACKEND=pthread
export KUPL_EXECUTOR_COUNT=33  # set to 32 when KUTACC_ASYNC_LAUNCH=0
export KUTACC_ASYNC_LAUNCH=1  # requires kutacc built from https://gitcode.com/zhengzhong722/kutacc/tree/br_sglang
export TORCH_COMPILE_DISABLE=1
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=4
export SGLANG_SET_ZMQ_CPU_AFFINITY_OFFSET=18

# SGLang
export SGLANG_LOG_MS=1
export SGLANG_USE_CPU_ENGINE=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export SGLANG_WARMUP_TIMEOUT=1600
export PYTHONWARNINGS="ignore::FutureWarning"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=300
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=0

# Kunpeng CPU
export SGLANG_USE_CPU_920F=1
export SGLANG_ENABLE_TOKENIZER_SEPERATE=0  # Only supports PD disaggregation mode
export TOKENIZER_WORKER_NUM=1
export SGLANG_KUNPENG_PROFILE=0
export SGLANG_KUNPENG_PP_PROFILE=0  # 1 = enable decode pipeline profiling
export SGLANG_ENABLE_BINARY_LAUNCH=1
export SGLANG_ENABLE_NUMA_DUPLICATION=1
export SGLANG_KUNPENG_DISABLE_MLA_ALL2ALL=0
export SGLANG_KUNPENG_USE_LONG_CONTEXT_INFERENCE=0
export SGLANG_KUNPENG_RDMA_ALLGATHER=1  # requires kutacc built from https://gitcode.com/zhengzhong722/kutacc/tree/br_sglang
export SGLANG_KUNPENG_RDMA_BCAST=1  # requires kutacc built from https://gitcode.com/zhengzhong722/kutacc/tree/br_sglang
export SGLANG_KUNPENG_MOE_FORCE_LOAD_BALANCE=0  # 1 = forced MoE load balancing (perf-test only, correctness not preserved)
# SGLANG_KUNPENG_DEBUG_EXPERT_LOAD=1  # build-time only: set before building sgl-kernel to enable expert load debug recording
export SGLANG_ENABLE_MTP=0
export SGLANG_ENABLE_OVERLAP=0
export SGLANG_ENABLE_OVERLAP_TRACE=0
export SGLANG_ENABLE_KUCCL=0  # set to 1 to use kuccl backend instead of gloo
# Kunpeng SHM pool
export SGLANG_KUNPENG_PREFILL_SHM_SIZE_MB=476
export SGLANG_KUNPENG_DECODE_SHM_SIZE_MB=50
export SGLANG_KUNPENG_ENABLE_SHM_FENCE=0
# SHM_ON_PACKAGE: requires kupl built from https://gitcode.com/kunpengcompute/kupl/tree/sglang_830
export KUPL_SHM_TYPE=sls
export KUPL_SHM_ON_PACKAGE=y
export KUPL_SHM_ENABLE_HUGEPAGE=y
# Kunpeng HBW pool
export SGLANG_ENABLE_HBW_POOL=1
export SGLANG_KUNPENG_WEIGTHS_HBW_POOL_SIZE_MB=3900
export SGLANG_KUNPENG_SWAP_KV_IN=0
export SGLANG_KUNPENG_SWAP_KV_OUT=0
export SGLANG_KUNPENG_SWAP_KV_BLOCKWISE=0
export SGLANG_KUNPENG_SWAP_MAX_KV_BLOCKS=512
export PREFILL_WEIGTHS_HBW_POOL_SIZE_MB=3400
export PREFILL_SWAP_KV_IN=0
export PREFILL_SWAP_KV_OUT=0
export PREFILL_SWAP_KV_BLOCKWISE=0
export DECODE_WEIGTHS_HBW_POOL_SIZE_MB=3900
export DECODE_SWAP_KV_IN=0
export DECODE_SWAP_KV_OUT=0
export DECODE_SWAP_KV_BLOCKWISE=0
# Kunpeng SDMA parameters
export SGLANG_KUNPENG_SDMA_MAX_EVENTS=10
export SGLANG_KUNPENG_SDMA_THRESHOLD=5
# Kunpeng graph capture
export SGLANG_ENABLE_GRAPH_CAPTURE=1
export SGLANG_ENABLE_GRAPH_PROFILE=0
export SGLANG_KUNPENG_GRAPH_CACHE_SIZE=8
# Kunpeng prefill graph padding to power 2 size
export SGLANG_KUNPENG_EXTEND_POWER_2_PADDING=0
# Load format (e.g. "kunpeng_state", leave empty for default)
export LOAD_FORMAT=""
# Other options
export DROP_CACHES=0
# Tokenizer-side cross-process batch timeline logging
export SGLANG_TOKENIZER_TIMELINE_LOG=0
# Scheduler stream interval (--stream-interval): flush a request's output every N tokens
export STREAM_INTERVAL=1
# Dedicated CPU list for the disaggregation bootstrap server thread (only
# effective in tokenizer-separate mode). 
export SGLANG_KUNPENG_BOOTSTRAP_SERVER_CPU=418-422
# Tokenizer backend: "huggingface" (default) or "fastokens"
# (fastokens requires transformers >= 5.12; on 5.6.0 it hits a Metaspace
#  pre-tokenizer error with DeepSeek-R1)
export SGLANG_TOKENIZER_BACKEND="huggingface"

# ------------------------------------------------------------
# Load local config
# ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_ENV_ROLE="${1:-native}"

case "$USER_ENV_ROLE" in
    decode|router) IS_PREFILL=0 ;;
    *) IS_PREFILL=1 ;;
esac
export IS_PREFILL

if [[ -f "$SCRIPT_DIR/.user_env.sh" ]]; then
    source "$SCRIPT_DIR/.user_env.sh" "$USER_ENV_ROLE"
fi

# Kuccl backend (UCX + UCG) environment
if [[ "$SGLANG_ENABLE_KUCCL" == "1" ]]; then
    export UCG_PLANC_UCX_BCAST_ATTR=I:1
    export UCX_MEM_EVENTS=no
    export UCX_UD_VERBS_ALLOC=thp,md,mmap,heap
    export UCX_RC_VERBS_ALLOC=thp,md,mmap,heap

    if [[ "$SGLANG_ENABLE_NUMA_DUPLICATION" != "1" ]]; then
        export HUCX_DIR="${HPCKIT_PATH}/26.1.RC1/hmpi/bisheng/release/hucx"
        export XUCG_DIR="${HPCKIT_PATH}/26.1.RC1/hmpi/bisheng/release/xucx"
        export UCX_MODULE_DIR="${HUCX_DIR}/lib/ucx"
        export UCX_PLANC=ucx
        export UCG_PLANC_PATH="${XUCG_DIR}/lib/planc"
        export LD_LIBRARY_PATH="${HUCX_DIR}/lib:${XUCG_DIR}/lib:${XUCG_DIR}/lib/planc:${LD_LIBRARY_PATH:-}"
        export PYTHONPATH="${KUCCL_PATH}:${PYTHONPATH:-}"
    fi
fi

# Defaults below use ":-" so explicit .user_env.sh overrides still win.
if [[ "$IS_PREFILL" == "1" ]]; then
    export SGLANG_KUNPENG_SWAP_EXPERT="${SGLANG_KUNPENG_SWAP_EXPERT:-1}"
    export SGLANG_KUNPENG_MAX_SEQ_NUM="${SGLANG_KUNPENG_MAX_SEQ_NUM:-8}"
    export SGLANG_KUNPENG_MAX_CUR_LEN="${SGLANG_KUNPENG_MAX_CUR_LEN:-512}"
    export SGLANG_KUNPENG_MAX_SEQ_LEN="${SGLANG_KUNPENG_MAX_SEQ_LEN:-4096}"
else
    export SGLANG_KUNPENG_MAX_SEQ_NUM="${SGLANG_KUNPENG_MAX_SEQ_NUM:-64}"
    # Decode MTP speculates 1 extra token per step: cur len must be 2
    if [[ -z "${SGLANG_KUNPENG_MAX_CUR_LEN:-}" ]]; then
        if [[ "$SGLANG_ENABLE_MTP" == "1" ]]; then
            export SGLANG_KUNPENG_MAX_CUR_LEN=2
        else
            export SGLANG_KUNPENG_MAX_CUR_LEN=1
        fi
    fi
fi

# ------------------------------------------------------------
# Helpers: resolve variables by role prefix (PREFILL / DECODE / NATIVE)
# ------------------------------------------------------------
_export_node_config() {
    local prefix="$1"
    local _var
    _var="${prefix}_IP_FILE"; local _ip_file="${!_var:-}"
    _var="${prefix}_IP_SPEC"; NODE_IPS=($(expand_ip_range "${!_var}" "$_ip_file"))
    _var="${prefix}_MASTER_ADDR"; export MASTER_ADDR="${!_var}"
    _var="${prefix}_MASTER_PORT"; export MASTER_PORT="${!_var}"
    export WORLD_SIZE=${#NODE_IPS[@]}
    export NODE_IPS_LIST="${NODE_IPS[*]}"
}

_export_pd_vars() {
    local prefix="$1"
    local _var
    _var="${prefix}_TP_SIZE";    export TP_SIZE="${!_var}"
    _var="${prefix}_DP_SIZE";    export DP_SIZE="${!_var}"
    _var="${prefix}_EP_SIZE";    export EP_SIZE="${!_var}"
    _var="${prefix}_PP_SIZE";    export PP_SIZE="${!_var}"
    _var="${prefix}_REDUNDANT_EXPERTS"; export REDUNDANT_EXPERTS="${!_var}"
    _var="${prefix}_INIT_EXPERT_LOCATION"; export INIT_EXPERT_LOCATION="${!_var}"
    _var="${prefix}_EP_DISPATCH_ALGORITHM"; export EP_DISPATCH_ALGORITHM="${!_var}"
    _var="${prefix}_SGLANG_KUNPENG_MOE_SHUFFLE_MODE"; export SGLANG_KUNPENG_MOE_SHUFFLE_MODE="${!_var}"
    _var="MODEL_PATH_${prefix}"; export MODEL_PATH="${!_var}"
    _var="SPECULATIVE_DRAFT_MODEL_PATH_${prefix}"; export SPECULATIVE_DRAFT_MODEL_PATH="${!_var}"
    _var="${prefix}_WEIGTHS_HBW_POOL_SIZE_MB"; export SGLANG_KUNPENG_WEIGTHS_HBW_POOL_SIZE_MB="${!_var}"
    _var="${prefix}_SWAP_KV_IN"; export SGLANG_KUNPENG_SWAP_KV_IN="${!_var}"
    _var="${prefix}_SWAP_KV_OUT"; export SGLANG_KUNPENG_SWAP_KV_OUT="${!_var}"
    _var="${prefix}_SWAP_KV_BLOCKWISE"; export SGLANG_KUNPENG_SWAP_KV_BLOCKWISE="${!_var}"
}

# ------------------------------------------------------------
# Per-role config functions (called via "${ACTION}_config")
# ------------------------------------------------------------
prefill_config() {
    _export_pd_vars "PREFILL"
    local instance="${PREFILL_INSTANCE:-1}"
    local _prefix
    if [[ "$instance" == "long_prompt" ]]; then
        export PP_SIZE=$PREFILL_LONG_PROMPT_PP_SIZE
        _prefix="PREFILL_LONG_PROMPT"
    else
        _prefix="PREFILL"
    fi
    _export_node_config "$_prefix"
    export SGLANG_SKIP_HTTP=1
}

decode_config() {
    _export_pd_vars "DECODE"
    _export_node_config "DECODE"
    export SGLANG_SKIP_HTTP=1
}

native_config() {
    _export_node_config "NATIVE"
}

router_config() {
    export NODE_IPS_LIST="$ROUTER_IP"
    export SGLANG_LAUNCH_HTTP_ONLY=1
}

build_config() {
    :
}

# ------------------------------------------------------------
# Main: dispatch based on command-line argument
# ------------------------------------------------------------
ACTION="${1:-native}"
shift

case "$ACTION" in
    prefill|decode|native|router|build)
        "${ACTION}_config"
        ;;
    *)
        echo "Usage: source env.sh [prefill|decode|native|router|build]" >&2
        return 1
        ;;
esac

if [[ "$ACTION" == "router" && "$SGLANG_ENABLE_TOKENIZER_SEPERATE" == "1" ]]; then
    export RAYON_NUM_THREADS=32
fi

source "${SCRIPT_DIR}/.time_env.sh"
export LOG_DIR="${LOG_BASE_DIR}/${LOG_DATE}/$ACTION/${LOG_TIME}"
export SGLANG_TORCH_PROFILER_DIR="${LOG_DIR}/torch_profiler"

if [[ "$SGLANG_ENABLE_NUMA_DUPLICATION" != "1" ]] || [[ "$ACTION" == "router" ]] || [[ "$ACTION" == "build" ]]; then
    source ${HPCKIT_PATH}/latest/compiler/bisheng/env/setvars.sh

    export LD_LIBRARY_PATH=${OpenBLAS_PATH}/lib:${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=/usr/lib64/libibverbs:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${KUPL_PATH}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${KUTACC_PATH}/lib:$LD_LIBRARY_PATH
fi

export KUTACC_LIB=${KUTACC_PATH}/lib
export KUTACC_INCLUDE=${KUTACC_PATH}/include

export CPATH=${KUPL_PATH}/include:$CPATH
export INCLUDE=${KUPL_PATH}/include:$INCLUDE
export LIBRARY_PATH=${KUPL_PATH}/lib:$LIBRARY_PATH

export CONDA_ACTIVATE_CMD="eval \"\$($CONDA_BASE_PATH/bin/conda shell.bash hook)\" && conda activate $CONDA_ENV_NAME"
eval "$CONDA_ACTIVATE_CMD"