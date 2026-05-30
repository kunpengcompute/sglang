#!/bin/bash
# Usage: source env.sh [prefill|decode|native]

# ------------------------------------------------------------
# Helper: expand IP range notation "base_ip | ranges"
# Example: "192.168.1. | 1-3,5" -> "192.168.1.1 192.168.1.2 192.168.1.3 192.168.1.5"
# ------------------------------------------------------------
expand_ip_range() {
    local spec="$1"
    local base="${spec%%|*}"            # part before '|'
    local ranges="${spec#*|}"           # part after '|'
    base="${base// /}"                  # trim spaces
    ranges="${ranges// /}"              # trim spaces

    IFS=',' read -ra parts <<< "$ranges"
    local ips=()
    for part in "${parts[@]}"; do
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
    echo "${ips[@]}"
}

# ------------------------------------------------------------
# Configuration variables (edit these as needed)
# ------------------------------------------------------------
# IP range for prefill nodes
PREFILL_IP_SPEC="xxx.xxx.xxx. | 1-16"
# IP range for decode nodes
DECODE_IP_SPEC="xxx.xxx.xxx. | 17-32"

NATIVE_IP_SPEC="xxx.xxx.xxx. | 1-16"

# Master address/port (common or per role)
PREFILL_MASTER_ADDR="xxx.xxx.xxx.1"
PREFILL_MASTER_PORT="5000"

DECODE_MASTER_ADDR="xxx.xxx.xxx.17"
DECODE_MASTER_PORT="5010"

NATIVE_MASTER_ADDR="xxx.xxx.xxx.1"
NATIVE_MASTER_PORT="5010"

# Paths
LOG_BASE_DIR="/path-to-logs"
CONDA_ENV_NAME="my_env"
CONDA_SH_PATH="/path-to-conda-start-sh"
MODEL_PATH="/path-to-deepseek-r1-channel-int8"

# TP/EP size
export TP_SIZE=256
export EP_SIZE=256

# Communication
export GLOO_SOCKET_IFNAME=enp26s0f0

# Thread
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export TORCH_USE_KUPL=0
export KUPL_EXECUTOR_BACKEND=pthread
export KUPL_EXECUTOR_COUNT=32
export TORCH_COMPILE_DISABLE=1
export SGLANG_ENABLE_TORCH_COMPILE=0

# SGLang
export SGLANG_LOG_MS=1
export SGLANG_USE_CPU_ENGINE=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export SGLANG_WARMUP_TIMEOUT=1600

# Kunpeng CPU
export SGLANG_USE_CPU_920F=1
export SGLANG_KUNPENG_PROFILE=0
export SGLANG_ENABLE_BINARY_LAUNCH=1
export SGLANG_ENABLE_KUTACC_COMM_OPS=0

HPCKIT_PATH=/path-to-HPCKit
source ${HPCKIT_PATH}/latest/compiler/bisheng/env/setvars.sh
source ${HPCKIT_PATH}/latest/kupl/bisheng/env/setvars.sh

# ------------------------------------------------------------
# Function: prefill_config
# ------------------------------------------------------------
prefill_config() {
    # Expand IP list
    NODE_IPS=($(expand_ip_range "$PREFILL_IP_SPEC"))
    export WORLD_SIZE=${#NODE_IPS[@]}
    export MASTER_ADDR="$PREFILL_MASTER_ADDR"
    export MASTER_PORT="$PREFILL_MASTER_PORT"
    export NODE_IPS_LIST="${NODE_IPS[*]}"
    export ROLE="prefill"
    export LOG_DIR="${LOG_BASE_DIR}/$(date +%y%m%d)/$ROLE/$(date +%H%M%S)"
    export CONDA_ACTIVATE_CMD="source ${CONDA_SH_PATH} && conda activate ${CONDA_ENV_NAME}"
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
    export LOG_DIR="${LOG_BASE_DIR}/$(date +%y%m%d)/$ROLE/$(date +%H%M%S)"
    export CONDA_ACTIVATE_CMD="source ${CONDA_SH_PATH} && conda activate ${CONDA_ENV_NAME}"
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
    export LOG_DIR="${LOG_BASE_DIR}/$(date +%y%m%d)/$(date +%H%M%S)"
    export CONDA_ACTIVATE_CMD="source ${CONDA_SH_PATH} && conda activate ${CONDA_ENV_NAME}"
}

# ------------------------------------------------------------
# Main: dispatch based on command-line argument
# ------------------------------------------------------------
ACTION="$1"
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
    *)
        echo "Usage: source env.sh [prefill|decode|native]" >&2
        return 1
        ;;
esac
