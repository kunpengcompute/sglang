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
# Usage: source env.sh [prefill|decode]

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

# Master address/port
PREFILL_MASTER_ADDR="xxx.xxx.xxx.1"
PREFILL_MASTER_PORT="5000"

DECODE_MASTER_ADDR="xxx.xxx.xxx.17"
DECODE_MASTER_PORT="5000"

# Paths
LOG_BASE_DIR="/path-to-logs"
CONDA_ENV_NAME="my_env"
CONDA_SH_PATH="/path-to-conda-start-sh"
MODEL_PATH="/path-to-model"

# Parallelism settings
TP_SIZE=256

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
    CONDA_ACTIVATE_CMD="source ${CONDA_SH_PATH} && conda activate ${CONDA_ENV_NAME}"
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
    CONDA_ACTIVATE_CMD="source ${CONDA_SH_PATH} && conda activate ${CONDA_ENV_NAME}"
}

# ------------------------------------------------------------
# Main: dispatch based on command-line argument
# ------------------------------------------------------------
case "$1" in
    prefill)
        prefill_config
        ;;
    decode)
        decode_config
        ;;
    *)
        echo "Usage: source env.sh [prefill|decode]" >&2
        return 1
        ;;
esac

export CONDA_ACTIVATE_CMD
export TP_SIZE

# Communication
export MC_ENABLE_PARALLEL_REG_MR=0
export GLOO_SOCKET_IFNAME="your-gloo-socket-name"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="2147483648"

# Thread
export OMP_NUM_THREADS=1
export TORCH_COMPILE_DISABLE=1
export SGLANG_ENABLE_TORCH_COMPILE=0

# SGLang
export SGLANG_USE_CPU_KUNPENG=1
export SGLANG_USE_CPU_ENGINE=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1000
