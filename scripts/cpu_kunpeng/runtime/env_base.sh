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
# Base environment setup for env.sh: configuration variables, local config
# loading (.user_env.sh) and role-dependent defaults.
# This file is sourced by env.sh and should not be used directly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Source helper functions (expand_ip_range, per-role config functions, etc.)
source "$SCRIPT_DIR/runtime/env_helper.sh"

# ------------------------------------------------------------
# Configuration variables (edit these as needed)
# ------------------------------------------------------------
# Deployment instances for `launch.sh all`: comma-separated entries, each
# "<role>" or "<role>_<instance>". Each entry starts one server group via
# `launch_cluster.sh <role> [instance]`; the instance suffix selects
# per-instance user env overrides (runtime/.user_env_<role>_<instance>.sh),
# e.g. "decode_128p" reads .user_env_decode_128p.sh on every node.
INSTANCES="prefill,decode"

# IP range, Master address/port for native nodes.
# prefill/decode topology now lives in runtime/env_prefill.sh / env_decode.sh.
# IP_FILE: file with one IP per line (e.g. "192.168.1.1" per line)
# IP_SPEC: alternative range notation, e.g. "192.168.1. | 1-3,5" (used only when IP_FILE is empty)
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
SPECULATIVE_DRAFT_MODEL_PATH="/path-to-deepseek-r1-channel-int8_mtp"

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
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=0 # WARNING: Setting SGLANG_SCHEDULER_SKIP_ALL_GATHER=1 can affect correctness in long-prompt prefill.

# Kunpeng CPU
export SGLANG_USE_CPU_920F=1
export TOKENIZER_WORKER_NUM=1
export SGLANG_KUNPENG_PROFILE=0
export SGLANG_KUNPENG_PP_PROFILE=0  # 1 = enable decode pipeline profiling
export SGLANG_ENABLE_BINARY_LAUNCH=1
export SGLANG_ENABLE_NUMA_DUPLICATION=1
export SGLANG_KUNPENG_DISABLE_MLA_ALL2ALL=0
export SGLANG_KUNPENG_ALLTOALL_FENCE=0  # 1 = explicit kupl_shm_fence after MLA shm alltoall
export SGLANG_KUNPENG_ALLREDUCE_NAIVE=0  # 1 = non-kutacc naive shm allreduce
export SGLANG_KUNPENG_LC_DP_RANKS=""  # comma-separated DP ranks running long-context decode CP (e.g. "14,15"); empty = all regular; must be identical between prefill and decode instances
export SGLANG_KUNPENG_LC_MIN_SEQ_LEN=4096  # requests with input_len + max_new_tokens >= this go to LC DP ranks; must exceed every regular rank's max_req_input_len
export SGLANG_KUNPENG_RDMA_ALLGATHER=1  # requires kutacc built from https://gitcode.com/zhengzhong722/kutacc/tree/br_sglang
export SGLANG_KUNPENG_RDMA_BCAST=1  # requires kutacc built from https://gitcode.com/zhengzhong722/kutacc/tree/br_sglang
export SGLANG_KUNPENG_RDMA_PP_COMM=1  # 1 = enable RDMA communication between PP ranks for rids
export SGLANG_KUNPENG_PP_LAYOUT=0  # PP layout: 0 = node_block (a node maps to one PP stage, legacy); 1 = interleave (every node hosts all PP stages: rin 0..7 -> PP0, 8..15 -> PP1).
export SGLANG_KUNPENG_MOE_FORCE_LOAD_BALANCE=0  # 1 = forced MoE load balancing (perf-test only, correctness not preserved)
# SGLANG_KUNPENG_DEBUG_EXPERT_LOAD=1  # build-time only: set before building sgl-kernel to enable expert load debug recording
export SGLANG_ENABLE_MTP=0
export SGLANG_DISABLE_RADIX_CACHE=0  # 1 = append --disable-radix-cache to server args
export SGLANG_ENABLE_OVERLAP=0
export SGLANG_ENABLE_OVERLAP_TRACE=0
export SGLANG_ENABLE_KUCCL=0  # set to 1 to use kuccl backend instead of gloo
# Kunpeng SHM pool
export SGLANG_KUNPENG_ENABLE_SHM_FENCE=0
export KUPL_SHM_TYPE=sls
export KUPL_SHM_ON_PACKAGE=y  # requires kupl built from https://gitcode.com/kunpengcompute/kupl/tree/sglang_830
export KUPL_SHM_ENABLE_HUGEPAGE=y
# Kunpeng HBW pool
export SGLANG_ENABLE_HBW_POOL=1
export SGLANG_KUNPENG_MEMORY_ALIGNMENT=4096
export SGLANG_KUNPENG_WEIGTHS_HBW_POOL_SIZE_MB=3400
export SGLANG_KUNPENG_SWAP_KV_IN=0
export SGLANG_KUNPENG_SWAP_KV_OUT=0
export SGLANG_KUNPENG_SWAP_KV_BLOCKWISE=0
export SGLANG_KUNPENG_SWAP_MAX_KV_BLOCKS=512
# Kunpeng SDMA parameters
export SGLANG_KUNPENG_SDMA_MAX_EVENTS=10
export SGLANG_KUNPENG_SDMA_THRESHOLD=5
# Kunpeng graph capture
export SGLANG_ENABLE_GRAPH_CAPTURE=1
export SGLANG_ENABLE_GRAPH_PROFILE=0
export SGLANG_KUNPENG_GRAPH_CACHE_SIZE=10
export SGLANG_KUNPENG_EXTEND_POWER_2_PADDING=1 # Kunpeng prefill graph padding to power 2 size
# Load format (e.g. "kunpeng_state", leave empty for default)
export LOAD_FORMAT=""
# Drop OS page cache (echo 3 > /proc/sys/vm/drop_caches) during stop.sh node
export DROP_CACHES=0
# Tokenizer-side cross-process batch timeline logging
export SGLANG_TOKENIZER_TIMELINE_LOG=0
# Scheduler stream interval (--stream-interval): flush a request's output every N tokens
export STREAM_INTERVAL=1
# Tokenizer backend: "huggingface" (default) or "fastokens" (fastokens requires transformers >= 5.12)
export SGLANG_TOKENIZER_BACKEND="huggingface"
# Tokenizer-separate mode: on by default for PD roles (prefill/decode/router), off for native.
# .user_env.sh can still override.
if [[ "${1:-native}" == "native" ]]; then
    export SGLANG_ENABLE_TOKENIZER_SEPERATE=0
else
    export SGLANG_ENABLE_TOKENIZER_SEPERATE=1
fi

export SGLANG_KUNPENG_MOE_TOKEN_MULTIPLE=2

# ------------------------------------------------------------
# Load local config
# ------------------------------------------------------------
USER_ENV_ROLE="${1:-native}"

case "$USER_ENV_ROLE" in
    decode|router|tokenizer) IS_PREFILL=0 ;;
    *) IS_PREFILL=1 ;;
esac
export IS_PREFILL

if [[ -f "$SCRIPT_DIR/.user_env.sh" ]]; then
    source "$SCRIPT_DIR/.user_env.sh" "$USER_ENV_ROLE"
fi

if [[ -f "$SCRIPT_DIR/runtime/.user_env_base.sh" ]]; then
    source "$SCRIPT_DIR/runtime/.user_env_base.sh" "$USER_ENV_ROLE"
fi
