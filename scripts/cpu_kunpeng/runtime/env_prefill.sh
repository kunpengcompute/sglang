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
# Prefill role config: cluster topology, parallel sizes and role defaults.
# Sourced by env.sh (after env_base.sh, so .user_env.sh overrides set there
# are respected via the ${VAR:-default} fallbacks below).
# Per-server overrides: runtime/.user_env_prefill.sh (sourced last).

# ------------------------------------------------------------
# Prefill cluster topology
# ------------------------------------------------------------
# IP_FILE: file with one IP per line (e.g. "192.168.1.1" per line)
# IP_SPEC: alternative range notation, e.g. "192.168.1. | 1-3,5" (used only when IP_FILE is empty)
PREFILL_IP_SPEC="${PREFILL_IP_SPEC:-}"
PREFILL_IP_FILE="${PREFILL_IP_FILE:-}"
PREFILL_MASTER_ADDR="${PREFILL_MASTER_ADDR:-xxx.xxx.xxx.1}"
PREFILL_MASTER_PORT="${PREFILL_MASTER_PORT:-5000}"

# Prefill model paths (default to the shared MODEL_PATH)
MODEL_PATH_PREFILL="${MODEL_PATH_PREFILL:-$MODEL_PATH}"
SPECULATIVE_DRAFT_MODEL_PATH_PREFILL="${SPECULATIVE_DRAFT_MODEL_PATH_PREFILL:-$SPECULATIVE_DRAFT_MODEL_PATH}"

# ------------------------------------------------------------
# Prefill TP/EP/PP sizes (default to the global values from env_base)
# ------------------------------------------------------------
export PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-${TP_SIZE}}"
export PREFILL_DP_SIZE="${PREFILL_DP_SIZE:-${DP_SIZE}}"
export PREFILL_EP_SIZE="${PREFILL_EP_SIZE:-${PREFILL_TP_SIZE}}"
export PREFILL_PP_SIZE="${PREFILL_PP_SIZE:-${PP_SIZE}}"
export PREFILL_REDUNDANT_EXPERTS="${PREFILL_REDUNDANT_EXPERTS:-0}"
export PREFILL_INIT_EXPERT_LOCATION="${PREFILL_INIT_EXPERT_LOCATION:-}"
export PREFILL_EP_DISPATCH_ALGORITHM="${PREFILL_EP_DISPATCH_ALGORITHM:-}"
export PREFILL_SGLANG_KUNPENG_MOE_SHUFFLE_MODE="${PREFILL_SGLANG_KUNPENG_MOE_SHUFFLE_MODE:-0}"


# Per-DP-rank chunked prefill size; server.sh multiplies it by DP_SIZE
# for the global --chunked-prefill-size.
export CHUNKED_PREFILL_SIZE_PER_DP=4096

export LONG_PROMPT_PREFILL_INSTANCE=0
if [[ $LONG_PROMPT_PREFILL_INSTANCE == "1" ]]; then
    export PREFILL_TP_SIZE=16
    export PREFILL_DP_SIZE=1
    export PREFILL_EP_SIZE="${PREFILL_TP_SIZE}"
    export PREFILL_PP_SIZE=16
fi

# ------------------------------------------------------------
# Prefill SHM / HBW pool
# ------------------------------------------------------------
export SGLANG_KUNPENG_PREFILL_SHM_SIZE_MB="${SGLANG_KUNPENG_PREFILL_SHM_SIZE_MB:-476}"
export PREFILL_WEIGTHS_HBW_POOL_SIZE_MB="${PREFILL_WEIGTHS_HBW_POOL_SIZE_MB:-3400}"
export PREFILL_SWAP_KV_IN="${PREFILL_SWAP_KV_IN:-0}"
export PREFILL_SWAP_KV_OUT="${PREFILL_SWAP_KV_OUT:-0}"
export PREFILL_SWAP_KV_BLOCKWISE="${PREFILL_SWAP_KV_BLOCKWISE:-0}"

# ------------------------------------------------------------
# Prefill role defaults
# ------------------------------------------------------------
export SGLANG_KUNPENG_SWAP_EXPERT="${SGLANG_KUNPENG_SWAP_EXPERT:-1}"
export SGLANG_KUNPENG_MAX_SEQ_NUM="${SGLANG_KUNPENG_MAX_SEQ_NUM:-8}"
export SGLANG_KUNPENG_MAX_CUR_LEN="${SGLANG_KUNPENG_MAX_CUR_LEN:-576}"
export SGLANG_KUNPENG_MAX_SEQ_LEN="${SGLANG_KUNPENG_MAX_SEQ_LEN:-65536}"

# Per-server overrides (sourced after role defaults so they take priority).
# With an instance (env.sh prefill 128p -> INSTANCE=128p), ONLY the instance
# file .user_env_prefill_<instance>.sh is loaded; the default file
# .user_env_prefill.sh is loaded only when no instance is given.
if [[ -n "${INSTANCE:-}" ]]; then
    if [[ -f "$SCRIPT_DIR/runtime/.user_env_prefill_${INSTANCE}.sh" ]]; then
        source "$SCRIPT_DIR/runtime/.user_env_prefill_${INSTANCE}.sh"
    else
        echo "ERROR: instance file runtime/.user_env_prefill_${INSTANCE}.sh not found (required when an instance is specified)" >&2
        return 1 2>/dev/null || exit 1
    fi
elif [[ -f "$SCRIPT_DIR/runtime/.user_env_prefill.sh" ]]; then
    source "$SCRIPT_DIR/runtime/.user_env_prefill.sh"
else
    echo "WARNING: runtime/.user_env_prefill.sh not found, using role defaults only" >&2
fi
