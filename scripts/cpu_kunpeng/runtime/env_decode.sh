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
# Decode role config: cluster topology, parallel sizes and role defaults.
# Sourced by env.sh (after env_base.sh, so .user_env.sh overrides set there
# are respected via the ${VAR:-default} fallbacks below).
# Per-server overrides: runtime/.user_env_decode.sh (sourced last).

# ------------------------------------------------------------
# Decode cluster topology
# ------------------------------------------------------------
# IP_FILE: file with one IP per line (e.g. "192.168.1.1" per line)
# IP_SPEC: alternative range notation, e.g. "192.168.1. | 1-3,5" (used only when IP_FILE is empty)
DECODE_IP_SPEC="${DECODE_IP_SPEC:-}"
DECODE_IP_FILE="${DECODE_IP_FILE:-}"
DECODE_MASTER_ADDR="${DECODE_MASTER_ADDR:-xxx.xxx.xxx.17}"
DECODE_MASTER_PORT="${DECODE_MASTER_PORT:-5010}"

# Decode model paths (default to the shared MODEL_PATH)
MODEL_PATH_DECODE="${MODEL_PATH_DECODE:-$MODEL_PATH}"
SPECULATIVE_DRAFT_MODEL_PATH_DECODE="${SPECULATIVE_DRAFT_MODEL_PATH_DECODE:-$SPECULATIVE_DRAFT_MODEL_PATH}"

# ------------------------------------------------------------
# Decode TP/EP/PP sizes (default to the global values from env_base)
# ------------------------------------------------------------
export DECODE_TP_SIZE="${DECODE_TP_SIZE:-${TP_SIZE}}"
export DECODE_DP_SIZE="${DECODE_DP_SIZE:-${DP_SIZE}}"
export DECODE_EP_SIZE="${DECODE_EP_SIZE:-${DECODE_TP_SIZE}}"
export DECODE_PP_SIZE="${DECODE_PP_SIZE:-${PP_SIZE}}"
export DECODE_REDUNDANT_EXPERTS="${DECODE_REDUNDANT_EXPERTS:-0}"
export DECODE_INIT_EXPERT_LOCATION="${DECODE_INIT_EXPERT_LOCATION:-}"
export DECODE_EP_DISPATCH_ALGORITHM="${DECODE_EP_DISPATCH_ALGORITHM:-}"
export DECODE_SGLANG_KUNPENG_MOE_SHUFFLE_MODE="${DECODE_SGLANG_KUNPENG_MOE_SHUFFLE_MODE:-0}"

# ------------------------------------------------------------
# Decode SHM / HBW pool
# ------------------------------------------------------------
export SGLANG_KUNPENG_DECODE_SHM_SIZE_MB="${SGLANG_KUNPENG_DECODE_SHM_SIZE_MB:-100}"
export DECODE_WEIGTHS_HBW_POOL_SIZE_MB="${DECODE_WEIGTHS_HBW_POOL_SIZE_MB:-3900}"
export DECODE_SWAP_KV_IN="${DECODE_SWAP_KV_IN:-0}"
export DECODE_SWAP_KV_OUT="${DECODE_SWAP_KV_OUT:-0}"
export DECODE_SWAP_KV_BLOCKWISE="${DECODE_SWAP_KV_BLOCKWISE:-0}"

# ------------------------------------------------------------
# Decode role defaults
# ------------------------------------------------------------
# Equivalent to MAX_SEQ_NUM / PP_SIZE in DeepSeek-V3-Sample (max_seq_num_per_mb):
export SGLANG_KUNPENG_MAX_SEQ_NUM="${SGLANG_KUNPENG_MAX_SEQ_NUM:-64}"
# Decode MTP: a verify micro-batch holds (root + speculative_num_steps
# draft) tokens per sequence, so the per-sequence in-flight width is
# speculative_num_steps + 1.
if [[ -z "${SGLANG_KUNPENG_MAX_CUR_LEN:-}" ]]; then
    if [[ "$SGLANG_ENABLE_MTP" == "1" ]]; then
        export SGLANG_KUNPENG_MAX_CUR_LEN=$(( ${SGLANG_SPECULATIVE_NUM_STEPS:-2} + 1 ))
    else
        export SGLANG_KUNPENG_MAX_CUR_LEN=1
    fi
fi

# Per-server overrides (sourced after role defaults so they take priority).
# With an instance (env.sh decode 128p -> INSTANCE=128p), ONLY the instance
# file .user_env_decode_<instance>.sh is loaded; the default file
# .user_env_decode.sh is loaded only when no instance is given.
if [[ -n "${INSTANCE:-}" ]]; then
    if [[ -f "$SCRIPT_DIR/runtime/.user_env_decode_${INSTANCE}.sh" ]]; then
        source "$SCRIPT_DIR/runtime/.user_env_decode_${INSTANCE}.sh"
    else
        echo "ERROR: instance file runtime/.user_env_decode_${INSTANCE}.sh not found (required when an instance is specified)" >&2
        return 1 2>/dev/null || exit 1
    fi
elif [[ -f "$SCRIPT_DIR/runtime/.user_env_decode.sh" ]]; then
    source "$SCRIPT_DIR/runtime/.user_env_decode.sh"
else
    echo "WARNING: runtime/.user_env_decode.sh not found, using role defaults only" >&2
fi
