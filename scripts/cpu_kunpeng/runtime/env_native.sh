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
# Native role config: role defaults for the non-disaggregated (single-server)
# mode. Sourced by env.sh (after env_base.sh, so .user_env.sh overrides set
# there are respected via the ${VAR:-default} fallbacks below).
# Node topology and parallel sizes are NOT set here: native_config in
# env_helper.sh reads the NATIVE_* vars plus the global TP_SIZE/DP_SIZE/
# EP_SIZE/PP_SIZE from env_base.sh. Native runs prefill and decode in one
# server, so it inherits the prefill-side role defaults below.
# Per-server overrides: runtime/.user_env_native.sh (sourced last).

# ------------------------------------------------------------
# Native role defaults (prefill-side values; native previously inherited
# these from env_prefill.sh)
# ------------------------------------------------------------
# Per-DP-rank chunked prefill size; server.sh multiplies it by DP_SIZE
# for the global --chunked-prefill-size.
export CHUNKED_PREFILL_SIZE_PER_DP="${CHUNKED_PREFILL_SIZE_PER_DP:-4096}"

export SGLANG_KUNPENG_MAX_SEQ_NUM="${SGLANG_KUNPENG_MAX_SEQ_NUM:-8}"
export SGLANG_KUNPENG_MAX_CUR_LEN="${SGLANG_KUNPENG_MAX_CUR_LEN:-576}"
export SGLANG_KUNPENG_MAX_SEQ_LEN="${SGLANG_KUNPENG_MAX_SEQ_LEN:-65536}"
export SGLANG_KUNPENG_SWAP_EXPERT="${SGLANG_KUNPENG_SWAP_EXPERT:-1}"

# ------------------------------------------------------------
# Native SHM / HBW pool
# ------------------------------------------------------------
export SGLANG_KUNPENG_PREFILL_SHM_SIZE_MB="${SGLANG_KUNPENG_PREFILL_SHM_SIZE_MB:-476}"

# Per-server overrides (sourced after role defaults so they take priority).
# With an instance (env.sh native 128p -> INSTANCE=128p), ONLY the instance
# file .user_env_native_<instance>.sh is loaded; the default file
# .user_env_native.sh is loaded only when no instance is given.
if [[ -n "${INSTANCE:-}" ]]; then
    if [[ -f "$SCRIPT_DIR/runtime/.user_env_native_${INSTANCE}.sh" ]]; then
        source "$SCRIPT_DIR/runtime/.user_env_native_${INSTANCE}.sh"
    else
        echo "ERROR: instance file runtime/.user_env_native_${INSTANCE}.sh not found (required when an instance is specified)" >&2
        return 1 2>/dev/null || exit 1
    fi
elif [[ -f "$SCRIPT_DIR/runtime/.user_env_native.sh" ]]; then
    source "$SCRIPT_DIR/runtime/.user_env_native.sh"
else
    echo "WARNING: runtime/.user_env_native.sh not found, using role defaults only" >&2
fi
