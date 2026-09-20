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
# Usage: source env.sh [prefill|decode|native|router|tokenizer|build|none] [instance]
#   instance (e.g. "128p") selects per-instance user env overrides
#   (runtime/.user_env_<role>_<instance>.sh); empty = default instance.
#   For tokenizer, the instance arg is "<side>[_<instance>]" (e.g.
#   "prefill", "decode_128p"): only that side's role env is loaded.
#   "none" loads only the base env (incl. .user_env.sh) and returns
#   immediately — no role config, no toolchain/conda setup.

ACTION="${1:-native}"
INSTANCE="${2:-}"
export INSTANCE

# Source base environment: configuration variables, .user_env.sh loading,
# helper functions, etc. (sets SCRIPT_DIR and IS_PREFILL too)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runtime/env_base.sh"

# "none" mode: base config only (for reading INSTANCES etc. from
# .user_env.sh); skip role config, conda activation, and all path setup.
if [[ "$ACTION" == "none" ]]; then
    return 0 2>/dev/null || exit 0
fi

# Role-dependent config + defaults.
# prefill/native/build -> prefill config; decode -> decode config.
# router/tokenizer resolve each instance's role env themselves (per-entry
# subshells in launch_router.sh / server_router.sh / launch_tokenizer.sh),
# so they only load the base env here — no prefill/decode exports leak
# into the control shell.
case "$ACTION" in
    prefill|native|build)
        source "$SCRIPT_DIR/runtime/env_prefill.sh" || return 1
        ;;
    router)
        : # base env only; backends are resolved per INSTANCES entry
        ;;
    tokenizer)
        _tok_side="${INSTANCE%%_*}"
        if [[ "$INSTANCE" == *_* ]]; then
            INSTANCE="${INSTANCE#*_}"
        else
            INSTANCE=""
        fi
        if [[ "$_tok_side" == "prefill" ]]; then
            source "$SCRIPT_DIR/runtime/env_prefill.sh" || return 1
        else
            source "$SCRIPT_DIR/runtime/env_decode.sh" || return 1
        fi
        ;;
    *)
        source "$SCRIPT_DIR/runtime/env_decode.sh" || return 1
        ;;
esac

# Optional: set SKIP_CONDA=1 before sourcing (SKIP_CONDA=1 source env.sh) to skip conda activation
SKIP_CONDA="${SKIP_CONDA:-0}"

case "$ACTION" in
    prefill|decode|native|router|tokenizer|build)
        "${ACTION}_config"
        ;;
    *)
        echo "Usage: source env.sh [prefill|decode|native|router|tokenizer|build]" >&2
        return 1
        ;;
esac

if [[ "$ACTION" == "tokenizer" && "$SGLANG_ENABLE_TOKENIZER_SEPERATE" == "1" ]]; then
    export RAYON_NUM_THREADS=16
    export SGLANG_SET_ZMQ_CPU_AFFINITY_OFFSET=17
    # Router tokenizer parent/workers are pinned to half-NUMA slices: resolve
    # the ZMQ offset as an index into each process's own slice (not absolute).
    export SGLANG_SET_ZMQ_CPU_AFFINITY_SLICE_RELATIVE=1
fi

# Static routing (matches DeepSeek-V3-Sample 64p/128p-decode toml
# enable_static_routing=1): enabled for decode when DP size >= 32
# (tp8*dp32 pp2 / tp8*dp64 pp2).
if [[ "$IS_PREFILL" != "1" && "${DP_SIZE:-0}" -ge 32 ]]; then
    export ENABLE_STATIC_ROUTING="${ENABLE_STATIC_ROUTING:-0}"
else
    export ENABLE_STATIC_ROUTING="${ENABLE_STATIC_ROUTING:-0}"
fi

if [[ ! -f "${SCRIPT_DIR}/.time_env.sh" ]]; then
    bash "${SCRIPT_DIR}/runtime/update_time.sh"
fi
source "${SCRIPT_DIR}/.time_env.sh"
# Instance-qualified log subdir ("prefill", "prefill_128p", ...) so
# same-role instances launched in one `launch.sh all` run never share a
# log directory (ssh logs, rank logs and "latest" symlinks stay per-instance).
# Exception: tokenizer — all its log files already carry the side/instance
# in their names, so one shared "tokenizer/" dir is enough.
if [[ "$ACTION" == "tokenizer" ]]; then
    export LOG_SUBDIR="tokenizer"
else
    export LOG_SUBDIR="$ACTION${INSTANCE:+_$INSTANCE}"
fi
export LOG_DIR="${LOG_BASE_DIR}/${LOG_DATE}/${LOG_SUBDIR}/${LOG_TIME}"
export SGLANG_TORCH_PROFILER_DIR="${LOG_DIR}/torch_profiler"

if [[ "$SGLANG_ENABLE_NUMA_DUPLICATION" != "1" ]] || [[ "$ACTION" == "router" ]] || [[ "$ACTION" == "tokenizer" ]] || [[ "$ACTION" == "build" ]]; then
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


# Kuccl backend (UCX + UCG) environment
if [[ "$SGLANG_ENABLE_KUCCL" == "1" ]]; then
    export UCG_PLANC_UCX_BCAST_ATTR=I:1
    export UCX_MEM_EVENTS=no
    export UCX_UD_VERBS_ALLOC=thp,md,mmap,heap
    export UCX_RC_VERBS_ALLOC=thp,md,mmap,heap

    if [[ "$SGLANG_ENABLE_NUMA_DUPLICATION" != "1" ]]; then
        export HUCX_DIR="${HPCKIT_PATH}/26.1.RC1/hmpi/bisheng/release/hucx"
        export XUCG_DIR="${HPCKIT_PATH}/26.1.RC1/hmpi/bisheng/release/xucg"
        export UCX_MODULE_DIR="${HUCX_DIR}/lib/ucx"
        export UCG_PLANC=ucx
        export UCG_PLANC_PATH="${XUCG_DIR}/lib/planc"
        export LD_LIBRARY_PATH="${HUCX_DIR}/lib:${XUCG_DIR}/lib:${XUCG_DIR}/lib/planc:${LD_LIBRARY_PATH:-}"
        export PYTHONPATH="${KUCCL_PATH}:${PYTHONPATH:-}"
    fi
fi

export CONDA_ACTIVATE_CMD="eval \"\$($CONDA_BASE_PATH/bin/conda shell.bash hook)\" && conda activate $CONDA_ENV_NAME"
if [[ "$SKIP_CONDA" != "1" ]]; then
    eval "$CONDA_ACTIVATE_CMD"
    echo "conda environment activated: $CONDA_ENV_NAME"
fi
