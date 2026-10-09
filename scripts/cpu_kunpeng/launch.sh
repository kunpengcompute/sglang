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
# launch.sh - Task dispatcher for cpu_kunpeng scripts.
# Usage: ./launch.sh <role> [args...]
#   prefill/decode/native -> runtime/launch_cluster.sh
#   router                -> runtime/launch_router.sh
#   mooncake              -> runtime/launch_mooncake.sh (HiCache L3 store master)
#   store                 -> runtime/launch_store.sh (HiCache L3 store service)
#   all                   -> inline: [mooncake + store] + prefill + decode + health-check
#                            + router
#   update                -> inline: update_time + update_numa_dup

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load base user overrides (INSTANCES etc.) early so the dispatch logic
# below can read them. "none" mode = base config only: no role config,
# no conda activation, no toolchain paths — children set those up via
# their own env.sh sourcing.
source "$SCRIPT_DIR/env.sh" none

show_usage() {
    cat <<EOF
Usage: $0 <role> [args...]

Roles:
  prefill    Launch prefill server (PD disaggregation, prefill side)
  decode     Launch decode server (PD disaggregation, decode side)
  native     Launch without PD disaggregation
  router     Launch gateway (route requests to prefill/decode)
  tokenizer  Launch one tokenizer HTTP server: $0 tokenizer <prefill|decode> [instance]
             (port auto-derived from the entry's position in INSTANCES,
             default 30001/30002; instance selects
             runtime/.user_env_<side>_<instance>.sh)
  mooncake   Launch the mooncake store master (HiCache L3 backend mooncake)
  store      Launch the mooncake store service on MOONCAKE_STORE_NODE (L3 pool)
  all        Launch prefill, decode, tokenizer, and router sequentially
             (the mooncake master + store service first, when a prefill entry
             selects the mooncake backend)
  update     Regenerate .time_env.sh + update NUMA binary replicas

Options:
  -h, --help    show this help and exit

Examples:
  $0 prefill
  $0 all
  $0 mooncake
  $0 store
  $0 update
EOF
}

if [[ $# -eq 0 ]]; then
    echo "Error: missing command" >&2
    show_usage >&2
    exit 1
fi

CMD="$1"
shift

case "$CMD" in
    -h|--help|help)
        show_usage
        exit 0
        ;;
    prefill|decode|native|router|tokenizer|mooncake|store|all|update)
        ROLE="$CMD"
        ;;
    *)
        echo "Error: unknown command '$CMD'" >&2
        show_usage >&2
        exit 1
        ;;
esac

role_uses_mooncake() {
    local _role="${1:-}" _inst="${2:-}" _sel
    [[ "$_role" == "prefill" || "$_role" == "native" ]] || return 1
    _sel="$(SKIP_CONDA=1 bash -c '
        source "$1/env.sh" "$2" "$3" >/dev/null 2>&1
        echo "${ENABLE_KUNPENG_HICACHE:-0}/${KUNPENG_HICACHE_BACKEND:-file}"' _ "$SCRIPT_DIR" "$_role" "$_inst")"
    [[ "$_sel" == "1/mooncake" ]]
}

start_mooncake_store() {
    SKIP_LOG=1 bash "$SCRIPT_DIR/runtime/stop_store.sh"
    SKIP_LOG=1 bash "$SCRIPT_DIR/runtime/launch_mooncake.sh"
    SKIP_LOG=1 bash "$SCRIPT_DIR/runtime/launch_store.sh"
}

bash "$SCRIPT_DIR/runtime/update_time.sh"
if [[ "$ROLE" == "router" || "$ROLE" == "tokenizer" || "$ROLE" == "mooncake" || "$ROLE" == "store" ]]; then
    echo -e "\033[33m[$(date +%T)] Skipping NUMA binary update for role '$ROLE'.\033[0m"
elif [[ "${SGLANG_ENABLE_NUMA_DUPLICATION:-1}" != "1" ]]; then
    echo -e "\033[33m[$(date +%T)] Skipping NUMA binary update (SGLANG_ENABLE_NUMA_DUPLICATION='${SGLANG_ENABLE_NUMA_DUPLICATION:-unset}').\033[0m"
else
    bash "$SCRIPT_DIR/runtime/update_numa_dup.sh"
fi

if [[ "$ROLE" == "update" ]]; then
    exit 0

elif [[ "$ROLE" == "all" ]]; then
    # Deployment instance list: comma-separated entries, each "<role>" or
    # "<role>_<instance>" (e.g. "prefill,decode_128p" -> prefill + decode
    # instance 128p; same-role multi-instance also supported, e.g.
    # "prefill,prefill_lc,decode" — see runtime/env_base.sh for the
    # per-instance port/NUMA requirements). Default lives in
    # runtime/env_base.sh; .user_env.sh (sourced via "env.sh none" at the
    # top) may override it.
    echo "[$(date +%T)] ===== Launching all roles ($INSTANCES + tokenizer + router) ====="

    IFS=',' read -ra _INST_LIST <<< "$INSTANCES"

    # HiCache L3 = mooncake: the store master has to be up before the prefill
    # servers register their L2 host pools with it. The backend is chosen in the
    # prefill entry's own role env, so resolve it there (a subshell, like
    # launch_router.sh does for the master addresses).
    _start_mooncake=0
    for _entry in "${_INST_LIST[@]}"; do
        _entry="${_entry//[[:space:]]/}"
        [[ -z "$_entry" ]] && continue
        _role="${_entry%%_*}"
        _inst=""
        [[ "$_entry" == *_* ]] && _inst="${_entry#*_}"
        role_uses_mooncake "$_role" "$_inst" && _start_mooncake=1
    done
    [[ "$_start_mooncake" == "1" ]] && start_mooncake_store

    for _entry in "${_INST_LIST[@]}"; do
        _entry="${_entry//[[:space:]]/}"
        [[ -z "$_entry" ]] && continue
        _role="${_entry%%_*}"   # "decode_128p" -> "decode"; "native" -> "native"
        _inst=""
        [[ "$_entry" == *_* ]] && _inst="${_entry#*_}"
        SKIP_LOG=1 bash "$SCRIPT_DIR/runtime/launch_cluster.sh" "$_role" "$_inst"
    done
    # Tokenizers follow the same instance list: one per prefill/decode
    # entry, loading that entry's instance env (master addr, DP size).
    for _entry in "${_INST_LIST[@]}"; do
        _entry="${_entry//[[:space:]]/}"
        [[ -z "$_entry" ]] && continue
        _role="${_entry%%_*}"
        _inst=""
        [[ "$_entry" == *_* ]] && _inst="${_entry#*_}"
        [[ "$_role" == "prefill" || "$_role" == "decode" ]] || continue
        SKIP_LOG=1 bash "$SCRIPT_DIR/runtime/launch_tokenizer.sh" "$_role" "$_inst"
    done
    bash "$SCRIPT_DIR/runtime/launch_router.sh"

elif [[ "$ROLE" == "router" ]]; then
    bash "$SCRIPT_DIR/runtime/launch_router.sh" "$@"

elif [[ "$ROLE" == "tokenizer" ]]; then
    bash "$SCRIPT_DIR/runtime/launch_tokenizer.sh" "$@"

elif [[ "$ROLE" == "mooncake" ]]; then
    bash "$SCRIPT_DIR/runtime/launch_mooncake.sh" "$@"

elif [[ "$ROLE" == "store" ]]; then
    bash "$SCRIPT_DIR/runtime/launch_store.sh" "$@"

else
    # prefill/decode/native
    role_uses_mooncake "$ROLE" "$@" && start_mooncake_store
    bash "$SCRIPT_DIR/runtime/launch_cluster.sh" "$ROLE" "$@"
fi
