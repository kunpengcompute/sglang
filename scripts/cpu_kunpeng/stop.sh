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
# stop.sh - Task dispatcher for cpu_kunpeng stop scripts.
# Pure dispatcher: delegates to the matching sub-script under runtime/.
# Usage: ./stop.sh <target> [args]
#   server    [prefill|decode|native] [instance] -> runtime/stop_server.sh
#   <role>    native|prefill|decode (shorthand for "server <role>")
#   router                                       -> runtime/stop_router.sh
#   tokenizer [prefill|decode|all]               -> runtime/stop_tokenizer.sh
#   mooncake                                     -> runtime/stop_mooncake.sh
#   store                                        -> runtime/stop_store.sh
#   all    -> router + tokenizer(both) + server(per INSTANCES in env_base.sh)
#             + store + mooncake

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load base user overrides (INSTANCES etc.) early so the dispatch logic
# below can read them. "none" mode = base config only: no role config,
# no conda activation, no toolchain paths — children set those up via
# their own env.sh sourcing.
source "$SCRIPT_DIR/env.sh" none

show_usage() {
    cat <<EOF
Usage: $0 <target> [side]

Targets:
  server    Kill sglang on the role's cluster nodes (side: prefill|decode|native, optional instance)
  <role>    native|prefill|decode: shorthand for "server <role>"
  router    Kill the gateway on the router node
  tokenizer Kill tokenizer HTTP server(s) (side: prefill|decode|all, default all)
  mooncake  Kill the mooncake store master on MOONCAKE_MASTER_NODE
  store     Kill the mooncake store service on MOONCAKE_STORE_NODE
  all       Stop router + both tokenizer sides + servers per INSTANCES + store + mooncake

Options:
  -h, --help    show this help and exit

Examples:
  $0 native
  $0 server decode
  $0 server decode 128p
  $0 tokenizer prefill
  $0 mooncake
  $0 store
  $0 all
EOF
}

if [[ $# -eq 0 ]]; then
    # Default: stop everything
    set -- all
fi

TARGET="$1"
shift

case "$TARGET" in
    -h|--help|help)
        show_usage
        exit 0
        ;;
    server|router)
        bash "$SCRIPT_DIR/runtime/stop_${TARGET}.sh" "$@"
        exit $?
        ;;
    # Shorthand: "stop.sh <role>" == "stop.sh server <role>"
    native|prefill|decode)
        bash "$SCRIPT_DIR/runtime/stop_server.sh" "$TARGET" "$@"
        exit $?
        ;;
    tokenizer)
        bash "$SCRIPT_DIR/runtime/stop_tokenizer.sh" "${1:-all}"
        exit $?
        ;;
    mooncake)
        bash "$SCRIPT_DIR/runtime/stop_mooncake.sh"
        exit $?
        ;;
    store)
        bash "$SCRIPT_DIR/runtime/stop_store.sh"
        exit $?
        ;;
    all)
        bash "$SCRIPT_DIR/runtime/stop_router.sh"
        bash "$SCRIPT_DIR/runtime/stop_tokenizer.sh" all
        # Stop servers per the deployment's instance list (same parsing
        # as launch.sh all): entries "<role>" or "<role>_<instance>".
        IFS=',' read -ra _INST_LIST <<< "$INSTANCES"
        _stop_mooncake=0
        for _entry in "${_INST_LIST[@]}"; do
            _entry="${_entry//[[:space:]]/}"
            [[ -z "$_entry" ]] && continue
            _role="${_entry%%_*}"
            _inst=""
            [[ "$_entry" == *_* ]] && _inst="${_entry#*_}"
            bash "$SCRIPT_DIR/runtime/stop_server.sh" "$_role" "$_inst"
            [[ "$_role" == "prefill" || "$_role" == "native" ]] || continue
            _hicache="$(SKIP_CONDA=1 bash -c "
                source '$SCRIPT_DIR/env.sh' '$_role' '$_inst' >/dev/null 2>&1
                echo \"\${ENABLE_KUNPENG_HICACHE:-0}\"")"
            [[ "$_hicache" != "1" ]] && continue
            _backend="$(SKIP_CONDA=1 bash -c "
                source '$SCRIPT_DIR/env.sh' '$_role' '$_inst' >/dev/null 2>&1
                echo \"\${KUNPENG_HICACHE_BACKEND:-file}\"")"
            [[ "$_backend" == "mooncake" ]] && _stop_mooncake=1
        done
        # Last: the prefill servers deregister from the store on the way down.
        if [[ "$_stop_mooncake" == "1" ]]; then
            bash "$SCRIPT_DIR/runtime/stop_store.sh"
            bash "$SCRIPT_DIR/runtime/stop_mooncake.sh"
        fi
        exit $?
        ;;
    *)
        echo "Error: unknown target '$TARGET'" >&2
        show_usage >&2
        exit 1
        ;;
esac
