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
#   router                                       -> runtime/stop_router.sh
#   tokenizer [prefill|decode|all]               -> runtime/stop_tokenizer.sh
#   all    -> router + tokenizer(both) + server(per INSTANCES in env_base.sh)

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
  router    Kill the gateway on the router node
  tokenizer Kill tokenizer HTTP server(s) (side: prefill|decode|all, default all)
  all       Stop router + both tokenizer sides + servers per INSTANCES

Options:
  -h, --help    show this help and exit

Examples:
  $0 server decode
  $0 server decode 128p
  $0 tokenizer prefill
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
    tokenizer)
        bash "$SCRIPT_DIR/runtime/stop_tokenizer.sh" "${1:-all}"
        exit $?
        ;;
    all)
        bash "$SCRIPT_DIR/runtime/stop_router.sh"
        bash "$SCRIPT_DIR/runtime/stop_tokenizer.sh" all
        # Stop servers per the deployment's instance list (same parsing
        # as launch.sh all): entries "<role>" or "<role>_<instance>".
        IFS=',' read -ra _INST_LIST <<< "$INSTANCES"
        for _entry in "${_INST_LIST[@]}"; do
            _entry="${_entry//[[:space:]]/}"
            [[ -z "$_entry" ]] && continue
            _role="${_entry%%_*}"
            _inst=""
            [[ "$_entry" == *_* ]] && _inst="${_entry#*_}"
            bash "$SCRIPT_DIR/runtime/stop_server.sh" "$_role" "$_inst"
        done
        exit $?
        ;;
    *)
        echo "Error: unknown target '$TARGET'" >&2
        show_usage >&2
        exit 1
        ;;
esac
