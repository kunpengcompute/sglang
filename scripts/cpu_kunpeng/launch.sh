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
#   all                   -> inline: prefill + decode + health-check + router
#   update                -> inline: update_time + update_numa_dup

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_usage() {
    cat <<EOF
Usage: $0 <role> [args...]

Roles:
  prefill    Launch prefill server (PD disaggregation, prefill side)
  decode     Launch decode server (PD disaggregation, decode side)
  native     Launch without PD disaggregation
  router     Launch gateway (route requests to prefill/decode)
  tokenizer  Launch tokenizer HTTP servers (prefill:30001 + decode:30002)
  all        Launch prefill, decode, tokenizer, and router sequentially
  update     Regenerate .time_env.sh + update NUMA binary replicas

Options:
  -h, --help    show this help and exit

Examples:
  $0 prefill
  $0 all
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
    prefill|decode|native|router|tokenizer|all|update)
        ROLE="$CMD"
        ;;
    *)
        echo "Error: unknown command '$CMD'" >&2
        show_usage >&2
        exit 1
        ;;
esac


bash "$SCRIPT_DIR/runtime/update_time.sh"
if [[ "$ROLE" == "router" ]]; then
    echo -e "\033[33m[$(date +%T)] Skipping NUMA binary update for role 'router'.\033[0m"
else
    bash "$SCRIPT_DIR/runtime/update_numa_dup.sh"
fi

if [[ "$ROLE" == "update" ]]; then
    exit 0

elif [[ "$ROLE" == "all" ]]; then
    echo "[$(date +%T)] ===== Launching all roles (prefill + decode + tokenizer + router) ====="

    SKIP_LOG=1 bash "$SCRIPT_DIR/runtime/launch_cluster.sh" prefill
    SKIP_LOG=1 bash "$SCRIPT_DIR/runtime/launch_cluster.sh" decode
    SKIP_LOG=1 bash "$SCRIPT_DIR/runtime/launch_tokenizer.sh"
    bash "$SCRIPT_DIR/runtime/launch_router.sh"

elif [[ "$ROLE" == "router" ]]; then
    bash "$SCRIPT_DIR/runtime/launch_router.sh" "$@"

elif [[ "$ROLE" == "tokenizer" ]]; then
    bash "$SCRIPT_DIR/runtime/launch_tokenizer.sh" "$@"

else
    # prefill/decode/native
    bash "$SCRIPT_DIR/runtime/launch_cluster.sh" "$ROLE" "$@"
fi
