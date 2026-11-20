#!/usr/bin/env bash
set -euo pipefail
export QWEN_KIT
QWEN_KIT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
action=${1:-help}
if [[ "$action" == help ]]; then
  echo 'Usage: bash run.sh build|pack|plan|deps|verify|check|start|request|bench|status|stop|prune /path/to/config.sh [--instance INSTANCE_ID | --apply]'
  echo 'QWEN_DEPLOY_MODE=single|pd in config selects node or PD cluster lifecycle; default: single.'
  exit 0
fi
export QWEN_CONFIG_PATH
QWEN_CONFIG_PATH=$(realpath "${2:-$QWEN_KIT/config.sh}")
while IFS= read -r key; do
  case "$key" in SGLANG_*|KUTACC_*|KUPL_*) unset "$key" ;; esac
done < <(compgen -e)
unset PYTHONHOME PYTHONPATH LD_PRELOAD
unset QWEN_PAYLOAD_ROOT
# Ownership and topology must come from this invocation's configuration.
unset QWEN_DEPLOY_MODE QWEN_CLUSTER_STATE_FILE QWEN_CLUSTER_TOKEN
unset QWEN_PD_MODE QWEN_PD_BOOTSTRAP_PORT QWEN_PD_IB_DEVICE
unset QWEN_TP_SIZE QWEN_PP_SIZE QWEN_NNODES QWEN_NODE_RANK QWEN_DIST_INIT_ADDR
source "$QWEN_CONFIG_PATH"
mode=${QWEN_DEPLOY_MODE:-single}
if [[ "$mode" != single && "$mode" != pd ]]; then
  echo 'QWEN_DEPLOY_MODE must be single or pd' >&2
  exit 2
fi
start_driver_log() {
  : "${DEPLOY_DIR:?Fill DEPLOY_DIR in config.sh}"
  mkdir -p "$DEPLOY_DIR/logs/driver"
  local driver_log="$DEPLOY_DIR/logs/driver/${action}_$(hostname)_$(date +%Y%m%d_%H%M%S)_$$.log"
  exec > >("$CLIENT_PYTHON" -B -S "$QWEN_KIT/tools/runtime_logging.py" "$driver_log") 2>&1
}
if [[ "$action" == prune ]]; then
  exec "$CLIENT_PYTHON" -B -S "$QWEN_KIT/tools/prune_logs.py" "${@:3}"
fi
if [[ "$mode" == pd ]]; then
  case "$action" in
    plan|check|start|status|stop)
      if [[ $# -ne 2 ]]; then
        echo 'PD lifecycle expects: run.sh ACTION /path/to/config.sh' >&2
        exit 2
      fi
      export PD_BASE_CONFIG="$QWEN_CONFIG_PATH"
      if [[ "$action" == start ]]; then start_driver_log; fi
      exec "${PD_CONTROL_PYTHON:-${CLIENT_PYTHON:-/usr/bin/python3}}" -u -B -S \
        "$QWEN_KIT/tools/pd_cluster.py" "$action"
      ;;
    build|pack) ;;
    *) echo 'PD mode supports build|pack|plan|check|start|status|stop|prune; send requests to the router.' >&2; exit 2 ;;
  esac
elif [[ "$action" == plan ]]; then
  echo 'plan requires QWEN_DEPLOY_MODE=pd in config' >&2
  exit 2
fi
: "${DEPLOY_DIR:?Fill DEPLOY_DIR in config.sh}"
export WORK_ROOT="$DEPLOY_DIR/instances"
export TILING_FILE="$QWEN_KIT/tools/gemm_tiling.plan"
export CHAT_TEMPLATE="$QWEN_KIT/tools/chat_template_no_think_default.jinja"
if [[ "$action" == build || "$action" == pack ]]; then
  if [[ -f "$QWEN_KIT/RUNTIME_LOCK.json" ]]; then
    echo '已冻结的release不能构建；请使用SGLang源码中的部署工具和新的版本目录' >&2
    exit 1
  fi
  export KERNEL_BUILD_RECEIPT="${KERNEL_BUILD_RECEIPT:-$DEPLOY_DIR/KERNEL_BUILD.json}"
  export KERNEL_PACKAGE_ROOT="$DEPLOY_DIR/build/kernel"
  if [[ "$action" == pack && -f "$KERNEL_BUILD_RECEIPT" ]]; then
    KERNEL_PACKAGE_ROOT=$("$CLIENT_PYTHON" -B -S -c 'import json,sys; print(json.load(open(sys.argv[1]))["kernel_root"])' "$KERNEL_BUILD_RECEIPT")
  fi
else
  # Use this entrypoint's tools, but always the selected deployment's frozen app.
  # Updating tools in SGLang does not modify/repack an existing release or images.
  export QWEN_PAYLOAD_ROOT="$DEPLOY_DIR/release"
  export SGLANG_SOURCE="$QWEN_PAYLOAD_ROOT/app/sglang"
  export KERNEL_PACKAGE_ROOT="$QWEN_PAYLOAD_ROOT/app/kernel"
  export TILING_FILE="$QWEN_PAYLOAD_ROOT/tools/gemm_tiling.plan"
  export CHAT_TEMPLATE="$QWEN_PAYLOAD_ROOT/tools/chat_template_no_think_default.jinja"
fi
start_driver_log
case "$action" in
  build) exec "$CLIENT_PYTHON" -u -B -S "$QWEN_KIT/tools/build_kernel.py" ;;
  pack) exec "$CLIENT_PYTHON" -u -B -S "$QWEN_KIT/tools/package.py" ;;
  deps|verify|check|start|request|bench|status|stop)
    exec "$CLIENT_PYTHON" -u -B -S "$QWEN_KIT/start.py" "$action" "${@:3}" ;;
  *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
