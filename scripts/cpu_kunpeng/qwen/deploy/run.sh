#!/usr/bin/env bash
set -euo pipefail
export DEPLOY_TOOLS
DEPLOY_TOOLS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
action=${1:-help}
if [[ "$action" == help ]]; then
  echo 'Usage: bash run.sh build|pack|plan|deps|verify|check|start|request|bench|status|stop|prune /path/to/config.sh [--instance INSTANCE_ID | --apply]'
  echo 'DEPLOY_MODE=single|pd in config selects node or PD cluster lifecycle; default: single.'
  exit 0
fi
export DEPLOY_CONFIG
DEPLOY_CONFIG=$(realpath "${2:-$DEPLOY_TOOLS/config.sh}")
while IFS= read -r key; do
  case "$key" in SGLANG_*|KUTACC_*|KUPL_*) unset "$key" ;; esac
done < <(compgen -e)
unset PYTHONHOME PYTHONPATH LD_PRELOAD
unset DEPLOY_PAYLOAD_ROOT
# Ownership and topology must come from this invocation's configuration.
unset DEPLOY_MODE CLUSTER_STATE_FILE CLUSTER_TOKEN
unset DISAGGREGATION_MODE DISAGGREGATION_BOOTSTRAP_PORT DISAGGREGATION_IB_DEVICE
unset TP_SIZE PP_SIZE NNODES NODE_RANK DIST_INIT_ADDR SHM_MIB
source "$DEPLOY_CONFIG"
mode=${DEPLOY_MODE:-single}
if [[ "$mode" != single && "$mode" != pd ]]; then
  echo 'DEPLOY_MODE must be single or pd' >&2
  exit 2
fi
if [[ "$mode" == single ]]; then
  : "${SHM_MIB:?Fill SHM_MIB for a single node; PD derives it from its role quota}"
  export SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
  export SGLANG_KUNPENG_SHARED_HBW_BACKING=split
  export SGLANG_KUNPENG_SHARED_HBW_TOTAL_MIB="$SGLANG_KUNPENG_WEIGHTS_HBW_POOL_SIZE_MB"
  export SGLANG_KUNPENG_SHARED_HBW_SHM_PREFIX_MIB="$SHM_MIB"
  export SGLANG_KUNPENG_PREFILL_SHM_SIZE_MB="$SHM_MIB"
  export SGLANG_KUNPENG_DECODE_SHM_SIZE_MB="$SHM_MIB"
  export SGLANG_KUNPENG_MAX_SEQ_LEN="$CONTEXT_LENGTH"
  export SGLANG_KUNPENG_MAX_SEQ_NUM="${SGLANG_KUNPENG_MAX_SEQ_NUM:-$MAX_RUNNING_REQUESTS}"
fi
start_driver_log() {
  : "${DEPLOY_DIR:?Fill DEPLOY_DIR in config.sh}"
  mkdir -p "$DEPLOY_DIR/logs/driver"
  local driver_log="$DEPLOY_DIR/logs/driver/${action}_$(hostname)_$(date +%Y%m%d_%H%M%S)_$$.log"
  exec > >("$CLIENT_PYTHON" -B -S "$DEPLOY_TOOLS/tools/runtime_logging.py" "$driver_log") 2>&1
}
if [[ "$action" == prune ]]; then
  exec "$CLIENT_PYTHON" -B -S "$DEPLOY_TOOLS/tools/prune_logs.py" "${@:3}"
fi
if [[ "$mode" == pd ]]; then
  case "$action" in
    plan|check|start|status|stop)
      if [[ $# -ne 2 ]]; then
        echo 'PD lifecycle expects: run.sh ACTION /path/to/config.sh' >&2
        exit 2
      fi
      export PD_BASE_CONFIG="$DEPLOY_CONFIG"
      if [[ "$action" == start ]]; then start_driver_log; fi
      exec "${PD_CONTROL_PYTHON:-${CLIENT_PYTHON:-/usr/bin/python3}}" -u -B -S \
        "$DEPLOY_TOOLS/tools/pd_cluster.py" "$action"
      ;;
    build|pack) ;;
    *) echo 'PD mode supports build|pack|plan|check|start|status|stop|prune; send requests to the router.' >&2; exit 2 ;;
  esac
elif [[ "$action" == plan ]]; then
  echo 'plan requires DEPLOY_MODE=pd in config' >&2
  exit 2
fi
: "${DEPLOY_DIR:?Fill DEPLOY_DIR in config.sh}"
export WORK_ROOT="$DEPLOY_DIR/instances"
export TILING_FILE="$DEPLOY_TOOLS/tools/gemm_tiling.plan"
export CHAT_TEMPLATE="$DEPLOY_TOOLS/tools/chat_template_no_think_default.jinja"
if [[ "$action" == build || "$action" == pack ]]; then
  if [[ -f "$DEPLOY_TOOLS/RUNTIME_LOCK.json" ]]; then
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
  export DEPLOY_PAYLOAD_ROOT="$DEPLOY_DIR/release"
  export SGLANG_SOURCE="$DEPLOY_PAYLOAD_ROOT/app/sglang"
  export KERNEL_PACKAGE_ROOT="$DEPLOY_PAYLOAD_ROOT/app/kernel"
  export TILING_FILE="$DEPLOY_PAYLOAD_ROOT/tools/gemm_tiling.plan"
  export CHAT_TEMPLATE="$DEPLOY_PAYLOAD_ROOT/tools/chat_template_no_think_default.jinja"
fi
start_driver_log
case "$action" in
  build) exec "$CLIENT_PYTHON" -u -B -S "$DEPLOY_TOOLS/tools/build_kernel.py" ;;
  pack) exec "$CLIENT_PYTHON" -u -B -S "$DEPLOY_TOOLS/tools/package.py" ;;
  deps|verify|check|start|request|bench|status|stop)
    exec "$CLIENT_PYTHON" -u -B -S "$DEPLOY_TOOLS/start.py" "$action" "${@:3}" ;;
  *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
