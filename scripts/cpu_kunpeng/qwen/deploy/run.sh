#!/usr/bin/env bash
set -euo pipefail
export QWEN_KIT
QWEN_KIT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
action=${1:-help}
if [[ "$action" == help ]]; then
  echo 'Usage: bash run.sh build|pack|deps|verify|check|start|request|bench|status|stop /path/to/config.sh [--instance INSTANCE_ID]'
  exit 0
fi
export QWEN_CONFIG_PATH
QWEN_CONFIG_PATH=$(realpath "${2:-$QWEN_KIT/config.sh}")
while IFS= read -r key; do
  case "$key" in SGLANG_*|KUTACC_*|KUPL_*) unset "$key" ;; esac
done < <(compgen -e)
unset PYTHONHOME PYTHONPATH LD_PRELOAD
unset QWEN_PAYLOAD_ROOT
source "$QWEN_CONFIG_PATH"
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
mkdir -p "$DEPLOY_DIR/logs/driver"
driver_log="$DEPLOY_DIR/logs/driver/${action}_$(hostname)_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -a "$driver_log") 2>&1
case "$action" in
  build) exec "$CLIENT_PYTHON" -u -B -S "$QWEN_KIT/tools/build_kernel.py" ;;
  pack) exec "$CLIENT_PYTHON" -u -B -S "$QWEN_KIT/tools/package.py" ;;
  deps|verify|check|start|request|bench|status|stop)
    exec "$CLIENT_PYTHON" -u -B -S "$QWEN_KIT/start.py" "$action" "${@:3}" ;;
  *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
