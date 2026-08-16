#!/bin/bash
set -eux

# ccache configuration
USE_CCACHE="${USE_CCACHE:-1}"

# Keep the ccache dir inside the sgl-kernel source tree (shared storage) so it
# survives node reboots / rebuilds on a different node. Override via CCACHE_DIR.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${USE_CCACHE}" = "1" ] && command -v ccache &>/dev/null; then
  export CCACHE_DIR="${CCACHE_DIR:-${SCRIPT_DIR}/.ccache}"
  export CCACHE_BASEDIR="$(pwd)"
  export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-10G}"
  export CCACHE_COMPILERCHECK=content
  export CCACHE_COMPRESS=true
  export CCACHE_SLOPPINESS=file_macro,time_macros,include_file_mtime,include_file_ctime
  export CMAKE_C_COMPILER_LAUNCHER=ccache
  export CMAKE_CXX_COMPILER_LAUNCHER=ccache
  export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
  mkdir -p "${CCACHE_DIR}"
  echo "=== ccache enabled, dir: ${CCACHE_DIR} ==="
  ccache -sV
else
  echo "=== ccache disabled ==="
fi

pip install -v ./ --no-build-isolation --no-deps

if [ "${USE_CCACHE}" = "1" ] && command -v ccache &>/dev/null; then
  echo "=== ccache stats (after) ==="
  ccache -s
fi