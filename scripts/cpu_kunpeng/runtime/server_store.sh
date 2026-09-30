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
# server_store.sh - Start the mooncake store service on this node.
# Invoked remotely by: runtime/launch_store.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR_ARG="${1:-}"

source "$SCRIPT_DIR/env.sh" store || exit 1

LOG_DIR="${LOG_DIR_ARG:-$LOG_DIR}"
mkdir -p "$LOG_DIR"
echo "[$(date +%T)] mooncake store service on $MOONCAKE_STORE_NODE:$MOONCAKE_STORE_PORT, segment $MOONCAKE_GLOBAL_SEGMENT_SIZE, master $MOONCAKE_MASTER"

exec python -m mooncake.mooncake_store_service --port="$MOONCAKE_STORE_PORT"