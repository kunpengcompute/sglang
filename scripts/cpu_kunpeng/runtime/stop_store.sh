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
# stop_store.sh - Kill the mooncake store service on MOONCAKE_STORE_NODE.
# Invoked by: ./stop.sh store

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

SKIP_CONDA=1 source ./env.sh none

echo "Killing mooncake store service on $MOONCAKE_STORE_NODE"
ssh "root@$MOONCAKE_STORE_NODE" '
    if pkill -f mooncake.mooncake_store_service; then
        echo "Mooncake store service stopped."
    else
        echo "No mooncake store service process found."
    fi
'