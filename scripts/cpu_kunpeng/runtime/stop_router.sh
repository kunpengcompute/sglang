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
# stop_router.sh - Kill the gateway (sgl-model-gateway) on the router node.
# Invoked by: ./stop.sh router

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Source config to get ROUTER_IP
SKIP_CONDA=1 source ./env.sh router

echo "Killing gateway on $ROUTER_IP"
ssh "root@$ROUTER_IP" '
    MAIN_PIDS=$(ps aux | grep "sgl-model-gateway" | grep -v grep | awk "{print \$2}")
    if [ -n "$MAIN_PIDS" ]; then
        echo "Killing process(es): $MAIN_PIDS"
        kill -15 $MAIN_PIDS 2>/dev/null
        sleep 5
        REMAINING=$(ps aux | grep "sgl-model-gateway" | grep -v grep | awk "{print \$2}")
        if [ -n "$REMAINING" ]; then
            kill -9 $REMAINING 2>/dev/null
        fi
        echo "Router stopped."
    else
        echo "No router process found."
    fi
'
