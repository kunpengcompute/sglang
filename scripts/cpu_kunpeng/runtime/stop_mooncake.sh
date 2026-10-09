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
# stop_mooncake.sh - Kill the mooncake store master on MOONCAKE_MASTER_NODE.
# Invoked by: ./stop.sh mooncake

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Base config only: stop only needs MOONCAKE_MASTER_NODE (no role topology, and
# no MOONCAKE_MASTER_BIN validation -- stopping must work even if the binary
# path is already gone). "none" mode skips the role env entirely.
SKIP_CONDA=1 source ./env.sh none

echo "Killing mooncake master on $MOONCAKE_MASTER_NODE"
ssh "root@$MOONCAKE_MASTER_NODE" '
    if pkill -f mooncake_master; then
        echo "Mooncake master stopped."
    else
        echo "No mooncake master process found."
    fi
'
