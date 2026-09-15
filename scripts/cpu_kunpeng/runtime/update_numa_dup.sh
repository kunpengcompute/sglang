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
# runtime/update_numa_dup.sh - Update NUMA binary replicas.
# Sources env.sh (native) to get paths + conda, then runs pyinstall/update.sh.
# Usage: bash runtime/update_numa_dup.sh [sglang|kernel|torch|all]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/env.sh" native

echo "Update binary sglang..."
bash "$SCRIPT_DIR/pyinstall/update.sh" "$@"
