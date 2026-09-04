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
# runtime/update_time.sh - Regenerate .time_env.sh with fresh LOG_DATE/LOG_TIME
# so all nodes source the same time-stamped log dir. Standalone runnable.
# Usage: bash runtime/update_time.sh

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cat > "$BASE_DIR/.time_env.sh" << EOF
export LOG_DATE="$(date +%y%m%d)"
export LOG_TIME="$(date +%H%M%S)"
EOF
