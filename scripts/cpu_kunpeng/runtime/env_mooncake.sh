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
# Mooncake role config: the store master behind KUNPENG_HICACHE_BACKEND=mooncake.
# Sourced by env.sh (after env_base.sh, so .user_env.sh overrides set there are
# respected via the ${VAR:-default} fallbacks below).

# ------------------------------------------------------------
# Mooncake store master
# ------------------------------------------------------------
# mooncake_master owns the global segment allocation, the object metadata and
# the eviction decisions; every prefill rank registers its L2 host pool with it.
# It is a native binary, not a python module: the pip wheel ships the client
# (mooncake.store) only, so MOONCAKE_MASTER_BIN must point at a locally built
# mooncake_master.
export MOONCAKE_MASTER_BIN="${MOONCAKE_MASTER_BIN:-}"
# Port of the master's embedded HTTP metadata server. This is NOT the port the
# store clients use: those go through MOONCAKE_MASTER
# (= MOONCAKE_MASTER_NODE:MOONCAKE_MASTER_PORT, the master's RPC port, see
# runtime/env_base.sh). The two must stay distinct.
export MOONCAKE_METADATA_PORT="${MOONCAKE_METADATA_PORT:-8080}"
# Evict objects once this fraction of the global segment is in use.
export MOONCAKE_EVICTION_WATERMARK="${MOONCAKE_EVICTION_WATERMARK:-0.9}"

if [[ -z "$MOONCAKE_MASTER_BIN" ]]; then
    echo "ERROR: MOONCAKE_MASTER_BIN is not set (path to the mooncake_master binary)." >&2
    return 1 2>/dev/null || exit 1
fi
if [[ ! -x "$MOONCAKE_MASTER_BIN" ]]; then
    echo "ERROR: MOONCAKE_MASTER_BIN='$MOONCAKE_MASTER_BIN' is not an executable file." >&2
    return 1 2>/dev/null || exit 1
fi
