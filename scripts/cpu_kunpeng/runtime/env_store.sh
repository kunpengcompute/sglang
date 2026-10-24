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
# store role config: the mooncake store service (real client) that holds the
# shared L3 segment. The SGLang nodes run with MOONCAKE_GLOBAL_SEGMENT_SIZE=0,
# so this process is where the L3 KV cache actually lives.

export MOONCAKE_STORE_NODE="${MOONCAKE_STORE_NODE:-$MOONCAKE_MASTER_NODE}"
export MOONCAKE_STORE_PORT="${MOONCAKE_STORE_PORT:-8081}"
export MOONCAKE_STORE_SEGMENT_SIZE="${MOONCAKE_STORE_SEGMENT_SIZE:-64gb}"

export MOONCAKE_GLOBAL_SEGMENT_SIZE="$MOONCAKE_STORE_SEGMENT_SIZE"
export MOONCAKE_LOCAL_HOSTNAME="${MOONCAKE_LOCAL_HOSTNAME:-$MOONCAKE_STORE_NODE}"
export MOONCAKE_LOCAL_BUFFER_SIZE=0
export MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-rdma}"
export MOONCAKE_TE_META_DATA_SERVER="${MOONCAKE_TE_META_DATA_SERVER:-P2PHANDSHAKE}"

if [[ "${MOONCAKE_ENABLE_SSD_OFFLOAD:-0}" == "1" ]]; then
    if [[ -z "${MOONCAKE_OFFLOAD_FILE_STORAGE_PATH:-}" ]]; then
        echo "ERROR: MOONCAKE_ENABLE_SSD_OFFLOAD=1 requires MOONCAKE_OFFLOAD_FILE_STORAGE_PATH (absolute SSD directory)" >&2
        return 1 2>/dev/null || exit 1
    fi
    export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH
    export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR="${MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR:-bucket_storage_backend}"
    export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES="${MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES:-1099511627776}"
    export MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY="${MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY:-lru}"
    echo "[store] SSD offload enabled -> $MOONCAKE_OFFLOAD_FILE_STORAGE_PATH"
fi
