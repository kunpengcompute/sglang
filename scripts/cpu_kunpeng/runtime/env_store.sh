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
export MOONCAKE_STORE_SEGMENT_SIZE="${MOONCAKE_STORE_SEGMENT_SIZE:-16gb}"

export MOONCAKE_GLOBAL_SEGMENT_SIZE="$MOONCAKE_STORE_SEGMENT_SIZE"
export MOONCAKE_LOCAL_HOSTNAME="${MOONCAKE_LOCAL_HOSTNAME:-$MOONCAKE_STORE_NODE}"
export MOONCAKE_LOCAL_BUFFER_SIZE=0
export MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-rdma}"
export MOONCAKE_TE_META_DATA_SERVER="${MOONCAKE_TE_META_DATA_SERVER:-P2PHANDSHAKE}"