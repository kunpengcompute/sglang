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
# run.sh - Single node execution for SGLang (prefill or decode).
# Usage: ./run.sh <role> <rank>

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <role> <rank>" >&2
    exit 1
fi

ROLE="$1"
RANK="$2"
IP="$(ifconfig enp26s0f0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}')"

# Source environment config (exports CONDA_ACTIVATE_CMD, PYTHON_SCRIPT, etc.)
source env.sh "$ROLE"

# Activate conda
eval "$CONDA_ACTIVATE_CMD"

# Base arguments common to both roles
BASE_ARGS=(
    --model "$MODEL_PATH"
    --device cpu
    --trust-remote-code
    --host "$IP"
    --port 30001
    --dist-init-addr "$MASTER_ADDR:$MASTER_PORT"
    --nnodes "$WORLD_SIZE"
    --node-rank "$RANK"
    --dist-timeout 600
    --enable-dp-attention
    --dp-size "$WORLD_SIZE"
    --tp-size "$TP_SIZE"
    --page-size 64
    --mem-fraction-static 0.88
    --chunked-prefill-size -1
    --skip-server-warmup
    --disable-custom-all-reduce
    --disable-radix-cache
    --disable-overlap-schedule
    --disaggregation-ib-device \
'{"0":"roceroh0","1":"roceroh0","2":"roceroh1","3":"roceroh1",'\
'"4":"roceroh2","5":"roceroh2","6":"roceroh3","7":"roceroh3",'\
'"8":"roceroh4","9":"roceroh4","10":"roceroh5","11":"roceroh5",'\
'"12":"roceroh6","13":"roceroh6", "14":"roceroh7","15":"roceroh7"}'
)

# Role-specific arguments
case "$ROLE" in
    prefill)
        SPECIFIC_ARGS=(
            --disaggregation-mode prefill
            --disaggregation-bootstrap-port 9001
            --max-prefill-tokens 4096
            --max-total-tokens 18496
            --prefill-max-requests 4
            --load-balance-method follow_bootstrap_room
        )
        ;;
    decode)
        SPECIFIC_ARGS=(
            --disaggregation-mode decode
            --max-total-tokens 139328
            --load-balance-method follow_bootstrap_room
        )
        ;;
    *)
        echo "Error: unknown role '$ROLE'" >&2
        exit 1
        ;;
esac

# Combine and execute
taskset -c 0,33-38,71-76,109-114,147-152,185-190,223-228,261-266,299-304,337-342,375-380,413-418,451-456,489-494,527-532,565-570,602-607 \
python -m sglang.launch_server "${BASE_ARGS[@]}" "${SPECIFIC_ARGS[@]}"