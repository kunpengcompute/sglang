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

#!/usr/bin/env bash
#
# Unified launcher script for Kunpeng CPU multi-process tests.
#
# Starts 16 Python processes individually, each wrapped with
# ``taskset -c`` so the CPU affinity is set *before* the process
# begins executing.  The gloo rendezvous is configured through
# environment variables (MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE)
# so ``dist.init_process_group(init_method="env://")`` works without
# torchrun.
#
# Usage:
#   bash run.sh <test_name>
#
# Available tests:
#   moe             -- RDMA MoE communication create/finalize smoke test
#   shm             -- Shared memory pool create/destroy smoke test
#   reduce_scatter  -- SHM reduce_scatter benchmark vs torch.distributed
#   allgather       -- SHM dual_allgather benchmark vs torch.distributed
#   allreduce       -- SHM allreduce benchmark vs torch.distributed
#
# Options (environment variables):
#   PYTHON        -- python interpreter       (default: python3)
#   CPU_PER_RANK  -- cores per rank           (default: 38)

set -euo pipefail

# ---- Parse argument ----------------------------------------------------------

if [[ $# -lt 1 ]]; then
    echo "Usage: bash run.sh <test_name>" >&2
    echo "Available tests: moe, shm, reduce_scatter, allgather, allreduce" >&2
    exit 1
fi

TEST_NAME="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${TEST_NAME}" in
    moe)
        TEST_FILE="${SCRIPT_DIR}/test_moe_comm_create.py"
        MASTER_PORT=5000
        TEST_LABEL="MoE comm create test"
        ;;
    shm)
        TEST_FILE="${SCRIPT_DIR}/test_shm_pool_create.py"
        MASTER_PORT=5001
        TEST_LABEL="SHM pool create test"
        ;;
    reduce_scatter)
        TEST_FILE="${SCRIPT_DIR}/test_reduce_scatter.py"
        MASTER_PORT=5002
        TEST_LABEL="SHM reduce_scatter benchmark"
        ;;
    allgather)
        TEST_FILE="${SCRIPT_DIR}/test_allgather.py"
        MASTER_PORT=5003
        TEST_LABEL="SHM dual_allgather benchmark"
        ;;
    allreduce)
        TEST_FILE="${SCRIPT_DIR}/test_allreduce.py"
        MASTER_PORT=5004
        TEST_LABEL="SHM allreduce benchmark"
        ;;
    *)
        echo "ERROR: unknown test '${TEST_NAME}'" >&2
        echo "Available tests: moe, shm, reduce_scatter, allgather, allreduce" >&2
        exit 1
        ;;
esac

if [[ ! -f "${TEST_FILE}" ]]; then
    echo "ERROR: test file not found: ${TEST_FILE}" >&2
    exit 1
fi

# ---- Configuration -----------------------------------------------------------

PYTHON="${PYTHON:-python3}"
CPU_PER_RANK="${CPU_PER_RANK:-38}"
MASTER_ADDR="127.0.0.1"
WORLD_SIZE=16

export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE

eval "$CONDA_ACTIVATE_CMD"

echo "============================================================"
echo " ${TEST_LABEL}"
echo " WORLD_SIZE  = ${WORLD_SIZE}"
echo " MASTER_ADDR = ${MASTER_ADDR}"
echo " MASTER_PORT = ${MASTER_PORT}"
echo " CPU_PER_RANK= ${CPU_PER_RANK}"
echo " PYTHON      = ${PYTHON}"
echo " TEST_FILE   = ${TEST_FILE}"
echo "============================================================"

# ---- Launch workers ----------------------------------------------------------

PIDS=()

for RANK in $(seq 0 $((WORLD_SIZE - 1))); do
    CPU_START=$((RANK * CPU_PER_RANK))
    CPU_RANGE="${CPU_START}-$((CPU_START + 15)),$((CPU_START + 21))-$((CPU_START + 36))"

    export RANK

    echo "[launcher] starting rank ${RANK} on cpus ${CPU_RANGE}"
    taskset -c "${CPU_RANGE}" "${PYTHON}" "${TEST_FILE}" &
    PIDS+=($!)
done

echo "[launcher] all ${WORLD_SIZE} workers launched, waiting..."

# ---- Wait for all workers and collect exit codes -----------------------------

ALL_OK=true
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    if wait "${PID}"; then
        echo "[launcher] rank ${i} (pid ${PID}) exited 0"
    else
        EC=$?
        echo "[launcher] rank ${i} (pid ${PID}) FAILED with exit code ${EC}" >&2
        ALL_OK=false
    fi
done

if ${ALL_OK}; then
    echo "[launcher] ALL ${WORLD_SIZE} workers passed."
    exit 0
else
    echo "[launcher] SOME WORKERS FAILED." >&2
    exit 1
fi
