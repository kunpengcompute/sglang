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
#   moe              -- RDMA MoE communication create/finalize smoke test
#   shm              -- Shared memory pool create/destroy smoke test
#   reduce_scatter   -- SHM reduce_scatter benchmark vs torch.distributed
#   dual_allgather   -- SHM dual_allgather benchmark vs torch.distributed
#   batch_allgather  -- SHM batched_allgather benchmark vs torch.distributed
#   allreduce        -- SHM allreduce benchmark vs torch.distributed
#   all_reduce_min_int8 -- SHM all_reduce min_int8 correctness vs torch.distributed
#   mla_alltoall     -- SHM MLA alltoall correctness + benchmark vs torch.distributed
#   rdma_allgather   -- RDMA full-mesh allgather correctness + benchmark vs torch.distributed
#
# Options (environment variables):
#   PYTHON        -- python interpreter       (default: python3)
#   CPU_PER_RANK  -- cores per rank           (default: 38)

# ---- Parse argument ----------------------------------------------------------

if [[ $# -lt 1 ]]; then
    echo "Usage: bash run.sh <test_name>" >&2
    echo "Available tests: moe, shm, reduce_scatter, dual_allgather, batch_allgather, allreduce, min_int8" >&2
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
    dual_allgather)
        TEST_FILE="${SCRIPT_DIR}/test_dual_allgather.py"
        MASTER_PORT=5003
        TEST_LABEL="SHM dual_allgather benchmark"
        ;;
    batch_allgather)
        TEST_FILE="${SCRIPT_DIR}/test_batch_allgather.py"
        MASTER_PORT=5005
        TEST_LABEL="SHM batched_allgather benchmark"
        ;;
    allreduce)
        TEST_FILE="${SCRIPT_DIR}/test_allreduce.py"
        MASTER_PORT=5004
        TEST_LABEL="SHM allreduce benchmark"
        ;;
    all_reduce_min_int8)
        TEST_FILE="${SCRIPT_DIR}/test_allreduce.py"
        MASTER_PORT=5010
        TEST_LABEL="SHM all_reduce min_int8 correctness"
        export SGLANG_TEST_TYPE="all_reduce_min_int8"
        ;;
    mla_alltoall)
        TEST_FILE="${SCRIPT_DIR}/test_mla_alltoall.py"
        MASTER_PORT=8006
        TEST_LABEL="MLA alltoall correctness + benchmark"
        ;;
    rdma_allgather)
        TEST_FILE="${SCRIPT_DIR}/test_rdma_allgather.py"
        MASTER_PORT=8007
        TEST_LABEL="RDMA full-mesh allgather correctness + benchmark"
        ;;
    pp_comm)
        TEST_FILE="${SCRIPT_DIR}/test_pp_comm.py"
        MASTER_PORT=5012
        TEST_LABEL="PP unified RDMA message (pyobj/tensor/ack) correctness"
        # PP p2p only needs two peers; fixed regardless of env.sh.
        export PP_TEST_WORLD_SIZE=2
        export PP_TEST_MASTER_PORT=5012
        ;;
    *)
        echo "ERROR: unknown test '${TEST_NAME}'" >&2
        echo "Available tests: moe, shm, reduce_scatter, dual_allgather, batch_allgather, allreduce, min_int8, mla_alltoall, rdma_allgather, pp_comm" >&2
        exit 1
        ;;
esac

if [[ ! -f "${TEST_FILE}" ]]; then
    echo "ERROR: test file not found: ${TEST_FILE}" >&2
    exit 1
fi

# ---- Configuration -----------------------------------------------------------

source ../../../scripts/cpu_kunpeng/env.sh native
source ${HPCKIT_PATH}/latest/compiler/bisheng/env/setvars.sh

export LD_LIBRARY_PATH=${OpenBLAS_PATH}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/lib64/libibverbs:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${KUPL_PATH}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${KUTACC_PATH}/install/lib:${LD_LIBRARY_PATH}

PYTHON="${PYTHON:-python3}"
CPU_PER_RANK="${CPU_PER_RANK:-38}"
MASTER_ADDR="127.0.0.1"
# PP p2p tests only need two peers; pp_comm exports PP_TEST_WORLD_SIZE=2
# (env.sh sets WORLD_SIZE from NODE_IPS, so prefer the test-specific one).
WORLD_SIZE="${PP_TEST_WORLD_SIZE:-${WORLD_SIZE:-16}}"
# Same for the rendezvous port (env.sh sets MASTER_PORT per role).
MASTER_PORT="${PP_TEST_MASTER_PORT:-${MASTER_PORT:-5012}}"
IS_PRFILL="0"

export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE
export IS_PRFILL

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
