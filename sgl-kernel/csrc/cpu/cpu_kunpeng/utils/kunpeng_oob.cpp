/*
 * Copyright 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

#include "kunpeng_oob.h"

#include <vector>

namespace kunpeng_oob {

// ============================================================================
// Data type conversion
// ============================================================================

at::ScalarType kurmcl_dtype_to_at(kutacc::kurmcl_datatype_t dt)
{
    switch (dt) {
        case kutacc::KURMCL_DATATYPE_CHAR:
            return at::kByte;
        case kutacc::KURMCL_DATATYPE_INT:
            return at::kInt;
        case kutacc::KURMCL_DATATYPE_LONG:
            return at::kLong;
        case kutacc::KURMCL_DATATYPE_FLOAT:
            return at::kFloat;
        case kutacc::KURMCL_DATATYPE_DOUBLE:
            return at::kDouble;
        default:
            return at::kByte;
    }
}

at::ScalarType kupl_shm_dtype_to_at(kupl_shm_datatype_t dt)
{
    switch (dt) {
        case KUPL_SHM_DATATYPE_CHAR:
            return at::kByte;
        case KUPL_SHM_DATATYPE_INT:
            return at::kInt;
        case KUPL_SHM_DATATYPE_LONG:
            return at::kLong;
        case KUPL_SHM_DATATYPE_FLOAT:
            return at::kFloat;
        case KUPL_SHM_DATATYPE_DOUBLE:
            return at::kDouble;
        default:
            return at::kByte;
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

static c10d::ProcessGroup *get_process_group(void *group_ptr)
{
    if (!group_ptr) return nullptr;
    return reinterpret_cast<c10d::ProcessGroup *>(group_ptr);
}

static int do_allgather(c10d::ProcessGroup *pg, const void *sendbuf, void *recvbuf, int count, at::ScalarType dtype)
{
    auto opts = at::TensorOptions().device(at::kCPU).dtype(dtype);
    auto send_tensor = at::from_blob(const_cast<void *>(sendbuf), {count}, opts);
    auto recv_tensor = at::from_blob(recvbuf, {pg->getSize() * count}, opts);

    auto work = pg->_allgather_base(recv_tensor, send_tensor);
    if (!work) return -1;
    work->wait();
    return 0;
}

static int do_barrier(c10d::ProcessGroup *pg)
{
    auto work = pg->barrier();
    if (!work) return -1;
    work->wait();
    return 0;
}

static int do_alltoall(c10d::ProcessGroup *pg, const void *sendbuf, int sendcount, at::ScalarType send_dtype,
                       void *recvbuf, int recvcount)
{
    int comm_size = pg->getSize();
    auto opts = at::TensorOptions().device(at::kCPU).dtype(send_dtype);
    auto send_tensor = at::from_blob(const_cast<void *>(sendbuf), {comm_size * sendcount}, opts);
    auto recv_tensor = at::from_blob(recvbuf, {comm_size * recvcount}, opts);
    std::vector<int64_t> send_list(comm_size, sendcount);

    auto work = pg->alltoall_base(recv_tensor, send_tensor, send_list, send_list);
    if (!work) return -1;
    work->wait();
    return 0;
}

// ============================================================================
// OOB callbacks for kurmcl (RDMA MoE)
// ============================================================================

int kurmcl_oob_allgather(const void *sendbuf, void *recvbuf, int count, void *group_ptr,
                         kutacc::kurmcl_datatype_t datatype)
{
    auto *pg = get_process_group(group_ptr);
    if (!pg) return -1;
    return do_allgather(pg, sendbuf, recvbuf, count, kurmcl_dtype_to_at(datatype));
}

int kurmcl_oob_barrier(void *group_ptr)
{
    auto *pg = get_process_group(group_ptr);
    if (!pg) return -1;
    return do_barrier(pg);
}

int kurmcl_oob_alltoall(const void *sendbuf, int sendcount, kutacc::kurmcl_datatype_t sendtype, void *recvbuf,
                        int recvcount, kutacc::kurmcl_datatype_t /*recvtype*/, void *group_ptr)
{
    auto *pg = get_process_group(group_ptr);
    if (!pg) return -1;
    return do_alltoall(pg, sendbuf, sendcount, kurmcl_dtype_to_at(sendtype), recvbuf, recvcount);
}

// ============================================================================
// OOB callbacks for kupl_shm (shared memory)
// ============================================================================

int kupl_shm_oob_allgather(const void *sendbuf, void *recvbuf, int count, void *group_ptr, kupl_shm_datatype_t datatype)
{
    auto *pg = get_process_group(group_ptr);
    if (!pg) return -1;
    return do_allgather(pg, sendbuf, recvbuf, count, kupl_shm_dtype_to_at(datatype));
}

int kupl_shm_oob_barrier(void *group_ptr)
{
    auto *pg = get_process_group(group_ptr);
    if (!pg) return -1;
    return do_barrier(pg);
}

}  // namespace kunpeng_oob
