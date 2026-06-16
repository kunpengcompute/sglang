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

#pragma once

/**
 * kunpeng_oob.h - Common OOB (Out-of-Band) callback utilities for Kunpeng.
 *
 * Shared by both RDMA MoE (kurmcl) and shared memory (kupl_shm) modules.
 * Provides:
 *   - Data type conversion: kurmcl_datatype_t / kupl_shm_datatype_t -> at::ScalarType
 *   - OOB callbacks using c10d::ProcessGroup for collective communication
 */

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "kutacc.h"
#include "kupl.h"

namespace kunpeng_oob {

// ============================================================================
// Data type conversion
// ============================================================================

/** Convert kurmcl_datatype_t to at::ScalarType. */
at::ScalarType kurmcl_dtype_to_at(kutacc::kurmcl_datatype_t dt);

/** Convert kupl_shm_datatype_t to at::ScalarType. */
at::ScalarType kupl_shm_dtype_to_at(kupl_shm_datatype_t dt);

// ============================================================================
// OOB callbacks for kurmcl (RDMA MoE)
// ============================================================================

/**
 * OOB allgather callback for kurmcl.
 * group_ptr must be a c10d::ProcessGroup*.
 */
int kurmcl_oob_allgather(const void *sendbuf, void *recvbuf, int count, void *group_ptr,
                         kutacc::kurmcl_datatype_t datatype);

/**
 * OOB barrier callback for kurmcl.
 * group_ptr must be a c10d::ProcessGroup*.
 */
int kurmcl_oob_barrier(void *group_ptr);

/**
 * OOB alltoall callback for kurmcl.
 * group_ptr must be a c10d::ProcessGroup*.
 */
int kurmcl_oob_alltoall(const void *sendbuf, int sendcount, kutacc::kurmcl_datatype_t sendtype,
                        void *recvbuf, int recvcount, kutacc::kurmcl_datatype_t recvtype,
                        void *group_ptr);

// ============================================================================
// OOB callbacks for kupl_shm (shared memory)
// ============================================================================

/**
 * OOB allgather callback for kupl_shm.
 * group_ptr must be a c10d::ProcessGroup*.
 */
int kupl_shm_oob_allgather(const void *sendbuf, void *recvbuf, int count, void *group_ptr,
                           kupl_shm_datatype_t datatype);

/**
 * OOB barrier callback for kupl_shm.
 * group_ptr must be a c10d::ProcessGroup*.
 */
int kupl_shm_oob_barrier(void *group_ptr);

} // namespace kunpeng_oob
