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
 *
 * Usage (Python):
 *     from sgl_kernel import pg_helper
 *     ptr_val = pg_helper.get_process_group_ptr(dist.group.WORLD)
 *     ptr_tensor = torch.tensor(ptr_val, dtype=torch.int64)
 *     torch.ops.sgl_kernel.kunpeng_moe_rdma_comm_create(ptr_tensor)
 */

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace py = pybind11;

static int64_t get_process_group_ptr(py::object group) {
  auto* raw_ptr = py::cast<c10d::ProcessGroup*>(group);
  if (!raw_ptr) {
    throw std::runtime_error("Extracted ProcessGroup pointer is null.");
  }

  return reinterpret_cast<int64_t>(raw_ptr);
}

PYBIND11_MODULE(pg_helper, m) {
  m.def(
      "get_process_group_ptr",
      &get_process_group_ptr,
      "Extract the C++ ProcessGroup pointer as an int64 scalar.\n\n"
      "Args:\n"
      "    group: A torch.distributed ProcessGroup object.\n\n"
      "Returns:\n"
      "    An int64 value containing the pointer address.");
}
