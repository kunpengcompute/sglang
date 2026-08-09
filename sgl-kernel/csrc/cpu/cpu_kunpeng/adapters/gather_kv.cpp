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

#include <ATen/Tensor.h>
#include <cstring>

#include "register_graph_kernels.h"

// Gather rows from kv_buffer by index into a contiguous output tensor.
// Used by chunked prefill to load prefix KV from the paged KV cache.
// At graph replay, kv_buffer (fixed) and indices (graph input) are updated,
// so the gathered prefix KV always reflects the current request state.
void gather_kv_kunpeng(at::Tensor kv_buffer, at::Tensor indices, at::Tensor out)
{
    int64_t num_indices = indices.size(0);
    int64_t row_bytes = out.stride(0) * out.element_size();
    int64_t buf_stride = kv_buffer.stride(0) * kv_buffer.element_size();
    int64_t out_stride = out.stride(0) * out.element_size();

    uint8_t* buf = static_cast<uint8_t*>(kv_buffer.data_ptr());
    uint8_t* dst = static_cast<uint8_t*>(out.data_ptr());

    if (indices.scalar_type() == at::kInt) {
        int32_t* idx = indices.data_ptr<int32_t>();
        for (int64_t i = 0; i < num_indices; i++) {
            std::memcpy(dst + i * out_stride,
                        buf + idx[i] * buf_stride, row_bytes);
        }
    } else {
        int64_t* idx = indices.data_ptr<int64_t>();
        for (int64_t i = 0; i < num_indices; i++) {
            std::memcpy(dst + i * out_stride,
                        buf + idx[i] * buf_stride, row_bytes);
        }
    }
}

static KernelRegistrar _r_gather_kv(
    "gather_kv_kunpeng",
    make_dispatch_v<decltype(&gather_kv_kunpeng), &gather_kv_kunpeng>);
