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

void set_kv_buffer_kunpeng(at::Tensor kv_buffer, at::Tensor loc,
                           at::Tensor cache_k)
{
    TORCH_CHECK(kv_buffer.dim() == 2, "kv_buffer must be 2D");
    TORCH_CHECK(loc.dim() == 1, "loc must be 1D");
    TORCH_CHECK(cache_k.dim() == 2, "cache_k must be 2D");
    TORCH_CHECK(cache_k.size(1) == kv_buffer.size(1),
                "cache_k dim-1 must match kv_buffer dim-1");
    TORCH_CHECK(cache_k.size(0) == loc.size(0),
                "cache_k and loc token count mismatch");

    TORCH_CHECK(kv_buffer.stride(1) == 1 && cache_k.stride(1) == 1,
                "kv_buffer and cache_k must have contiguous dim-1 (stride==1)");

    int64_t kdim = kv_buffer.size(1);
    int64_t tokens = loc.size(0);
    int64_t elem_sz = kv_buffer.element_size();
    int64_t buf_stride_bytes = kv_buffer.stride(0) * elem_sz;
    int64_t src_stride_bytes = cache_k.stride(0) * elem_sz;
    int64_t row_bytes = kdim * elem_sz;

    uint8_t* buf = static_cast<uint8_t*>(kv_buffer.data_ptr());
    uint8_t* src = static_cast<uint8_t*>(cache_k.data_ptr());

    if (loc.scalar_type() == at::kInt) {
        int32_t* idx = loc.data_ptr<int32_t>();
        for (int64_t i = 0; i < tokens; i++) {
            std::memcpy(buf + idx[i] * buf_stride_bytes,
                        src + i * src_stride_bytes, row_bytes);
        }
    } else {
        int64_t* idx = loc.data_ptr<int64_t>();
        for (int64_t i = 0; i < tokens; i++) {
            std::memcpy(buf + idx[i] * buf_stride_bytes,
                        src + i * src_stride_bytes, row_bytes);
        }
    }
}

static KernelRegistrar _r_set_kv_buffer(
    "set_kv_buffer_kunpeng",
    make_dispatch_v<decltype(&set_kv_buffer_kunpeng),
                    &set_kv_buffer_kunpeng>);

void set_kv_buffer_2_kunpeng(at::Tensor kv_buffer, at::Tensor loc,
                             at::Tensor k_nope, at::Tensor k_pe)
{
    TORCH_CHECK(kv_buffer.dim() == 2, "kv_buffer must be 2D");
    TORCH_CHECK(loc.dim() == 1, "loc must be 1D");
    TORCH_CHECK(k_nope.dim() == 2, "k_nope must be 2D");
    TORCH_CHECK(k_pe.dim() == 2, "k_pe must be 2D");
    TORCH_CHECK(k_nope.size(0) == loc.size(0) && k_pe.size(0) == loc.size(0),
                "k_nope/k_pe and loc token count mismatch");
    TORCH_CHECK(k_nope.size(1) + k_pe.size(1) == kv_buffer.size(1),
                "k_nope dim-1 + k_pe dim-1 must match kv_buffer dim-1");
    TORCH_CHECK(kv_buffer.stride(1) == 1 && k_nope.stride(1) == 1 &&
                    k_pe.stride(1) == 1,
                "kv_buffer, k_nope and k_pe must have contiguous dim-1 (stride==1)");

    int64_t kdim_n = k_nope.size(1);
    int64_t kdim_p = k_pe.size(1);
    int64_t tokens = loc.size(0);
    int64_t elem_sz = kv_buffer.element_size();
    int64_t buf_stride_bytes = kv_buffer.stride(0) * elem_sz;
    int64_t nope_stride_bytes = k_nope.stride(0) * elem_sz;
    int64_t pe_stride_bytes = k_pe.stride(0) * elem_sz;
    int64_t nope_bytes = kdim_n * elem_sz;
    int64_t pe_bytes = kdim_p * elem_sz;

    uint8_t* buf = static_cast<uint8_t*>(kv_buffer.data_ptr());
    uint8_t* nope = static_cast<uint8_t*>(k_nope.data_ptr());
    uint8_t* pe = static_cast<uint8_t*>(k_pe.data_ptr());

    if (loc.scalar_type() == at::kInt) {
        int32_t* idx = loc.data_ptr<int32_t>();
        for (int64_t i = 0; i < tokens; i++) {
            uint8_t* dst = buf + idx[i] * buf_stride_bytes;
            std::memcpy(dst, nope + i * nope_stride_bytes, nope_bytes);
            std::memcpy(dst + nope_bytes, pe + i * pe_stride_bytes, pe_bytes);
        }
    } else {
        int64_t* idx = loc.data_ptr<int64_t>();
        for (int64_t i = 0; i < tokens; i++) {
            uint8_t* dst = buf + idx[i] * buf_stride_bytes;
            std::memcpy(dst, nope + i * nope_stride_bytes, nope_bytes);
            std::memcpy(dst + nope_bytes, pe + i * pe_stride_bytes, pe_bytes);
        }
    }
}

static KernelRegistrar _r_set_kv_buffer_2(
    "set_kv_buffer_2_kunpeng",
    make_dispatch_v<decltype(&set_kv_buffer_2_kunpeng),
                    &set_kv_buffer_2_kunpeng>);
