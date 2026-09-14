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

#include <torch/extension.h>

#include "memory_pool.h"

void MemoryPool::allocate(size_t size, size_t base_alignment)
{
    if (size == 0) {
        tensor_ = torch::Tensor();
        base_off_ = 0;
        return;
    }
    auto options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCPU);
    if (base_alignment > 1) {
        // Over-allocate and start at an aligned base so per-storage offset
        // alignment in pack_intervals equals absolute-address alignment.
        tensor_ = torch::empty({static_cast<int64_t>(size + base_alignment)}, options);
        uintptr_t p = reinterpret_cast<uintptr_t>(tensor_.data_ptr());
        base_off_ = (base_alignment - p % base_alignment) % base_alignment;
    } else {
        tensor_ = torch::empty({static_cast<int64_t>(size)}, options);
        base_off_ = 0;
    }
}

void MemoryPool::adopt(torch::Tensor tensor)
{
    TORCH_CHECK(tensor.device().is_cpu(), "MemoryPool::adopt: tensor must be on CPU");
    TORCH_CHECK(tensor.dtype() == torch::kByte,
                "MemoryPool::adopt: tensor must be uint8, got ", tensor.dtype());
    tensor_ = std::move(tensor);
    base_off_ = 0;
}
