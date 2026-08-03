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

#include <cstddef>
#include <cstdint>
#include <vector>

// A contiguous block of memory.
// Multiple TensorViews can share one StorageBuf.
enum class MemoryType : int {
    REGULAR = 0,  // regular CPU memory (DDR/HBW via external pool)
    SHM = 1,      // shared memory (KuTACC SHM pool)
};

struct StorageBuf {
    int id = -1;
    void* storage_base = nullptr;  // for identity matching in registry
    int born_op = -1;       // -1 = external input
    int death_op = -1;
    size_t size = 0;        // bytes needed (max extent over all views)
    bool in_pool = true;    // allocated in memory pool; false = external/fixed
    MemoryType memory_type = MemoryType::REGULAR;
    void* data_ptr = nullptr; // runtime data address (pool or external)
};

// A tensor view into a StorageBuf.
struct TensorView {
    int id = -1;
    int storage_id = -1;        // which StorageBuf
    int64_t storage_offset = 0; // offset in elements from storage base
    size_t numel = 0;
    size_t element_size = 0;
    int scalar_type = 0;        // c10::ScalarType enum value
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    bool is_return = false;     // is an op return value
};
