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

#include <torch/extension.h>

#include <cstddef>
#include <cstdint>

class MemoryPool {
public:
    MemoryPool() = default;
    ~MemoryPool() = default;

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&& other) noexcept = default;
    MemoryPool& operator=(MemoryPool&& other) noexcept = default;

    void allocate(size_t size);
    void* data() { return tensor_.defined() ? tensor_.data_ptr() : nullptr; }
    size_t size() const { return tensor_.defined() ? tensor_.nbytes() : 0; }

    void* ptr(size_t offset) { return static_cast<char*>(data()) + offset; }

private:
    torch::Tensor tensor_;
};
