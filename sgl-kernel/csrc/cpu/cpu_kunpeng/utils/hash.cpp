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

#include <ATen/ATen.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <torch/library.h>

#include "../adapters/register_graph_kernels.h"

namespace utils {

static inline uint64_t fnv1a_hash_bytes(const uint8_t *data, size_t len)
{
    uint64_t hash = 0xcbf29ce484222325ULL;  // FNV offset basis
    for (size_t i = 0; i < len; ++i) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= 0x100000001b3ULL;  // FNV prime
    }
    return hash;
}

static inline void hash_combine(uint64_t &seed, uint64_t value)
{
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

uint64_t tensor_hash(const at::Tensor &tensor)
{
    at::Tensor contig = tensor.contiguous().cpu();

    if (contig.numel() == 0) {
        return 0;
    }

    size_t num_bytes = contig.numel() * contig.itemsize();
    const uint8_t *data = static_cast<const uint8_t *>(contig.data_ptr());

    uint64_t hash = fnv1a_hash_bytes(data, num_bytes);

    auto sizes = contig.sizes();
    for (auto s : sizes) {
        hash_combine(hash, static_cast<uint64_t>(s));
    }

    uint64_t dtype_val = static_cast<uint64_t>(contig.scalar_type());
    hash_combine(hash, dtype_val);

    return hash;
}

}  // namespace utils

void print_hash_kunpeng(at::Tensor t, std::string name)
{
    uint64_t h = utils::tensor_hash(t);
    std::cout << "[kunpeng.print_hash" << (name.empty() ? "" : ":" + name) << "]"
              << " hash=0x" << std::hex << h << std::dec << " shape=[";
    for (int64_t i = 0; i < t.dim(); ++i) {
        if (i) std::cout << ",";
        std::cout << t.size(i);
    }
    std::cout << "] dtype=" << static_cast<int>(t.scalar_type()) << std::endl;
}

static KernelRegistrar _r_print_hash("print_hash", make_dispatch_v<decltype(&print_hash_kunpeng), &print_hash_kunpeng>);
static KernelRegistrar _r_print_hash_kunpeng("print_hash_kunpeng", make_dispatch_v<decltype(&print_hash_kunpeng), &print_hash_kunpeng>);

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m)
{
    m.def("print_hash_kunpeng(Tensor t, str name='') -> ()");
    m.impl("print_hash_kunpeng", print_hash_kunpeng);
}