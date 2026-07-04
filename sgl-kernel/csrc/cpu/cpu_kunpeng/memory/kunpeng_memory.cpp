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

#include <torch/all.h>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>

constexpr int64_t PAGE_SIZE = 2 * 1024 * 1024;  // 2MB huge page
constexpr int BENCH_LOOPS = 10;

at::Tensor hbw_allocator_kunpeng(int64_t size)
{
    int64_t aligned_size = (size + PAGE_SIZE - 1) / PAGE_SIZE * PAGE_SIZE;
    int64_t num_pages = aligned_size / PAGE_SIZE;

    // Step 1: mmap with MAP_HUGETLB + MAP_POPULATE
    auto *hbm = static_cast<uint8_t *>(mmap(NULL, aligned_size, PROT_READ | PROT_WRITE,
                                            MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0));
    if (hbm == MAP_FAILED) {
        std::cerr << "    FAILED: errno=" << errno << " (" << strerror(errno) << ")" << std::endl;
        throw std::runtime_error("mmap with MAP_HUGETLB failed");
    }

    // Step 2: mbind to HBM NUMA node
    int cpu = sched_getcpu();
    int cpu_node = numa_node_of_cpu(cpu);
    int hbm_node = cpu_node + 16;
    unsigned long mask = 1UL << hbm_node;

    int ret = mbind(hbm, aligned_size, MPOL_BIND, &mask, sizeof(mask) * 8, MPOL_MF_STRICT);
    if (ret != 0) {
        std::cerr << "    FAILED: errno=" << errno << " (" << strerror(errno) << ")" << std::endl;
        munmap(hbm, aligned_size);
        throw std::runtime_error("mbind to HBM NUMA node failed");
    }

    // Create a uint8 tensor that wraps the HBM memory directly (no copy)
    at::Tensor data_tensor = at::from_blob(hbm, {aligned_size}, at::dtype(at::kByte).device(at::kCPU));
    return data_tensor;
}

void hbw_destroy_kunpeng(at::Tensor data_tensor)
{
    void *ptr = data_tensor.data_ptr();
    if (ptr) {
        munmap(ptr, data_tensor.numel());
    }
}
