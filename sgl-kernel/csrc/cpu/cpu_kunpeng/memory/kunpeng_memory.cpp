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
#include <errno.h>
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

    int cpu_node = numa_node_of_cpu(sched_getcpu());
    int hbm_node = cpu_node + 16;
    unsigned long mask = 1UL << hbm_node;
    unsigned long previous_mask = 0;
    int previous_mode = MPOL_DEFAULT;
    constexpr auto mask_bits = sizeof(mask) * 8;
    if (get_mempolicy(&previous_mode, &previous_mask, mask_bits, nullptr, 0) != 0)
        throw std::runtime_error("Cannot read allocating thread's NUMA policy");
    if (set_mempolicy(MPOL_BIND, &mask, mask_bits) != 0)
        throw std::runtime_error("Cannot bind HBW allocation to HBM");

    // Hugepage reservation also obeys the allocating thread's policy. Restore
    // the caller's policy immediately so later DDR/KV allocations stay local.
    auto *hbm = static_cast<uint8_t *>(mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0));
    const int mapping_errno = errno;
    const int restored = set_mempolicy(previous_mode,
        previous_mode == MPOL_DEFAULT ? nullptr : &previous_mask, mask_bits);
    if (restored != 0) {
        if (hbm != MAP_FAILED) munmap(hbm, aligned_size);
        throw std::runtime_error("Cannot restore allocating thread's NUMA policy");
    }
    if (hbm == MAP_FAILED) {
        std::cerr << "HBW mmap failed: " << strerror(mapping_errno) << std::endl;
        throw std::runtime_error("mmap with MAP_HUGETLB failed");
    }

    int ret = mbind(hbm, aligned_size, MPOL_BIND, &mask, sizeof(mask) * 8, MPOL_MF_STRICT);
    if (ret != 0) {
        std::cerr << "    FAILED: errno=" << errno << " (" << strerror(errno) << ")" << std::endl;
        munmap(hbm, aligned_size);
        throw std::runtime_error("mbind to HBM NUMA node failed");
    }

    // Pre-mark the whole pool MADV_DONTFORK (same workaround as the SHM
    // pool): ibv_reg_mr sub-range DONTFORK calls then no-op instead of
    // splitting misaligned hugetlb VMAs (EINVAL). Buffers are RDMA-registered.
    if (madvise(hbm, aligned_size, MADV_DONTFORK) != 0) {
        std::cerr << "[KuTACC] WARNING: madvise(MADV_DONTFORK) on HBW pool failed: " << errno << " (" << strerror(errno)
                  << "), RDMA reg_mr on HBW buffers may fail" << std::endl;
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
