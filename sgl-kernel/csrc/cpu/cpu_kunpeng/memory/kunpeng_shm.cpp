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
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <algorithm>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <unistd.h>
#include <arm_bf16.h>
#include <arm_fp16.h>

#include "sgl_kernel_ops.h"
#include "../utils/kunpeng_oob.h"
#include "../memory/kunpeng_shm.h"
#include <kutacc.h>

// The shm size must be aligned to 2MB
constexpr int64_t SHM_ALIGNMENT = 2 * 1024 * 1024;
static int64_t g_shm_size = 0;

kupl_shm_win_h kupl_win_intra_node;
kupl_shm_win_h kupl_win_intra_socket;
kupl_shm_win_h kupl_win_intra_die;

kupl_shm_comm_h kupl_intra_node_comm;
kupl_shm_comm_h kupl_intra_socket_comm;
kupl_shm_comm_h kupl_intra_die_comm;

void *kupl_shm_baseptr;
void *kupl_intra_socketfence_baseptr;
void *kupl_intra_diefence_baseptr;
std::vector<void *> kupl_shm_group_baseptr;

static int intra_node_size = 0;
static int intra_node_rank = 0;
static int intra_die_size = 0;
static int intra_die_rank = 0;
static int intra_socket_size = 0;
static int intra_socket_rank = 0;

static c10d::ProcessGroup *g_intra_node_group = nullptr;
static c10d::ProcessGroup *g_intra_die_group = nullptr;
static c10d::ProcessGroup *g_intra_socket_group = nullptr;

static bool g_shm_initialized = false;

static int64_t g_max_seq_num = 0;
static int64_t g_max_cur_len = 0;
static int64_t g_max_tokens = 0;

template <typename T>
struct span {
    T *ptr;
    size_t size;

    span subspan(size_t offset, size_t count = size_t(-1u))
    {
        if (count == size_t(-1u)) {
            TORCH_CHECK(offset <= size, "offset > size");
            return {ptr + offset, size - offset};
        } else {
            TORCH_CHECK(offset + count <= size, "offset + count > size");
            return {ptr + offset, count};
        }
    }
};
using u8span = span<uint8_t>;

u8span shm_pool;
u8span shm_available;

void shm_pool_create_kunpeng(int64_t intra_node_pg, int64_t intra_socket_pg, int64_t intra_die_pg, int64_t shm_size_mb)
{
    if (g_shm_initialized) return;

    TORCH_CHECK(shm_size_mb > 0, "shm_size_mb must be positive, got ", shm_size_mb);
    int64_t raw_size = static_cast<int64_t>(shm_size_mb) * 1024 * 1024;
    g_shm_size = (raw_size + SHM_ALIGNMENT - 1) / SHM_ALIGNMENT * SHM_ALIGNMENT;

    g_intra_node_group = reinterpret_cast<c10d::ProcessGroup *>(intra_node_pg);
    TORCH_CHECK(g_intra_node_group != nullptr, "intra_node_pg pointer is null");

    g_intra_die_group = reinterpret_cast<c10d::ProcessGroup *>(intra_die_pg);
    TORCH_CHECK(g_intra_die_group != nullptr, "intra_die_pg pointer is null");

    g_intra_socket_group = reinterpret_cast<c10d::ProcessGroup *>(intra_socket_pg);
    TORCH_CHECK(g_intra_socket_group != nullptr, "intra_socket_pg pointer is null");

    intra_node_size = g_intra_node_group->getSize();
    intra_node_rank = g_intra_node_group->getRank();

    intra_die_size = g_intra_die_group->getSize();
    intra_die_rank = g_intra_die_group->getRank();

    intra_socket_size = g_intra_socket_group->getSize();
    intra_socket_rank = g_intra_socket_group->getRank();

    TORCH_CHECK(intra_node_size == 16, "intra_node_size != 16");
    TORCH_CHECK(intra_socket_size == 8, "intra_socket_size != 8");
    TORCH_CHECK(intra_die_size == 4, "intra_die_size != 4");

    int pid = getpid();

    kupl_shm_oob_cb_t oob_cbs;
    kupl_shm_oob_cb_h oob_cbs_h = &oob_cbs;
    oob_cbs_h->oob_allgather = kunpeng_oob::kupl_shm_oob_allgather;
    oob_cbs_h->oob_barrier = kunpeng_oob::kupl_shm_oob_barrier;

    kupl_shm_comm_create(intra_node_size, intra_node_rank, pid, oob_cbs_h, g_intra_node_group, &kupl_intra_node_comm);
    kupl_shm_comm_create(intra_socket_size, intra_socket_rank, pid, oob_cbs_h, g_intra_socket_group,
                         &kupl_intra_socket_comm);
    kupl_shm_comm_create(intra_die_size, intra_die_rank, pid, oob_cbs_h, g_intra_die_group, &kupl_intra_die_comm);

    kupl_shm_win_alloc(64, kupl_intra_socket_comm, (void **)&kupl_intra_socketfence_baseptr, &kupl_win_intra_socket);
    kupl_shm_win_alloc(64, kupl_intra_die_comm, (void **)&kupl_intra_diefence_baseptr, &kupl_win_intra_die);
    kupl_shm_win_alloc(g_shm_size, kupl_intra_node_comm, (void **)&kupl_shm_baseptr, &kupl_win_intra_node);

    memset(kupl_shm_baseptr, 0, g_shm_size);

    auto work = g_intra_node_group->barrier();
    work->wait();

    kupl_shm_group_baseptr.resize(intra_node_size, nullptr);
    for (int i = 0; i < intra_node_size; ++i) {
        kupl_shm_win_query(kupl_win_intra_node, i, (void **)&kupl_shm_group_baseptr[i]);
    }

    auto shm = reinterpret_cast<uint8_t *>(kupl_shm_baseptr);
    shm_pool = shm_available = u8span{shm, static_cast<size_t>(g_shm_size)};

    std::cout << "[KuTACC] Init shared memory pool, shm_size= " << shm_available.size << std::endl;

    g_shm_initialized = true;

    auto str = std::getenv("SGLANG_KUNPENG_MAX_SEQ_NUM");
    if (str != nullptr) {
        g_max_seq_num = std::atoll(str);
    }
    str = std::getenv("SGLANG_KUNPENG_MAX_CUR_LEN");
    if (str != nullptr) {
        g_max_cur_len = std::atoll(str);
    }
    TORCH_CHECK(g_max_seq_num > 0, "SGLANG_KUNPENG_MAX_SEQ_NUM must be set to a positive value");
    TORCH_CHECK(g_max_cur_len > 0, "SGLANG_KUNPENG_MAX_CUR_LEN must be set to a positive value");
    g_max_tokens = g_max_seq_num * g_max_cur_len;
}

void shm_pool_destroy_kunpeng()
{
    if (!g_shm_initialized) return;

    kupl_shm_win_free(kupl_win_intra_node);
    kupl_shm_win_free(kupl_win_intra_socket);
    kupl_shm_win_free(kupl_win_intra_die);

    kupl_shm_comm_destroy(kupl_intra_node_comm);
    kupl_shm_comm_destroy(kupl_intra_socket_comm);
    kupl_shm_comm_destroy(kupl_intra_die_comm);

    g_shm_initialized = false;
}

bool is_shm(void *ptr)
{
    return reinterpret_cast<size_t>(ptr) >= reinterpret_cast<size_t>(kupl_shm_group_baseptr[intra_node_rank]) &&
           reinterpret_cast<size_t>(ptr) <
               reinterpret_cast<size_t>(kupl_shm_group_baseptr[intra_node_rank]) + g_shm_size;
}

bool is_shm_tensor(at::Tensor tensor)
{
    void *ptr = tensor.data_ptr();
    return is_shm(ptr);
}

void get_peer_shm_baseptr(int64_t peer_rank, void *local_base_ptr, void **remote_base_ptr)
{
    TORCH_CHECK(is_shm(local_base_ptr), "input pointer is not shared memory");
    *remote_base_ptr = (char *)kupl_shm_group_baseptr[peer_rank] +
                       ((char *)local_base_ptr - (char *)kupl_shm_group_baseptr[intra_node_rank]);
}

static inline uint8_t *alignup(uint8_t *ptr, int64_t alignment)
{
    return reinterpret_cast<uint8_t *>((reinterpret_cast<int64_t>(ptr) + alignment - 1) / alignment * alignment);
}

at::Tensor create_shm_tensor_kunpeng(at::ScalarType dtype, c10::ArrayRef<int64_t> shape)
{
    TORCH_CHECK(g_shm_initialized, "create_shm_tensor_kunpeng called before shm_pool_create_kunpeng");

    int64_t element_size = at::elementSize(dtype);
    int64_t numel = 1;
    for (auto s : shape)
        numel *= s;
    int64_t total_bytes = numel * element_size;

    int64_t alignment = 1;
    uint8_t *cur_ptr = reinterpret_cast<uint8_t *>(shm_available.ptr);
    int64_t align_gap = alignup(cur_ptr, alignment) - cur_ptr;
    uint8_t *allocated_ptr = cur_ptr + align_gap;
    int64_t needed = total_bytes + align_gap;

    TORCH_CHECK(needed <= static_cast<int64_t>(shm_available.size), "Not enough shared memory, need ", needed,
                ", remain ", shm_available.size);

    shm_available = shm_available.subspan(needed);

    std::cout << "[KuTACC] Allocated " << total_bytes << " bytes from shared memory pool, remaining "
              << shm_available.size << std::endl;

    auto torch_tensor = at::from_blob(allocated_ptr, shape, at::TensorOptions().dtype(dtype));
    torch_tensor.set_requires_grad(false);

    return torch_tensor;
}

static std::unordered_map<int64_t, at::Tensor> g_shm_tensor_cache_small;
static std::unordered_map<int64_t, at::Tensor> g_shm_tensor_cache_large;

at::Tensor get_or_create_shm_tensor(int64_t dim, int64_t bs)
{
    TORCH_CHECK(g_shm_initialized, "get_or_create_shm_tensor called before shm_pool_create_kunpeng");
    TORCH_CHECK(g_max_tokens > 0, "set_shm_max_tokens must be called before get_or_create_shm_tensor");
    TORCH_CHECK(bs > 0, "bs must be a positive value, got ", bs);

    int64_t batch_size = std::max(g_max_seq_num, static_cast<int64_t>(intra_node_size));
    if (bs <= batch_size) {
        auto it = g_shm_tensor_cache_small.find(dim);
        if (it != g_shm_tensor_cache_small.end()) {
            return it->second;
        }
        int64_t shape[2] = {batch_size, dim};
        at::Tensor tensor = create_shm_tensor_kunpeng(at::kBFloat16, c10::ArrayRef<int64_t>(shape, 2));
        g_shm_tensor_cache_small[dim] = tensor;
        return tensor;
    }

    auto it = g_shm_tensor_cache_large.find(dim);
    if (it != g_shm_tensor_cache_large.end()) {
        return it->second;
    }
    int64_t shape[2] = {g_max_tokens, dim};
    at::Tensor tensor = create_shm_tensor_kunpeng(at::kBFloat16, c10::ArrayRef<int64_t>(shape, 2));
    g_shm_tensor_cache_large[dim] = tensor;
    return tensor;
}

int get_intra_node_rank()
{
    return intra_node_rank;
}

int get_intra_node_size()
{
    return intra_node_size;
}

bool is_shm_initialized()
{
    return g_shm_initialized;
}

void *alloc_shm_raw(size_t bytes)
{
    TORCH_CHECK(g_shm_initialized, "alloc_shm_raw called before shm_pool_create_kunpeng");

    int64_t alignment = 1;
    uint8_t *cur_ptr = reinterpret_cast<uint8_t *>(shm_available.ptr);
    int64_t align_gap = alignup(cur_ptr, alignment) - cur_ptr;
    uint8_t *allocated_ptr = cur_ptr + align_gap;
    int64_t needed = static_cast<int64_t>(bytes) + align_gap;

    TORCH_CHECK(needed <= static_cast<int64_t>(shm_available.size), "Not enough shared memory, need ", needed,
                ", remain ", shm_available.size);

    shm_available = shm_available.subspan(needed);

    std::cout << "[KuTACC] Allocated " << bytes << " raw bytes from shared memory pool, remaining "
              << shm_available.size << std::endl;

    return allocated_ptr;
}