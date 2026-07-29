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
#include <torch/library.h>

#include <cstring>

#include "cpu_kunpeng/adapters/register_graph_kernels.h"
#include "cpu_kunpeng/utils/sdma_util.h"
#include "cpu_kunpeng/utils/sdma_thres_util.h"

int64_t get_sdma_event_num()
{
    return static_cast<int64_t>(utils::EVENT_NUM);
}

void kupl_sdma_init_torch(int64_t sdmathreshold)
{
    SdmaCtlThredInit();
    SetSdmaThreshold(sdmathreshold);
    utils::kupl_sdma_init();
}

void kupl_sdma_clear_torch()
{
    utils::kupl_sdma_clear();
    DevmemFdDestroy();
}

int64_t kupl_get_free_event_id_torch()
{
    return static_cast<int64_t>(utils::kupl_get_free_event_id());
}

void kupl_sdma_async_torch(int64_t event_id, at::Tensor dst, at::Tensor src, int64_t dst_byte_offset,
                           int64_t src_byte_offset, int64_t byte_counts)
{
    TORCH_CHECK(dst.is_contiguous(), "kupl_sdma_async: dst must be contiguous");
    TORCH_CHECK(src.is_contiguous(), "kupl_sdma_async: src must be contiguous");
    TORCH_CHECK(dst_byte_offset >= 0, "kupl_sdma_async: dst_byte_offset must be non-negative");
    TORCH_CHECK(src_byte_offset >= 0, "kupl_sdma_async: src_byte_offset must be non-negative");
    TORCH_CHECK(byte_counts > 0, "kupl_sdma_async: byte_counts must be positive");
    void *dst_ptr = static_cast<char *>(dst.data_ptr()) + dst_byte_offset;
    const void *src_ptr = static_cast<const char *>(src.data_ptr()) + src_byte_offset;
    utils::kupl_sdma_async(static_cast<int>(event_id), dst_ptr, src_ptr, static_cast<int>(byte_counts), 0);
}

void kupl_sdma_wait_torch(int64_t event_id)
{
    utils::kupl_sdma_wait(static_cast<int>(event_id));
}

// ==========================================================================
// Graph-compatible SDMA chunked copy (graph-op entry point)
// ==========================================================================

void kupl_sdma_memcpy_chunked(at::Tensor dst, at::Tensor src,
                              at::Tensor event_tensor, at::Tensor event_num_tensor,
                              int64_t dst_byte_offset, int64_t src_byte_offset,
                              int64_t total_bytes,
                              int64_t chunk_bytes, int64_t max_pending_events)
{
    // No-op when dst and src alias the same region.
    if (dst.data_ptr() == src.data_ptr() && dst_byte_offset == src_byte_offset)
        return;

    TORCH_CHECK(dst.dtype() == src.dtype(),
                "kupl_sdma_memcpy_chunked: dtype mismatch (dst=",
                dst.dtype(), ", src=", src.dtype(), ")");
    TORCH_CHECK(dst.element_size() == src.element_size(),
                "kupl_sdma_memcpy_chunked: element size mismatch");

    // dst must already be contiguous.
    TORCH_CHECK(dst.is_contiguous(),
                "kupl_sdma_memcpy_chunked: dst must be contiguous");

    // Materialise src if it is a non-contiguous view.
    at::Tensor src_contig = src.is_contiguous() ? src : src.contiguous();

    if (total_bytes == 0)
        return;

    int64_t dst_capacity = dst.numel() * dst.element_size();
    TORCH_CHECK(dst_capacity >= dst_byte_offset + total_bytes,
                "kupl_sdma_memcpy_chunked: dst too small "
                "(dst=", dst_capacity, "B, dst_byte_offset=", dst_byte_offset,
                ", need=", dst_byte_offset + total_bytes, "B)");
    int64_t src_capacity = src_contig.numel() * src_contig.element_size();
    TORCH_CHECK(src_capacity >= src_byte_offset + total_bytes,
                "kupl_sdma_memcpy_chunked: src too small "
                "(src=", src_capacity, "B, src_byte_offset=", src_byte_offset,
                ", need=", src_byte_offset + total_bytes, "B)");

    int event_count = event_num_tensor.item<int>();
    int next_slot = event_count;
    int n_chunks = static_cast<int>((total_bytes + chunk_bytes - 1) / chunk_bytes);

    for (int i = 0; i < n_chunks; i++) {
        int64_t rel_offset = static_cast<int64_t>(i) * chunk_bytes;
        int64_t size = std::min(chunk_bytes, total_bytes - rel_offset);

        // Drain the oldest pending event to make room if saturated.
        if (next_slot >= static_cast<int>(max_pending_events)) {
            int old_id = event_tensor[0].item<int>();
            utils::kupl_sdma_wait(old_id);
            // Shift remaining pending ids left by one slot.
            if (next_slot > 1)
                std::memmove(event_tensor.data_ptr<int>(),
                             event_tensor.data_ptr<int>() + 1,
                             (next_slot - 1) * sizeof(int));
            next_slot -= 1;
        }

        int event_id = utils::kupl_get_free_event_id();
        void *dst_ptr = static_cast<char *>(dst.data_ptr()) + dst_byte_offset + rel_offset;
        const void *src_ptr =
            static_cast<const char *>(src_contig.data_ptr()) + src_byte_offset + rel_offset;
        utils::kupl_sdma_async(event_id, dst_ptr, src_ptr, static_cast<int>(size), 0);

        event_tensor[next_slot] = event_id;
        next_slot += 1;
    }

    event_num_tensor[0] = next_slot;
}

// ==========================================================================
// Graph-compatible SDMA event drain (graph-op entry point)
// ==========================================================================

void kupl_sdma_wait_all(at::Tensor event_tensor, at::Tensor event_num_tensor)
{
    int event_num = event_num_tensor.item<int>();
    if (event_num == 0)
        return;

    auto *event_data = event_tensor.data_ptr<int>();
    for (int i = 0; i < event_num; i++) {
        utils::kupl_sdma_wait(event_data[i]);
    }
    event_num_tensor.zero_();
}

void kupl_sdma_set_kv_buffer(at::Tensor kv_buffer, at::Tensor loc, at::Tensor cache_k, at::Tensor event_tensor,
                             at::Tensor event_num_tensor, int64_t max_pending_events, int64_t chunk_bytes)
{
    TORCH_CHECK(kv_buffer.dim() == 2, "kv_buffer must be 2D");
    TORCH_CHECK(loc.dim() == 1, "loc must be 1D");
    TORCH_CHECK(cache_k.dim() == 2, "cache_k must be 2D");
    TORCH_CHECK(cache_k.size(1) == kv_buffer.size(1), "cache_k dim-1 must match kv_buffer dim-1");
    TORCH_CHECK(cache_k.size(0) == loc.size(0), "cache_k and loc token count mismatch");
    TORCH_CHECK(kv_buffer.is_contiguous(), "kv_buffer must be contiguous");
    TORCH_CHECK(cache_k.is_contiguous(), "cache_k must be contiguous");
    TORCH_CHECK(kv_buffer.stride(1) == 1 && cache_k.stride(1) == 1, "kv_buffer and cache_k must have contiguous dim-1");

    int64_t tokens = loc.size(0);
    if (tokens == 0) return;

    int64_t kdim = kv_buffer.size(1);
    int64_t elem_sz = kv_buffer.element_size();
    int64_t dst_stride = kv_buffer.stride(0) * elem_sz;
    int64_t src_stride = cache_k.stride(0) * elem_sz;
    int64_t row_bytes = kdim * elem_sz;

    if (loc.scalar_type() == at::kInt) {
        int32_t *idx32 = loc.data_ptr<int32_t>();
        int64_t run_start = 0;
        while (run_start < tokens) {
            int64_t run_end = run_start + 1;
            while (run_end < tokens && idx32[run_end] == idx32[run_end - 1] + 1) {
                run_end++;
            }
            int64_t run_count = run_end - run_start;
            int64_t dst_offset = static_cast<int64_t>(idx32[run_start]) * dst_stride;
            int64_t src_offset = run_start * src_stride;
            int64_t total_bytes = run_count * row_bytes;

            kupl_sdma_memcpy_chunked(kv_buffer, cache_k, event_tensor, event_num_tensor, dst_offset, src_offset,
                                     total_bytes, chunk_bytes, max_pending_events);

            run_start = run_end;
        }
        return;
    }

    int64_t *idx_arr = loc.data_ptr<int64_t>();

    int64_t run_start = 0;
    while (run_start < tokens) {
        int64_t run_end = run_start + 1;
        while (run_end < tokens && idx_arr[run_end] == idx_arr[run_end - 1] + 1) {
            run_end++;
        }
        int64_t run_count = run_end - run_start;
        int64_t dst_offset = idx_arr[run_start] * dst_stride;
        int64_t src_offset = run_start * src_stride;
        int64_t total_bytes = run_count * row_bytes;

        kupl_sdma_memcpy_chunked(kv_buffer, cache_k, event_tensor, event_num_tensor, dst_offset, src_offset,
                                 total_bytes, chunk_bytes, max_pending_events);

        run_start = run_end;
    }
}

// ==========================================================================
// Torch library registration (eager mode)
// ==========================================================================

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m)
{
    m.def("get_sdma_event_num() -> int");
    m.impl("get_sdma_event_num", get_sdma_event_num);

    m.def("kupl_sdma_init(int sdmathreshold) -> ()");
    m.impl("kupl_sdma_init", kupl_sdma_init_torch);

    m.def("kupl_sdma_clear() -> ()");
    m.impl("kupl_sdma_clear", kupl_sdma_clear_torch);

    m.def("kupl_get_free_event_id() -> int");
    m.impl("kupl_get_free_event_id", kupl_get_free_event_id_torch);

    m.def(
        "kupl_sdma_async(int event_id, Tensor dst, Tensor src, int dst_byte_offset, int src_byte_offset, int "
        "byte_counts) -> ()");
    m.impl("kupl_sdma_async", kupl_sdma_async_torch);

    m.def("kupl_sdma_wait(int event_id) -> ()");
    m.impl("kupl_sdma_wait", kupl_sdma_wait_torch);

    m.def(
        "kupl_sdma_memcpy_chunked(Tensor dst, Tensor src, "
        "Tensor(a!) event_tensor, Tensor(b!) event_num_tensor, "
        "int dst_byte_offset, int src_byte_offset, "
        "int total_bytes, int chunk_bytes, int max_pending_events) -> ()");
    m.impl("kupl_sdma_memcpy_chunked", kupl_sdma_memcpy_chunked);

    m.def(
        "kupl_sdma_wait_all(Tensor event_tensor, Tensor(b!) event_num_tensor) -> ()");
    m.impl("kupl_sdma_wait_all", kupl_sdma_wait_all);

    m.def(
        "kupl_sdma_set_kv_buffer(Tensor kv_buffer, Tensor loc, Tensor cache_k, "
        "Tensor(a!) event_tensor, Tensor(b!) event_num_tensor, "
        "int max_pending_events, int chunk_bytes) -> ()");
    m.impl("kupl_sdma_set_kv_buffer", kupl_sdma_set_kv_buffer);
}

// ==========================================================================
// Graph-op registration (replay mode)
// ==========================================================================

static KernelRegistrar _r_sdma_memcpy_chunked(
    "kupl_sdma_memcpy_chunked",
    make_dispatch_v<decltype(&kupl_sdma_memcpy_chunked), &kupl_sdma_memcpy_chunked>);

static KernelRegistrar _r_sdma_wait_all(
    "kupl_sdma_wait_all",
    make_dispatch_v<decltype(&kupl_sdma_wait_all), &kupl_sdma_wait_all>);

static KernelRegistrar _r_sdma_set_kv_buffer(
    "kupl_sdma_set_kv_buffer", make_dispatch_v<decltype(&kupl_sdma_set_kv_buffer), &kupl_sdma_set_kv_buffer>);
