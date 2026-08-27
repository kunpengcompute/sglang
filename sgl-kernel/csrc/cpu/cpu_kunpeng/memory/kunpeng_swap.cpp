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

constexpr int64_t kSwapChunkBytes = 14 * 1024 * 1024;  // 14 MB
constexpr int64_t kSwapMaxPendingEvents = 512;

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
    int *event_ptr = event_tensor.data_ptr<int>();

    for (int i = 0; i < n_chunks; i++) {
        int64_t rel_offset = static_cast<int64_t>(i) * chunk_bytes;
        int64_t size = std::min(chunk_bytes, total_bytes - rel_offset);


        int event_id = utils::kupl_get_free_event_id();
        void *dst_ptr = static_cast<char *>(dst.data_ptr()) + dst_byte_offset + rel_offset;
        const void *src_ptr =
            static_cast<const char *>(src_contig.data_ptr()) + src_byte_offset + rel_offset;
        utils::kupl_sdma_async(event_id, dst_ptr, src_ptr, static_cast<int>(size), 0);

        event_ptr[next_slot] = event_id;
        next_slot += 1;
    }

    event_num_tensor.data_ptr<int>()[0] = next_slot;

}

// ==========================================================================
// Graph-compatible SDMA event drain (graph-op entry point)
// ==========================================================================

void kupl_sdma_wait_all(at::Tensor event_tensor, at::Tensor event_num_tensor)
{

    int event_num = event_num_tensor.data_ptr<int>()[0];
    if (event_num == 0)
        return;

    auto *event_data = event_tensor.data_ptr<int>();
    for (int i = 0; i < event_num; i++) {
        utils::kupl_sdma_wait(event_data[i]);
    }
    event_num_tensor.data_ptr<int>()[0] = 0;
}

void kupl_sdma_set_kv_buffer(at::Tensor kv_buffer, at::Tensor loc, at::Tensor cache_k, at::Tensor event_tensor,
                             at::Tensor event_num_tensor)
{
    TORCH_CHECK(kv_buffer.dim() == 2, "kv_buffer must be 2D");
    TORCH_CHECK(loc.dim() == 1, "loc must be 1D");
    TORCH_CHECK(cache_k.dim() == 2, "cache_k must be 2D");
    TORCH_CHECK(cache_k.size(1) == kv_buffer.size(1), "cache_k dim-1 must match kv_buffer dim-1");
    TORCH_CHECK(cache_k.size(0) == loc.size(0), "cache_k and loc token count mismatch");
    TORCH_CHECK(kv_buffer.stride(1) == 1 && cache_k.stride(1) == 1,
                "kv_buffer and cache_k must have contiguous dim-1 (stride==1)");

    int64_t kdim = kv_buffer.size(1);
    int64_t tokens = loc.size(0);
    if (tokens == 0) return;

    int64_t elem_sz = kv_buffer.element_size();
    int64_t dst_stride = kv_buffer.stride(0) * elem_sz;
    int64_t src_stride = cache_k.stride(0) * elem_sz;
    int64_t row_bytes = kdim * elem_sz;

    uint8_t *buf = static_cast<uint8_t *>(kv_buffer.data_ptr());
    uint8_t *src = static_cast<uint8_t *>(cache_k.data_ptr());

    int event_count = event_num_tensor.item<int>();
    int next_slot = event_count;
    int *event_ptr = event_tensor.data_ptr<int>();

    if (loc.scalar_type() == at::kInt) {
        int32_t *idx = loc.data_ptr<int32_t>();
        for (int64_t i = 0; i < tokens; i++) {
            int event_id = utils::kupl_get_free_event_id();
            utils::kupl_sdma_async(event_id, buf + static_cast<int64_t>(idx[i]) * dst_stride, src + i * src_stride,
                                   static_cast<int>(row_bytes), 0);
            event_ptr[next_slot] = event_id;
            next_slot++;
        }
    } else {
        int64_t *idx = loc.data_ptr<int64_t>();
        for (int64_t i = 0; i < tokens; i++) {
            int event_id = utils::kupl_get_free_event_id();
            utils::kupl_sdma_async(event_id, buf + idx[i] * dst_stride, src + i * src_stride,
                                   static_cast<int>(row_bytes), 0);
            event_ptr[next_slot] = event_id;
            next_slot++;
        }
    }
    event_num_tensor.data_ptr<int>()[0] = next_slot;
}

void kupl_sdma_set_kv_buffer_2(at::Tensor kv_buffer, at::Tensor loc, at::Tensor k_nope, at::Tensor k_pe,
                               at::Tensor event_tensor, at::Tensor event_num_tensor)
{
    TORCH_CHECK(kv_buffer.dim() == 2, "kv_buffer must be 2D");
    TORCH_CHECK(loc.dim() == 1, "loc must be 1D");
    TORCH_CHECK(k_nope.dim() == 2, "k_nope must be 2D");
    TORCH_CHECK(k_pe.dim() == 2, "k_pe must be 2D");
    TORCH_CHECK(k_nope.size(0) == loc.size(0) && k_pe.size(0) == loc.size(0),
                "k_nope/k_pe and loc token count mismatch");
    TORCH_CHECK(k_nope.size(1) + k_pe.size(1) == kv_buffer.size(1),
                "k_nope dim-1 + k_pe dim-1 must match kv_buffer dim-1");
    TORCH_CHECK(kv_buffer.stride(1) == 1 && k_nope.stride(1) == 1 && k_pe.stride(1) == 1,
                "kv_buffer, k_nope and k_pe must have contiguous dim-1 (stride==1)");

    int64_t kdim_n = k_nope.size(1);
    int64_t kdim_p = k_pe.size(1);
    int64_t tokens = loc.size(0);
    if (tokens == 0) return;

    int64_t elem_sz = kv_buffer.element_size();
    int64_t dst_stride = kv_buffer.stride(0) * elem_sz;
    int64_t nope_stride = k_nope.stride(0) * elem_sz;
    int64_t pe_stride = k_pe.stride(0) * elem_sz;
    int64_t nope_bytes = kdim_n * elem_sz;
    int64_t pe_bytes = kdim_p * elem_sz;

    uint8_t *buf = static_cast<uint8_t *>(kv_buffer.data_ptr());
    uint8_t *nope = static_cast<uint8_t *>(k_nope.data_ptr());
    uint8_t *pe = static_cast<uint8_t *>(k_pe.data_ptr());

    int event_count = event_num_tensor.item<int>();
    int next_slot = event_count;
    int *event_ptr = event_tensor.data_ptr<int>();

    if (loc.scalar_type() == at::kInt) {
        int32_t *idx = loc.data_ptr<int32_t>();
        for (int64_t i = 0; i < tokens; i++) {
            // Long-context decode CP: foreign pages carry slot -1 and must be
            // skipped (they used to be dropped by an eager boolean-mask filter,
            // which is not graph-capture safe).
            if (idx[i] < 0) continue;
            uint8_t *dst = buf + static_cast<int64_t>(idx[i]) * dst_stride;
            int event_id = utils::kupl_get_free_event_id();
            utils::kupl_sdma_async(event_id, dst, nope + i * nope_stride,
                                   static_cast<int>(nope_bytes), 0);
            event_ptr[next_slot] = event_id;
            next_slot++;
            event_id = utils::kupl_get_free_event_id();
            utils::kupl_sdma_async(event_id, dst + nope_bytes, pe + i * pe_stride,
                                   static_cast<int>(pe_bytes), 0);
            event_ptr[next_slot] = event_id;
            next_slot++;
        }
    } else {
        int64_t *idx = loc.data_ptr<int64_t>();
        for (int64_t i = 0; i < tokens; i++) {
            if (idx[i] < 0) continue;
            uint8_t *dst = buf + idx[i] * dst_stride;
            int event_id = utils::kupl_get_free_event_id();
            utils::kupl_sdma_async(event_id, dst, nope + i * nope_stride,
                                   static_cast<int>(nope_bytes), 0);
            event_ptr[next_slot] = event_id;
            next_slot++;
            event_id = utils::kupl_get_free_event_id();
            utils::kupl_sdma_async(event_id, dst + nope_bytes, pe + i * pe_stride,
                                   static_cast<int>(pe_bytes), 0);
            event_ptr[next_slot] = event_id;
            next_slot++;
        }
    }
    event_num_tensor.data_ptr<int>()[0] = next_slot;
}

void kupl_sdma_kv_swapin(at::Tensor dst, at::Tensor src, at::Tensor event_tensor, at::Tensor event_num_tensor,
                         int64_t total_bytes)
{
    kupl_sdma_memcpy_chunked(dst, src, event_tensor, event_num_tensor,
                             0,  // dst_byte_offset
                             0,  // src_byte_offset
                             total_bytes, kSwapChunkBytes, kSwapMaxPendingEvents);
}

// ==========================================================================
// Block-wise KV swap-in (DDR -> HBM), graph-op entry point
// ==========================================================================
//
// Issues one SDMA event per block (independent of whether ids are
// consecutive), appending the events to the caller's event table.

void kupl_sdma_kv_block_swapin(at::Tensor dst_hbm, at::Tensor src_ddr, at::Tensor ddr_block_ids,
                               at::Tensor hbw_block_ids, int64_t block_bytes, at::Tensor event_tensor,
                               at::Tensor event_num_tensor)
{
    TORCH_CHECK(dst_hbm.is_contiguous(), "kupl_sdma_kv_block_swapin: dst_hbm must be contiguous");
    TORCH_CHECK(src_ddr.is_contiguous(), "kupl_sdma_kv_block_swapin: src_ddr must be contiguous");
    TORCH_CHECK(ddr_block_ids.dim() == 1 && hbw_block_ids.dim() == 1,
                "kupl_sdma_kv_block_swapin: block id tensors must be 1D");
    TORCH_CHECK(ddr_block_ids.size(0) == hbw_block_ids.size(0),
                "kupl_sdma_kv_block_swapin: ddr_block_ids and hbw_block_ids length mismatch");
    TORCH_CHECK(block_bytes > 0, "kupl_sdma_kv_block_swapin: block_bytes must be positive");
    TORCH_CHECK(dst_hbm.dtype() == src_ddr.dtype(), "kupl_sdma_kv_block_swapin: dtype mismatch (dst=", dst_hbm.dtype(),
                ", src=", src_ddr.dtype(), ")");
    TORCH_CHECK(ddr_block_ids.scalar_type() == at::kInt && hbw_block_ids.scalar_type() == at::kInt,
                "kupl_sdma_kv_block_swapin: block id tensors must be int32");

    int64_t n_blocks = ddr_block_ids.size(0);
    if (n_blocks == 0) return;

    uint8_t *dst_base = static_cast<uint8_t *>(dst_hbm.data_ptr());
    const uint8_t *src_base = static_cast<const uint8_t *>(src_ddr.data_ptr());
    int64_t dst_capacity = dst_hbm.numel() * dst_hbm.element_size();
    int64_t src_capacity = src_ddr.numel() * src_ddr.element_size();

    int event_count = event_num_tensor.item<int>();
    int next_slot = event_count;
    int *event_ptr = event_tensor.data_ptr<int>();

    const int32_t *ddr_ids = ddr_block_ids.data_ptr<int32_t>();
    const int32_t *hbw_ids = hbw_block_ids.data_ptr<int32_t>();

    for (int64_t i = 0; i < n_blocks; i++) {
        int64_t ddr_offset = static_cast<int64_t>(ddr_ids[i]) * block_bytes;
        int64_t hbw_offset = static_cast<int64_t>(hbw_ids[i]) * block_bytes;
        TORCH_CHECK(hbw_offset + block_bytes <= dst_capacity, "kupl_sdma_kv_block_swapin: dst too small for hbw block ",
                    hbw_ids[i]);
        TORCH_CHECK(ddr_offset + block_bytes <= src_capacity, "kupl_sdma_kv_block_swapin: src too small for ddr block ",
                    ddr_ids[i]);
        int event_id = utils::kupl_get_free_event_id();
        utils::kupl_sdma_async(event_id, dst_base + hbw_offset, src_base + ddr_offset, static_cast<int>(block_bytes),
                               0);
        event_ptr[next_slot] = event_id;
        next_slot++;
    }

    event_num_tensor.data_ptr<int>()[0] = next_slot;
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
        "kupl_sdma_kv_swapin(Tensor dst, Tensor src, "
        "Tensor(a!) event_tensor, Tensor(b!) event_num_tensor, "
        "int total_bytes) -> ()");
    m.impl("kupl_sdma_kv_swapin", kupl_sdma_kv_swapin);

    m.def(
        "kupl_sdma_kv_block_swapin(Tensor dst_hbm, Tensor src_ddr, "
        "Tensor ddr_block_ids, Tensor hbw_block_ids, "
        "int block_bytes, Tensor(a!) event_tensor, Tensor(b!) event_num_tensor) -> ()");
    m.impl("kupl_sdma_kv_block_swapin", kupl_sdma_kv_block_swapin);

    m.def(
        "kupl_sdma_set_kv_buffer(Tensor kv_buffer, Tensor loc, Tensor cache_k, "
        "Tensor(a!) event_tensor, Tensor(b!) event_num_tensor) -> ()");
    m.impl("kupl_sdma_set_kv_buffer", kupl_sdma_set_kv_buffer);

    m.def(
        "kupl_sdma_set_kv_buffer_2(Tensor kv_buffer, Tensor loc, "
        "Tensor k_nope, Tensor k_pe, "
        "Tensor(a!) event_tensor, Tensor(b!) event_num_tensor) -> ()");
    m.impl("kupl_sdma_set_kv_buffer_2", kupl_sdma_set_kv_buffer_2);
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

static KernelRegistrar _r_sdma_kv_swapin("kupl_sdma_kv_swapin",
                                         make_dispatch_v<decltype(&kupl_sdma_kv_swapin), &kupl_sdma_kv_swapin>);

static KernelRegistrar _r_sdma_kv_block_swapin(
    "kupl_sdma_kv_block_swapin", make_dispatch_v<decltype(&kupl_sdma_kv_block_swapin), &kupl_sdma_kv_block_swapin>);

static KernelRegistrar _r_sdma_set_kv_buffer(
    "kupl_sdma_set_kv_buffer", make_dispatch_v<decltype(&kupl_sdma_set_kv_buffer), &kupl_sdma_set_kv_buffer>);

static KernelRegistrar _r_sdma_set_kv_buffer_2(
    "kupl_sdma_set_kv_buffer_2", make_dispatch_v<decltype(&kupl_sdma_set_kv_buffer_2), &kupl_sdma_set_kv_buffer_2>);
