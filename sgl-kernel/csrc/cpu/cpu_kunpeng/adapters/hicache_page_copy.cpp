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
#include <cstdint>
#include <cstring>

#include <kutacc.h>

// Row-wise gather-scatter between two KV buffers with the same row width.
//
//    for i in [0, dst_indices.numel()):  dst[dst_indices[i]] = src[src_indices[i]]
//
// This is the L1 <-> L2 page mover of the Kunpeng hierarchical cache: both the
// main KV pool (L1) and the HiCache host pool (L2) live in host DDR, so the
// transfer is a pure in-memory row permutation instead of the CUDA
// `transfer_kv_*` kernels used on GPU/HIP.
//
// Interface (all CPU, all fully contiguous):
//   dst, src                 : 2D tensors, same dtype and same size(1).
//                              Row strides may differ, because the two pools
//                              have different slot counts / paddings.
//   dst_indices, src_indices : 1D int32 or int64 with equal numel. Their
//                              integer widths may differ (host slots are int64,
//                              paged device slots can be int32).
//
// Negative indices are skipped, matching set_kv_buffer_2_kunpeng: long-context
// decode CP leaves -1 holes for non-local pages, and a skip keeps the eager and
// captured behaviour identical.
//
// This operator is deliberately NOT registered into GraphOpRegistry: it is
// issued from the scheduler thread by HiCacheController when writing through /
// loading back the L2 host pool, never from model.forward, so it must not be
// replayed by the static graph.
void hicache_page_copy_kunpeng(at::Tensor dst, at::Tensor src,
                               at::Tensor dst_indices, at::Tensor src_indices)
{
    TORCH_CHECK(dst.dim() == 2, "hicache_page_copy_kunpeng: dst must be 2D, got dim=", dst.dim());
    TORCH_CHECK(src.dim() == 2, "hicache_page_copy_kunpeng: src must be 2D, got dim=", src.dim());
    TORCH_CHECK(dst.is_cpu() && src.is_cpu(),
                "hicache_page_copy_kunpeng: dst/src must be CPU tensors");
    TORCH_CHECK(dst.is_contiguous() && src.is_contiguous(),
                "hicache_page_copy_kunpeng: dst/src must be contiguous");
    TORCH_CHECK(dst.scalar_type() == src.scalar_type(),
                "hicache_page_copy_kunpeng: dtype mismatch (dst=", dst.scalar_type(),
                ", src=", src.scalar_type(), ")");
    TORCH_CHECK(dst.size(1) == src.size(1),
                "hicache_page_copy_kunpeng: row width mismatch (dst=", dst.size(1),
                ", src=", src.size(1), ")");
    TORCH_CHECK(dst_indices.dim() == 1 && src_indices.dim() == 1,
                "hicache_page_copy_kunpeng: index tensors must be 1D (dst=", dst_indices.dim(),
                ", src=", src_indices.dim(), ")");
    TORCH_CHECK(dst_indices.numel() == src_indices.numel(),
                "hicache_page_copy_kunpeng: index count mismatch (dst=", dst_indices.numel(),
                ", src=", src_indices.numel(), ")");

    // The two index tensors are allowed to use different integer widths: the
    // host allocator hands out int64 slots while the paged device allocator can
    // return int32 (alloc_decode), and both end up here in the same call.
    const bool dst_is_i32 = dst_indices.scalar_type() == at::kInt;
    const bool src_is_i32 = src_indices.scalar_type() == at::kInt;
    const bool dst_is_i64 = dst_indices.scalar_type() == at::kLong;
    const bool src_is_i64 = src_indices.scalar_type() == at::kLong;
    TORCH_CHECK(dst_is_i32 || dst_is_i64,
                "hicache_page_copy_kunpeng: dst_indices must be int32 or int64, got ",
                dst_indices.scalar_type());
    TORCH_CHECK(src_is_i32 || src_is_i64,
                "hicache_page_copy_kunpeng: src_indices must be int32 or int64, got ",
                src_indices.scalar_type());

    const int64_t rows = dst_indices.numel();
    if (rows == 0) {
        return;
    }

    const int64_t src_rows = src.size(0);
    const int64_t dst_rows = dst.size(0);
    const int64_t elem_sz = dst.element_size();
    const int64_t row_bytes = dst.size(1) * elem_sz;
    const int64_t src_stride_bytes = src.stride(0) * elem_sz;
    const int64_t dst_stride_bytes = dst.stride(0) * elem_sz;

    const int32_t *d32 = dst_is_i32 ? dst_indices.data_ptr<int32_t>() : nullptr;
    const int64_t *d64 = dst_is_i32 ? nullptr : dst_indices.data_ptr<int64_t>();
    const int32_t *s32 = src_is_i32 ? src_indices.data_ptr<int32_t>() : nullptr;
    const int64_t *s64 = src_is_i32 ? nullptr : src_indices.data_ptr<int64_t>();

    auto read_dst_index = [&](int64_t i) -> int64_t {
        return dst_is_i32 ? static_cast<int64_t>(d32[i]) : d64[i];
    };
    auto read_src_index = [&](int64_t i) -> int64_t {
        return src_is_i32 ? static_cast<int64_t>(s32[i]) : s64[i];
    };

    // Validate every index up front (O(rows), rows is bounded by the pool size)
    // so that the parallel body below can never throw from a worker thread.
    int64_t max_dst = -1;
    int64_t max_src = -1;
    for (int64_t i = 0; i < rows; i++) {
        const int64_t di = read_dst_index(i);
        const int64_t si = read_src_index(i);
        max_dst = di > max_dst ? di : max_dst;
        max_src = si > max_src ? si : max_src;
    }
    TORCH_CHECK(max_dst < dst_rows,
                "hicache_page_copy_kunpeng: dst index out of range (max=", max_dst,
                ", dst_rows=", dst_rows, ")");
    TORCH_CHECK(max_src < src_rows,
                "hicache_page_copy_kunpeng: src index out of range (max=", max_src,
                ", src_rows=", src_rows, ")");

    const uint8_t *src_ptr = static_cast<const uint8_t *>(src.data_ptr());
    uint8_t *dst_ptr = static_cast<uint8_t *>(dst.data_ptr());

    // Each row writes a disjoint destination slot, so the iterations are
    // independent and safe to split across threads.
    kutacc::parallel_for(0, rows, 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; i++) {
            const int64_t di = read_dst_index(i);
            const int64_t si = read_src_index(i);
            if (di < 0 || si < 0) {
                continue;  // non-local page hole, nothing to move
            }
            std::memcpy(dst_ptr + di * dst_stride_bytes, src_ptr + si * src_stride_bytes, row_bytes);
        }
    });
}

// ---------------------------------------------------------------------------
// L2 <-> L3 page (de)serialization for the layer_first MLA host pool.
//
// The storage backend stores one page as a flat blob whose encoding is
// (layer, token, 1, kv_dim) flattened. For layer_first the page slice lives at
// kv_buffer[layer, index:index+page_size, 0, :], which is NOT contiguous across
// layers, so producing / consuming that blob needs a real copy.
//
// Both ops below are deliberately SERIAL (no kutacc::parallel_for, no torch
// elementwise op): they are called from HiCache's prefetch/backup threads, which
// are plain Python threads that libkupl does not know about, and a parallel copy
// issued from such a thread has been observed to segfault inside
// kupl_parallel_for. The work is small anyway -- because a page slice is
// contiguous *within* one layer, each op is just one memcpy per layer
// (page_size * kv_dim elements), i.e. 61 memcpys of 72 KB for a DeepSeek page.
// ---------------------------------------------------------------------------

void hicache_page_flatten_kunpeng(at::Tensor kv_buffer, at::Tensor out,
                                 int64_t index, int64_t page_size)
{
    constexpr const char *op = "hicache_page_flatten_kunpeng";
    TORCH_CHECK(kv_buffer.dim() == 4, op, ": kv_buffer must be 4D (layer, slots, 1, kv_dim), got dim=", kv_buffer.dim());
    TORCH_CHECK(kv_buffer.size(2) == 1, op, ": kv_buffer dim-2 must be 1, got ", kv_buffer.size(2));
    TORCH_CHECK(kv_buffer.is_cpu() && kv_buffer.is_contiguous(), op, ": kv_buffer must be a contiguous CPU tensor");
    TORCH_CHECK(out.dim() == 1, op, ": out must be 1D, got dim=", out.dim());
    TORCH_CHECK(out.is_cpu() && out.is_contiguous(), op, ": out must be a contiguous CPU tensor");
    TORCH_CHECK(out.scalar_type() == kv_buffer.scalar_type(),
                op, ": dtype mismatch (out=", out.scalar_type(), ", kv_buffer=", kv_buffer.scalar_type(), ")");
    TORCH_CHECK(page_size > 0, op, ": page_size must be positive, got ", page_size);

    const int64_t layers = kv_buffer.size(0);
    const int64_t slots = kv_buffer.size(1);
    const int64_t kv_dim = kv_buffer.size(3);
    const int64_t page_bytes = page_size * kv_dim * kv_buffer.element_size();
    TORCH_CHECK(index >= 0 && index + page_size <= slots,
                op, ": page out of range (index=", index, ", page_size=", page_size, ", slots=", slots, ")");
    TORCH_CHECK(out.numel() * out.element_size() == layers * page_bytes,
                op, ": out byte size mismatch (out=", out.numel() * out.element_size(),
                ", expected=", layers * page_bytes, ")");

    const uint8_t *src = static_cast<const uint8_t *>(kv_buffer.data_ptr());
    uint8_t *dst = static_cast<uint8_t *>(out.data_ptr());
    const int64_t layer_stride_bytes = slots * kv_dim * kv_buffer.element_size();

    for (int64_t layer = 0; layer < layers; layer++) {
        std::memcpy(dst + layer * page_bytes,
                    src + layer * layer_stride_bytes + index * kv_dim * kv_buffer.element_size(),
                    page_bytes);
    }
}

void hicache_page_unflatten_kunpeng(at::Tensor kv_buffer, at::Tensor flat,
                                   int64_t index, int64_t page_size)
{
    constexpr const char *op = "hicache_page_unflatten_kunpeng";
    TORCH_CHECK(kv_buffer.dim() == 4, op, ": kv_buffer must be 4D (layer, slots, 1, kv_dim), got dim=", kv_buffer.dim());
    TORCH_CHECK(kv_buffer.size(2) == 1, op, ": kv_buffer dim-2 must be 1, got ", kv_buffer.size(2));
    TORCH_CHECK(kv_buffer.is_cpu() && kv_buffer.is_contiguous(), op, ": kv_buffer must be a contiguous CPU tensor");
    TORCH_CHECK(flat.dim() == 1, op, ": flat must be 1D, got dim=", flat.dim());
    TORCH_CHECK(flat.is_cpu() && flat.is_contiguous(), op, ": flat must be a contiguous CPU tensor");
    TORCH_CHECK(flat.scalar_type() == kv_buffer.scalar_type(),
                op, ": dtype mismatch (flat=", flat.scalar_type(), ", kv_buffer=", kv_buffer.scalar_type(), ")");
    TORCH_CHECK(page_size > 0, op, ": page_size must be positive, got ", page_size);

    const int64_t layers = kv_buffer.size(0);
    const int64_t slots = kv_buffer.size(1);
    const int64_t kv_dim = kv_buffer.size(3);
    const int64_t page_bytes = page_size * kv_dim * kv_buffer.element_size();
    TORCH_CHECK(index >= 0 && index + page_size <= slots,
                op, ": page out of range (index=", index, ", page_size=", page_size, ", slots=", slots, ")");
    TORCH_CHECK(flat.numel() * flat.element_size() == layers * page_bytes,
                op, ": flat byte size mismatch (flat=", flat.numel() * flat.element_size(),
                ", expected=", layers * page_bytes, ")");

    const uint8_t *src = static_cast<const uint8_t *>(flat.data_ptr());
    uint8_t *dst = static_cast<uint8_t *>(kv_buffer.data_ptr());
    const int64_t layer_stride_bytes = slots * kv_dim * kv_buffer.element_size();

    for (int64_t layer = 0; layer < layers; layer++) {
        std::memcpy(dst + layer * layer_stride_bytes + index * kv_dim * kv_buffer.element_size(),
                    src + layer * page_bytes,
                    page_bytes);
    }
}