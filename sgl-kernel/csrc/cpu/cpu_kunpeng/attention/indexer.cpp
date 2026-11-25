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

#include <kutacc.h>

#ifndef SGLANG_KUTACC_HAS_DSA_INDEXER
#define SGLANG_KUTACC_HAS_DSA_INDEXER 1
#endif
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

#include "fp8_quant.h"
#include "../utils/utils.h"

// ---------------------------------------------------------------------------
// DSA indexer scoring kernel (kutacc adapter, bfloat16 path).
//
// Semantics (identical to kutacc::paged_mqa_logits_topk bf16 overload):
//   score[ti, si] = sum_h max(dot(q[ti, h], k[slot(b, si)]), 0) * weight[ti, h]
//     - per-head ReLU gating, fp32 accumulation, NO pre-scaling (global
//       positive constants do not change the top-k ranking; NPU bf16 path).
//   - sequence lookup: accumulate cur_lens to find the owning sequence b of
//     query row ti, pos = ti - offset (cur_lens = query rows per sequence:
//     1 for decode, extend_seq_lens for extend phases).
//   - causal visibility (bf16 uncompressed): k_end = seq_lens[b] - cur_lens[b] + pos + 1.
//   - slot(b, si) = block_tables[b, si / block_size] * block_size + (si % block_size).
//   - top-k: score descending, ties broken by the smaller id; -1 padding.
//
// Tensor contract (torch ops are direct-write, graph-replay friendly):
//   q             [T, H, D]                 bf16 (indexer query, post-RoPE)
//   cur_kv_cache  [num_blocks, block_size, D] bf16 paged index-K cache
//   weight        [T, H]                    bf16 per query-head gate (unscaled)
//   block_tables  [B, max_blocks]           int32
//   seq_lens      [B]                       int32 KV lengths
//   cur_lens      [B]                       int32 query rows per sequence
//   output        [T, topk]                 int32 direct-write, -1 padding
//   workspace     uint8 scratch, >= get_paged_mqa_logits_topk_workspace_size()
//   topk          top-k entries per query row
//   output_coord  0 = absolute paged slot (decode consumer),
//                 1 = in-sequence position (sparse prefill consumer with
//                 key_start_loc; also feeds dsa_topk_slots_kunpeng).
//   mode          0 = kTokenBalance (prefill: the token dimension already
//                 saturates the pool), 1 = kSplitKV (decode/verify/
//                 draft-extend: splits each row's KV range across threads;
//                 the library falls back to kTokenBalance when
//                 row_rows > n_threads -- semantically identical).
//   row_start / row_rows  Scoring row window into the EFFECTIVE row stream
//                 [0, sum(cur_lens)) (TP-sharded scoring: each attn-TP rank
//                 scores only its own window on the shared-shape inputs;
//                 rows OUTSIDE the window are neither read nor written --
//                 the caller relies on this to keep one full-shape output
//                 per rank). row_rows < 0 = "from row_start to the end of
//                 the stream"; defaults (0, -1) = full stream (legacy
//                 behavior). Sequence lookup / pos / k_end / output
//                 addressing all follow GLOBAL row numbers, so a windowed
//                 call is element-for-element identical to the full call's
//                 corresponding rows (see kutacc.h attention_schedule).
// ---------------------------------------------------------------------------
void paged_mqa_logits_topk_kunpeng(at::Tensor q, at::Tensor cur_kv_cache,
                                   at::Tensor weight, at::Tensor block_tables,
                                   at::Tensor seq_lens, at::Tensor cur_lens,
                                   at::Tensor output, at::Tensor workspace,
                                   int64_t topk, int64_t output_coord,
                                   int64_t mode, int64_t row_start,
                                   int64_t row_rows)
{
#if SGLANG_KUTACC_HAS_DSA_INDEXER
    auto kt_q = to_kutacc<bfloat16_t, 3>(q);
    auto kt_kv = to_kutacc<bfloat16_t, 3>(cur_kv_cache);
    auto kt_weight = to_kutacc<bfloat16_t, 2>(weight);
    auto kt_block_tables = to_kutacc<int, 2>(block_tables);
    auto kt_seq = to_kutacc<int, 1>(seq_lens);
    auto kt_cur = to_kutacc<int, 1>(cur_lens);
    auto kt_output = to_kutacc<int, 2>(output);
    auto kt_workspace = to_kutacc<uint8_t, 1>(workspace);

    kutacc::attention_schedule schedule;
    schedule.mode = mode == 1 ? kutacc::attention_schedule::Mode::kSplitKV
                              : kutacc::attention_schedule::Mode::kTokenBalance;
    schedule.output_coord = output_coord == 1
                                ? kutacc::attention_schedule::OutputCoord::kPosition
                                : kutacc::attention_schedule::OutputCoord::kSlot;
    schedule.row_start = static_cast<int32_t>(row_start);
    schedule.row_rows = static_cast<int32_t>(row_rows);

    // Windowed (TP-sharded) scoring only writes the rows inside this rank's
    // window; rows outside keep whatever the buffer held from a previous
    // batch (raw heap bytes / stale index ids) and a NaN query row produces no
    // candidate at all, so the consumer would read that residue as KV slot
    // ids. Pre-fill with the "no candidate" sentinel -1 so every unwritten
    // row is a no-op for the consumer's pos >= 0 filter.
    if (row_start != 0 || row_rows >= 0) {
        int32_t *out_p = output.data_ptr<int32_t>();
        std::fill(out_p, out_p + output.numel(), -1);
    }

    kutacc::paged_mqa_logits_topk(kt_q, kt_kv, kt_weight, kt_block_tables,
                                  kt_seq, kt_cur, kt_output, schedule,
                                  kt_workspace, static_cast<int>(topk));

#else
    TORCH_CHECK(false, "This KuTACC build does not provide SGLANG_KUTACC_HAS_DSA_INDEXER; install a matching KuTACC release to use paged_mqa_logits_topk_kunpeng");

#endif
}

// Workspace bytes for paged_mqa_logits_topk_kunpeng: one (fp32 score, int id)
// pair (8 bytes) per visible KV position per worker thread. max_k_end is an
// upper bound of the visible KV count of any query row (seq_lens[b] for the
// bfloat16 uncompressed path).
int64_t get_paged_mqa_logits_topk_workspace_size(int64_t max_k_end)
{
#if SGLANG_KUTACC_HAS_DSA_INDEXER
    TORCH_CHECK(max_k_end > 0, "max_k_end must be positive, got ", max_k_end);
    return kutacc::get_thread_num() * max_k_end * 8;

#else
    TORCH_CHECK(false, "This KuTACC build does not provide SGLANG_KUTACC_HAS_DSA_INDEXER; install a matching KuTACC release to use get_paged_mqa_logits_topk_workspace_size");

#endif
}

// ---------------------------------------------------------------------------
// FP8 (e4m3) index-K cache path.
//
// Storage: per-layer uint8 code buffer [num_slots, D] plus one ue8m0 scale
// byte per absolute slot -- a single scale covers the whole [D] token row
// (the DSA index head_dim D is fixed at 128): scale = 2^(b-127), the exact power-of-two ceil of amax/448
// (ceil >= amax/448, so nothing clips). The ue8m0 scale row is
// deliberately DDR-resident and never mirrored into HBW -- the kernel
// reads it straight from DDR (1 B/token, negligible); the HBW residency
// (swap-in HBM mirror) covers only the code rows. Scoring runs the kutacc fp8
// overload: decode e4m3 -> bf16 (exact), decode the ue8m0 byte to 2^(b-127)
// and fold the per-slot scale after the head sum inside the kernel; q stays
// bf16.
// ---------------------------------------------------------------------------

#if defined(KUTACC_FEATURE_INDEXER_FP8) && SGLANG_KUTACC_HAS_DSA_INDEXER
#define SGLANG_KUTACC_HAS_DSA_INDEXER_FP8 1
#else
#define SGLANG_KUTACC_HAS_DSA_INDEXER_FP8 0
#endif

// Real DSA indexer scoring kernel (kutacc adapter, fp8 e4m3 index-K cache).
// Tensor contract mirrors paged_mqa_logits_topk_kunpeng with cur_kv_cache a
// uint8 code tensor [num_blocks, block_size, D] and one extra uint8 k_scale
// [num_slots] of ue8m0 scale bytes (scale = 2^(b-127), indexed by absolute
// slot; byte 0 decodes to 0.0f).
void paged_mqa_logits_topk_fp8_kunpeng(at::Tensor q, at::Tensor cur_kv_cache,
                                       at::Tensor k_scale, at::Tensor weight,
                                       at::Tensor block_tables, at::Tensor seq_lens,
                                       at::Tensor cur_lens, at::Tensor output,
                                       at::Tensor workspace, int64_t topk,
                                       int64_t output_coord, int64_t mode,
                                       int64_t row_start, int64_t row_rows)
{
#if SGLANG_KUTACC_HAS_DSA_INDEXER_FP8
    auto kt_q = to_kutacc<bfloat16_t, 3>(q);
    auto kt_kv = to_kutacc<uint8_t, 3>(cur_kv_cache);
    auto kt_ks = to_kutacc<uint8_t, 1>(k_scale);
    auto kt_weight = to_kutacc<bfloat16_t, 2>(weight);
    auto kt_block_tables = to_kutacc<int, 2>(block_tables);
    auto kt_seq = to_kutacc<int, 1>(seq_lens);
    auto kt_cur = to_kutacc<int, 1>(cur_lens);
    auto kt_output = to_kutacc<int, 2>(output);
    auto kt_workspace = to_kutacc<uint8_t, 1>(workspace);

    kutacc::attention_schedule schedule;
    schedule.mode = mode == 1 ? kutacc::attention_schedule::Mode::kSplitKV
                              : kutacc::attention_schedule::Mode::kTokenBalance;
    schedule.output_coord = output_coord == 1
                                ? kutacc::attention_schedule::OutputCoord::kPosition
                                : kutacc::attention_schedule::OutputCoord::kSlot;
    schedule.row_start = static_cast<int32_t>(row_start);
    schedule.row_rows = static_cast<int32_t>(row_rows);

    // Windowed (TP-sharded) calls pre-fill the whole output with -1 (same
    // rationale as the bf16 adapter).
    if (row_start != 0 || row_rows >= 0) {
        int32_t *out_p = output.data_ptr<int32_t>();
        std::fill(out_p, out_p + output.numel(), -1);
    }

    kutacc::paged_mqa_logits_topk(kt_q, kt_kv, kt_ks, kt_weight, kt_block_tables,
                                  kt_seq, kt_cur, kt_output, schedule,
                                  kt_workspace, static_cast<int>(topk));
#else
    TORCH_CHECK(false, "This KuTACC build does not provide the fp8 index-K overload (KUTACC_FEATURE_INDEXER_FP8); install a matching KuTACC release to use paged_mqa_logits_topk_fp8_kunpeng");
#endif
}

// Store-side quantization: bf16 index-K rows [T, D] -> e4m3 codes written
// into the uint8 pool buffer at flat slots `loc`, plus one ue8m0 scale byte
// per row (scale = 2^ceil(log2(amax/448)), see fp8_quant.h) covering the
// whole [D] row. loc < 0 (padding / foreign rows) is skipped, same contract
// contract as set_kv_buffer_kunpeng, so the write is graph-replay safe.
void set_index_k_fp8_buffer_kunpeng(at::Tensor kv_buffer, at::Tensor k_scale_buffer,
                                    at::Tensor loc, at::Tensor cache_k)
{
    TORCH_CHECK(kv_buffer.dim() == 2 && kv_buffer.scalar_type() == at::kByte,
                "kv_buffer must be a 2D uint8 tensor");
    TORCH_CHECK(k_scale_buffer.dim() == 1 && k_scale_buffer.scalar_type() == at::kByte,
                "k_scale_buffer must be a 1D uint8 (ue8m0 scale byte) tensor");
    TORCH_CHECK(loc.dim() == 1, "loc must be 1D");
    TORCH_CHECK(cache_k.dim() == 2 && cache_k.scalar_type() == at::kBFloat16,
                "cache_k must be a 2D bf16 tensor");
    const int64_t tokens = loc.size(0);
    const int64_t kdim = kv_buffer.size(1);
    TORCH_CHECK(cache_k.size(0) == tokens, "cache_k and loc token count mismatch");
    TORCH_CHECK(cache_k.size(1) == kdim, "cache_k dim-1 must match kv_buffer dim-1");
    TORCH_CHECK(kv_buffer.stride(1) == 1 && cache_k.stride(1) == 1,
                "kv_buffer and cache_k must have contiguous dim-1 (stride==1)");
    TORCH_CHECK(kdim == 128, "fp8 index-K store requires head_dim == 128 (one ue8m0 scale byte per token row), got ", kdim);

    uint8_t *buf = kv_buffer.data_ptr<uint8_t>();
    uint8_t *scale_buf = k_scale_buffer.data_ptr<uint8_t>();
    const int64_t buf_stride = kv_buffer.stride(0);
    const int64_t src_stride = cache_k.stride(0);
    // Same 2-byte layout as __bf16; at::BFloat16 provides the float cast.
    const at::BFloat16 *src = cache_k.data_ptr<at::BFloat16>();

    auto quant_row = [&](int64_t slot, int64_t t) {
        float amax = 0.0f;
        const at::BFloat16 *row = src + t * src_stride;
        for (int64_t d = 0; d < kdim; ++d) {
            amax = std::max(amax, std::fabs(static_cast<float>(row[d])));
        }
        // All-zero row: scale byte 0 decodes to 0.0f and every code is 0, so
        // the row round-trips to 0 (inv_scale 0 keeps the divide finite).
        const uint8_t scale_byte =
            amax > 0.0f ? fp32_to_ue8m0_byte(amax / 448.0f) : 0u;
        // Exact power of two (bit trick in fp8_quant.h): the divide below
        // introduces no rounding of its own.
        const float scale = ue8m0_byte_to_f32(scale_byte);
        const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        uint8_t *dst = buf + slot * buf_stride;
        for (int64_t d = 0; d < kdim; ++d) {
            // ceil scale >= amax/448 keeps |x * inv_scale| <= 448 by
            // construction; the clamp additionally absorbs any fp32
            // rounding above 448 so the NaN codes are never produced.
            float v = static_cast<float>(row[d]) * inv_scale;
            v = std::min(448.0f, std::max(-448.0f, v));
            dst[d] = fp32_to_e4m3fn_byte(v);
        }
        scale_buf[slot] = scale_byte;
    };

    // Token rows are independent (each writes its own slot; loc values are
    // unique per token by contract), so split over the kupl executors with
    // kutacc::parallel_for, the same pattern as the kunpeng attention
    // kernels. Only the forward/graph-replay thread reaches this op (the
    // nsa_indexer store path); HiCache background threads must not call it
    // and never do.
    const bool loc_is_int = loc.scalar_type() == at::kInt;
    if (!loc_is_int) {
        TORCH_CHECK(loc.scalar_type() == at::kLong, "loc must be int32 or int64");
    }
    kutacc::parallel_for(0, tokens, 1, [&](int64_t start, int64_t end) {
        if (loc_is_int) {
            const int32_t *idx = loc.data_ptr<int32_t>();
            for (int64_t i = start; i < end; ++i) {
                if (idx[i] < 0) continue;
                quant_row(idx[i], i);
            }
        } else {
            const int64_t *idx = loc.data_ptr<int64_t>();
            for (int64_t i = start; i < end; ++i) {
                if (idx[i] < 0) continue;
                quant_row(idx[i], i);
            }
        }
    });
}
