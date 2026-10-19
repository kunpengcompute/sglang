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
#include <torch/extension.h>

#include <cstdint>
#include <optional>

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

    kutacc::paged_mqa_logits_topk(kt_q, kt_kv, kt_weight, kt_block_tables,
                                  kt_seq, kt_cur, kt_output, schedule,
                                  kt_workspace, static_cast<int>(topk));
}

// Workspace bytes for paged_mqa_logits_topk_kunpeng: one (fp32 score, int id)
// pair (8 bytes) per visible KV position per worker thread. max_k_end is an
// upper bound of the visible KV count of any query row (seq_lens[b] for the
// bfloat16 uncompressed path).
int64_t get_paged_mqa_logits_topk_workspace_size(int64_t max_k_end)
{
    TORCH_CHECK(max_k_end > 0, "max_k_end must be positive, got ", max_k_end);
    return kutacc::get_thread_num() * max_k_end * 8;
}
