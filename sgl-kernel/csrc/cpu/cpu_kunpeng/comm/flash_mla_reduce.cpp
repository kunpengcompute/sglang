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

// ============================================================================
// MLA long-context decode CP: merge the per-shard partial attention outputs.
//
// After the SHM exchange each rank holds, for its OWN local head block:
//   o_contrib   (cp, B, Nh_local, kv_lora_rank) bf16  partial outputs
//   lse_contrib (cp, B, Nh_local)               fp32  log-sum-exp values
//   topk_length (cp*B,)                         int32 per-(shard, seq) local
//                                                 KV counts (0 == empty shard)
// A rank whose sequence has NO local KV on a given shard contributes weight 0.
//
// This is the reference flash_mla_reduce semantics: a max-based online
// softmax reduction over the cp dimension, accumulated in fp32 and written
// back as bf16.  It replaces the eager Python where/exp/max/sum chain, which
// allocates intermediate tensors and cannot run under graph capture.
// ============================================================================

#include <torch/extension.h>

#include <arm_bf16.h>
#include <cmath>
#include <cstdint>

void flash_mla_reduce_kunpeng(at::Tensor o_contrib, at::Tensor lse_contrib,
                              at::Tensor topk_length, at::Tensor out)
{
    TORCH_CHECK(o_contrib.scalar_type() == at::kBFloat16 && o_contrib.is_contiguous(),
                "o_contrib must be contiguous bf16");
    TORCH_CHECK(lse_contrib.scalar_type() == at::kFloat && lse_contrib.is_contiguous(),
                "lse_contrib must be contiguous fp32");
    TORCH_CHECK(topk_length.scalar_type() == at::kInt && topk_length.is_contiguous(),
                "topk_length must be contiguous int32");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous(),
                "out must be contiguous bf16");
    TORCH_CHECK(o_contrib.dim() == 4, "o_contrib must be 4D (cp, B, Nh_local, D)");
    TORCH_CHECK(lse_contrib.dim() == 3, "lse_contrib must be 3D (cp, B, Nh_local)");
    TORCH_CHECK(topk_length.dim() == 1, "topk_length must be 1D (cp*B)");
    TORCH_CHECK(out.dim() == 3, "out must be 3D (B, Nh_local, D)");

    const int64_t cp = o_contrib.size(0);
    const int64_t B = o_contrib.size(1);
    const int64_t Nh = o_contrib.size(2);
    const int64_t D = o_contrib.size(3);
    TORCH_CHECK(D <= 512, "kv_lora_rank (D=", D, ") exceeds the fixed accumulator size 512");
    TORCH_CHECK(lse_contrib.size(0) == cp && lse_contrib.size(1) == B &&
                    lse_contrib.size(2) == Nh,
                "lse_contrib shape mismatch");
    TORCH_CHECK(topk_length.size(0) == cp * B, "topk_length numel mismatch");
    TORCH_CHECK(out.size(0) == B && out.size(1) == Nh && out.size(2) == D,
                "out shape mismatch");

    const bfloat16_t *o_ptr =
        reinterpret_cast<const bfloat16_t *>(o_contrib.data_ptr());
    const float *lse_ptr = reinterpret_cast<const float *>(lse_contrib.data_ptr());
    const int32_t *topk_ptr = reinterpret_cast<const int32_t *>(topk_length.data_ptr());
    bfloat16_t *out_ptr = reinterpret_cast<bfloat16_t *>(out.data_ptr());

    const int64_t o_row = Nh * D;  // per-(p, b) row of o_contrib
    const int64_t lse_row = Nh;    // per-(p, b) row of lse_contrib
    constexpr float kEmptyLse = -1e30f;  // LSE of an empty shard

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < Nh; ++h) {
            // Max over the cp shards; empty shards contribute -1e30 so they
            // never win the max.
            float mx = kEmptyLse;
            for (int64_t p = 0; p < cp; ++p) {
                const bool valid = topk_ptr[p * B + b] > 0;
                const float l = valid ? lse_ptr[(p * B + b) * lse_row + h]
                                      : kEmptyLse;
                if (l > mx) mx = l;
            }

            float acc[512];  // D == kv_lora_rank == 512
            float denom = 0.0f;
            for (int64_t d = 0; d < D; ++d) acc[d] = 0.0f;
            for (int64_t p = 0; p < cp; ++p) {
                const bool valid = topk_ptr[p * B + b] > 0;
                const float w = valid
                                    ? std::exp(lse_ptr[(p * B + b) * lse_row + h] - mx)
                                    : 0.0f;
                denom += w;
                const bfloat16_t *o_p = o_ptr + (p * B + b) * o_row + h * D;
                for (int64_t d = 0; d < D; ++d) {
                    acc[d] += static_cast<float>(o_p[d]) * w;
                }
            }

            // Sequences whose shards are ALL empty (e.g. padding rows) yield 0.
            const float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
            bfloat16_t *out_p = out_ptr + (b * Nh + h) * D;
            for (int64_t d = 0; d < D; ++d) {
                out_p[d] = static_cast<bfloat16_t>(acc[d] * inv);
            }
        }
    }
}
