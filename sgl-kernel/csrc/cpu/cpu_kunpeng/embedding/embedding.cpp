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

#include <torch/extension.h>
#include <kutacc.h>

#include "sgl_kernel_ops.h"

/*
 * Fused mask + embedding lookup for Kunpeng CPU.
 *
 * Replaces the two-step Python-side pattern:
 *   1) get_masked_input_and_mask() — map token IDs to local shard indices
 *   2) F.embedding() + masked_fill_() — lookup + zero out-of-shard tokens
 * with a single fused kernel.
 *
 * For each token i:
 *   if org_vocab_start <= indices[i] < org_vocab_end:
 *       output[i] = weight[indices[i] - org_vocab_start]
 *   elif added_vocab_start <= indices[i] < added_vocab_end:
 *       output[i] = weight[indices[i] - added_offset]
 *   else:
 *       output[i] = 0
 *
 * where added_offset = added_vocab_start - (org_vocab_end - org_vocab_start) - num_org_vocab_padding
 */
at::Tensor embedding_kunpeng(at::Tensor indices, at::Tensor weight, at::Tensor output, int64_t org_vocab_start,
                             int64_t org_vocab_end, int64_t num_org_vocab_padding, int64_t added_vocab_start,
                             int64_t added_vocab_end)
{
    TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [vocab_shard, hidden_dim]");
    TORCH_CHECK(output.dim() == 2, "output must be 2D [num_tokens, hidden_dim]");

    TORCH_CHECK(org_vocab_start <= org_vocab_end, "org_vocab_start must be <= org_vocab_end");

    TORCH_CHECK(num_org_vocab_padding == 0, "embedding_kunpeng does not support org_vocab_padding > 0");
    TORCH_CHECK(added_vocab_end <= added_vocab_start,
                "embedding_kunpeng does not support two concatenated vocabulary ranges (LoRA added vocab). "
                "added_vocab_end must be <= added_vocab_start, got added_vocab_start=",
                added_vocab_start, ", added_vocab_end=", added_vocab_end);

    int64_t num_tokens = indices.numel();
    int64_t hidden_dim = weight.size(1);

    kutacc::embedding(indices.data_ptr<int64_t>(), weight.data_ptr(), output.data_ptr(), weight.element_size(),
                      num_tokens, hidden_dim, org_vocab_start, org_vocab_end);

    return output;
}
