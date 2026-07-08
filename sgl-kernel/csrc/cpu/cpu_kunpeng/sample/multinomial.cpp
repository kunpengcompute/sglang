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
#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <algorithm>
#include <random>
#include <vector>

#include "common.h"

at::Tensor multinomial_kunpeng(const at::Tensor &probs, int64_t num_samples, bool replacement)
{
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(probs);
    CHECK_DIM(2, probs);
    TORCH_CHECK(probs.scalar_type() == at::kFloat, "probs must be float32, got ", probs.scalar_type());
    TORCH_CHECK(num_samples >= 1, "num_samples must be >= 1, got ", num_samples);
    if (!replacement) {
        TORCH_CHECK(num_samples <= probs.size(1),
                    "num_samples must be <= vocab_size when replacement=False, got ", num_samples, " vs ", probs.size(1));
    }

    if (num_samples == 1) {
        // Gumbel-max trick via PyTorch's vectorized exponential_ + div + argmax
        auto noise = at::empty_like(probs);
        noise.exponential_(1.0);
        auto ratio = probs / noise;
        return at::argmax(ratio, 1, true);
    }

    // num_samples > 1: cumsum + binary search
    int64_t batch = probs.size(0);
    int64_t vocab = probs.size(1);

    auto result = at::empty({batch, num_samples}, at::dtype(at::kLong));
    const float *probs_data = probs.data_ptr<float>();
    int64_t *result_data = result.data_ptr<int64_t>();
    int64_t stride = probs.stride(0);

    at::parallel_for(0, batch, 1, [&](int64_t start, int64_t end) {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> cumsum(vocab);
        std::vector<float> row_work;

        for (int64_t b = start; b < end; b++) {
            const float *row = probs_data + b * stride;

            if (replacement) {
                float total = 0.0f;
                for (int64_t j = 0; j < vocab; j++) {
                    total += row[j];
                    cumsum[j] = total;
                }
                for (int64_t s = 0; s < num_samples; s++) {
                    float r = dist(rng) * total;
                    auto it = std::lower_bound(cumsum.begin(), cumsum.end(), r);
                    result_data[b * num_samples + s] = static_cast<int64_t>(it - cumsum.begin());
                }
            } else {
                row_work.assign(row, row + vocab);
                for (int64_t s = 0; s < num_samples; s++) {
                    float total_w = 0.0f;
                    for (int64_t j = 0; j < vocab; j++) {
                        total_w += row_work[j];
                        cumsum[j] = total_w;
                    }
                    float r = dist(rng) * total_w;
                    auto it = std::lower_bound(cumsum.begin(), cumsum.end(), r);
                    int64_t idx = static_cast<int64_t>(it - cumsum.begin());
                    result_data[b * num_samples + s] = idx;
                    row_work[idx] = 0.0f;
                }
            }
        }
    });

    return result;
}
