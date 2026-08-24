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
#include <kutacc.h>

#include <algorithm>
#include <vector>

#include "common.h"

// Page-aligned extend KV slot allocation for Kunpeng CPU.
//
// Mirrors the original Python loop in
// `kunpeng_allocator.py::alloc_extend_kernel_kunpeng` (per-request three
// segments: continuation from last_loc, full new pages from free_pages, and a
// final partial page tail), but vectorized over the batch with
// kutacc::parallel_for instead of a Python loop that triggers per-element
// `.item()` synchronizations under single-threaded torch.
void alloc_extend_kernel_kunpeng(at::Tensor prefix_lens, at::Tensor seq_lens, at::Tensor last_loc,
                                 at::Tensor free_pages, at::Tensor out_indices, int64_t page_size)
{
    CHECK_INPUT(prefix_lens);
    CHECK_INPUT(seq_lens);
    CHECK_INPUT(last_loc);
    CHECK_INPUT(free_pages);
    CHECK_INPUT(out_indices);
    TORCH_CHECK(prefix_lens.scalar_type() == at::kLong, "prefix_lens must be int64");
    TORCH_CHECK(seq_lens.scalar_type() == at::kLong, "seq_lens must be int64");
    TORCH_CHECK(free_pages.scalar_type() == at::kLong, "free_pages must be int64");
    TORCH_CHECK(out_indices.scalar_type() == at::kLong, "out_indices must be int64");
    TORCH_CHECK(last_loc.scalar_type() == at::kLong || last_loc.scalar_type() == at::kInt,
                "last_loc must be int64 or int32");

    int64_t bs = prefix_lens.size(0);
    TORCH_CHECK(seq_lens.size(0) == bs && last_loc.size(0) == bs, "shape mismatch");

    const int64_t *prefix_ptr = prefix_lens.data_ptr<int64_t>();
    const int64_t *seq_ptr = seq_lens.data_ptr<int64_t>();
    const int64_t *free_ptr = free_pages.data_ptr<int64_t>();
    int64_t *out_ptr = out_indices.data_ptr<int64_t>();
    const bool last_is_i64 = last_loc.scalar_type() == at::kLong;
    const int64_t *last_ptr = last_loc.data_ptr<int64_t>();
    const int32_t *last_ptr_i32 = last_is_i64 ? nullptr : last_loc.data_ptr<int32_t>();

    // Prefix scan: per-request extend token start/end offsets and per-request
    // new-page start/end/need counts. All derived from prefix/seq lengths.
    std::vector<int64_t> start_pos(bs), end_pos(bs);
    std::vector<int64_t> start_new_pages(bs), end_new_pages(bs), need_page(bs);
    {
        int64_t acc_tok = 0, acc_pg = 0;
        for (int64_t i = 0; i < bs; i++) {
            int64_t pref = prefix_ptr[i], seq = seq_ptr[i];
            int64_t ext = seq - pref;
            start_pos[i] = acc_tok;
            acc_tok += ext;
            end_pos[i] = acc_tok;

            int64_t num_new = (seq + page_size - 1) / page_size - (pref + page_size - 1) / page_size;
            int64_t num_full = seq / page_size - (pref + page_size - 1) / page_size;
            need_page[i] = num_new - num_full;
            start_new_pages[i] = acc_pg;
            acc_pg += num_new;
            end_new_pages[i] = acc_pg;
        }
    }

    kutacc::parallel_for(0, bs, 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; i++) {
            int64_t pref = prefix_ptr[i], seq = seq_ptr[i];
            int64_t last = last_is_i64 ? last_ptr[i] : (int64_t)last_ptr_i32[i];

            // Segment 1: continuation after the aligned prefix boundary.
            int64_t num1 = std::min(seq, (pref + page_size - 1) / page_size * page_size) - pref;
            if (num1 > 0) {
                int64_t base = last + 1;
                for (int64_t k = 0; k < num1; k++) {
                    out_ptr[start_pos[i] + k] = base + k;
                }
            }

            // Segment 2: full new pages.
            int64_t num2 = std::max<int64_t>(0, seq / page_size - (pref + page_size - 1) / page_size) * page_size;
            if (num2 > 0) {
                int64_t n_pages = end_new_pages[i] - need_page[i] - start_new_pages[i];
                int64_t out_base = start_pos[i] + num1;
                for (int64_t p = 0; p < n_pages; p++) {
                    int64_t page_base = free_ptr[start_new_pages[i] + p] * page_size;
                    for (int64_t k = 0; k < page_size; k++) {
                        out_ptr[out_base + p * page_size + k] = page_base + k;
                    }
                }
            }

            // Segment 3: final partial page tail.
            int64_t num3 = std::max<int64_t>(
                0, seq - std::max(seq / page_size, (pref + page_size - 1) / page_size) * page_size);
            if (num3 > 0) {
                int64_t page_base = free_ptr[end_new_pages[i] - 1] * page_size;
                int64_t out_base = end_pos[i] - num3;
                for (int64_t k = 0; k < num3; k++) {
                    out_ptr[out_base + k] = page_base + k;
                }
            }
        }
    });
}

// Get last token location for each request from req_to_token table.
//
// Mirrors `mem_cache/common.py::get_last_loc_torch`: for each request i,
// if prefix_lens[i] > 0, out[i] = req_to_token[req_pool_indices[i], prefix_lens[i]-1];
// otherwise out[i] = -1. Output is int64 (matching torch.where promote).
void get_last_loc_kunpeng(at::Tensor req_to_token, at::Tensor req_pool_indices,
                          at::Tensor prefix_lens, at::Tensor out)
{
    CHECK_INPUT(req_to_token);
    CHECK_INPUT(req_pool_indices);
    CHECK_INPUT(prefix_lens);
    CHECK_INPUT(out);
    TORCH_CHECK(req_to_token.scalar_type() == at::kInt, "req_to_token must be int32");
    TORCH_CHECK(req_pool_indices.scalar_type() == at::kLong, "req_pool_indices must be int64");
    TORCH_CHECK(prefix_lens.scalar_type() == at::kLong, "prefix_lens must be int64");
    TORCH_CHECK(out.scalar_type() == at::kLong, "out must be int64");

    int64_t B = prefix_lens.size(0);
    int64_t max_ctx = req_to_token.size(1);

    const int32_t *token_ptr = req_to_token.data_ptr<int32_t>();
    const int64_t *pool_ptr = req_pool_indices.data_ptr<int64_t>();
    const int64_t *pref_ptr = prefix_lens.data_ptr<int64_t>();
    int64_t *out_ptr = out.data_ptr<int64_t>();

    kutacc::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; i++) {
            int64_t pref = pref_ptr[i];
            if (pref > 0) {
                int64_t row = pool_ptr[i];
                out_ptr[i] = (int64_t)token_ptr[row * max_ctx + pref - 1];
            } else {
                out_ptr[i] = -1;
            }
        }
    });
}
