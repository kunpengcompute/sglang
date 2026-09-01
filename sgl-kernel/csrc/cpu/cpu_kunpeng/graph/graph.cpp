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

#include <torch/csrc/autograd/profiler.h>
#include <torch/extension.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "capture.h"
#include "graph.h"

Graph::Graph(std::vector<StorageBuf> storages, std::vector<TensorView> views, std::vector<OpRecord> ops,
             std::vector<int> output_view_ids, int num_inputs, const std::unordered_map<int, torch::Tensor> &fixed,
             torch::Tensor external_pool, torch::Tensor external_shm_pool, int memory_alignment)
    : num_inputs_(num_inputs),
      storages_(std::move(storages)),
      views_(std::move(views)),
      op_records_(std::move(ops)),
      output_view_ids_(std::move(output_view_ids))
{
    total_ops_ = static_cast<int>(op_records_.size());
    finalize(fixed, std::move(external_pool), std::move(external_shm_pool), memory_alignment);

    profile_row_.assign(total_ops_ + 1, 0);
}

void Graph::finalize(const std::unordered_map<int, torch::Tensor> &fixed, torch::Tensor external_pool,
                     torch::Tensor external_shm_pool, int memory_alignment)
{
    compute_death_ops();
    detect_outputs();
    plan_memory(std::move(external_pool), std::move(external_shm_pool), memory_alignment);
    precompute_replay();
    hold_fixed(fixed);
    for (int i = 0; i < total_ops_; ++i) {
        op_names_.push_back(op_records_[i].profile_name.empty() ? op_records_[i].op_name : op_records_[i].profile_name);
    }
}

void Graph::enable_profile(bool enable)
{
    profile_enabled_ = enable;
}

void Graph::compute_death_ops()
{
    for (auto &s : storages_)
        s.death_op = std::max(0, s.born_op);

    for (int op_idx = 0; op_idx < total_ops_; ++op_idx) {
        const auto &op = op_records_[op_idx];
        for (int vid : op.input_view_ids) {
            int sid = views_[vid].storage_id;
            storages_[sid].death_op = std::max(storages_[sid].death_op, op_idx);
        }
    }
}

void Graph::detect_outputs()
{
    for (size_t i = 0; i < storages_.size(); ++i) {
        if (storages_[i].born_op == -1) {
            if (static_cast<int>(i) < num_inputs_)
                input_storage_ids_.push_back(static_cast<int>(i));
            else
                fixed_storage_ids_.push_back(static_cast<int>(i));
        }
    }

    for (int vid : output_view_ids_) {
        int sid = views_[vid].storage_id;
        storages_[sid].death_op = total_ops_ - 1;
    }
}

namespace {

struct Interval {
    int idx;
    int born;
    int death;
    size_t size;
};

struct PlaceEntry {
    int idx;
    size_t offset;
    size_t size;
    int born;
    int death;
};

bool intervals_overlap(int a_born, int a_death, int b_born, int b_death)
{
    return !(a_death < b_born || b_death < a_born);
}

std::vector<PlaceEntry> pack_intervals(std::vector<Interval> intervals, int memory_alignment)
{
    std::sort(intervals.begin(), intervals.end(),
              [](const auto &a, const auto &b) { return a.size != b.size ? a.size > b.size : a.idx < b.idx; });

    std::vector<PlaceEntry> placed;
    for (const auto &[idx, born, death, size] : intervals) {
        if (size == 0) continue;

        std::vector<size_t> candidates = {0};
        for (const auto &pe : placed) {
            if (intervals_overlap(born, death, pe.born, pe.death)) {
                size_t aligned_end = (pe.offset + pe.size + memory_alignment - 1) / memory_alignment * memory_alignment;
                candidates.push_back(aligned_end);
            }
        }
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

        size_t best_offset = 0;
        for (size_t cand : candidates) {
            bool valid = true;
            for (const auto &pe : placed) {
                if (intervals_overlap(born, death, pe.born, pe.death)) {
                    if (cand < pe.offset + pe.size && cand + size > pe.offset) {
                        valid = false;
                        break;
                    }
                }
            }
            if (valid) {
                best_offset = cand;
                break;
            }
        }

        placed.push_back({idx, best_offset, size, born, death});
    }
    return placed;
}

}  // namespace

void Graph::plan_memory(torch::Tensor external_pool, torch::Tensor external_shm_pool, int memory_alignment)
{
    std::vector<Interval> regular_intervals;
    std::vector<Interval> shm_intervals;
    for (size_t i = 0; i < storages_.size(); ++i) {
        const auto &s = storages_[i];
        if (!s.in_pool) continue;
        int born = std::max(0, s.born_op);
        Interval iv{static_cast<int>(i), born, s.death_op, s.size};
        if (s.memory_type == MemoryType::SHM)
            shm_intervals.push_back(iv);
        else
            regular_intervals.push_back(iv);
    }

    auto assign_pool = [this](std::vector<Interval> intervals, MemoryPool &pool, torch::Tensor external, bool is_shm,
                              const char *tag, int memory_alignment) {
        if (intervals.empty()) return;

        auto placed = pack_intervals(std::move(intervals), memory_alignment);

        size_t pool_size = 0;
        for (const auto &pe : placed)
            pool_size = std::max(pool_size, pe.offset + pe.size);
        pool_size = (pool_size + memory_alignment - 1) / memory_alignment * memory_alignment;

        if (external.defined()) {
            TORCH_CHECK(external.nbytes() >= static_cast<int64_t>(pool_size), "plan_memory: external ", tag,
                        " pool too small (", external.nbytes(), " bytes vs needed ", pool_size, " bytes)");
            pool.adopt(std::move(external));
        } else {
            TORCH_CHECK(!is_shm, "plan_memory: ", tag, " pool requires external pool");
            pool.allocate(pool_size);
        }

        if constexpr (kGraphDebugPrint) {
            std::cout << "===== plan_memory: " << tag << " pool allocated " << pool_size << " bytes ("
                      << (pool_size / 1024) << " KiB) for " << placed.size() << " storages =====" << std::endl;
            for (const auto &pe : placed) {
                std::cout << "  storage[" << pe.idx << "] offset=" << pe.offset << " size=" << pe.size
                          << " born=" << pe.born << " death=" << pe.death << std::endl;
            }
        }

        for (const auto &pe : placed)
            storages_[pe.idx].data_ptr = pool.ptr(pe.offset);
    };

    assign_pool(std::move(regular_intervals), pool_, std::move(external_pool), false, "regular", memory_alignment);
    assign_pool(std::move(shm_intervals), shm_pool_, std::move(external_shm_pool), true, "shm", memory_alignment);
}

void Graph::precompute_replay()
{
    // Runtime input primary views: 0-dim placeholders (replaced in run())
    input_view_ids_.clear();
    for (int sid : input_storage_ids_)
        input_view_ids_.push_back(sid);

    // Build cached_tensors_ indexed by view_id
    cached_tensors_.resize(views_.size());

    // Input primary views: 0-dim placeholder tensors
    for (int sid : input_storage_ids_)
        cached_tensors_[sid] = torch::empty({0});

    // Build alias map: for each runtime input, collect all secondary views
    // that alias the same storage (vid != storage_id).
    input_alias_vids_.resize(input_storage_ids_.size());
    for (size_t vi = 0; vi < views_.size(); ++vi) {
        int sid = views_[vi].storage_id;
        if (static_cast<int>(vi) < num_inputs_) continue;  // skip primary input views
        if (storages_[sid].born_op != -1) continue;        // skip intermediates
        if (storages_[sid].in_pool) continue;              // skip pool

        // Find which input index this storage belongs to
        for (size_t i = 0; i < input_storage_ids_.size(); ++i) {
            if (input_storage_ids_[i] == sid) {
                input_alias_vids_[i].push_back(static_cast<int>(vi));
                break;
            }
        }
    }

    // All views: from_blob into pool or fixed memory.
    // Fixed storage views use data_ptr set during begin_capture.
    for (size_t vid = 0; vid < views_.size(); ++vid) {
        const auto &view = views_[vid];
        const auto &storage = storages_[view.storage_id];

        // Skip input primary views (vid 0..num_inputs_-1).
        // They are handled as placeholders above and swapped in run().
        if (static_cast<int>(vid) < num_inputs_) continue;

        if (view.numel == 0 && view.element_size == 0) {
            cached_tensors_[vid] = at::Tensor();
            continue;
        }
        if (!storage.data_ptr) {
            auto dtype = static_cast<c10::ScalarType>(view.scalar_type);
            cached_tensors_[vid] = torch::empty(view.shape, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
            continue;
        }

        void *ptr = static_cast<char *>(storage.data_ptr) + view.storage_offset * view.element_size;
        auto dtype = static_cast<c10::ScalarType>(view.scalar_type);
        cached_tensors_[vid] =
            torch::from_blob(ptr, view.shape, view.strides, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
    }

    // Pre-compute per-op dispatch data (one-time lookups, amortized over replays)
    op_dispatch_.resize(total_ops_);
    op_vid_begin_.resize(total_ops_);
    op_vid_count_.resize(total_ops_);

    int total_vids = 0;
    for (int op_idx = 0; op_idx < total_ops_; ++op_idx) {
        const auto &op = op_records_[op_idx];
        int nvids = static_cast<int>(op.input_view_ids.size() + op.output_view_ids.size());
        op_vid_begin_[op_idx] = total_vids;
        op_vid_count_[op_idx] = nvids;
        total_vids += nvids;
        max_tensor_count_ = std::max(max_tensor_count_, nvids);

        auto *info = GraphOpRegistry::instance().lookup(op.op_name);
        if (!info || !info->dispatch_fn) TORCH_CHECK(false, "precompute_replay: op '", op.op_name, "' not registered");
        op_dispatch_[op_idx] = info->dispatch_fn;
    }

    // Build flat view_ids
    flat_vids_.resize(total_vids);
    for (int op_idx = 0; op_idx < total_ops_; ++op_idx) {
        const auto &op = op_records_[op_idx];
        int pos = op_vid_begin_[op_idx];
        for (int vid : op.input_view_ids)
            flat_vids_[pos++] = vid;
        for (int vid : op.output_view_ids)
            flat_vids_[pos++] = vid;
    }

    // Pre-allocate saved_ and op_tensors_
    saved_.resize(input_view_ids_.size());
    op_tensors_.reserve(max_tensor_count_);
}

void Graph::hold_fixed(const std::unordered_map<int, torch::Tensor> &fixed)
{
    fixed_tensors_.resize(fixed_storage_ids_.size());
    for (const auto &[pos, tensor] : fixed) {
        int idx = pos - num_inputs_;
        TORCH_CHECK(idx >= 0 && idx < static_cast<int>(fixed_storage_ids_.size()),
                    "hold_fixed: invalid fixed position ", pos);
        TORCH_CHECK(tensor.defined(), "hold_fixed: tensor at position ", pos, " is undefined");
        fixed_tensors_[idx] = tensor;
    }
}

std::vector<torch::Tensor> Graph::run(const std::vector<torch::Tensor> &inputs)
{
    const int n = static_cast<int>(input_view_ids_.size());
    TORCH_CHECK(static_cast<int>(inputs.size()) == n, "Graph::run: expected ", n, " inputs, got ", inputs.size());

    // Swap runtime input tensors into cached_tensors_;
    // update storage data_ptr, view shape/strides, and alias views.
    for (int i = 0; i < n; ++i) {
        const int vid = input_view_ids_[i];
        saved_[i] = cached_tensors_[vid];
        TORCH_CHECK(inputs[i].defined(), "Graph::run: input at position ", i, " is undefined");
        cached_tensors_[vid] = inputs[i];

        storages_[vid].data_ptr =
            static_cast<char *>(inputs[i].data_ptr()) - inputs[i].storage_offset() * inputs[i].element_size();

        // Update view shape/strides for dynamic batch sizes
        views_[vid].shape.assign(inputs[i].sizes().begin(), inputs[i].sizes().end());
        views_[vid].strides.assign(inputs[i].strides().begin(), inputs[i].strides().end());

        // Sync alias views to the new data_ptr
        for (int alias_vid : input_alias_vids_[i]) {
            const auto &av = views_[alias_vid];
            void *ptr = static_cast<char *>(storages_[vid].data_ptr) + av.storage_offset * av.element_size;
            cached_tensors_[alias_vid] = torch::from_blob(
                ptr, av.shape, av.strides,
                torch::TensorOptions().dtype(static_cast<c10::ScalarType>(av.scalar_type)).device(torch::kCPU));
        }
    }

    // Execute ops — zero heap allocation in inner loop
    for (int op_idx = 0; op_idx < total_ops_; ++op_idx) {
        const int begin = op_vid_begin_[op_idx];
        const int count = op_vid_count_[op_idx];

        op_tensors_.clear();
        for (int j = 0; j < count; ++j)
            op_tensors_.push_back(cached_tensors_[flat_vids_[begin + j]]);

        if (profile_enabled_) {
            auto ts = std::chrono::high_resolution_clock::now();
            profile_row_[op_idx] = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(ts.time_since_epoch()).count());
        }

        RECORD_FUNCTION(op_names_[op_idx].c_str(), std::vector<c10::IValue>{});
        if constexpr (kGraphDebugPrint) {
            std::cout << "[replay op " << op_idx << "] " << op_names_[op_idx] << std::endl;
            const auto &op = op_records_[op_idx];
            for (int j = 0; j < count; ++j) {
                int vid = flat_vids_[begin + j];
                bool is_in = j < static_cast<int>(op.input_view_ids.size());
                const auto &t = cached_tensors_[vid];
                std::cout << "  " << (is_in ? "in " : "out") << " vid=" << vid;
                if (t.defined()) {
                    std::cout << " dtype=" << t.scalar_type() << " shape=[";
                    for (size_t d = 0; d < t.sizes().size(); ++d) {
                        if (d) std::cout << ", ";
                        std::cout << t.size(d);
                    }
                    std::cout << "]";
                } else {
                    std::cout << " undefined";
                }
                std::cout << std::endl;
            }
        }
        op_dispatch_[op_idx](op_tensors_, op_records_[op_idx].scalar_args);
    }

    if (profile_enabled_) {
        auto ts = std::chrono::high_resolution_clock::now();
        profile_row_[total_ops_] =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(ts.time_since_epoch()).count());
    }

    // Restore 0-dim placeholder tensors
    for (int i = 0; i < n; ++i)
        cached_tensors_[input_view_ids_[i]] = saved_[i];

    // Build outputs from cached tensors
    std::vector<torch::Tensor> outputs;
    for (int vid : output_view_ids_)
        outputs.push_back(cached_tensors_[vid]);
    return outputs;
}
