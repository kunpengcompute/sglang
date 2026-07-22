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

#pragma once

#include <torch/extension.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "tensor_meta.h"
#include "op_record.h"
#include "memory_pool.h"

class Graph {
public:
    Graph(std::vector<StorageBuf> storages,
          std::vector<TensorView> views,
          std::vector<OpRecord> ops,
          std::vector<int> output_view_ids,
          int num_inputs,
          const std::unordered_map<int, torch::Tensor>& fixed,
          torch::Tensor external_pool = {});

    // Backward-compatible: hold fixed tensor references to keep memory alive.
    // data_ptr is already set during begin_capture; this only stores references.
    void set_fixed(const std::unordered_map<int, torch::Tensor>& fixed) {
        hold_fixed(fixed);
    }
    std::vector<torch::Tensor> run(const std::vector<torch::Tensor>& inputs);

private:
    void finalize(const std::unordered_map<int, torch::Tensor>& fixed,
                  torch::Tensor external_pool);
    void compute_death_ops();
    void plan_memory(torch::Tensor external_pool);
    void detect_outputs();
    void precompute_replay();
    void hold_fixed(const std::unordered_map<int, torch::Tensor>& fixed);

    static bool intervals_overlap(int a_born, int a_death, int b_born, int b_death) {
        return !(a_death < b_born || b_death < a_born);
    }

    int num_inputs_ = 0;
    std::vector<int> input_storage_ids_;          // runtime input storage IDs (first N)
    std::vector<int> fixed_storage_ids_;           // fixed tensor storage IDs
    std::vector<at::Tensor> fixed_tensors_;        // live Python tensor refs for fixed (GC guard)
    std::vector<std::vector<int>> input_alias_vids_; // per-input alias view IDs

    std::vector<StorageBuf> storages_;
    std::vector<TensorView> views_;
    std::vector<OpRecord> op_records_;
    MemoryPool pool_;
    std::vector<int> input_view_ids_;             // = input_storage_ids_ (primary view == storage_id)
    std::vector<int> output_view_ids_;
    std::vector<std::string> op_names_;
    int total_ops_ = 0;

    // --- pre-computed replay data (zero-allocation hot path) ---

    std::vector<at::Tensor> cached_tensors_;

    std::vector<DispatchFn> op_dispatch_;

    std::vector<int> flat_vids_;
    std::vector<int> op_vid_begin_;
    std::vector<int> op_vid_count_;
    int max_tensor_count_ = 0;

    std::vector<at::Tensor> saved_;
    std::vector<at::Tensor> op_tensors_;
};
