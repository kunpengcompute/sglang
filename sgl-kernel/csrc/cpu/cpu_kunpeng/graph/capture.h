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

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensor_meta.h"
#include "op_record.h"

constexpr bool kGraphDebugPrint = false;

class CaptureManager {
public:
    static CaptureManager& instance();

    void begin_capture(const std::vector<StorageBuf>& external_bufs,
                       const std::vector<TensorView>& external_views,
                       int num_inputs);
    std::pair<std::vector<StorageBuf>, std::vector<TensorView>>
        end_capture();
    bool is_capturing() const { return capturing_; }
    int num_external_inputs() const { return num_external_inputs_; }

    int lookup_storage(void* storage_base) const;
    int register_storage(StorageBuf buf);
    int register_output_storage(StorageBuf buf);

    int register_view(TensorView view);
    int find_or_register_view(int storage_id, int64_t storage_offset,
                              const torch::Tensor& t);

    void record_op(OpRecord op);
    std::vector<OpRecord>& op_records() { return op_records_; }

private:
    void debug_print_op(const OpRecord& op) const;

    CaptureManager() = default;

    bool capturing_ = false;
    int num_external_inputs_ = 0;
    std::unordered_map<void*, int> storage_registry_;
    std::vector<StorageBuf> storages_;
    std::vector<TensorView> views_;
    std::vector<OpRecord> op_records_;
};

class GraphOpRegistry {
public:
    static GraphOpRegistry& instance();

    struct OpInfo {
        DispatchFn dispatch_fn = nullptr;
    };

    void register_op(const std::string& name, OpInfo info);
    const OpInfo* lookup(const std::string& name) const;

private:
    GraphOpRegistry() = default;
    std::unordered_map<std::string, OpInfo> ops_;
};
