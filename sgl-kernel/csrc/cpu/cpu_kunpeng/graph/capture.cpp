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

#include <algorithm>
#include <iostream>

#include "capture.h"

CaptureManager& CaptureManager::instance()
{
    static CaptureManager mgr;
    return mgr;
}

void CaptureManager::begin_capture(const std::vector<StorageBuf>& external_bufs,
                                   const std::vector<TensorView>& external_views,
                                   int num_inputs)
{
    capturing_ = true;
    num_external_inputs_ = num_inputs;
    storages_.clear();
    views_.clear();
    op_records_.clear();
    storage_registry_.clear();

    for (size_t i = 0; i < external_bufs.size(); ++i) {
        auto buf = external_bufs[i];
        buf.id = static_cast<int>(storages_.size());
        buf.born_op = -1;
        buf.death_op = -1;
        buf.in_pool = false;
        storage_registry_[buf.storage_base] = buf.id;
        storages_.push_back(buf);
    }

    for (size_t i = 0; i < external_views.size(); ++i) {
        auto view = external_views[i];
        view.id = static_cast<int>(views_.size());
        auto& s = storages_[view.storage_id];
        s.size = std::max(s.size,
            static_cast<size_t>(view.storage_offset) * view.element_size +
                view.numel * view.element_size);
        views_.push_back(view);
    }
}

std::pair<std::vector<StorageBuf>, std::vector<TensorView>>
CaptureManager::end_capture()
{
    TORCH_CHECK(capturing_, "end_capture: not in capture context");
    capturing_ = false;
    return {std::move(storages_), std::move(views_)};
}

int CaptureManager::lookup_storage(void* storage_base) const
{
    auto it = storage_registry_.find(storage_base);
    if (it != storage_registry_.end()) return it->second;
    return -1;
}

int CaptureManager::register_storage(StorageBuf buf)
{
    buf.id = static_cast<int>(storages_.size());
    storage_registry_[buf.storage_base] = buf.id;
    storages_.push_back(buf);
    return buf.id;
}

int CaptureManager::register_output_storage(StorageBuf buf)
{
    TORCH_CHECK(capturing_, "register_output_storage: not in capture context");
    buf.born_op = static_cast<int>(op_records_.size());
    return register_storage(buf);
}

int CaptureManager::register_view(TensorView view)
{
    TORCH_CHECK(capturing_, "register_view: not in capture context");
    view.id = static_cast<int>(views_.size());
    views_.push_back(view);
    return view.id;
}

int CaptureManager::find_or_register_view(
        int storage_id, int64_t storage_offset,
        const torch::Tensor& t)
{
    TORCH_CHECK(capturing_, "find_or_register_view: not in capture context");

    if (!t.defined()) {
        for (const auto& v : views_) {
            if (v.storage_id == storage_id && v.storage_offset == 0 &&
                v.numel == 0 && v.element_size == 0 &&
                v.shape.empty() && !v.is_return) {
                return v.id;
            }
        }
        TORCH_CHECK(storage_id >= num_external_inputs_,
            "input tensor (storage=", storage_id, ") cannot be undefined/None");
        TensorView view;
        view.storage_id = storage_id;
        view.storage_offset = 0;
        view.numel = 0;
        view.element_size = 0;
        view.scalar_type = 0;
        view.is_return = false;
        return register_view(view);
    }

    size_t numel = static_cast<size_t>(t.numel());
    size_t element_size = static_cast<size_t>(t.element_size());
    int scalar_type = static_cast<int>(t.scalar_type());
    std::vector<int64_t> shape(t.sizes().begin(), t.sizes().end());
    std::vector<int64_t> strides(t.strides().begin(), t.strides().end());

    for (const auto& v : views_) {
        if (v.storage_id == storage_id &&
            v.storage_offset == storage_offset &&
            v.numel == numel &&
            v.element_size == element_size &&
            v.scalar_type == scalar_type &&
            v.shape == shape &&
            v.strides == strides &&
            !v.is_return) {
            return v.id;
        }
    }
    // No matching view. For input storages: must match the pre-registered view.
    TORCH_CHECK(storage_id >= num_external_inputs_,
        "input tensor (storage=", storage_id, ") has no matching view");
    TensorView view;
    view.storage_id = storage_id;
    view.storage_offset = storage_offset;
    view.numel = numel;
    view.element_size = element_size;
    view.scalar_type = scalar_type;
    view.shape = shape;
    view.strides = strides;
    view.is_return = false;
    return register_view(view);
}

void CaptureManager::record_op(OpRecord op)
{
    TORCH_CHECK(capturing_, "record_op: not in capture context");
    if constexpr (kGraphDebugPrint)
        debug_print_op(op);
    op_records_.push_back(std::move(op));
}

void CaptureManager::debug_print_op(const OpRecord& op) const
{
    auto print_view = [this](int vid, const char* tag) {
        const auto& v = views_[vid];
        int sid = v.storage_id;
        const auto& s = storages_[sid];
        const char* source;
        if (s.born_op < 0) {
            source = (vid < num_external_inputs_) ? "input" : "fixed";
        } else {
            source = "intermediate";
        }
        std::cout << "  " << tag << " vid=" << vid << " sid=" << sid
                  << " source=" << source << " shape=[";
        for (size_t i = 0; i < v.shape.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << v.shape[i];
        }
        std::cout << "]" << std::endl;
    };

    int op_idx = static_cast<int>(op_records_.size());
    std::cout << "[op " << op_idx << "] " << op.op_name << std::endl;
    for (int vid : op.input_view_ids)  print_view(vid, " in ");
    for (int vid : op.output_view_ids) print_view(vid, " out");
    std::cout << std::endl;
}

GraphOpRegistry& GraphOpRegistry::instance()
{
    static GraphOpRegistry reg;
    return reg;
}

void GraphOpRegistry::register_op(const std::string& name, OpInfo info)
{
    ops_[name] = std::move(info);
}

const GraphOpRegistry::OpInfo* GraphOpRegistry::lookup(const std::string& name) const
{
    auto it = ops_.find(name);
    if (it != ops_.end()) return &it->second;
    return nullptr;
}
