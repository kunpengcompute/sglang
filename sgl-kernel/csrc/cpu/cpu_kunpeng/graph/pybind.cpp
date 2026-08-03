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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <memory>
#include <variant>

#include "capture.h"
#include "graph.h"

namespace py = pybind11;

static int64_t tensor_storage_offset(const torch::Tensor& t)
{
    if (!t.defined()) return 0;
    return t.storage_offset();
}

static uintptr_t tensor_storage_base(const torch::Tensor& t)
{
    if (!t.defined()) return 0;
    return reinterpret_cast<uintptr_t>(t.storage().data());
}

static std::pair<StorageBuf, TensorView> tensor_to_buf_and_view(const torch::Tensor& t)
{
    if (!t.defined()) {
        StorageBuf buf;
        buf.storage_base = nullptr;
        buf.size = 0;

        TensorView view;
        view.storage_offset = 0;
        view.numel = 0;
        view.element_size = 0;
        view.scalar_type = 0;
        return {buf, view};
    }

    int64_t so = t.storage_offset();
    void* base = const_cast<void*>(t.storage().data());

    StorageBuf buf;
    buf.storage_base = base;
    buf.size = static_cast<size_t>(so) * t.element_size() +
               static_cast<size_t>(t.numel()) * t.element_size();

    TensorView view;
    view.storage_offset = so;
    view.numel = static_cast<size_t>(t.numel());
    view.element_size = static_cast<size_t>(t.element_size());
    view.scalar_type = static_cast<int>(t.scalar_type());
    for (int64_t s : t.sizes()) view.shape.push_back(s);
    for (int64_t s : t.strides()) view.strides.push_back(s);

    return {buf, view};
}

// Manual ScalarArg ↔ Python conversion (avoid custom type_caster)
static std::vector<ScalarArg> _py_to_scalar_args(py::list src)
{
    std::vector<ScalarArg> out;
    for (auto item : src) {
        if (PyBool_Check(item.ptr()))       out.push_back(item.cast<bool>());
        else if (PyLong_Check(item.ptr()))  out.push_back(item.cast<int64_t>());
        else if (PyFloat_Check(item.ptr())) out.push_back(item.cast<double>());
        else throw std::runtime_error("scalar_args: unsupported type");
    }
    return out;
}

static py::list _scalar_args_to_py(const std::vector<ScalarArg>& src)
{
    py::list out;
    for (const auto& v : src)
        std::visit([&](auto&& x) { out.append(py::cast(x)); }, v);
    return out;
}

void init_graph_cpp(py::module& m)
{
    m.doc() = "Graph computation graph engine";

    py::enum_<MemoryType>(m, "MemoryType")
        .value("REGULAR", MemoryType::REGULAR)
        .value("SHM", MemoryType::SHM);

    py::class_<StorageBuf>(m, "StorageBuf")
        .def(py::init<>())
        .def_readwrite("id", &StorageBuf::id)
        .def_property("storage_base",
            [](const StorageBuf& b) { return reinterpret_cast<uintptr_t>(b.storage_base); },
            [](StorageBuf& b, uintptr_t v) { b.storage_base = reinterpret_cast<void*>(v); })
        .def_readwrite("born_op", &StorageBuf::born_op)
        .def_readwrite("death_op", &StorageBuf::death_op)
        .def_readwrite("size", &StorageBuf::size)
        .def_readwrite("in_pool", &StorageBuf::in_pool)
        .def_readwrite("memory_type", &StorageBuf::memory_type)
        .def_property("data_ptr",
            [](const StorageBuf& b) { return reinterpret_cast<uintptr_t>(b.data_ptr); },
            [](StorageBuf& b, uintptr_t v) { b.data_ptr = reinterpret_cast<void*>(v); });

    py::class_<TensorView>(m, "TensorView")
        .def(py::init<>())
        .def_readwrite("id", &TensorView::id)
        .def_readwrite("storage_id", &TensorView::storage_id)
        .def_readwrite("storage_offset", &TensorView::storage_offset)
        .def_readwrite("numel", &TensorView::numel)
        .def_readwrite("element_size", &TensorView::element_size)
        .def_readwrite("scalar_type", &TensorView::scalar_type)
        .def_readwrite("shape", &TensorView::shape)
        .def_readwrite("strides", &TensorView::strides)
        .def_readwrite("is_return", &TensorView::is_return);

    py::class_<OpRecord>(m, "OpRecord")
        .def(py::init<>())
        .def_readwrite("op_name", &OpRecord::op_name)
        .def_readwrite("profile_name", &OpRecord::profile_name)
        .def_readwrite("input_view_ids", &OpRecord::input_view_ids)
        .def_readwrite("output_view_ids", &OpRecord::output_view_ids)
        .def_property("scalar_args",
            [](const OpRecord& r) { return _scalar_args_to_py(r.scalar_args); },
            [](OpRecord& r, py::list v) { r.scalar_args = _py_to_scalar_args(v); });

    py::class_<CaptureManager>(m, "CaptureManager")
        .def_static("instance", &CaptureManager::instance,
                    py::return_value_policy::reference)
        .def("begin_capture", [](CaptureManager& self,
                                  const std::vector<torch::Tensor>& inputs,
                                  const std::vector<torch::Tensor>& fixed) {
            std::vector<StorageBuf> all_bufs;
            std::vector<TensorView> all_views;

            auto add_input = [&](const torch::Tensor& t, bool is_fixed) {
                TORCH_CHECK(t.defined(),
                            "begin_capture: external input tensor is undefined (None)");
                auto [buf, view] = tensor_to_buf_and_view(t);
                int sid = static_cast<int>(all_bufs.size());
                buf.id = sid;
                view.storage_id = sid;
                if (is_fixed) {
                    TORCH_CHECK(buf.storage_base != nullptr,
                                "begin_capture: fixed tensor has nullptr storage");
                    buf.data_ptr = buf.storage_base;
                    all_bufs.push_back(buf);
                } else {
                    all_bufs.push_back(buf);
                    all_views.push_back(view);
                }
            };

            for (const auto& t : inputs) add_input(t, false);
            for (const auto& t : fixed)  add_input(t, true);

            self.begin_capture(all_bufs, all_views,
                               static_cast<int>(inputs.size()));
        })
        .def("end_capture", [](CaptureManager& self) -> py::tuple {
            auto [storages, views] = self.end_capture();
            return py::make_tuple(storages, views, self.op_records(),
                                  self.num_external_inputs());
        })
        .def("is_capturing", &CaptureManager::is_capturing)
        .def("lookup_storage", [](CaptureManager& self, uintptr_t base) {
            return self.lookup_storage(reinterpret_cast<void*>(base));
        })
        .def("register_storage", &CaptureManager::register_storage)
        .def("register_output_storage", [](CaptureManager& self, StorageBuf buf, MemoryType memory_type) {
            return self.register_output_storage(std::move(buf), memory_type);
        })
        .def("upgrade_storage_memory_type", &CaptureManager::upgrade_storage_memory_type)
        .def("register_view", &CaptureManager::register_view)
        .def("find_or_register_view", [](CaptureManager& self, int storage_id, int64_t storage_offset, py::object t) {
            torch::Tensor tensor;
            if (!t.is_none()) tensor = t.cast<torch::Tensor>();
            return self.find_or_register_view(storage_id, storage_offset, tensor);
        })
        .def("record_op", &CaptureManager::record_op);

    py::class_<Graph>(m, "Graph")
        .def(py::init([](const std::vector<StorageBuf>& storages,
                         const std::vector<TensorView>& views,
                         const std::vector<OpRecord>& ops,
                         const std::vector<int>& output_view_ids,
                         int num_inputs,
                         const std::unordered_map<int, torch::Tensor>& fixed,
                         py::object external_pool,
                         py::object external_shm_pool) {
            torch::Tensor pool_tensor;
            if (!external_pool.is_none()) pool_tensor = external_pool.cast<torch::Tensor>();
            torch::Tensor shm_pool_tensor;
            if (!external_shm_pool.is_none()) shm_pool_tensor = external_shm_pool.cast<torch::Tensor>();
            return std::make_unique<Graph>(storages, views, ops,
                                           output_view_ids, num_inputs, fixed,
                                           pool_tensor, shm_pool_tensor);
        }), py::arg("storages"), py::arg("views"), py::arg("ops"),
            py::arg("output_view_ids"), py::arg("num_inputs"),
            py::arg("fixed"), py::arg("external_pool") = py::none(),
            py::arg("external_shm_pool") = py::none())
        .def_readwrite("has_hidden_states", &Graph::has_hidden_states)
        .def("run", &Graph::run)
        .def("set_fixed", &Graph::set_fixed)
        .def("enable_profile", &Graph::enable_profile)
        .def("get_profile_row", &Graph::get_profile_row)
        .def("profile_op_names", &Graph::profile_op_names,
             py::return_value_policy::reference_internal);

    py::class_<GraphOpRegistry::OpInfo>(m, "OpInfo")
        .def(py::init<>())
        .def_property("dispatch_fn",
            [](const GraphOpRegistry::OpInfo& info) {
                return reinterpret_cast<uintptr_t>(info.dispatch_fn);
            },
            [](GraphOpRegistry::OpInfo& info, uintptr_t v) {
                info.dispatch_fn = reinterpret_cast<DispatchFn>(v);
            });

    py::class_<GraphOpRegistry>(m, "GraphOpRegistry")
        .def_static("instance", &GraphOpRegistry::instance,
                    py::return_value_policy::reference)
        .def("register_op", &GraphOpRegistry::register_op)
        .def("lookup", &GraphOpRegistry::lookup,
             py::return_value_policy::reference);

    m.def("storage_base", &tensor_storage_base);
    m.def("storage_offset", &tensor_storage_offset);
    m.def("tensor_to_buf_and_view", &tensor_to_buf_and_view);
}
