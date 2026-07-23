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
/*
 * KUPL Async — submit task to KUPL worker thread, wait for result.
 * submit: non-blocking, submits fn(*args) to a KUPL executor via kupl_queue
 * wait: blocks until task completes, returns the result
 */

#include <pybind11/pybind11.h>
#include <atomic>
#include <vector>
#include "kupl.h"

namespace py = pybind11;

struct TaskArgs {
    PyObject *fn;
    PyObject *args;
    PyObject *result;
};

static void py_task_body(void *args) {
    auto *ta = (TaskArgs *)args;
    PyGILState_STATE g = PyGILState_Ensure();
    ta->result = PyObject_CallObject(ta->fn, ta->args);
    if (ta->result == nullptr) {
        PyErr_Print();
        ta->result = Py_None;
        Py_INCREF(ta->result);
    }
    PyGILState_Release(g);
}

class PyKuplExecutor {
    kupl_egroup_h egroup_ = nullptr;
    kupl_queue_h queue_ = nullptr;
    TaskArgs *task_ = nullptr;

public:
    PyKuplExecutor() {
        int n = kupl_get_num_executors();
        if (n <= 1) {
            throw std::runtime_error("KUPL executor count must be > 1 for async launch");
        }
        std::vector<int> executors(n - 1);
        for (int i = 0; i < n - 1; i++) executors[i] = i + 1;
        egroup_ = kupl_egroup_create(executors.data(), n - 1);
    }

    ~PyKuplExecutor() {
        if (task_) {
            {
                py::gil_scoped_release release;
                kupl_queue_wait(queue_);
            }
            Py_DECREF(task_->fn);
            Py_DECREF(task_->args);
            delete task_;
        }
        if (queue_) kupl_queue_destroy(queue_);
        if (egroup_) kupl_egroup_destroy(egroup_);
    }

    void submit(py::object fn, py::args args) {
        if (queue_) kupl_queue_destroy(queue_);
        queue_ = kupl_queue_create();

        task_ = new TaskArgs();
        task_->fn = fn.ptr();
        task_->args = args.ptr();
        task_->result = nullptr;
        Py_INCREF(task_->fn);
        Py_INCREF(task_->args);

        kupl_queue_item_desc_t desc = {};
        desc.func = py_task_body;
        desc.args = task_;
        desc.egroup = egroup_;
        desc.field_mask = KUPL_QUEUE_ITEM_DESC_FIELD_EGROUP;

        kupl_queue_submit(queue_, &desc);
    }

    py::object wait() {
        if (!queue_ || !task_) {
            throw std::runtime_error("no task to wait");
        }

        {
            py::gil_scoped_release release;
            kupl_queue_wait(queue_);
        }

        PyObject *result = task_->result;
        Py_DECREF(task_->fn);
        Py_DECREF(task_->args);
        delete task_;
        task_ = nullptr;

        kupl_queue_destroy(queue_);
        queue_ = nullptr;

        return py::reinterpret_steal<py::object>(result);
    }
};

PYBIND11_MODULE(_kupl_async, m) {
    py::class_<PyKuplExecutor>(m, "PyKuplExecutor")
        .def(py::init<>())
        .def("submit", &PyKuplExecutor::submit, py::arg("fn"))
        .def("wait", &PyKuplExecutor::wait);
}
