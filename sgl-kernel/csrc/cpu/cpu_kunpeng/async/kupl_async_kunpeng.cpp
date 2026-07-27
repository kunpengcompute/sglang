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
#include <string>
#include <vector>
#include "kupl.h"
#include <pybind11/stl.h>
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

class PyKuplEgroup {
    kupl_egroup_h handle_ = nullptr;

public:
    PyKuplEgroup(std::vector<int> executors) {
        if (executors.empty()) {
            throw std::runtime_error("executors list must not be empty");
        }
        int n = kupl_get_num_executors();
        for (int eid : executors) {
            if (eid < 1 || eid >= n) {
                throw std::runtime_error(
                    "executor id " + std::to_string(eid) +
                    " out of range [1, " + std::to_string(n - 1) + "]");
            }
        }
        handle_ = kupl_egroup_create(executors.data(), executors.size());
        if (!handle_) {
            throw std::runtime_error("kupl_egroup_create failed");
        }
    }

    ~PyKuplEgroup() {
        if (handle_) kupl_egroup_destroy(handle_);
    }

    PyKuplEgroup(const PyKuplEgroup &) = delete;
    PyKuplEgroup &operator=(const PyKuplEgroup &) = delete;

    kupl_egroup_h get() const { return handle_; }
};

class PyKuplExecutor {
    kupl_egroup_h egroup_ = nullptr;
    kupl_queue_h queue_ = nullptr;
    std::vector<TaskArgs*> tasks_;

public:
    PyKuplExecutor() {
        int n = kupl_get_num_executors();
        if (n <= 1) {
            throw std::runtime_error("KUPL executor count must be > 1 for async launch");
        }
        std::vector<int> executors(n - 1);
        for (int i = 0; i < n - 1; i++) executors[i] = i + 1;
        egroup_ = kupl_egroup_create(executors.data(), n - 1);
        queue_ = kupl_queue_create();
    }

    ~PyKuplExecutor() {
        for (auto *ta : tasks_) {
            if (ta) {
                PyGILState_STATE g = PyGILState_Ensure();
                Py_DECREF(ta->fn);
                Py_DECREF(ta->args);
                PyGILState_Release(g);
                delete ta;
            }
        }
        tasks_.clear();
        if (queue_) kupl_queue_destroy(queue_);
        if (egroup_) kupl_egroup_destroy(egroup_);
    }

    void submit(py::object fn, py::args args, py::object egroup) {
        auto *ta = new TaskArgs();
        ta->fn = fn.ptr();
        ta->args = args.ptr();
        ta->result = nullptr;
        Py_INCREF(ta->fn);
        Py_INCREF(ta->args);
        tasks_.push_back(ta);

        kupl_egroup_h eg;
        if (egroup.is_none()) {
            eg = egroup_;
        } else {
            eg = egroup.cast<PyKuplEgroup &>().get();
        }

        kupl_queue_item_desc_t desc = {};
        desc.func = py_task_body;
        desc.args = ta;
        desc.egroup = eg;
        desc.field_mask = KUPL_QUEUE_ITEM_DESC_FIELD_EGROUP;

        int ret = kupl_queue_submit(queue_, &desc);
        if (ret != 0) {
            printf("[kupl_async] kupl_queue_submit FAILED ret=%d\n", ret);
            fflush(stdout);
            throw std::runtime_error("kupl_queue_submit failed");
        }
    }

    py::list wait() {
        if (tasks_.empty()) {
            throw std::runtime_error("no task to wait");
        }

        {
            py::gil_scoped_release release;
            kupl_queue_wait(queue_);
        }

        py::list results;
        for (auto *ta : tasks_) {
            results.append(py::reinterpret_steal<py::object>(ta->result));
            Py_DECREF(ta->fn);
            Py_DECREF(ta->args);
            delete ta;
        }
        tasks_.clear();

        return results;
    }
};

PYBIND11_MODULE(_kupl_async, m) {
    py::class_<PyKuplEgroup>(m, "PyKuplEgroup")
        .def(py::init<std::vector<int>>());

    py::class_<PyKuplExecutor>(m, "PyKuplExecutor")
        .def(py::init<>())
        .def("submit", &PyKuplExecutor::submit,
             py::arg("fn"),
             py::arg("egroup") = py::none())
        .def("wait", &PyKuplExecutor::wait);
}