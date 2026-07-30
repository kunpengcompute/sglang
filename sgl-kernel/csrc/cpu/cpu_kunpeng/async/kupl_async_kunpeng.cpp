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
#include <unordered_set>
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

class PyKuplQueue {
    kupl_queue_h handle_ = nullptr;

public:
    PyKuplQueue() {
        handle_ = kupl_queue_create();
        if (!handle_) {
            throw std::runtime_error("kupl_queue_create failed");
        }
    }

    ~PyKuplQueue() {
        if (handle_) kupl_queue_destroy(handle_);
    }

    PyKuplQueue(const PyKuplQueue &) = delete;
    PyKuplQueue &operator=(const PyKuplQueue &) = delete;

    kupl_queue_h get() const { return handle_; }
};

struct PendingTask {
    TaskArgs *args;
    kupl_queue_h queue;
};

class PyKuplExecutor {
    kupl_egroup_h default_egroup_ = nullptr;
    kupl_queue_h default_queue_ = nullptr;
    std::vector<PendingTask> tasks_;

public:
    PyKuplExecutor() {
        int n = kupl_get_num_executors();
        if (n <= 1) {
            throw std::runtime_error("KUPL executor count must be > 1 for async launch");
        }
        std::vector<int> executors(n - 1);
        for (int i = 0; i < n - 1; i++) executors[i] = i + 1;
        default_egroup_ = kupl_egroup_create(executors.data(), n - 1);
        default_queue_ = kupl_queue_create();
    }

    ~PyKuplExecutor() {
        for (auto &pt : tasks_) {
            if (pt.args) {
                PyGILState_STATE g = PyGILState_Ensure();
                Py_DECREF(pt.args->fn);
                Py_DECREF(pt.args->args);
                PyGILState_Release(g);
                delete pt.args;
            }
        }
        tasks_.clear();
        if (default_queue_) kupl_queue_destroy(default_queue_);
        if (default_egroup_) kupl_egroup_destroy(default_egroup_);
    }

    void submit(py::object fn, py::args args, py::object egroup, py::object queue) {
        auto *ta = new TaskArgs();
        ta->fn = fn.ptr();
        ta->args = args.ptr();
        ta->result = nullptr;
        Py_INCREF(ta->fn);
        Py_INCREF(ta->args);

        kupl_egroup_h eg;
        if (egroup.is_none()) {
            eg = default_egroup_;
        } else {
            eg = egroup.cast<PyKuplEgroup &>().get();
        }

        kupl_queue_h q;
        if (queue.is_none()) {
            q = default_queue_;
        } else {
            q = queue.cast<PyKuplQueue &>().get();
        }

        kupl_queue_item_desc_t desc = {};
        desc.func = py_task_body;
        desc.args = ta;
        desc.egroup = eg;
        desc.field_mask = KUPL_QUEUE_ITEM_DESC_FIELD_EGROUP;

        int ret = kupl_queue_submit(q, &desc);
        if (ret != 0) {
            printf("[kupl_async] kupl_queue_submit FAILED ret=%d\n", ret);
            fflush(stdout);
            Py_DECREF(ta->fn);
            Py_DECREF(ta->args);
            delete ta;
            throw std::runtime_error("kupl_queue_submit failed");
        }

        tasks_.push_back({ta, q});
    }

    py::list wait() {
        if (tasks_.empty()) {
            throw std::runtime_error("no task to wait");
        }

        // 每个 queue 只 wait 一次
        std::unordered_set<kupl_queue_h> waited;
        {
            py::gil_scoped_release release;
            for (auto &pt : tasks_) {
                auto [it, inserted] = waited.insert(pt.queue);
                if (inserted) {
                    kupl_queue_wait(pt.queue);
                }
            }
        }

        py::list results;
        for (auto &pt : tasks_) {
            results.append(py::reinterpret_steal<py::object>(pt.args->result));
            Py_DECREF(pt.args->fn);
            Py_DECREF(pt.args->args);
            delete pt.args;
        }
        tasks_.clear();

        return results;
    }
};

PYBIND11_MODULE(_kupl_async, m) {
    py::class_<PyKuplEgroup>(m, "PyKuplEgroup")
        .def(py::init<std::vector<int>>());

    py::class_<PyKuplQueue>(m, "PyKuplQueue")
        .def(py::init<>());

    py::class_<PyKuplExecutor>(m, "PyKuplExecutor")
        .def(py::init<>())
        .def("submit", &PyKuplExecutor::submit,
             py::arg("fn"),
             py::arg("egroup") = py::none(),
             py::arg("queue") = py::none())
        .def("wait", &PyKuplExecutor::wait);
}