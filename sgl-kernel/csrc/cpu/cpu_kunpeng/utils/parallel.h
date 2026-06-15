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

#include <kutacc.h>
#include <kupl.h>
#include <arm_sve.h>
#include "math.h"

#pragma once

struct func_args {
    int64_t begin;
    int64_t end;
    int64_t chunk_size;
    const std::function<void(int64_t, int64_t)> &f;
};

void parallel_for(int64_t begin, int64_t end, int64_t grain_size, const std::function<void(int64_t, int64_t)> &f)
{
    TORCH_CHECK(grain_size > 0, "grain_size invalid: ", grain_size);
    if (begin >= end) {
        return;
    }
    int64_t num_threads = std::min(kutacc::get_thread_num(), kmath::divup(end - begin, grain_size));
    int64_t chunk_size = kmath::divup(end - begin, num_threads);
    if (num_threads == 1) {
        f(begin, end);
    } else {
#if defined(USE_OMP_PARALLEL)
#pragma omp parallel
        {
            int64_t tid = get_thread_id();
            int64_t begin_tid = begin + tid * chunk_size;
            if (begin_tid < end) {
                f(begin_tid, std::min(end, chunk_size + begin_tid));
            }
        }
#elif defined(USE_KSPIN_PARALLEL)
        struct Task {
            int64_t begin;
            int64_t end;
            int64_t chunk_size;
            const std::function<void(int64_t, int64_t)> &f;
            static void call(void *_task)
            {
                auto task = static_cast<Task *>(_task);
                int64_t tid = get_thread_id();
                int64_t beginTid = task->begin + tid * task->chunk_size;
                if (beginTid < task->end) {
                    task->f(beginTid, std::min(task->end, task->chunk_size + beginTid));
                }
            }
        };
        Task task{begin, end, chunk_size, f};
        kspin_run_with_pool(Task::call, &task);
#elif defined(USE_KUPL_PARALLEL)
        kupl_parallel_for_desc_t desc = {
            .field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT,
            .range = NULL,
            .egroup = NULL,
            .concurrency = static_cast<int>(num_threads),
            .policy = KUPL_LOOP_POLICY_STATIC
        };
        func_args args = {begin, end, chunk_size, f};
        kupl_parallel_for(&desc, parallel_for_kernel, &args);
#else
        for (int64_t tid = 0; tid < num_threads; tid++) {
            int64_t begin_tid = begin + tid * chunk_size;
            if (begin_tid < end) {
                f(begin_tid, std::min(end, begin_tid + chunk_size));
            }
        }
#endif
    }
}

static void parallel_for_kernel(kupl_nd_range_t *nd_range, void *args, int tid, int tnum)
{
    auto data = (func_args *) args;
    int64_t begin = data->begin;
    int64_t end = data->end;
    int64_t chunk_size = data->chunk_size;
    const std::function<void(int64_t, int64_t)> &f = data->f;

    int64_t begin_tid = begin + tid * chunk_size;
    if (begin_tid < end) {
        f(begin_tid, std::min(end, begin_tid + chunk_size));
    }
}