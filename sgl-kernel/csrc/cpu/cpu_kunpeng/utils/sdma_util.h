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

#include <kupl.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>

namespace utils {
const inline int EVENT_NUM = 512;
const inline int QUEUE_NUM = 3;
inline kupl_event_h event[EVENT_NUM];
inline kupl_queue_h que[QUEUE_NUM];
inline int wait_flag[EVENT_NUM];

inline void kupl_sdma_init()
{
    for (int i = 0; i < EVENT_NUM; i++) {
        wait_flag[i] = 0;
        event[i] = kupl_event_create();
        TORCH_CHECK(event[i] != nullptr, "SDMA init failed: event[", i, "] creation returned null");
        if (i < QUEUE_NUM) {
            que[i] = kupl_queue_create();
            TORCH_CHECK(que[i] != nullptr, "SDMA init failed: queue[", i, "] creation returned null");
        }
    }
}

inline void kupl_sdma_async(int event_id, void* dest, const void* src, int byte_counts, int que_id = 0)
{
    TORCH_CHECK(que_id < QUEUE_NUM, "SDMA async failed: que_id ", que_id, " >= QUEUE_NUM ", QUEUE_NUM);
    TORCH_CHECK(wait_flag[event_id] == 0, "SDMA async failed: event_id ", event_id, " is already in use");
    wait_flag[event_id] = 1;
    int ret = kupl_memcpy_async(dest, src, byte_counts, nullptr, event[event_id]);
    TORCH_CHECK(ret == KUPL_OK, "SDMA async failed: kupl_memcpy_async returned ", ret);
}

inline void kupl_sdma_wait(int event_id)
{
    TORCH_CHECK(wait_flag[event_id] == 1, "SDMA wait failed: event_id ", event_id, " is not pending");
    kupl_event_wait(event[event_id]);
    wait_flag[event_id] = 0;
}

inline int kupl_get_free_event_id()
{
    for (int e_id = 0; e_id < EVENT_NUM; e_id++) {
        if (wait_flag[e_id] == 0) {
            return e_id;
        }
    }
    TORCH_CHECK(false, "SDMA: no free event ID available (all ", EVENT_NUM, " events in use)");
    return -1;
}

inline void kupl_sdma_close(int event_id)
{
    if (event[event_id] != nullptr) {
        kupl_event_destroy(event[event_id]);
        event[event_id] = nullptr;
    }
}

inline void kupl_sdma_clear()
{
    for (int i = 0; i < EVENT_NUM; i++) {
        if (wait_flag[i] == 1) {
            kupl_sdma_wait(i);
        }
        kupl_sdma_close(i);
    }
    for (int i = 0; i < QUEUE_NUM; i++) {
        if (que[i] != nullptr) {
            kupl_queue_destroy(que[i]);
            que[i] = nullptr;
        }
    }
}

};  // namespace utils
