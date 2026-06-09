#pragma once

#include <kupl.h>

#include "check.h"

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
        PARAMETER_CHECK(event[i] != nullptr);
        if (i < QUEUE_NUM) {
            que[i] = kupl_queue_create();
            PARAMETER_CHECK(que[i] != nullptr);
        }
    }
}

inline void kupl_sdma_async(int event_id, void* dest, const void* src, int byte_counts, int que_id = 0)
{
    PARAMETER_CHECK(que_id < QUEUE_NUM);
    PARAMETER_CHECK(wait_flag[event_id] == 0);
    wait_flag[event_id] = 1;
    int ret = kupl_memcpy_async(dest, src, byte_counts, nullptr, event[event_id]);
    PARAMETER_CHECK(ret == KUPL_OK);
}

inline void kupl_sdma_wait(int event_id)
{
    PARAMETER_CHECK(wait_flag[event_id] == 1);
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
    PARAMETER_CHECK(false, "no free event id");
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
