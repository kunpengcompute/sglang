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

#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <kupl.h>
#include "sgl_kernel_ops.h"
#include <cstdint>
#include <iostream>
#include "../utils/prf_memcpy.h"
#include "../utils/sdma_util.h"
#include "../utils/sdma_thres_util.h"

constexpr bool debug = false;
// 分块拷贝的阈值，用于将大块内存拷贝操作分割成多个小块异步拷贝任务
const int64_t ASYNC_COPY_THRES_SIZE = 14 * 1024 * 1024;

void sync_swap_kunpeng(at::Tensor dst, at::Tensor src, int64_t byte_size) {
    void* dst_ptr = dst.data_ptr();
    void* src_ptr = src.data_ptr();

    utils::prf_memcpy<true, true, 3 * 1024, SV_PLDL2STRM>(dst_ptr, src_ptr, static_cast<size_t>(byte_size));
}

// on_package_memory -> ddr, 异步将数据从on_package_memory拷贝回ddr
void queue_async_swapout_kunpeng(
    int64_t index,
    int64_t byte_size,
    int64_t byte_offset,
    at::Tensor src,                    // 源 tensor
    at::Tensor dst,                    // 目标 tensor
    at::Tensor ddr2swap,               // 1D tensor, shape: [BLOCK_NUM]
    at::Tensor swapout_tables,         // 2D tensor, shape: [BLOCK_NUM, MAX_EVENTS]
    at::Tensor swapout_lengths         // 1D tensor, shape: [BLOCK_NUM],
) {
    if (debug) {
        std::cout << "queue_async_swapout: index=" << index
                  << ", byte_size=" << byte_size
                  << ", byte_offset=0x" << std::hex << byte_offset << std::dec << std::endl;
    }
    // 获取指针
    int* ddr2swap_ptr = ddr2swap.data_ptr<int>();
    int swap_index = ddr2swap_ptr[index];
    TORCH_CHECK(swap_index != -1, "swap_index == -1, no swapin before");

    if (byte_size == -1) {
        // byte_size = swap_buffer_size;
        TORCH_CHECK(false, "byte_size == -1 not supported without swap_buffer_size");
    }

    int64_t buffer_stride = src.size(1) * src.size(2) * src.size(3) * src.element_size();
    void* src_ptr = (void*)(((int8_t*)src.data_ptr()) + swap_index * buffer_stride + byte_offset);
    void* dst_ptr = (void*)(((int8_t*)dst.data_ptr()) + byte_offset);

    int event_id = utils::kupl_get_free_event_id();
    utils::kupl_sdma_async(event_id, dst_ptr, src_ptr, byte_size);

    int* lengths_ptr = swapout_lengths.data_ptr<int>();
    int current_len = lengths_ptr[index];
    int64_t stride = swapout_tables.size(1);
    int* table_ptr = swapout_tables.data_ptr<int>();
    table_ptr[index * stride + current_len] = event_id;
    lengths_ptr[index] = current_len + 1;
    ddr2swap_ptr[index] = -1;
}

// ============ queue_async_swapin ============
int64_t queue_async_swapin_kunpeng(
    int64_t index,
    int64_t byte_size,
    int64_t now_buf_id,
    at::Tensor src,
    at::Tensor dst,
    at::Tensor ddr2swap,
    at::Tensor swapin_tables,
    at::Tensor swapin_lengths,
    int64_t num_swap_buffers
) {
    if (debug) {
        std::cout << "queue_async_swapin: index=" << index
                  << ", byte_size=" << byte_size << std::endl;
    }

    void* src_ptr = src.data_ptr();
    int64_t buffer_stride = dst.size(1) * dst.size(2) * dst.size(3) * dst.element_size();
    void* dst_ptr = (void*)(((int8_t*)dst.data_ptr()) + now_buf_id * buffer_stride);

    int* ddr2swap_ptr = ddr2swap.data_ptr<int>();
    ddr2swap_ptr[index] = now_buf_id;
    // 更新 now_buf_id
    now_buf_id = (now_buf_id + 1) % num_swap_buffers;

    if (byte_size == -1) {
        TORCH_CHECK(false, "byte_size == -1 not supported without swap_buffer_size");
    }

    int64_t send_num = (byte_size + ASYNC_COPY_THRES_SIZE - 1) / ASYNC_COPY_THRES_SIZE;

    int64_t stride = swapin_tables.size(1);
    int* table_ptr = swapin_tables.data_ptr<int>();
    int* lengths_ptr = swapin_lengths.data_ptr<int>();
    int current_len = lengths_ptr[index];

    for (int64_t i = 0; i < send_num; i++) {
        int event_id = utils::kupl_get_free_event_id();
        void* src_p = (void*)(((int8_t*)src_ptr) + i * ASYNC_COPY_THRES_SIZE);
        void* dst_p = (void*)(((int8_t*)dst_ptr) + i * ASYNC_COPY_THRES_SIZE);
        int64_t copy_size = std::min(byte_size - i * ASYNC_COPY_THRES_SIZE, ASYNC_COPY_THRES_SIZE);
        utils::kupl_sdma_async(event_id, dst_p, src_p, copy_size);
        table_ptr[index * stride + current_len + i] = event_id;
    }

    lengths_ptr[index] = current_len + send_num;
    return now_buf_id;
}

// ============ get_safe_on_package_memory_index ============
int64_t get_safe_on_package_memory_index_kunpeng(
    int64_t index,
    at::Tensor ddr2swap,
    at::Tensor swap2ddr,
    at::Tensor swapin_tables,
    at::Tensor swapout_tables,
    at::Tensor swapin_lengths,
    at::Tensor swapout_lengths
) {
    int* ddr2swap_ptr = ddr2swap.data_ptr<int>();
    int* swap2ddr_ptr = swap2ddr.data_ptr<int>();
    int* swapin_lengths_ptr = swapin_lengths.data_ptr<int>();
    int* swapout_lengths_ptr = swapout_lengths.data_ptr<int>();
    int swap_index = ddr2swap_ptr[index];
    TORCH_CHECK(swap_index != -1, "swap_index == -1, no swapin before");

    int ddr_index = swap2ddr_ptr[swap_index];
    if (ddr_index == index) {
        return swap_index;
    }

    int64_t swapout_stride = swapout_tables.size(1);
    int* swapout_tables_ptr = swapout_tables.data_ptr<int>();
    int64_t swapin_stride = swapin_tables.size(1);
    int* swapin_tables_ptr = swapin_tables.data_ptr<int>();

    bool need_wait0 = (ddr_index != -1 && swapout_lengths_ptr[ddr_index] > 0);
    bool need_wait1 = (swapin_lengths_ptr[index] > 0);

    if (need_wait0) {
        int len = swapout_lengths_ptr[ddr_index];
        for (int i = 0; i < len; i++) {
            int eid = swapout_tables_ptr[ddr_index * swapout_stride + i];
            utils::kupl_sdma_wait(eid);
        }
        swapout_lengths_ptr[ddr_index] = 0;
    }

    if (need_wait1) {
        int len = swapin_lengths_ptr[index];
        for (int i = 0; i < len; i++) {
            int eid = swapin_tables_ptr[index * swapin_stride + i];
            utils::kupl_sdma_wait(eid);
        }
        swapin_lengths_ptr[index] = 0;
    }

    swap2ddr_ptr[swap_index] = index;
    return swap_index;
}

void init_sdma(int64_t sdmathreshold)
{
    SdmaCtlThredInit();
    SetSdmaThreshold(sdmathreshold);
    utils::kupl_sdma_init();
}

void finalize_sdma()
{
    utils::kupl_sdma_clear();
    DevmemFdDestroy();
}
