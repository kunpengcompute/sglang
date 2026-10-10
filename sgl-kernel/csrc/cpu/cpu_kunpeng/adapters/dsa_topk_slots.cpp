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

#include "register_graph_kernels.h"
#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <cstdint>

void dsa_topk_slots_kunpeng(const at::Tensor &req_to_token, const at::Tensor &req_pool_indices,
                            const c10::optional<at::Tensor> &remapped_block_table,
                            const at::Tensor &topk_indices, const at::Tensor &seq_lens,
                            int64_t page_size, int64_t row_start,
                            at::Tensor slots, at::Tensor topk_length);

// 直写式: 张量序 = [req_to_token, req_pool_indices, remapped_block_table,
// topk_indices, seq_lens (输入), slots, topk_length (输出)], 标量序 =
// [page_size, row_start] -- 与 Python GraphOp 调用序一致. 图分发不接受
// optional 张量类型 (make_dispatch_v 只认裸 at::Tensor), 回放把 None 记录为
// undefined 视图后以未定义张量传入, 这里转回 c10::optional 再调用原函数.
void dsa_topk_slots_graph(const at::Tensor &req_to_token, const at::Tensor &req_pool_indices,
                          const at::Tensor &remapped_block_table,
                          const at::Tensor &topk_indices, const at::Tensor &seq_lens,
                          int64_t page_size, int64_t row_start,
                          at::Tensor slots, at::Tensor topk_length)
{
    dsa_topk_slots_kunpeng(
        req_to_token, req_pool_indices,
        remapped_block_table.defined() ? c10::optional<at::Tensor>(remapped_block_table)
                                       : c10::nullopt,
        topk_indices, seq_lens, page_size, row_start, slots, topk_length);
}

static KernelRegistrar _r_dsa_topk_slots(
    "dsa_topk_slots_kunpeng",
    make_dispatch_v<decltype(&dsa_topk_slots_graph), &dsa_topk_slots_graph>);
