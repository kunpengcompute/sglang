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

void fake_indexer_topk_kunpeng(const at::Tensor &seq_lens, int64_t topk, at::Tensor indices);
void fake_indexer_topk_mtp_kunpeng(const at::Tensor &seq_lens, const c10::optional<at::Tensor> &extend_seq_lens,
                                   int64_t num_rows, int64_t topk, at::Tensor indices);

// 直写式: 张量序 = [seq_lens (输入), indices (输出)], 标量 = [topk],
// 与 Python GraphOp 调用序一致, 直接注册原函数.
static KernelRegistrar _r_fake_indexer(
    "fake_indexer_topk_kunpeng",
    make_dispatch_v<decltype(&fake_indexer_topk_kunpeng), &fake_indexer_topk_kunpeng>);

// MTP (TARGET_VERIFY / DRAFT_EXTEND / absorbed prefill) 变体: 张量序 =
// [seq_lens, extend_seq_lens (输入, verify 传 None -> undefined),
// indices (输出)], 标量 = [num_rows, topk]. 内核侧自行处理 undefined 的
// extend_seq_lens. 图分发不接受 optional 张量类型 (make_dispatch_v 只认
// 裸 at::Tensor), 回放把 None 记录为 undefined 视图后以未定义张量传入,
// 这里转回 c10::optional 再调用原函数.
void fake_indexer_topk_mtp_graph(const at::Tensor &seq_lens, const at::Tensor &extend_seq_lens,
                                 int64_t num_rows, int64_t topk, at::Tensor indices)
{
    fake_indexer_topk_mtp_kunpeng(
        seq_lens,
        extend_seq_lens.defined() ? c10::optional<at::Tensor>(extend_seq_lens)
                                  : c10::nullopt,
        num_rows, topk, indices);
}

static KernelRegistrar _r_fake_indexer_mtp(
    "fake_indexer_topk_mtp_kunpeng",
    make_dispatch_v<decltype(&fake_indexer_topk_mtp_graph),
                    &fake_indexer_topk_mtp_graph>);
