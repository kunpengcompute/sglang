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

#include <ATen/Tensor.h>

#include "register_graph_kernels.h"

// Definitions live in moe/kunpeng_moe.cpp (same common_ops module).
void moe_etp_dispatch_share_kunpeng(at::Tensor packed_recv_x, at::Tensor token_ids,
                                    at::Tensor experts_offset, at::Tensor dense_buf,
                                    at::Tensor etp_experts_offset, at::Tensor etp_total,
                                    at::Tensor etp_seq, int64_t row_bytes, int64_t moe_tp_size);

void moe_etp_reduce_kunpeng(at::Tensor moe_down, at::Tensor etp_total, at::Tensor etp_seq,
                            int64_t hidden, int64_t moe_tp_size);

void etp_remap_topk_ids_kunpeng(at::Tensor topk_ids, at::Tensor topk_ids_index_buf,
                                int64_t num_tokens, int64_t topk, int64_t num_local_experts,
                                int64_t moe_tp_size);

static KernelRegistrar _r_moe_etp_dispatch_share(
    "moe_etp_dispatch_share_kunpeng",
    make_dispatch_v<decltype(&moe_etp_dispatch_share_kunpeng), &moe_etp_dispatch_share_kunpeng>);

static KernelRegistrar _r_moe_etp_reduce(
    "moe_etp_reduce_kunpeng",
    make_dispatch_v<decltype(&moe_etp_reduce_kunpeng), &moe_etp_reduce_kunpeng>);

static KernelRegistrar _r_etp_remap_topk_ids(
    "etp_remap_topk_ids_kunpeng",
    make_dispatch_v<decltype(&etp_remap_topk_ids_kunpeng), &etp_remap_topk_ids_kunpeng>);
