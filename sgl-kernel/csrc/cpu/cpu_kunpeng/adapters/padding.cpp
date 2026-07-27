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

void pad_q_left_mtp_kunpeng(at::Tensor q_heads, at::Tensor ext_lens, at::Tensor q_padded);

void unpad_o_right_mtp_kunpeng(at::Tensor o_padded, at::Tensor ext_lens, at::Tensor o_flat);

static KernelRegistrar _r_pad_q_left_mtp("pad_q_left_mtp_kunpeng",
                                         make_dispatch_v<decltype(&pad_q_left_mtp_kunpeng), &pad_q_left_mtp_kunpeng>);

static KernelRegistrar _r_unpad_o_right_mtp(
    "unpad_o_right_mtp_kunpeng", make_dispatch_v<decltype(&unpad_o_right_mtp_kunpeng), &unpad_o_right_mtp_kunpeng>);