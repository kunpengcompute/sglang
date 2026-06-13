# Copyright 2023-2024 SGLang Team
# Modifications Copyright 2026 Huawei Technologies Co., Ltd.
# This file has been modified from the original version by Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from enum import IntEnum, auto


class AttnForwardMethod(IntEnum):
    # Use multi-head attention
    MHA = auto()
    MHA_KUNPENG = auto()

    # Use absorbed multi-latent attention
    MLA = auto()
    MLA_KUNPENG = auto()

    # Use multi-head attention, but with KV cache chunked.
    # This method can avoid OOM when prefix lengths are long.
    MHA_CHUNKED_KV = auto()

    # Use multi-head attention, execute the MHA for prefix and extended kv in a single kernel
    # when the sequence lengths are below the threshold.
    MHA_ONE_SHOT = auto()

    # Use MLA but with fused RoPE
    MLA_FUSED_ROPE_ROCM = auto()

    # Use MLA with fused RoPE kernel for CPU
    MLA_FUSED_ROPE_CPU = auto()

    # Use multi-head attention for NPU
    MHA_NPU = auto()

    # Use absorbed multi-latent attention for NPU
    MLA_NPU = auto()

    # Use Deepseek V3.2 sparse multi-latent attention for NPU
    DSA_NPU = auto()
