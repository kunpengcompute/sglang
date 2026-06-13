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

from .forward_methods import AttnForwardMethod
from .forward_mha import DeepseekMHAForwardMixin
from .forward_mha_kunpeng import DeepseekMHAKunpengForwardMixin
from .forward_mla import DeepseekMLAForwardMixin
from .forward_mla_fused_rope_cpu import DeepseekMLACpuForwardMixin
from .forward_mla_fused_rope_rocm import DeepseekMLARocmForwardMixin
from .forward_mla_kunpeng import DeepseekMLAKunpengForwardMixin

__all__ = [
    "AttnForwardMethod",
    "DeepseekMHAForwardMixin",
    "DeepseekMLACpuForwardMixin",
    "DeepseekMHAKunpengForwardMixin",
    "DeepseekMLAForwardMixin",
    "DeepseekMLAKunpengForwardMixin",
    "DeepseekMLARocmForwardMixin",
]
