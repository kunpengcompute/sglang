# Copyright 2026 Huawei Technologies Co., Ltd.
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

from typing import TYPE_CHECKING, Optional
import os
import mmap
import numpy as np
import torch

from sglang.srt.mem_cache.memory_pool import (
    MLATokenToKVPool,
    get_tensor_size_bytes,
)

class KunpnengCPUMLATokenToKVPool(MLATokenToKVPool):
    "MLA Token pool for KunpengCPU"

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_nsa: bool = False,
        override_kv_cache_dim: Optional[int] = None,
        tp_rank: int = 0,
    ):
        self.tp_rank = tp_rank
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
            use_nsa=use_nsa,
            override_kv_cache_dim=override_kv_cache_dim,
        )


    def _create_buffers(self):
        shm_path = "/dev/shm/deepseek_kvcache_" + str(self.tp_rank)
        print(
            f"[KVCache] Opened shared memory: {shm_path}, initialized cache shape [{self.layer_num}, {self.size + self.page_size}, 1, {self.kv_cache_dim}]")
        fd = os.open(shm_path, os.O_CREAT | os.O_RDWR, mode=0o600)
        total_elements = self.layer_num * (self.size + self.page_size) * self.kv_cache_dim
        os.ftruncate(fd, total_elements * 2)  # bfloat16 2 bytes
        mm = mmap.mmap(fd, total_elements * 2, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        shared_kv_buffer_np = np.frombuffer(mm, dtype=np.uint16).reshape(self.layer_num,
                                                                         self.size + self.page_size, 1,
                                                                         self.kv_cache_dim)
        shared_kv_buffer_np[:] = 0
        mm.flush()
        self.kv_buffer = [
            torch.from_numpy(shared_kv_buffer_np[i].view(np.uint16)).view(torch.bfloat16)
            for i in range(self.layer_num)
        ]
