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

import mmap
import os
import subprocess
import struct
import numpy as np
import time
from numpy.typing import ArrayLike
import atomics
import logging
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class KunpengShmConnector:
    """
    负责将 tokens / slot_mapping / block_table / ctrl 信息
    写入 /dev/shm/deepseek_*_{local_rank}。
    共享内存文件由外部预先创建和分配好，不在此类中调整大小。
    """
    def __init__(self, local_rank: int, shm_dir: str = "/dev/shm", prefix: str = "deepseek"):
        self.local_rank = int(local_rank)
        self.shm_dir = shm_dir.rstrip("/")
        self.prefix = prefix

        self.input_ids_path = f"{self.shm_dir}/{self.prefix}_input_ids_{self.local_rank}"
        self.out_cache_loc_path = f"{self.shm_dir}/{self.prefix}_out_cache_loc_{self.local_rank}" # 128 * 2048 * 8
        self.seq_lens_path = f"{self.shm_dir}/{self.prefix}_out_cache_loc_lens_{self.local_rank}" # 128 * 8
        self.seq_to_slot_path = f"{self.shm_dir}/{self.prefix}_seq_to_slot_{self.local_rank}" # 128 * 4096 * 8
        self.seq_to_slot_lens_path = f"{self.shm_dir}/{self.prefix}_seq_to_slot_lens_{self.local_rank}" # 128 * 8
        self.ctrl_path = f"{self.shm_dir}/{self.prefix}_ctrl_{self.local_rank}" # 64 bytes
        self.output_path = f"{self.shm_dir}/{self.prefix}_output_{self.local_rank}" # 128 * 129280 * 2
        self.output_tokens_path = f"{self.shm_dir}/{self.prefix}_output_tokens_{self.local_rank}"  # 128 * 1 * 8

        # ================== 预分配 ==================
        self._create_file(self.input_ids_path, 128 * 2048, 8)
        self._create_file(self.out_cache_loc_path, 128 * 2048, 8)
        self._create_file(self.seq_lens_path, 128, 8)
        self._create_file(self.seq_to_slot_path, 128 * 8192, 8)
        self._create_file(self.seq_to_slot_lens_path, 128, 8)
        self._create_file(self.ctrl_path, 64, 1)
        self._create_file(self.output_path, 128 * 129280, 2)
        self._create_file(self.output_tokens_path, 128 * 1, 8)

        logging.basicConfig(
            level=logging.INFO,
            format=f"[%(asctime)s.%(msecs)03d TP{self.local_rank}] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )

    # ================== 工具函数 ==================
    def _create_file(self, path: str, total_elements: int, data: int) -> None:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, mode=0o600)
        os.ftruncate(fd, total_elements * data)
        mm = mmap.mmap(fd, total_elements * data, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

    @staticmethod
    def _write_bytes(path: str, data: bytes) -> None:
        """将字节写入预分配的共享内存文件中。"""
        fd = os.open(path, os.O_CREAT | os.O_RDWR, mode=0o600)
        try:
            # 文件通过 truncate 预分配到足够大小，避免出现error: mmap length is greater than file size
            logger.debug("[write] %s, size=%s bytes", path, len(data))
            mm = mmap.mmap(fd, len(data), access=mmap.ACCESS_WRITE)
            try:
                mm.seek(0)
                mm.write(data)
                mm.flush()
            finally:
                mm.close()
        finally:
            os.close(fd)

    # ================== 数据写入接口 ==================

    def write_data_array(self, data: ArrayLike, path: str) -> None:
        arr = np.asarray(data, dtype=np.int64)
        self._write_bytes(path, arr.tobytes())

    def write_ctrl(
        self,
        *,
        is_prefill: int,
        n_seqs: int,
        seq_len: int,
        cur_len: int,
        extend_num_tokens: int,
        seq_lens_sum: int,
        ready_state: int = 3,
    ) -> None:
        fd = os.open(self.ctrl_path, os.O_CREAT | os.O_RDWR, mode=0o600)
        try:
            # 预分配文件大小
            os.ftruncate(fd, 64)
            mm = mmap.mmap(fd, 64, access=mmap.ACCESS_WRITE)
            try:
                # 初始化阶段
                struct.pack_into("i", mm, 4, int(is_prefill))
                struct.pack_into("i", mm, 8, int(n_seqs))
                struct.pack_into("i", mm, 12, int(seq_len))
                struct.pack_into("i", mm, 16, int(cur_len))
                struct.pack_into("i", mm, 20, int(extend_num_tokens))
                struct.pack_into("i", mm, 24, int(seq_lens_sum))
                mm.flush()

                # 切换到 READY
                buf = memoryview(mm).cast('B')
                view = buf[0:4]
                with atomics.atomicview(buffer=view, offset=0, atype=atomics.INT) as a:
                    a.store(int(ready_state))
                mm.flush()

                time_break = 8000
                res = 0
                while time_break > 0:
                    with atomics.atomicview(buffer=view, offset=0, atype=atomics.INT) as a:
                        st = a.load()
                    if st == 5:  # DONE
                        res = 1
                        break
                    time.sleep(0.001)
                    time_break -= 1

                if res == 0:
                    logger.info("forward failed")
            finally:
                del buf
                del view
                mm.close()
        finally:
            os.close(fd)

        logger.debug("[kpinfer_shm_connector] ctrl written (state=%s, is_prefill=%s, n_seqs=%s, seq_len=%s, cur_len=%s)",
                     ready_state, is_prefill, n_seqs, seq_len, cur_len)

    def send_task(
        self,
        forward_batch: ForwardBatch,
        *,
        is_prefill: int,
    ):
        """
        一次性写入 tokens / slot_mapping / block_table / ctrl。
        """
        input_ids_arr = np.asarray(forward_batch.input_ids, dtype=np.int64)
        out_cache_loc_arr = np.asarray(forward_batch.out_cache_loc, dtype=np.int64)
        if is_prefill:
            out_cache_loc_lens_arr = np.asarray(forward_batch.extend_seq_lens, dtype=np.int64)
            extend_num_tokens = forward_batch.extend_num_tokens
        else :
            decode_seq_lens = np.ones(forward_batch.batch_size, dtype=np.int64)
            out_cache_loc_lens_arr = np.asarray(decode_seq_lens, dtype=np.int64)
            extend_num_tokens = forward_batch.batch_size

        seq_to_slot = np.empty(forward_batch.seq_lens_sum, dtype=np.int32)
        offset = 0
        seq_lens_arr = np.asarray(forward_batch.seq_lens, dtype=np.int64)
        req_to_token_arr = np.asarray(forward_batch.req_to_token_pool.req_to_token, dtype=np.int32)
        req_pool_indices_arr = np.asarray(forward_batch.req_pool_indices, dtype=np.int64)
        for i, req_idx in enumerate(req_pool_indices_arr):
            L = seq_lens_arr[i]
            src = req_to_token_arr[req_idx, :L]
            seq_to_slot[offset:offset + L] = src
            offset += L
        seq_to_slot_arr = np.asarray(seq_to_slot, dtype=np.int32)
        seq_to_slot_lens_arr = np.asarray(forward_batch.seq_lens, dtype=np.int64)

        self._write_bytes(self.input_ids_path, input_ids_arr.tobytes())
        self._write_bytes(self.out_cache_loc_path, out_cache_loc_arr.tobytes())
        self._write_bytes(self.seq_lens_path, out_cache_loc_lens_arr.tobytes())
        self._write_bytes(self.seq_to_slot_path, seq_to_slot_arr.tobytes())
        self._write_bytes(self.seq_to_slot_lens_path, seq_to_slot_lens_arr.tobytes())

        self.write_ctrl(
            is_prefill=is_prefill,
            n_seqs=forward_batch.batch_size,
            seq_len=0,
            cur_len=0,
            extend_num_tokens=extend_num_tokens,
            seq_lens_sum=forward_batch.seq_lens_sum,
        )

    def read_output(self, batch_size: int) -> torch.Tensor:
        forward_batch_size = batch_size * 129280 * 2
        with open(self.output_path, "rb") as f:
            data = f.read(forward_batch_size)
            if len(data) < forward_batch_size:
                raise EOFError(f"File is too small; Expected batch size: {forward_batch_size}; Current len:{len(data)}")
        out_tensor = torch.frombuffer(data, dtype=torch.bfloat16)
        return out_tensor.reshape(batch_size, 129280)

    def get_greedy_tokens(self, batch_size: int) -> torch.Tensor:
        forward_batch_size = batch_size * 1 * 8
        with open(self.output_tokens_path, "rb") as f:
            data = f.read(forward_batch_size)
            if len(data) < forward_batch_size:
                raise EOFError(f"File is too small; Expected batch size: {forward_batch_size}; Current len:{len(data)}")
        out_tensor = torch.frombuffer(data, dtype=torch.int64)
        out_tensor = out_tensor.reshape(batch_size)
        return out_tensor
