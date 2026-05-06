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

import logging
import numpy as np
from sgl_kernel.cpu_kunpeng.shm_tools import KunpengShmConnector
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.utils import (get_bool_env_var)
import torch

from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_size,
    initialize_dp_attention,
)

logger = logging.getLogger(__name__)


class KunpengModelRunner:
    """A KunpengCPUModelRunner runs the forward passes of the models."""

    def __init__(self, kp_shm_connector: KunpengShmConnector):
        self.kp_shm_connector = kp_shm_connector


    def forward(
        self,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        logger.debug(
            "forward batch mode: %s, batchsize: %s, outcache: %s",
            forward_batch.forward_mode, forward_batch.batch_size, forward_batch.out_cache_loc)
        if forward_batch.forward_mode.is_decode():
            self.kp_shm_connector.send_task(forward_batch, is_prefill=0)
        elif forward_batch.forward_mode.is_extend():
            self.kp_shm_connector.send_task(forward_batch, is_prefill=1)
        elif forward_batch.forward_mode.is_idle():
            self.kp_shm_connector.write_ctrl(is_prefill=2, n_seqs=2, seq_len=128, cur_len=1, extend_num_tokens=1,
                                             seq_lens_sum=1)

        ret = LogitsProcessorOutput(next_token_logits=None)

        return ret
