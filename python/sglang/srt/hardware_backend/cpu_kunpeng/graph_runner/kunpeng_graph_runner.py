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
"""Kunpeng CPU 920F graph capture/replay runner and HBW pool management.

Integrates into ModelRunner via self.graph_runner (same pattern as
CPUGraphRunner / NPUGraphRunner).  Provides can_run() / replay() so
_forward_raw() intercepts decode, extend and idle modes before they
reach forward_decode/forward_extend/forward_idle.
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, List, Optional, Union

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.utils import (
    is_cpu_920f,
    is_kunpeng_graph_capture,
    is_kunpeng_graph_profile,
    is_kunpeng_hbw_pool,
    is_kunpeng_swap_expert,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

_is_kunpeng_hbw_pool = is_kunpeng_hbw_pool()
_is_kunpeng_swap_expert = is_kunpeng_swap_expert()
_is_kunpeng_graph_capture = is_kunpeng_graph_capture()
_is_kunpeng_graph_profile = is_kunpeng_graph_profile()
_is_scheduler_skip_all_gather = (
    os.environ.get("SGLANG_SCHEDULER_SKIP_ALL_GATHER", "0") == "1"
)


class KunpengGraphRunner:
    """Manages Kunpeng CPU 920F specific functionality:

    - HBW memory pool for model weights
    - Expert/KV swap buffers for HBM offloading
    - sglang graph capture/replay for decode, extend, and idle forward passes

    Fits into ModelRunner via the standard graph_runner protocol:
      can_run(forward_batch) → bool
      replay(forward_batch, skip_attn_backend_init, pp_proxy_tensors) → output
    """

    # Process-global graph pools shared across KunpengGraphRunner instances
    # (e.g. multi-layer eagle runners in one process). The SHM/HBW bump
    # allocators are process-global, so the first capture claims the remaining
    # bytes and all runners must reuse the same pool tensors. Lazily created
    # on first capture; never reset once created (SHM bytes cannot be
    # reclaimed).
    _graph_hbw_tensor: Optional[torch.Tensor] = None
    _graph_shm_tensor: Optional[torch.Tensor] = None

    def __init__(self, model_runner: ModelRunner):
        self.model_runner = model_runner

        # Graph capture state
        self.graph_fixed_weights: Optional[List[torch.Tensor]] = None
        self._sglang_graph_cache: OrderedDict = OrderedDict()
        self._sglang_graph_max_cache = int(
            os.environ.get("SGLANG_KUNPENG_GRAPH_CACHE_SIZE", "4")
        )

        # HBW pool and swap manager
        self.weight_hbw_pool = None
        self.swap_mgr = None

    @classmethod
    def create(
        cls, model_runner: ModelRunner
    ) -> Optional["KunpengGraphRunner"]:
        """Create and initialize the Kunpeng runner if on Kunpeng CPU 920F.

        Returns None on other platforms. HBW pool (when enabled) and swap
        manager are initialized here; ModelRunner only assigns the result.
        """
        if not is_cpu_920f():
            return None
        runner = cls(model_runner)
        if _is_kunpeng_hbw_pool:
            runner.init_hbw_pool()
        runner.init_swap_manager()
        return runner

    # ── Initialization ───────────────────────────────────────────────────

    def init_hbw_pool(self):
        """Initialize HBW memory pool for Kunpeng CPU backend."""
        from sglang.srt.hardware_backend.cpu_kunpeng.allocator.kunpeng_hbw_allocator import (
            KunpengHBWPool,
        )

        weights_pool_size_mb = int(
            os.environ.get("SGLANG_KUNPENG_WEIGTHS_HBW_POOL_SIZE_MB")
        )
        assert (
            weights_pool_size_mb > 0
        ), f"weights_pool_size_mb must be positive, got {weights_pool_size_mb}"
        weights_pool_size_bytes = weights_pool_size_mb * 1024 * 1024
        self.weight_hbw_pool = KunpengHBWPool.get_instance(
            pool_size_bytes=weights_pool_size_bytes,
            alignment=1024,
        )

    def init_swap_manager(self):
        """Initialize the Kunpeng swap manager for expert/KV offloading."""
        from sglang.srt.hardware_backend.cpu_kunpeng.swap_manager import (
            KunpengSwapManager,
        )

        self.swap_mgr = KunpengSwapManager.get_instance()

    def init_swap_buffers(self):
        """Initialize HBM swap buffers for expert weights and KV cache."""
        model = self.model_runner.model.model  # DeepseekV2Model or DeepseekModelNextN

        if self.swap_mgr.enable_swap_expert and (
            hasattr(model, "decoder") or model._first_moe_layer_idx is not None
        ):
            if hasattr(model, "decoder"):
                # DeepseekModelNextN: single decoder layer with MoE
                experts = model.decoder.mlp.experts
            else:
                # DeepseekV2Model: multiple layers
                experts = model.layers[model._first_moe_layer_idx].mlp.experts
            self.swap_mgr.init_expert_buffer(
                hidden_size=experts.hidden_size,
                moe_intermediate_size=experts.intermediate_size_per_partition,
                num_experts=experts.num_local_experts,
                moe_expert_dtype=experts.w13_weight.dtype,
            )

        if self.swap_mgr.enable_swap_kv_in:
            num_tokens_ddr = self.model_runner.token_to_kv_pool.kv_buffer[0].shape[0]
            head_num = self.model_runner.token_to_kv_pool.kv_buffer[0].shape[1]
            kv_cache_dim = self.model_runner.token_to_kv_pool.kv_buffer[0].shape[2]
            if self.swap_mgr.enable_swap_kv_blockwise:
                max_blocks = int(os.environ.get("SGLANG_KUNPENG_SWAP_MAX_KV_BLOCKS"))
                num_tokens = max_blocks * self.model_runner.page_size
            else:
                num_tokens = num_tokens_ddr

            logger.info(
                f"num_tokens={num_tokens}, num_tokens_ddr={num_tokens_ddr}, "
                f"head_num={head_num}, kv_cache_dim={kv_cache_dim}"
            )

            self.swap_mgr.init_kv_buffer(
                num_tokens=num_tokens,
                head_num=head_num,
                kv_cache_dim=kv_cache_dim,
                dtype=self.model_runner.kv_cache_dtype,
            )

    def move_weights_to_hbw(self):
        """Move model parameters to HBW memory pool."""
        if self.model_runner.is_draft_worker:
            if (
                self.model_runner.server_args.disaggregation_mode == "decode"
                and int(os.environ.get("DECODE_PP_SIZE", "1")) > 1
            ):
                logger.info(
                    "[weight load] decode (DECODE_PP_SIZE=%s): draft weights placed on HBW",
                    os.environ.get("DECODE_PP_SIZE", "1"),
                )
            else:
                return

        for name, param in self.model_runner.model.named_parameters():
            if "embed_tokens" in name:
                continue
            if "mlp.experts" in name:
                if _is_kunpeng_swap_expert:
                    continue
                tensor_hbw = self._move_expert_weights_to_hbw(name, param)
            else:
                tensor_hbw = self.weight_hbw_pool.move_to_hbw(param)
            param.data = tensor_hbw

    def _move_expert_weights_to_hbw(
        self, name: str, param: torch.nn.Parameter
    ) -> torch.Tensor:
        """Move one MoE expert weight param to HBW, dropping invalid (-1) slots.

        Mirrors the reference implementation: expert weights are stored only
        for the actual per-layer/per-rank expert count, so redundant/invalid
        (-1) slots never occupy HBM.

        Requires all valid slots to be a leading prefix ([0..k-1]) of the
        local slot range so the router's local expert index maps 1:1 onto
        the compressed tensor.
        """
        logger.debug("moe weight %s", name)

        import re
        import math

        from sglang.srt.distributed.parallel_state import (
            get_moe_expert_parallel_rank,
            get_moe_expert_parallel_world_size,
        )
        from sglang.srt.eplb.expert_location import (
            get_global_expert_location_metadata,
        )

        metadata = get_global_expert_location_metadata()
        if metadata is None:
            return self.weight_hbw_pool.move_to_hbw(param)

        m = re.match(r"^model\.layers\.(\d+)\.mlp\.experts\.\S+$", name)
        if m is None:
            logger.info("non-moe weight %s", name)
            return self.weight_hbw_pool.move_to_hbw(param)

        layer_id = int(m.group(1))
        ep_rank = get_moe_expert_parallel_rank()
        # Use the MOE EP world size (not metadata.ep_size, which is the global
        # world size and differs under PP), consistent with the routing remap.
        slots_per_rank = (
            metadata.num_physical_experts // get_moe_expert_parallel_world_size()
        )
        slots = metadata.physical_to_logical_map[
            layer_id,
            ep_rank * slots_per_rank : (ep_rank + 1) * slots_per_rank,
        ].tolist()
        logger.debug(
            "[KunpengHBW] %s param_shape=%s layer_id=%d ep_rank=%d "
            "ep_world=%d num_physical=%d slots_per_rank=%d slots=%s",
            name, tuple(param.shape), layer_id, ep_rank,
            get_moe_expert_parallel_world_size(), metadata.num_physical_experts,
            slots_per_rank, slots,
        )

        valid_local = [i for i, s in enumerate(slots) if s != -1]
        logger.debug("[KunpengHBW] %s valid_local=%s", name, valid_local)
        if len(valid_local) == len(slots):
            # No invalid slot on this layer/rank: plain move.
            return self.weight_hbw_pool.move_to_hbw(param)

        if valid_local != list(range(len(valid_local))):
            raise ValueError(
                f"Kunpeng HBW: invalid expert slot in the middle of layer "
                f"{layer_id} rank {ep_rank} slots={slots}; valid slots must "
                f"be a leading prefix for compressed expert storage."
            )

        compressed_shape = (len(valid_local),) + tuple(param.shape[1:])
        logger.debug(
            "[KunpengHBW] %s compressed_shape=%s numel=%d bytes=%d",
            name,
            compressed_shape,
            math.prod(compressed_shape),
            math.prod(compressed_shape) * param.dtype.itemsize,
        )
        hbw_tensor = self.weight_hbw_pool.alloc(compressed_shape, param.dtype)
        hbw_tensor.copy_(param[: len(valid_local)])
        logger.debug(
            "Kunpeng HBW: compressed %s %s -> %s",
            name,
            tuple(param.shape),
            tuple(hbw_tensor.shape),
        )
        return hbw_tensor

    def init_graph_capture(self):
        """Initialize graph capture by collecting model fixed weights."""
        from sglang.srt.graph.collect_weights import collect_model_weights

        self.graph_fixed_weights = collect_model_weights(self.model_runner.model)

    def init_after_model_load(self):
        """Post model-load initialization (HBW weight move + graph capture).

        Called by ModelRunner right after the model weights are loaded; the
        feature flags are checked internally.
        """
        if _is_kunpeng_hbw_pool:
            self.move_weights_to_hbw()
        if _is_kunpeng_graph_capture:
            self.init_graph_capture()

    def is_graph_active(self) -> bool:
        """Check if sglang graph capture/replay is enabled."""
        return _is_kunpeng_graph_capture

    # ── Standard graph_runner protocol ───────────────────────────────────

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        """Return True for all Kunpeng-supported modes.

        The runner handles both graph and eager internally via is_graph_active().
        When graph is disabled it falls through to plain model.forward().
        """
        return True

    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        """Unified graph replay entry point (called from _forward_raw).

        Handles attn backend init and kwargs construction internally,
        then dispatches to the correct mode-specific forward.
        """

        self.swap_mgr.set_idle_forward(forward_batch.forward_mode.is_idle())

        # Build common kwargs (mirrors forward_decode/forward_extend/forward_idle)
        kwargs: dict = {}
        if self.model_runner.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors

        if forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
            # ── Extend ──
            if not skip_attn_backend_init:
                if hasattr(self.model_runner.model, "prepare_forward_batch"):
                    self.model_runner.model.prepare_forward_batch(forward_batch)
                self.model_runner.attn_backend.init_forward_metadata(forward_batch)

            if forward_batch.input_embeds is not None:
                kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()
            if (
                forward_batch.replace_embeds is not None
                and forward_batch.replace_positions is not None
            ):
                # Token embedding overrides: get base embeddings, scatter replacements
                if "input_embeds" not in kwargs:
                    embed_layer = self.model_runner.model.get_input_embeddings()
                    kwargs["input_embeds"] = embed_layer(forward_batch.input_ids)
                kwargs["input_embeds"][forward_batch.replace_positions] = (
                    forward_batch.replace_embeds.to(kwargs["input_embeds"].dtype)
                )
            if not self.model_runner.is_generation:
                kwargs["get_embedding"] = True

            output = self._forward_extend(forward_batch, **kwargs)
            return self._wrap_output(output)

        elif forward_batch.forward_mode.is_decode_or_idle():
            if forward_batch.forward_mode.is_idle():
                # ── Idle ──
                if forward_batch.batch_size > 0:
                    self.model_runner.attn_backend.init_forward_metadata(
                        forward_batch
                    )

                if self.model_runner.support_pp and pp_proxy_tensors is not None:
                    hidden_states = pp_proxy_tensors.tensors["hidden_states"]
                    residual = pp_proxy_tensors.tensors["residual"]
                    kwargs["pp_proxy_tensors"].tensors["hidden_states"] = torch.empty(
                        1, dtype=hidden_states.dtype
                    )[:0].view(hidden_states.shape)
                    kwargs["pp_proxy_tensors"].tensors["residual"] = torch.empty(
                        1, dtype=residual.dtype
                    )[:0].view(residual.shape)

                output = self._forward_idle(forward_batch, **kwargs)
            else:
                # ── Decode ──
                if not skip_attn_backend_init:
                    if hasattr(self.model_runner.model, "prepare_forward_batch"):
                        self.model_runner.model.prepare_forward_batch(forward_batch)
                    if self.model_runner.server_args.enable_pdmux:
                        self.model_runner.decode_attn_backend.init_forward_metadata(
                            forward_batch
                        )
                        forward_batch.attn_backend = (
                            self.model_runner.decode_attn_backend
                        )
                    else:
                        self.model_runner.attn_backend.init_forward_metadata(
                            forward_batch
                        )

                output = self._forward_decode(forward_batch, **kwargs)

            return self._wrap_output(output)

        # Unsupported mode – should not reach here
        raise ValueError(
            f"KunpengGraphRunner: unsupported forward mode "
            f"{forward_batch.forward_mode}"
        )

    @staticmethod
    def _wrap_output(
        output: Union[PPProxyTensors, LogitsProcessorOutput, tuple],
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        """Normalise output into LogitsProcessorOutput or PPProxyTensors.

        Graph mode returns (logits, hidden_states) tuples; eager mode returns
        LogitsProcessorOutput directly.
        """
        if isinstance(output, (LogitsProcessorOutput, PPProxyTensors)):
            return output
        # Graph mode: (logits, hidden_states) tuple
        logits, hidden_states = output
        return LogitsProcessorOutput(
            next_token_logits=logits, hidden_states=hidden_states
        )

    # ── Mode-specific forward (internal) ─────────────────────────────────

    def _forward_decode(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Union[PPProxyTensors, tuple]:
        if not self.is_graph_active():
            # Eager fallback – plain model.forward
            return self.model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

        inputs = self._build_kunpeng_graph_inputs(forward_batch, kwargs)
        return self._graph_forward(
            forward_batch, inputs, "decode", use_hbw=True, **kwargs
        )

    def _forward_extend(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Union[PPProxyTensors, tuple]:
        if not self.is_graph_active():
            return self.model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

        inputs = self._build_kunpeng_graph_inputs(forward_batch, kwargs)
        return self._graph_forward(
            forward_batch,
            inputs,
            forward_batch.forward_mode.name,
            use_hbw=not self.model_runner.is_draft_worker,
            **kwargs,
        )

    def _forward_idle(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Union[PPProxyTensors, tuple]:
        if not self.is_graph_active():
            return self.model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

        inputs = self._build_kunpeng_graph_inputs(forward_batch, kwargs)
        return self._graph_forward(
            forward_batch,
            inputs,
            "idle",
            use_hbw=not self.model_runner.is_draft_worker,
            **kwargs,
        )

    # ── Graph internals ───────────────────────────────────────────────────

    def _build_kunpeng_graph_inputs(
        self, forward_batch: ForwardBatch, kwargs: dict
    ) -> List[torch.Tensor]:
        """Build the graph inputs for the Kunpeng graph path.

        The input set differs per forward mode; PP proxy tensors are appended
        when pipeline parallelism is active, and None entries are filtered out.
        """
        forward_mode = forward_batch.forward_mode
        meta = self.model_runner.attn_backend.forward_metadata

        if forward_mode.is_idle():
            forward_batch.input_ids = torch.tensor([], dtype=torch.int64)
            forward_batch.positions = torch.tensor([], dtype=torch.int64)
            inputs = [forward_batch.input_ids, forward_batch.positions]
        elif forward_mode.is_decode():
            inputs = [
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch.num_token_non_padded,
            ]
            if meta is not None:
                inputs.extend([meta.block_table, meta.seq_lens])
            inputs.extend(
                [forward_batch.out_cache_loc, self.model_runner.attn_backend._decode_meta]
            )
            if self.swap_mgr._blockwise_ddr_block_ids is not None:
                # Block-wise swap: the block_table passed to the attention
                # kernel is the remapped (HBM slot) version; the per-step
                # DDR/HBW block ids and the remapped cache_loc (for writing
                # new K/V into HBM) must be registered as graph inputs too.
                # meta.block_table is at index 3 here.
                inputs[3] = self.swap_mgr._blockwise_remapped_block_table
                inputs.append(self.swap_mgr._blockwise_hbw_cache_loc)
                inputs.append(self.swap_mgr._blockwise_ddr_block_ids)
                inputs.append(self.swap_mgr._blockwise_hbw_block_ids)
        else:
            inputs = [
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch.extend_seq_lens,
                forward_batch.out_cache_loc,
                forward_batch.num_token_non_padded,
                self.model_runner.attn_backend._decode_meta,
            ]
            if meta is not None:
                inputs.extend([meta.block_table, meta.seq_lens, meta.extend_seq_lens])
                if self.swap_mgr._blockwise_ddr_block_ids is not None:
                    inputs[-3] = self.swap_mgr._blockwise_remapped_block_table
            if self.swap_mgr._blockwise_ddr_block_ids is not None:
                inputs.append(self.swap_mgr._blockwise_ddr_block_ids)
                inputs.append(self.swap_mgr._blockwise_hbw_block_ids)
                inputs.append(self.swap_mgr._blockwise_hbw_cache_loc)
            if forward_batch.extend_prefix_lens is not None:
                inputs.append(forward_batch.extend_prefix_lens)

        spec_info = getattr(forward_batch, "spec_info", None)
        if self.model_runner.is_draft_worker and spec_info is not None:
            if forward_mode.is_idle():
                spec_info.hidden_states = torch.tensor(
                    [0, 7168], dtype=torch.bfloat16
                )
            if getattr(spec_info, "hidden_states", None) is not None:
                inputs.append(spec_info.hidden_states)

        # PP: proxy tensors consumed by this pipeline stage must be registered
        # as graph inputs so the capture system can track them.
        pp = kwargs.get("pp_proxy_tensors")
        if self.model_runner.support_pp and pp is not None:
            inputs.extend([pp.tensors["hidden_states"], pp.tensors["residual"]])

        return [item for item in inputs if item is not None]

    def _build_graph_fixed_tensors(
        self, forward_batch: ForwardBatch
    ) -> List[torch.Tensor]:
        fixed = self.graph_fixed_weights[:]
        for kv in self.model_runner.token_to_kv_pool.kv_buffer:
            fixed.append(kv)
        if self.model_runner.is_draft_worker:
            fixed.extend(
                [
                    self.model_runner.model.model.embed_tokens.weight,
                    self.model_runner.model.lm_head.weight,
                ]
            )
        else:
            # cos_sin_cache shared across all layers (skip PPMissingLayer on non-first PP rank)
            for _layer in self.model_runner.model.model.layers:
                if (
                    hasattr(_layer, "self_attn")
                    and _layer.self_attn.rotary_emb is not None
                ):
                    fixed.append(_layer.self_attn.rotary_emb.cos_sin_cache)
                    break
        if forward_batch.next_token_logits_buffer is not None:
            fixed.append(forward_batch.next_token_logits_buffer)
        # Expert HBM swap buffers (reused across layers, allocated once)
        if self.swap_mgr.enable_swap_expert:
            fixed.append(self.swap_mgr._expert_buffer_w13)
            fixed.append(self.swap_mgr._expert_buffer_w2)
            fixed.append(self.swap_mgr._expert_event_tensor)
            fixed.append(self.swap_mgr._expert_event_num_tensor)
        if self.swap_mgr.enable_swap_kv_in:
            fixed.append(self.swap_mgr._cur_kv_hbm)
            fixed.append(self.swap_mgr._kv_swap_in_event_tensor)
            fixed.append(self.swap_mgr._kv_swap_in_event_num_tensor)
        if self.swap_mgr.enable_swap_kv_out:
            fixed.append(self.swap_mgr._kv_ddr_event_tensor)
            fixed.append(self.swap_mgr._kv_ddr_event_num_tensor)
        try:
            from sglang.srt.layers.moe.token_dispatcher.kunpeng import (
                _KunpengDispatcherState,
            )

            state = _KunpengDispatcherState.get()
            for attr in (
                "parallel_policy",
                "dispatch_call_count",
                "dispatch_send_buf",
                "dispatch_recv_buf",
                "combine_send_buf",
                "combine_recv_buf",
                "recv_token_ids_buf",
                "recv_experts_offset",
                "combined_x",
                "topk_weights_buf",
                "topk_ids_flat_buf",
                "topk_ids_index_buf",
                "dynamic_remap_counter",
            ):
                t = getattr(state, attr, None)
                if t is not None:
                    fixed.append(t)
        except Exception:
            pass
        # EPLB static dispatch map (logical -> physical expert per layer),
        # consumed by remap_topk_ids_to_rank_slot_kunpeng during graph
        # capture.  Each MoE layer reads its own `[layer_id, :]` slice view,
        # so the FULL tensor must be registered as a fixed storage or the
        # slice lookup fails with "non-return-value parameter tensor not
        # registered".
        try:
            from sglang.srt.eplb.expert_location import (
                get_global_expert_location_metadata,
            )

            metadata = get_global_expert_location_metadata()
            if metadata is not None:
                if metadata.logical_to_rank_dispatch_physical_map is not None:
                    fixed.append(metadata.logical_to_rank_dispatch_physical_map)
                # Dynamic redundant-expert maps, consumed by
                # remap_topk_ids_to_rank_slot_dynamic_kunpeng during capture.
                # Like the static map, each MoE layer reads a `[layer_id, :]`
                # slice, so the FULL tensors must be registered as fixed storage.
                fixed.append(metadata.logical_to_all_physical_map)
                fixed.append(metadata.logical_to_all_physical_map_num_valid)
        except Exception:
            pass
        return fixed

    def _graph_forward(
        self,
        forward_batch: ForwardBatch,
        inputs: list,
        forward_mode: str,
        use_hbw: bool = False,
        **kwargs,
    ) -> Union[PPProxyTensors, tuple]:
        from sglang.srt.graph import capture, finalize

        assert (
            forward_batch.input_ids.dtype == torch.int64
        ), f"graph input_ids must be int64, got {forward_batch.input_ids.dtype}"
        total_tokens = forward_batch.input_ids.shape[0]
        # Received proxy tensors must be registered as graph inputs so the
        # capture system can track them (otherwise consuming graph ops fail).
        is_pp_graph = (
            kwargs.get("pp_proxy_tensors") is not None
            and self.model_runner.support_pp
        )
        # Batch size must be part of the key: per-sequence ops (e.g.
        # last_tokens, MTP pad/unpad) fix their output shape at capture, so two
        # batches with equal total_tokens but different sequence counts must
        # not share a graph.
        graph_cache_key = (
            forward_batch.forward_mode,
            total_tokens,
            forward_batch.batch_size,
            is_pp_graph,
        )

        if graph_cache_key not in self._sglang_graph_cache:
            while len(self._sglang_graph_cache) >= self._sglang_graph_max_cache:
                self._sglang_graph_cache.popitem(last=False)

            fixed = self._build_graph_fixed_tensors(forward_batch)

            with capture(inputs=inputs, fixed=fixed):
                # pp_proxy_tensors must be passed to model.forward() even in
                # non-decode modes (e.g. PP1 idle), otherwise the model's
                # assert (pp_proxy_tensors is not None) will fail.
                ret = self.model_runner.model.forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                )
                is_pp_output = isinstance(ret, PPProxyTensors)
                if is_pp_output:
                    hidden_states = ret.tensors["hidden_states"]
                    residual = ret.tensors["residual"]
                else:
                    logits = ret.next_token_logits
                    hidden_states = ret.hidden_states

            if is_pp_output:
                graph_outputs = [hidden_states, residual]
            else:
                graph_outputs = (
                    [logits, hidden_states] if hidden_states is not None else [logits]
                )

            if KunpengGraphRunner._graph_shm_tensor is None:
                remaining = torch.ops.sgl_kernel.shm_remaining_bytes_kunpeng()
                if remaining > 0:
                    KunpengGraphRunner._graph_shm_tensor = (
                        torch.ops.sgl_kernel.create_shm_tensor_kunpeng(
                            torch.uint8, [remaining]
                        )
                    )

            if use_hbw and _is_kunpeng_hbw_pool:
                if KunpengGraphRunner._graph_hbw_tensor is None:
                    remaining = self.weight_hbw_pool.largest_free_bytes
                    KunpengGraphRunner._graph_hbw_tensor = self.weight_hbw_pool.alloc(
                        (remaining,), torch.uint8
                    )
                graph = finalize(
                    graph_outputs,
                    external_pool=KunpengGraphRunner._graph_hbw_tensor,
                    external_shm_pool=KunpengGraphRunner._graph_shm_tensor,
                )
            else:
                graph = finalize(
                    graph_outputs,
                    external_shm_pool=KunpengGraphRunner._graph_shm_tensor,
                )
            graph.has_hidden_states = (not is_pp_output) and hidden_states is not None

            if _is_kunpeng_graph_profile:
                graph.enable_profile(True)
            self._sglang_graph_cache[graph_cache_key] = (
                graph,
                is_pp_graph,
                is_pp_output,
            )
            logger.info(
                f"[graph] captured mode={forward_batch.forward_mode.name}, "
                f"total_tokens={total_tokens}, batch_size={forward_batch.batch_size}, "
                f"pp={is_pp_graph}"
            )
        else:
            self._sglang_graph_cache.move_to_end(graph_cache_key)
            graph, is_pp_graph, is_pp_output = self._sglang_graph_cache[
                graph_cache_key
            ]

        t0 = time.time()
        outputs = graph.run(inputs)
        t1 = time.time()
        # With SGLANG_SCHEDULER_SKIP_ALL_GATHER the scheduler issues an idle
        # forward every iteration even when no requests arrive; skip logging
        # those replays so an idle server does not flood the log file.
        skip_idle_log = (
            _is_scheduler_skip_all_gather and forward_batch.forward_mode.is_idle()
        )
        if (
            not skip_idle_log
            and os.environ.get("SGLANG_KUNPENG_PP_PROFILE", "0") == "0"
        ):
            logger.info(f"[graph] run {1000 * (t1 - t0):.3f} ms")

        # Idle replays would grow the profile jsonl unboundedly.
        if _is_kunpeng_graph_profile and not skip_idle_log:
            from sglang.srt.graph.profile import write_profile

            profile_dir = os.environ.get("SGLANG_TORCH_PROFILER_DIR", "/tmp")
            path = os.path.join(
                profile_dir,
                f"sglang_graph_rank{self.model_runner.tp_rank}.jsonl",
            )
            row = graph.get_profile_row()
            op_names = graph.profile_op_names()
            write_profile(
                path,
                row,
                op_names,
                {
                    "forward_mode": forward_mode,
                    "count": len(op_names),
                    "total_tokens": total_tokens,
                    "batch_size": forward_batch.batch_size,
                },
            )

        if is_pp_output:
            hidden_states, residual = outputs
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        if graph.has_hidden_states:
            logits, hidden_states = outputs
            return logits, hidden_states
        (logits,) = outputs
        return logits, None
