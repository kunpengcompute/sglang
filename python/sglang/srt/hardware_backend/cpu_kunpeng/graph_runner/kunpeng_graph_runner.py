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
from pathlib import Path
import threading
import time
import weakref
from collections import OrderedDict
from contextlib import nullcontext
from typing import TYPE_CHECKING, List, Optional, Union

import torch

from sglang.srt.hardware_backend.cpu_kunpeng.pp_perf import pp_span
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.utils import (
    is_cpu_920f,
    is_kunpeng_graph_capture,
    is_kunpeng_hbw_pool,
    is_kunpeng_hbw_resident_expert_weights,
    is_kunpeng_hbw_resident_graph,
    is_kunpeng_hbw_resident_non_expert_weights,
    is_kunpeng_swap_expert,
    validate_kunpeng_hbw_residency,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

_is_kunpeng_hbw_pool = is_kunpeng_hbw_pool()
# Residency switches: raise on conflicting config at import time so the
# process fails at the earliest possible point.
_hbw_resident_non_expert = is_kunpeng_hbw_resident_non_expert_weights()
_hbw_resident_expert = is_kunpeng_hbw_resident_expert_weights()
_hbw_resident_graph = is_kunpeng_hbw_resident_graph()
_is_kunpeng_swap_expert = is_kunpeng_swap_expert()
_is_kunpeng_graph_capture = is_kunpeng_graph_capture()
_is_cpu_920f = is_cpu_920f()
_is_scheduler_skip_all_gather = (
    os.environ.get("SGLANG_SCHEDULER_SKIP_ALL_GATHER", "0") == "1"
)
last_call_timestamp = 0

# ── Request-driven graph profile session (mirrors the torch profiler) ──────
# /start_profile (SchedulerProfilerMixin.start_profile) starts collecting a
# profile row per graph replay into an in-memory buffer; /stop_profile
# flushes the buffer into the per-rank jsonl files. Steady-state serving
# pays only a bool check per replay — no per-replay file I/O.
_graph_profile_lock = threading.Lock()
_graph_profile_active = False
_graph_profile_dir: Optional[str] = None
_graph_profile_records: list = []
_graph_profile_runners = weakref.WeakSet()


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
        # Register for the request-driven profile session so start/stop can
        # toggle enable_profile on all cached graphs of every runner
        # (target + MTP draft runners live in the same process).
        _graph_profile_runners.add(self)

        # Graph capture state
        self.graph_fixed_weights: Optional[List[torch.Tensor]] = None
        self._graph_provider = None
        self._graph_provider_resolved = False
        self._graph_bindings = None
        self._sglang_graph_cache: OrderedDict = OrderedDict()
        self._graph_capture_count = 0
        self._graph_replay_count = 0
        self._decode_coverage = {}
        self._prefill_coverage = {}
        self._sglang_graph_max_cache = int(
            os.environ.get("SGLANG_KUNPENG_GRAPH_CACHE_SIZE", "4")
        )

        # HBW pool and swap manager
        self.weight_hbw_pool = None
        self._forward_hbw = None
        self.swap_mgr = None

    @classmethod
    def create(cls, model_runner: ModelRunner) -> Optional["KunpengGraphRunner"]:
        """Create and initialize the Kunpeng runner if on Kunpeng CPU 920F.

        Returns None on other platforms. HBW pool (when enabled) and swap
        manager are initialized here; ModelRunner only assigns the result.
        """
        if not is_cpu_920f():
            return None
        residency = validate_kunpeng_hbw_residency()
        logger.info(
            "[HBW residency] non_expert_weights=%s expert_weights=%s graph=%s",
            residency["non_expert_weights"],
            residency["expert_weights"],
            residency["graph"],
        )
        runner = cls(model_runner)
        target = getattr(model_runner, "borrowed_qwen_target_runner", None)
        if target is not None:
            runner._borrowed_target_runner = target  # Strong owner/lifetime reference.
            owner = target.kunpeng_runner
            if (
                owner is None
                or owner.weight_hbw_pool is None
                or owner._forward_hbw is None
            ):
                raise ValueError(
                    "Attached MTP requires the initialized target pool and arena"
                )
            runner.weight_hbw_pool = owner.weight_hbw_pool
            runner._forward_hbw = owner._forward_hbw
            runner.swap_mgr = owner.swap_mgr
            return runner
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
            os.environ.get("SGLANG_KUNPENG_WEIGHTS_HBW_POOL_SIZE_MB")
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
        if getattr(self.model_runner, "borrowed_qwen_target_runner", None) is not None:
            if self.swap_mgr.enable_swap_expert or self.swap_mgr.enable_swap_kv_in:
                raise ValueError(
                    "Attached MTP requires independent DDR KV with swap disabled"
                )
            return
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

    def _move_parameter_to_hbw(self, name, param, label=None):
        """Move a parameter while preserving CPU920F scale backing views."""
        if not name.endswith(".weight_scale"):
            return self.weight_hbw_pool.move_to_hbw(param, label=label)
        owner = self.model_runner.model.get_submodule(name.rsplit(".", 1)[0])
        storage = getattr(owner, "_cpu920f_w8a8_scale_storage", None)
        if storage is None:
            return self.weight_hbw_pool.move_to_hbw(param, label=label)
        n = param.shape[0]
        if (
            param.dtype != torch.float32
            or tuple(param.shape) != (n, 1)
            or storage.dtype != torch.float32
            or tuple(storage.shape) != (n + 16, 1)
            or not param.is_contiguous()
            or not storage.is_contiguous()
            or param.storage_offset() != storage.storage_offset()
            or storage.storage_offset() != 0
            or param.data_ptr() != storage.data_ptr()
            or param.untyped_storage().data_ptr()
            != storage.untyped_storage().data_ptr()
            or storage.untyped_storage().nbytes() < (n + 16) * 4
        ):
            raise ValueError(
                "CPU920F logical weight scale lost its N+16 backing before HBW move: "
                + name
            )
        storage_hbw = self.weight_hbw_pool.move_to_hbw(storage, label=label)
        logical_hbw = storage_hbw[:n]
        if (
            logical_hbw.data_ptr() != storage_hbw.data_ptr()
            or logical_hbw.untyped_storage().data_ptr()
            != storage_hbw.untyped_storage().data_ptr()
            or logical_hbw.untyped_storage().nbytes() < (n + 16) * 4
        ):
            raise ValueError(
                "HBW scale view does not retain its physical tail: " + name
            )
        owner._cpu920f_w8a8_scale_storage = storage_hbw
        self._int8_scale_backing_hbw_count += 1
        self._int8_scale_backing_hbw_bytes += (
            storage_hbw.numel() * storage_hbw.element_size()
        )
        return logical_hbw

    def move_weights_to_hbw(self):
        """Move model parameters to HBW memory pool.

        Per-category residency switches (see utils/common.py):
        - SGLANG_KUNPENG_HBW_RESIDENT_NON_EXPERT_WEIGHTS=0 keeps non-expert
          weights in DDR.
        - SGLANG_KUNPENG_HBW_RESIDENT_EXPERT_WEIGHTS=0 keeps expert weights
          in DDR (layer-wise swap path when SGLANG_KUNPENG_SWAP_EXPERT=1).
        """
        if self.model_runner.is_draft_worker:
            if (
                self.model_runner.server_args.disaggregation_mode == "decode"
                and int(os.environ.get("DECODE_PP_SIZE", "1")) > 1
            ):
                logger.info(
                    "[weight load] decode (DECODE_PP_SIZE=%s): draft weights "
                    "moved to HBW per residency switches "
                    "(non_expert=%s expert=%s)",
                    os.environ.get("DECODE_PP_SIZE", "1"),
                    _hbw_resident_non_expert,
                    _hbw_resident_expert,
                )
            else:
                return

        pool = self.weight_hbw_pool
        non_expert_bytes = 0
        expert_bytes = 0

        self._int8_scale_backing_hbw_count = 0
        self._int8_scale_backing_hbw_bytes = 0
        for name, param in self.model_runner.model.named_parameters():
            if "embed_tokens" in name:
                continue
            if "mlp.experts" in name:
                if _is_kunpeng_swap_expert or not _hbw_resident_expert:
                    continue
                used_before = pool.used_bytes
                tensor_hbw = self._move_expert_weights_to_hbw(name, param)
                expert_bytes += pool.used_bytes - used_before
            else:
                if not _hbw_resident_non_expert:
                    continue
                used_before = pool.used_bytes
                try:
                    tensor_hbw = self._move_parameter_to_hbw(
                        name, param, label="non_expert_weights"
                    )
                except RuntimeError as exc:
                    if os.environ.get(
                        "SGLANG_KUNPENG_HBW_ALLOW_DDR_FALLBACK", "0"
                    ) != "1" or "HBW pool exhausted" not in str(exc):
                        raise
                    logger.warning(
                        "[KunpengHBMSpill] parameter=%s bytes=%d remains in DDR",
                        name,
                        param.numel() * param.element_size(),
                    )
                    continue
                non_expert_bytes += pool.used_bytes - used_before
            param.data = tensor_hbw
        if os.environ.get("SGLANG_KUNPENG_HBW_LOG_PLACEMENT", "0") == "1":
            logger.info(
                "[KunpengHBMPool] used_bytes=%d free_bytes=%d allocations=%d "
                "int8_scale_backings=%d int8_scale_backing_bytes=%d",
                self.weight_hbw_pool.used_bytes,
                self.weight_hbw_pool.free_bytes,
                self.weight_hbw_pool.num_allocated,
                self._int8_scale_backing_hbw_count,
                self._int8_scale_backing_hbw_bytes,
            )

        mb = 1024 * 1024
        logger.info(
            "[HBW weights] moved non_expert=%.1f MB (%s), expert=%.1f MB (%s); "
            "pool used=%.1f MB free=%.1f MB largest_free=%.1f MB",
            non_expert_bytes / mb,
            "resident" if _hbw_resident_non_expert else "DDR",
            expert_bytes / mb,
            "resident" if _hbw_resident_expert else "DDR",
            pool.used_bytes / mb,
            (pool.pool_size - pool.used_bytes) / mb,
            pool.largest_free_bytes / mb,
        )

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
            return self.weight_hbw_pool.move_to_hbw(param, label="expert_weights")

        # Target model: model.layers.<N>.mlp.experts.*
        # MTP draft model: decoder.mlp.experts.* (no layer id; use the last
        # layer's mapping row since the metadata has no MTP layer row).
        m = re.match(r"^(?:model\.layers\.(\d+)|decoder)\.mlp\.experts\.\S+$", name)
        if m is None:
            logger.info("non-moe weight %s", name)
            return self.weight_hbw_pool.move_to_hbw(param, label="expert_weights")

        layer_id = int(m.group(1)) if m.group(1) is not None else metadata.num_layers - 1
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
            return self.weight_hbw_pool.move_to_hbw(param, label="expert_weights")

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
        hbw_tensor = self.weight_hbw_pool.alloc(
            compressed_shape, param.dtype, label="expert_weights"
        )
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
        if getattr(self.model_runner, "borrowed_qwen_target_runner", None) is not None:
            owner = self._borrowed_target_runner.kunpeng_runner
            if (
                self.weight_hbw_pool is not owner.weight_hbw_pool
                or self._forward_hbw is not owner._forward_hbw
            ):
                raise RuntimeError("Borrowed target pool/arena identity changed")
            if _is_kunpeng_graph_capture:
                self.init_graph_capture()
            return  # No second move/pack/reservation of target-owned weights.
        if _is_kunpeng_hbw_pool:
            self.move_weights_to_hbw()
            from sglang.srt.hardware_backend.cpu_kunpeng.qwen import (
                enabled as qwen_enabled,
            )

            if qwen_enabled() and not self.model_runner.is_draft_worker:
                from sglang.srt.hardware_backend.cpu_kunpeng.qwen.backend import (
                    prepare_model_lm_head,
                )

                self._packed_lm_head_enabled = prepare_model_lm_head(
                    self.model_runner.model, self.weight_hbw_pool
                )
            initial_backing = self.weight_hbw_pool.backing_snapshot()
            self._shm_native_init_used_bytes = initial_backing.get("shm_used_bytes")
            arena_mib = int(os.environ.get("SGLANG_KUNPENG_FORWARD_HBW_ARENA_MB", "0"))
            if qwen_enabled() and arena_mib:
                from sglang.srt.hardware_backend.cpu_kunpeng.allocator.forward_hbw import (
                    ForwardHBWArena,
                )

                # Ordinary ATen and native tensor outputs need their own reusable
                # reservation before the common graph claims the remaining pool.
                # Gemma effective weights are buffers rather than Parameters.
                # Preserve the existing placement policy for embedding/vision/MTP.
                seen = {}
                for module in self.model_runner.model.modules():
                    if not getattr(module, "_kunpeng_qwen", False):
                        continue
                    buffer = getattr(module, "gemma_weight", None)
                    if buffer is not None:
                        key = buffer.untyped_storage().data_ptr()
                        if key not in seen:
                            seen[key] = self.weight_hbw_pool.move_to_hbw(buffer)
                        module.gemma_weight = seen[key]
                self._static_hbw_bytes = self.weight_hbw_pool.used_bytes - getattr(
                    self.weight_hbw_pool, "_payload_reserved_bytes", 0
                )
                self._forward_hbw = ForwardHBWArena(
                    self.weight_hbw_pool,
                    os.environ.get(
                        "SGLANG_KUNPENG_FORWARD_HBW_LIBRARY",
                        str(
                            Path(__import__("sgl_kernel").__file__).parent
                            / "libscoped_hbw_allocator.so"
                        ),
                    ),
                    arena_mib * 1024 * 1024,
                )
                logger.info(
                    "[ForwardHBWArena] capacity_bytes=%d graph_remaining_bytes=%d",
                    arena_mib * 1024 * 1024,
                    self.weight_hbw_pool.free_bytes,
                )
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

    def _graph_profile_path(self) -> str:
        # With PP>1, stages sharing a tp_rank would concurrently append
        # to the same file; split per pp_rank. The MTP draft runner is
        # constructed with pp_size=1/pp_rank=0 but only ever runs on the
        # last PP rank, so merge its records into that stage's file.
        if self.model_runner.is_draft_worker:
            pp_rank = self.model_runner.server_args.pp_size - 1
        else:
            pp_rank = self.model_runner.pp_rank
        rank_part = (
            f"pp{pp_rank}"
            f"_dp{self.model_runner.dp_rank}"
            f"_tp{self.model_runner.tp_rank}"
        )
        return os.path.join(_graph_profile_dir, f"sglang_graph_{rank_part}.jsonl")

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

        # DSA topk carry: expose the incoming top-k to the attention backend so
        # it can pre-window it into this rank's row slice before the capture (an
        # input storage may not get new views while capturing).
        if _is_cpu_920f:
            forward_batch.pp_topk_indices = (
                pp_proxy_tensors.tensors.get("topk_indices")
                if pp_proxy_tensors is not None
                else None
            )

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
                    if hidden_states.numel() != 0 or residual.numel() != 0:
                        # Aborted batch (e.g. PD transfer failure) arrives as
                        # idle with non-empty proxy tensors, which the idle
                        # graph cannot replay -- run eagerly to avoid a
                        # shape-mismatch crash.
                        return self.model_runner.model.forward(
                            forward_batch.input_ids,
                            forward_batch.positions,
                            forward_batch,
                            **kwargs,
                        )
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
                    with pp_span("prepare_forward_batch"):
                        if hasattr(self.model_runner.model, "prepare_forward_batch"):
                            self.model_runner.model.prepare_forward_batch(forward_batch)
                    with pp_span("init_forward_metadata"):
                        if self.model_runner.server_args.enable_pdmux:
                            self.model_runner.decode_attn_backend.init_forward_metadata(
                                forward_batch
                            )
                            forward_batch.attn_backend = (
                                self.model_runner.decode_attn_backend
                            )
                        else:
                            # Honor a per-step backend switch (multi-step MTP
                            # chain drafts pre-init draft_attn_backend.attn_backends[i]
                            # and skip re-init here); otherwise init the default.
                            self._resolve_attn_backend(
                                self.model_runner, forward_batch
                            ).init_forward_metadata(forward_batch)

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

    def memory_placement_snapshot(self):
        """Small on-demand ledger; never scans tensors in the timed forward."""
        pool = self.weight_hbw_pool
        if pool is None:
            return {"hbm_enabled": False}
        pool.sweep()
        graph = KunpengGraphRunner._graph_hbw_tensor
        maps = []
        with open("/proc/self/numa_maps") as stream:
            for line in stream:
                try:
                    address = int(line.split()[0], 16)
                except (ValueError, IndexError):
                    continue
                if address == pool._base_addr:
                    maps.append(line.strip())
        arena = self._forward_hbw.stats() if self._forward_hbw is not None else None
        backing = pool.backing_snapshot()
        from sglang.srt.hardware_backend.cpu_kunpeng.allocator.placement_snapshot import (
            physical_window_snapshot,
        )

        expected_node = int(self.model_runner.tp_rank) + 16
        if backing.get("transport_mode") == "split":
            shm_placement = physical_window_snapshot(
                backing["shm_base"],
                backing["shm_bytes"],
                expected_node,
                allow_regular=True,
            )
            private_placement = physical_window_snapshot(
                pool._base_addr, pool.pool_size, expected_node
            )
            placement = {
                "complete": shm_placement["complete"] and private_placement["complete"],
                "total_budget_bytes": backing["total_budget_bytes"],
                "ranges": {
                    "shm_payload": shm_placement,
                    "private_hbw": private_placement,
                },
            }
        else:
            placement = physical_window_snapshot(
                pool._base_addr, pool.pool_size, expected_node
            )
        raw_graph_bytes = getattr(self, "_graph_hbw_raw_bytes", 0)
        arena_bytes = 0 if arena is None else arena["capacity_bytes"]
        static_bytes = getattr(self, "_static_hbw_bytes", 0)
        prefix_bytes = backing.get("shm_prefix_bytes", 0)
        shm_base = backing.get("shm_base", pool._base_addr)
        private_reserved = getattr(pool, "_payload_reserved_bytes", 0)
        other_bytes = (
            pool.used_bytes
            - private_reserved
            - static_bytes
            - arena_bytes
            - raw_graph_bytes
        )
        from sglang.srt.hardware_backend.cpu_kunpeng.qwen import enabled as qwen_enabled

        prefill = None
        if qwen_enabled():
            from sglang.srt.hardware_backend.cpu_kunpeng.qwen import prefill

        arena_lo = (
            0 if self._forward_hbw is None else self._forward_hbw.backing.data_ptr()
        )
        arena_hi = arena_lo + arena_bytes
        workspace_rows = []
        for name, tensor in zip(
            ("q", "packed_k", "packed_v", "out", "scratch"),
            getattr(getattr(prefill, "_shared_flash_workspace", None), "_hbm_buffers", ()),
        ):
            workspace_rows.append(
                {
                    "name": "prefill." + name,
                    "bytes": tensor.nbytes,
                    "address": tensor.data_ptr(),
                    "in_forward_arena": arena_lo <= tensor.data_ptr()
                    and tensor.data_ptr() + tensor.nbytes <= arena_hi,
                }
            )
        provider = self._graph_provider
        for batch, (_, tensor) in getattr(provider, "dense_plans", {}).items():
            workspace_rows.append(
                {
                    "name": "dense." + str(batch),
                    "bytes": tensor.nbytes,
                    "address": tensor.data_ptr(),
                    "in_forward_arena": arena_lo <= tensor.data_ptr()
                    and tensor.data_ptr() + tensor.nbytes <= arena_hi,
                }
            )
        plans = []
        from collections import Counter

        for key, entry in self._sglang_graph_cache.items():
            captured = entry[0]
            row = {
                "cache_key": str(key),
                "plan_available": hasattr(captured, "describe_plan"),
            }
            if not row["plan_available"]:
                row["plan_unavailable_reason"] = "pinned executor has no describe_plan"
            if row["plan_available"]:
                (
                    storages,
                    views,
                    records,
                    inputs,
                    needed,
                    capacity,
                    shm_needed,
                    shm_capacity,
                ) = captured.describe_plan()
                export_path = os.environ.get("SGLANG_KUNPENG_M1_PLAN_EXPORT_PATH")
                if (
                    export_path
                    and int(self.model_runner.tp_rank) == 0
                    and not getattr(self, "_m1_plan_exported", False)
                    and len(key) >= 4
                    and key[0].is_decode()
                    and int(key[1])
                    == int(os.environ.get("SGLANG_KUNPENG_PLAN_EXPORT_TOKENS", "1"))
                    and int(key[2])
                    == int(os.environ.get("SGLANG_KUNPENG_PLAN_EXPORT_BATCH", "1"))
                    and not key[3]
                ):
                    import json

                    if not os.path.isabs(export_path) or os.path.exists(export_path):
                        raise RuntimeError(
                            "Graph plan export requires a fresh absolute path"
                        )
                    export = {
                        "pid": os.getpid(),
                        "rank": 0,
                        "cache_key": str(key),
                        "num_inputs": int(inputs),
                        "required_regular_bytes": int(needed),
                        "external_pool_bytes": int(capacity),
                        "required_shm_bytes": int(shm_needed),
                        "external_shm_pool_bytes": int(shm_capacity),
                        "storages": [
                            {
                                "id": int(s.id),
                                "pointer": int(s.data_ptr),
                                "bytes": int(s.size),
                                "born": int(s.born_op),
                                "death": int(s.death_op),
                                "type": str(s.memory_type),
                                "in_pool": bool(s.in_pool),
                            }
                            for s in storages
                        ],
                        "views": [
                            {
                                "id": int(v.id),
                                "storage": int(v.storage_id),
                                "shape": list(v.shape),
                                "stride": list(v.strides),
                                "offset": int(v.storage_offset),
                                "scalar_type": int(v.scalar_type),
                                "element_size": int(v.element_size),
                            }
                            for v in views
                        ],
                        "records": [
                            {
                                "ordinal": i,
                                "op_name": str(r.op_name),
                                "profile_name": str(r.profile_name),
                                "input_view_ids": list(r.input_view_ids),
                                "output_view_ids": list(r.output_view_ids),
                            }
                            for i, r in enumerate(records)
                        ],
                    }
                    temporary = export_path + ".tmp." + str(os.getpid())
                    with open(temporary, "x") as stream:
                        json.dump(export, stream, indent=2, allow_nan=False)
                        stream.write("\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, export_path)
                    self._m1_plan_exported = True
                row.update(
                    required_regular_bytes=needed,
                    external_pool_bytes=capacity,
                    required_shm_bytes=shm_needed,
                    external_shm_pool_bytes=shm_capacity,
                    storage_count=len(storages),
                    op_count=len(records),
                    op_counts=dict(Counter(record.op_name for record in records)),
                    profile_name_counts=dict(
                        Counter(record.profile_name for record in records)
                    ),
                    shm_outputs_outside_prefix=sum(
                        bool(storage.size)
                        and not (
                            shm_base <= storage.data_ptr
                            and storage.data_ptr + storage.size
                            <= shm_base + prefix_bytes
                        )
                        for storage in storages
                        if storage.in_pool and str(storage.memory_type).endswith("SHM")
                    ),
                    regular_outputs_outside_hbm=sum(
                        bool(storage.size)
                        and not (
                            pool._base_addr + private_reserved <= storage.data_ptr
                            and storage.data_ptr + storage.size
                            <= pool._base_addr + pool.pool_size
                        )
                        for storage in storages
                        if storage.in_pool
                        and str(storage.memory_type).endswith("REGULAR")
                    ),
                )
            plans.append(row)
        arena_ok = (
            arena is not None
            and placement["complete"]
            and arena["allocations"] > 0
            and arena["allocation_failures"] == 0
            and arena["allocator_ddr_fallback_count"] == 0
        )
        scope_coverage = {
            "c10_storage_allocations": arena_ok,
            "aten_tensor_outputs": arena_ok,
            "boxed_graph_replay": arena_ok
            and any(p.get("op_counts", {}).get("aten_boxed", 0) for p in plans)
            and self._graph_replay_count > 0,
            "persistent_prefill_and_dense_workspace": arena_ok
            and len(workspace_rows) >= 5
            and all(r["in_forward_arena"] for r in workspace_rows),
        }
        packed_lm_heads = []
        if getattr(self, "_packed_lm_head_enabled", False):
            from sglang.srt.hardware_backend.cpu_kunpeng.bf16_lm_head import (
                placement_snapshot,
            )

            packed_lm_heads = placement_snapshot(self.model_runner.model, pool)
        return {
            "hbm_enabled": True,
            "packed_lm_head_buffers": packed_lm_heads,
            "shared_hbw_backing": backing,
            "physical_hbm": placement,
            "allocation_path_coverage": scope_coverage,
            "forward_ddr_fallback_count": (
                None if arena is None else arena["allocator_ddr_fallback_count"]
            ),
            "persistent_workspace_buffers": workspace_rows,
            "prefill_shared_flash_calls": getattr(
                getattr(prefill, "_kutacc_prefill", None), "_shared_flash_calls", 0
            ),
            "tail_class_bytes": {
                "static_weight_scale_norm": static_bytes,
                "forward_arena_backing": arena_bytes,
                "regular_graph_raw": raw_graph_bytes,
                "other_allocations_and_alignment": other_bytes,
            },
            "native_init_used_bytes": getattr(
                self, "_shm_native_init_used_bytes", None
            ),
            "prefix_before_graph_bytes": getattr(
                self, "_shm_prefix_before_graph_bytes", None
            ),
            "pool_capacity_bytes": pool.pool_size,
            "rank_total_hbm_budget_bytes": getattr(
                pool, "total_budget_bytes", pool.pool_size
            ),
            "pool_base_address": pool._base_addr,
            "pool_used_bytes": pool.used_bytes,
            "static_weight_scale_buffer_bytes": getattr(
                self, "_static_hbw_bytes", None
            ),
            "largest_free_block_bytes": pool.largest_free_bytes,
            "pool_free_bytes": pool.free_bytes,
            "graph_backing_bytes": raw_graph_bytes,
            "graph_effective_backing_bytes": 0 if graph is None else graph.numel(),
            "graph_plans": plans,
            "eager_prefill_arena": arena,
            "int8_scale_backing_count": self._int8_scale_backing_hbw_count,
            "int8_scale_backing_bytes": self._int8_scale_backing_hbw_bytes,
            "pool_numa_maps": maps,
            "coverage_note": "allocator counters do not certify non-Tensor native scratch; inspect explicit scratch contract",
        }

    def _forward_hbw_scope(self):
        return (
            self._forward_hbw.forward()
            if self._forward_hbw is not None
            else nullcontext()
        )

    def _provider_for_forward(self):
        if not self._graph_provider_resolved:
            from sglang.srt.graph.bindings import resolve_graph_provider

            self._graph_provider = resolve_graph_provider(self.model_runner)
            self._graph_provider_resolved = True
        return self._graph_provider

    def _graph_supports_forward(self, forward_batch, kwargs):
        if getattr(self.model_runner, "borrowed_qwen_target_runner", None) is not None:
            # Generic inputs/fixed tensors are DeepSeek-specific; no draft provider yet.
            return False
        if not self.is_graph_active():
            return False
        provider = self._provider_for_forward()
        return provider is None or provider.supports(forward_batch, kwargs)

    def graph_coverage_snapshot(self):
        """Cumulative metadata only; safe to collect once after a test batch."""
        return {
            "capture_count": self._graph_capture_count,
            "replay_count": self._graph_replay_count,
            "decode": [dict(row) for _, row in sorted(self._decode_coverage.items())],
            "prefill": [dict(row) for _, row in sorted(self._prefill_coverage.items())],
            "prefill_backend": os.environ.get(
                "SGLANG_KUNPENG_QWEN_PREFILL_BACKEND", "torch"
            ),
            "shared_prefill_kv": os.environ.get(
                "SGLANG_KUNPENG_QWEN_SHARED_PREFILL_KV", "0"
            ),
        }

    def _decode_coverage_row(self, fb, supported, reason):
        key = (int(fb.batch_size), int(fb.input_ids.numel()), bool(supported), reason)
        row = self._decode_coverage.get(key)
        if row is None:
            row = dict(
                batch=key[0],
                tokens=key[1],
                supported=key[2],
                reason=reason,
                attempts=0,
                cache_misses=0,
                captures=0,
                replays=0,
                eager=0,
            )
            self._decode_coverage[key] = row
        return row

    def _forward_decode(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Union[PPProxyTensors, tuple]:
        supported = self._graph_supports_forward(forward_batch, kwargs)
        provider = self._provider_for_forward()
        reason = "supported"
        if not supported:
            if (
                getattr(self.model_runner, "borrowed_qwen_target_runner", None)
                is not None
            ):
                reason = "attached_draft_provider_unavailable"
            elif not self.is_graph_active():
                reason = "graph_disabled"
            elif provider is not None and hasattr(provider, "unsupported_reason"):
                reason = provider.unsupported_reason(forward_batch, kwargs)
            else:
                reason = "provider_unsupported"
        coverage = (
            self._decode_coverage_row(forward_batch, supported, reason)
            if provider is not None
            else None
        )
        if coverage is not None:
            coverage["attempts"] += 1
        if not supported:
            if coverage is not None:
                coverage["eager"] += 1
            # Eager fallback – plain model.forward
            with self._forward_hbw_scope():
                return self.model_runner.model.forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                )

        with pp_span("build_graph_inputs"):
            inputs = self._build_kunpeng_graph_inputs(forward_batch, kwargs)
        return self._graph_forward(
            forward_batch, inputs, "decode", use_hbw=True, **kwargs
        )

    def _forward_extend(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Union[PPProxyTensors, tuple]:
        provider = self._provider_for_forward()
        row = None
        if provider is not None:
            key = (int(forward_batch.input_ids.numel()), int(forward_batch.batch_size))
            row = self._prefill_coverage.setdefault(
                key,
                dict(
                    tokens=key[0],
                    batch=key[1],
                    calls=0,
                    max_prefix=0,
                    supported=0,
                    eager=0,
                    cache_misses=0,
                    cache_hits=0,
                    captures=0,
                    replays=0,
                    reasons={},
                ),
            )
            row["calls"] += 1
            prefixes = forward_batch.extend_prefix_lens_cpu
            if prefixes is not None:
                row["max_prefix"] = max(
                    row["max_prefix"], max(map(int, prefixes), default=0)
                )
        supported = self._graph_supports_forward(forward_batch, kwargs)
        if not supported:
            reason = (
                "attached_draft_provider_unavailable"
                if getattr(self.model_runner, "borrowed_qwen_target_runner", None)
                is not None
                else (
                    "graph_disabled"
                    if not self.is_graph_active()
                    else (
                        provider.unsupported_reason(forward_batch, kwargs)
                        if provider is not None
                        and hasattr(provider, "unsupported_reason")
                        else "provider_unsupported"
                    )
                )
            )
            if row is not None:
                row["eager"] += 1
                row["reasons"][reason] = row["reasons"].get(reason, 0) + 1
            with self._forward_hbw_scope():
                return self.model_runner.model.forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                )

        if row is not None:
            row["supported"] += 1
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
        if not self._graph_supports_forward(forward_batch, kwargs):
            with self._forward_hbw_scope():
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

    @staticmethod
    def _resolve_attn_backend(
        model_runner, forward_batch: ForwardBatch
    ):
        """Return the attention backend the current forward actually uses.

        EAGLEWorker.draft_forward switches `forward_batch.attn_backend` to a
        per-step backend (`draft_attn_backend.attn_backends[i]`) for multi-step
        MTP chain drafts. The graph capture/replay inputs must be built from
        THAT backend's metadata, otherwise the per-step switch is silently
        ignored and the captured attention runs with stale metadata.
        """
        backend = getattr(forward_batch, "attn_backend", None)
        return backend if backend is not None else model_runner.attn_backend

    def _build_kunpeng_graph_inputs(
        self, forward_batch: ForwardBatch, kwargs: dict
    ) -> List[torch.Tensor]:
        """Build the graph inputs for the Kunpeng graph path.

        The input set differs per forward mode; PP proxy tensors are appended
        when pipeline parallelism is active, and None entries are filtered out.
        """
        provider = self._provider_for_forward()
        if provider is not None:
            with self._forward_hbw_scope():
                self._graph_bindings = provider.bind(
                    self.model_runner, forward_batch, kwargs
                ).validate()
            return list(self._graph_bindings.inputs.values())
        self._graph_bindings = None
        forward_mode = forward_batch.forward_mode
        attn_backend = self._resolve_attn_backend(self.model_runner, forward_batch)
        meta = attn_backend.forward_metadata

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
            inputs.extend([forward_batch.out_cache_loc, attn_backend._decode_meta])
            if getattr(attn_backend, "_nsa_enabled", False):
                # DSA: full-batch seq lens consumed by the (fake) CPU indexer
                # graph op; the real indexer reads the per-step full-batch
                # view (metadata.indexer_*, built outside the capture).
                # Register the SAME views the ops consume.
                inputs.append(meta.full_seq_lens[: forward_batch.seq_lens.shape[0]])
            if getattr(attn_backend, "_nsa_enabled", False) and not getattr(
                attn_backend, "_lc_enabled", False
            ):
                # dsa_topk_slots reads req_to_token directly; register the
                # SAME views the op consumes.
                inputs.append(forward_batch.req_to_token_pool.req_to_token)
                inputs.append(meta.req_pool_indices)
                # index_share_for_mtp_iteration reuse steps (decode with a
                # carried top-k) never call the indexer inside the graph,
                # so the scoring view has no in-graph consumer -- skip it
                # (the carried top-k itself registers as a graph input in
                # the draft-worker block below). Scoring steps (step 0 /
                # the share disabled) register the SAME views the scoring
                # op consumes.
                _mtp_reuse_step = (
                    getattr(forward_batch, "reuse_dsa_topk_indices", False)
                    and getattr(
                        getattr(forward_batch, "spec_info", None),
                        "dsa_topk_indices",
                        None,
                    )
                    is not None
                )
                if (
                    meta.indexer_block_table is not None
                    and not _mtp_reuse_step
                ):
                    inputs.extend(
                        [
                            meta.indexer_seq_lens,
                            meta.indexer_cur_lens,
                            meta.indexer_block_table,
                        ]
                    )
            if getattr(attn_backend, "_lc_enabled", False):
                # Long-context decode CP: per-step sparse-attention metadata
                # (local KV indices + per-sequence counts) consumed by the
                # graph ops. Shapes vary per step, so they must be graph inputs.
                # The indices buffers must be registered as the SAME view the
                # kernel consumes: _forward_mla_paged_cp slices them to
                # [:, :seqlen_q, :] and seqlen_q == 1 in decode. With MTP the
                # persistent buffer has speculative_num_draft_tokens > 1 rows,
                # so registering the full buffer instead of the slice fails
                # capture with "input tensor has no matching view".
                inputs.extend(
                    [
                        meta.long_context_indices[:, :1, :],
                        meta.long_context_topk_length,
                        meta.long_context_real_topk_length,
                    ]
                )
                if meta.long_context_hbm_indices is not None:
                    # LC + block-wise KV swap: the per-step slot-remapped
                    # index buffer consumed by the sparse flash MLA kernel
                    # (reads the compacted HBM swap buffer).
                    inputs.append(meta.long_context_hbm_indices[:, :1, :])
            if self.swap_mgr._blockwise_ddr_block_ids is not None:
                # Block-wise swap: the block_table passed to the attention
                # kernel is the remapped (HBM slot) version; the per-step
                # DDR/HBW block ids and the remapped cache_loc (for writing
                # new K/V into HBM) must be registered as graph inputs too.
                # meta.block_table is at index 3 here. LC decode CP is
                # index-driven (no block_table consumer), so the replacement
                # only applies when a block_table was actually added.
                if meta.block_table is not None:
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
                attn_backend._decode_meta,
            ]
            if meta is not None:
                # Flat absorbed prefill: the per-row valid counts (fake:
                # exact values; real: the metadata-time upper bound) are
                # consumed by the sparse sched/kernel on BOTH paths, so
                # they register independently of the fake-only slots.
                if getattr(meta, "absorbed_topk_length", None) is not None:
                    inputs.append(meta.absorbed_topk_length)
                if getattr(meta, "absorbed_slots", None) is not None:
                    inputs.append(meta.absorbed_slots)
                inputs.extend([meta.block_table, meta.seq_lens, meta.extend_seq_lens])
                if self.swap_mgr._blockwise_ddr_block_ids is not None:
                    # Only replace when a block_table entry exists (non-LC
                    # extend); LC decode CP is index-driven and has no
                    # block_table (meta.block_table is None there, and the
                    # LC remapped table is None too).
                    if meta.block_table is not None:
                        inputs[-3] = self.swap_mgr._blockwise_remapped_block_table
            if getattr(attn_backend, "_lc_enabled", False) and (
                forward_mode.is_target_verify() or forward_mode.is_draft_extend()
            ):
                # Long-context decode CP + MTP: the sparse LC metadata
                # (persistent indices buffer + per-step topk lengths) is read
                # inside the captured forward (flash_mla_sparse_decode / LC
                # partial-O exchange), so it must be registered as graph
                # inputs here, mirroring the DECODE capture above. The shapes
                # are constant for a given batch size (the indices buffer is
                # fixed at (B, speculative_num_draft_tokens, MAX_TOPK)), so
                # the same tensors can back TARGET_VERIFY / DRAFT_EXTEND
                # captures as well.
                inputs.extend(
                    [
                        meta.long_context_indices,
                        meta.long_context_topk_length,
                        meta.long_context_real_topk_length,
                    ]
                )
                if meta.long_context_hbm_indices is not None:
                    # LC + block-wise KV swap: per-step slot-remapped index
                    # buffer consumed by the sparse flash MLA kernel.
                    inputs.append(meta.long_context_hbm_indices)
            absorbed_dsa_prefill = (
                forward_mode.is_extend_without_speculative()
                and getattr(attn_backend, "_nsa_enabled", False)
                and not getattr(attn_backend, "_lc_enabled", False)
            )
            if (
                getattr(attn_backend, "_nsa_enabled", False)
                and not getattr(attn_backend, "_lc_enabled", False)
                and (forward_mode.is_target_verify() or forward_mode.is_draft_extend())
            ):
                # DSA + MTP verify/draft-extend: the (fake) MTP indexer op
                # (inside the capture) reads the persistent FULL-batch seq
                # lens; the real indexer reads the per-step full-batch view
                # (metadata.indexer_*, built outside the capture). Register
                # the SAME views the ops consume. (The absorbed prefill no
                # longer consumes them.)
                inputs.append(meta.full_seq_lens[: forward_batch.seq_lens.shape[0]])
                # dsa_topk_slots reads req_to_token directly; register the
                # SAME views the op consumes.
                inputs.append(forward_batch.req_to_token_pool.req_to_token)
                inputs.append(meta.req_pool_indices)
                if meta.indexer_block_table is not None:
                    inputs.extend(
                        [
                            meta.indexer_seq_lens,
                            meta.indexer_cur_lens,
                            meta.indexer_block_table,
                        ]
                    )
                # DRAFT_EXTEND (R3 pre-pad scoring): the per-sequence ext
                # lens feed the indexer's pre-pad (pad_q_left) and the
                # post-scoring pad-row restore (fill_mtp_pad_prefix) inside
                # the capture; forward_cpu reads forward_batch.
                # extend_seq_lens directly (natively int32), so register
                # THAT tensor as the graph input anchor (same pattern as
                # the extend branch's inputs list).
                if forward_mode.is_draft_extend():
                    inputs.append(forward_batch.extend_seq_lens)
            if (
                absorbed_dsa_prefill
                and meta is not None
                and meta.indexer_block_table is not None
            ):
                # Real absorbed prefill: the indexer scoring view (built
                # outside the capture) is consumed by the scoring op
                # inside; register the SAME views (mirrors the decode /
                # MTP registration above).
                inputs.extend(
                    [
                        meta.indexer_seq_lens,
                        meta.indexer_cur_lens,
                        meta.indexer_block_table,
                    ]
                )
            if self.swap_mgr._blockwise_ddr_block_ids is not None:
                inputs.append(self.swap_mgr._blockwise_ddr_block_ids)
                inputs.append(self.swap_mgr._blockwise_hbw_block_ids)
                inputs.append(self.swap_mgr._blockwise_hbw_cache_loc)
            if (
                forward_batch.extend_prefix_lens is not None
                and not absorbed_dsa_prefill
            ):
                # The absorbed DSA prefill path never reads the prefix lens
                # (its windows come from full_seq_lens + extend_seq_lens);
                # only the MHA prefill kernels consume them.
                inputs.append(forward_batch.extend_prefix_lens)

        spec_info = getattr(forward_batch, "spec_info", None)
        if self.model_runner.is_draft_worker and spec_info is not None:
            if forward_mode.is_idle():
                spec_info.hidden_states = torch.tensor([0, 7168], dtype=torch.bfloat16)
            if getattr(spec_info, "hidden_states", None) is not None:
                inputs.append(spec_info.hidden_states)
            # index_share_for_mtp_iteration: the carried top-k (the
            # draft-extend seed / the previous step's publish, stuffed
            # by the worker or by _graph_forward after the producing
            # step's replay) feeds the later chain steps' captured
            # graphs, like spec_info.hidden_states above.
            if (
                forward_mode.is_decode()
                and getattr(forward_batch, "reuse_dsa_topk_indices", False)
                and getattr(spec_info, "dsa_topk_indices", None) is not None
            ):
                inputs.append(spec_info.dsa_topk_indices)

        # PP: proxy tensors consumed by this pipeline stage must be registered
        # as graph inputs so the capture system can track them.
        pp = kwargs.get("pp_proxy_tensors")
        if self.model_runner.support_pp and pp is not None:
            inputs.extend([pp.tensors["hidden_states"], pp.tensors["residual"]])
            # DSA topk carried from the previous stage; present only when that
            # stage had one to forward. The absorbed prefill path consumes the
            # backend's pre-windowed view (built outside the capture); every
            # other path consumes the raw tensor directly.
            pp_topk = pp.tensors.get("topk_indices")
            if pp_topk is not None:
                carry_slots = (
                    getattr(meta, "absorbed_carry_slots", None)
                    if forward_mode.is_extend_without_speculative()
                    else None
                )
                inputs.append(carry_slots if carry_slots is not None else pp_topk)

        return [item for item in inputs if item is not None]

    def _build_graph_fixed_tensors(
        self, forward_batch: ForwardBatch
    ) -> List[torch.Tensor]:
        if self._graph_bindings is not None:
            return list(self._graph_bindings.all_fixed_tensors())
        fixed = self.graph_fixed_weights[:]
        for kv in self.model_runner.token_to_kv_pool.kv_buffer:
            fixed.append(kv)
        # DSA real indexer: the index-K pool buffers (allocated separately
        # from the main kv_buffer by KunpengNSATokenToKVPool) and the
        # shared max-sized scoring workspace (stable tensor identity, see
        # KunpengCpuBackend._get_shared_indexer_workspace). The per-layer
        # indexer weights (wq_b / wk / weights_proj / k_norm) and the
        # rotary cos_sin caches are already covered by
        # collect_model_weights.
        attn_backend = self._resolve_attn_backend(self.model_runner, forward_batch)
        index_k_buffers = getattr(
            self.model_runner.token_to_kv_pool, "index_k_buffer", None
        )
        if index_k_buffers:
            for buf in index_k_buffers:
                if buf is not None:
                    fixed.append(buf)
        indexer_ws = getattr(attn_backend, "indexer_workspace", None)
        if indexer_ws is not None:
            fixed.append(indexer_ws)
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
        # Expert HBM swap buffers (reused across layers, allocated once).
        # Buffers may legitimately be None when the flag is on but this rank
        # has no expert/KV layers (e.g. disagg prefill PP0 with only dense
        # layers sets _first_moe_layer_idx=None), so skip None entries instead
        # of passing them to begin_capture (which rejects non-Tensor items).
        if self.swap_mgr.enable_swap_expert:
            for buf in (
                self.swap_mgr._expert_buffer_w13,
                self.swap_mgr._expert_buffer_w2,
                self.swap_mgr._expert_event_tensor,
                self.swap_mgr._expert_event_num_tensor,
            ):
                if buf is not None:
                    fixed.append(buf)
        if self.swap_mgr.enable_swap_kv_in:
            for buf in (
                self.swap_mgr._cur_kv_hbm,
                self.swap_mgr._kv_swap_in_event_tensor,
                self.swap_mgr._kv_swap_in_event_num_tensor,
            ):
                if buf is not None:
                    fixed.append(buf)
        if self.swap_mgr.enable_swap_kv_out:
            for buf in (
                self.swap_mgr._kv_ddr_event_tensor,
                self.swap_mgr._kv_ddr_event_num_tensor,
            ):
                if buf is not None:
                    fixed.append(buf)
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
                "recv_token_src_buf",
                "moe_sum_token_weights",
                "combined_x",
                "topk_weights_buf",
                "topk_ids_flat_buf",
                "topk_ids_index_buf",
                "dynamic_remap_counter",
                # ETP node-shared buffers: consumed by
                # moe_etp_dispatch_share_kunpeng / moe_etp_reduce_kunpeng
                # and (dense_buf/token_ids/offset) by the downstream IGEMM
                # ops via KunpengDispatchOutput.  None on non-ETP ranks.
                "etp_dense_buf",
                "etp_token_ids_buf",
                "etp_experts_offset",
                "etp_total",
                "etp_seq",
            ):
                t = getattr(state, attr, None)
                if t is not None:
                    fixed.append(t)
        except Exception:
            pass
        # at_trace cumulative activation counter, mutated in-place by
        # record_expert_activation_kunpeng on every replay. It must be a fixed
        # storage so each replay writes into the same memory Python diffs later.
        try:
            from sglang.srt.hardware_backend.cpu_kunpeng.at_trace import (
                get_counter_tensor,
            )

            at_counter = get_counter_tensor()
            if at_counter is not None:
                fixed.append(at_counter)
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
        # Long-context decode CP: the persistent SHM region views the sparse
        # flash MLA writes O/LSE into directly (lc_stage_base_buffers_kunpeng,
        # initialized at backend construction -- always exists here). Each
        # layer derives slice views (o_base[:b*sq], lse_base[:b*sq]) from
        # them, so the FULL base tensors must be registered as fixed storage
        # or the slice lookup fails with "non-return-value parameter tensor
        # not registered".
        stage_base = self._resolve_attn_backend(
            self.model_runner, forward_batch
        )._lc_stage_base
        if stage_base is not None:
            fixed.append(stage_base[0])
            fixed.append(stage_base[1])

        # Kunpeng SHM communicator tensors consumed by shm_dual_allgather_kunpeng
        # in the logits all-gather path (last PP rank, or every rank when PP is
        # disabled). They are allocated lazily by KunpengCommunicator outside the
        # graph, so their storages must be registered as fixed inputs; otherwise
        # the capture-time lookup fails with "non-return-value parameter tensor
        # not registered".
        if self.model_runner.pp_group.is_last_rank:
            try:
                from sglang.srt.distributed.parallel_state import get_tp_group

                comm = get_tp_group().kunpeng_communicator
                if comm is not None:
                    fixed.append(comm.dummy_tensor)
                    # shm_all_gather_into_tensor uses dim = input.size(1) as the key,
                    # i.e. the local vocab of logits (vocab_size // tp_size),
                    # not the global vocab_size, otherwise it would be registered as a different SHM tensor.
                    local_vocab_size = -(
                        -self.model_runner.model_config.vocab_size
                        // self.model_runner.tp_size
                    )
                    fixed.append(comm.get_shm_tensor(local_vocab_size))
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
            kwargs.get("pp_proxy_tensors") is not None and self.model_runner.support_pp
        )
        # Batch size must be part of the key: per-sequence ops (e.g.
        # last_tokens, MTP pad/unpad) fix their output shape at capture, so two
        # batches with equal total_tokens but different sequence counts must
        # not share a graph.
        # The backend identity joins the key when forward_batch.attn_backend is
        # switched away from the runner's default (multi-step MTP chain draft
        # steps assign draft_attn_backend.attn_backends[i] per step). Each such
        # backend owns distinct metadata tensors, so they must not share a
        # captured graph.
        attn_backend = self._resolve_attn_backend(self.model_runner, forward_batch)
        # Absorbed DSA prefill: the flat row count fixes the kernel/slots
        # shapes and is not a function of total_tokens/batch_size, so it
        # joins the cache key (shapes never depend on
        # sum(extend_seq_lens); the per-row data varies via the inputs).
        absorbed_prefill_key = (
            getattr(
                attn_backend.forward_metadata,
                "absorbed_flat_rows",
                None,
            )
            if (
                forward_batch.forward_mode.is_extend_without_speculative()
                and getattr(attn_backend, "_nsa_enabled", False)
                and not getattr(attn_backend, "_lc_enabled", False)
            )
            else None
        )
        # All-padding batches take the zero-output skip branch inside
        # _forward_mla_sparse_paged -- a Python branch frozen at capture,
        # so the flag must split the graph cache key.
        all_padding_key = bool(
            getattr(
                getattr(attn_backend, "forward_metadata", None), "sparse_all_padding", False
            )
        )
        # index_share_for_mtp_iteration: whether this draft chain step
        # enters the graph WITH a carried top-k (reuse branch -- no
        # scoring op, the carry is a graph input) or WITHOUT one (the
        # no-seed fallback re-scores). The branch is a Python condition
        # frozen at capture and the carry comes and goes across rounds
        # (first round after prefill has no seed), so the two variants
        # must not share a cached graph.
        mtp_carry_key = None
        if (
            self.model_runner.is_draft_worker
            and forward_batch.forward_mode.is_decode()
            and getattr(forward_batch, "reuse_dsa_topk_indices", False)
        ):
            _carry_si = forward_batch.spec_info
            mtp_carry_key = _carry_si is not None and (
                getattr(_carry_si, "dsa_topk_indices", None) is not None
            )
        graph_cache_key = (
            forward_batch.forward_mode,
            total_tokens,
            forward_batch.batch_size,
            is_pp_graph,
            (
                id(attn_backend)
                if attn_backend is not self.model_runner.attn_backend
                else None
            ),
            absorbed_prefill_key,
            all_padding_key,
            mtp_carry_key,
        )

        # index_share_for_mtp_iteration: the incoming carry (the
        # draft-extend seed for the chain's step 0, or the previous
        # step's publish inside the chain; None when absent). The
        # capture branch below distinguishes a freshly computed top-k
        # from a re-published one by object identity.
        _spec_info = forward_batch.spec_info
        _pre_topk_carry = (
            getattr(_spec_info, "dsa_topk_indices", None)
            if _spec_info is not None
            else None
        )
        if self._graph_bindings is not None:
            graph_cache_key += (self._graph_bindings.cache_signature(),)

        decode_coverage = None
        if self._graph_bindings is not None and forward_batch.forward_mode.is_decode():
            decode_coverage = self._decode_coverage_row(
                forward_batch, True, "supported"
            )
        prefill_coverage = None
        if self._graph_bindings is not None and forward_batch.forward_mode.name == "EXTEND":
            prefill_coverage = self._prefill_coverage[
                (int(total_tokens), int(forward_batch.batch_size))
            ]
        if graph_cache_key not in self._sglang_graph_cache:
            if prefill_coverage is not None:
                prefill_coverage["cache_misses"] += 1
            if decode_coverage is not None:
                decode_coverage["cache_misses"] += 1
            while len(self._sglang_graph_cache) >= self._sglang_graph_max_cache:
                self._sglang_graph_cache.popitem(last=False)

            fixed = self._build_graph_fixed_tensors(forward_batch)

            provider = self._provider_for_forward()
            context = (
                provider.capture_context()
                if provider is not None and hasattr(provider, "capture_context")
                else nullcontext()
            )
            capture_options = getattr(provider, "capture_options", {})
            with capture(inputs=inputs, fixed=fixed, **capture_options), context:
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
                pp_topk = None
                if is_pp_output:
                    hidden_states = ret.tensors["hidden_states"]
                    residual = ret.tensors["residual"]
                    pp_topk = ret.tensors.get("topk_indices")
                else:
                    logits = ret.next_token_logits
                    hidden_states = ret.hidden_states

            if is_pp_output:
                # DSA topk carry: the model appends the indexer top-k when this
                # stage ends on a skip-topk (shared) layer, for the next stage.
                if pp_topk is not None:
                    graph_outputs = [hidden_states, residual, pp_topk]
                else:
                    graph_outputs = [hidden_states, residual]
            else:
                graph_outputs = (
                    [logits, hidden_states] if hidden_states is not None else [logits]
                )

            # index_share_for_mtp_iteration: register the freshly
            # published top-k as an extra graph output so every replay
            # returns the fresh values (the model's Python publish
            # cannot run inside a captured graph). Two producers:
            #   - draft chain steps (decode + reuse flag): the carried /
            #     step-0 top-k for the later chain steps;
            #   - draft-extend with the seed-capture flag: the
            #     [bs, nv, topk] block the worker mines the root-row
            #     seed from.
            has_mtp_topk = False
            if (
                self.model_runner.is_draft_worker
                and not is_pp_output
                and (
                    (
                        forward_batch.forward_mode.is_decode()
                        and getattr(
                            forward_batch, "reuse_dsa_topk_indices", False
                        )
                    )
                    or (
                        forward_batch.forward_mode.is_draft_extend()
                        and getattr(
                            getattr(
                                forward_batch, "spec_info", None
                            ),
                            "dsa_seed_topk_capture",
                            None,
                        )
                        is not None
                    )
                )
            ):
                published = getattr(forward_batch.spec_info, "dsa_topk_indices", None)
                if published is not None and published is not _pre_topk_carry:
                    graph_outputs.append(published)
                    has_mtp_topk = True

            if KunpengGraphRunner._graph_shm_tensor is None:
                remaining = torch.ops.sgl_kernel.shm_remaining_bytes_kunpeng()
                backing = (
                    self.weight_hbw_pool.backing_snapshot()
                    if self.weight_hbw_pool
                    else {}
                )
                if backing.get("enabled"):
                    self._shm_prefix_before_graph_bytes = (
                        backing["shm_prefix_bytes"] - remaining
                    )
                if remaining > 0:
                    KunpengGraphRunner._graph_shm_tensor = (
                        torch.ops.sgl_kernel.create_shm_tensor_kunpeng(
                            torch.uint8, [remaining]
                        )
                    )

            if use_hbw and _is_kunpeng_hbw_pool and _hbw_resident_graph:
                if KunpengGraphRunner._graph_hbw_tensor is None:
                    remaining = self.weight_hbw_pool.largest_free_bytes
                    if remaining < 4 * 1024 * 1024:
                        pool = self.weight_hbw_pool
                        raise RuntimeError(
                            f"HBW pool has only {remaining / 1024 / 1024:.1f} MB "
                            f"largest free block left for graph capture "
                            f"(min 4 MB). pool used={pool.used_bytes / 1024 / 1024:.1f} MB "
                            f"free={(pool.pool_size - pool.used_bytes) / 1024 / 1024:.1f} MB. "
                            f"Enlarge SGLANG_KUNPENG_WEIGHTS_HBW_POOL_SIZE_MB, or free "
                            f"HBW via SGLANG_KUNPENG_HBW_RESIDENT_NON_EXPERT_WEIGHTS=0 / "
                            f"SGLANG_KUNPENG_HBW_RESIDENT_EXPERT_WEIGHTS=0, or disable graph "
                            f"HBW with SGLANG_KUNPENG_HBW_RESIDENT_GRAPH=0."
                        )
                    raw = self.weight_hbw_pool.alloc(
                        (remaining,), torch.uint8, label="graph_capture"
                    )
                    self._graph_hbw_raw_bytes = raw.numel()
                    # Round the pool base up to 2M so that per-storage offset
                    # alignment in pack_intervals (2M/512K/1K workspace bufs)
                    # equals absolute-address alignment.
                    _align = 2 * 1024 * 1024
                    off = (_align - (raw.data_ptr() % _align)) % _align
                    KunpengGraphRunner._graph_hbw_tensor = raw[off:]
                    logger.info(
                        "[HBW graph] claimed %d MB for graph capture buffers",
                        remaining // (1024 * 1024),
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

            if _graph_profile_active:
                graph.enable_profile(True)
            has_pp_topk = bool(is_pp_output and pp_topk is not None)
            self._sglang_graph_cache[graph_cache_key] = (
                graph,
                is_pp_graph,
                is_pp_output,
                has_pp_topk,
                has_mtp_topk,
            )
            if self._graph_bindings is not None:
                self._graph_capture_count += 1
            if prefill_coverage is not None:
                prefill_coverage["captures"] += 1
            if decode_coverage is not None:
                decode_coverage["captures"] += 1
            logger.info(
                f"[graph] captured mode={forward_batch.forward_mode.name}, "
                f"total_tokens={total_tokens}, batch_size={forward_batch.batch_size}, "
                f"pp={is_pp_graph}, mtp_topk={has_mtp_topk}"
            )
        else:
            if prefill_coverage is not None:
                prefill_coverage["cache_hits"] += 1
            self._sglang_graph_cache.move_to_end(graph_cache_key)
            (
                graph,
                is_pp_graph,
                is_pp_output,
                has_pp_topk,
                has_mtp_topk,
            ) = self._sglang_graph_cache[graph_cache_key]

        t0 = time.time()
        with pp_span("graph_run"), self._forward_hbw_scope():
            outputs = graph.run(inputs)
        if self._graph_bindings is not None:
            self._graph_replay_count += 1
        if prefill_coverage is not None:
            prefill_coverage["replays"] += 1
        if decode_coverage is not None:
            decode_coverage["replays"] += 1
        t1 = time.time()
        # With SGLANG_SCHEDULER_SKIP_ALL_GATHER the scheduler issues an idle
        # forward every iteration even when no requests arrive; skip logging
        # those replays so an idle server does not flood the log file.
        skip_idle_log = (
            _is_scheduler_skip_all_gather and forward_batch.forward_mode.is_idle()
        )
        global last_call_timestamp
        if (
            not skip_idle_log
            and os.environ.get("SGLANG_KUNPENG_GRAPH_LOG_REPLAY", "1") != "0"
            and os.environ.get("SGLANG_KUNPENG_PP_PROFILE", "0") == "0"
        ):
            logger.info(
                f"[graph] run {1000 * (t1 - t0):.3f} ms, others {1000 * (t0 - last_call_timestamp):.3f} ms."
            )

        # Idle replays would grow the profile records unboundedly.
        if _graph_profile_active and not skip_idle_log:
            row = graph.get_profile_row()
            op_names = graph.profile_op_names()
            record = (
                self._graph_profile_path(),
                row,
                op_names,
                {
                    "forward_mode": forward_mode,
                    "count": len(op_names),
                    "total_tokens": total_tokens,
                    "batch_size": forward_batch.batch_size,
                },
            )
            with _graph_profile_lock:
                # Re-check: the session may have been stopped meanwhile.
                if _graph_profile_active:
                    _graph_profile_records.append(record)
        last_call_timestamp = time.time()

        if is_pp_output:
            if has_pp_topk:
                hidden_states, residual, topk_indices = outputs
                return PPProxyTensors(
                    {
                        "hidden_states": hidden_states,
                        "residual": residual,
                        "topk_indices": topk_indices,
                    }
                )
            hidden_states, residual = outputs
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        # index_share_for_mtp_iteration: return the fresh top-k (the
        # chain's carried/step top-k, or the draft-extend seed block)
        # through spec_info. CLONE it out of the graph pool: per-step
        # draft graphs share the pool, so later replays would overwrite
        # the buffer in place (same reason the chain clones
        # hidden_states).
        if has_mtp_topk:
            if graph.has_hidden_states:
                logits, hidden_states, mtp_topk = outputs
            else:
                logits, mtp_topk = outputs
                hidden_states = None
            if forward_batch.spec_info is not None:
                forward_batch.spec_info.dsa_topk_indices = mtp_topk.clone()
            return logits, hidden_states

        if graph.has_hidden_states:
            logits, hidden_states = outputs
            return logits, hidden_states
        (logits,) = outputs
        return logits, None

    def _graph_profile_path(self) -> str:
        # With PP>1, stages sharing a tp_rank would concurrently append
        # to the same file; split per pp_rank. The MTP draft runner is
        # constructed with pp_size=1/pp_rank=0 but only ever runs on the
        # last PP rank, so merge its records into that stage's file.
        if self.model_runner.is_draft_worker:
            pp_rank = self.model_runner.server_args.pp_size - 1
        else:
            pp_rank = self.model_runner.pp_rank
        rank_part = (
            f"pp{pp_rank}"
            f"_dp{self.model_runner.dp_rank}"
            f"_tp{self.model_runner.tp_rank}"
        )
        return os.path.join(_graph_profile_dir, f"sglang_graph_{rank_part}.jsonl")


def kunpeng_graph_profile_active() -> bool:
    return _graph_profile_active


def kunpeng_graph_profile_start(output_dir: Optional[str] = None) -> bool:
    """Start collecting one profile row per graph replay (like the torch
    profiler's /start_profile). Records are buffered in memory and flushed
    by kunpeng_graph_profile_stop(). Returns False if a session is already
    active.
    """
    global _graph_profile_active, _graph_profile_dir, _graph_profile_records
    with _graph_profile_lock:
        if _graph_profile_active:
            return False
        _graph_profile_records = []
        _graph_profile_dir = output_dir or os.environ.get(
            "SGLANG_TORCH_PROFILER_DIR", "/tmp"
        )
        _graph_profile_active = True
    # Graphs captured before this session need in-graph timestamps enabled;
    # graphs captured during the session are handled at capture time.
    for runner in list(_graph_profile_runners):
        for entry in runner._sglang_graph_cache.values():
            entry[0].enable_profile(True)
    logger.info(
        "[graph] profile session started, records will be flushed to %s "
        "on stop_profile",
        _graph_profile_dir,
    )
    return True


def kunpeng_graph_profile_stop() -> bool:
    """Stop collecting and flush the buffered records into the per-rank
    jsonl files. Returns False if no session is active.
    """
    global _graph_profile_active, _graph_profile_records
    with _graph_profile_lock:
        if not _graph_profile_active:
            return False
        _graph_profile_active = False
        records = _graph_profile_records
        _graph_profile_records = []
    for runner in list(_graph_profile_runners):
        for entry in runner._sglang_graph_cache.values():
            entry[0].enable_profile(False)
    by_path: dict = {}
    for path, row, op_names, meta in records:
        by_path.setdefault(path, []).append((row, op_names, meta))
    if by_path:
        from sglang.srt.graph.profile import write_profiles

        for path, rows in by_path.items():
            write_profiles(path, rows)
    logger.info(
        "[graph] profile session stopped, flushed %d record(s) into %d file(s)",
        len(records),
        len(by_path),
    )
    return True
