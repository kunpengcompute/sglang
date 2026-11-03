"""Opt-in Qwen native primitive bindings for the common Kunpeng graph runner.

No model graph is authored here. Model.forward supplies the sequence through
automatic ATen capture and explicit adapters for existing native primitives.
"""

import contextlib
import os
import torch
from sglang.srt.graph import ops
from . import prefill_graph
from sglang.srt.graph.bindings import (
    GraphBindings,
    StaticTensorBindings,
    tensor_signature,
)
from sglang.srt.graph.torch_capture import (
    AutomaticTorchCapture,
    contiguous_clone,
)


class QwenHybridGraphProvider:
    # Explicit capabilities: the common runner's default capture contract is unchanged.
    capture_options = {
        "allow_input_aliases": True,
        "suppress_adapter_dispatch": True,
    }

    def __init__(self, runner):
        self.backend = runner.attn_backend
        self.weights = None
        self.static_weights = None
        self.use_static_weights = True
        self.context = None
        self.decompose_dense = True
        self.decompose_clone = True
        self.dense_plans = {}
        self.dense_context = None
        self.eager_collective_buffers = None

    def unsupported_reason(self, fb, kwargs):
        if fb.forward_mode.name == "EXTEND":
            reason = prefill_graph.unsupported_reason(fb)
            if reason is not None:
                return reason
            if kwargs.get("pp_proxy_tensors") is not None:
                return "pp_proxy_tensors"
            if kwargs.get("get_embedding", False):
                return "get_embedding"
            return None
        if not (fb.forward_mode.is_decode() or fb.forward_mode.is_target_verify()):
            return "forward_mode"
        if not 1 <= fb.batch_size <= 32:
            return "batch_outside_1_32"
        tokens = fb.input_ids.numel()
        if not 1 <= tokens <= 32:
            return "tokens_outside_1_32"
        if not fb.forward_mode.is_target_verify() and tokens != fb.batch_size:
            return "decode_token_batch_mismatch"
        if fb.contains_image_inputs():
            return "image_inputs"
        if fb.mamba_track_mask is not None:
            return "mamba_state_tracking"
        if fb.return_logprob:
            return "return_logprob"
        if kwargs.get("pp_proxy_tensors") is not None:
            return "pp_proxy_tensors"
        if kwargs.get("get_embedding", False):
            return "get_embedding"
        return None

    def supports(self, fb, kwargs):
        return self.unsupported_reason(fb, kwargs) is None

    def bind(self, runner, fb, kwargs):
        from sglang.srt.graph.collect_weights import collect_model_weights

        # Sampling returns int32 IDs; the embedding graph uses int64 for both
        # greedy and sampled decode steps. Normalize before binding inputs.
        fb.input_ids = fb.input_ids.to(torch.int64)
        if self.weights is None:
            for module in runner.model.modules():
                rotary = getattr(module, "rotary_emb", None)
                if rotary is not None and hasattr(module, "q_norm"):
                    # The fused row-wise C ABI reads BF16; preserve the original
                    # cache dtype used by the ordinary CPU M-RoPE path.
                    from .backend import native_rope_cache

                    native_rope_cache(module)
            self.weights = collect_model_weights(runner.model)
            if self.use_static_weights:
                self.static_weights = StaticTensorBindings.create(
                    {f"weight.{i}": tensor for i, tensor in enumerate(self.weights)}
                )
                print(
                    f"[QwenRegisteredGraph] static_weight_bindings={len(self.weights)}",
                    flush=True,
                )
        inputs = {}
        seen = set()
        aliases = []

        def dynamic(name, tensor):
            if tensor is None:
                return
            base = tensor.untyped_storage().data_ptr()
            aliases.append((name, tensor_signature(tensor)))
            # Metadata aliases are resolved by the common input-view registry.
            if base not in seen:
                inputs[name] = tensor
                seen.add(base)

        for name in (
            "input_ids",
            "positions",
            "mrope_positions",
            "req_pool_indices",
            "seq_lens",
            "out_cache_loc",
            "num_token_non_padded",
            "extend_seq_lens",
            "extend_prefix_lens",
            "extend_start_loc",
        ):
            dynamic(name, getattr(fb, name, None))
        if fb.spec_info is not None:
            dynamic("spec.hidden_states", getattr(fb.spec_info, "hidden_states", None))
        linear = getattr(self.backend, "linear_attn_backend", None)
        if linear is not None:
            meta = linear.forward_metadata
            for name in ("mamba_cache_indices", "query_start_loc"):
                dynamic("linear." + name, getattr(meta, name))
            if fb.forward_mode.is_target_verify():
                dynamic(
                    "linear.checkpoint_indices",
                    linear.verify_intermediate_state_indices,
                )
        fixed = {}
        fixed_seen = (
            set(self.static_weights.storage_bases) if self.static_weights else set()
        )

        def persistent(name, tensor):
            if tensor is None:
                return
            base = tensor.untyped_storage().data_ptr()
            if base and base not in fixed_seen and base not in seen:
                fixed[name] = tensor
                fixed_seen.add(base)

        if self.static_weights is None:
            for i, tensor in enumerate(self.weights):
                persistent(f"weight.{i}", tensor)
        if self.decompose_dense:
            batch = fb.input_ids.numel()
            if batch not in self.dense_plans:
                shapes = {
                    tuple(module.weight.shape)
                    for module in runner.model.modules()
                    if getattr(module, "_kunpeng_bf16_weight_packed", False)
                }
                assert shapes, "No packed dense weights for decomposition"
                tilings = {
                    shape: tuple(
                        map(
                            int,
                            torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(
                                batch, *shape
                            ),
                        )
                    )
                    for shape in shapes
                }
                # Prefill's K1 native dense path does not access workspace.
                # A nonempty sentinel still gives GraphOp a valid Storage.
                # Keep the established decode allocation contract unchanged.
                size = max(
                    (
                        1
                        if fb.forward_mode.name == "EXTEND" and k == tilings[n, k][2]
                        else (k // tilings[n, k][2]) * n * batch + 1024
                    )
                    for n, k in shapes
                )
                from sglang.srt.hardware_backend.cpu_kunpeng.allocator.forward_hbw import (
                    current_arena,
                )

                arena = current_arena()
                with arena.forward() if arena is not None else contextlib.nullcontext():
                    workspace = torch.empty(size, dtype=torch.bfloat16)
                self.dense_plans[batch] = (tilings, workspace)
            tilings, workspace = self.dense_plans[batch]
            self.dense_context = (batch, tilings, workspace)
            persistent("dense.workspace", workspace)
        pool = getattr(runner.token_to_kv_pool, "full_kv_pool", runner.token_to_kv_pool)
        for name in ("k_buffer", "v_buffer"):
            for i, tensor in enumerate(getattr(pool, name)):
                persistent(f"{name}.{i}", tensor)
        if linear is not None:
            cache = runner.req_to_token_pool.mamba_pool.mamba_cache
            for i, tensor in enumerate(cache.conv):
                persistent(f"conv.{i}", tensor)
            persistent("ssm", cache.temporal)
            if fb.forward_mode.is_target_verify():
                persistent("checkpoint.ssm", cache.intermediate_ssm)
                for i, tensor in enumerate(cache.intermediate_conv_window):
                    persistent(f"checkpoint.conv.{i}", tensor)
        if fb.forward_mode.name == "EXTEND":
            prefill_graph.register()
            for name, tensor in zip(
                ("q", "packed_k", "packed_v", "out", "scratch"),
                prefill_graph.resources(),
            ):
                persistent("prefill.shared." + name, tensor)
        persistent("request_table", runner.req_to_token_pool.req_to_token)
        persistent("logits_buffer", fb.next_token_logits_buffer)
        # Standard TP logits gathering stages through communicator-owned SHM.
        # Allocate its known width before capture/pool finalization and bind
        # the complete backing buffers, including the dual-gather barrier.
        from sglang.srt.distributed import get_tp_group

        communicator = get_tp_group().kunpeng_communicator
        if self.eager_collective_buffers is None:
            # Embedding all-reduce uses the native eager cache between decode
            # graphs. Reserve both row buckets before the graph claims SHM.
            width = runner.model_config.hidden_size
            reserve = torch.ops.sgl_kernel.get_or_create_shm_tensor_kunpeng
            self.eager_collective_buffers = (
                reserve(width, 1),
                reserve(width, communicator.max_tokens),
            )
        communicator.get_shm_tensor(runner.model.lm_head.weight.shape[0])
        for width, tensor in sorted(communicator.shm_tensors.items()):
            persistent(f"collective.shm.{width}", tensor)
        persistent("collective.dummy", communicator.dummy_tensor)
        return GraphBindings(
            inputs,
            fixed,
            (
                "qwen-hybrid",
                fb.capture_hidden_mode,
                getattr(fb.spec_info, "draft_token_num", None),
                1,
                16,
                32,
                "bf16-native-canonical-ar",
                self.decompose_dense,
                self.use_static_weights,
                self.decompose_clone,
                "1",
                tuple(aliases),
            ),
            static=(self.static_weights,) if self.static_weights else (),
        )

    @contextlib.contextmanager
    def capture_context(self):
        from . import ops as primitive_registry

        rows_before = primitive_registry.rows_capture_count
        decompositions = {}
        if self.decompose_clone:
            decompositions[("aten::clone", "")] = contiguous_clone
        if self.decompose_dense:
            batch, tilings, workspace = self.dense_context

            def dense(x, weight, bias=None):
                assert x.ndim == 2 and x.shape[0] == batch and x.is_contiguous()
                assert x.dtype == weight.dtype == torch.bfloat16 and bias is None
                tile_m, _, tile_k = tilings[tuple(weight.shape)]
                packed = ops.bf16_gemm_pack_kunpeng(x, tile_m, tile_k)
                return ops.bf16_packed_gemm_kunpeng(packed, weight, workspace, 32)

            decompositions[("sgl_kernel::bf16_linear_kunpeng", "")] = dense
        self.context = AutomaticTorchCapture(decompositions)
        with self.context:
            yield self.context
        rows_count = primitive_registry.rows_capture_count - rows_before
        if rows_count:
            print(f"[QwenRegisteredGraph] gqa_rows_nodes={rows_count}", flush=True)
        if self.decompose_dense:
            count = self.context.decomposition_counts.get(
                ("sgl_kernel::bf16_linear_kunpeng", ""), 0
            )
            assert count > 0, "Dense decomposition requested but never captured"
            print(
                f"[QwenRegisteredGraph] dense_decompositions={count} batch={batch} "
                f"workspace_bytes={workspace.numel()*workspace.element_size()}",
                flush=True,
            )
        if self.decompose_clone:
            count = self.context.decomposition_counts.get(("aten::clone", ""), 0)
            print(
                f"[QwenRegisteredGraph] planned_contiguous_clones={count}", flush=True
            )
