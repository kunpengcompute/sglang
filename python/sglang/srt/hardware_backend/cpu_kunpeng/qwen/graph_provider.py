"""Opt-in Qwen native primitive bindings for the common Kunpeng graph runner.

No model graph is authored here. Model.forward supplies the sequence through
automatic ATen capture and explicit adapters for existing native primitives.
"""

import contextlib
import os
import torch
from sglang.srt.graph import ops
from . import image_decode_graph_enabled, image_prefill_graph_enabled, prefill_graph
from .capabilities import get_capabilities
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
        # Retired: this legacy mechanism decomposed the sgl_kernel::bf16_linear_kunpeng
        # op (emitted by the retired _kunpeng_qwen_dense packing path, which also set
        # _kunpeng_bf16_weight_packed) into pack+GEMM graph ops under a provider-planned
        # tiling/workspace. The upstream rebuild replaced that execution path with the
        # _cpu_projection_role hooks (configure_modules -> prepare_dense/apply_dense):
        # apply_dense calls the registered packed-GEMM GraphOps directly, so the
        # projections self-register during capture and neither the marker nor the
        # intercepted op exists anymore. Keeping this True aborts bind with
        # "No packed dense weights for decomposition".
        self.decompose_dense = False
        self.decompose_clone = True
        self.dense_plans = {}
        self.dense_context = None
        # One-shot guard for the vision SHM width reservations in bind().
        self._vision_shm_reserved = False

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
        tokens = fb.input_ids.numel()
        if fb.forward_mode.is_target_verify():
            # Verify rows are batch_size * draft_token_num; `rowwise_gqa`
            # chunks them, so only the declared batch and total row capacity
            # bound the graph.
            requests = get_capabilities().decode_max_requests
            if not 1 <= fb.batch_size <= requests:
                return f"batch_outside_1_{requests}"
            if not prefill_graph.tokens_supported(tokens):
                return "verify_tokens_outside_prefill_limit"
        else:
            reason = get_capabilities().decode_unsupported_reason(
                fb.batch_size, tokens
            )
            if reason is not None:
                return reason
        if fb.contains_image_inputs() and not (
            # Decode batches carry mm_inputs until
            # the request finishes but never consume them; decode M-RoPE
            # rows are equal (see qwen/__init__.image_decode_graph_enabled),
            # so replay-safe with text decode graphs. Target-verify stays
            # excluded, and image PREFILL still falls to eager.
            fb.forward_mode.is_decode()
            and image_decode_graph_enabled()
        ):
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

    def _embed_mm_inputs_for_graph(self, runner, fb, kwargs):
        """Eager vision tower + multimodal merge for image-prefill graphs.

        Mirrors the embed stage general_mm_embed_routine runs inside the
        eager forward (the same embed_mm_inputs call with the same argument
        assembly), then mirrors its post-merge state (mm_inputs consumed,
        mm_input_embeds published) and hands the merged tensors to the
        model forward via kwargs. The model's forward override detects
        input_embeds and runs the language tower + logits only, which is
        the capturable body; every replay re-runs this pre-stage because
        vision features differ per request.
        """
        from sglang.srt.managers.mm_utils import embed_mm_inputs

        model = runner.model
        mm_inputs_list = [
            mm_input for mm_input in fb.mm_inputs if mm_input is not None
        ]
        extend_prefix_lens = [
            prefix_len
            for i, prefix_len in enumerate(fb.extend_prefix_lens_cpu)
            if fb.mm_inputs[i] is not None
        ]
        extend_seq_lens = [
            seq_len
            for i, seq_len in enumerate(fb.extend_seq_lens_cpu)
            if fb.mm_inputs[i] is not None
        ]
        use_deepstack = getattr(model, "use_deepstack", None)
        if use_deepstack is None:
            # qwen3_vl publishes {IMAGE: True, VIDEO: True}; derive the same
            # policy from the tower's deepstack indexes if the wrapper does
            # not expose the dict.
            if getattr(model, "deepstack_visual_indexes", None):
                from sglang.srt.managers.schedule_batch import Modality

                use_deepstack = {Modality.IMAGE: True, Modality.VIDEO: True}
            else:
                use_deepstack = {}
        input_embeds, other_info = embed_mm_inputs(
            mm_inputs_list=mm_inputs_list,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            input_ids=fb.input_ids,
            input_embedding=model.get_input_embeddings(),
            multimodal_model=model,
            use_deepstack=use_deepstack,
        )
        fb.mm_inputs = None
        fb.mm_input_embeds = input_embeds
        kwargs["input_embeds"] = input_embeds
        deepstack = other_info.get("input_deepstack_embeds")
        if deepstack is not None:
            kwargs["input_deepstack_embeds"] = deepstack

    def bind(self, runner, fb, kwargs):
        from sglang.srt.graph.collect_weights import collect_model_weights

        # Sampling returns int32 IDs; the embedding graph uses int64 for both
        # greedy and sampled decode steps. Normalize before binding inputs.
        fb.input_ids = fb.input_ids.to(torch.int64)
        if (
            fb.forward_mode.name == "EXTEND"
            and fb.contains_image_inputs()
            and image_prefill_graph_enabled()
        ):
            # War B: the image gate released this batch; run the vision
            # tower + multimodal merge eagerly here (bind runs outside any
            # capture context), exactly as the eager forward would. The
            # merged tensors are bound as dynamic inputs below and handed
            # to the model forward via kwargs, so the captured body is the
            # language tower + logits only.
            self._embed_mm_inputs_for_graph(runner, fb, kwargs)
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
        if kwargs.get("input_embeds") is not None:
            # War B merged multimodal embeddings (see
            # _embed_mm_inputs_for_graph): bound like any other dynamic
            # input so every replay refreshes them, and their presence in
            # the aliases list splits the graph cache signature from
            # text-prefill graphs of the same (total_tokens, batch_size).
            dynamic("input_embeds", kwargs["input_embeds"])
        if kwargs.get("input_deepstack_embeds") is not None:
            dynamic("input_deepstack_embeds", kwargs["input_deepstack_embeds"])
        ragged = getattr(fb, "_draft_extend_ragged", None)
        if ragged is not None:
            # Data-dependent row mapping of the ragged draft-extend; rebuilt
            # per forward and replayed as an ordinary graph input.
            dynamic("draft_extend.requests", ragged[0])
            dynamic("draft_extend.lengths", ragged[1])
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
                # Verify row metadata is built once in prepare_for_verify and
                # shared by every layer, so it must be bound like any other
                # dynamic input instead of being rebuilt per layer in-graph.
                dynamic(
                    "attn.verify_requests",
                    getattr(fb.spec_info, "_kunpeng_verify_requests", None),
                )
                dynamic(
                    "attn.verify_lengths",
                    getattr(fb.spec_info, "_kunpeng_verify_lengths", None),
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
        # The explicit Qwen collectives read graph-owned SHM, so the legacy
        # communicator buckets are optional here: bind the ones that exist, but
        # never allocate. By now the already captured graphs have claimed the
        # remaining pool, so a fresh create would fail with "remain 0".
        from sglang.srt.distributed import get_tp_group

        communicator = get_tp_group().kunpeng_communicator
        # Vision tower TP reductions stage through the communicator's
        # dim-keyed SHM cache, but their widths only first appear at the
        # initial image request - long after the text graphs claimed the
        # pool, which is exactly the exhaustion that used to crash image
        # batches ("remain 0"). The upstream rebuild dropped the legacy
        # embedding/lm-head eager reservations here (the explicit
        # collectives read graph-owned SHM now), but the vision path still
        # stages through the dim-keyed cache: reserve both row buckets for
        # every vision reduce width once, at the first bind, and publish
        # the reserved capacity so RowParallelLinear's vision branch can
        # fall back to gloo beyond it instead of asking the pool to grow
        # an entry post-capture.
        if not self._vision_shm_reserved:
            self._vision_shm_reserved = True
            visual = getattr(runner.model, "visual", None)
            if visual is not None:
                from sglang.srt.layers.linear import (
                    RowParallelLinear,
                    _vision_shm_rows as vision_registry,
                )

                vision_widths = sorted(
                    {
                        int(module.output_size)
                        for module in visual.modules()
                        if isinstance(module, RowParallelLinear)
                        and getattr(module, "_runtime_component", None) == "vision"
                    }
                )
                if vision_widths:
                    vision_budget = min(
                        int(
                            os.environ.get(
                                "SGLANG_KUNPENG_VISION_SHM_MAX_ROWS", "4096"
                            )
                        ),
                        int(communicator.max_tokens),
                    )
                    reserve = torch.ops.sgl_kernel.get_or_create_shm_tensor_kunpeng
                    for vision_width in vision_widths:
                        # The kernel's cache keys by dim: the small call pins
                        # the [batch_size, dim] entry, the large call pins a
                        # bucket-rounded [budget, dim] entry. Row counts up to
                        # the returned capacity then always hit a reservation.
                        small = reserve(vision_width, 1)
                        large = reserve(vision_width, vision_budget)
                        capacity = max(int(small.size(0)), int(large.size(0)))
                        vision_registry[vision_width] = capacity
                    print(
                        f"[QwenRegisteredGraph] vision_shm_rows={vision_registry} "
                        f"budget={vision_budget}",
                        flush=True,
                    )
        for shm_width, tensor in sorted(communicator.shm_tensors.items()):
            persistent(f"collective.shm.{shm_width}", tensor)
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

        # The BF16 projection hooks (qwen/linear.py apply_dense, wired via
        # unquant.py's _cpu_linear_apply) query the host-side tiling planner
        # and then parameterize the registered packed-GEMM graph ops with the
        # returned tile sizes. The planner is a scalar-only custom op (int
        # M/N/K -> int tiles, no tensors), and the capture mode intercepts
        # every op entering torch dispatch - scalar-only calls included - so
        # capture aborts with "Native op needs a registered graph adapter".
        # Decompose it to eager execution: the callback already runs under
        # _DisableTorchDispatch (torch_capture.py), the plan depends only on
        # the static GEMM shape (fixed per captured graph), and the explicit
        # runtime calls the same planner outside any dispatch mode while
        # capturing.
        def tiling_plan(M, N, K):
            return torch.ops.sgl_kernel.bgemm_find_optimal_tiling_plan(M, N, K)

        decompositions[("sgl_kernel::bgemm_find_optimal_tiling_plan", "")] = tiling_plan
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


def draft_extend_pad_enabled() -> bool:
    """Whether the ragged draft-extend batch is padded to a canonical row count.

    Padding only stabilizes the graph-cache key; the row-wise attention chunks
    rows, so padding off still captures every draft-extend shape.
    """
    return os.environ.get("SGLANG_KUNPENG_QWEN_DRAFT_EXTEND_PAD", "1") == "1"


def draft_extend_max_rows() -> int:
    """Accepted tokens per request cannot exceed the drafted token count."""
    from sglang.srt.server_args import get_global_server_args

    return max(int(get_global_server_args().speculative_num_draft_tokens or 1), 1)


_PAD_BUFFERS = {}


def _pad_into(t, pad, name, dim=0):
    """Append `pad` copies of the last slice along `dim` into a persistent
    staging buffer, and return the buffer (exactly the target length).

    The graph key carries the binding signature of every bound tensor, so
    rebuilding these tensors each step would change the key and force a fresh
    capture per step; the staging buffers are kept per (name, shape, dim) and
    refilled in place.
    """
    shape = list(t.shape)
    shape[dim] += pad
    key = (name, dim, tuple(shape), t.dtype, t.device)
    buf = _PAD_BUFFERS.get(key)
    if buf is None:
        buf = torch.empty(shape, dtype=t.dtype, device=t.device)
        _PAD_BUFFERS[key] = buf
    keep = [slice(None)] * t.dim()
    keep[dim] = slice(0, t.shape[dim])
    buf[tuple(keep)].copy_(t)
    tail = [slice(None)] * t.dim()
    tail[dim] = slice(t.shape[dim], shape[dim])
    last = [slice(None)] * t.dim()
    last[dim] = slice(t.shape[dim] - 1, t.shape[dim])
    expand = list(t.shape)
    expand[dim] = pad
    buf[tuple(tail)].copy_(t[tuple(last)].expand(expand))
    return buf


def pad_draft_extend_batch(fb) -> None:
    """Canonicalize the ragged draft-extend batch to `batch_size * max_rows`.

    Surplus rows duplicate the last row. Because input_ids, positions and
    out_cache_loc are copied as well, the native attention recomputes the same
    k/v and rewrites the same KV slot, so the extra rows are idempotent, and
    their outputs are never selected by the per-request last-token gather.
    Without this the row count is data dependent, so every step captures a new
    graph.

    Every row-count tensor must be normalized, not just input_ids: the graph
    key includes each binding's shape, so an unnormalized `out_cache_loc`
    (allocated before finished requests are filtered, hence longer than the
    surviving tokens) or `mrope_positions` (built from spec_info.positions)
    would reintroduce one capture per distinct length.
    """
    from .backend import draft_extend_ragged

    max_rows = draft_extend_max_rows()
    rows = fb.batch_size * max_rows
    tokens = fb.input_ids.numel()
    requests, lengths = draft_extend_ragged(fb)
    loc = fb.out_cache_loc
    if loc is not None and loc.numel() > tokens:
        # Only the leading `tokens` slots belong to surviving rows (same
        # prefix assumption as the eager draft-extend KV write).
        loc = loc[:tokens]
    pad = rows - tokens
    if pad > 0:
        fb.input_ids = _pad_into(fb.input_ids, pad, "input_ids")
        fb.positions = _pad_into(fb.positions, pad, "positions")
        if loc is not None:
            loc = _pad_into(loc, pad, "out_cache_loc")
        requests = _pad_into(requests, pad, "requests")
        lengths = _pad_into(lengths, pad, "lengths")
        hidden = getattr(fb.spec_info, "hidden_states", None)
        if hidden is not None:
            fb.spec_info.hidden_states = _pad_into(hidden, pad, "hidden_states")
    fb.out_cache_loc = loc
    fb._draft_extend_ragged = (requests, lengths)
    mrope = getattr(fb, "mrope_positions", None)
    if mrope is not None:
        if mrope.shape[1] > tokens:
            mrope = mrope[:, :tokens]
        extra = rows - mrope.shape[1]
        if extra > 0:
            mrope = _pad_into(mrope, extra, "mrope_positions", dim=1)
        fb.mrope_positions = mrope


class QwenAttachedMTPGraphProvider(QwenHybridGraphProvider):
    """Capture provider for the attached one-layer MTP draft.

    Unlike the target provider this only declares the chained draft *decode*
    steps (one token per sequence, per-step attention backend) plus the
    draft-extend pass that seeds them, and the prefill-extend that seeds the
    chain for a freshly prefilled batch. Draft-extend is ragged -- each request
    contributes 1..N accepted rows -- so its row mapping is built eagerly and
    bound as an ordinary graph input. Prefill-extend is a plain EXTEND batch and
    reuses the segmented prefill attention.

    Like the provider-less Kunpeng backends, the draft declares no shape
    admission: row counts are absorbed by ``rowwise_gqa`` and the segmented
    prefill kernel, so every draft forward is captured. Only a missing chain
    input (the previous hidden states) or a ragged length mapping that cannot
    be built makes a batch unboundable; those are input-contract violations,
    not a batch-size policy.
    """

    def bind(self, runner, fb, kwargs):
        if self.decompose_dense and not any(
            getattr(module, "_kunpeng_bf16_weight_packed", False)
            for module in runner.model.modules()
        ):
            # The borrowed MTP component keeps its dense projections on the
            # plain F.linear path (never prepacked), so there is nothing to
            # decompose. ATen capture records those linear/matmul ops
            # directly instead.
            self.decompose_dense = False
            print(
                "[QwenAttachedMTPGraph] dense decomposition disabled: "
                "no packed dense weights in the draft component",
                flush=True,
            )
        if fb.forward_mode.is_draft_extend():
            if draft_extend_pad_enabled():
                pad_draft_extend_batch(fb)
            else:
                from .backend import draft_extend_ragged

                fb._draft_extend_ragged = draft_extend_ragged(fb)
        return super().bind(runner, fb, kwargs)

    def unsupported_reason(self, fb, kwargs):
        if fb.forward_mode.is_draft_extend():
            return self._draft_extend_unsupported_reason(fb, kwargs)
        if fb.forward_mode.name == "EXTEND":
            return self._draft_prefill_extend_unsupported_reason(fb, kwargs)
        if not fb.forward_mode.is_decode():
            return "forward_mode"
        return self._draft_unsupported_reason(fb, kwargs)

    def _draft_unsupported_reason(self, fb, kwargs):
        # The chained decode step must carry the previous step's hidden states;
        # nothing else can make it unboundable. Batch/token rows are covered by
        # the declared capacity and the native row kernels, so they are not an
        # admission gate (an over-capacity shape fails in the kernel instead of
        # silently falling back to eager, which cannot allocate SHM after the
        # first capture).
        spec_info = fb.spec_info
        if spec_info is None or getattr(spec_info, "hidden_states", None) is None:
            return "draft_hidden_states_missing"
        return None

    def _draft_prefill_extend_unsupported_reason(self, fb, kwargs):
        """The attached MTP prefill-extend is an ordinary EXTEND batch.

        It inherits EXTEND (never DRAFT_EXTEND) and uses spec_info only to carry
        the target's full hidden states, so the segmented prefill attention that
        covers the target prefill covers it as well. Prefill limits are enforced
        by ``prefill_graph`` itself, exactly as for the target prefill.
        """
        tokens = fb.input_ids.numel()
        spec_info = fb.spec_info
        hidden = getattr(spec_info, "hidden_states", None) if spec_info else None
        if hidden is None or hidden.shape[0] != tokens:
            return "draft_prefill_extend_hidden_states_missing"
        return None

    def _draft_extend_unsupported_reason(self, fb, kwargs):
        tokens = fb.input_ids.numel()
        counts = fb.extend_seq_lens
        # The row mapping is data dependent and is built eagerly for the graph
        # input; padding only stabilizes the graph-cache key. Row counts are not
        # an admission gate because `rowwise_gqa` chunks them.
        if counts is None or counts.numel() != fb.batch_size:
            return "draft_extend_lengths_missing"
        if int(counts.sum()) != tokens or int(counts.min()) < 1:
            return "draft_extend_lengths_mismatch"
        # out_cache_loc may be allocated before finished requests are filtered,
        # so it can exceed the surviving token count; the attention path clips
        # it to the leading `tokens` slots.
        loc = fb.out_cache_loc
        if loc is None or loc.numel() < tokens:
            return "draft_extend_cache_loc_mismatch"
        spec_info = fb.spec_info
        hidden = getattr(spec_info, "hidden_states", None) if spec_info else None
        if hidden is None or hidden.shape[0] != tokens:
            return "draft_extend_hidden_states_missing"
        return None
