"""Explicit Qwen CPU model and attention hooks using common registered operators."""

import ctypes
import os
import threading
from functools import lru_cache

import torch
from sgl_kernel import common_ops

from sglang.srt.graph import is_capturing, ops
from sglang.srt.graph.torch_capture import install_boxed_bridge, register_native_shape
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm, rms_norm_ref
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm

from . import ops as primitives
from .prefill import conv_prefill, gating


@lru_cache()
def initialize():
    library = primitives.register_primitives()
    primitives.register_rows()
    install_boxed_bridge(common_ops.__file__)
    register_native_shape(
        torch.ops.sgl_kernel.bf16_linear_kunpeng.default,
        lambda x, w, bias=None: torch.empty(
            (*x.shape[:-1], w.shape[0]), dtype=x.dtype, device="meta"
        ),
    )

    def split_shape(qkvz, ba, hk, hv, dk, dv):
        n = qkvz.shape[0]
        return tuple(
            torch.empty(shape, dtype=qkvz.dtype, device="meta")
            for shape in ((n, 2 * hk * dk + hv * dv), (n, hv, dv), (n, hv), (n, hv))
        )

    register_native_shape(
        torch.ops.sgl_kernel.fused_qkvzba_split_reshape_cat_contiguous_cpu.default,
        split_shape,
    )
    library.qwen36_gdn_prefill_with_workspace.restype = ctypes.c_int
    library.qwen36_gdn_prefill_with_workspace.argtypes = (
        [ctypes.c_void_p] * 8 + [ctypes.c_int64] * 6 + [ctypes.c_void_p, ctypes.c_int64]
    )
    library.qwen36_gdn_prefill_parallel_with_workspace.restype = ctypes.c_int
    library.qwen36_gdn_prefill_parallel_with_workspace.argtypes = (
        [ctypes.c_void_p] * 8 + [ctypes.c_int64] * 7 + [ctypes.c_void_p, ctypes.c_int64]
    )
    return library


def configure_modules(model):
    initialize()
    shapes = {
        (2176, 5120),
        (5120, 1088),
        (1024, 5120),
        (5120, 384),
        (1536, 5120),
        (5120, 512),
    }
    from sglang.srt.layers.quantization.modelslim.schemes.cpu920f_w8a8_dynamic import (
        CPU920FModelSlimW8A8Dynamic,
    )

    for module in model.modules():
        # Declare capability from the actual downstream projection. GDN's
        # BF16 BA projection shares its input norm and keeps BF16 output.
        mlp = getattr(module, "mlp", None)
        gate_up = getattr(mlp, "gate_up_proj", None)
        post_norm = getattr(module, "post_attention_layernorm", None)
        if post_norm is not None and isinstance(
            getattr(gate_up, "scheme", None), CPU920FModelSlimW8A8Dynamic
        ):
            post_norm._kunpeng_int8_consumer = True
        qkv = getattr(module, "qkv_proj", None)
        input_norm = getattr(module, "input_layernorm", None)
        if input_norm is not None and isinstance(
            getattr(qkv, "scheme", None), CPU920FModelSlimW8A8Dynamic
        ):
            input_norm._kunpeng_int8_consumer = True
        weight = getattr(module, "weight", None)
        if weight is not None and tuple(weight.shape) in shapes:
            module._kunpeng_qwen_dense = True


def prepare_model_lm_head(model, pool):
    # The architecture backend supplies a semantic role, not a shape whitelist.
    # Shared MTP references retain this same module and its logical weight.
    head = getattr(model, "lm_head", None)
    if head is None or not hasattr(head, "weight"):
        return False
    head._kunpeng_lm_head_role = True
    from sglang.srt.hardware_backend.cpu_kunpeng.bf16_lm_head import prepare_weight

    return prepare_weight(head, pool)


class QwenGemmaRMSNorm(GemmaRMSNorm):
    _kunpeng_qwen = True

    def forward_cpu(
        self, x, residual=None, post_residual_addition=None, quantize=False
    ):
        emit_quantized = quantize or getattr(self, "_kunpeng_int8_consumer", False)
        if emit_quantized:
            # The public DeepSeek norm-quant algebra measures the F32 weighted
            # max before normalizing, then quantizes from the BF16 residual.
            # It intentionally does not materialize a BF16 norm-output tensor.
            flat_x = x.reshape(-1, x.shape[-1])
            if residual is None:
                return ops.rmsnorm_quant_kunpeng(
                    flat_x, self.gemma_weight, self.variance_epsilon
                )
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            q, scale = ops.fused_add_rmsnorm_quant_kunpeng(
                flat_x,
                residual.reshape(-1, residual.shape[-1]),
                self.gemma_weight,
                self.variance_epsilon,
            )
            if quantize:
                # Match the common RMSNorm explicit quantize=True protocol.
                return q, scale, residual
            return (q, scale), residual
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            out = ops.fused_add_rmsnorm_kunpeng(
                x.reshape(-1, x.shape[-1]),
                residual.reshape(-1, residual.shape[-1]),
                self.gemma_weight,
                self.variance_epsilon,
            )
            return out.reshape_as(x), residual
        return ops.rmsnorm_kunpeng(
            x.reshape(-1, x.shape[-1]), self.gemma_weight, self.variance_epsilon
        ).reshape_as(x)


def _prefill_gated_rms_norm_capture_ref(x, weight, z, eps):
    """Selected rms_norm_ref contract, with explicit scalar overload schemas.

    Keep sqrt -> reciprocal -> multiply1, not rsqrt or fused native math.
    This caller fixes biasNone, groupNone, norm_before_gate=True, upcast=True.
    """
    dtype = x.dtype
    weight = weight.float()
    x = x.float()
    z = z.float()
    variance = x.square().mean(dim=-1, keepdim=True)
    variance = torch.ops.aten.add.Scalar(variance, eps)
    denominator = torch.sqrt(variance)
    rstd = torch.ops.aten.mul.Scalar(torch.reciprocal(denominator), 1)
    out = x * rstd * weight
    out *= torch.nn.functional.silu(z)
    return out.to(dtype)


class QwenGatedRMSNorm(RMSNorm):
    def forward(self, x, z=None):
        if (
            z is not None
            and x.ndim == 2
            and x.shape == z.shape
            and 1 <= x.shape[0] <= 96
        ):
            return ops.qwen_gated_norm(
                x.contiguous(), self.weight, z.contiguous(), self.eps
            )
        if (
            is_capturing()
            and os.environ.get("SGLANG_KUNPENG_ENABLE_PREFILL_GRAPH", "0") == "1"
            and z is not None
            and x.ndim == 2
            and x.shape == z.shape
            and 1 <= x.shape[0] <= 3 * 4096
            and x.shape[0] % 3 == 0
            and x.shape[1] == self.weight.numel() == 128
            and x.device.type == z.device.type == self.weight.device.type == "cpu"
            and x.dtype == z.dtype == torch.bfloat16
        ):
            # Local HV3 unfolds each legal C1 prefill M into 3*M rows of128.
            # Eager and the established <=96-row native path stay unchanged.
            return _prefill_gated_rms_norm_capture_ref(x, self.weight, z, self.eps)
        return rms_norm_ref(
            x=x,
            weight=self.weight,
            bias=None,
            z=z,
            eps=self.eps,
            group_size=None,
            norm_before_gate=True,
        )


class QwenGDNKernel(LinearAttnKernelBase):
    _kunpeng_validate_extend_metadata = True

    def decode(
        self,
        q,
        k,
        v,
        a,
        b,
        *,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        query_start_loc,
        **kwargs,
    ):
        q0, k0, v0 = q[0].contiguous(), k[0].contiguous(), v[0].contiguous()
        workspace = ops.alloc_buffer(
            q0.shape[0] * (2 * q0.shape[1] + 3 * v0.shape[2]), dtype=torch.float32
        )
        return ops.qwen_gdn_decode(
            A_log,
            dt_bias,
            q0,
            k0,
            v0,
            a.contiguous(),
            b.contiguous(),
            ssm_states,
            cache_indices,
            workspace,
        ).unsqueeze(1)

    def extend(
        self, q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc, **kwargs
    ):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        g, beta = g.float().contiguous(), beta.float().contiguous()
        state = ssm_states[cache_indices].contiguous()
        from . import prefill_graph

        if prefill_graph.enabled() and prefill_graph.gdn_supported(q, k, v):
            prefill_graph.register()
            grain = 32
            size = prefill_graph.gdn_workspace_numel(
                q.shape[1], query_start_loc.numel() - 1
            )
            workspace = ops.alloc_buffer(size, dtype=torch.float32)
            output = ops.qwen_gdn_prefill(
                q, k, v, g, beta, state, query_start_loc, workspace, grain
            )
            return output, state, None
        output = torch.empty_like(v)
        library = initialize()
        parallel = q.shape[1] > 32
        sequences = query_start_loc.numel() - 1
        grain = (
            32
            if (prefill_graph.gdn_supported(q, k, v) or query_start_loc.numel() <= 3)
            else 128
        )
        if parallel:
            slices = (v.shape[3] + grain - 1) // grain
            size = (
                2 * q.shape[1] * q.shape[2]
                + sequences * v.shape[2] * slices * 3 * grain
            )
        else:
            size = 2 * q.shape[2] + 3 * v.shape[3]
        workspace = ops.alloc_buffer(size, dtype=torch.float32)
        kernel = (
            library.qwen36_gdn_prefill_parallel_with_workspace
            if parallel
            else library.qwen36_gdn_prefill_with_workspace
        )
        # Keep short tails serial. Larger chunks reuse each worker's state
        # slice across the token recurrence, with one pool launch per layer.
        rc = kernel(
            *(t.data_ptr() for t in (q, k, v, g, beta, state, query_start_loc, output)),
            query_start_loc.numel() - 1,
            q.shape[1],
            q.shape[2],
            v.shape[2],
            q.shape[3],
            v.shape[3],
            *([grain] if parallel else []),
            workspace.data_ptr(),
            workspace.numel(),
        )
        if rc:
            raise RuntimeError(f"Qwen GDN prefill failed: {rc}")
        return output, state, None

    def target_verify(
        self,
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        *,
        ssm_states,
        cache_indices,
        query_start_loc,
        intermediate_states_buffer,
        intermediate_state_indices,
        cache_steps,
        **kwargs,
    ):
        from .speculative import verify_gdn

        return verify_gdn(
            A_log,
            dt_bias,
            q,
            k,
            v,
            a,
            b,
            ssm_states,
            cache_indices[: query_start_loc.numel() - 1],
            intermediate_states_buffer,
            intermediate_state_indices[: query_start_loc.numel() - 1],
            cache_steps,
        )


def conv_decode(x, state, weight, bias, activation, conv_state_indices, **kwargs):
    assert bias is None and activation in ("silu", "swish")
    if x.ndim == 3:
        from .speculative import verify_conv

        return verify_conv(
            x,
            state,
            weight,
            conv_state_indices,
            kwargs["intermediate_conv_window"],
            kwargs["intermediate_state_indices"],
        )
    return ops.qwen_causal_conv(x.contiguous(), state, weight, conv_state_indices)


class QwenAttention:
    def __init__(self):
        initialize()

    def forward_verify(self, q, k, v, layer, fb, save_kv_cache=True):
        kb = fb.token_to_kv_pool.get_key_buffer(layer.layer_id)
        vb = fb.token_to_kv_pool.get_value_buffer(layer.layer_id)
        if save_kv_cache:
            ops.set_kv_buffer_kunpeng(
                kb.view(-1, 256), fb.out_cache_loc, k.reshape(-1, 256).contiguous()
            )
            ops.set_kv_buffer_kunpeng(
                vb.view(-1, 256), fb.out_cache_loc, v.reshape(-1, 256).contiguous()
            )
        steps = fb.spec_info.draft_token_num
        lengths = (
            fb.seq_lens[:, None]
            + torch.arange(1, steps + 1, dtype=fb.seq_lens.dtype, device=q.device)[
                None, :
            ]
        ).reshape(-1)
        requests = fb.req_pool_indices.repeat_interleave(steps)
        query = q.reshape(-1, layer.tp_q_head_num, 256).contiguous()
        op = (
            ops.qwen_gqa_rows
            if is_capturing() and query.shape[0] >= 8
            else ops.qwen_gqa_decode
        )
        output, _ = op(
            query,
            kb,
            vb,
            fb.req_to_token_pool.req_to_token,
            requests,
            lengths,
            float(layer.scaling),
        )
        return output.reshape(-1, layer.tp_q_head_num * 256)

    def forward_decode(self, q, k, v, layer, fb, save_kv_cache=True):
        kb = fb.token_to_kv_pool.get_key_buffer(layer.layer_id)
        vb = fb.token_to_kv_pool.get_value_buffer(layer.layer_id)
        if save_kv_cache:
            ops.set_kv_buffer_kunpeng(
                kb.view(-1, 256), fb.out_cache_loc, k.reshape(-1, 256).contiguous()
            )
            ops.set_kv_buffer_kunpeng(
                vb.view(-1, 256), fb.out_cache_loc, v.reshape(-1, 256).contiguous()
            )
        query = q.reshape(-1, layer.tp_q_head_num, 256).contiguous()
        op = (
            ops.qwen_gqa_rows
            if is_capturing() and query.shape[0] >= 8
            else ops.qwen_gqa_decode
        )
        output, _ = op(
            query,
            kb,
            vb,
            fb.req_to_token_pool.req_to_token,
            fb.req_pool_indices,
            fb.seq_lens,
            float(layer.scaling),
        )
        return output.reshape(-1, layer.tp_q_head_num * 256)


def compact_weight_loaders(module):
    from .compact_attention import (
        load_compact_o_weight,
        load_compact_qkv_weight,
        load_compact_o_scale,
        load_compact_qkv_scale,
    )

    rank = module.attn_tp_rank

    def bind(param, loader):
        if hasattr(param, "_weight_loader"):
            param._weight_loader = loader
        else:
            param.weight_loader = loader

    bind(
        module.qkv_proj.weight,
        lambda param, weight, loaded_shard_id=None: load_compact_qkv_weight(
            param.data, weight, loaded_shard_id, rank
        ),
    )
    bind(
        module.o_proj.weight,
        lambda param, weight: load_compact_o_weight(param.data, weight, rank),
    )
    for name, dummy_value in (("weight_scale", 1.0), ("weight_offset", 0.0)):
        qkv_param = getattr(module.qkv_proj, name, None)
        if qkv_param is not None:
            bind(
                qkv_param,
                lambda param, scale, loaded_shard_id=None, value=dummy_value: load_compact_qkv_scale(
                    param.data, scale, loaded_shard_id, rank, value
                ),
            )
        o_param = getattr(module.o_proj, name, None)
        if o_param is not None:
            bind(o_param, lambda param, scale: load_compact_o_scale(param.data, scale))


def qkv_layout(qkv):
    q, gate, k, v = ops.qwen_qkv_layout(qkv)
    return q, k, v, gate


def native_rope_cache(module):
    # get_rope() already shares one rotary module between equal layers.
    # Keep its BF16 conversion on that owner instead of copying it per layer.
    rotary = module.rotary_emb
    cache = getattr(rotary, "kunpeng_bf16_cache", None)
    if cache is None:
        cache = rotary.cos_sin_cache.to(torch.bfloat16).contiguous()
        rotary.kunpeng_bf16_cache = cache
    module.registered_graph_rope_cache = cache
    return cache


def qknorm_rope(module, positions, q, k):
    cache = native_rope_cache(module)
    return ops.qwen_qknorm_rope(
        q,
        k,
        module.q_norm.gemma_weight,
        module.k_norm.gemma_weight,
        cache,
        positions,
        float(module.q_norm.variance_epsilon),
        0,
    )


# The native collective request and its eager SHM caches are process-global.
# A busy call must not fall back into a cache borrowed by another call.
_allreduce_cache_lock = threading.Lock()


def allreduce(tensor):
    if not _allreduce_cache_lock.acquire(blocking=False):
        raise RuntimeError(
            "Qwen allreduce is already using the native collective/cache"
        )
    try:
        # Preserve B1's 256 feature partitions and its owner-die/rounding order.
        rows = tensor.shape[0]
        if tensor.ndim == 2 and tensor.shape[1] == 5120 and 1 < rows <= 4096:
            packed = None
            blocked = False
            eligible_blocked = (
                rows > 128
                and tensor.device.type == "cpu"
                and tensor.dtype == torch.bfloat16
                and tensor.is_contiguous()
            )
            if eligible_blocked and is_capturing():
                packed = ops.qwen_ar_pack_block(tensor, 256)
                blocked = True
            if (
                packed is None
                and not is_capturing()
                and tensor.device.type == "cpu"
                and tensor.dtype == torch.bfloat16
                and tensor.is_contiguous()
            ):
                # This is the same small/large cache already reserved by the
                # graph provider for eager collectives, not a new SHM buffer.
                cache = torch.ops.sgl_kernel.get_or_create_shm_tensor_kunpeng(
                    5120, rows
                )
                if (
                    cache.device.type == "cpu"
                    and cache.dtype == torch.bfloat16
                    and cache.ndim == 2
                    and cache.shape[1] == 5120
                    and cache.shape[0] >= rows
                    and cache.is_contiguous()
                    and torch.ops.sgl_kernel.is_shm_tensor(cache)
                ):
                    src_lo, cache_lo = tensor.data_ptr(), cache.data_ptr()
                    if (
                        src_lo + tensor.nbytes <= cache_lo
                        or cache_lo + cache.nbytes <= src_lo
                    ):
                        # Each partition has M*20 live elements. Slicing a
                        # [256, capacity, 20] view would give the wrong stride.
                        packed = cache.view(-1)[: rows * 5120].view(rows, 5120)
                        if eligible_blocked:
                            torch.ops.sgl_kernel.qwen_registered_ar_pack_block_into(
                                packed, tensor, 256
                            )
                            blocked = True
                        else:
                            packed.view(256, rows, 20).copy_(
                                tensor.view(rows, 256, 20).transpose(0, 1)
                            )
            if packed is None:
                # Unknown layouts retain the original allocation and alias
                # semantics; never attach the eager shared scratch to a graph.
                packed = (
                    tensor.reshape(rows, 256, 20)
                    .transpose(0, 1)
                    .contiguous()
                    .view(rows, 5120)
                )
            result = ops.shm_allreduce_kunpeng(packed)
            # Keep the original destination storage and its aliases. The
            # synchronous collective and this copy finish before cache release.
            if blocked:
                ops.qwen_ar_unpack_block(tensor, packed, 256)
            else:
                tensor.view(rows, 256, 20).copy_(
                    packed.view(256, rows, 20).transpose(0, 1)
                )
            return result
        return ops.shm_allreduce_kunpeng(tensor)
    finally:
        _allreduce_cache_lock.release()
