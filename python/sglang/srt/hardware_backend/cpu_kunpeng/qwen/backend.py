"""Explicit Kunpeng Qwen primitives. No automatic capture, provider, or arena."""
import ctypes
import os
from functools import lru_cache
import torch
from sgl_kernel import common_ops
from sglang.srt.graph import ops, is_capturing
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm
from sglang.srt.layers.attention.linear.kernels.kernel_backend import LinearAttnKernelBase
from . import ops as primitives
from .capabilities import get_capabilities

def contiguous(x):
    return x if x.is_contiguous() else ops.contiguous_kunpeng(x)

def cast(x, dtype):
    if x.dtype == dtype:
        return x
    out = ops.alloc_buffer(x.numel(), dtype=dtype).view(x.shape)
    ops.copy_kunpeng(out, x)
    return out

def select(x, indices):
    return ops.tensor_select(x, indices)

def scatter(x, indices, source):
    ops.tensor_scatter(x, cast(indices, torch.int64), cast(source, x.dtype))

@lru_cache()
def initialize():
    lib = primitives.register_primitives()
    primitives.register_rows()
    from ..activation_ops import register_sigmoid_mul_bf16
    register_sigmoid_mul_bf16()
    from .layout import register_projection_split
    register_projection_split(common_ops.__file__)
    return lib

from .prefill import conv_prefill, gating

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
                residual = ops.tensor_binary(residual, post_residual_addition, 0)
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
                residual = ops.tensor_binary(residual, post_residual_addition, 0)
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


class QwenGDNKernel(LinearAttnKernelBase):
    _kunpeng_validate_extend_metadata = True
    _kunpeng_accepts_postconv_base = True

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
        get_capabilities().require_decode(q.shape[1], q.shape[1])
        post_conv_base = kwargs.get("post_conv_base")
        if is_capturing() and primitives.p20_enabled_for_base(post_conv_base):
            if (8 <= post_conv_base.shape[0] <= 32
                    and q.ndim == k.ndim == v.ndim == 4
                    and q.shape[0] == k.shape[0] == v.shape[0] == 1
                    and q.shape[2:] == k.shape[2:] == (1, 128)
                    and v.shape[2:] == (3, 128)
                    and q.shape[1] == k.shape[1] == v.shape[1] == post_conv_base.shape[0]
                    and primitives.p2_gdn_head_selected(
                        post_conv_base.shape[0], q.shape[2], v.shape[2], q.shape[3], v.shape[3])):
                q_view, k_view, v_view = q[0], k[0], v[0]
                if primitives.p20_selected(post_conv_base, q_view, k_view, v_view):
                    workspace = ops.alloc_buffer(
                        q_view.shape[0] * (2 * q_view.shape[1] + 3 * v_view.shape[2]),
                        dtype=torch.float32,
                    )
                    return ops.qwen_p20_gdn_postconv(
                        A_log, dt_bias, post_conv_base, contiguous(a), contiguous(b),
                        ssm_states, cache_indices, workspace, q_view.shape[1],
                    ).unsqueeze(1)
        q0, k0, v0 = contiguous(q[0]), contiguous(k[0]), contiguous(v[0])
        workspace = ops.alloc_buffer(
            q0.shape[0] * (2 * q0.shape[1] + 3 * v0.shape[2]), dtype=torch.float32
        )
        operation = ops.qwen_gdn_decode
        if q0.shape[0] >= 8 and primitives.p2_gdn_head_selected(
            q0.shape[0], q0.shape[1], v0.shape[1], q0.shape[2], v0.shape[2]
        ):
            operation = ops.qwen_p2_gdn_head
        return operation(
            A_log,
            dt_bias,
            q0,
            k0,
            v0,
            contiguous(a),
            contiguous(b),
            ssm_states,
            cache_indices,
            workspace,
        ).unsqueeze(1)

    def extend(
        self, q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc, **kwargs
    ):
        q, k, v = contiguous(q), contiguous(k), contiguous(v)
        g, beta = contiguous(cast(g, torch.float32)), contiguous(cast(beta, torch.float32))
        state = select(ssm_states, cache_indices)
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
        if is_capturing():
            raise RuntimeError(
                "Qwen GDN prefill geometry has no registered graph primitive; "
                "ctypes execution during capture is forbidden"
            )
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
    get_capabilities().require_decode(x.shape[0], x.shape[0])
    if os.environ.get("SGLANG_KUNPENG_P2_DECODE_CONV_REQUESTS", "0") == "1" and 8 <= x.shape[0] <= 32:
        return ops.qwen_p2_decode_conv_requests(contiguous(x), state, weight, conv_state_indices)
    return ops.qwen_causal_conv(contiguous(x), state, weight, conv_state_indices)


class QwenAttention:
    def __init__(self):
        initialize()

    def forward_verify(self, q, k, v, layer, fb, save_kv_cache=True):
        kb = fb.token_to_kv_pool.get_key_buffer(layer.layer_id)
        vb = fb.token_to_kv_pool.get_value_buffer(layer.layer_id)
        if save_kv_cache:
            ops.set_kv_buffer_kunpeng(
                kb.view(-1, 256), fb.out_cache_loc, contiguous(k.reshape(-1, 256))
            )
            ops.set_kv_buffer_kunpeng(
                vb.view(-1, 256), fb.out_cache_loc, contiguous(v.reshape(-1, 256))
            )
        steps = fb.spec_info.draft_token_num
        lengths = (
            fb.seq_lens[:, None]
            + torch.arange(1, steps + 1, dtype=fb.seq_lens.dtype, device=q.device)[
                None, :
            ]
        ).reshape(-1)
        requests = fb.req_pool_indices.repeat_interleave(steps)
        query = contiguous(q.reshape(-1, layer.tp_q_head_num, 256))
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
        # Reject unsupported geometry before the first persistent KV write.
        get_capabilities().require_decode(fb.batch_size, fb.input_ids.numel())
        kb = fb.token_to_kv_pool.get_key_buffer(layer.layer_id)
        vb = fb.token_to_kv_pool.get_value_buffer(layer.layer_id)
        if save_kv_cache:
            ops.set_kv_buffer_kunpeng(
                kb.view(-1, 256), fb.out_cache_loc, contiguous(k.reshape(-1, 256))
            )
            ops.set_kv_buffer_kunpeng(
                vb.view(-1, 256), fb.out_cache_loc, contiguous(v.reshape(-1, 256))
            )
        query = contiguous(q.reshape(-1, layer.tp_q_head_num, 256))
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


def configure_modules(model):
    initialize()
    torch.set_num_threads(int(os.environ.get("SGLANG_KUNPENG_TORCH_NUM_THREADS", "32")))
    from .linear import apply_dense, declare_projection, prepare_dense
    from sglang.srt.layers.quantization.modelslim.schemes.cpu920f_w8a8_dynamic import CPU920FModelSlimW8A8Dynamic
    for module in model.modules():
        mlp = getattr(module, 'mlp', None)
        gate = getattr(mlp, 'gate_up_proj', None)
        post = getattr(module, 'post_attention_layernorm', None)
        if post is not None:
            post._kunpeng_int8_consumer = isinstance(getattr(gate, 'scheme', None), CPU920FModelSlimW8A8Dynamic)
        qkv = getattr(module, 'qkv_proj', None)
        norm = getattr(module, 'input_layernorm', None)
        if norm is not None:
            norm._kunpeng_int8_consumer = isinstance(getattr(qkv, 'scheme', None), CPU920FModelSlimW8A8Dynamic)
        if getattr(module, '_cpu_projection_role', None):
            if module.weight.dtype == torch.bfloat16:
                module._cpu_linear_prepare = prepare_dense
                module._cpu_linear_apply = apply_dense
            module._cpu_allreduce = allreduce
        if isinstance(module, QwenGemmaRMSNorm):
            module._cpu_allreduce = allreduce

class QwenGatedRMSNorm(RMSNorm):
    def forward(self, x, z=None):
        if z is None:
            raise ValueError('GDN gated Norm requires its gate')
        if x.shape[0] <= 96:
            from .norm_rowgroups_full_occupancy import decode_norm
            return decode_norm(contiguous(x), self.weight, contiguous(z), self.eps)
        if os.environ.get("SGLANG_KUNPENG_P2_NORM_REDUCE_KEEP_SILU", "0") == "1":
            from . import prefill_graph
            prefill_graph.register()
            activated = ops.tensor_unary(cast(z, torch.float32), 5)
            width = int(os.environ["SGLANG_KUNPENG_P2_REFERENCE_VECTOR_WIDTH"])
            return ops.qwen_p2_norm_reduce_keep_silu(x, self.weight, activated, float(self.eps), width)
        original = x.dtype
        x, z, weight = cast(x, torch.float32), cast(z, torch.float32), cast(self.weight, torch.float32)
        variance = ops.tensor_mean_last(ops.tensor_unary(x, 0))
        denominator = ops.tensor_unary(ops.tensor_scalar(variance, float(self.eps), 0), 1)
        rstd = ops.tensor_unary(denominator, 2)
        value = ops.tensor_binary(ops.tensor_binary(x, rstd, 1), weight, 1)
        value = ops.tensor_binary(value, ops.tensor_unary(z, 5), 1)
        return cast(value, original)

def apply_gqa_output_gate(output, gate, forward_batch):
    return ops.sigmoid_mul_bf16_kunpeng(output, gate)

def allreduce(tensor):
    rows = tensor.shape[0]
    if tensor.ndim == 2 and tensor.shape[1] == 5120 and rows > 1:
        packed = contiguous(tensor.view(rows, 256, 20).transpose(0, 1)).view(rows, 5120)
        ops.shm_allreduce_kunpeng(packed)
        ops.copy_kunpeng(tensor.view(rows, 256, 20), packed.view(256, rows, 20).transpose(0, 1))
    else:
        ops.shm_allreduce_kunpeng(tensor)
    return tensor


def zeros_like(x):
    out = ops.alloc_buffer(x.numel(), dtype=x.dtype).view(x.shape)
    ops.zero_(out)
    return out
