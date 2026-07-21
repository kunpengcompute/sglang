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

import torch

from sglang.srt.graph import register_op


def _setup_fused_add_rmsnorm_kunpeng():
    def shape_infer(act, residual, weight, eps):
        return [list(residual.shape)]

    def eager_fn(act, residual, weight, eps):
        out = torch.empty(residual.shape, dtype=residual.dtype)
        torch.ops.sgl_kernel.fused_add_rmsnorm_kunpeng(
            act, residual, weight, eps, out)
        return out

    register_op('fused_add_rmsnorm_kunpeng', shape_infer, eager_fn)


def _setup_rmsnorm_kunpeng():
    def shape_infer(acts, weights, eps):
        return [(acts.shape, acts.dtype)]

    def eager_fn(acts, weights, eps):
        out = torch.empty(acts.shape, dtype=acts.dtype)
        torch.ops.sgl_kernel.rmsnorm_kunpeng(acts, weights, eps, out)
        return out

    register_op('rmsnorm_kunpeng', shape_infer, eager_fn)


def _setup_quant_kunpeng():
    def shape_infer(input):
        return [(list(input.shape), torch.int8),
                ((input.shape[0],), torch.float32)]

    def eager_fn(input):
        out = torch.empty(input.shape, dtype=torch.int8)
        scale = torch.empty(input.shape[0], dtype=torch.float32)
        torch.ops.sgl_kernel.quant_kunpeng(input, out, scale)
        return out, scale

    register_op('quant_kunpeng', shape_infer, eager_fn)


def _setup_quant_inplace_kunpeng():
    def shape_infer(input, out, scale):
        return []

    def eager_fn(input, out, scale):
        torch.ops.sgl_kernel.quant_kunpeng(input, out, scale)
        return None

    register_op('quant_inplace_kunpeng', shape_infer, eager_fn)


def _setup_batched_gemm_pack_allthreads_kunpeng():
    def shape_infer(input):
        return [(input.shape, input.dtype)]

    def eager_fn(input):
        out = torch.empty(input.shape, dtype=input.dtype)
        torch.ops.sgl_kernel.batched_gemm_pack_allthreads_kunpeng(input, out)
        return out

    register_op('batched_gemm_pack_allthreads_kunpeng', shape_infer, eager_fn)


def _setup_batched_gemm_woqs8_allthreads_kunpeng():
    def shape_infer(act, weight, rscale, cscale):
        bs = act.shape[0]
        m = act.shape[1]
        n = weight.shape[1]
        return [((bs, m, n), act.dtype)]

    def eager_fn(act, weight, rscale, cscale):
        bs = act.shape[0]
        m = act.shape[1]
        n = weight.shape[1]
        out = torch.empty((bs, m, n), dtype=act.dtype)
        torch.ops.sgl_kernel.batched_gemm_woqs8_allthreads_kunpeng(
            act, weight, rscale, cscale, out)
        return out

    register_op('batched_gemm_woqs8_allthreads_kunpeng', shape_infer, eager_fn)


def _setup_rope_kunpeng():
    def shape_infer(positions, q, k, cos_sin_cache):
        return [(q.shape, q.dtype), (k.shape, k.dtype)]

    def eager_fn(positions, q, k, cos_sin_cache):
        q_out = torch.empty(q.shape, dtype=q.dtype)
        k_out = torch.empty(k.shape, dtype=k.dtype)
        torch.ops.sgl_kernel.rope_kunpeng(
            positions, q, k, q_out, k_out, cos_sin_cache)
        return q_out, k_out

    register_op('rope_kunpeng', shape_infer, eager_fn)


def _setup_s8_gemm_pack_kunpeng():
    def shape_infer(input, split_r, split_c):
        return [(input.shape, input.dtype)]

    def eager_fn(input, split_r, split_c):
        out = torch.empty(input.shape, dtype=input.dtype)
        torch.ops.sgl_kernel.s8_gemm_pack_kunpeng(
            input, out, split_r, split_c)
        return out

    register_op('s8_gemm_pack_kunpeng', shape_infer, eager_fn)


def _setup_alloc_buffer():
    def shape_infer(size_bytes, dtype=torch.uint8):
        return [((size_bytes,), dtype)]

    def eager_fn(size_bytes, dtype=torch.uint8):
        return torch.empty(size_bytes, dtype=dtype)

    register_op('alloc_buffer', shape_infer, eager_fn)


def _setup_s8_s8_packed_gemm_bf16_dq_kunpeng():
    def shape_infer(input, weight, weight_scale, scale, workspace,
                    tile_m, tile_n, tile_k):
        M = input.shape[0]
        N = weight.shape[0]
        return [((M, N), torch.bfloat16)]

    def eager_fn(input, weight, weight_scale, scale, workspace,
                 tile_m, tile_n, tile_k):
        M = input.shape[0]
        N = weight.shape[0]
        output = torch.empty((M, N), dtype=torch.bfloat16)
        torch.ops.sgl_kernel.s8_s8_packed_gemm_bf16_dq_kunpeng(
            input, weight, weight_scale, scale,
            output, workspace, tile_m, tile_n, tile_k)
        return output

    register_op('s8_s8_packed_gemm_bf16_dq_kunpeng', shape_infer, eager_fn)


def _setup_bf16_gemm_pack_kunpeng():
    def shape_infer(input, split_r, split_c):
        return [(input.shape, input.dtype)]

    def eager_fn(input, split_r, split_c):
        out = torch.empty(input.shape, dtype=input.dtype)
        torch.ops.sgl_kernel.bf16_gemm_pack_kunpeng(input, out, split_r, split_c)
        return out

    register_op('bf16_gemm_pack_kunpeng', shape_infer, eager_fn)


def _setup_bf16_packed_gemm_kunpeng():
    def shape_infer(input, weight, workspace, num_threads):
        M = input.shape[0]
        N = weight.shape[0]
        return [((M, N), input.dtype)]

    def eager_fn(input, weight, workspace, num_threads):
        M = input.shape[0]
        N = weight.shape[0]
        output = torch.empty((M, N), dtype=input.dtype)
        torch.ops.sgl_kernel.bf16_packed_gemm_kunpeng(
            input, weight, output, workspace, num_threads)
        return output

    register_op('bf16_packed_gemm_kunpeng', shape_infer, eager_fn)


def _setup_grouped_topk_kunpeng():
    def shape_infer(router_logits, bias, topk, num_expert_group, topk_group,
                    renormalize, scoring_func_sigmoid, moe_balance, v2):
        bs = router_logits.shape[0]
        return [((bs, topk), torch.float32), ((bs, topk), torch.int16)]

    def eager_fn(router_logits, bias, topk, num_expert_group, topk_group,
                 renormalize, scoring_func_sigmoid, moe_balance, v2):
        bs = router_logits.shape[0]
        tw = torch.empty((bs, topk), dtype=torch.float32)
        ti = torch.empty((bs, topk), dtype=torch.int16)
        torch.ops.sgl_kernel.grouped_topk_kunpeng(
            router_logits, tw, ti,
            topk, num_expert_group, topk_group,
            bias=bias, experts_offset=None,
            renormalize=bool(renormalize),
            scoring_func_sigmoid=bool(scoring_func_sigmoid),
            moe_balance=bool(moe_balance), v2=v2)
        return tw, ti

    register_op('grouped_topk_kunpeng', shape_infer, eager_fn)


def _setup_load_balance_padded_tokens_kunpeng():
    def shape_infer(topk_ids, topk_weights, num_token_non_padded, num_experts, topk):
        return []

    def eager_fn(topk_ids, topk_weights, num_token_non_padded, num_experts, topk):
        torch.ops.sgl_kernel.load_balance_padded_tokens_kunpeng(
            topk_ids, topk_weights, num_token_non_padded, num_experts, topk)
        return None

    register_op('load_balance_padded_tokens_kunpeng', shape_infer, eager_fn)


def _setup_multinomial_kunpeng():
    def shape_infer(probs, num_samples, replacement):
        batch = probs.shape[0]
        return [((batch, num_samples), torch.int64)]

    def eager_fn(probs, num_samples, replacement):
        batch = probs.shape[0]
        out = torch.empty((batch, num_samples), dtype=torch.int64)
        torch.ops.sgl_kernel.multinomial_kunpeng(
            probs, out, int(num_samples), bool(replacement))
        return out

    register_op('multinomial_kunpeng', shape_infer, eager_fn)


def _setup_shm_batched_allgather_kunpeng():
    def shape_infer(input, comm_size):
        out_shape = list(input.shape)
        out_shape[-1] *= comm_size
        return [(out_shape, input.dtype)]

    def eager_fn(input, comm_size):
        out_shape = list(input.shape)
        out_shape[-1] *= comm_size
        output = torch.empty(out_shape, dtype=input.dtype)
        torch.ops.sgl_kernel.shm_batched_allgather_kunpeng(
            input, output, comm_size)
        return output

    register_op('shm_batched_allgather_kunpeng', shape_infer, eager_fn)


def _setup_shm_dual_allgather_kunpeng():
    def shape_infer(src0, dst0, src1, dst1):
        return []  # in-place on dst0, dst1

    def eager_fn(src0, dst0, src1, dst1):
        torch.ops.sgl_kernel.shm_dual_allgather_kunpeng(
            src0, dst0, src1, dst1)
        return None

    register_op('shm_dual_allgather_kunpeng', shape_infer, eager_fn)


def _setup_shm_mla_q_alltoall_kunpeng():
    def shape_infer(q, tp_size):
        B, Nl, D = q.shape
        Btp, Nh = B // tp_size, Nl * tp_size
        return [((Btp, Nh, D), q.dtype)]

    def eager_fn(q, tp_size):
        B, Nl, D = q.shape
        output = torch.empty((B // tp_size, Nl * tp_size, D),
                             dtype=q.dtype, device=q.device)
        torch.ops.sgl_kernel.shm_mla_q_alltoall_kunpeng(q, output)
        return output

    register_op('shm_mla_q_alltoall_kunpeng', shape_infer, eager_fn)


def _setup_shm_mla_o_alltoall_kunpeng():
    def shape_infer(o, tp_size):
        Btp, Nh, D = o.shape
        B, Nl = Btp * tp_size, Nh // tp_size
        return [((B, Nl, D), o.dtype)]

    def eager_fn(o, tp_size):
        Btp, Nh, D = o.shape
        output = torch.empty((Btp * tp_size, Nh // tp_size, D),
                             dtype=o.dtype, device=o.device)
        torch.ops.sgl_kernel.shm_mla_o_alltoall_kunpeng(o, output)
        return output

    register_op('shm_mla_o_alltoall_kunpeng', shape_infer, eager_fn)


def _setup_shm_allreduce_kunpeng():
    def shape_infer(input):
        return []

    def eager_fn(input):
        torch.ops.sgl_kernel.shm_allreduce_kunpeng(input)
        return None

    register_op('shm_allreduce_kunpeng', shape_infer, eager_fn)


def _setup_shm_reduce_scatter_kunpeng():
    def shape_infer(input):
        return []

    def eager_fn(input):
        torch.ops.sgl_kernel.shm_reduce_scatter_kunpeng(input)
        return None

    register_op('shm_reduce_scatter_kunpeng', shape_infer, eager_fn)


def _setup_bmm_kunpeng():
    def shape_infer(input, weight):
        B, M, K = input.shape
        N = weight.shape[1]
        return [((B, M, N), input.dtype)]

    def eager_fn(input, weight):
        B, M, K = input.shape
        N = weight.shape[1]
        out = torch.empty((B, M, N), dtype=input.dtype, device=input.device)
        torch.ops.sgl_kernel.bmm_kunpeng(input, weight, out)
        return out

    register_op('bmm_kunpeng', shape_infer, eager_fn)


def _setup_embedding_kunpeng():
    def shape_infer(indices, weight, org_vocab_start, org_vocab_end,
                    num_org_vocab_padding, added_vocab_start, added_vocab_end):
        num_tokens = indices.numel()
        hidden_dim = weight.shape[1]
        return [((num_tokens, hidden_dim), weight.dtype)]

    def eager_fn(indices, weight, org_vocab_start, org_vocab_end,
                 num_org_vocab_padding, added_vocab_start, added_vocab_end):
        num_tokens = indices.numel()
        hidden_dim = weight.shape[1]
        output = torch.empty((num_tokens, hidden_dim), dtype=weight.dtype)
        torch.ops.sgl_kernel.embedding_kunpeng(
            indices, weight, output,
            org_vocab_start, org_vocab_end,
            num_org_vocab_padding, added_vocab_start, added_vocab_end)
        return output

    register_op('embedding_kunpeng', shape_infer, eager_fn)


def _setup_silu_mul_quant_kunpeng():
    def shape_infer(gateup):
        M = gateup.shape[0]
        inter_dim = gateup.shape[-1] // 2
        return [((M, inter_dim), torch.int8),
                ((M, 1), torch.float32)]

    def eager_fn(gateup):
        M = gateup.shape[0]
        inter_dim = gateup.shape[-1] // 2
        outs = torch.empty((M, inter_dim), dtype=torch.int8)
        scales = torch.empty((M, 1), dtype=torch.float32)
        torch.ops.sgl_kernel.silu_mul_quant_kunpeng(gateup, outs, scales)
        return outs, scales

    register_op('silu_mul_quant_kunpeng', shape_infer, eager_fn)


def _setup_moe_silu_mul_quant_kunpeng():
    def shape_infer(gateup, outs, scales, experts_offset):
        return []

    def eager_fn(gateup, outs, scales, experts_offset):
        recv_tokens = int(experts_offset[-1].item())
        torch.ops.sgl_kernel.silu_mul_quant_kunpeng(
            gateup[:recv_tokens], outs[:recv_tokens], scales[:recv_tokens])
        return None

    register_op('moe_silu_mul_quant_kunpeng', shape_infer, eager_fn)


def _setup_moe_dispatch_send_kunpeng():
    def shape_infer(x, topk_idx, num_experts, num_max_dispatch_tokens_per_rank,
                    parallel_policy, num_tokens, batch_id):
        return []

    def eager_fn(x, topk_idx, num_experts, num_max_dispatch_tokens_per_rank,
                 parallel_policy, num_tokens, batch_id):
        torch.ops.sgl_kernel.moe_dispatch_send_kunpeng(
            x, topk_idx, num_experts, num_max_dispatch_tokens_per_rank,
            parallel_policy, num_tokens, batch_id)
        return None

    register_op('moe_dispatch_send_kunpeng', shape_infer, eager_fn)


def _setup_moe_dispatch_recv_kunpeng():
    def shape_infer(batch_id):
        return []

    def eager_fn(batch_id):
        torch.ops.sgl_kernel.moe_dispatch_recv_kunpeng(batch_id)
        return None

    register_op('moe_dispatch_recv_kunpeng', shape_infer, eager_fn)


def _setup_moe_combine_send_kunpeng():
    def shape_infer(x, src_info, num_max_dispatch_tokens_per_rank,
                    num_experts, hidden, parallel_sizes, batch_id,
                    combined_x, topk_idx, topk_weights, num_tokens,
                    num_topk, enable_allgather):
        return []

    def eager_fn(x, src_info, num_max_dispatch_tokens_per_rank,
                 num_experts, hidden, parallel_sizes, batch_id,
                 combined_x, topk_idx, topk_weights, num_tokens,
                 num_topk, enable_allgather):
        torch.ops.sgl_kernel.moe_combine_send_kunpeng(
            x, src_info, num_max_dispatch_tokens_per_rank,
            num_experts, hidden, parallel_sizes, batch_id,
            combined_x, topk_idx, topk_weights, num_tokens,
            num_topk, enable_allgather)
        return None

    register_op('moe_combine_send_kunpeng', shape_infer, eager_fn)


def _setup_moe_combine_recv_kunpeng():
    def shape_infer(combined_x, topk_idx, topk_weights, num_tokens,
                    num_max_dispatch_tokens_per_rank, num_topk, hidden,
                    batch_id):
        return []

    def eager_fn(combined_x, topk_idx, topk_weights, num_tokens,
                 num_max_dispatch_tokens_per_rank, num_topk, hidden,
                 batch_id):
        torch.ops.sgl_kernel.moe_combine_recv_kunpeng(
            combined_x, topk_idx, topk_weights, num_tokens,
            num_max_dispatch_tokens_per_rank, num_topk, hidden,
            batch_id)
        return None

    register_op('moe_combine_recv_kunpeng', shape_infer, eager_fn)


def _setup_cat_kunpeng():
    def shape_infer(a, b, dim):
        shape = list(a.shape)
        shape[dim] += b.shape[dim]
        return [(tuple(shape), a.dtype)]

    def eager_fn(a, b, dim):
        return torch.cat([a, b], dim=dim)

    register_op('cat_kunpeng', shape_infer, eager_fn)


def _setup_contiguous_kunpeng():
    def shape_infer(x):
        return [(x.shape, x.dtype)]

    def eager_fn(x):
        return x.contiguous()

    register_op('contiguous_kunpeng', shape_infer, eager_fn)


def _setup_set_kv_buffer_kunpeng():
    def shape_infer(kv_buffer, loc, cache_k):
        return []

    def eager_fn(kv_buffer, loc, cache_k):
        torch.ops.sgl_kernel.set_kv_buffer_kunpeng(kv_buffer, loc, cache_k)
        return None

    register_op('set_kv_buffer_kunpeng', shape_infer, eager_fn)


def _setup_copy_kunpeng():
    def shape_infer(dst, src):
        return []

    def eager_fn(dst, src):
        torch.ops.sgl_kernel.copy_kunpeng(dst, src)
        return None

    register_op('copy_kunpeng', shape_infer, eager_fn)


def _setup_flash_mla_dense_decode_kunpeng():
    def shape_infer(q, kcache, block_table, seqlens_kv,
                    softmax_scale, is_causal, extra_buffer, meta,
                    head_dim_v):
        bsz = q.shape[0]
        n_heads = q.shape[2]
        return [((bsz, 1, n_heads, head_dim_v), torch.bfloat16),
                ((bsz, 1, n_heads), torch.float32)]

    def eager_fn(q, kcache, block_table, seqlens_kv,
                 softmax_scale, is_causal, extra_buffer, meta,
                 head_dim_v):
        bsz = q.shape[0]
        n_heads = q.shape[2]
        o = torch.empty((bsz, 1, n_heads, head_dim_v), dtype=torch.bfloat16)
        softmax_lse = torch.empty((bsz, 1, n_heads), dtype=torch.float32)
        torch.ops.sgl_kernel.flash_mla_dense_decode_kunpeng(
            q, kcache, None,
            block_table, seqlens_kv,
            o, softmax_lse,
            float(softmax_scale), bool(is_causal),
            extra_buffer, meta)
        return o, softmax_lse

    register_op('flash_mla_dense_decode_kunpeng', shape_infer, eager_fn)


def _setup_topk_convert_kunpeng():
    def shape_infer(src_info, token_ids, experts_offset, num_ranks,
                    num_local_experts, num_max_dispatch_tokens_per_rank, is_prefill):
        return []

    def eager_fn(src_info, token_ids, experts_offset, num_ranks,
                 num_local_experts, num_max_dispatch_tokens_per_rank, is_prefill):
        torch.ops.sgl_kernel.topk_convert_kunpeng(
            src_info, token_ids, experts_offset, num_ranks,
            num_local_experts, num_max_dispatch_tokens_per_rank, is_prefill)
        return None

    register_op('topk_convert_kunpeng', shape_infer, eager_fn)


def _setup_igemm_fusedmoe_gateup_kunpeng():
    def shape_infer(act, scale, experts_w13, experts_w13_scale,
                    token_ids, experts_offset, moe_gateup,
                    tmpx, tmpy, tmp_scales):
        return []

    def eager_fn(act, scale, experts_w13, experts_w13_scale,
                 token_ids, experts_offset, moe_gateup,
                 tmpx, tmpy, tmp_scales):
        recv_tokens = int(experts_offset[-1].item())
        torch.ops.sgl_kernel.igemm_fusedmoe_gateup_kunpeng(
            act[:recv_tokens], scale[:recv_tokens],
            experts_w13, experts_w13_scale,
            token_ids[:recv_tokens], experts_offset,
            moe_gateup, tmpx, tmpy, tmp_scales)
        return None

    register_op('igemm_fusedmoe_gateup_kunpeng', shape_infer, eager_fn)


def _setup_igemm_fusedmoe_down_kunpeng():
    def shape_infer(moe_silu_int8, experts_w2, moe_silu_scale, experts_w2_scale,
                    token_ids, experts_offset, moe_down,
                    tmpx, tmpy, tmp_scales):
        return []

    def eager_fn(moe_silu_int8, experts_w2, moe_silu_scale, experts_w2_scale,
                 token_ids, experts_offset, moe_down,
                 tmpx, tmpy, tmp_scales):
        recv_tokens = int(experts_offset[-1].item())
        torch.ops.sgl_kernel.igemm_fusedmoe_down_kunpeng(
            moe_silu_int8[:recv_tokens], experts_w2,
            moe_silu_scale[:recv_tokens], experts_w2_scale,
            token_ids[:recv_tokens], experts_offset,
            moe_down, tmpx, tmpy, tmp_scales)
        return None

    register_op('igemm_fusedmoe_down_kunpeng', shape_infer, eager_fn)


def _setup_moe_comm_barrier_kunpeng():
    def shape_infer():
        return []

    def eager_fn():
        torch.ops.sgl_kernel.moe_comm_barrier_kunpeng()
        return None

    register_op('moe_comm_barrier_kunpeng', shape_infer, eager_fn)


def _setup_mul_scalar_add_kunpeng():
    def shape_infer(src, dst, alpha):
        return []

    def eager_fn(src, dst, alpha):
        torch.ops.sgl_kernel.mul_scalar_add_kunpeng(src, dst, alpha)
        return None

    register_op('mul_scalar_add_kunpeng', shape_infer, eager_fn)


def setup():
    _setup_fused_add_rmsnorm_kunpeng()
    _setup_rmsnorm_kunpeng()
    _setup_quant_kunpeng()
    _setup_quant_inplace_kunpeng()
    _setup_batched_gemm_pack_allthreads_kunpeng()
    _setup_batched_gemm_woqs8_allthreads_kunpeng()
    _setup_rope_kunpeng()
    _setup_s8_gemm_pack_kunpeng()
    _setup_alloc_buffer()
    _setup_s8_s8_packed_gemm_bf16_dq_kunpeng()
    _setup_bf16_gemm_pack_kunpeng()
    _setup_bf16_packed_gemm_kunpeng()
    _setup_grouped_topk_kunpeng()
    _setup_load_balance_padded_tokens_kunpeng()
    _setup_multinomial_kunpeng()
    _setup_shm_batched_allgather_kunpeng()
    _setup_shm_mla_q_alltoall_kunpeng()
    _setup_shm_mla_o_alltoall_kunpeng()
    _setup_shm_allreduce_kunpeng()
    _setup_shm_reduce_scatter_kunpeng()
    _setup_shm_dual_allgather_kunpeng()
    _setup_bmm_kunpeng()
    _setup_embedding_kunpeng()
    _setup_silu_mul_quant_kunpeng()
    _setup_moe_silu_mul_quant_kunpeng()
    _setup_mul_scalar_add_kunpeng()
    _setup_moe_comm_barrier_kunpeng()
    _setup_moe_dispatch_send_kunpeng()
    _setup_moe_dispatch_recv_kunpeng()
    _setup_moe_combine_send_kunpeng()
    _setup_moe_combine_recv_kunpeng()
    _setup_cat_kunpeng()
    _setup_contiguous_kunpeng()
    _setup_set_kv_buffer_kunpeng()
    _setup_copy_kunpeng()
    _setup_flash_mla_dense_decode_kunpeng()
    _setup_topk_convert_kunpeng()
    _setup_igemm_fusedmoe_gateup_kunpeng()
    _setup_igemm_fusedmoe_down_kunpeng()

setup()
