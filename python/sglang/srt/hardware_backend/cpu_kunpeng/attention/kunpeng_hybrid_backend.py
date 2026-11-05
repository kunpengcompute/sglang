"""Kunpeng GQA/GDN metadata and persistent resources for the common graph runner."""
import os

import torch
from sglang.srt.graph import ops
from sglang.srt.layers.attention.hybrid_linear_attn_backend import HybridLinearAttnBackend
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.hardware_backend.cpu_kunpeng.qwen.backend import (
    QwenAttention, QwenGDNKernel, conv_decode, conv_prefill, gating, scatter, native_rope_cache,
)
from sglang.srt.hardware_backend.cpu_kunpeng.qwen import prefill_graph

class KunpengGQABackend(QwenAttention):
    def init_forward_metadata(self, fb):
        self.forward_metadata = None
    def forward_extend(self, q, k, v, layer, fb, save_kv_cache=True, **kwargs):
        if fb.forward_mode.is_target_verify():
            # Verify reuses the extend entry point; the Qwen verify attention
            # (row-wise GQA over the speculative lengths) is the same path the
            # torch-native backend takes.
            return self.forward_verify(q, k, v, layer, fb, save_kv_cache)
        return prefill_graph.forward_attention(q, k, v, layer, fb, save_kv_cache)

class KunpengGDNBackend(GDNAttnBackend):
    def __init__(self, runner):
        super().__init__(runner)
        self.kernel_dispatcher.decode_kernel = QwenGDNKernel()
        self.kernel_dispatcher.extend_kernel = self.kernel_dispatcher.decode_kernel
        self.kernel_dispatcher.supports_packed_decode = False

    def _state(self, layer, fb):
        if fb.mamba_track_mask is not None:
            raise ValueError('State tracking has not been explicitly registered for this backend')
        cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        return cache.conv[0], cache.temporal, self.forward_metadata.mamba_cache_indices, self.forward_metadata.query_start_loc

    def _split(self, layer, x):
        q, k, v = x.split((layer.q_dim, layer.k_dim, layer.v_dim), -1)
        return (q.view(1, x.shape[0], layer.num_q_heads, layer.head_q_dim),
                k.view(1, x.shape[0], layer.num_k_heads, layer.head_k_dim),
                v.view(1, x.shape[0], layer.num_v_heads, layer.head_v_dim))

    def forward_decode(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
        conv, state, indices, qsl = self._state(layer, forward_batch)
        x = conv_decode(mixed_qkv, conv, layer.conv_weights, layer.bias, layer.activation,
                        conv_state_indices=indices)
        q, k, v = self._split(layer, x)
        return self.kernel_dispatcher.decode(q, k, v, a, b, A_log=layer.A_log, dt_bias=layer.dt_bias,
                                             ssm_states=state, cache_indices=indices, query_start_loc=qsl, post_conv_base=x)

    def forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
        if forward_batch.forward_mode.is_target_verify():
            # Speculative verify has a different contract (per-step conv
            # checkpoints + the GDN verify recurrence, no extend metadata);
            # the base class already routes it to the Qwen native kernel.
            return super().forward_extend(layer, forward_batch, mixed_qkv, a, b, **kwargs)
        fb=forward_batch
        conv, state, indices, qsl = self._state(layer, fb)
        prefill_graph.validate_extend_metadata(qsl, indices, mixed_qkv.shape[0], state.shape[0])
        initial = ops.tensor_scalar(fb.extend_prefix_lens, 0., 2)
        x = conv_prefill(mixed_qkv.transpose(0,1), layer.conv_weights, layer.bias, layer.activation,
                         conv, initial, indices, qsl, fb.extend_seq_lens_cpu, metadata_validated=True).transpose(0,1)
        q, k, v = self._split(layer, x)
        g, beta = gating(layer.A_log, a, b, layer.dt_bias)
        output, updated, _ = self.kernel_dispatcher.extend(q,k,v,g,beta,ssm_states=state,
                                                           cache_indices=indices,query_start_loc=qsl)
        scatter(state, indices, updated)
        return output

class KunpengHybridBackend(HybridLinearAttnBackend):
    _kunpeng_explicit_graph = True
    forward_metadata = None
    # Verify slices registered index tables inside the graph; secondary views of
    # input storages must register as aliases during capture.
    capture_options = {"allow_input_aliases": True}

    def __init__(self, runner, full):
        self.runner=runner
        super().__init__(full, KunpengGDNBackend(runner), runner.mambaish_config.full_attention_layer_ids)
        self.workspace = prefill_graph.resources()
        from sglang.srt.hardware_backend.cpu_kunpeng.bf16_lm_head import (
            prepare,
            prepare_weight,
        )

        head = runner.model.lm_head
        pool = runner.kunpeng_runner.weight_hbw_pool
        # Prefer the full-K packed capability: it also covers M1, so the MTP
        # draft chain leaves aten::mm. When it declines (head not resident in
        # HBW, or the small-M full-K plan is off), keep the established
        # pack-once path used by the explicit model.
        head._kunpeng_lm_head_role = True
        if not prepare_weight(head, pool):
            prepare(head, pool)
        for module in runner.model.modules():
            conv = getattr(module, 'conv1d', None)
            attention = getattr(module, 'attn', None)
            if conv is not None and hasattr(attention, 'conv_weights'):
                attention.conv_weights = conv.weight.view(attention.conv_weights.shape)
            if hasattr(module,'q_norm') and getattr(module,'rotary_emb',None) is not None:
                native_rope_cache(module)
        # The attached MTP draft embeds eagerly (outside every graph), so its
        # all-reduce needs the eager SHM caches to exist before the first
        # capture claims the remaining pool: the small bucket covers decode
        # rows, the large one covers draft-extend rows. Same reservation the
        # graph provider performs for eager collectives.
        from sglang.srt.distributed import get_tp_group

        communicator = get_tp_group().kunpeng_communicator
        capacity = int(
            os.environ.get(
                "SGLANG_KUNPENG_COLLECTIVE_MAX_TOKENS",
                int(os.environ.get("SGLANG_KUNPENG_MAX_SEQ_NUM", "1"))
                * int(os.environ.get("SGLANG_KUNPENG_MAX_CUR_LEN", "1")),
            )
        )
        width = runner.model_config.hidden_size
        reserve = torch.ops.sgl_kernel.get_or_create_shm_tensor_kunpeng
        reserve(width, 1)
        reserve(width, max(1, min(communicator.max_tokens, capacity)))
        # New backend resources exist before the first capture; refresh the common collector once.
        runner.kunpeng_runner.init_graph_capture()

    def init_forward_metadata(self, fb):
        super().init_forward_metadata(fb)
        metadata = self.linear_attn_backend.forward_metadata
        metadata.mamba_cache_indices = metadata.mamba_cache_indices.to(torch.int32)

    def graph_unsupported_reason(self, fb, kwargs):
        # The per-step (non-fused) MTP verify fallback gathers and scatters the
        # recurrent conv/SSM state with plain ATen, which an explicit graph
        # cannot record; keep TARGET_VERIFY eager while the fused verify is off.
        if fb.forward_mode.is_target_verify():
            from sglang.srt.hardware_backend.cpu_kunpeng.qwen.speculative import (
                verify_fused_enabled,
            )

            if not verify_fused_enabled():
                return "verify_fused_disabled"
        return None

    def get_graph_inputs(self, fb):
        fb.input_ids=fb.input_ids.to(torch.int64)
        values=[fb.input_ids,fb.positions,fb.req_pool_indices,fb.seq_lens,fb.out_cache_loc,
                self.linear_attn_backend.forward_metadata.mamba_cache_indices,
                self.linear_attn_backend.forward_metadata.query_start_loc]
        if fb.forward_mode.is_extend():
            values.extend([fb.extend_seq_lens,fb.extend_prefix_lens,fb.extend_start_loc])
        if fb.forward_mode.is_target_verify():
            # Verify row metadata is rebuilt outside the graph once per verify
            # and shared by every layer, so it is bound as a dynamic input.
            values.extend([getattr(fb.spec_info, "_kunpeng_verify_requests", None),
                           getattr(fb.spec_info, "_kunpeng_verify_lengths", None),
                           self.linear_attn_backend.verify_intermediate_state_indices])
        result=[];seen=set()
        for tensor in values:
            if tensor is not None and tensor.untyped_storage().data_ptr() not in seen:
                seen.add(tensor.untyped_storage().data_ptr());result.append(tensor)
        return result

    def get_graph_fixed_tensors(self, fb):
        pool=getattr(self.runner.token_to_kv_pool,'full_kv_pool',self.runner.token_to_kv_pool)
        state=self.runner.req_to_token_pool.mamba_pool.mamba_cache
        values=list(pool.k_buffer)+list(pool.v_buffer)+list(state.conv)+[state.temporal,
                    self.runner.req_to_token_pool.req_to_token]+list(self.workspace)
        if fb.forward_mode.is_target_verify() and hasattr(
            state, "intermediate_conv_window"
        ):
            # Speculative verify writes per-step conv/SSM checkpoints into these
            # persistent buffers; the index table that drives them is a dynamic
            # input (see get_graph_inputs).
            values.extend(state.intermediate_conv_window)
            values.append(state.intermediate_ssm)
        if fb.next_token_logits_buffer is not None:values.append(fb.next_token_logits_buffer)
        return values
