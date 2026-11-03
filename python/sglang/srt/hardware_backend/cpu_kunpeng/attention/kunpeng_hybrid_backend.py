"""Kunpeng GQA/GDN metadata and persistent resources for the common graph runner."""
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

    def __init__(self, runner, full):
        self.runner=runner
        super().__init__(full, KunpengGDNBackend(runner), runner.mambaish_config.full_attention_layer_ids)
        self.workspace = prefill_graph.resources()
        from sglang.srt.hardware_backend.cpu_kunpeng.bf16_lm_head import prepare
        prepare(runner.model.lm_head, runner.kunpeng_runner.weight_hbw_pool)
        for module in runner.model.modules():
            conv = getattr(module, 'conv1d', None)
            attention = getattr(module, 'attn', None)
            if conv is not None and hasattr(attention, 'conv_weights'):
                attention.conv_weights = conv.weight.view(attention.conv_weights.shape)
            if hasattr(module,'q_norm') and getattr(module,'rotary_emb',None) is not None:
                native_rope_cache(module)
        # New backend resources exist before the first capture; refresh the common collector once.
        runner.kunpeng_runner.init_graph_capture()

    def init_forward_metadata(self, fb):
        super().init_forward_metadata(fb)
        metadata = self.linear_attn_backend.forward_metadata
        metadata.mamba_cache_indices = metadata.mamba_cache_indices.to(torch.int32)

    def get_graph_inputs(self, fb):
        fb.input_ids=fb.input_ids.to(torch.int64)
        values=[fb.input_ids,fb.positions,fb.req_pool_indices,fb.seq_lens,fb.out_cache_loc,
                self.linear_attn_backend.forward_metadata.mamba_cache_indices,
                self.linear_attn_backend.forward_metadata.query_start_loc]
        if fb.forward_mode.is_extend():
            values.extend([fb.extend_seq_lens,fb.extend_prefix_lens,fb.extend_start_loc])
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
        if fb.next_token_logits_buffer is not None:values.append(fb.next_token_logits_buffer)
        return values
