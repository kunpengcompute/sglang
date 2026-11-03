"""Borrow the enabled attached Qwen MTP component for the existing EAGLE worker."""

from copy import deepcopy


def attached_target_for_draft(target_worker, server_args):
    from sglang.srt.hardware_backend.cpu_kunpeng.qwen import enabled

    if not enabled():
        return None
    target = target_worker.model_runner
    model = target.model
    capabilities = getattr(model, "runtime_capabilities", None)
    if capabilities is None or not capabilities.policy_enabled("mtp"):
        return None
    capabilities.require("mtp")
    if (
        target.device != "cpu"
        or target.tp_size != 16
        or target.pp_size != 1
        or server_args.speculative_algorithm != "EAGLE"
        or server_args.speculative_eagle_topk != 1
        or server_args.speculative_num_steps not in (2, 3)
        or server_args.speculative_num_draft_tokens
        != server_args.speculative_num_steps + 1
        or server_args.speculative_adaptive
        or server_args.speculative_token_map is not None
        or server_args.speculative_draft_attention_backend
        not in (None, "torch_native", "kunpeng_cpu")
    ):
        raise ValueError(
            "Attached Qwen MTP requires fixed CPU TP16 EAGLE topk1 steps2/3"
        )
    mtp = getattr(model, "mtp", None)
    if (
        mtp is None
        or getattr(mtp, "_runtime_component", None) != "mtp"
        or getattr(mtp, "_runtime_capabilities", None) is not capabilities
        or mtp.model.embed_tokens is not model.model.embed_tokens
        or mtp.lm_head is not model.lm_head
    ):
        raise ValueError(
            "Attached Qwen MTP must retain target-owned weights and shared modules"
        )
    return target


def configure_borrowed_draft(model_config, target):
    """Copy metadata only; the component and its parameters remain target-owned."""
    result = deepcopy(model_config)
    if result.hf_config.architectures != ["Qwen3_5ForCausalLMMTP"]:
        raise ValueError(
            "Borrowed component requires the existing Qwen MTP model configuration"
        )
    # Use the already TP-adjusted geometry of the actual, loaded component.
    # A fresh checkpoint ModelConfig still describes raw Q24/KV4, not Q32/KV16.
    text = deepcopy(target.model.mtp.config)
    result.hf_text_config = text
    if hasattr(result.hf_config, "text_config"):
        result.hf_config.text_config = text
    else:
        raise ValueError("Attached Qwen MTP requires its text subconfiguration")
    text.num_hidden_layers = 1
    text.full_attention_interval = 1
    if hasattr(text, "layer_types"):
        text.layer_types = ["full_attention"]
    text.num_nextn_predict_layers = 1
    result.hf_config.num_nextn_predict_layers = 1
    result._derive_model_shapes()
    result.num_nextn_predict_layers = 1
    result.num_hidden_layers = 1
    result.num_attention_layers = 1
    result.full_attention_interval = 1
    result.is_hybrid_swa = False
    result.is_hybrid_swa_compress = False
    result.swa_attention_layer_ids = []
    result.full_attention_layer_ids = [0]
    if (
        result.context_len != target.model_config.context_len
        or text.hidden_size != target.model_config.hidden_size
        or result.dtype != target.model_config.dtype
    ):
        raise ValueError("Borrowed MTP geometry/context must match its target")
    return result


def validate_borrowed_config(runner):
    target = runner.borrowed_qwen_target_runner
    target.model.runtime_capabilities.require("mtp")
    config = runner.model_config
    owner = target.model.mtp.config
    if (
        not runner.is_draft_worker
        or runner.device != "cpu"
        or runner.tp_size != 16
        or runner.pp_size != 1
        or runner.server_args.nnodes != 1
        or config.hf_config.architectures != ["Qwen3_5ForCausalLMMTP"]
        or config.num_hidden_layers != 1
        or config.num_attention_layers != 1
        or config.num_nextn_predict_layers != 1
        or config.full_attention_layer_ids != [0]
        or config.is_hybrid_swa
        or config.is_hybrid_swa_compress
        or config.context_len != target.model_config.context_len
        or config.dtype != target.model_config.dtype
        or (config.num_attention_heads, config.num_key_value_heads, config.head_dim)
        != (owner.num_attention_heads, owner.num_key_value_heads, owner.head_dim)
    ):
        raise ValueError("Borrowed MTP must retain loaded one-layer CPU TP16 geometry")


def bind_borrowed_model(runner):
    validate_borrowed_config(runner)
    target = runner.borrowed_qwen_target_runner
    if (
        not runner.is_draft_worker
        or runner.device != "cpu"
        or runner.tp_rank != target.tp_rank
        or runner.tp_size != target.tp_size
        or runner.pp_size != target.pp_size
        or runner.req_to_token_pool is not target.req_to_token_pool
        or runner.token_to_kv_pool_allocator is not target.token_to_kv_pool_allocator
    ):
        raise ValueError(
            "Borrowed MTP runner must share target rank and request allocator"
        )
    target.model.runtime_capabilities.require("mtp")
    return target.model.mtp


def validate_borrowed_pools(runner):
    target = runner.borrowed_qwen_target_runner
    draft = runner.token_to_kv_pool
    owner = getattr(target.token_to_kv_pool, "full_kv_pool", target.token_to_kv_pool)
    if (
        draft is target.token_to_kv_pool
        or draft is owner
        or draft.layer_num != 1
        or runner.mambaish_config is not None
        or runner.req_to_token_pool is not target.req_to_token_pool
        or runner.token_to_kv_pool_allocator is not target.token_to_kv_pool_allocator
    ):
        raise ValueError(
            "Attached MTP requires independent one-layer KV and shared slot ownership"
        )
    # Allocated by the ordinary CPU KV pool, outside every forward-arena scope.
    # Check that no storage was borrowed from target KV or the private HBW pool.
    target_ranges = [
        (t.data_ptr(), t.data_ptr() + t.nbytes) for t in owner.k_buffer + owner.v_buffer
    ]
    pool = target.kunpeng_runner.weight_hbw_pool
    target_ranges.append((pool._base_addr, pool._base_addr + pool.pool_size))
    for tensor in draft.k_buffer + draft.v_buffer:
        lo, hi = tensor.data_ptr(), tensor.data_ptr() + tensor.nbytes
        if tensor.device.type != "cpu" or any(
            lo < b and a < hi for a, b in target_ranges
        ):
            raise ValueError("Draft DDR KV overlaps target KV/HBW storage")


class AttachedQwenMultiStepDraftBackend:
    """CPU topk1 step metadata, using the existing TorchNative attention path."""

    def __init__(self, model_runner, topk, num_steps):
        from sglang.srt.layers.attention.torch_native_backend import (
            TorchNativeAttnBackend,
        )

        if (
            model_runner.borrowed_qwen_target_runner is None
            or topk != 1
            or num_steps not in (2, 3)
        ):
            raise ValueError("This CPU draft container is limited to attached Qwen MTP")
        self.attn_backends = [
            TorchNativeAttnBackend(model_runner) for _ in range(num_steps - 1)
        ]
        self.num_steps = num_steps
        self._forward_batch_id = None

    def init_forward_metadata(self, forward_batch):
        self._forward_batch_id = id(forward_batch)
        self._base_lengths = forward_batch.seq_lens.clone()
        cpu = forward_batch.seq_lens_cpu
        self._base_lengths_cpu = (
            cpu.clone() if cpu is not None else self._base_lengths.cpu().clone()
        )

    def prepare_step(self, index, forward_batch):
        # Metadata is bound before ModelRunner/graph inputs, never patched
        # after an attention op or inferred from a mutable positions tensor.
        if forward_batch.forward_mode.is_idle():
            return
        if (
            id(forward_batch) != self._forward_batch_id
            or not 0 <= index < self.num_steps - 1
        ):
            raise ValueError("CPU draft step has no matching metadata snapshot")
        forward_batch.positions = (self._base_lengths + index).long()
        forward_batch.seq_lens = self._base_lengths + index + 1
        forward_batch.seq_lens_cpu = self._base_lengths_cpu + index + 1
        forward_batch.seq_lens_sum = int(forward_batch.seq_lens_cpu.sum().item())
