from __future__ import annotations

import dataclasses
import functools
import logging
import math
import os
import warnings
from functools import lru_cache, partial
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sglang.jit_kernel.norm import can_use_fused_inplace_qknorm as can_use_jit_qk_norm
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.utils import (
    get_bool_env_var,
    get_device_capability,
    is_blackwell_supported,
    is_cpu_920f,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
    print_info_once,
)
from sglang.srt.utils.multi_stream_utils import (
    maybe_execute_in_parallel,
    with_multi_stream,
)

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_musa = is_musa()
_is_npu = is_npu()
_is_hip = is_hip()
_is_xpu = is_xpu()

if _is_cuda:
    from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache

    from sglang.jit_kernel.flash_attention import (
        flash_attn_varlen_func,
    )

if _is_musa:
    from flash_attn_interface import flash_attn_varlen_func

if _is_npu:
    import torch_npu

from sglang.srt.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
)
from sglang.srt.distributed import utils as dist_utils
from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.rotary_embedding import apply_rotary_pos_emb
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, get_bool_env_var

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

ROTARY_EMBED_CLASSES = {
    "normal": apply_rotary_pos_emb,
}

# === Vision Encoder === #
FLASHINFER_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024

# Batch buckets for cuDNN graph caching - graphs are cached per bucket size
# This avoids creating a new graph for each unique batch size at runtime
BATCH_BUCKETS = [8, 16, 32, 64]

# Bucketized max seqlens to reduce cuDNN recompilation frequency while
# preserving a tighter upper bound than a single fixed max seqlen.
FLASHINFER_MAX_SEQLEN_BUCKETS = [
    4 * 1024,
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
]


@dataclasses.dataclass
class SingletonCache:
    data: Any = None

    def set_data(self, value: Any) -> None:
        self.data = value

    def get_data(self) -> Optional[Any]:
        return self.data

    def empty(self) -> bool:
        return self.get_data() is None


# TODO: requires real seqlens from images
@functools.lru_cache(maxsize=128)
def _get_cu_seqlens_for_shape(batch_size: int, seqlen: int, device) -> torch.Tensor:
    """
    Generates cumulative sequence lengths (cu_seqlens) for a given batch_size, seqlen, and device.
    Caches the result based on these parameters.
    """
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * seqlen,
        step=seqlen,
        dtype=torch.int32,
        device=device,
    )
    return cu_seqlens


def resolve_seqlens(
    cu_seqlens: torch.Tensor | SingletonCache | None,
    bsz: int,
    seq_len: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if cu_seqlens is None:
        resolved_seqlens = _get_cu_seqlens_for_shape(bsz, seq_len, device=device)
    elif isinstance(cu_seqlens, SingletonCache):
        if cu_seqlens.empty():
            cu_seqlens.set_data(_get_cu_seqlens_for_shape(bsz, seq_len, device=device))
        resolved_seqlens = cu_seqlens.get_data()
    else:
        resolved_seqlens = cu_seqlens
    assert isinstance(
        resolved_seqlens, torch.Tensor
    ), "cu_seqlens must be a torch.Tensor"
    return resolved_seqlens


# ---------------------------------------------------------------------------
# Kunpeng 920F CPU flash attention for vision towers.
#
# F.scaled_dot_product_attention has no CPU flash backend, so it decomposes
# into the math chain (bmm / mask / softmax / bmm) of tiny per-rank GEMMs:
# at TP16 the Qwen3-VL tower runs [S, 1, head_dim] operands, and per-op
# thread-pool barriers dominate - a profiled S=256 graph replay spent 60%
# of its 547 ms in 81 aten::bmm calls averaging ~4 ms each. The Kunpeng C++
# kernel behind flash_attention_with_workspace_kunpeng (the same registered
# graph op the text prefill uses) computes the whole attention in ONE op:
# native bf16 loads, fp32 online softmax, no S-matrix materialization, and
# no mask at all - per-image bidirectional attention comes from varlen
# boundaries. It records natively inside the Kunpeng ViT graph.
# ---------------------------------------------------------------------------

_is_cpu_920f = is_cpu_920f()


def kunpeng_vit_flash_enabled() -> bool:
    """Gate for the Kunpeng vision flash attention: default ON on the 920F
    CPU backend; opt out with SGLANG_KUNPENG_VIT_FLASH_ATTN=0. Applies to
    eager tower forwards and the Kunpeng ViT graph alike - both run the
    same kernel, so graph replays stay identical to eager."""
    return _is_cpu_920f and os.environ.get(
        "SGLANG_KUNPENG_VIT_FLASH_ATTN", "1"
    ) not in ("", "0")


def kunpeng_vision_flash_max_rows() -> int:
    """Largest total token count the shared flash workspace can serve.
    Reuses the vision SHM row budget's env and default, since the graph
    runner already refuses to capture beyond that budget."""
    return int(os.environ.get("SGLANG_KUNPENG_VISION_SHM_MAX_ROWS", "4096"))


# ---------------------------------------------------------------------------
# Kunpeng 920F vision rope: bf16 split-half fast path
# ---------------------------------------------------------------------------
# The vision tower computes the rotary cos/sin tables once per forward and
# hands the same tensors to every block, but the legacy path re-pays per
# block: the call site re-duplicates the half-width tables (torch.cat) and
# apply_rotary_pos_emb_native (selected on this platform: no NPU, no AMX)
# upcasts q/k/cos/sin to fp32 for a ~16-kernel eager chain. Measured on the
# 920F Qwen-VL tower (S=1600, 27 blocks): 1.8ms/block, 17% of the whole
# attention total. The fast path below evaluates the identical neox
# split-half math in bf16 - o1 = t1*c - t2*s; o2 = t2*c + t1*s, equal to
# t*cat(c,c) + rotate_half(t)*cat(s,s) - with the prepared tables cached
# per tower pass. Measured: rope bucket -39%, attention total -7%, with a
# worst-case deviation of one bf16 ulp vs the fp32 reference path (logged
# once per process by the self-check in _vit_rope_fast).
# ---------------------------------------------------------------------------


def kunpeng_vit_rope_bf16_enabled() -> bool:
    """Gate for the bf16 vision rope fast path: default ON on the 920F CPU
    backend; opt out with SGLANG_KUNPENG_VIT_ROPE_BF16=0."""
    return _is_cpu_920f and os.environ.get(
        "SGLANG_KUNPENG_VIT_ROPE_BF16", "1"
    ) not in ("", "0")


_VIT_ROPE = {
    "cos": None,
    "sin": None,
    "hs": None,
    "cc": None,
    "ss": None,
    "checked": False,
}


def _vit_rope_tables(cos, sin, head_size):
    """Prepared bf16 tables [T, 1, rot_dim] for the split-half form, built
    once per tower pass: the tower hands the same cos/sin tensors to every
    block, and holding strong refs to them keeps the identity check immune
    to id() reuse after gc."""
    st = _VIT_ROPE
    if st["cos"] is cos and st["sin"] is sin and st["hs"] == head_size:
        return st["cc"], st["ss"]
    cc = cos.to(torch.bfloat16).unsqueeze(1).contiguous()
    ss = sin.to(torch.bfloat16).unsqueeze(1).contiguous()
    st.update(cos=cos, sin=sin, hs=head_size, cc=cc, ss=ss)
    return cc, ss


def _vit_rope_one(t, cc, ss):
    """neox split-half rotary in bf16 for one tensor: t [T, h, 2r] -> [T, h, 2r]."""
    half = t.shape[-1] // 2
    t1 = t[..., :half]
    t2 = t[..., half:]
    o1 = t1 * cc
    o1.addcmul_(t2, ss, value=-1.0)
    o2 = t2 * cc
    o2.addcmul_(t1, ss, value=1.0)
    return torch.cat((o1, o2), dim=-1)


def _vit_rope_fast(q, k, cos, sin, head_size):
    cc, ss = _vit_rope_tables(cos, sin, head_size)
    q_out = _vit_rope_one(q, cc, ss)
    k_out = _vit_rope_one(k, cc, ss)
    if not _VIT_ROPE["checked"]:
        # One-shot numerics gate: compare against the exact legacy path
        # (fp32 math over the doubled tables) on the same inputs.
        _VIT_ROPE["checked"] = True
        try:
            q_ref, _ = apply_rotary_pos_emb(
                q, k, torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)
            )
            d = (q_out.float() - q_ref.float()).abs()
            maxabs = d.max().item()
            scale = q_ref.float().abs().max().item()
            logger.info(
                "[vit-rope] bf16 one-shot check: maxabs=%.2e scale=%.2e"
                " rel-to-scale=%.2e",
                maxabs,
                scale,
                maxabs / max(scale, 1e-6),
            )
        except Exception as e:
            logger.warning("[vit-rope] bf16 check failed: %s", e)
    return q_out, k_out


@lru_cache(maxsize=1)
def _kunpeng_flash_geometry() -> Tuple[int, int, int]:
    """(BR, BC, thread_num) tile/thread geometry of the Kunpeng flash
    attention kernel, queried once from C++."""
    br, bc = torch.ops.sgl_kernel.get_flash_attention_block_kunpeng()
    threads = torch.ops.sgl_kernel.get_flash_attention_thread_num()
    return int(br), int(bc), int(threads)


_KUNPENG_VISION_FLASH_WORKSPACE: Optional[torch.Tensor] = None


def kunpeng_vision_flash_workspace(
    qk_head_dim: int, vo_head_dim: int
) -> Optional[torch.Tensor]:
    """Shared scratch tensor for the Kunpeng vision flash attention.

    Sized once for the vision row budget (every capturable image shape and
    any per-image split of it fits), because the kernel bump-allocates its
    per-thread pack buffers from this tensor exactly like the text
    prefill's workspace (kunpeng_attention.cpp flash_attention_with_
    workspace: pack_attn_k/v sized threads * pack_len * head_dim, plus the
    q/attn_s/out_block/max/base tiles). One buffer per process; graph
    captures register it as a fixed storage, so its identity never changes
    under any captured graph.

    Head dims are rounded up to 64 multiples internally: the kernel's pack
    stages step the head dimension in 64-column groups (1VL trans pack) /
    32-column groups (2VL pack) with full-width SVE loads, so
    _forward_kunpeng_flash zero-pads its operands to the same grid and the
    workspace must cover the padded dims (D=72 -> 128)."""
    global _KUNPENG_VISION_FLASH_WORKSPACE
    if not kunpeng_vit_flash_enabled():
        return None
    qk_head_dim = (qk_head_dim + 63) // 64 * 64
    vo_head_dim = (vo_head_dim + 63) // 64 * 64
    br, bc, threads = _kunpeng_flash_geometry()
    # 2x: non-causal calls hand the kernel seq_lens = 2 * cur_lens (the
    # fake per-sequence prefix that disables its causal mask), and the
    # wrapper sizes the per-thread K/V pack buffers from those doubled
    # lengths - so the workspace must cover twice the row budget.
    pack_len = (2 * kunpeng_vision_flash_max_rows() + bc - 1) // bc * bc

    def align64(n: int) -> int:
        return (n + 63) // 64 * 64

    bf16, f32 = 2, 4
    ws_bytes = (
        align64(threads * pack_len * qk_head_dim * bf16)  # pack_attn_k
        + align64(threads * pack_len * vo_head_dim * bf16)  # pack_attn_v
        + align64(threads * br * qk_head_dim * bf16)  # pack_attn_q
        + align64(threads * bc * br * f32)  # attn_s
        + align64(threads * br * vo_head_dim * f32) * 2  # out_block old/new
        + align64(threads * br * f32) * 4  # max/base old/new
    )
    if (
        _KUNPENG_VISION_FLASH_WORKSPACE is None
        or _KUNPENG_VISION_FLASH_WORKSPACE.numel() < ws_bytes
    ):
        _KUNPENG_VISION_FLASH_WORKSPACE = torch.empty(
            ws_bytes, dtype=torch.uint8
        )
    return _KUNPENG_VISION_FLASH_WORKSPACE


class VisionSdpaAttention(nn.Module):
    r"""
    Scaled Dot Product Attention inner product

    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        flatten_batch: bool = False,
        softmax_in_single_precision: bool = False,
        softmax_scale: float | None = None,
        **kwargs,
    ):
        super().__init__()
        self.head_size = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.flatten_batch = flatten_batch
        self.softmax_in_single_precision = softmax_in_single_precision
        self.dropout = dropout
        self.scale = (
            softmax_scale
            if softmax_scale is not None
            else 1.0 / math.sqrt(self.head_size)
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def _generate_mask_cache(
        s: int, flatten_batch: bool, cu_seqlens: tuple
    ) -> torch.BoolTensor:
        """
        Generate a boolean attention mask with caching mechanism.
        Args:
            s: sequence length
            flatten_batch: whether to flatten batch dimension
            cu_seqlens: tuple of cumulative sequence lengths
        Returns:
            attention mask tensor of shape [b, 1, s, s] or [1, s, s]
        """
        if flatten_batch:
            mask = torch.zeros([1, s, s], dtype=torch.bool)
            for i in range(1, len(cu_seqlens)):
                start = cu_seqlens[i - 1]
                end = cu_seqlens[i]
                mask[..., start:end, start:end] = True
        else:
            # [1, 1, 1, s]
            row_indices = torch.arange(s).view(1, 1, 1, s)
            # [1, 1, s, 1]
            col_indices = torch.arange(s).view(1, 1, s, 1)
            # [b, 1, 1, 1]
            seq_lens = torch.tensor(
                [end - start for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])],
            ).view(-1, 1, 1, 1)

            mask = (row_indices < seq_lens) & (col_indices < seq_lens)

        return mask

    def generate_patch_attention_mask(
        self,
        s: int,
        cu_seqlens: Optional[torch.Tensor],
        flatten_batch: bool = False,
    ) -> Optional[torch.Tensor]:
        r"""
        Creates a non-causal 4D mask of shape `(b, 1, s, s)` or `(1, 1, s, s)`.
        Args:
            s: sequence length
            cu_seqlens: cumulative sequence lengths tensor. If not, returns an empty mask
            flatten_batch: whether to flatten batch dimension
        Returns:
            attention mask tensor or None
        """
        if cu_seqlens is None:
            return None

        cu_seqlens_tuple = tuple(cu_seqlens.cpu().tolist())

        return self._generate_mask_cache(s, flatten_batch, cu_seqlens_tuple)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz: int,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        flash_seq_lens: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if self.flatten_batch:
            assert bsz == 1, "flatten_batch is True, bsz must be 1"

        assert q.dim() == 3, q.shape

        # Kunpeng 920F: one fused kernel replaces the CPU SDPA math chain.
        # Eager forwards and the Kunpeng ViT graph capture body both run
        # this same branch, so graph replays stay identical to eager.
        # flash_seq_lens carries the graph input's per-image token counts;
        # when absent (every eager path) it is derived from cu_seqlens.
        if (
            self.flatten_batch
            and bsz == 1
            and q.dtype == torch.bfloat16
            and kunpeng_vit_flash_enabled()
        ):
            out = self._forward_kunpeng_flash(q, k, v, cu_seqlens, flash_seq_lens)
            if out is not None:
                return out

        s = q.shape[0] // bsz

        # [b, 1, s, s]
        if attention_mask is None:
            attention_mask = self.generate_patch_attention_mask(
                s, cu_seqlens, flatten_batch=self.flatten_batch
            )

        if attention_mask is None:
            if self.softmax_in_single_precision:
                raise RuntimeError("Empty attention mask")
        else:
            attention_mask = attention_mask.to(device=q.device)

        q, k, v = [rearrange(x, "(b s) h d -> b h s d", b=bsz) for x in [q, k, v]]

        if self.softmax_in_single_precision:
            k = rearrange(k, "b h s d -> b h d s")
            attn_weights = torch.matmul(q, k) * self.scale
            del k
            # masking
            attention_mask = (~attention_mask) * torch.finfo(q.dtype).min
            attn_weights = attn_weights + attention_mask
            del attention_mask
            # full-precision
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(q.dtype)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.dropout, training=False
            )
            output = torch.matmul(attn_weights, v)
            del attn_weights, v
        else:
            # SDPA
            # [b, h, s, head_size]
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout,
                is_causal=False,
                scale=self.scale,
            )

        # [b, h, s, head_size] --> [b * s, h, head_size]
        output = rearrange(output, "b h s d -> (b s) h d")

        return output

    def _forward_kunpeng_flash(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor],
        flash_seq_lens: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Fused varlen attention via the Kunpeng flash kernel; returns
        None to fall back to the SDPA path below.

        q/k/v are the flattened contiguous [S, heads, head_dim] operands of
        a flatten_batch tower; the kernel is bf16-only and attends within
        each cu_seqlens-delimited image bidirectionally (causal=False) -
        exactly the block-diagonal semantics of the SDPA mask path. The
        padded buffers mirror the text prefill's usage: the kernel loads Q
        and writes O in BR-row tiles and reads K/V in BC-row tiles, so each
        operand is padded past S and zeroed to keep tile over-reads inside
        allocated memory. Head dims are zero-padded to a 64-column grid for
        the same reason (the kernel's pack stages over-run otherwise), which
        is exact for attention and only widens the gemms."""
        s = q.shape[0]
        if s == 0 or s > kunpeng_vision_flash_max_rows():
            # Empty or beyond the shared workspace's capacity; the SDPA
            # path handles any size.
            return None
        # The kernel zero-pads each image's K to 16-row tiles and, with
        # causal=False, nothing masks those padding rows: zero rows carry
        # logit 0 and bias the softmax denominator unless the row max is
        # large. Require 16-aligned token counts and fall back to SDPA
        # otherwise. On the captured single-image path (flash_seq_lens)
        # no data-dependent host read may run inside the capture, but
        # there s is exactly the one image's token count, so this shape
        # check is exact; the eager branch re-checks every image below.
        if s % 16 != 0:
            return None
        if flash_seq_lens is not None:
            if flash_seq_lens.dtype != torch.int32:
                return None
            seq_lens = flash_seq_lens
            # Captured single-image path: the constant counts are [S], so
            # the true max sequence length is S itself - no data-dependent
            # host read may run inside the capture.
            chunked_prefill_size = s
        else:
            if cu_seqlens is None:
                return None
            seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
            if seq_lens.numel() == 0:
                return None
            if bool((seq_lens % 16 != 0).any()):
                # Multi-image eager: a 16-aligned total token count does
                # not imply aligned images (100 + 124 = 224); each image
                # must align on its own or its K padding rows bias the
                # softmax denominator.
                return None
            chunked_prefill_size = int(seq_lens.max())

        num_heads = q.shape[1]
        qk_head_dim = q.shape[2]
        vo_head_dim = v.shape[2]
        # The kernel's pack stages require head dims on a 64-column grid
        # (1VL trans pack for Q/K; the 2VL V pack needs 32) - full-width
        # SVE loads and an n-based store layout both over-run otherwise
        # (D=72 reads the next row's data into the packed tail and writes
        # past the pack block extent). Zero-padding the head dimension is
        # mathematically exact for attention - q/k zeros add nothing to
        # the dot products, v zeros produce zero output columns that get
        # sliced off - and only widens the gemms (72 -> 128 also lands on
        # the production text-path geometry). The scale stays 1/sqrt(real
        # head_dim).
        pad_qk = (qk_head_dim + 63) // 64 * 64
        pad_vo = (vo_head_dim + 63) // 64 * 64
        if pad_qk > 256 or pad_vo > 256:
            # Beyond the shared workspace's intended footprint; SDPA
            # handles any head dim.
            return None
        workspace = kunpeng_vision_flash_workspace(qk_head_dim, vo_head_dim)
        if workspace is None:
            return None

        from sglang.srt.graph import ops as kunpeng

        br, bc, _threads = _kunpeng_flash_geometry()

        def padded_rows(t: torch.Tensor, rows: int, pad_dim: int) -> torch.Tensor:
            buf = kunpeng.alloc_buffer(
                rows * num_heads * pad_dim, dtype=t.dtype
            )
            buf = buf.view(rows, num_heads, pad_dim)
            kunpeng.zero_(buf)
            # Strided destination: copy_kunpeng is aten copy_, which
            # handles the [:, :, :d] head-dim slice fine (and records
            # identically under graph capture).
            kunpeng.copy_kunpeng(buf[: t.shape[0], :, : t.shape[2]], t)
            return buf

        padded_q = padded_rows(q, s + br, pad_qk)
        padded_k = padded_rows(k, s + bc, pad_qk)
        padded_v = padded_rows(v, s + bc, pad_vo)
        out = kunpeng.flash_attention_with_workspace_kunpeng(
            padded_q,
            padded_k,
            padded_v,
            workspace,
            seq_lens,
            False,  # causal: each image attends bidirectionally
            self.scale,
            chunked_prefill_size,
        )
        # Drop the row and head-dim padding; contiguous so downstream
        # projections get a plain dense operand.
        out = out[:s, :, :vo_head_dim].contiguous()
        return out


class VisionTritonAttention(nn.Module):
    """
    Triton-implemented attention without a causal mask
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        use_data_parallel = (
            kwargs["use_data_parallel"] if "use_data_parallel" in kwargs else False
        )
        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
            softmax_scale: override softmax scale (default 1/sqrt(head_dim))
        Returns:
             [b * s, h, head_size]
        """
        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
            if "output_ws" not in kwargs:
                raise RuntimeError("output_ws should be prepared for cuda-graph mode")

            if not isinstance(cu_seqlens, list):
                raise RuntimeError("cuda-graph mode cu_seqlens should be a list")

            output = kwargs["output_ws"]
            context_attention_fwd(
                q,
                k,
                v,
                output,
                cu_seqlens[0],
                cu_seqlens[1],
                cu_seqlens[2],
                is_causal=False,
                sm_scale=softmax_scale,
            )
        else:
            cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)

            # [b * s, head, head_size]
            output = torch.empty_like(q)

            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            max_seqlen = seq_lens.max().item()
            context_attention_fwd(
                q,
                k,
                v,
                output,
                cu_seqlens.to(q.device),
                seq_lens.to(q.device),
                max_seqlen,
                is_causal=False,
                sm_scale=softmax_scale,
            )

        return output


class VisionFlash3Attention(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        if not (_is_cuda or _is_musa):
            raise Exception("VisionFlash3Attention is only available for cuda or musa")
        super().__init__()
        use_data_parallel = (
            kwargs["use_data_parallel"] if "use_data_parallel" in kwargs else False
        )
        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        window_size = kwargs.get("window_size", (-1, -1))
        s_aux = kwargs.get("s_aux", None)

        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
            max_seqlen = cu_seqlens[1]
            fa_kwargs = dict(
                cu_seqlens_q=cu_seqlens[0],
                cu_seqlens_k=cu_seqlens[0],
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                window_size=window_size,
            )
            if s_aux is not None:
                fa_kwargs["sinks"] = s_aux
            output = flash_attn_varlen_func(q, k, v, **fa_kwargs)
        else:
            cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)
            cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(q.device)
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            max_seqlen = seq_lens.max().item()

            fa_kwargs = dict(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                window_size=window_size,
            )
            if s_aux is not None:
                fa_kwargs["sinks"] = s_aux
            output = flash_attn_varlen_func(q, k, v, **fa_kwargs)

        return output


class VisionFlash4Attention(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        if not _is_cuda:
            raise Exception("VisionFlash4Attention is only available for cuda")
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if cu_seqlens is None:
            cu_seqlens = _get_cu_seqlens_for_shape(bsz, seq_len, device=q.device)
        elif isinstance(cu_seqlens, SingletonCache):
            if cu_seqlens.empty():
                cu_seqlens.set_data(
                    _get_cu_seqlens_for_shape(bsz, seq_len, device=q.device)
                )
            cu_seqlens = cu_seqlens.get_data()

        cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(q.device)
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seq_lens.max().item()

        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            ver=4,
        )

        return output


class VisionFlashInferAttention(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        if not _is_cuda:
            raise Exception("VisionFlashInferAttention is only available for cuda")
        super().__init__()
        self.workspace_buffer = (
            kwargs["workspace_buffer"] if "workspace_buffer" in kwargs else None
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if "sequence_lengths" not in kwargs:
            raise RuntimeError(
                "sequence_lengths should be prepared for vision flashinfer_cudnn attention backend"
            )
        if "max_seqlen" not in kwargs:
            raise RuntimeError(
                "max_seqlen should be prepared for vision flashinfer_cudnn attention backend"
            )

        sequence_lengths = kwargs["sequence_lengths"]  # (B_padded,) or (B_padded,1,1,1)
        max_seqlen = kwargs["max_seqlen"]

        # max_seqlen must be python int
        if isinstance(max_seqlen, torch.Tensor):
            if max_seqlen.is_cuda:
                max_seqlen = int(max_seqlen.detach().cpu().item())
            else:
                max_seqlen = int(max_seqlen.item())
        else:
            max_seqlen = int(max_seqlen)

        # flatten if caller gives (b, s, h, d)
        is_reshaped = q.dim() == 4
        if is_reshaped:
            reshape_batch_size = q.shape[0]
            q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])

        if not isinstance(cu_seqlens, torch.Tensor):
            raise RuntimeError(
                "flashinfer_cudnn expects packed indptrs as a torch.Tensor"
            )

        # sequence_lengths -> (B,)
        if not isinstance(sequence_lengths, torch.Tensor):
            raise RuntimeError("sequence_lengths must be a torch.Tensor")
        seq_lens_1d = sequence_lengths.view(-1).to(device=q.device, dtype=torch.int32)
        B = int(seq_lens_1d.numel())

        # cu_seqlens contains packed *element indptrs*:
        # [qk_indptr(B+1), v_indptr(B+1), o_indptr(B+1)] => total 3*(B+1)
        cu_seqlens_1d = cu_seqlens.view(-1).to(device=q.device, dtype=torch.int32)
        expected = 3 * (B + 1)
        if int(cu_seqlens_1d.numel()) != expected:
            raise RuntimeError(
                f"packed indptr numel mismatch: got {cu_seqlens_1d.numel()}, expected {expected} (= 3*(B+1))"
            )

        split = B + 1
        indptr_qk = cu_seqlens_1d[:split].view(split, 1, 1, 1)
        indptr_v = cu_seqlens_1d[split : 2 * split].view(split, 1, 1, 1)
        indptr_o = cu_seqlens_1d[2 * split :].view(split, 1, 1, 1)

        # cuDNN style: (B,1,1,1)
        seq_lens_4d = seq_lens_1d.view(B, 1, 1, 1)

        # indptr are in ELEMENT offsets (not token offsets)
        token_width_q = int(q.shape[1] * q.shape[2])  # heads * head_dim on this rank
        total_elems_q = int(q.numel())

        # check each real sequence fits
        # (skip padded tail where seq_len==0)
        start_elems = indptr_qk.view(-1)[:-1]  # (B,)
        end_elems = start_elems + seq_lens_1d * token_width_q
        if (end_elems > total_elems_q).any():
            raise RuntimeError("offset + len out of bounds; packed indptr is wrong")

        _, _, head_size = q.shape
        scale = softmax_scale if softmax_scale is not None else head_size**-0.5

        output, _ = cudnn_batch_prefill_with_kv_cache(
            q,
            k,
            v,
            scale,
            self.workspace_buffer,
            max_token_per_sequence=max_seqlen,
            max_sequence_kv=max_seqlen,
            actual_seq_lens_q=seq_lens_4d,
            actual_seq_lens_kv=seq_lens_4d,
            causal=False,
            return_lse=True,
            batch_offsets_q=indptr_qk,
            batch_offsets_k=indptr_qk,
            batch_offsets_v=indptr_v,
            batch_offsets_o=indptr_o,
            is_cuda_graph_compatible=True,
        )

        if is_reshaped:
            output = rearrange(output, "(b s) h d -> b s h d", b=reshape_batch_size)

        return output


class VisionAiterAttention(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        if not _is_hip:
            raise Exception("aiter_attn is only available for AMD")
        try:
            from aiter import flash_attn_varlen_func as aiter_flash_attn_varlen_func
        except ImportError as e:
            raise ImportError(
                "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
            ) from e

        self.flash_attn_varlen_func = aiter_flash_attn_varlen_func
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)

        cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(q.device)
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seq_lens.max().item()

        return self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
        )


class VisionAscendAttention(nn.Module):

    def __init__(
        self,
        **kwargs,
    ):
        if not _is_npu:
            raise Exception("VisionAscendAttention is only available for ascend npu")
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
            if "output_ws" not in kwargs:
                raise RuntimeError("output_ws should be prepared for npu-graph mode")
            output = kwargs["output_ws"]
            seq_len_arg = cu_seqlens
        else:
            cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device="cpu")
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            if seq_lens.is_npu:
                seq_lens = seq_lens.to("cpu")
            output = torch.empty_like(q)
            seq_len_arg = seq_lens.to(torch.int32)

        _, num_heads, head_size = q.shape
        num_kv_heads = k.shape[1]

        scale_value = softmax_scale if softmax_scale is not None else head_size**-0.5

        torch_npu._npu_flash_attention_unpad(
            query=q,
            key=k,
            value=v,
            seq_len=seq_len_arg,
            scale_value=scale_value,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            out=output,
        )
        return output


QKV_BACKEND_IMPL = {
    "triton_attn": VisionTritonAttention,
    "sdpa": VisionSdpaAttention,
    "fa3": VisionFlash3Attention,
    "fa4": VisionFlash4Attention,
    "flashinfer_cudnn": VisionFlashInferAttention,
    "ascend_attn": VisionAscendAttention,
    "aiter_attn": VisionAiterAttention,
}


class VisionAttention(nn.Module):
    r"""
        Multi-headed attention without any cache, mostly used for multimodal transformers.


    Args:
        use_qkv_parallel (bool, optional): If True, use QKV-parallel attention.
        softmax_in_single_precision (bool, default to False):
            if ``True``, the softmax will be performed in single-precision
            Otherwise, it will be performed in half-precision

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        use_qkv_parallel: bool,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        qkv_backend: Optional[str] = None,
        quant_config: Optional[QuantizationConfig] = None,
        dropout: float = 0.0,
        softmax_in_single_precision: bool = False,
        softmax_scale: Optional[float] = None,
        flatten_batch: bool = False,
        prefix: str = "",
        proj_bias: bool = True,
        num_dummy_heads: int = 0,
        qkv_bias: bool = True,
        qk_normalization: bool = False,
        qk_normalization_by_head_size: bool = False,
        layer_norm_eps: float = 1e-06,
        customized_position_embedding_applier: Callable[
            [torch.Tensor, torch.Tensor, Any, Any], Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        use_data_parallel: bool = False,
        use_dp_attention_reduce: bool = False,
        aux_stream: Optional[torch.cuda.Stream] = None,
        workspace_buffer: Optional[torch.Tensor] = None,
        use_sink: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        **kwargs,
    ):
        super().__init__()
        if head_dim is None and "head_size" in kwargs:
            head_dim = kwargs.pop("head_size")
            warnings.warn(
                "VisionAttention(head_size=...) is deprecated; use head_dim=...",
                DeprecationWarning,
                stacklevel=2,
            )
        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()
        self.tp_rank = 0 if use_data_parallel else get_attention_tp_rank()
        self.dropout = dropout
        num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_size = head_dim if head_dim is not None else embed_dim // num_heads
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads
        )
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_dummy_heads + num_heads, self.tp_size
        )
        self.num_attention_kv_heads_per_partition = dist_utils.divide(
            num_dummy_heads + num_kv_heads, self.tp_size
        )

        self.q_size = self.num_attention_heads_per_partition * self.head_size
        self.kv_size = self.num_attention_kv_heads_per_partition * self.head_size

        self.qk_normalization = qk_normalization
        self.qk_normalization_by_head_size = qk_normalization_by_head_size

        # Additional dummy heads are used to enable TP for common GPU counts.
        self.dummy_dim = (num_dummy_heads + num_heads) * self.head_size

        if self.qk_normalization:
            self.q_norm, self.k_norm = self._init_qk_norm(
                self.dummy_dim, layer_norm_eps, embed_dim
            )

        elif self.qk_normalization_by_head_size:
            self.q_norm, self.k_norm = self._init_qk_norm(
                self.head_size, layer_norm_eps
            )

        # Select attention backend via a unified method
        _passed_backend = qkv_backend
        qkv_backend = self._determine_attention_backend(_passed_backend)
        if (
            get_global_server_args().mm_attention_backend is None
            and _passed_backend is None
        ):
            print_info_once(f"Multimodal attention backend not set. Use {qkv_backend}.")
        print_info_once(f"Using {qkv_backend} as multimodal attention backend.")

        self.customized_position_embedding_applier = (
            customized_position_embedding_applier
        )
        self.softmax_scale = softmax_scale
        self.qkv_backend = QKV_BACKEND_IMPL[qkv_backend](
            head_dim=self.head_size,
            num_heads=self.num_attention_heads_per_partition,
            num_kv_heads=self.num_attention_kv_heads_per_partition,
            dropout=dropout,
            flatten_batch=flatten_batch,
            softmax_in_single_precision=softmax_in_single_precision,
            softmax_scale=softmax_scale,
            use_data_parallel=use_data_parallel,
            workspace_buffer=workspace_buffer,
        )

        self.use_qkv_parallel = use_qkv_parallel
        if use_qkv_parallel:
            self.qkv_proj = QKVParallelLinear(
                hidden_size=embed_dim,
                head_size=self.head_size,
                total_num_heads=num_dummy_heads + num_heads,
                total_num_kv_heads=num_dummy_heads + num_kv_heads,
                bias=qkv_bias,
                quant_config=quant_config,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                prefix=add_prefix("qkv_proj", prefix),
            )
        else:
            self.qkv_proj = ColumnParallelLinear(
                input_size=embed_dim,
                output_size=3 * self.dummy_dim,
                bias=qkv_bias,
                quant_config=quant_config,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                prefix=add_prefix("qkv_proj", prefix),
            )
        self.proj = RowParallelLinear(
            input_size=self.dummy_dim,
            output_size=embed_dim,
            bias=proj_bias,
            quant_config=quant_config,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            prefix=add_prefix("proj", prefix),
            use_dp_attention_reduce=use_dp_attention_reduce,
        )

        self.workspace_buffer = workspace_buffer
        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()] if aux_stream else []

        self.window_size = window_size
        if use_sink:
            # Allocate the full (unsharded) sink tensor for weight loading;
            # only the local TP slice is used in forward.
            self.sinks = nn.Parameter(
                torch.empty(
                    self.num_attention_heads_per_partition * self.tp_size,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
        else:
            self.sinks = None

    def _init_qk_norm(
        self, norm_dim: int, eps: float, var_hidden_size: Optional[int] = None
    ):
        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
            )
            if get_global_server_args().rl_on_policy_target is not None
            else {}
        )
        q_norm = RMSNorm(
            norm_dim,
            eps=eps,
            var_hidden_size=var_hidden_size,
            **norm_kwargs,
        )
        k_norm = RMSNorm(
            norm_dim,
            eps=eps,
            var_hidden_size=var_hidden_size,
            **norm_kwargs,
        )
        return q_norm, k_norm

    def _determine_attention_backend(self, passed_backend: Optional[str]) -> str:
        """Decide the multimodal attention backend string.

        Priority: server args override > constructor arg > platform default.

        Platform defaults:
        - CUDA (Hopper SM90): "fa3"
        - CUDA (Blackwell SM100): "fa4"
        - CUDA (other): "triton_attn"
        - Non-CUDA: "sdpa"
        """
        override_backend = get_global_server_args().mm_attention_backend
        if override_backend is not None:
            backend = override_backend
        elif passed_backend is not None:
            backend = passed_backend
        elif is_cuda():
            major, minor = get_device_capability()
            if major == 9:
                backend = "fa3"
            elif major == 10:
                backend = "fa4"
            else:
                backend = "triton_attn"
        elif _is_musa:
            if get_device_capability() >= (3, 1):
                backend = "fa3"
            else:
                backend = "triton_attn"
        elif _is_hip:
            if get_device_capability() >= (9, 4) and _use_aiter:
                backend = "aiter_attn"
            else:
                backend = "triton_attn"
        elif _is_xpu:
            backend = "triton_attn"
        else:
            backend = "sdpa"
        if backend == "fa3" and is_blackwell_supported():
            raise ValueError("The 'fa3' backend is not supported on Blackwell GPUs")

        return backend

    def _apply_qk_norm_head_size(self, q: torch.Tensor, k: torch.Tensor):
        """apply qk norm for GLM-OCR vit attn"""
        q_by_head = q.reshape(-1, self.head_size)
        q_by_head = self.q_norm(q_by_head)
        k_by_head = k.reshape(-1, self.head_size)
        k_by_head = self.k_norm(k_by_head)
        q = q_by_head.view(q.shape)
        k = k_by_head.view(k.shape)
        return q, k

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor):
        """apply qk norm for internvl vit attn"""

        def q_l2norm():
            q_ = q.flatten(1, 2)
            if self.tp_size > 1:
                q_ = tensor_model_parallel_all_gather(q_.contiguous())
            q_ = self.q_norm(q_)
            if self.tp_size > 1:
                splitter = partial(
                    split_tensor_along_last_dim, num_partitions=self.tp_size
                )
                q_ = splitter(q_)[self.tp_rank]
            q_ = q_.unflatten(-1, (-1, self.head_size))
            return q_

        def k_l2norm():
            k_ = k.flatten(1, 2)
            if self.tp_size > 1:
                k_ = tensor_model_parallel_all_gather(k_.contiguous())
            k_ = self.k_norm(k_)
            if self.tp_size > 1:
                splitter = partial(
                    split_tensor_along_last_dim, num_partitions=self.tp_size
                )
                k_ = splitter(k_)[self.tp_rank]
            k_ = k_.unflatten(-1, (-1, self.head_size))
            return k_

        with with_multi_stream(True):
            q, k = maybe_execute_in_parallel(
                q_l2norm,
                k_l2norm,
                self.ln_events,
                self.aux_stream,
            )
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        flash_seq_lens: Optional[torch.Tensor] = None,
        full_attn: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            x: [b, s, embed_dim]
            cu_seqlens: [b]
        Returns:
             [s, b, head * head_size]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        assert x.dim() == 3, x.shape
        if (
            get_global_server_args().rl_on_policy_target is not None
            and position_embeddings is not None
        ):
            assert isinstance(position_embeddings, tuple), (
                "expected position_embeddings to be a tuple of two tensors,\n"
                f"but got {type(position_embeddings)}, change if needed"
            )
            position_embeddings = tuple(p.to(x.dtype) for p in position_embeddings)
        x_shape = x.shape
        bsz, s, _ = x_shape
        head = self.num_attention_heads_per_partition
        kv_head = self.num_attention_kv_heads_per_partition

        attn_output_ws = kwargs["output_ws"] if "output_ws" in kwargs else None
        max_seqlen = kwargs["max_seqlen"] if "max_seqlen" in kwargs else None
        sequence_lengths = (
            kwargs["sequence_lengths"] if "sequence_lengths" in kwargs else None
        )
        if self.use_qkv_parallel:
            # [b, s, embed_dim] --> [b, s, embed_dim]
            qkv, _ = self.qkv_proj(x)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # [b, s, embed_dim] --> [b * s, head, head_size]
            q = q.reshape(bsz * s, head, -1).contiguous()
            k = k.reshape(bsz * s, kv_head, -1).contiguous()
            v = v.reshape(bsz * s, kv_head, -1).contiguous()
            if self.qk_normalization_by_head_size:
                q, k = self._apply_qk_norm_head_size(q, k)
        else:
            # [b, s, embed_dim] --> [s, b, embed_dim]
            x = rearrange(x, "b s ... -> s b ...")
            # [s, b, embed_dim] --> [s, b, head * 3 * head_size]
            qkv, _ = self.qkv_proj(x)

            # [s, b, head, head_dim_sum]
            new_x_shape = qkv.size()[:-1] + (
                head,
                self.q_size + 2 * self.kv_size,
            )
            qkv = qkv.view(*new_x_shape)

            # [s, b, head, 3 * head_size] --> 3 [s, b, head, head_size]
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # [s, b, head, head_size] --> [b, s, head, head_size]
            q, k, v = [
                rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v)
            ]

            if self.qk_normalization_by_head_size:
                q, k = self._apply_qk_norm_head_size(q, k)

        cos = None
        sin = None

        if position_embeddings is not None:
            if self.customized_position_embedding_applier is not None:
                q, k = self.customized_position_embedding_applier(
                    q, k, position_embeddings, x_shape
                )
            else:
                cos, sin = position_embeddings
        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            cos = rotary_pos_emb_cos
            sin = rotary_pos_emb_sin

        if cos is not None and sin is not None:
            original_q_shape = q.shape
            original_k_shape = k.shape

            # [total_tokens, head, head_size] for q / [total_tokens, kv_head, head_size] for k
            q = q.view(-1, head, self.head_size)
            k = k.view(-1, kv_head, self.head_size)

            if (
                cos.size(-1) * 2 == self.head_size
                and kunpeng_vit_rope_bf16_enabled()
                and q.dtype == torch.bfloat16
            ):
                # bf16 fast path: half-width tables, split-half math, and a
                # per-tower-pass table cache; see _vit_rope_fast. The legacy
                # branch keeps the exact old behavior for every other
                # shape/dtype/platform.
                q, k = _vit_rope_fast(q, k, cos, sin, self.head_size)
            else:
                if cos.size(-1) * 2 == self.head_size:
                    cos = torch.cat([cos, cos], dim=-1)
                    sin = torch.cat([sin, sin], dim=-1)

                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            q = q.view(original_q_shape)
            k = k.view(original_k_shape)

        if q.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            q = rearrange(q, "b s ... -> (b s) ...")
        if k.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            k = rearrange(k, "b s ... -> (b s) ...")
        if v.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            v = rearrange(v, "b s ... -> (b s) ...")

        assert q.dim() == 3, q.dim()
        assert k.dim() == 3, k.dim()
        assert v.dim() == 3, v.dim()

        # internvl
        if self.qk_normalization and not self.qk_normalization_by_head_size:
            # jit kernel
            if can_use_jit_qk_norm(self.head_size, q.dtype):

                # q: [tokens, head, head_size]  ->  [tokens, embed_dim]
                head_dim_for_norm = head * self.head_size

                q, k = apply_qk_norm(
                    q=q,
                    k=k,
                    q_norm=self.q_norm,
                    k_norm=self.k_norm,
                    head_dim=head_dim_for_norm,
                    alt_stream=self.aux_stream,
                )

            else:
                q, k = self._apply_qk_norm(q, k)

        if full_attn or self.sinks is None:
            effective_window_size = (-1, -1)
            s_aux = None
        else:
            effective_window_size = self.window_size
            q_head_start = self.tp_rank * self.num_attention_heads_per_partition
            q_head_end = (self.tp_rank + 1) * self.num_attention_heads_per_partition
            s_aux = self.sinks[q_head_start:q_head_end]

        output = self.qkv_backend.forward(
            q=q,
            k=k,
            v=v,
            bsz=bsz,
            seq_len=s,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            flash_seq_lens=flash_seq_lens,
            sequence_lengths=sequence_lengths,
            max_seqlen=max_seqlen,
            output_ws=attn_output_ws,
            softmax_scale=self.softmax_scale,
            window_size=effective_window_size,
            s_aux=s_aux,
        )

        assert output.dim() == 3, output.shape

        if self.use_qkv_parallel:
            # [b * s, h, head_size] --> [b, s, h * head_size]
            output = rearrange(output, "(b s) ... h d -> b s ... (h d)", b=bsz)

            # [b, s, h * head_size] --> [b, s, h * head_size]
            output, _ = self.proj(output)
        else:
            # [b * s, h, head_size] --> [s, b, h * head_size]
            context_layer = rearrange(
                output, "(b s) h d -> s b (h d)", b=bsz, s=s
            ).contiguous()

            # [s, b, h * head_size] --> [s, b, h * head_size]
            output, _ = self.proj(context_layer)

            # [s, b, h * head_size] --> [b, s, h * head_size]
            output = output.view(bsz, s, -1)

        return output
