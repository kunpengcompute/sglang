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
"""Shared KV-cache pool for Kunpeng 920F (step 1: allocation + attach).

Switch: ``SGLANG_KUNPENG_KV_SHARE_GROUP``
    0  (default) = every rank allocates its own KV pool (baseline)
    4 / 8 / 16   = that many consecutive TP ranks share ONE KV pool

Group layout::

    group_size = SGLANG_KUNPENG_KV_SHARE_GROUP
    group_base = tp_rank // group_size * group_size
    intra      = tp_rank - group_base        # 0 = leader, rest = followers

What happens with the switch on:

* The leader creates ONE POSIX shm segment holding the whole KV pool (main
  ``kv_buffer`` + DSA ``index_k_buffer``), places every page STRICTLY on its own
  NUMA node (``MPOL_BIND``), pre-faults all pages and verifies the placement via
  ``/proc/self/numa_maps`` before publishing the header's ``ready`` flag.
  Rank and NUMA node are 1:1 on 920F, so "the leader's node" is unambiguously
  the CPU-affinity node of that process -- there is no env override for it.
* The other ``group_size - 1`` ranks attach to that segment and build the exact
  same tensor views over it. They allocate nothing.
* A process may hold TWO pools -- the target model's and the MTP / draft
  model's. They have different layer ranges, so each gets its OWN segment and
  the ``role`` (``target`` / ``draft``) is part of the segment name; opening the
  same role twice is rejected.

The segment is the SAME SIZE as the private pool each rank used to allocate, so
per-node demand never grows; only ``/dev/shm`` is new (checked up front).

Failure policy: NO FALLBACK. Every failure raises :class:`SharedKvPoolError`
and aborts startup -- one rank reading a private pool while its peers read the
shared one would serve stale KV, which is worse than not starting.

Not done yet (next steps, do NOT assume):
1. No write gating: every rank still writes the segment. The ranks of a group
   compute byte-identical KV for the same slot, so the bytes are correct and
   each rank reads what it wrote itself -- this step does not depend on
   cross-process write->read visibility. Leader-only writes require a
   group-level fence/drain and must land together.
2. Placement is single-node (the leader's). The next placement step is to
   spread the segment over ALL NUMA nodes of the group, one slice per rank
   (rank <-> node is 1:1); that needs a per-rank "done" handshake in the header
   before ``ready`` is published.
3. Attach uses polling + the ``ready`` flag, not a group barrier; there is no
   failure agreement (if the leader dies, followers wait until the timeout).
4. Memory budget / cell_size (D6), HiCache (D7), PD single-receiver (D9) and
   runtime invariant checks are untouched: every rank still sizes its pool as
   if it were going to allocate the whole thing.
"""

from __future__ import annotations

import atexit
import ctypes
import logging
import os
import re
import struct
import time
import zlib
from contextlib import contextmanager
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KV_SHARE_GROUP_DISABLED: int = 0
#: Legal ``SGLANG_KUNPENG_KV_SHARE_GROUP`` values (besides 0).
KV_SHARE_GROUP_CHOICES: Tuple[int, ...] = (4, 8, 16)

#: One process can hold TWO pools: the target model's and the MTP/draft
#: model's. They have different geometry (different layer ranges), so each gets
#: its own segment and the role is part of the segment name.
POOL_ROLE_TARGET: str = "target"
POOL_ROLE_DRAFT: str = "draft"

#: Bumped whenever the byte layout changes; part of the shm name AND the header,
#: so a stale segment from an older build is never attached.
_LAYOUT_VERSION: int = 1

#: Reserved header area at the start of the segment; data starts after it.
_HEADER_SIZE: int = 4096
_MAGIC: bytes = b"SGLKVS01"
_HEADER_VERSION: int = 1

_DTYPE_CANDIDATES: Tuple[Tuple[str, int], ...] = (
    ("bfloat16", 1),
    ("float16", 2),
    ("float32", 3),
    ("float8_e4m3fn", 4),
    ("float8_e5m2", 5),
    ("uint8", 6),
)

# (name, struct code) in byte order; little-endian, no padding.
_HEADER_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("magic", "8s"),
    ("version", "H"),
    ("header_size", "H"),
    ("group_size", "I"),
    ("group_base", "I"),
    ("dp_rank", "I"),
    ("pp_rank", "I"),
    ("pp_size", "I"),
    ("start_layer", "I"),
    ("layout_version", "I"),
    ("total_bytes", "Q"),
    ("kv_buffer_bytes", "Q"),
    ("index_k_bytes", "Q"),
    ("num_slots", "I"),
    ("page_size", "I"),
    ("layer_num", "I"),
    ("kv_cache_dim", "I"),
    ("index_head_dim", "I"),
    ("num_full_layers", "I"),
    ("store_dtype_code", "I"),
    ("index_k_dtype_code", "I"),
    ("numa_node", "i"),
    ("checksum", "I"),
    ("ready", "I"),
)
_FMT_SIZE = {"8s": 8, "H": 2, "I": 4, "i": 4, "Q": 8}
_HEADER_FORMAT: str = "<" + "".join(code for _, code in _HEADER_FIELDS)


def _compute_header_offsets() -> Dict[str, int]:
    offsets: Dict[str, int] = {}
    cursor = 0
    for name, code in _HEADER_FIELDS:
        offsets[name] = cursor
        cursor += _FMT_SIZE[code]
    return offsets


_HEADER_OFFSETS: Dict[str, int] = _compute_header_offsets()
_HEADER_NAMES: Tuple[str, ...] = tuple(name for name, _ in _HEADER_FIELDS)
_CHECKSUM_IDX: int = _HEADER_NAMES.index("checksum")
_CHECKSUM_OFFSET: int = _HEADER_OFFSETS["checksum"]
_READY_OFFSET: int = _HEADER_OFFSETS["ready"]
_HEADER_USED_BYTES: int = _READY_OFFSET + _FMT_SIZE["I"]
if _HEADER_USED_BYTES > _HEADER_SIZE:
    raise RuntimeError(
        f"KV-share header ({_HEADER_USED_BYTES} B) exceeds the reserved "
        f"area ({_HEADER_SIZE} B)"
    )

#: How long a follower waits for the leader to publish ``ready``.
_ATTACH_TIMEOUT_S: float = 300.0
_ATTACH_POLL_S: float = 0.5

#: Pages on the leader's node must be at least this share of the segment;
#: anything lower means the memory policy did not take effect.
_PLACEMENT_MIN_RATIO: float = 0.99

#: Only "not published yet" is retryable; everything else is a hard mismatch.
_RETRYABLE_ERROR_PREFIXES = ("not_ready",)

#: Comparison order for the header check (semantic fields, then byte sizes).
_HEADER_COMPARE_ORDER: Tuple[str, ...] = (
    "group_size",
    "group_base",
    "dp_rank",
    "pp_rank",
    "pp_size",
    "start_layer",
    "layout_version",
    "num_slots",
    "page_size",
    "layer_num",
    "kv_cache_dim",
    "index_head_dim",
    "num_full_layers",
    "store_dtype_code",
    "index_k_dtype_code",
    "kv_buffer_bytes",
    "index_k_bytes",
    "total_bytes",
)

# ---------------------------------------------------------------------------
# Process-local state
# ---------------------------------------------------------------------------

_ENABLED: bool = False
_INITIALIZED: bool = False
#: role -> segment name, for every pool opened in this process. A second pool
#: with a DIFFERENT role (target + MTP draft) is expected; the same role twice
#: is a bug.
_OPENED_ROLES: Dict[str, str] = {}
_IS_LEADER: bool = True
_GROUP_SIZE: int = KV_SHARE_GROUP_DISABLED
_GROUP_BASE: int = 0
_INTRA_GROUP_RANK: int = 0
_NUMA_NODE: int = -1
_SEGMENT_BYTES: int = 0

# Keeps ``shm.buf`` (and therefore every tensor built on it) alive for the whole
# process, so ``data_ptr()`` / ``untyped_storage()`` stay stable across graph
# capture and replay.
_LIVE_SEGMENTS: List[shared_memory.SharedMemory] = []
_CREATED_SEGMENTS: List[shared_memory.SharedMemory] = []


class SharedKvPoolError(RuntimeError):
    """Fatal error of the shared KV pool path (never recovered from)."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_kv_share_group() -> int:
    """Read ``SGLANG_KUNPENG_KV_SHARE_GROUP`` (0 = disabled)."""
    try:
        from sglang.srt.environ import envs

        return int(envs.SGLANG_KUNPENG_KV_SHARE_GROUP.get())
    except Exception:
        return KV_SHARE_GROUP_DISABLED


def is_kv_share_enabled() -> bool:
    """True iff this process' KV pool lives in the shared segment."""
    return _ENABLED


def get_kv_share_state() -> Dict[str, Any]:
    """Introspection helper for startup logging / QA."""
    return {
        "enabled": _ENABLED,
        "initialized": _INITIALIZED,
        "group_size": _GROUP_SIZE,
        "group_base": _GROUP_BASE,
        "intra_group_rank": _INTRA_GROUP_RANK,
        "is_leader": _IS_LEADER,
        "numa_node": _NUMA_NODE,
        "segment_bytes": _SEGMENT_BYTES,
        "roles": dict(_OPENED_ROLES),
    }


class SharedKvRequest:
    """Everything needed to open the segment, resolved once per rank.

    Built by :func:`shared_kv_request` from the ``ModelRunner`` before the pool
    is constructed, consumed by :func:`open_shared_kv_storage` while the pool
    builds its buffers.
    """

    __slots__ = (
        "model_runner",
        "role",
        "group_size",
        "group_base",
        "intra_group_rank",
        "dp_rank",
        "pp_rank",
        "pp_size",
        "start_layer",
        "numa_node",
        "instance_key",
    )

    def __init__(
        self,
        model_runner: Any,
        role: str,
        group_size: int,
        group_base: int,
        intra_group_rank: int,
        dp_rank: int,
        pp_rank: int,
        pp_size: int,
        start_layer: int,
        numa_node: int,
        instance_key: str,
    ) -> None:
        self.model_runner = model_runner
        self.role = role
        self.group_size = int(group_size)
        self.group_base = int(group_base)
        self.intra_group_rank = int(intra_group_rank)
        self.dp_rank = int(dp_rank)
        self.pp_rank = int(pp_rank)
        self.pp_size = int(pp_size)
        self.start_layer = int(start_layer)
        self.numa_node = int(numa_node)
        self.instance_key = instance_key

    @property
    def is_leader(self) -> bool:
        return self.intra_group_rank == 0

    def __repr__(self) -> str:
        return (
            "SharedKvRequest("
            f"role={self.role}, group={self.group_size}, base={self.group_base}, "
            f"intra={self.intra_group_rank}, dp_rank={self.dp_rank}, "
            f"pp_rank={self.pp_rank}, pp_size={self.pp_size}, "
            f"start_layer={self.start_layer}, numa_node={self.numa_node}, "
            f"instance={self.instance_key})"
        )


class KvShareLayout:
    """Byte layout of ONE group's segment, derived from the pool geometry.

    Everything here is a pure function of the (shared) pool geometry, so the
    leader and the followers compute the same layout and the header check is an
    exact comparison. The size does NOT depend on ``group_size``: the segment is
    exactly what one rank used to allocate privately.
    """

    def __init__(
        self,
        pool: Any,
        group_size: int,
        group_base: int,
        dp_rank: int,
        instance_key: str,
        pp_rank: int = 0,
        pp_size: int = 1,
        start_layer: int = 0,
        role: str = POOL_ROLE_TARGET,
    ) -> None:
        self.role = role
        self.group_size = int(group_size)
        self.group_base = int(group_base)
        self.dp_rank = int(dp_rank)
        self.instance_key = instance_key
        self.pp_rank = int(pp_rank)
        self.pp_size = int(pp_size)
        self.start_layer = int(start_layer)
        self.layout_version = _LAYOUT_VERSION

        self.num_slots = int(pool.size) + int(pool.page_size)
        self.page_size = int(pool.page_size)
        self.layer_num = int(pool.layer_num)
        self.kv_cache_dim = int(pool.kv_cache_dim)
        self.store_dtype = pool.store_dtype
        self.store_dtype_code = _dtype_to_code(self.store_dtype)
        if self.store_dtype_code == 0:
            raise SharedKvPoolError(
                f"unsupported KV dtype {self.store_dtype} for the shared pool"
            )
        self.index_head_dim = int(pool.index_head_dim)
        self.index_k_dtype = pool.index_k_dtype
        self.index_k_dtype_code = _dtype_to_code(self.index_k_dtype)
        if self.index_k_dtype_code == 0:
            raise SharedKvPoolError(
                f"unsupported index-K dtype {self.index_k_dtype} for the "
                "shared pool"
            )
        self.skip_topk_layers: List[bool] = [bool(s) for s in pool.skip_topk_layers]
        if len(self.skip_topk_layers) != self.layer_num:
            raise SharedKvPoolError(
                f"skip_topk_layers length {len(self.skip_topk_layers)} != "
                f"layer_num {self.layer_num}"
            )
        self.num_full_layers = sum(1 for s in self.skip_topk_layers if not s)
        self.device = pool.device

        self.kv_layer_stride = (
            self.num_slots * self.kv_cache_dim * self.store_dtype.itemsize
        )
        self.index_layer_stride = (
            self.num_slots * self.index_head_dim * self.index_k_dtype.itemsize
        )
        self.kv_buffer_bytes = self.layer_num * self.kv_layer_stride
        self.index_k_bytes = self.num_full_layers * self.index_layer_stride

        self.kv_offset = _HEADER_SIZE
        self.index_k_offset = _align_up(self.kv_offset + self.kv_buffer_bytes)
        self.total_bytes = _align_up(self.index_k_offset + self.index_k_bytes)

        self.name = (
            f"sgkv{self.group_size}_{instance_key}"
            f"_p{self.pp_rank}_{self.dp_rank}_{self.group_base}_{self.role}"
            f"_v{self.layout_version}"
        )

    def header_fields(self, numa_node: int) -> Dict[str, Any]:
        return {
            "magic": _MAGIC,
            "version": _HEADER_VERSION,
            "header_size": _HEADER_SIZE,
            "group_size": self.group_size,
            "group_base": self.group_base,
            "dp_rank": self.dp_rank,
            "pp_rank": self.pp_rank,
            "pp_size": self.pp_size,
            "start_layer": self.start_layer,
            "layout_version": self.layout_version,
            "total_bytes": self.total_bytes,
            "kv_buffer_bytes": self.kv_buffer_bytes,
            "index_k_bytes": self.index_k_bytes,
            "num_slots": self.num_slots,
            "page_size": self.page_size,
            "layer_num": self.layer_num,
            "kv_cache_dim": self.kv_cache_dim,
            "index_head_dim": self.index_head_dim,
            "num_full_layers": self.num_full_layers,
            "store_dtype_code": self.store_dtype_code,
            "index_k_dtype_code": self.index_k_dtype_code,
            "numa_node": int(numa_node),
        }

    def __repr__(self) -> str:
        return (
            "KvShareLayout("
            f"name={self.name}, role={self.role}, total_bytes={self.total_bytes}, "
            f"kv_bytes={self.kv_buffer_bytes}, index_k_bytes={self.index_k_bytes}, "
            f"num_slots={self.num_slots}, layers={self.layer_num}, "
            f"full_layers={self.num_full_layers})"
        )


class SharedKvStorage:
    """The segment plus the tensors that alias it, for one rank.

    ``kv_views`` / ``index_k_views`` are installed into the pool in place of the
    private ``torch.zeros`` buffers. The segment is kept alive in
    ``_LIVE_SEGMENTS`` for the whole process.
    """

    __slots__ = (
        "layout",
        "shm",
        "kv_views",
        "index_k_views",
        "is_leader",
        "group_size",
        "group_base",
        "intra_group_rank",
        "dp_rank",
        "numa_node",
    )

    def __init__(
        self,
        layout: KvShareLayout,
        shm: shared_memory.SharedMemory,
        kv_views: List[torch.Tensor],
        index_k_views: List[Optional[torch.Tensor]],
        is_leader: bool,
        group_size: int,
        group_base: int,
        intra_group_rank: int,
        dp_rank: int,
        numa_node: int,
    ) -> None:
        self.layout = layout
        self.shm = shm
        self.kv_views = kv_views
        self.index_k_views = index_k_views
        self.is_leader = is_leader
        self.group_size = group_size
        self.group_base = group_base
        self.intra_group_rank = intra_group_rank
        self.dp_rank = dp_rank
        self.numa_node = numa_node


# ---------------------------------------------------------------------------
# Entry points used by the model runner / the pool
# ---------------------------------------------------------------------------


def pool_role(model_runner: Any) -> str:
    """``"draft"`` for the MTP / speculative draft worker, else ``"target"``."""
    return (
        POOL_ROLE_DRAFT
        if bool(getattr(model_runner, "is_draft_worker", False))
        else POOL_ROLE_TARGET
    )


def shared_kv_request(model_runner: Any) -> Optional[SharedKvRequest]:
    """Return a request when sharing is on, else ``None``.

    Called from ``_init_pools`` before the 920F pool is constructed. With the
    switch off this returns ``None`` and nothing at all changes.

    The process can hold TWO pools -- the target model's and the MTP / draft
    model's. They have different geometry (different layer ranges), so each gets
    its OWN segment: ``role`` is part of the segment name and of the process
    state. Opening the same role twice is a bug and is rejected.
    """
    group_size = get_kv_share_group()
    if group_size == KV_SHARE_GROUP_DISABLED:
        return None
    if group_size not in KV_SHARE_GROUP_CHOICES:
        raise SharedKvPoolError(
            f"SGLANG_KUNPENG_KV_SHARE_GROUP={group_size} is not supported; "
            f"valid values are 0 and {list(KV_SHARE_GROUP_CHOICES)}"
        )
    role = pool_role(model_runner)
    if role in _OPENED_ROLES:
        raise SharedKvPoolError(
            f"the {role} KV pool is already shared in this process "
            f"(segment {_OPENED_ROLES[role]}); a second pool with the same "
            "role would need the same segment"
        )

    tp_rank = int(getattr(model_runner, "tp_rank", 0))
    group_base = (tp_rank // group_size) * group_size
    intra_group_rank = tp_rank - group_base
    ok, reasons = validate_share_preconditions(model_runner, group_size)
    if not ok:
        raise SharedKvPoolError(
            "SGLANG_KUNPENG_KV_SHARE_GROUP=%d preconditions failed "
            "(tp_rank=%s tp_size=%s dp_rank=%s pp_rank=%s pp_size=%s "
            "start_layer=%s attn_tp_size=%s): %s"
            % (
                group_size,
                tp_rank,
                getattr(model_runner, "tp_size", 0),
                _resolve_dp_rank(model_runner),
                getattr(model_runner, "pp_rank", 0),
                getattr(model_runner, "pp_size", 1),
                getattr(model_runner, "start_layer", 0),
                _resolve_attn_tp_size(model_runner),
                ", ".join(reasons),
            )
        )

    # Only the leader allocates; the node is resolved on the leader. Followers
    # learn the leader's node from the header.
    numa_node = -1
    if intra_group_rank == 0:
        numa_node = resolve_leader_numa_node()

    return SharedKvRequest(
        model_runner=model_runner,
        role=role,
        group_size=group_size,
        group_base=group_base,
        intra_group_rank=intra_group_rank,
        dp_rank=_resolve_dp_rank(model_runner),
        pp_rank=int(getattr(model_runner, "pp_rank", 0)),
        pp_size=int(getattr(model_runner, "pp_size", 1)),
        start_layer=int(getattr(model_runner, "start_layer", 0)),
        numa_node=numa_node,
        instance_key=_instance_key(),
    )


def open_shared_kv_storage(
    request: Optional[SharedKvRequest], pool_geometry: Any
) -> SharedKvStorage:
    """Create (leader) or attach to (follower) the group segment.

    Called from ``KunpengNSATokenToKVPool._create_buffers`` INSTEAD of
    allocating the rank's private ``torch.zeros`` buffers, so with the switch on
    the process never holds a private pool at all.

    ``pool_geometry`` is the pool being constructed; only geometry attributes
    are read from it (``size`` / ``page_size`` / ``layer_num`` / ``kv_cache_dim``
    / ``store_dtype`` / ``index_head_dim`` / ``index_k_dtype`` /
    ``skip_topk_layers`` / ``device``).

    Every failure raises :class:`SharedKvPoolError`; nothing is ever silently
    downgraded to a per-rank pool.
    """
    global _ENABLED, _INITIALIZED, _IS_LEADER, _GROUP_SIZE
    global _GROUP_BASE, _INTRA_GROUP_RANK, _NUMA_NODE, _SEGMENT_BYTES

    if request is None:
        raise SharedKvPoolError("open_shared_kv_storage() called without a request")
    role = request.role
    if role in _OPENED_ROLES:
        raise SharedKvPoolError(
            f"the {role} KV pool is already open in this process "
            f"(segment {_OPENED_ROLES[role]}); a pool has one segment per role"
        )

    model_runner = request.model_runner
    ok, reasons = validate_share_preconditions(model_runner, request.group_size)
    if not ok:
        raise SharedKvPoolError(
            "preconditions failed at pool creation: " + ", ".join(reasons)
        )

    server_args = getattr(model_runner, "server_args", None)
    if server_args is not None and getattr(server_args, "enable_memory_saver", False):
        raise SharedKvPoolError(
            "SGLANG_KUNPENG_KV_SHARE_GROUP cannot be combined with "
            "--enable-memory-saver: the shm segment is not reclaimable per rank"
        )

    layout = KvShareLayout(
        pool_geometry,
        request.group_size,
        request.group_base,
        request.dp_rank,
        request.instance_key,
        pp_rank=request.pp_rank,
        pp_size=request.pp_size,
        start_layer=request.start_layer,
        role=role,
    )

    is_leader = request.is_leader
    created = False
    shm: Optional[shared_memory.SharedMemory] = None

    if is_leader:
        _check_capacity(layout, request.numa_node)
        created = True
        shm = _create_segment(layout)
        _CREATED_SEGMENTS.append(shm)
        _LIVE_SEGMENTS.append(shm)
        try:
            _place_on_node(shm, layout, request.numa_node)
            _publish_header(shm, layout, request.numa_node)
        except BaseException:
            _release_segment(shm, created=True)
            raise
        numa_node = request.numa_node
        logger.info(
            "[kv-share] leader created %s (%s pool): %.2f GiB, %d layers "
            "(%d full), %d slots, numa_node=%d (strict membind, prefaulted, "
            "verified)",
            layout.name,
            role,
            layout.total_bytes / 2**30,
            layout.layer_num,
            layout.num_full_layers,
            layout.num_slots,
            numa_node,
        )
    else:
        shm = _attach_segment(layout)
        _LIVE_SEGMENTS.append(shm)
        numa_node = _read_header(shm)["numa_node"]
        logger.info(
            "[kv-share] follower tp_rank=%s attached %s (%s pool, %.2f GiB, "
            "leader numa_node=%d)",
            getattr(model_runner, "tp_rank", "?"),
            layout.name,
            role,
            layout.total_bytes / 2**30,
            numa_node,
        )

    try:
        kv_views, index_views = _build_shared_views(shm, layout)
    except BaseException:
        _release_segment(shm, created=created)
        raise

    _ENABLED = True
    _IS_LEADER = is_leader
    _GROUP_SIZE = request.group_size
    _GROUP_BASE = request.group_base
    _INTRA_GROUP_RANK = request.intra_group_rank
    _NUMA_NODE = int(numa_node)
    _SEGMENT_BYTES += layout.total_bytes
    _OPENED_ROLES[role] = layout.name
    _INITIALIZED = True

    return SharedKvStorage(
        layout=layout,
        shm=shm,
        kv_views=kv_views,
        index_k_views=index_views,
        is_leader=is_leader,
        group_size=request.group_size,
        group_base=request.group_base,
        intra_group_rank=request.intra_group_rank,
        dp_rank=request.dp_rank,
        numa_node=int(numa_node),
    )


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def validate_share_preconditions(
    model_runner: Any, group_size: int = 0
) -> Tuple[bool, List[str]]:
    """Legality checks; a non-empty ``reasons`` list is always fatal.

    The caller raises :class:`SharedKvPoolError`; there is no baseline fallback.
    PP (``PP_SIZE > 1``) is allowed as long as the whole group stays inside one
    PP stage; the definitive cross-process guarantee is the header, since a
    process can only observe its own ``pp_rank``.
    """
    reasons: List[str] = []
    if group_size <= 0:
        group_size = get_kv_share_group()

    tp_rank = int(getattr(model_runner, "tp_rank", 0))
    tp_size = int(getattr(model_runner, "tp_size", 0))
    pp_rank = int(getattr(model_runner, "pp_rank", 0))
    pp_size = int(getattr(model_runner, "pp_size", 1))
    start_layer = int(getattr(model_runner, "start_layer", 0))
    attn_tp_size = _resolve_attn_tp_size(model_runner)
    group_base = (tp_rank // group_size) * group_size if group_size > 0 else 0

    def _ctx() -> str:
        return (
            f"[pp_rank={pp_rank},pp_size={pp_size},start_layer={start_layer},"
            f"group_base={group_base},tp_rank={tp_rank},tp_size={tp_size},"
            f"attn_tp_size={attn_tp_size},group_size={group_size}]"
        )

    if group_size not in KV_SHARE_GROUP_CHOICES:
        reasons.append(f"group_size({group_size})_not_supported {_ctx()}")
    if not _is_920f_host():
        reasons.append("not_cpu_920f " + _ctx())
    if attn_tp_size <= 0 or attn_tp_size % group_size != 0:
        reasons.append(
            f"attn_tp_size({attn_tp_size})_mod_group_size({group_size})_ne_0 "
            + _ctx()
        )
    if attn_tp_size > 0 and (
        group_base // attn_tp_size
        != (group_base + group_size - 1) // attn_tp_size
    ):
        # The whole group must belong to the SAME attn-TP group, otherwise its
        # KV contents are fundamentally different.
        reasons.append("group_crosses_attn_tp_boundary " + _ctx())
    if tp_size > 0 and tp_size % group_size != 0:
        reasons.append("group_crosses_tp_boundary " + _ctx())

    stage_span = _resolve_pp_stage_span(pp_size, tp_size)
    if stage_span is not None and stage_span % group_size != 0:
        reasons.append(
            f"group_crosses_pp_stage:stage_span({stage_span})"
            f"%group_size({group_size})!=0 " + _ctx()
        )

    return (not reasons), reasons


def _resolve_pp_stage_span(pp_size: int, tp_size: int) -> Optional[int]:
    """Ranks per PP stage, or ``None`` when PP is off / not applicable."""
    if pp_size <= 1:
        return None
    layout = os.environ.get("SGLANG_KUNPENG_PP_LAYOUT", "").strip().lower()
    local_world = os.environ.get("LOCAL_WORLD_SIZE", "").strip()
    if layout in ("1", "interleave") and local_world.isdigit():
        world = int(local_world)
        if world > 0 and world % pp_size == 0:
            return world // pp_size
    return tp_size


def _resolve_attn_tp_size(model_runner: Any) -> int:
    try:
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        return int(get_attention_tp_size())
    except Exception:
        pass
    tp_size = int(getattr(model_runner, "tp_size", 0) or 0)
    dp_size = int(getattr(model_runner, "dp_size", 1) or 1)
    if tp_size <= 0:
        return 0
    return max(1, tp_size // max(1, dp_size))


def _resolve_dp_rank(model_runner: Any) -> int:
    try:
        from sglang.srt.layers.dp_attention import get_attention_dp_rank

        return int(get_attention_dp_rank())
    except Exception:
        return int(getattr(model_runner, "dp_rank", 0) or 0)


def _is_920f_host() -> bool:
    try:
        from sglang.srt.utils.common import is_cpu_920f

        return bool(is_cpu_920f())
    except Exception:
        return False


def _instance_key() -> str:
    """Identifier of THIS run; isolates co-located instances on one host."""
    raw = "_".join(
        part
        for part in (
            os.environ.get("MASTER_PORT", ""),
            os.environ.get("LOG_DATE", ""),
            os.environ.get("LOG_TIME", ""),
        )
        if part
    )
    if not raw:
        raw = f"ppid{os.getppid()}"
    key = re.sub(r"[^0-9A-Za-z]", "", raw)[:32]
    return key or "0"


# ---------------------------------------------------------------------------
# NUMA: resolve the leader's node, bind, place and verify
# ---------------------------------------------------------------------------


def resolve_leader_numa_node() -> int:
    """NUMA node the leader allocates on: the one its CPUs are bound to.

    Rank and NUMA node are 1:1 on 920F, so the process' CPU affinity is the
    single source of truth -- there is deliberately no env override, so a
    misconfiguration cannot silently place the segment on somebody else's node.
    The affinity must resolve to exactly ONE node; an unbound / multi-node
    process is an error rather than "pick some node".
    """
    try:
        from sglang.srt.utils.numa_utils import get_current_numa_node

        node = get_current_numa_node()
    except Exception as exc:
        raise SharedKvPoolError(
            "cannot import the NUMA helpers (numa_utils); the shared KV pool "
            f"needs libnuma to place pages: {exc}"
        ) from exc
    if node is None:
        raise SharedKvPoolError(
            "cannot determine a single NUMA node for this rank (libnuma "
            "unavailable, or the CPU affinity spans several nodes). Launch the "
            "rank pinned to its node (numactl --cpunodebind=N --membind=N)"
        )
    return node


def _check_capacity(layout: KvShareLayout, numa_node: int) -> None:
    """Fail before touching memory: neither the node nor /dev/shm may be small."""
    node_free = _node_free_bytes(numa_node)
    if node_free is not None and layout.total_bytes > node_free:
        raise SharedKvPoolError(
            f"the shared KV segment needs {layout.total_bytes / 2**30:.1f} GiB "
            f"but NUMA node {numa_node} has only {node_free / 2**30:.1f} GiB "
            f"free. Lower --mem-fraction-static / --max-total-tokens, or "
            f"disable SGLANG_KUNPENG_KV_SHARE_GROUP"
        )
    shm_free = _dev_shm_free_bytes()
    if shm_free is not None and layout.total_bytes > shm_free:
        raise SharedKvPoolError(
            f"the shared KV segment needs {layout.total_bytes / 2**30:.1f} GiB "
            f"but /dev/shm has only {shm_free / 2**30:.1f} GiB free; remount "
            f"/dev/shm larger, remove leaked sgkv* segments, or disable "
            f"SGLANG_KUNPENG_KV_SHARE_GROUP"
        )


def _node_free_bytes(node: int) -> Optional[int]:
    try:
        from sglang.srt.utils.numa_utils import get_node_free_bytes

        return get_node_free_bytes(node)
    except Exception:
        return None


def _place_on_node(
    shm: shared_memory.SharedMemory, layout: KvShareLayout, numa_node: int
) -> None:
    """Bind the calling thread to ``numa_node``, fault every page, restore.

    The bind is STRICT (``MPOL_BIND``) and its failure is fatal: a segment that
    silently landed on another node would look correct but read remotely.
    """
    try:
        from sglang.srt.utils.numa_utils import (
            set_local_memory_policy,
            set_membind_memory_policy,
        )
    except Exception as exc:
        raise SharedKvPoolError(f"cannot import the NUMA helpers: {exc}") from exc

    if not set_membind_memory_policy(numa_node):
        raise SharedKvPoolError(
            f"numa_set_membind(node={numa_node}) failed; the leader cannot "
            "guarantee that the shared KV pages live on its own NUMA node"
        )
    started = time.monotonic()
    try:
        _prefault_region(shm, layout.total_bytes)
    finally:
        # Back to first-touch for everything this thread allocates later.
        set_local_memory_policy()
    elapsed = time.monotonic() - started
    pages = _verify_placement(shm, layout, numa_node)
    logger.info(
        "[kv-share] prefaulted %.2f GiB in %.1fs; numa_maps: %s (target node "
        "%d, ratio %.4f)",
        layout.total_bytes / 2**30,
        elapsed,
        {node: count * 4096 for node, count in sorted(pages.items())},
        numa_node,
        _placement_ratio(pages, numa_node),
    )


def _prefault_region(shm: shared_memory.SharedMemory, total_bytes: int) -> None:
    """Write every page so it is allocated now, under the current policy."""
    addr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
    try:
        ctypes.memset(ctypes.c_void_p(addr), 0, total_bytes)
    except Exception as exc:  # pragma: no cover - depends on the host
        raise SharedKvPoolError(
            f"pre-faulting {total_bytes} bytes of {shm.name} failed: {exc}"
        ) from exc


def _verify_placement(
    shm: shared_memory.SharedMemory, layout: KvShareLayout, numa_node: int
) -> Dict[int, int]:
    """Assert the pages landed on ``numa_node``; raise otherwise.

    ``/proc/self/numa_maps`` is the only reliable way to confirm that a memory
    policy actually took effect for the pages of an mmap'ed tmpfs file.
    """
    pages = _numa_maps_pages(shm.name)
    if not pages:
        raise SharedKvPoolError(
            f"cannot find the numa_maps entry of /dev/shm/{shm.name}; cannot "
            f"verify that the shared KV pages were placed on NUMA node "
            f"{numa_node}. Refusing to start unverified"
        )
    ratio = _placement_ratio(pages, numa_node)
    if ratio < _PLACEMENT_MIN_RATIO:
        raise SharedKvPoolError(
            f"only {ratio:.2%} of the shared KV pages are on NUMA node "
            f"{numa_node} (expected >= {_PLACEMENT_MIN_RATIO:.0%}); pages per "
            f"node: { {n: c for n, c in sorted(pages.items())} }. The memory "
            "policy did not take effect -- refusing to start"
        )
    return pages


def _placement_ratio(pages: Dict[int, int], numa_node: int) -> float:
    total = sum(pages.values())
    if total <= 0:
        return 0.0
    return pages.get(numa_node, 0) / total


def _numa_maps_pages(shm_name: str) -> Dict[int, int]:
    """Per-node page counts of ``/dev/shm/<shm_name>`` in our own numa_maps."""
    target = f"file=/dev/shm/{shm_name}"
    pages: Dict[int, int] = {}
    try:
        with open("/proc/self/numa_maps", "r") as handle:
            for line in handle:
                if target not in line:
                    continue
                for token in line.split()[2:]:
                    if token.startswith("N") and "=" in token:
                        node_str, _, count = token[1:].partition("=")
                        if node_str.isdigit() and count.isdigit():
                            node = int(node_str)
                            pages[node] = pages.get(node, 0) + int(count)
    except OSError:
        return {}
    return pages


def _dev_shm_free_bytes() -> Optional[int]:
    try:
        stat = os.statvfs("/dev/shm")
    except OSError:
        return None
    return stat.f_bavail * stat.f_frsize


# ---------------------------------------------------------------------------
# Segment create / attach / release
# ---------------------------------------------------------------------------


def _create_segment(layout: KvShareLayout) -> shared_memory.SharedMemory:
    """Leader: create the segment. The header is UNPUBLISHED until later."""
    try:
        shm = shared_memory.SharedMemory(
            name=layout.name, create=True, size=layout.total_bytes
        )
    except FileExistsError:
        _remove_stale_segment(layout.name)
        try:
            shm = shared_memory.SharedMemory(
                name=layout.name, create=True, size=layout.total_bytes
            )
        except Exception as exc:
            raise SharedKvPoolError(
                f"cannot create the shared KV segment {layout.name} even after "
                f"removing the stale one: {exc}"
            ) from exc
    except Exception as exc:
        raise SharedKvPoolError(
            f"cannot create the shared KV segment {layout.name} "
            f"({layout.total_bytes} bytes): {exc}"
        ) from exc
    return shm


def _remove_stale_segment(name: str) -> None:
    """Best-effort unlink of a leftover segment with the same name."""
    shm = None
    try:
        with _untracked_shared_memory():
            shm = shared_memory.SharedMemory(name=name)
        if len(shm.buf) >= _READY_OFFSET + 4:
            # Clear ``ready`` first: a follower that opens the segment between
            # these two steps must not attach to a half-dead segment.
            shm.buf[_READY_OFFSET : _READY_OFFSET + 4] = struct.pack("<I", 0)
        shm.close()
        shm.unlink()
        logger.warning("[kv-share] removed stale segment %s", name)
    except FileNotFoundError:
        return
    except Exception as exc:
        logger.warning("[kv-share] failed to remove stale segment %s: %s", name, exc)
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass


def _attach_segment(layout: KvShareLayout) -> shared_memory.SharedMemory:
    """Follower: wait for the leader's segment and validate its header."""
    deadline = time.monotonic() + _ATTACH_TIMEOUT_S
    last_error = "segment_not_created"
    warned = False

    while True:
        shm: Optional[shared_memory.SharedMemory] = None
        try:
            with _untracked_shared_memory():
                shm = shared_memory.SharedMemory(name=layout.name)
        except FileNotFoundError:
            shm = None
        except Exception as exc:
            last_error = f"attach_exception:{exc}"

        if shm is not None:
            try:
                error = _validate_header(shm, layout)
            except Exception as exc:
                error = f"header_validation_exception:{exc}"
            if error is None:
                return shm
            last_error = error
            shm.close()
            if not error.startswith(_RETRYABLE_ERROR_PREFIXES):
                raise SharedKvPoolError(
                    f"refusing segment {layout.name}: {error}. Every rank of a "
                    "group must build an identical pool; fix the launch or "
                    "disable SGLANG_KUNPENG_KV_SHARE_GROUP"
                )

        if not warned:
            logger.info(
                "[kv-share] waiting for the group leader to publish %s ...",
                layout.name,
            )
            warned = True
        if time.monotonic() >= deadline:
            raise SharedKvPoolError(
                f"timed out after {_ATTACH_TIMEOUT_S:.0f}s waiting for the group "
                f"leader's segment {layout.name} (last error: {last_error})"
            )
        time.sleep(_ATTACH_POLL_S)


def _release_segment(shm: shared_memory.SharedMemory, created: bool) -> None:
    for registry in (_LIVE_SEGMENTS, _CREATED_SEGMENTS):
        if shm in registry:
            registry.remove(shm)
    try:
        shm.close()
    except Exception:
        pass
    if created:
        try:
            shm.unlink()
        except Exception:
            pass


@contextmanager
def _untracked_shared_memory():
    """Context manager disabling the resource_tracker for ONE SharedMemory.

    Python tracks every attached segment and would unlink a segment it did not
    create at interpreter exit. Same workaround as
    ``distributed/parallel_state.py:2855-2861``.
    """
    import multiprocessing.resource_tracker as resource_tracker

    original = resource_tracker.register
    resource_tracker.register = lambda *args, **kwargs: None
    try:
        yield
    finally:
        resource_tracker.register = original


@atexit.register
def _unlink_created_segments() -> None:
    for shm in list(_CREATED_SEGMENTS):
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def _pack_header(values: Dict[str, Any]) -> bytes:
    """Serialise the header, computing the CRC over everything before it."""
    ordered: List[Any] = []
    for name, _ in _HEADER_FIELDS:
        # ``ready`` is written separately (last) by ``_publish_header``.
        ordered.append(0 if name in ("checksum", "ready") else values[name])
    payload = struct.pack(_HEADER_FORMAT, *ordered)
    ordered[_CHECKSUM_IDX] = zlib.crc32(payload[:_CHECKSUM_OFFSET]) & 0xFFFFFFFF
    return struct.pack(_HEADER_FORMAT, *ordered)


def _read_header(shm: shared_memory.SharedMemory) -> Dict[str, Any]:
    raw = bytes(shm.buf[:_HEADER_USED_BYTES])
    return dict(zip(_HEADER_NAMES, struct.unpack(_HEADER_FORMAT, raw)))


def _publish_header(
    shm: shared_memory.SharedMemory, layout: KvShareLayout, numa_node: int
) -> None:
    """Write every field, then ``ready`` LAST so followers never read garbage."""
    payload = _pack_header(layout.header_fields(numa_node))
    shm.buf[0 : len(payload)] = payload
    shm.buf[_READY_OFFSET : _READY_OFFSET + 4] = struct.pack("<I", 1)


def _validate_header(
    shm: shared_memory.SharedMemory, layout: KvShareLayout
) -> Optional[str]:
    """Return ``None`` when the header matches, else a reason string.

    ``ready`` is published last by the leader, so "not ready" simply means the
    leader is still pre-faulting; every other mismatch (version, CRC, geometry,
    PP stage, dtype) is FATAL -- retrying cannot fix it, and attaching to the
    wrong segment would make this rank read another stage's KV.
    """
    values = _read_header(shm)
    if values["ready"] != 1:
        return "not_ready"
    if values["magic"] != _MAGIC:
        return f"bad_magic:{values['magic']!r}"
    if values["version"] != _HEADER_VERSION:
        return f"version_mismatch:{values['version']}!={_HEADER_VERSION}"
    if values["header_size"] != _HEADER_SIZE:
        return f"header_size_mismatch:{values['header_size']}!={_HEADER_SIZE}"
    if values["layout_version"] != _LAYOUT_VERSION:
        return (
            f"layout_version_mismatch:{values['layout_version']}"
            f"!={_LAYOUT_VERSION}"
        )
    checksum = zlib.crc32(bytes(shm.buf[:_CHECKSUM_OFFSET])) & 0xFFFFFFFF
    if checksum != values["checksum"]:
        return f"checksum_mismatch:{checksum}!={values['checksum']}"

    expected = layout.header_fields(values["numa_node"])
    # Semantic fields first, then the derived byte sizes: the first mismatch is
    # what the operator will read in the log.
    for name in _HEADER_COMPARE_ORDER:
        if values[name] != expected[name]:
            return f"{name}_mismatch:{values[name]}!={expected[name]}"
    return None


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def _build_shared_views(
    shm: shared_memory.SharedMemory, layout: KvShareLayout
) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]:
    kv_views: List[torch.Tensor] = []
    for layer_idx in range(layout.layer_num):
        shape = (layout.num_slots, 1, layout.kv_cache_dim)
        offset = layout.kv_offset + layer_idx * layout.kv_layer_stride
        view = _make_view(shm, offset, shape, layout.store_dtype)
        _assert_view_geometry(
            view, shape, layout.store_dtype, layout.device, f"kv_buffer[{layer_idx}]"
        )
        kv_views.append(view)

    index_views: List[Optional[torch.Tensor]] = []
    full_idx = 0
    for layer_idx, skip in enumerate(layout.skip_topk_layers):
        if skip:
            # Shared (skip-topk) layers have no index-K buffer in the baseline
            # either; keep the same ``None`` placeholder so the list stays
            # layer-aligned for get_state_buf_infos().
            index_views.append(None)
            continue
        shape = (layout.num_slots, layout.index_head_dim)
        offset = layout.index_k_offset + full_idx * layout.index_layer_stride
        view = _make_view(shm, offset, shape, layout.index_k_dtype)
        _assert_view_geometry(
            view,
            shape,
            layout.index_k_dtype,
            layout.device,
            f"index_k_buffer[{layer_idx}]",
        )
        index_views.append(view)
        full_idx += 1
    return kv_views, index_views


def _make_view(
    shm: shared_memory.SharedMemory,
    offset: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    """``torch.frombuffer`` view with exactly the requested shape.

    Mirrors the CPU shm pattern of ``managers/cache_controller.py``. The
    returned tensor aliases ``shm.buf``; the segment is kept alive for the whole
    process so ``data_ptr()`` never changes (graph capture invariant).
    """
    count = 1
    for dim in shape:
        count *= int(dim)
    tensor = torch.frombuffer(shm.buf, dtype=dtype, count=count, offset=int(offset))
    return tensor.view(shape)


def _assert_view_geometry(
    view: torch.Tensor,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: Any,
    label: str,
) -> None:
    """The shared view must be indistinguishable from the private buffer."""
    if tuple(view.shape) != tuple(shape):
        raise SharedKvPoolError(f"{label}: shape {tuple(view.shape)} != {shape}")
    if view.dtype != dtype:
        raise SharedKvPoolError(f"{label}: dtype {view.dtype} != {dtype}")
    if tuple(view.stride()) != tuple(_contiguous_stride(shape)):
        raise SharedKvPoolError(
            f"{label}: stride {tuple(view.stride())} is not contiguous"
        )
    if str(view.device) != str(device):
        raise SharedKvPoolError(f"{label}: device {view.device} != {device}")
    expected_numel = 1
    for dim in shape:
        expected_numel *= int(dim)
    if view.numel() != expected_numel:
        raise SharedKvPoolError(
            f"{label}: numel {view.numel()} != {expected_numel}"
        )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _align_up(value: int, alignment: int = _HEADER_SIZE) -> int:
    return (value + alignment - 1) // alignment * alignment


def _contiguous_stride(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    stride = [1] * len(shape)
    acc = 1
    for idx in range(len(shape) - 1, -1, -1):
        stride[idx] = acc
        acc *= int(shape[idx])
    return tuple(stride)


def _dtype_to_code(dtype: torch.dtype) -> int:
    for attr, code in _DTYPE_CANDIDATES:
        candidate = getattr(torch, attr, None)
        if candidate is not None and dtype == candidate:
            return code
    return 0
