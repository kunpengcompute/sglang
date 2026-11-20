import hashlib
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Set, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.utils import is_cpu_920f

logger = logging.getLogger(__name__)


@dataclass
class HiCacheStorageConfig:
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    attn_cp_rank: int
    attn_cp_size: int
    is_mla_model: bool
    enable_storage_metrics: bool
    is_page_first_layout: bool
    model_name: Optional[str]
    tp_lcm_size: Optional[int] = None
    should_split_heads: bool = False
    extra_config: Optional[dict] = None
    # Topology-free page-blob identity (see flat_kv_key_digest). Set by the
    # two-tier path so every PP/TP layout writes the same file names.
    key_digest: Optional[str] = None


def flat_kv_key_digest(
    model_name: Optional[str],
    dtype: torch.dtype,
    page_size: int,
    kv_dim: int,
    index_head_dim: int = 0,
) -> str:
    """Topology-free identity of a page blob: what it holds, not who wrote it.

    No tp/pp/cp rank enters the digest, so every PP (and TP) layout derives the
    same file names and can share one pool. Layer counts are left out on
    purpose: a mismatch already shows up as a wrong object size, i.e. a miss.
    """
    raw = "|".join(
        [
            model_name or "",
            str(dtype).replace("torch.", ""),
            str(int(page_size)),
            str(int(kv_dim)),
            str(int(index_head_dim)),
        ]
    )
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


@dataclass
class HiCacheStorageExtraInfo:
    prefix_keys: Optional[List[str]] = (None,)
    extra_info: Optional[dict] = None


# Width of the ``.stages`` sidecar, in bytes, one bit per PP stage. It is a
# constant (not ``layout.writers``) so a PP1 process and a PP16 process agree on
# the same completion record: PP1 writes all slots, PP16 each writes its own.
L3_STAGE_SLOTS = 16


@dataclass(frozen=True)
class FlatPageLayout:
    """Where this PP stage's bytes live inside a whole-page L3 object.

    PP1 and PP16 must read and write the same files, so the object keeps the
    PP1 byte layout: the target layers in layer order, then the MTP draft
    layers. Every stage writes only the slice it owns, at the offset a PP1
    process would have put that slice at, so either layout can read what the
    other wrote.

    ``segments`` are ``(buffer_offset, file_offset, length)`` byte triples,
    which lets the target and the draft slices be discontiguous in both the
    flat buffer and the object. ``writer``/``writers`` index the byte of the
    sidecar ``.stages`` marker that records this stage's write, over the stages
    that actually contribute bytes to this object (a stage whose layer range
    has no index-K blob does not take part in the index-K marker).
    """

    segments: Tuple[Tuple[int, int, int], ...]
    total_bytes: int
    writer: int = 0
    writers: int = 1


class PoolName(str, Enum):
    """Well-known pool names used as PoolTransfer/PoolEntry identifiers."""

    KV = "kv"
    MAMBA = "mamba"
    INDEXER = "indexer"

    def __str__(self) -> str:
        return self.value


class PoolHitPolicy(str, Enum):
    """Hit policy for batch_exists_v2 per-pool prefix matching.

    ALL_PAGES      : every page in [0, kv_hit) must exist (e.g. DSA).
    TRAILING_PAGES : only the last N pages must exist (e.g. Mamba/SWA states).
    """

    ALL_PAGES = "all_pages"
    TRAILING_PAGES = "trailing_pages"


@dataclass
class PoolTransfer:
    """Unified per-pool transfer descriptor for batch v2 interface.

    device<->host path : host_indices + device_indices
    host<->storage path: host_indices + keys
    """

    name: PoolName
    host_indices: Optional[torch.Tensor] = None
    device_indices: Optional[torch.Tensor] = None
    keys: Optional[List[str]] = None
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES


@dataclass
class PoolTransferResult:
    """Tracks how many pages were successfully processed per pool."""

    kv_hit_pages: int
    extra_pool_hit_pages: dict[str, int]

    @classmethod
    def empty(cls) -> "PoolTransferResult":
        return cls(0, {})

    def update_kv_hit_pages(self, kv_hit_pages: int) -> None:
        """Accumulate kv_hit_pages across batches (max = last successful batch)."""
        self.kv_hit_pages = max(self.kv_hit_pages, kv_hit_pages)

    def update_extra_pool_hit_pages(self, results: dict[str, List[bool]]) -> None:
        """Record actual load/write success counts per extra pool."""
        self.extra_pool_hit_pages.update(
            {name: sum(rs) for name, rs in results.items()}
        )


class HiCacheStorage(ABC):
    """
    HiCacheStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
    It abstracts the underlying storage mechanism, allowing different implementations to be used.
    """

    # todo, the page size of storage backend does not have to be the same as the same as host memory pool
    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        self.mem_pool_host = mem_pool_host

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        if not hasattr(self, "registered_pools"):
            self.registered_pools = {}
        self.registered_pools[host_pool_name] = host_pool

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        """Check which cache pages exist in storage, respecting per-pool hit policies.

        Longest-prefix semantics
        Extra-pool hit policies (``PoolTransfer.hit_policy``)
        ------------------------------------------------------
        Each ``PoolTransfer`` in ``pool_transfers`` describes a secondary
        cache pool (e.g. Mamba SSM states) that must be co-present with the
        KV pages.  The final ``final_pages`` is the minimum across all pools,
        so a missing auxiliary page shrinks the usable prefix.

        - ``"all_pages"`` (default):  every page in [0, kv_hit) must exist
          for this pool.  Used for pools that are required for every token
          in the prefix (e.g. DeepSeek DSA pool).

        - ``"trailing_pages"``:  only the *last* ``len(transfer.keys)`` pages
          of the KV prefix need to exist.  Used for pools whose data covers
          only the tail of a prefix (e.g. Mamba/SWA Pool).

        Returns
        -------
        PoolTransferResult
            ``kv_hit_pages`` = length of the usable KV prefix.
            ``extra_pool_hit_pages`` maps each pool name to the number of pages
            that were found.
        """
        raise NotImplementedError()

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> dict[str, List[bool]]:
        """Read data from storage into host memory for each PoolTransfer.

        Returns a dict mapping pool name to a per-entry success list.
        """
        raise NotImplementedError()

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> dict[str, List[bool]]:
        """Write data from host memory to storage for each PoolTransfer.

        Returns a dict mapping pool name to a per-entry success list.
        """
        raise NotImplementedError()

    def supports_coalesced_pages(self) -> bool:
        """True when the backend keeps a page's draft KV inside the page object.

        Page-oriented backends that pay a high per-object cost (shared
        filesystems charge an open/close round trip per page) can store the
        draft blob in the same object as the target blob, halving the object
        count per prefetched page. Callers then use
        :meth:`batch_set_coalesced_pages` / :meth:`batch_get_coalesced_pages`
        instead of the separate ``"d:"``-prefixed draft keys.
        """
        return False

    def batch_set_coalesced_pages(
        self,
        keys: List[str],
        values: List[Any],
        draft_values: List[Any],
    ) -> bool:
        """Store each page with its target and draft blobs coalesced."""
        raise NotImplementedError()

    def batch_get_coalesced_pages(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        draft_locations: List[torch.Tensor],
    ) -> tuple:
        """Read each coalesced object once, filling the target and draft buffers.

        Returns ``(targets, draft_hits)``: ``targets[i]`` is the filled target
        buffer, or None when the object is missing; ``draft_hits[i]`` reports
        whether the trailing draft section was present and fully read. Objects
        written before coalescing existed carry no draft section, which is not
        an error -- the draft pool simply keeps its current contents.
        """
        raise NotImplementedError()

    def supports_batched_page_load(self) -> bool:
        """True when ``batch_get_pages_into_pools`` can be used.

        That call reads a whole batch of pages straight into the L2 pools in one
        Python -> C++ call, which matters for backends whose caller is a plain
        Python thread (HiCache's storage threads): each round trip costs it a GIL
        re-acquisition, i.e. tens of ms under load.
        """
        return False

    def batch_get_pages_into_pools(
        self,
        keys: List[str],
        target_pool,
        target_indices: torch.Tensor,
        draft_pool=None,
        draft_indices: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Read a batch of pages straight into the L2 pools; see the backends.

        Returns ``(target_hits, draft_hits)``, 0/1 per page. Only called when
        :meth:`supports_batched_page_load` is True.
        """
        raise NotImplementedError()

    def supports_flat_io(self) -> bool:
        """True when the two-tier (L1+L3) flat-blob interface is implemented.

        That mode has no L2 host pool, so ``batch_get_v1``/``batch_set_v1`` (which
        resolve their buffers through ``mem_pool_host.get_page_buffer_meta``)
        cannot be used. Backends implementing this return True from
        :meth:`supports_flat_io` and take caller-provided 1-D tensors instead;
        everything else keeps the pool-based interface (default: unsupported).
        """
        return False

    def register_io_buffer(self, buffer: torch.Tensor) -> None:
        """Register a fixed I/O buffer so zero-copy transfers may target it.

        Called once per flat I/O buffer at attach time. Backends that copy
        through the CPU (e.g. the file backend) need no registration, so the
        default is a no-op.
        """
        pass

    def set_flat_layout(
        self,
        kv_layout: Optional["FlatPageLayout"],
        idx_layout: Optional["FlatPageLayout"] = None,
        idx_key_suffix: str = "@idx",
    ) -> None:
        """Declare the shared PP page geometry used by the flat interface.

        Backends that keep one whole page per object (the file backend when
        ``SGLANG_HICACHE_FILE_PP_SHARED`` is on) use this to pwrite/pread only
        the slice of the current PP stage, so a PP1 and a PP16 process can
        share one pool. The default keeps whole-page I/O, i.e. the flat
        interface behaves exactly as before.
        """
        pass

    def batch_get_flat(
        self,
        keys: List[str],
        buffers: List[torch.Tensor],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """Read one page blob per key into the matching 1-D buffer.

        ``buffers[i]`` is a view of the caller's flat I/O buffer and is exactly
        one page blob wide; the caller guarantees it stays alive until this
        returns. Returns per-key success, same contract as :meth:`batch_get_v1`.
        Only called when :meth:`supports_flat_io` is True.
        """
        raise NotImplementedError()

    def batch_set_flat(
        self,
        keys: List[str],
        buffers: List[torch.Tensor],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """Write one page blob per key out of the matching 1-D buffer.

        Mirror of :meth:`batch_get_flat`; returns per-key success.
        """
        raise NotImplementedError()

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Retrieve values for multiple keys.
        Returns a list of booleans indicating success for each key.
        """
        pass

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Store multiple key-value pairs.
        Returns a list of booleans indicating success for each key.
        """
        pass

    @abstractmethod
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store multiple key-value pairs.
        Returns True if all operations were successful, False otherwise.
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        pass

    # TODO: Use a finer-grained return type (e.g., List[bool])
    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """
        Check if the keys exist in the storage.
        return the number of consecutive existing keys from the start.
        Can be overridden by subclasses for more efficient implementation.
        """
        for i in range(len(keys)):
            if not self.exists(keys[i]):
                return i
        return len(keys)

    def batch_exists_bits(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> List[bool]:
        """Per-page existence for the sharded L3 hit query."""
        return [self.batch_exists([key], extra_info) == 1 for key in keys]

    def clear(self) -> None:
        pass

    def get_stats(self):
        return None


class HiCacheFile(HiCacheStorage):

    def __init__(
        self, storage_config: HiCacheStorageConfig, file_path: str = "/tmp/hicache"
    ):
        self.file_path = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get() or file_path

        tp_rank, tp_size, pp_rank, pp_size, model_name, is_mla_model = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.pp_rank,
            storage_config.pp_size,
            storage_config.model_name,
            storage_config.is_mla_model,
        )
        model_name = "-".join(model_name.split("/")) if model_name else ""
        self.pp_rank = int(pp_rank)
        self.pp_size = int(pp_size)
        self.pp_shared = envs.SGLANG_HICACHE_FILE_PP_SHARED.get()
        if storage_config.key_digest and self.pp_shared:
            # Two-tier mode: the name carries only what the blob holds, so a
            # PP1 and a PP16 process (and any TP split) address the same files.
            self.config_suffix = (
                f"_{model_name}-{storage_config.key_digest}"
                if model_name
                else f"_{storage_config.key_digest}"
            )
        else:
            self.config_suffix = f"_{model_name}"
            if not is_mla_model:
                self.config_suffix += f"_{tp_rank}_{tp_size}"
            # With the shared key off, a pp>1 run keeps its private per-stage
            # files, exactly as before this change.
            if not self.pp_shared and self.pp_size > 1:
                self.config_suffix += f"_{self.pp_size}_{self.pp_rank}"
        self.kv_layout: Optional[FlatPageLayout] = None
        self.idx_layout: Optional[FlatPageLayout] = None
        self.idx_key_suffix = "@idx"
        if not os.path.exists(self.file_path) and tp_rank == 0:
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")

    def set_flat_layout(
        self,
        kv_layout: Optional[FlatPageLayout],
        idx_layout: Optional[FlatPageLayout] = None,
        idx_key_suffix: str = "@idx",
    ) -> None:
        if not self.pp_shared:
            return
        self.kv_layout = kv_layout
        self.idx_layout = idx_layout
        self.idx_key_suffix = idx_key_suffix

    def _layout_for(self, key: str) -> Optional[FlatPageLayout]:
        """Page layout of one key; ``None`` means the legacy whole-file I/O."""
        if self.kv_layout is None:
            return None
        if self.idx_layout is not None and self.idx_key_suffix in key:
            return self.idx_layout
        return self.kv_layout

    @staticmethod
    def _stage_marker_path(path: str, stage: int) -> str:
        """Per-stage completion sentinel (zero bytes) next to the page blob.

        One file per stage instead of one shared 16-byte bitmap: a shared file
        that 16 nodes pwrite into loses updates on a distributed filesystem
        (observed: the byte lands locally, the server keeps the old value), which
        leaves pages stuck at 15/16 forever. Creating a private file with
        ``O_CREAT|O_EXCL`` is atomic and cannot be undone by another node.
        """
        stem = path[:-4] if path.endswith(".bin") else path
        return f"{stem}.s{stage}"

    def _marker_slots(self, layout: FlatPageLayout) -> int:
        """How many stage sentinels this page has; ``0`` = not representable."""
        if layout.writers > L3_STAGE_SLOTS:
            return 0
        return L3_STAGE_SLOTS

    def _marker_need(self, layout: FlatPageLayout) -> int:
        """How many sentinels must exist for the page to count as complete.

        A whole-object writer (PP1) creates every sentinel, and must also
        *require* every one: otherwise a PP16 page that another stage is still
        assembling would look complete to it.
        """
        if layout.writers <= 1:
            return L3_STAGE_SLOTS
        return layout.writers

    def _marker_count(self, path: str, need: int) -> int:
        """Number of the first *need* sentinels that exist (cheap, no read)."""
        n = 0
        for i in range(max(0, need)):
            if os.path.exists(self._stage_marker_path(path, i)):
                n += 1
        return n

    def _page_ready(self, path: str, layout: FlatPageLayout) -> bool:
        """Whole page present: full size and every stage sentinel present.

        The size check alone is not enough: a stage's slice can sit at the very
        end of the object (the MTP draft layer does), so the first write already
        stretches the file to ``total_bytes`` while the rest is holes.
        """
        need = self._marker_need(layout)
        if need <= 0 or self._marker_slots(layout) <= 0:
            return False
        try:
            size = os.stat(path).st_size
        except OSError:
            return False
        if size != layout.total_bytes:
            return False
        return self._marker_count(path, need) == need

    def _wait_incomplete(
        self, path: str, layout: FlatPageLayout, budget: float = None
    ) -> bool:
        """Wait for a page that another stage is still assembling.

        A reader and a writer of the same prefix race by design: the writer's
        last stages may land seconds after a page is already readable in
        substance. Stopping the whole contiguous prefix on such a page turns a
        nearly-complete hit into a full recompute, so look at the marker first:
        ``0 < set < need`` means "in progress" (retry for a bounded time), while
        a missing marker or zero slots means "not started" (do not wait).
        """
        need = self._marker_need(layout)
        if need <= 0 or self._marker_slots(layout) <= 0:
            return False
        filled = self._marker_count(path, need)
        if filled == 0 or filled >= need:
            return False
        if budget is None:
            budget = float(envs.SGLANG_HICACHE_INCOMPLETE_WAIT.get())
        deadline = time.monotonic() + budget
        while time.monotonic() < deadline:
            time.sleep(0.05)
            if self._page_ready(path, layout):
                return True
        logger.warning(
            "[hicache] L3 page %s stayed incomplete for %.1fs (%d/%d stages); "
            "treating it as a miss",
            os.path.basename(path),
            budget,
            filled,
            need,
        )
        return False

    def _read_slices(
        self, path: str, buffer: torch.Tensor, layout: FlatPageLayout
    ) -> bool:
        """Read each segment at its file offset; a short read means a miss."""
        buf = memoryview(buffer.view(torch.uint8).contiguous().numpy())
        with open(path, "rb", buffering=0) as f:
            for src, dst, length in layout.segments:
                f.seek(dst)
                if f.readinto(buf[src : src + length]) != length:
                    return False
        return True

    @staticmethod
    def _create_sentinel(path: str, stage: int) -> bool:
        """Atomically record that *stage* finished this page.

        ``O_CREAT|O_EXCL`` makes "create once" a single atomic operation, so two
        nodes racing here can never lose each other's record (which is what a
        shared bitmap does on a distributed filesystem).
        """
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644))
            return True
        except FileExistsError:
            return True
        except OSError as e:
            logger.warning("[hicache] sentinel create failed for %s: %s", path, e)
            return False

    def _write_slices(
        self, path: str, buffer: torch.Tensor, layout: FlatPageLayout
    ) -> bool:
        slots = self._marker_slots(layout)
        need = self._marker_need(layout)
        if slots <= 0 or need <= 0:
            logger.warning(
                "[hicache] L3 page %s needs %d PP stage slots but only %d are "
                "supported; skipping write",
                os.path.basename(path),
                layout.writers,
                L3_STAGE_SLOTS,
            )
            return False
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1
        if size > layout.total_bytes:
            # Another writer used a different page geometry. Truncating would
            # wipe the slices it already wrote, so leave the object alone.
            logger.warning(
                "[hicache] L3 page %s is larger than the shared layout "
                "(%d > %d bytes), skipping write",
                os.path.basename(path),
                size,
                layout.total_bytes,
            )
            return False
        if size == layout.total_bytes and self._marker_count(path, need) == need:
            # Genuinely complete (readiness uses the same predicate), so there
            # is nothing to add. Size alone never gets here.
            return True
        data = memoryview(buffer.view(torch.uint8).contiguous().numpy())
        fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            for src, dst, length in layout.segments:
                view = data[src : src + length]
                if len(view) < length:
                    logger.warning(
                        "[hicache] L3 page %s buffer shorter than its slice "
                        "(%d < %d), skipping write",
                        os.path.basename(path),
                        len(view),
                        length,
                    )
                    return False
                done = 0
                while done < length:
                    n = os.pwrite(fd, view[done:], dst + done)
                    if n <= 0:
                        logger.warning(
                            "[hicache] L3 page %s write stalled at %d/%d bytes",
                            os.path.basename(path),
                            done,
                            length,
                        )
                        return False
                    done += n
        finally:
            os.close(fd)
        # Record this stage. Each stage owns a private sentinel file, so no
        # node can ever overwrite another stage's record: the create is atomic
        # and idempotent. PP1 covers the whole object, so it creates them all.
        if layout.writers == 1:
            ok = all(self._create_sentinel(self._stage_marker_path(path, i), 0)
                     for i in range(slots))
        else:
            ok = self._create_sentinel(
                self._stage_marker_path(path, layout.writer), 0
            )
        if not ok:
            return False
        if (
            self._marker_count(path, need) == need
            and size == layout.total_bytes
            and envs.SGLANG_HICACHE_DEBUG.get()
        ):
            logger.info(
                "[hicache] L3 page %s complete (%d/%d stages)",
                os.path.basename(path),
                need,
                need,
            )
        return True

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def _get_component_key(self, key: str, component_name: Optional[str] = None) -> str:
        if component_name is None or component_name in ("__default__", PoolName.KV):
            return self._get_suffixed_key(key)
        return self._get_suffixed_key(f"{key}.{component_name}")

    def _get_component_path(
        self, key: str, component_name: Optional[str] = None
    ) -> str:
        return os.path.join(
            self.file_path, f"{self._get_component_key(key, component_name)}.bin"
        )

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            expected = target_location.numel() * target_location.element_size()
            with open(tensor_path, "rb", buffering=0) as f:
                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())
                if f.readinto(buf) != expected:
                    raise IOError(f"Short read for {key}")
            return target_location
        except FileNotFoundError:
            logger.warning(f"Failed to fetch {key} from HiCacheFile storage.")
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        return [
            self.get(key, target_location)
            for key, target_location in zip(
                keys, target_locations or [None] * len(keys)
            )
        ]

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if self.exists(key):
            logger.debug(f"Key {key} already exists. Skipped.")
            return True

        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            value.contiguous().view(dtype=torch.uint8).numpy().tofile(tensor_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        for key, value in zip(keys, values):
            if not self.set(key, value):
                return False
        return True

    def supports_coalesced_pages(self) -> bool:
        return True

    def batch_set_coalesced_pages(
        self,
        keys: List[str],
        values: List[Any],
        draft_values: List[Any],
    ) -> bool:
        """Store each page as one ``.bin`` holding ``[target blob][draft blob]``.

        The target blob comes first, so the plain read path is unaffected:
        :meth:`get` reads exactly target-sized bytes from offset 0 and never sees
        the draft section, which keeps ``exists`` / ``batch_exists`` and the
        per-page prefix match identical to the non-coalesced layout.
        """
        for key, value, draft_value in zip(keys, values, draft_values):
            if self.exists(key):
                logger.debug(f"Key {key} already exists. Skipped.")
                continue

            key = self._get_suffixed_key(key)
            tensor_path = os.path.join(self.file_path, f"{key}.bin")
            try:
                with open(tensor_path, "wb") as f:
                    value.contiguous().view(dtype=torch.uint8).numpy().tofile(f)
                    draft_value.contiguous().view(dtype=torch.uint8).numpy().tofile(f)
            except Exception as e:
                logger.error(f"Failed to save tensor {key}: {e}")
                return False
        return True

    def batch_get_coalesced_pages(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        draft_locations: List[torch.Tensor],
    ) -> tuple:
        """Read each coalesced page object once, into both buffers.

        The single open/read pair per page is the point of the coalesced layout:
        the draft section is read from the same handle right after the target
        blob instead of from its own ``"d:"`` object.
        """
        targets = []
        draft_hits = []
        for key, target, draft in zip(keys, target_locations, draft_locations):
            tensor_path = os.path.join(
                self.file_path, f"{self._get_suffixed_key(key)}.bin"
            )
            expected = target.numel() * target.element_size()
            draft_expected = draft.numel() * draft.element_size()
            try:
                with open(tensor_path, "rb", buffering=0) as f:
                    buf = memoryview(target.view(torch.uint8).contiguous().numpy())
                    if f.readinto(buf) != expected:
                        raise IOError(f"Short read for {key}")
                    # A page stored before coalescing has no trailing section;
                    # that is not an error, the draft pool keeps its contents.
                    draft_buf = memoryview(
                        draft.view(torch.uint8).contiguous().numpy()
                    )
                    draft_hits.append(f.readinto(draft_buf) == draft_expected)
                    targets.append(target)
            except FileNotFoundError:
                logger.warning(
                    f"Failed to fetch {key} from HiCacheFile storage."
                )
                targets.append(None)
                draft_hits.append(False)
            except Exception as e:
                logger.error(f"Failed to fetch {key}: {e}")
                targets.append(None)
                draft_hits.append(False)
        return targets, draft_hits

    def supports_batched_page_load(self) -> bool:
        # The batched load kernel is Kunpeng-only; other platforms keep the
        # per-page interface above.
        return is_cpu_920f()

    def batch_get_pages_into_pools(
        self,
        keys: List[str],
        target_pool,
        target_indices: torch.Tensor,
        draft_pool=None,
        draft_indices: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Read a batch of pages and scatter them into the L2 pools in one call.

        Same data as :meth:`batch_get_coalesced_pages` plus one
        ``set_from_flat_data_page`` per page, without the per-page
        Python -> C++ round trips. ``target_indices`` follows the per-page loop's
        convention (page ``i`` owns slots ``[i * page_size, (i + 1) * page_size)``);
        ``draft_pool`` is given only for coalescing backends, where the draft blob
        sits inside the target object. Returns ``(target_hits, draft_hits)``, 0/1
        per page (0 = not in storage, or no draft section).
        """
        from sglang.srt.hardware_backend.cpu_kunpeng.hicache import (
            hicache_page_load_coalesced_batch,
        )

        paths = [self._get_component_path(key) for key in keys]
        return hicache_page_load_coalesced_batch(
            target_pool.kv_buffer,
            target_indices,
            target_pool.page_size,
            paths,
            draft_pool.kv_buffer if draft_pool is not None else None,
            draft_indices if draft_pool is not None else None,
            draft_pool.page_size if draft_pool is not None else 0,
        )

    def exists(self, key: str) -> bool:
        tensor_path = os.path.join(self.file_path, f"{self._get_suffixed_key(key)}.bin")
        layout = self._layout_for(key)
        if layout is None:
            return os.path.exists(tensor_path)
        # A page another stage is still assembling is not usable yet, and one
        # written by the other layout (PP1 vs PP16) is only usable at full size.
        return self._page_ready(tensor_path, layout)

    def batch_exists_bits(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> List[bool]:
        """One existence check per page; the sharded hit query splits these
        across ranks."""
        return [self.exists(key) for key in keys]

    def supports_flat_io(self) -> bool:
        # Only the kunpeng two-tier HiCache path uses the flat interface; same
        # capability-flag shape as supports_batched_page_load above.
        return is_cpu_920f()

    def batch_get_flat(
        self,
        keys: List[str],
        buffers: List[torch.Tensor],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        results = []
        for key, buffer in zip(keys, buffers):
            tensor_path = os.path.join(
                self.file_path, f"{self._get_suffixed_key(key)}.bin"
            )
            layout = self._layout_for(key)
            try:
                if layout is not None:
                    if not self._page_ready(tensor_path, layout) and not (
                        # The prefix is being assembled by another PP stage right
                        # now: waiting keeps a nearly-complete hit instead of
                        # recomputing the whole suffix.
                        self._wait_incomplete(tensor_path, layout)
                    ):
                        results.append(False)
                        continue
                    results.append(self._read_slices(tensor_path, buffer, layout))
                    continue
                expected = buffer.numel() * buffer.element_size()
                with open(tensor_path, "rb", buffering=0) as f:
                    buf = memoryview(buffer.view(torch.uint8).contiguous().numpy())
                    results.append(f.readinto(buf) == expected)
            except FileNotFoundError:
                logger.warning(f"Failed to fetch {key} from HiCacheFile storage.")
                results.append(False)
        return results

    def batch_set_flat(
        self,
        keys: List[str],
        buffers: List[torch.Tensor],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        results = []
        for key, buffer in zip(keys, buffers):
            tensor_path = os.path.join(
                self.file_path, f"{self._get_suffixed_key(key)}.bin"
            )
            layout = self._layout_for(key)
            try:
                if layout is not None:
                    # Writes only this stage's slices, at the offsets the PP1
                    # layout puts them at; concurrent stages fill the rest.
                    results.append(
                        self._write_slices(tensor_path, buffer, layout)
                    )
                    continue
                needed = buffer.numel() * buffer.element_size()
                # Write-once, unless the existing blob was written by an older
                # page layout (e.g. before the draft layers were coalesced).
                # Rewriting on a size mismatch keeps those prefixes usable
                # instead of turning them into permanent misses.
                try:
                    if os.path.getsize(tensor_path) == needed:
                        results.append(True)
                        continue
                except OSError:
                    pass
                buffer.contiguous().view(dtype=torch.uint8).numpy().tofile(tensor_path)
                results.append(True)
            except Exception as e:
                logger.error(f"Failed to save tensor {key}: {e}")
                results.append(False)
        return results

    def _collect_existing_component_keys(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
    ) -> Set[str]:
        target_files = {f"{self._get_component_key(key)}.bin" for key in keys}
        for transfer in pool_transfers or []:
            for key in keys:
                target_files.add(f"{self._get_component_key(key, transfer.name)}.bin")

        existing_files = set()
        with os.scandir(self.file_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name in target_files:
                    existing_files.add(entry.name)
        return existing_files

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        existing_files = self._collect_existing_component_keys(keys, pool_transfers)

        def has_component(page_idx: int, name: str) -> bool:
            return (
                f"{self._get_component_key(keys[page_idx], name)}.bin" in existing_files
            )

        # Longest contiguous KV prefix present in storage.
        kv_pages = next(
            (
                i
                for i in range(len(keys))
                if f"{self._get_component_key(keys[i])}.bin" not in existing_files
            ),
            len(keys),
        )

        hit_count: dict[str, int] = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages

        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            name = transfer.name
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = next(
                    (i for i in range(kv_pages) if not has_component(i, name)), kv_pages
                )
            else:  # trailing_pages
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                boundary = 0
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        has_component(i, name)
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break
            if boundary:
                hit_count[name] = boundary
            final_pages = min(final_pages, boundary)

        return PoolTransferResult(final_pages, hit_count)

    def _log_key(self, pool_name: str, key: str) -> str:
        return key if pool_name == PoolName.KV else f"{key}.{pool_name}"

    def _read_page(self, pool_name: str, key: str, host_pool, page_offset: int) -> bool:
        """Read one page from storage into host_pool at page_offset."""
        storage_key = self._log_key(pool_name, key)
        data_page = self.get(storage_key, host_pool.get_dummy_flat_data_page())
        if data_page is None:
            return False
        host_pool.set_from_flat_data_page(page_offset, data_page)
        return True

    def _write_page(
        self, pool_name: str, key: str, host_pool, page_offset: int
    ) -> bool:
        """Write one page from host_pool at page_offset to storage as raw bytes."""
        storage_key = self._log_key(pool_name, key)
        data_page = host_pool.get_data_page(page_offset, flat=True)
        return self.set(storage_key, data_page)

    def _batch_io_v2(self, transfers: List[PoolTransfer], op_fn):
        results: dict[str, List[bool]] = {}
        for transfer in transfers:
            host_pool = self.registered_pools[transfer.name]
            keys = transfer.keys or []
            page_size = getattr(host_pool, "page_size", 1) or 1
            expected = len(keys) * page_size
            host_indices = transfer.host_indices

            if host_indices is None or host_indices.numel() != expected:
                logger.error(
                    "%s indices length mismatch for %s: expected %s, got %s",
                    op_fn.__name__,
                    transfer.name,
                    expected,
                    host_indices.numel() if host_indices is not None else 0,
                )
                results[transfer.name] = [False] * len(keys)
                continue

            results[transfer.name] = [
                op_fn(transfer.name, key, host_pool, host_indices[i * page_size].item())
                for i, key in enumerate(keys)
            ]
        return results

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, self._read_page)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, self._write_page)

    def clear(self) -> bool:
        try:
            for filename in os.listdir(self.file_path):
                file_path = os.path.join(self.file_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cleared all entries in HiCacheFile storage.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear HiCacheFile storage: {e}")
            return False
