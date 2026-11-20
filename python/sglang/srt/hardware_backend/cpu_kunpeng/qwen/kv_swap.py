"""Qwen GQA KV staging and Draft residency, owned by each swap manager.

DDR remains authoritative. Fixed HBW mirrors and lifecycle metadata are shared
by eager execution and captured graphs; no hooks are installed by default.
"""

import atexit
import logging
import os
import threading
from dataclasses import dataclass

import torch

from sglang.srt.graph import is_capturing, ops

logger = logging.getLogger(__name__)


@dataclass
class _GQAKVSwap:
    # Keep the pool alive: target/draft may have identical layer IDs, but own
    # independent DDR, staging and events throughout every captured replay.
    owner: object
    layers: tuple
    keys: tuple
    values: tuple
    key_hbw: torch.Tensor
    value_hbw: torch.Tensor
    events: torch.Tensor
    event_count: torch.Tensor
    pages: torch.Tensor
    stats: torch.Tensor
    page_rows: int
    source_signature: tuple
    is_target: bool = False
    extra_pair: tuple = None  # optional (K,V,events,count), fixed for graph lifetime
    draft_reuse: object = None


class QwenKVSwap:
    def __init__(self, manager):
        self.manager = manager
        self.pools = {}
        # Registered after the public manager: counters are read before SDMA clear.
        atexit.register(self.log_counters)

    def log_counters(self):
        if not self.manager._sdma_initialized:
            return
        for state in self.pools.values():
            logger.info(
                "GQA SDMA final counters owner=%x layers=%s "
                "[prepares,logical_rows,pages,last_pair_bytes,total_bytes,"
                "layer_pairs,ranges,events]=%s",
                id(state.owner), state.layers, state.stats.tolist(),
            )

    @staticmethod
    def _source_signature(tensors):
        return tuple((t.data_ptr(), tuple(t.shape), t.dtype) for t in tensors)

    def register_gqa_kv_pool(self, owner, layer_ids, *, is_target=False) -> None:
        """Allocate one full K/V mirror per Qwen pool before graph planning.

        No compaction: slot/page indices retain their DDR meaning. Runtime
        graph operations copy the union of pages read by this forward. Every new KV
        row is still written synchronously to DDR by the attention caller.
        """
        if not self.manager.enable_swap_kv_in:
            return
        layers = tuple(layer_ids)
        double = is_target and (
            os.environ.get("SGLANG_KUNPENG_QWEN_GQA_DOUBLE_PAIR", "0") == "1"
        )
        if double and os.environ.get("SGLANG_KUNPENG_QWEN_PREFILL_WORKSPACE_HBM") != "0":
            raise ValueError("Target double pair requires D prefill workspace outside HBW")
        old = self.pools.get(id(owner))
        if old is not None:
            if (
                old.owner is not owner or old.layers != layers
                or old.is_target != is_target
                or (old.extra_pair is not None) != double
            ):
                raise ValueError("GQA swap pool identity/layer order changed")
            current = tuple(owner.get_key_buffer(i) for i in layers) + tuple(
                owner.get_value_buffer(i) for i in layers
            )
            signature = self._source_signature(current)
            if signature != old.source_signature:
                raise ValueError("GQA KV storage changed; recreate manager and captured graphs")
            return
        if self.manager.enable_swap_kv_blockwise:
            raise ValueError("Qwen GQA SDMA uses physical page mirrors, not compact blockwise mode")
        if not layers or len(set(layers)) != len(layers):
            raise ValueError("GQA swap needs a nonempty unique local layer order")
        keys = tuple(owner.get_key_buffer(i) for i in layers)
        values = tuple(owner.get_value_buffer(i) for i in layers)
        key, value = keys[0], values[0]
        for group, reference in ((keys, key), (values, value)):
            if any(
                t.shape != reference.shape or t.dtype != reference.dtype
                or t.device.type != "cpu" or not t.is_contiguous()
                for t in group
            ):
                raise ValueError("GQA mirror requires uniform contiguous CPU KV per pool")
        if key.dtype != value.dtype:
            raise ValueError("GQA K/V mirror dtype mismatch")
        # Native chunked copy appends events and does not enforce its nominal
        # max_pending_events argument; bound the combined K/V count here.
        chunk_bytes = 14 * 1024 * 1024
        if key.shape != value.shape:
            raise ValueError("GQA K/V shape mismatch")
        row_bytes = key.nbytes // key.shape[0]
        if row_bytes <= 0 or (2 * 1024 * 1024) % row_bytes:
            raise ValueError("GQA SDMA requires row size dividing 2MiB")
        page_rows = (2 * 1024 * 1024) // row_bytes
        page_count = (key.shape[0] + page_rows - 1) // page_rows
        # Worst case: each selected page is an isolated range. This is safe
        # for arbitrary noncontiguous request slots, not just dense C1 tables.
        pending = 2 * page_count * ((page_rows * row_bytes + chunk_bytes - 1) // chunk_bytes)
        capacity = self.manager._kv_swap_in_event_tensor.numel()
        if pending > capacity:
            raise ValueError(f"GQA K/V mirror requires {pending} events, capacity {capacity}")
        required = key.nbytes + value.nbytes
        pool = self.manager._hbw_pool
        pool.sweep()
        staging_bytes = required * (2 if double else 1)
        if staging_bytes > pool.free_bytes:
            raise RuntimeError(
                f"GQA K/V staging requires {staging_bytes} HBW bytes; free={pool.free_bytes}"
            )
        if double:
            source_ranges = sorted((t.data_ptr(), t.data_ptr() + t.nbytes) for t in keys + values)
            if any(a[1] > b[0] for a, b in zip(source_ranges, source_ranges[1:])):
                raise ValueError("Target double pair requires disjoint layer DDR sources")
        allocated = []
        pairs = []
        try:
            for pair in range(2 if double else 1):
                kh = pool.alloc(key.shape, key.dtype, label=f"gqa_sdma_k{pair}")
                allocated.append(kh)
                vh = pool.alloc(value.shape, value.dtype, label=f"gqa_sdma_v{pair}")
                allocated.append(vh)
                pairs.append((
                    kh, vh, torch.zeros(capacity, dtype=torch.int32),
                    torch.zeros(1, dtype=torch.int32),
                ))
        except Exception:
            for tensor in reversed(allocated):
                pool.free(tensor)
            raise
        state = _GQAKVSwap(
            owner, layers, keys, values, *pairs[0],
            torch.zeros(page_count, dtype=torch.int32),
            torch.zeros(8, dtype=torch.int64), page_rows,
            self._source_signature(keys + values), bool(is_target),
            pairs[1] if double else None,
        )
        self.pools[id(owner)] = state
        logger.info("GQA SDMA dynamic-page mirror owner=%x layers=%s K=%s V=%s "
                    "staging_bytes=%d maximum_bytes_per_forward=%d maximum_events_per_layer=%d "
                    "pool_used_bytes=%d pool_free_bytes=%d",
                    id(owner), layers, tuple(key.shape), tuple(value.shape),
                    required * len(pairs), required * len(layers), pending,
                    pool.used_bytes, pool.free_bytes)

    @staticmethod
    def _gqa_pair(state, position):
        if state.extra_pair is not None and position % 2:
            return state.extra_pair
        return state.key_hbw, state.value_hbw, state.events, state.event_count

    def register_draft_mirror_reuse(self, owner, req_pool, server_args):
        state = self.pools.get(id(owner))
        requested = os.environ.get("SGLANG_KUNPENG_QWEN_DRAFT_MIRROR_REUSE", "0") == "1"
        if state is None or state.is_target:
            return
        if state.draft_reuse is not None:
            if not requested or state.draft_reuse.req_pool is not req_pool:
                raise ValueError("draft mirror reuse mode/owner changed; recreate graphs")
            return
        if not requested:
            return
        unsupported = (
            "enable_hierarchical_cache", "enable_hisparse", "enable_memory_saver",
            "disaggregation_decode_enable_offload_kvcache",
            "disaggregation_decode_enable_radix_cache", "enable_pdmux",
        )
        eligible = (
            len(state.layers) == 1 and state.extra_pair is None
            and (server_args.max_running_requests == 1 or (
                os.environ.get("SGLANG_KUNPENG_QWEN_DRAFT_RESIDENCY_V2", "0") == "1"
                and server_args.max_running_requests <= 32
                and server_args.disable_radix_cache
            ))
            and server_args.speculative_eagle_topk == 1
            and server_args.disable_overlap_schedule
            and not any(getattr(server_args, name, False) for name in unsupported)
            and os.environ.get("SGLANG_DISAGG_STAGING_BUFFER", "0") in ("0", "")
        )
        state.draft_reuse = DraftMirrorReuse(
            state, req_pool, eligible=eligible,
            tail_guard=2 * server_args.speculative_num_draft_tokens,
        )
        logger.info("Qwen draft mirror reuse requested owner=%x eligible=%s", id(owner), eligible)

    def gqa_kv_fixed_tensors(self, owner) -> tuple:
        state = self.pools.get(id(owner))
        if state is None:
            return ()
        return (
            self._gqa_pair(state, 0) + (state.extra_pair or ())
            + (state.pages, state.stats)
            + (state.draft_reuse.fixed_tensors() if state.draft_reuse is not None else ())
        )

    def _gqa_drain(self, state):
        for position in range(2 if state.extra_pair is not None else 1):
            _, _, events, count = self._gqa_pair(state, position)
            ops.kupl_sdma_wait_all(events, count)

    def _gqa_swap_in(self, state, position) -> None:
        kh, vh, events, count = self._gqa_pair(state, position)
        ops.kupl_sdma_wait_all(events, count)
        ops.kupl_sdma_gqa_copy_pages(
            kh, vh, state.keys[position], state.values[position],
            state.pages, state.stats, events, count, state.page_rows)

    def get_gqa_kv(self, owner, layer_id, table, requests, lengths):
        state = self.pools[id(owner)]
        position = state.layers.index(layer_id)
        if position == 0:
            if state.extra_pair is not None:
                self._gqa_drain(state)
            reuse = state.draft_reuse
            if reuse is None:
                ops.kupl_sdma_gqa_prepare_pages(
                    table, requests, lengths, state.pages, state.stats,
                    state.keys[0].shape[0], state.page_rows)
            else:
                self._gqa_drain(state)
                prepare = ops.qwen_draft_reuse_prepare_v2 if reuse.v2 else ops.qwen_draft_reuse_prepare
                scan = (reuse.scan_lengths,) if reuse.v2 else ()
                tail = (reuse.tail_guard,) if reuse.v2 else ()
                prepare(
                    state.keys[0], state.values[0], state.key_hbw, state.value_hbw,
                    table, requests, lengths, state.pages, state.stats,
                    reuse.resident, reuse.key, reuse.external, reuse.generations,
                    reuse.counters, *scan, state.keys[0].shape[0], state.page_rows, *tail)
            self._gqa_swap_in(state, 0)
        kh, vh, events, count = self._gqa_pair(state, position)
        # Required before the caller writes BOTH DDR source and HBW rows.
        ops.kupl_sdma_wait_all(events, count)
        return kh, vh

    def commit_draft_mirror_reuse(self, owner, table, requests):
        reuse = getattr(owner, "_qwen_draft_reuse", None)
        if reuse is None:
            return  # default/target: no dictionary lookup, new op or log
        state = reuse.state
        commit = ops.qwen_draft_reuse_commit_v2 if reuse.v2 else ops.qwen_draft_reuse_commit
        commit(
            state.keys[0], state.values[0], state.key_hbw, state.value_hbw,
            table, requests, state.pages, reuse.resident, reuse.key,
            reuse.external, reuse.generations, state.event_count, reuse.counters)

    def prefetch_gqa_kv_before_attention(self, owner, layer_id):
        state = self.pools[id(owner)]
        if state.extra_pair is not None:
            position = state.layers.index(layer_id) + 1
            if position < len(state.layers):
                self._gqa_swap_in(state, position)

    def swap_next_gqa_kv(self, owner, layer_id) -> None:
        state = self.pools[id(owner)]
        position = state.layers.index(layer_id) + 1
        if state.extra_pair is not None:
            if position == len(state.layers):
                self._gqa_drain(state)
        elif position < len(state.layers):
            self._gqa_swap_in(state, position)


class DraftMirrorReuse:
    def __init__(self, state, req_pool, *, eligible, tail_guard=8):
        self.state = state
        self.v2 = os.environ.get("SGLANG_KUNPENG_QWEN_DRAFT_RESIDENCY_V2", "0") == "1"
        self.tail_guard = tail_guard
        self.scan_lengths = (
            torch.zeros((req_pool.req_to_token.shape[0] + 1, 2), dtype=torch.int64)
            if self.v2 else None
        )
        self.req_pool = req_pool
        self.thread = threading.get_ident()
        self.resident = torch.zeros_like(state.pages)
        self.key = torch.zeros(16, dtype=torch.int64)
        # epoch, pending RDMA tickets, armed, irreversible unsafe latch
        self.external = torch.tensor([0, 0, int(eligible), 0], dtype=torch.int64)
        self.generations = torch.zeros(req_pool.req_to_token.shape[0], dtype=torch.int64)
        self.counters = torch.zeros(10, dtype=torch.int64)
        self.next_generation = 0
        self.next_ticket = 0
        self.tickets = {}
        self.first_unsafe_reason = None
        self._control()
        for slot in range(1, self.generations.numel()):
            if slot not in req_pool.free_slots:
                self._advance(slot)
        if getattr(req_pool, "_qwen_draft_reuse", self) is not self:
            raise ValueError("request pool already belongs to another draft residency owner")
        req_pool._qwen_draft_reuse = self
        state.owner._qwen_draft_reuse = self
        getattr(state.owner, "full_kv_pool", state.owner)._qwen_draft_reuse = self

    def fixed_tensors(self):
        return (
            self.resident, self.key, self.external, self.generations, self.counters
        ) + ((self.scan_lengths,) if self.v2 else ())

    def _control(self):
        # Never assume the GIL makes native graph/SDMA metadata atomic.
        if threading.get_ident() != self.thread or is_capturing():
            raise RuntimeError("draft lifecycle metadata requires scheduler thread between graph.run calls")

    def _advance(self, slot):
        self.next_generation += 1
        if self.next_generation >= 2**63 - 1:
            raise RuntimeError("draft request generation exhausted; rebuild owner")
        self.generations[slot] = self.next_generation

    def invalidate_graph(self):
        s = self.state
        ops.qwen_draft_reuse_invalidate(
            self.resident, self.key, self.counters, s.events, s.event_count)

    def before_pool_write(self):
        # When called by a captured writer, only record native invalidation.
        # No capture-time Tensor.add_/index_put or host epoch mutation.
        if is_capturing():
            self.invalidate_graph()
            return
        self._control()
        self.invalidate_graph()  # drain before an external DDR writer begins
        if int(self.external[0]) >= 2**63 - 2:
            raise RuntimeError("draft external epoch exhausted; rebuild owner")
        self.external[0] += 1

    def fresh_slot(self, slot):
        self._control()
        self.invalidate_graph()
        self._advance(slot)

    def release_slot(self, req):
        self._control()
        ticket = getattr(req, "_qwen_draft_receive_ticket", None)
        if ticket in self.tickets:
            self.fail_receive(req)  # free/abort does not establish RDMA quiescence
        self.invalidate_graph()
        self._advance(req.req_pool_idx)

    def clear_slots(self):
        self._control()
        if self.v2:
            logger.info(
                "DraftResidencyV2 counters=%s scan_slots_and_logical=%s copy_stats=%s",
                self.counters.tolist(), self.scan_lengths[-1].tolist(),
                self.state.stats.tolist(),
            )
        self.invalidate_graph()
        if self.tickets:
            self._unsafe("request-pool clear while RDMA tickets lack completion")
        for slot in range(1, self.generations.numel()):
            self._advance(slot)  # never reset generations to an earlier value

    def begin_receive(self, req):
        self._control()
        if int(self.external[3]):
            self.before_pool_write()
            return  # permanently disabled: never accumulate later request objects
        if getattr(req, "_qwen_draft_receive_ticket", None) in self.tickets:
            raise RuntimeError("duplicate draft receive advertisement")
        self.before_pool_write()
        if len(self.tickets) >= self.generations.numel() - 1:
            self._unsafe("receive tickets exceeded actual request-pool capacity")
            return
        self.next_ticket += 1
        ticket = self.next_ticket
        self.tickets[ticket] = req
        req._qwen_draft_receive_ticket = ticket
        self.external[1] = len(self.tickets)

    def finish_receive(self, req):
        self._control()
        ticket = getattr(req, "_qwen_draft_receive_ticket", None)
        if self.tickets.get(ticket) is not req:
            # An untracked completion is uncertainty, never evidence to re-arm.
            self._unsafe("receive completion had no matching owner ticket")
            self.before_pool_write()
            return
        self.before_pool_write()
        del self.tickets[ticket]
        req._qwen_draft_receive_ticket = None
        self.external[1] = len(self.tickets)
        # Do NOT clear external[3]. Another failed/aborted ticket may still write.

    def fail_receive(self, req):
        self._control()
        self._unsafe("receive abort/failure has no proved RDMA quiescence")
        self.before_pool_write()
        # Retain ticket and pending count until owner rebuild. abort/clear has
        # no proved RDMA cancel/flush fence. No blocking/recovery loop is added.

    def _unsafe(self, reason):
        if self.first_unsafe_reason is None:
            self.first_unsafe_reason = reason
        self.external[3] = 1  # Only a new owner starts with zero.
