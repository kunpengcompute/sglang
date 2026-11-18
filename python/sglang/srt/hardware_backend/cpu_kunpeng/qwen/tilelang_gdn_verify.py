"""TileLang (kunpeng_pto) implementation of ``qwen_gdn_verify``.

Experimental replacement for the C++ SVE kernel behind
``SGLANG_KUNPENG_GDN_VERIFY_TILELANG=1``. The kernel is copied from
``test/srt/cpu_kunpeng/test_qwen_ops_tilelang.py`` (branch
``gitcode/qwen_native_ops``) with one change: ``V`` is declared as
``[tokens, v_heads, value_dim]`` instead of the test's
``[tokens, 2 * v_heads, value_dim]``, whose second half the kernel never reads.

The kernel is compiled per concrete shape, because tilelang bakes every tensor
shape into the generated code. Its ``extern "C" void call(uint8_t* ...)`` entry
is registered into ``common_ops`` by shape so ``qwen_gdn_verify_tilelang`` can
be replayed by the common graph through ``qwen_graph_gdn_verify_tilelang_dispatch``
(``sgl-kernel/csrc/cpu/cpu_kunpeng/qwen/qwen_tilelang_gdn_verify.cpp``). The
graph planner owns every buffer: ``QK``/``V``/``AB`` must be contiguous, so the
caller packs them from the strided qkv/BA views with ``alloc_buffer`` plus
``copy_kunpeng`` -- this target captures without the automatic ATen adapter, and
plain ``torch.cat``/``clone`` would allocate unregistered storages. ``State`` and
``Checkpoints`` stay bound model tensors.
"""

import contextlib
import ctypes
import fcntl
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from tqdm import tqdm

import torch

import tilelang
import tilelang.language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}

_bridge = None
_register_call = None
_registered = set()

# Set while precompile_batches() runs so a shape-by-shape log line is not
# emitted for the whole sweep; the progress bar carries that information.
_precompiling = False

# Startup precompilation (SGLANG_KUNPENG_TL_OP_PRECOMPILE=1): rank 0 of the TP
# group compiles every batch size ahead of the first request. The pool size is
# the deployment node's logical CPU count; it is only an upper bound, since one
# shape per worker is submitted.
_PRECOMPILE_JOBS = 608


def _parse_fallback_batches(value):
    """Parse ``"1"`` / ``"1,3,4"`` into ``{1, 3, 4}``; empty means none."""
    batches = set()
    for item in (value or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            batch = int(item)
        except ValueError:
            batch = 0
        if batch < 1:
            raise ValueError(
                "SGLANG_KUNPENG_GDN_VERIFY_TILELANG_FALLBACK_BATCH_SIZE must "
                f"be a comma-separated list of positive ints, got {value!r}"
            )
        batches.add(batch)
    return batches


# Batch sizes that always use the reference C++ kernel, even though the
# TileLang kernel is available and compiled. bs=1 is the usual entry: with a
# single request the packing overhead of the TileLang path outweighs its
# parallelism, so the native kernel is faster.
_CONFIGURED_FALLBACK_BATCHES = _parse_fallback_batches(
    os.environ.get(
        "SGLANG_KUNPENG_GDN_VERIFY_TILELANG_FALLBACK_BATCH_SIZE", ""
    )
)

# Batch sizes whose TileLang compilation failed. They are served by the
# reference C++ kernel instead of crashing the worker.
_fallback_batches: set[int] = set()


def _gdn_verify_kernel(batch, steps, q_heads, v_heads, key_dim, value_dim,
                       state_slots, ckpt_slots):
    group = v_heads // q_heads
    tokens = batch * steps
    num_tasks = batch * v_heads
    query_scale = 1.0 / math.sqrt(key_dim)

    @T.prim_func
    def main(
        A_log: T.Tensor((v_heads,), 'float'),
        dt_bias: T.Tensor((v_heads,), 'bfloat16'),
        QK: T.Tensor((tokens, 2 * q_heads, key_dim), 'bfloat16'),
        V: T.Tensor((tokens, v_heads, value_dim), 'bfloat16'),
        AB: T.Tensor((tokens, 2 * v_heads), 'bfloat16'),
        State: T.Tensor((state_slots, v_heads, key_dim, value_dim), 'float'),
        Indices: T.Tensor((batch,), 'int32'),
        Checkpoints: T.Tensor((ckpt_slots, steps, v_heads, key_dim, value_dim), 'float'),
        Ckpt_indices: T.Tensor((batch,), 'int32'),
        O: T.Tensor((tokens, v_heads, value_dim), 'bfloat16'),
    ):
        with T.Kernel(num_tasks, is_cpu=True) as tid:
            r = tid // v_heads
            h = tid % v_heads
            kh = h // group

            S = T.alloc_buffer([key_dim, value_dim], dtype='float', scope='shared')
            Qb = T.alloc_buffer([key_dim], dtype='float', scope='shared')
            Kb = T.alloc_buffer([key_dim], dtype='float', scope='shared')
            Db = T.alloc_buffer([value_dim], dtype='float', scope='shared')
            Vb = T.alloc_buffer([value_dim], dtype='float', scope='shared')
            SC = T.alloc_buffer([2], dtype='float', scope='shared')

            # direct scalar reads
            slot = Indices[r]
            ckpt = Ckpt_indices[r]
            A_val = A_log[h]
            bias_f = dt_bias[h] * 1.0    # bf16 -> fp32 promotion

            # step 0 source: the persistent state (read-only afterwards)
            for i in T.serial(key_dim):
                srow = T.sve.load(State[slot, h, i, 0], value_dim, dtype='float')
                T.sve.store(srow, S[i, 0])

            for s in T.serial(steps):
                token = r * steps + s

                # ---- q: normalize per q-head ----
                q_raw = T.sve.load(QK[token, kh, 0], key_dim, dtype='bfloat16')
                q_v = T.sve.convert(q_raw, 'float')
                T.sve.store(q_v, Qb[0])
                SC[0] = 0.0
                for d in T.serial(key_dim):
                    SC[0] = SC[0] + Qb[d] * Qb[d]
                q_s = query_scale / T.sqrt(SC[0] + 1e-6)
                T.sve.mul(q_v, q_s)
                T.sve.store(q_v, Qb[0])

                # ---- k: normalize per q-head (no query_scale) ----
                k_raw = T.sve.load(QK[token, q_heads + kh, 0], key_dim, dtype='bfloat16')
                k_v = T.sve.convert(k_raw, 'float')
                T.sve.store(k_v, Kb[0])
                SC[1] = 0.0
                for d in T.serial(key_dim):
                    SC[1] = SC[1] + Kb[d] * Kb[d]
                k_s = 1.0 / T.sqrt(SC[1] + 1e-6)
                T.sve.mul(k_v, k_s)
                T.sve.store(k_v, Kb[0])

                # ---- gate / decay / beta (direct scalar reads) ----
                a_f = AB[token, h] * 1.0
                b_f = AB[token, v_heads + h] * 1.0
                x = a_f + bias_f
                sp = T.Select(x > 20.0, x,
                              T.Select(x < -20.0, T.exp(x),
                                      T.log(1.0 + T.exp(x))))
                gate = -T.exp(A_val) * sp
                decay = T.exp(gate)
                beta = 1.0 / (1.0 + T.exp(-b_f))

                # ---- memory: mem[j] = decay * sum_i k[i] * S[i][j] ----
                mem = T.sve.load(S[0, 0], value_dim, dtype='float')
                T.sve.mul(mem, 0.0)
                for i in T.serial(key_dim):
                    row = T.sve.load(S[i, 0], value_dim, dtype='float')
                    T.sve.mul(row, Kb[i])
                    T.sve.add(mem, row)
                T.sve.mul(mem, decay)
                T.sve.store(mem, Db[0])

                # ---- delta[j] = (v[j] - mem[j]) * beta ----
                v_raw = T.sve.load(V[token, h, 0], value_dim, dtype='bfloat16')
                v_v = T.sve.convert(v_raw, 'float')
                T.sve.store(v_v, Vb[0])
                for j in T.serial(value_dim):
                    Db[j] = (Vb[j] - Db[j]) * beta

                # ---- S = S * decay + k (outer) delta ----
                for i in T.serial(key_dim):
                    row = T.sve.load(S[i, 0], value_dim, dtype='float')
                    T.sve.mul(row, decay)
                    dv = T.sve.load(Db[0], value_dim, dtype='float')
                    T.sve.mul(dv, Kb[i])
                    T.sve.add(row, dv)
                    T.sve.store(row, S[i, 0])

                # ---- persist the snapshot for this step ----
                for i in T.serial(key_dim):
                    row = T.sve.load(S[i, 0], value_dim, dtype='float')
                    T.sve.store(row, Checkpoints[ckpt, s, h, i, 0])

                # ---- out[j] = sum_i S[i][j] * q[i] ----
                out = T.sve.load(Db[0], value_dim, dtype='float')
                T.sve.mul(out, 0.0)
                for i in T.serial(key_dim):
                    row = T.sve.load(S[i, 0], value_dim, dtype='float')
                    T.sve.mul(row, Qb[i])
                    T.sve.add(out, row)
                # explicit narrowing (see the gqa note in the reference test)
                out16 = T.sve.convert(out, 'bfloat16')
                T.sve.store(out16, O[token, h, 0])

    return main


def _disk_cache_has_entry(prim):
    """Whether the disk cache already holds this PrimFunc's built kernel.

    Used only to report cache hits honestly: ``tilelang.compile`` is silent, and
    probing the same key avoids claiming a compile when the kernel is reused
    from a previous run.
    """
    try:
        from tilelang.cache.kernel_cache import (
            KERNEL_LIB_PATH,
            PARAMS_PATH,
            WRAPPED_KERNEL_PATH,
            KernelCache,
        )
        from tilelang.env import is_cache_enabled
        from tilelang.utils.target import determine_platform

        if not is_cache_enabled():
            return False
        cache = KernelCache()
        key = cache._generate_key(
            func=prim,
            out_idx=[9],
            workspace_idx=None,
            execution_backend="cython",
            args=(),
            target="pto",
            target_host=None,
            platform=determine_platform("auto"),
            pass_configs=_PASS_CONFIGS,
        )
        cache_path = os.path.join(cache.cache_dir, key)
        return all(
            os.path.exists(os.path.join(cache_path, name))
            for name in (WRAPPED_KERNEL_PATH, PARAMS_PATH, KERNEL_LIB_PATH)
        )
    except Exception:
        return False


@contextlib.contextmanager
def _key_compile_lock(key):
    """Serialize one shape's build across processes.

    ``KernelCache`` only guards its in-memory map and the disk read/write; the
    compile runs outside that lock, and saving copies over ``kernel_lib.so``.
    Two nodes sharing a cache directory would race, so hold an flock keyed by
    the shape tuple for the whole build. Different shapes use different files
    and stay parallel.
    """
    try:
        from tilelang.env import TILELANG_CACHE_DIR

        lock_dir = os.path.join(TILELANG_CACHE_DIR, "locks")
        os.makedirs(lock_dir, exist_ok=True)
        path = os.path.join(
            lock_dir, "gdn_verify_%s.lock" % "_".join(str(v) for v in key)
        )
        fp = open(path, "w")
    except Exception:
        yield
        return
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        finally:
            fp.close()


@lru_cache(maxsize=None)
def _compiled(batch, steps, q_heads, v_heads, key_dim, value_dim, state_slots,
              ckpt_slots):
    prim = _gdn_verify_kernel(batch, steps, q_heads, v_heads, key_dim,
                              value_dim, state_slots, ckpt_slots)
    key = (batch, steps, q_heads, v_heads, key_dim, value_dim, state_slots,
           ckpt_slots)
    with _key_compile_lock(key):
        cache_hit = _disk_cache_has_entry(prim)
        t0 = time.perf_counter()
        kernel = tilelang.compile(prim, out_idx=[9], target='pto',
                                  pass_configs=_PASS_CONFIGS)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        # Keep the ctypes handle alive next to the kernel; replay calls its raw
        # `call` entry directly, without the cython wrapper.
        lib = getattr(kernel.adapter, "lib", None)
        if lib is None:
            lib = ctypes.CDLL(kernel.adapter.lib_generator.libpath)
        if not hasattr(lib, "call"):
            raise RuntimeError("TileLang GDN verify library has no 'call' entry")
    # Report once per shape per process: prepare() is called once per GDN
    # layer, but this body runs only on the first call for the key.
    if not cache_hit and not _precompiling:
        print(
            "[QwenVerify] batch=%d compiled in %.0f ms" % (batch, elapsed_ms),
            flush=True,
        )
    return kernel, lib, cache_hit, elapsed_ms


def _shape_key(qk, v, state, indices, checkpoints):
    return (
        int(indices.numel()),
        int(checkpoints.shape[1]),
        int(qk.shape[1]) // 2,
        int(v.shape[1]),
        int(qk.shape[2]),
        int(v.shape[2]),
        int(state.shape[0]),
        int(checkpoints.shape[0]),
    )


def _check(A_log, dt_bias, qk, v, ab, state, indices, checkpoints,
           ckpt_indices):
    assert A_log.dtype == state.dtype == checkpoints.dtype == torch.float32
    assert indices.dtype == ckpt_indices.dtype == torch.int32
    assert all(t.dtype == torch.bfloat16 for t in (dt_bias, qk, v, ab))
    # The persistent state is read-only and never copied, so it must already be
    # laid out the way the kernel addresses it.
    assert all(t.is_contiguous() for t in (state, indices, checkpoints, ckpt_indices))


def prepare(A_log, dt_bias, qk, v, ab, state, indices, checkpoints,
            ckpt_indices):
    """Compile this shape and publish its entry to the graph dispatcher.

    Returns ``(kernel, cache_hit, elapsed_ms)``; the startup precompile uses
    the timing/cache flag for its progress bar, the lazy path ignores them.
    """
    key = _shape_key(qk, v, state, indices, checkpoints)
    kernel, lib, cache_hit, elapsed_ms = _compiled(*key)
    if key not in _registered:
        if _bridge is None:
            raise RuntimeError(
                "TileLang GDN verify graph dispatch is not registered; "
                "SGLANG_KUNPENG_GDN_VERIFY_TILELANG requires a rebuilt sgl-kernel"
            )
        _register_call(*key, ctypes.cast(lib.call, ctypes.c_void_p))
        _registered.add(key)
    return kernel, cache_hit, elapsed_ms


def gdn_verify(A_log, dt_bias, qk, v, ab, state, indices, checkpoints,
               ckpt_indices):
    """Packed-input form used by both eager and graph replay.

    ``qk`` is ``[tokens, 2 * q_heads, key_dim]`` and ``ab`` is
    ``[tokens, 2 * v_heads]``; the caller repacks the strided projections
    (captured as ordinary ATen nodes when replaying from a graph).
    """
    _check(A_log, dt_bias, qk, v, ab, state, indices, checkpoints,
           ckpt_indices)
    kernel, _, _ = prepare(A_log, dt_bias, qk, v, ab, state, indices,
                           checkpoints, ckpt_indices)
    return kernel(A_log.contiguous(), dt_bias.contiguous(), qk,
                  v.contiguous(), ab, state, indices.contiguous(),
                  checkpoints, ckpt_indices.contiguous())


def _shape(A_log, dt_bias, qk, v, ab, state, indices, checkpoints,
           ckpt_indices):
    return [(tuple(v.shape), v.dtype)]


def fallback_batches() -> set:
    """Batch sizes whose TileLang compile failed in this process."""
    return _fallback_batches


def configured_fallback_batches() -> set:
    """Batch sizes pinned to the C++ kernel by env configuration."""
    return _CONFIGURED_FALLBACK_BATCHES


def tilelang_disabled(batch) -> bool:
    """True when this batch must use the reference C++ kernel."""
    return batch in _CONFIGURED_FALLBACK_BATCHES or batch in _fallback_batches


def mark_failed(batch, exc, qk, v, ab, state, checkpoints):
    """Log a failed shape and degrade it to the reference C++ kernel."""
    if batch in _fallback_batches:
        return
    _fallback_batches.add(batch)
    print(
        "[QwenVerify] batch=%d TileLang fallback to C++: %s: %s "
        "QK=%s AB=%s V=%s state=%s checkpoints=%s"
        % (
            batch, type(exc).__name__, exc,
            (tuple(qk.shape), qk.dtype),
            (tuple(ab.shape), ab.dtype),
            (tuple(v.shape), v.dtype),
            (tuple(state.shape), state.dtype),
            (tuple(checkpoints.shape), checkpoints.dtype),
        ),
        flush=True,
    )


def precompile_batches(*, state, checkpoints, q_heads, v_heads, key_dim,
                       value_dim, max_batch, jobs=_PRECOMPILE_JOBS, skip=()):
    """Compile and register ``batch = 1..max_batch`` except ``skip``.

    Each shape is one thread pinned to a distinct logical CPU: a single compile
    is single threaded (codegen plus one clang invocation), so the speedup comes
    from compiling different shapes concurrently. Threads exit when the pool is
    closed. Returns the sorted list of batch sizes that failed.
    """
    wanted = [b for b in range(1, max_batch + 1) if b not in set(skip)]
    if not wanted:
        return []
    steps = int(checkpoints.shape[1])
    dtype = torch.bfloat16
    tokens = max_batch * steps
    # Only shapes and dtypes reach the compiler; contents are never read.
    qk = torch.empty((tokens, 2 * q_heads, key_dim), dtype=dtype)
    v = torch.empty((tokens, v_heads, value_dim), dtype=dtype)
    ab = torch.empty((tokens, 2 * v_heads), dtype=dtype)
    a_log = torch.empty(v_heads, dtype=torch.float32)
    dt_bias = torch.empty(v_heads, dtype=dtype)
    indices = torch.arange(max_batch, dtype=torch.int32)
    ckpt_indices = torch.arange(max_batch, dtype=torch.int32)

    total_cpus = os.cpu_count() or 1
    workers = max(1, min(int(jobs), len(wanted), total_cpus))
    started = time.perf_counter()
    failures = []
    bind_error = []

    def _compile_one(batch, cpu):
        try:
            os.sched_setaffinity(0, {cpu})
        except OSError as exc:
            # Usually the container cpuset is narrower than the node; report
            # once and keep compiling instead of failing the shape.
            if not bind_error:
                bind_error.append(exc)
                print(
                    "[QwenVerify] precompile cannot bind cpu %d: %s"
                    % (cpu, exc),
                    flush=True,
                )
        t0 = time.perf_counter()
        try:
            _, cache_hit, _ = prepare(
                a_log, dt_bias,
                qk[: batch * steps], v[: batch * steps], ab[: batch * steps],
                state, indices[:batch], checkpoints, ckpt_indices[:batch],
            )
        except Exception as exc:  # reported by mark_failed, degraded at runtime
            mark_failed(batch, exc, qk[: batch * steps], v[: batch * steps],
                        ab[: batch * steps], state, checkpoints)
            return batch, None, (time.perf_counter() - t0) * 1e3
        return batch, cache_hit, (time.perf_counter() - t0) * 1e3

    global _precompiling
    hits = 0
    compiles = 0
    _precompiling = True
    # Always draw: the deployment redirects scheduler output to a log file
    # (non-TTY); tqdm still renders there when the log is followed with
    # `tail -f`. stdout keeps it next to the other [QwenVerify] lines.
    bar = tqdm(
        total=len(wanted),
        desc="[QwenVerify] TL gdn_verify",
        unit="shape",
        dynamic_ncols=True,
        mininterval=1.0,
        disable=False,
        file=sys.stdout,
    )
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_compile_one, batch, (batch - 1) % total_cpus)
                for batch in wanted
            ]
            for future in as_completed(futures):
                batch, cache_hit, elapsed_ms = future.result()
                if cache_hit is None:
                    failures.append(batch)
                elif cache_hit:
                    hits += 1
                else:
                    compiles += 1
                bar.set_postfix(
                    hit=hits, compiled=compiles,
                    last="%.0fms" % elapsed_ms, refresh=False,
                )
                bar.update(1)
    finally:
        _precompiling = False
        bar.close()

    failures.sort()
    print(
        "[QwenVerify] precompile done: %d shapes (%d compiled, %d hit) in "
        "%.1fs, workers=%d, failed=%s"
        % (
            len(wanted), compiles, hits, time.perf_counter() - started,
            workers, failures,
        ),
        flush=True,
    )
    return failures


def _first_gdn_layer(model):
    """Return a linear-attention layer, whose attributes describe the shape."""
    from sglang.srt.layers.radix_linear_attention import RadixLinearAttention

    for module in model.modules():
        if isinstance(module, RadixLinearAttention) and module.A_log is not None:
            return module
    return None


def precompile_gdn_verify(runner) -> None:
    """Compile and register this shape family for batch=1..M at startup.

    Called once per scheduler process after the memory pool exists. The
    TileLang cache lives on local disk, so every node compiles its own copy:
    within a node only the rank whose in-node slot is 0 compiles, and every
    other rank waits on the TP-group barrier so a long compile cannot
    desynchronize the group around a later collective. Peers register a shape
    lazily on first use, which is a local disk-cache hit. The kernel is never
    executed here.
    """
    from sglang.srt.hardware_backend.cpu_kunpeng.distributed import rank_layout

    pp_size = int(runner.server_args.pp_size)
    rin_in_node = rank_layout.compute_rin_in_node(
        int(getattr(runner, "pp_rank", 0)),
        int(getattr(runner, "tp_rank", 0)),
        pp_size,
    )
    verbose = rin_in_node == 0  # one reporter per node

    def skip(reason):
        if verbose:
            print(f"[QwenVerify] precompile skipped: {reason}", flush=True)

    if os.environ.get("SGLANG_KUNPENG_GDN_VERIFY_TILELANG", "0") != "1":
        skip("SGLANG_KUNPENG_GDN_VERIFY_TILELANG != 1")
        return
    # A PP stage may not own any linear-attention layer, so skipping must stay
    # symmetric; keep the feature off for layouts where the barrier group would
    # differ.
    if pp_size != 1 or runner.server_args.dp_size != 1:
        skip(
            f"pp_size={pp_size} "
            f"dp_size={runner.server_args.dp_size} (only 1/1 supported)"
        )
        return
    if runner.server_args.disaggregation_mode == "prefill":
        # Target verify only runs on the decode instance; the prefill side has
        # no use for these kernels.
        skip("prefill instance (target verify runs on decode)")
        return
    if runner.is_draft_worker:
        return  # the target runner in the same process precompiles
    pool = getattr(
        getattr(getattr(runner, "req_to_token_pool", None), "mamba_pool", None),
        "mamba_cache",
        None,
    )
    checkpoints_all = getattr(pool, "intermediate_ssm", None)
    if pool is None or checkpoints_all is None:
        skip(
            "mamba speculative checkpoints unavailable "
            "(speculative decoding off?)"
        )
        return
    gdn = _first_gdn_layer(runner.model)
    if gdn is None:
        skip("no linear-attention (GDN) layer found")
        return
    # Only shapes reach the compiler, so reuse the real cache views instead of
    # allocating state_slots-sized stubs. The cache is authoritative for the
    # per-rank head geometry; only the query head count is not in it.
    state = pool.temporal[0]
    checkpoints = checkpoints_all[0]
    v_heads, key_dim, value_dim = (
        int(state.shape[1]), int(state.shape[2]), int(state.shape[3])
    )
    q_heads = int(gdn.num_q_heads)
    assert (
        q_heads > 0
        and v_heads % q_heads == 0
        and int(gdn.num_v_heads) == v_heads
        and int(gdn.head_k_dim) == key_dim
        and int(gdn.head_v_dim) == value_dim
    ), (
        f"unexpected GDN head geometry hq={q_heads} hv={v_heads} "
        f"kd={key_dim} vd={value_dim} state={tuple(state.shape)} "
        f"layer_hv={int(gdn.num_v_heads)} layer_kd={int(gdn.head_k_dim)} "
        f"layer_vd={int(gdn.head_v_dim)}"
    )
    from sglang.srt.hardware_backend.cpu_kunpeng.qwen.capabilities import (
        get_capabilities,
    )

    max_batch = min(
        get_capabilities().decode_max_requests,
        int(runner.max_running_requests),
        int(checkpoints.shape[0]) - 1,
    )
    if max_batch <= 0:
        skip(f"max_batch={max_batch}")
        return
    if verbose:
        print(
            "[QwenVerify] precompile start: tp_rank=%d rin_in_node=%d "
            "batches=1..%d steps=%d hq=%d hv=%d kd=%d vd=%d "
            "state_slots=%d ckpt_slots=%d cpp_fallback=%s"
            % (
                int(getattr(runner, "tp_rank", 0)), rin_in_node, max_batch,
                int(checkpoints.shape[1]), q_heads, v_heads, key_dim,
                value_dim, int(state.shape[0]), int(checkpoints.shape[0]),
                sorted(_CONFIGURED_FALLBACK_BATCHES),
            ),
            flush=True,
        )

    import torch.distributed as dist

    from sglang.srt.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    failure = None
    # One warmer per node: the TileLang cache is local to the node, and every
    # node's rank-in-node 0 compiles the full range concurrently.
    if rin_in_node == 0:
        try:
            failures = precompile_batches(
                state=state,
                checkpoints=checkpoints,
                q_heads=q_heads,
                v_heads=v_heads,
                key_dim=key_dim,
                value_dim=value_dim,
                max_batch=max_batch,
                skip=_CONFIGURED_FALLBACK_BATCHES,
            )
            if failures:
                failure = (
                    "TileLang gdn_verify precompile failed for batch="
                    f"{failures}"
                )
        except BaseException as exc:
            failure = f"TileLang gdn_verify precompile crashed: {exc!r}"
    # Barrier first: a failed node warmer must not strand its peers on a later
    # collective before the assertion takes the process down.
    dist.barrier(group=tp_group.cpu_group)
    assert failure is None, failure


def register_tilelang_verify(bridge, register_dispatch):
    """Register the graph op and its C++ replay dispatch into common_ops."""
    global _bridge, _register_call
    from sglang.srt.graph import register_op

    try:
        register_call = bridge.qwen_tilelang_register_gdn_verify
    except AttributeError as exc:
        raise RuntimeError(
            "sgl-kernel is missing the TileLang GDN verify graph dispatch; "
            "rebuild it from sources that include qwen_tilelang_gdn_verify.cpp"
        ) from exc
    register_call.argtypes = [ctypes.c_int64] * 8 + [ctypes.c_void_p]
    register_call.restype = None
    _register_call = register_call
    _bridge = bridge
    register_op("qwen_gdn_verify_tilelang", _shape, gdn_verify)
    register_dispatch(bridge, "qwen_gdn_verify_tilelang",
                      "qwen_graph_gdn_verify_tilelang_dispatch")
