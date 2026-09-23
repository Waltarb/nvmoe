"""3-Tier NVMe MoE Expert Cache for FreeToken (GLM-5.3-Flash NVFP4).

Architecture:
  Tier 1: GPU VRAM Slot Cache (288 slots = 3.80 GiB)
  Tier 2: Pinned Host RAM Slot Cache (288 slots = 3.80 GiB, bounded)
  Tier 3: Micron 3400 NVMe (.ftw shards, direct I/O paging on demand)

When an expert is missing from GPU VRAM:
  1. Check if resident in Pinned Host RAM (Tier 2).
  2. If miss, read expert rows (13.52 MB across 6 banks) from NVMe shards into host slot (evicting host LRU).
  3. Rewrite src_indices to host slot ID.
  4. Fused CUDA kernel fast_index_copy_multi_jit gathers from Host -> GPU VRAM over PCIe unmodified.
"""
from __future__ import annotations

import atexit
import bisect
import json
import math
import ctypes
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from flashlib.kernels.slot_cache import lru_ensure
from freetoken.core import get_global_ctx
from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
from freetoken.kernel.pinned import device_ptr
from freetoken.moe.offload_cache import (
    _BANK_SCHEMAS,
    MARLIN_MAX_CACHE_SIZE,
    N_STATS,
    OffloadMoeCache,
)
from freetoken.moe.offload_kernels import (
    ensure_experts as base_ensure_experts,
    materialize_layer as base_materialize_layer,
)
from freetoken.utils import init_logger

logger = init_logger(__name__, "NvmeOffloadCache")

# O_DIRECT block alignment for offsets, lengths, and destination buffers. 4096 covers both
# 512e and 4Kn devices; this model's bank offsets/row_bytes are already 4096-multiples
# (verified by bench_read_shape.py), so nothing needs padding.
_O_DIRECT_ALIGN = 4096

# --------------------------------------------------------------------------
# io_uring FFI Bindings for zero-copy batched Direct I/O
# --------------------------------------------------------------------------
_liburing = None
_RING_STRUCT_SIZE = 512

try:
    _liburing = ctypes.CDLL("liburing-ffi.so.2")

    class _Cqe(ctypes.Structure):
        _fields_ = [
            ("user_data", ctypes.c_uint64),
            ("res", ctypes.c_int32),
            ("flags", ctypes.c_uint32),
        ]

    _liburing.io_uring_queue_init.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
    _liburing.io_uring_queue_init.restype = ctypes.c_int
    _liburing.io_uring_queue_exit.argtypes = [ctypes.c_void_p]
    _liburing.io_uring_queue_exit.restype = None
    _liburing.io_uring_get_sqe.argtypes = [ctypes.c_void_p]
    _liburing.io_uring_get_sqe.restype = ctypes.c_void_p
    _liburing.io_uring_prep_read.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint64
    ]
    _liburing.io_uring_prep_read.restype = None
    _liburing.io_uring_sqe_set_data64.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    _liburing.io_uring_sqe_set_data64.restype = None
    _liburing.io_uring_submit_and_wait.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    _liburing.io_uring_submit_and_wait.restype = ctypes.c_int
    _liburing.io_uring_submit.argtypes = [ctypes.c_void_p]
    _liburing.io_uring_submit.restype = ctypes.c_int
    _liburing.io_uring_wait_cqe.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    _liburing.io_uring_wait_cqe.restype = ctypes.c_int
    _liburing.io_uring_peek_cqe.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    _liburing.io_uring_peek_cqe.restype = ctypes.c_int
    _liburing.io_uring_cqe_seen.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _liburing.io_uring_cqe_seen.restype = None

    class FastIoUringBatch:
        """Zero-copy batched Direct I/O ring. Submits N requests in a single syscall."""

        def __init__(self, queue_depth: int = 128):
            self._ring_buf = ctypes.create_string_buffer(_RING_STRUCT_SIZE)
            self._ring = ctypes.addressof(self._ring_buf)
            rc = _liburing.io_uring_queue_init(queue_depth, self._ring, 0)
            if rc < 0:
                raise RuntimeError(f"io_uring_queue_init failed: {os.strerror(-rc)}")
            self._cqe_ptr = ctypes.c_void_p()
            self._closed = False
            self._in_flight_jobs: dict[int, Any] = {}      # job_id -> tag
            self._tag_remaining: dict[Any, int] = {}       # tag -> pending job count
            self._tag_completed_set: set[Any] = set()
            self._tag_callbacks: dict[Any, Any] = {}
            self._job_id_counter: int = 0

        def outstanding_jobs(self) -> int:
            return len(self._in_flight_jobs)

        def submit_tagged_requests(
            self,
            tagged_jobs: list[tuple[Any, int, int, int, int]],  # (tag, fd, ptr, count, off)
            tag_sizes: dict[Any, int],
            on_tag_ready: Any = None,
        ) -> int:
            if not tagged_jobs:
                return 0
            for tag, sz in tag_sizes.items():
                self._tag_remaining[tag] = self._tag_remaining.get(tag, 0) + sz
                if on_tag_ready is not None:
                    self._tag_callbacks[tag] = on_tag_ready

            for tag, fd, ptr, count, off in tagged_jobs:
                sqe = _liburing.io_uring_get_sqe(self._ring)
                if not sqe:
                    _liburing.io_uring_submit(self._ring)
                    sqe = _liburing.io_uring_get_sqe(self._ring)
                    if not sqe:
                        raise RuntimeError("io_uring submission queue full")
                self._job_id_counter += 1
                job_id = self._job_id_counter
                _liburing.io_uring_prep_read(sqe, fd, ctypes.c_void_p(ptr), count, off)
                _liburing.io_uring_sqe_set_data64(sqe, job_id)
                self._in_flight_jobs[job_id] = tag

            rc = _liburing.io_uring_submit(self._ring)
            if rc < 0:
                raise RuntimeError(f"io_uring_submit failed: {os.strerror(-rc)}")
            return len(tagged_jobs)

        def peek_completed(self, on_tag_completed: Any = None) -> set[Any]:
            newly_completed = set()
            while True:
                rc = _liburing.io_uring_peek_cqe(self._ring, ctypes.byref(self._cqe_ptr))
                if rc < 0:
                    break
                cqe = ctypes.cast(self._cqe_ptr, ctypes.POINTER(_Cqe)).contents
                job_id = cqe.user_data
                _liburing.io_uring_cqe_seen(self._ring, self._cqe_ptr)
                tag = self._in_flight_jobs.pop(job_id, None)
                if tag is not None and tag in self._tag_remaining:
                    self._tag_remaining[tag] -= 1
                    if self._tag_remaining[tag] == 0:
                        del self._tag_remaining[tag]
                        self._tag_completed_set.add(tag)
                        newly_completed.add(tag)
                        cb = self._tag_callbacks.pop(tag, None) or on_tag_completed
                        if cb is not None:
                            cb(tag)
            return newly_completed

        def wait_for_required_tags(self, required_tags: set[Any], on_tag_completed: Any = None) -> None:
            if not required_tags:
                return
            self.peek_completed(on_tag_completed=on_tag_completed)
            while any(t in self._tag_remaining for t in required_tags):
                rc = _liburing.io_uring_wait_cqe(self._ring, ctypes.byref(self._cqe_ptr))
                if rc < 0:
                    raise RuntimeError(f"io_uring_wait_cqe failed: {os.strerror(-rc)}")
                cqe = ctypes.cast(self._cqe_ptr, ctypes.POINTER(_Cqe)).contents
                job_id = cqe.user_data
                _liburing.io_uring_cqe_seen(self._ring, self._cqe_ptr)
                tag = self._in_flight_jobs.pop(job_id, None)
                if tag is not None and tag in self._tag_remaining:
                    self._tag_remaining[tag] -= 1
                    if self._tag_remaining[tag] == 0:
                        del self._tag_remaining[tag]
                        self._tag_completed_set.add(tag)
                        cb = self._tag_callbacks.pop(tag, None) or on_tag_completed
                        if cb is not None:
                            cb(tag)

        def read_batch(self, jobs: list[tuple[int, int, int, int]]) -> None:
            """jobs: list of (fd, buf_ptr, count, offset) tuples.
            Submits all tasks in ONE syscall, reaps all CQEs in-place."""
            n = len(jobs)
            if n == 0:
                return

            for i, (fd, ptr, count, off) in enumerate(jobs):
                sqe = _liburing.io_uring_get_sqe(self._ring)
                if not sqe:
                    _liburing.io_uring_submit_and_wait(self._ring, 0)
                    sqe = _liburing.io_uring_get_sqe(self._ring)
                    if not sqe:
                        raise RuntimeError("io_uring submission queue full")
                _liburing.io_uring_prep_read(sqe, fd, ctypes.c_void_p(ptr), count, off)
                _liburing.io_uring_sqe_set_data64(sqe, i)

            rc = _liburing.io_uring_submit_and_wait(self._ring, n)
            if rc < 0:
                raise RuntimeError(f"io_uring_submit_and_wait failed: {os.strerror(-rc)}")

            reaped = 0
            while reaped < n:
                rc = _liburing.io_uring_wait_cqe(self._ring, ctypes.byref(self._cqe_ptr))
                if rc < 0:
                    raise RuntimeError(f"io_uring_wait_cqe failed: {os.strerror(-rc)}")
                cqe = ctypes.cast(self._cqe_ptr, ctypes.POINTER(_Cqe)).contents
                if cqe.res < 0:
                    raise RuntimeError(f"io_uring read failed on task {cqe.user_data}: {os.strerror(-cqe.res)}")
                _liburing.io_uring_cqe_seen(self._ring, self._cqe_ptr)
                reaped += 1

        def read_batch_grouped(self, jobs, group_sizes, on_group_ready) -> None:
            """Like read_batch, but submits without waiting (`io_uring_submit`) and reaps
            CQEs one at a time exactly as before -- the only difference from read_batch is
            that CQEs aren't all drained before the caller gets control back. `group_sizes`
            partitions `jobs` (e.g. one group per expert, one job per NVMe-backed bank of
            that expert); `on_group_ready(group_index)` fires the instant every job in that
            group has completed, so the caller can dispatch that group's H2D copy while
            other groups' reads are still in flight (RESULTS_AND_FINDINGS.md #38/#39).
            A zero-size group (every bank of that expert already resident, e.g. preloaded)
            has nothing to wait on, so its callback fires immediately, before submission.
            """
            n = len(jobs)
            assert sum(group_sizes) == n
            remaining = list(group_sizes)
            job_group = []
            for g, sz in enumerate(group_sizes):
                job_group.extend([g] * sz)
                if sz == 0:
                    on_group_ready(g)
            if n == 0:
                return

            ratio_env = os.getenv("FREETOKEN_ARTIFICIAL_NVME_RATIO", None)
            if ratio_env is not None:
                try:
                    ratio = float(ratio_env)
                    if ratio <= 0.0:
                        for g in range(len(group_sizes)):
                            on_group_ready(g)
                        return
                    elif ratio < 1.0:
                        tot_bytes = sum(count for _, _, count, _ in jobs)
                        sim_delay = (tot_bytes / 3.13e9) * ratio
                        t_end = time.perf_counter() + sim_delay
                        while time.perf_counter() < t_end:
                            pass
                        for g in range(len(group_sizes)):
                            on_group_ready(g)
                        return
                except Exception:
                    pass

            for i, (fd, ptr, count, off) in enumerate(jobs):
                sqe = _liburing.io_uring_get_sqe(self._ring)
                if not sqe:
                    _liburing.io_uring_submit(self._ring)
                    sqe = _liburing.io_uring_get_sqe(self._ring)
                    if not sqe:
                        raise RuntimeError("io_uring submission queue full")
                _liburing.io_uring_prep_read(sqe, fd, ctypes.c_void_p(ptr), count, off)
                _liburing.io_uring_sqe_set_data64(sqe, i)

            rc = _liburing.io_uring_submit(self._ring)
            if rc < 0:
                raise RuntimeError(f"io_uring_submit failed: {os.strerror(-rc)}")

            reaped = 0
            while reaped < n:
                rc = _liburing.io_uring_wait_cqe(self._ring, ctypes.byref(self._cqe_ptr))
                if rc < 0:
                    raise RuntimeError(f"io_uring_wait_cqe failed: {os.strerror(-rc)}")
                cqe = ctypes.cast(self._cqe_ptr, ctypes.POINTER(_Cqe)).contents
                if cqe.res < 0:
                    raise RuntimeError(f"io_uring read failed on task {cqe.user_data}: {os.strerror(-cqe.res)}")
                job_idx = cqe.user_data
                _liburing.io_uring_cqe_seen(self._ring, self._cqe_ptr)
                reaped += 1
                g = job_group[job_idx]
                remaining[g] -= 1
                if remaining[g] == 0:
                    on_group_ready(g)

        def close(self) -> None:
            if not self._closed and _liburing is not None:
                _liburing.io_uring_queue_exit(self._ring)
                self._closed = True

except Exception as _uring_err:
    FastIoUringBatch = None
    logger.info(f"io_uring not available ({_uring_err}), will use ThreadPoolExecutor")


class SimpleHostLru:
    """Thin OrderedDict wrapper matching SegmentedHostLru's interface, used when
    FREETOKEN_SLRU_HOST_TIER is off -- lets call sites stay policy-agnostic."""

    def __init__(self) -> None:
        self._od: "OrderedDict[tuple[int, int], None]" = OrderedDict()

    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self._od

    def __len__(self) -> int:
        return len(self._od)

    def set_capacity_hint(self, capacity: int) -> None:
        pass

    def insert_new(self, key: tuple[int, int]) -> None:
        self._od[key] = None

    def touch(self, key: tuple[int, int]) -> None:
        self._od.move_to_end(key)

    def pop_lru(self, protect: set[tuple[int, int]] | None = None) -> tuple[tuple[int, int], None]:
        if not protect:
            return self._od.popitem(last=False)
        for k in self._od:
            if k not in protect:
                del self._od[k]
                return k, None
        return self._od.popitem(last=False)

    def discard(self, key: tuple[int, int]) -> None:
        self._od.pop(key, None)


class SegmentedHostLru:
    """Item 2 (RESULTS_AND_FINDINGS.md #40): 2Q/SLRU dynamic host tier. Naive single-queue
    LRU evicts a recurring expert the instant it's least-recently-used, even if it had
    already proven itself worth a second reference -- live measurement (#40.2) found 7.2%
    of dynamic-tier evictions under naive LRU had 2+ prior hits. This splits the dynamic
    pool into a probationary queue (first-time misses) and a protected queue (promoted on
    a 2nd reference), capped at `protected_ratio` of total dynamic capacity so protected
    can't swallow the whole pool and starve eviction candidates. Eviction always drains
    probationary's LRU end first, falling back to protected's LRU end only once
    probationary is empty -- so a genuine one-hit wonder still gets evicted quickly, it
    just can't take a proven-recurring expert down with it.

    Physical host slots are shared across both queues (this cache is slot-indexed, not
    partitioned by address range), so promotion/demotion only moves dict entries, never
    memory.
    """

    def __init__(self, protected_ratio: float = 0.8) -> None:
        self.probation: "OrderedDict[tuple[int, int], None]" = OrderedDict()
        self.protected: "OrderedDict[tuple[int, int], None]" = OrderedDict()
        self._protected_ratio = protected_ratio
        self._capacity: int | None = None

    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self.probation or key in self.protected

    def __len__(self) -> int:
        return len(self.probation) + len(self.protected)

    def set_capacity_hint(self, capacity: int) -> None:
        """Fix total dynamic-tier slot count once known -- constant after startup pinning
        completes, so only needs setting once (first caller wins)."""
        if self._capacity is None:
            self._capacity = capacity

    def _protected_cap(self) -> int | None:
        if self._capacity is None:
            return None
        return max(1, int(self._capacity * self._protected_ratio))

    def insert_new(self, key: tuple[int, int]) -> None:
        self.probation[key] = None

    def touch(self, key: tuple[int, int]) -> None:
        """Record a hit: promote probation -> protected on the 2nd reference; a key
        already in protected just gets its recency refreshed there."""
        if key in self.protected:
            self.protected.move_to_end(key)
            return
        if key in self.probation:
            del self.probation[key]
            cap = self._protected_cap()
            if cap is not None and len(self.protected) >= cap and self.protected:
                demoted, _ = self.protected.popitem(last=False)
                self.probation[demoted] = None
            self.protected[key] = None

    def pop_lru(self, protect: set[tuple[int, int]] | None = None) -> tuple[tuple[int, int], None]:
        if not protect:
            if self.probation:
                return self.probation.popitem(last=False)
            return self.protected.popitem(last=False)
        for k in self.probation:
            if k not in protect:
                del self.probation[k]
                return k, None
        for k in self.protected:
            if k not in protect:
                del self.protected[k]
                return k, None
        if self.probation:
            return self.probation.popitem(last=False)
        return self.protected.popitem(last=False)

    def discard(self, key: tuple[int, int]) -> None:
        self.probation.pop(key, None)
        self.protected.pop(key, None)


class LayerBankLoc:
    # fd_direct: the O_DIRECT fd for per-expert row reads (RESULTS_AND_FINDINGS.md #42).
    # Equals `fd` when O_DIRECT is off or failed its alignment check, so per-expert read
    # sites can use it unconditionally and get transparent buffered fallback.
    __slots__ = ("fd", "fd_direct", "offset_in_shard", "row_bytes", "total_bytes")

    def __init__(self, fd: int, offset_in_shard: int, row_bytes: int, total_bytes: int,
                 fd_direct: int | None = None):
        self.fd = fd
        self.fd_direct = fd if fd_direct is None else fd_direct
        self.offset_in_shard = offset_in_shard
        self.row_bytes = row_bytes
        self.total_bytes = total_bytes


def _pread_all(fd: int, buf: memoryview, offset: int, count: int) -> None:
    """Read exactly count bytes, looping across the Linux kernel 2GB preadv VFS ceiling."""
    total = 0
    while total < count:
        n = os.preadv(fd, [buf[total:count]], offset + total)
        if n <= 0:
            raise RuntimeError(f"preadv EOF after {total}/{count} bytes at offset {offset}")
        total += n


def _read_job(job: tuple) -> None:
    _pread_all(job[0], job[1], job[2], job[3])


class NvmeOffloadMoeCache(OffloadMoeCache):
    """3-Tier MoE Expert Cache with bounded pinned host RAM and NVMe backing."""

    def __init__(
        self,
        model_path: str,
        num_layers: int,
        num_experts: int,
        cache_size: int,
        device: torch.device,
        host_cache_size: int = 288,
        quant_format: str = "nvfp4",
        cache_policy: str = "lru",
    ):
        # Validate geometry
        if cache_size < num_experts:
            raise ValueError(f"cache_size {cache_size} < num_experts {num_experts}")

        self.model_path = model_path
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.cache_size = cache_size
        self.host_cache_size = host_cache_size
        self.device = device
        self.quant_format = quant_format
        self.cache_policy = cache_policy

        self.prefill_overlap = False
        self.prefill_hit_d2d = False
        self.decode_target = "gpu"
        self.hybrid_max_fetch = -1
        self.hybrid_fetch_fraction = 0.0
        self.cpu_layer_ids = []
        self._unpinned_layers = frozenset()
        self.layer_residency = ["PINNED"] * num_layers

        self.bank_schema = _BANK_SCHEMAS[self.quant_format]
        self.gate_up_alpha = None
        self.down_alpha = None
        self.collect_stats = False
        self.collect_decode_freq = os.getenv("FREETOKEN_COLLECT_ROUTING", "1") == "1"
        self.decode_freq = torch.zeros(
            (self.num_layers, self.num_experts), dtype=torch.int64, device=self.device
        )
        self._initial_freq: torch.Tensor | None = None
        _model_basename = os.path.basename(self.model_path.rstrip("/")).lower()
        _default_freq_name = (
            "glm5_routing_freq.pt"
            if "glm" in _model_basename
            else "qwen38_routing_freq.pt"
        )
        _default_freq_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), _default_freq_name
        )
        self._freq_path = os.getenv("FREETOKEN_ROUTING_FREQ_PATH", _default_freq_path)
        if self.collect_decode_freq and os.path.exists(self._freq_path):
            try:
                prior = torch.load(self._freq_path, map_location="cpu", weights_only=True)
                if prior.shape == (self.num_layers, self.num_experts):
                    self._initial_freq = prior
            except Exception:
                pass
        atexit.register(self.close)

        # Bookkeeping structures for GPU slot cache
        self.slot_for_id = torch.full(
            (self.num_layers, self.num_experts), -1, dtype=torch.int32, device=self.device
        )
        self.id_of_slot = torch.full(
            (self.cache_size,), -1, dtype=torch.int32, device=self.device
        )
        self.usage = torch.zeros((self.cache_size,), dtype=torch.int64, device=self.device)
        self.step = torch.zeros((), dtype=torch.int64, device=self.device)
        self.active_mask = torch.zeros((self.num_experts,), dtype=torch.int32, device=self.device)

        # GPU VRAM Frequency Pinning (Item 1 from Haberstroh analysis)
        self.gpu_pinned_per_layer = int(os.getenv("FREETOKEN_GPU_PINNED_PER_LAYER", "4"))
        max_gpu_pinned_total = max(0, self.cache_size - 64)
        max_pin_k = max_gpu_pinned_total // self.num_layers
        self.gpu_pinned_per_layer = min(self.gpu_pinned_per_layer, max_pin_k)
        self.num_gpu_pinned = self.gpu_pinned_per_layer * self.num_layers
        self.num_dynamic = self.cache_size - self.num_gpu_pinned
        self.gpu_pinned_experts: set[tuple[int, int]] = set()
        self._gpu_pinned_list: list[list[int]] = [[] for _ in range(self.num_layers)]
        self.id_of_slot_dynamic = self.id_of_slot[:self.num_dynamic]
        self.usage_dynamic = self.usage[:self.num_dynamic]

        # Zero-sync device-side host slot remapping (Phase 2 from Haberstroh analysis)
        self._zero_sync_host_remap = os.getenv("FREETOKEN_ZERO_SYNC_HOST_REMAP", "0") == "1"
        self.expert_to_host_slot = torch.full(
            (self.num_layers, self.num_experts), -1, dtype=torch.int32, device=self.device
        )
        self._topk_pinned = torch.empty(self.num_experts, dtype=torch.int32, pin_memory=True)
        self._topk_event = torch.cuda.Event()

        plan_slots = max(self.num_experts, self.cache_size)
        self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.num_indices = torch.zeros((1,), dtype=torch.int64, device=self.device)
        self.num_missing_full = torch.zeros((1,), dtype=torch.int64, device=self.device)
        self.expert_recency = torch.full(
            (self.num_layers, self.num_experts), -1, dtype=torch.int64, device=self.device
        )
        self.lru_stats = torch.zeros(
            (self.num_layers, N_STATS), dtype=torch.int64, device=self.device
        )

        self._pending_src_layer: int | None = None
        self._pending_whole_layer = False

        # Parse FTW manifest and open shard fds
        self._init_nvme_manifest()

        # Preload tiny global scale banks (gate_up_global, down_global: ~187 MB total)
        self._init_preloaded_global_banks()

        # Allocate pinned host memory banks (Tier 2)
        self._init_host_banks()

        # Allocate GPU VRAM slot cache (Tier 1)
        self._init_gpu_cache()

        # Build fused copy plan (pointing to our host banks)
        self._build_fused_plan()

        # Initialize Host LRU state
        self.host_slot_for_expert: dict[tuple[int, int], int] = {}
        self.expert_for_host_slot: list[tuple[int, int] | None] = [None] * self.host_cache_size
        self._slru_host_tier = os.getenv("FREETOKEN_SLRU_HOST_TIER", "0") == "1"
        if self._slru_host_tier:
            protected_ratio = float(os.getenv("FREETOKEN_SLRU_PROTECTED_RATIO", "0.8"))
            self.host_lru: SegmentedHostLru | SimpleHostLru = SegmentedHostLru(protected_ratio)
        else:
            self.host_lru = SimpleHostLru()
        # Phase E: Permanently pinned host cache tier (immune to LRU eviction)
        self.pinned_experts: set[tuple[int, int]] = set()
        # When prefill-narrowing is active, materialize_layer's identity slot range [0, num_experts)
        # is never used, so all host cache slots can be pooled into free_host_slots.
        start_host_slot = 0 if os.getenv("FREETOKEN_PREFILL_NARROW", "1") == "1" else self.num_experts
        self.free_host_slots = list(reversed(range(start_host_slot, self.host_cache_size)))

        # Performance counters
        self.stat_nvme_reads = 0
        self.stat_host_hits = 0
        self.stat_pinned_hits = 0
        self.stat_gpu_hits = 0
        self.stat_total_requests = 0

        # Telemetry pacing: 1 token = num_layers * active_experts_per_token
        is_glm = self.num_layers == 42 or "glm" in self.model_path.lower()
        topk_ovr = os.getenv("FREETOKEN_TOPK_OVERRIDE", os.getenv("FREETOKEN_GLM_TOPK", None))
        if is_glm and topk_ovr is not None:
            try:
                self._telemetry_step = self.num_layers * int(topk_ovr)
            except ValueError:
                self._telemetry_step = self.num_layers * 8
        elif is_glm:
            self._telemetry_step = self.num_layers * 8  # 336
        else:
            self._telemetry_step = 480
        self._next_telemetry_threshold = self._telemetry_step

        # Fine-grained latency breakdown accumulators
        self.time_nvme_read_s = 0.0
        self.bytes_nvme_read = 0
        self.time_host_bookkeeping_s = 0.0
        self.time_gpu_lru_s = 0.0
        self.time_h2d_copy_s = 0.0

        # Toggleable (default off) instrumentation for the .item()/.cpu().tolist() GPU-queue-
        # drain cost specifically -- separate from time_gpu_lru_s (which only wraps the async
        # kernel launch, not the sync) and from time_nvme_read_s (measured after the sync
        # already returned). See RESULTS_AND_FINDINGS.md #19 for why this exists.
        self._trace_sync = os.getenv("FREETOKEN_TRACE_SYNC", "0") == "1"
        self.time_item_sync_s = 0.0
        self.time_item_sync_zero_miss_s = 0.0
        self.time_item_sync_with_miss_s = 0.0
        self.count_zero_miss_layers = 0
        self.count_miss_layers = 0
        self.time_cpu_tolist_s = 0.0  # kept at 0.0; .cpu().tolist() was eliminated (see #20)

        # Async sync removal (RESULTS_AND_FINDINGS.md #20): one event-gated pinned D2H
        # transfer replaces the previous two separate blocking round-trips
        # (.item() then .cpu().tolist()). Pinned staging buffers are sized lazily on the
        # first ensure_experts call (top_k is constant across calls at this deployment's
        # decode batch size of 1).
        self._sync_event = torch.cuda.Event()
        self._num_indices_pinned: torch.Tensor | None = None
        self._src_indices_pinned: torch.Tensor | None = None
        self._evict_slots_pinned: torch.Tensor | None = None

        # Phase A (RESULTS_AND_FINDINGS.md #22): copy_missing's H2D scatter-gather cost,
        # via deferred CUDA events (flushed at the same periodic point as the telemetry
        # log below) so this doesn't add its own sync per layer. time_h2d_copy_s was
        # previously declared but never populated -- a dead duplicate `copy_missing`
        # definition shadowed the one that actually ran; fixed by merging into the one
        # definition that's live.
        self._h2d_pending: list = []  # (start_ev, end_ev, had_miss) tuples
        self._last_num_misses = 0
        self.time_h2d_copy_zero_miss_s = 0.0
        self.time_h2d_copy_with_miss_s = 0.0
        self.count_h2d_zero_miss = 0
        self.count_h2d_with_miss = 0

        # I/O Backend: io_uring vs threadpool
        # Default: iouring (zero-copy single-syscall DMA into pinned PyTorch tensors)
        self.io_backend_name = os.getenv("FREETOKEN_IO_BACKEND", "iouring").lower()
        self.uring = None
        self.io_pool = None

        if self.io_backend_name == "iouring" and FastIoUringBatch is not None:
            try:
                self.uring = FastIoUringBatch(queue_depth=128)
                logger.info("NvmeOffloadMoeCache: initialized zero-copy io_uring I/O backend (queue_depth=128)")
            except Exception as e:
                logger.warning(f"NvmeOffloadMoeCache: io_uring init failed ({e}), falling back to threadpool")
                self.uring = None

        if self.uring is None:
            io_workers = int(os.getenv("FREETOKEN_IO_WORKERS", "16"))
            self.io_pool = ThreadPoolExecutor(max_workers=io_workers)
            logger.info(f"NvmeOffloadMoeCache: initialized ThreadPoolExecutor I/O backend ({io_workers} workers)")

        self.qd_hist = {"0": 0, "1_4": 0, "5_8": 0, "9_16": 0, "gt16": 0}

        # Item 1 (RESULTS_AND_FINDINGS.md #38/#39): overlap per-expert NVMe reads with
        # their H2D PCIe transfer instead of waiting for the whole layer's miss batch to
        # land before starting any H2D copy. Only wired up for the io_uring backend --
        # ThreadPoolExecutor's blocking .map()/single-job path has no per-completion hook
        # to dispatch against. Default off pending live A/B verification.
        self._pipeline_h2d = os.getenv("FREETOKEN_PIPELINE_H2D", "0") == "1" and self.uring is not None
        if self._pipeline_h2d:
            self._h2d_stream = torch.cuda.Stream()
            self._h2d_pipeline_event = torch.cuda.Event()
            self._pending_h2d_pipelined = False
            logger.info("NvmeOffloadMoeCache: FREETOKEN_PIPELINE_H2D=1, overlapping per-expert NVMe reads with H2D copy")

        # Item 2 investigation (improvements.md, RESULTS_AND_FINDINGS.md #40): does naive
        # LRU actually evict recurring dynamic-tier experts before they'd earn a second
        # hit? Tracks hits-since-insertion per dynamic-tier key, bucketed at eviction time.
        # A large "2+" bucket means real experts that had already proven they'd be reused
        # were evicted anyway -- justifying an SLRU/2Q split. A large "0" bucket means
        # naive LRU is already doing the right thing (correctly identifying one-hit
        # wonders) and a segmented tier would mostly add bookkeeping overhead for little
        # hit-rate gain.
        self._trace_host_evict = os.getenv("FREETOKEN_TRACE_HOST_EVICT", "0") == "1"
        self._host_access_count: dict[tuple[int, int], int] = {}
        self._evict_hist = {"0": 0, "1": 0, "2+": 0}

        # Re-prioritized item (improvements.md): confirm whether materialize_layer's
        # per-layer NVMe/preloaded read cost during prefill is close to constant
        # regardless of prompt length (leading hypothesis for §23's flat-TTFT finding).
        # Logs one line per materialize_layer call: layer_id, this forward's new-token
        # count (batch.log_new_tokens), bytes moved, and wall time -- flat bytes/time
        # across varying log_new_tokens confirms the full-pool-read mechanism.
        self._trace_materialize = os.getenv("FREETOKEN_TRACE_MATERIALIZE", "0") == "1"

        # Item 3 substitute (RESULTS_AND_FINDINGS.md #41): layer L+1's *exact* routed
        # experts genuinely can't be known before L's own router runs (a same-layer data
        # dependency, not an implementation gap -- see improvements.md's Item 3 note), so
        # instead of prefetching the exact set, prefetch a *speculative* candidate set
        # seeded from the existing qwen38_routing_freq.pt calibration data (self._initial_freq,
        # already loaded above for prewarm_cache). Reads run on a dedicated background
        # thread pool that never touches host_slot_for_expert/host_lru/free_host_slots
        # directly -- only _drain_speculative, called on the main thread at the top of
        # every resolve_experts_in_host_batched, makes a finished read visible to the rest
        # of the cache, so there's no cross-thread race on cache bookkeeping. Strictly
        # opportunistic: only claims genuinely free slots, never evicts a resident expert
        # to make room for a guess. Prefill-only in practice (only
        # _install_prefill_narrow_patch calls maybe_speculative_prefetch); decode never
        # calls it, so _drain_speculative is a no-op fast path there.
        # Always-present (possibly always-empty) so resolve_experts_in_host_batched and
        # _drain_speculative can check them unconditionally without an enabled-flag branch
        # on every call.
        self._speculative_pending: set[tuple[int, int]] = set()
        self._speculative_slots: dict[tuple[int, int], int] = {}
        self._speculative_futures: dict[tuple[int, int], Any] = {}
        self._speculative_pool: ThreadPoolExecutor | None = None
        self.stat_speculative_prefetched = 0
        self.stat_speculative_hits = 0

        self._speculative_prefetch_enabled = (
            os.getenv("FREETOKEN_SPECULATIVE_PREFETCH", "0") == "1" and self._initial_freq is not None
        )
        if self._speculative_prefetch_enabled:
            spec_workers = int(os.getenv("FREETOKEN_SPECULATIVE_WORKERS", "4"))
            self._speculative_k = int(os.getenv("FREETOKEN_SPECULATIVE_K", "16"))
            self._speculative_pool = ThreadPoolExecutor(max_workers=spec_workers)
            logger.info(
                f"NvmeOffloadMoeCache: FREETOKEN_SPECULATIVE_PREFETCH=1, speculative "
                f"layer-ahead prefill prefetch (K={self._speculative_k}, {spec_workers} workers)"
            )
        elif os.getenv("FREETOKEN_SPECULATIVE_PREFETCH", "0") == "1":
            logger.warning(
                "NvmeOffloadMoeCache: FREETOKEN_SPECULATIVE_PREFETCH=1 but no routing "
                f"calibration found at {self._freq_path}; speculative prefetch disabled"
            )

        if self._o_direct:
            n_direct = sum(1 for loc in self.bank_locations.values() if loc.fd_direct != loc.fd)
            logger.info(
                f"NvmeOffloadMoeCache: FREETOKEN_O_DIRECT=1, per-expert reads bypass the page "
                f"cache ({n_direct}/{len(self.bank_locations)} banks aligned)"
            )

        self.expert_bytes = sum(self.bank_locations[(0, n)].row_bytes for n in self.bank_schema)
        logger.info(
            f"NvmeOffloadMoeCache initialized: GPU cache={self.cache_size} slots "
            f"({self.cache_size * self.expert_bytes / (1024**3):.2f} GiB approx), "
            f"Host cache={self.host_cache_size} slots, {self.num_layers} MoE layers on NVMe"
        )

        if os.getenv("FREETOKEN_PREWARM_CACHE", "0") == "1":
            self.prewarm_cache()

        # Dynamic-tier capacity is fixed now: prewarm's pinning (if any) already ran, and
        # nothing after this point changes how many slots are reachable by the LRU/SLRU
        # policy -- only which specific slots cycle through it.
        self.host_lru.set_capacity_hint(len(self.free_host_slots))
        self.prefetcher = None
        if os.getenv("FREETOKEN_SPECULATIVE_LOOKAHEAD", "0") == "1":
            self.prefetcher = RollingLookaheadPrefetcher(self)

    def _init_nvme_manifest(self) -> None:
        manifest_path = os.path.join(self.model_path, "freetoken_weight.json")
        with open(manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.shards = sorted(self.manifest["shards"], key=lambda s: s["global_off"])
        self.shard_offsets = [s["global_off"] for s in self.shards]
        self.shard_fds: dict[str, int] = {}
        # Per-expert reads go through a second, O_DIRECT fd set (RESULTS_AND_FINDINGS.md
        # #42): bench_read_shape.py measured the decode read shape at 2.19 GB/s buffered
        # vs 3.78 GB/s O_DIRECT (1.72x), because a buffered read pays a page-cache ->
        # pinned-buffer memcpy per byte and flatlines at ~2.2 GB/s regardless of queue
        # depth. Whole-bank reads (global preload, materialize_layer) keep the buffered
        # fd -- their buffers aren't alignment-guaranteed and they aren't on the hot path.
        self.shard_fds_direct: dict[str, int] = {}
        self._o_direct = os.getenv("FREETOKEN_O_DIRECT", "0") == "1"
        for s in self.shards:
            path = os.path.join(self.model_path, s["file"])
            self.shard_fds[s["file"]] = os.open(path, os.O_RDONLY)
            if self._o_direct:
                try:
                    self.shard_fds_direct[s["file"]] = os.open(path, os.O_RDONLY | os.O_DIRECT)
                except OSError as e:
                    logger.warning(f"O_DIRECT open failed for {s['file']} ({e}); using buffered reads")
                    self._o_direct = False
                    break

        # Map (layer_id, bank_name) -> LayerBankLoc
        self.bank_locations: dict[tuple[int, str], LayerBankLoc] = {}
        self.bank_shapes: dict[str, tuple[int, ...]] = {}
        self.bank_dtypes: dict[str, torch.dtype] = {}

        tensors_by_name = {t["name"]: t for t in self.manifest["tensors"] if t["kind"] == "experts_bank"}
        for layer_id in range(self.num_layers):
            for bank_name in self.bank_schema:
                entry_name = f"{bank_name}#L{layer_id:05d}"
                entry = tensors_by_name.get(entry_name)
                if entry is None:
                    raise KeyError(f"FTW manifest missing expert entry {entry_name}")

                global_off = entry["global_off"]
                idx = bisect.bisect_right(self.shard_offsets, global_off) - 1
                shard = self.shards[idx]
                off_in_shard = global_off - shard["global_off"]
                row_shape = tuple(entry["shape"][1:])
                num_rows = entry["shape"][0]
                row_bytes = entry["nbytes"] // num_rows

                fd = self.shard_fds[shard["file"]]
                # O_DIRECT requires the offset AND length of every read to be block-aligned
                # (the destination buffer too -- checked once in _init_host_banks). Any bank
                # that doesn't qualify silently keeps its buffered fd rather than failing
                # reads with EINVAL at serving time.
                fd_direct = fd
                if self._o_direct and off_in_shard % _O_DIRECT_ALIGN == 0 and row_bytes % _O_DIRECT_ALIGN == 0:
                    fd_direct = self.shard_fds_direct[shard["file"]]
                self.bank_locations[(layer_id, bank_name)] = LayerBankLoc(
                    fd, off_in_shard, row_bytes, entry["nbytes"], fd_direct
                )

                if bank_name not in self.bank_shapes:
                    self.bank_shapes[bank_name] = row_shape
                    # Map dtype string to torch dtype
                    dt_str = entry["dtype"]
                    dt_map = {
                        "uint8": torch.uint8,
                        "float8_e4m3fn": torch.float8_e4m3fn,
                        "float16": torch.float16,
                        "bfloat16": torch.bfloat16,
                        "float32": torch.float32,
                    }
                    self.bank_dtypes[bank_name] = dt_map[dt_str]

    def _init_preloaded_global_banks(self) -> None:
        """Preload the tiny global scale banks (gate_up_global 62MB, down_global 125MB) into RAM."""
        self.preloaded_banks: dict[str, bytearray] = {}
        for name in ("gate_up_global", "down_global"):
            if name in self.bank_schema:
                t0 = time.perf_counter()
                bank_data = bytearray()
                for layer_id in range(self.num_layers):
                    loc = self.bank_locations[(layer_id, name)]
                    buf = bytearray(loc.total_bytes)
                    _pread_all(loc.fd, memoryview(buf), loc.offset_in_shard, loc.total_bytes)
                    bank_data.extend(buf)
                self.preloaded_banks[name] = bank_data
                dur = time.perf_counter() - t0
                logger.info(
                    f"Preloaded {name} across all {self.num_layers} layers "
                    f"({len(bank_data)/(1024*1024):.1f} MB in RAM, {dur:.3f}s)"
                )

    def _init_host_banks(self) -> None:
        # Guarantee physical RAM safety:
        # 1. Desktop environment consumes ~5.0 - 6.0 GiB.
        # 2. FreeToken model loader transiently reads ~9.2 GiB off disk during startup.
        # 3. Reserve headroom so free RAM never drops to megabytes:
        # At 5,120 slots (13.22 GiB pinned), raw free RAM is measured at 1.7 GiB.
        # At 5,632 slots (+1.35 GiB pinned), raw free RAM collapses to 229 MB (danger zone).
        # Therefore, 5,120 slots is the hard physical ceiling on this 31 GiB machine.
        min_system_reserve_bytes = int(17.5 * 1024 * 1024 * 1024)
        total_ram_bytes = 0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_ram_bytes = int(line.split()[1]) * 1024
                        break
        except Exception:
            pass

        slot_bytes = 0
        for name in self.bank_schema:
            numel = 1
            for d in self.bank_shapes[name]:
                numel *= d
            elem_sz = torch.empty((), dtype=self.bank_dtypes[name]).element_size()
            slot_bytes += numel * elem_sz

        if total_ram_bytes > 0 and slot_bytes > 0:
            max_safe_bytes = max(0, total_ram_bytes - min_system_reserve_bytes)
            max_safe_slots = min(5120, max(self.num_experts, max_safe_bytes // slot_bytes))
            force_size = os.getenv("FREETOKEN_FORCE_HOST_CACHE_SIZE", "0") == "1"
            if not force_size and self.host_cache_size > max_safe_slots:
                logger.warning(
                    f"Requested host_cache_size {self.host_cache_size} exceeds safe physical RAM limit. "
                    f"Clamping to {max_safe_slots} slots to preserve at least 1.5 GiB free RAM."
                )
                self.host_cache_size = max_safe_slots

        self.host_banks: dict[str, torch.Tensor] = {}
        self.host_banks_np: dict[str, Any] = {}
        for name in self.bank_schema:
            shape = (self.host_cache_size, *self.bank_shapes[name])
            dtype = self.bank_dtypes[name]
            # Allocate pinned host memory
            t = torch.empty(shape, dtype=dtype, pin_memory=True)
            self.host_banks[name] = t
            # View as flat uint8 numpy buffer for byte-exact preadv direct I/O
            self.host_banks_np[name] = t.view(torch.uint8).flatten().numpy()
        self.host_bank_ptrs: dict[str, int] = {
            name: self.host_banks[name].data_ptr() for name in self.bank_schema
        }
        # O_DIRECT's third alignment requirement (after offset and length): the destination
        # buffer address. Pinned allocations are page-aligned in practice, but verify rather
        # than assume -- an unaligned buffer fails every read with EINVAL at serving time,
        # so fall back to buffered reads instead if this doesn't hold.
        if self._o_direct:
            bad = [n for n, p in self.host_bank_ptrs.items() if p % _O_DIRECT_ALIGN]
            if bad:
                logger.warning(
                    f"O_DIRECT disabled: pinned host banks {bad} are not "
                    f"{_O_DIRECT_ALIGN}-byte aligned"
                )
                self._o_direct = False
                for loc in self.bank_locations.values():
                    loc.fd_direct = loc.fd

    def _init_gpu_cache(self) -> None:
        self.bank_caches: dict[str, torch.Tensor] = {}
        for name in self.bank_schema:
            shape = (self.cache_size, *self.bank_shapes[name])
            dtype = self.bank_dtypes[name]
            self.bank_caches[name] = torch.empty(shape, dtype=dtype, device=self.device)
        self.banks = [(None, self.bank_caches[name]) for name in self.bank_schema]

    def _build_fused_plan(self) -> None:
        dst_ptrs = []
        feat_bytes = []
        host_src_ptrs = []
        for name in self.bank_schema:
            cache = self.bank_caches[name]
            host_t = self.host_banks[name]
            feat = math.prod(self.bank_shapes[name]) * cache.element_size()
            dst_ptrs.append(cache.data_ptr())
            feat_bytes.append(feat)
            host_src_ptrs.append(device_ptr(host_t))

        self._copy_dst_ptrs = torch.tensor(dst_ptrs, dtype=torch.int64, device=self.device)
        self._copy_feat_bytes = torch.tensor(feat_bytes, dtype=torch.int64, device=self.device)
        # All layers share the same host bank base pointers
        self._copy_src_ptrs = [
            torch.tensor(host_src_ptrs, dtype=torch.int64, device=self.device)
            for _ in range(self.num_layers)
        ]
        self._copy_fused_ok = True

    def is_cpu_layer(self, layer_id: int) -> bool:
        return False

    def bank_views(self, n: int | None = None) -> tuple[torch.Tensor, ...]:
        if n is None:
            return tuple(self.bank_caches[name] for name in self.bank_schema)
        return tuple(self.bank_caches[name][:n] for name in self.bank_schema)

    def alphas_for_slots(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        return None

    def alphas_for_layer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        return None

    def resolve_expert_in_host(self, layer_id: int, expert_id: int) -> int:
        t_start = time.perf_counter()
        key = (layer_id, expert_id)
        if key in self.host_slot_for_expert:
            slot = self.host_slot_for_expert[key]
            if key in self.pinned_experts:
                self.stat_host_hits += 1
                self.stat_pinned_hits += 1
                self.time_host_bookkeeping_s += (time.perf_counter() - t_start)
                return slot
            self.host_lru.touch(key)
            self.stat_host_hits += 1
            self.time_host_bookkeeping_s += (time.perf_counter() - t_start)
            return slot

        # NVMe Miss -> Fetch into Host Slot
        self.stat_nvme_reads += 1
        if self.free_host_slots:
            slot = self.free_host_slots.pop()
        else:
            evict_key, _ = self.host_lru.pop_lru()
            slot = self.host_slot_for_expert.pop(evict_key)
            self.expert_for_host_slot[slot] = None

        t_pre_io = time.perf_counter()
        self.time_host_bookkeeping_s += (t_pre_io - t_start)

        t_io_0 = time.perf_counter()
        # Direct I/O read across all 6 banks into host slot
        for name in self.bank_schema:
            loc = self.bank_locations[(layer_id, name)]
            dest_slice = memoryview(self.host_banks_np[name])[
                slot * loc.row_bytes : (slot + 1) * loc.row_bytes
            ]
            if name in self.preloaded_banks:
                src_off = layer_id * (self.num_experts * loc.row_bytes) + expert_id * loc.row_bytes
                dest_slice[:] = memoryview(self.preloaded_banks[name])[src_off : src_off + loc.row_bytes]
            else:
                off = loc.offset_in_shard + expert_id * loc.row_bytes
                _pread_all(loc.fd_direct, dest_slice, off, loc.row_bytes)
                self.bytes_nvme_read += loc.row_bytes
        t_io_1 = time.perf_counter()
        self.time_nvme_read_s += (t_io_1 - t_io_0)

        t_post_0 = time.perf_counter()
        self.host_slot_for_expert[key] = slot
        self.expert_for_host_slot[slot] = key
        self.host_lru.insert_new(key)
        self.time_host_bookkeeping_s += (time.perf_counter() - t_post_0)
        return slot

    def _build_expert_jobs(
        self, layer_id: int, expert_id: int, slot: int, tag: Any
    ) -> list[tuple[Any, int, int, int, int]]:
        """Returns list of (tag, fd, ptr, count, off) for NVMe-backed banks of this expert.
        Preloaded banks are copied synchronously here into host_banks_np."""
        jobs = []
        for name in self.bank_schema:
            loc = self.bank_locations[(layer_id, name)]
            if name in self.preloaded_banks:
                dest_slice = memoryview(self.host_banks_np[name])[
                    slot * loc.row_bytes : (slot + 1) * loc.row_bytes
                ]
                src_off = layer_id * (self.num_experts * loc.row_bytes) + expert_id * loc.row_bytes
                dest_slice[:] = memoryview(self.preloaded_banks[name])[src_off : src_off + loc.row_bytes]
            else:
                off = loc.offset_in_shard + expert_id * loc.row_bytes
                ptr = self.host_bank_ptrs[name] + slot * loc.row_bytes
                jobs.append((tag, loc.fd_direct, ptr, loc.row_bytes, off))
        return jobs

    def _pipelined_fetch(
        self, layer_id: int, host_slots: list[int], fetch_needed: list[tuple[int, int, int]]
    ) -> None:
        """Item 1 (RESULTS_AND_FINDINGS.md #38/#39): overlap NVMe reads with H2D PCIe
        transfer instead of waiting for the whole miss batch before starting any copy.

        `host_slots[i]` is already the final host slot for position `i` of this layer's
        miss set (hit or NVMe-bound alike -- slot assignment is pure bookkeeping done
        above, independent of when bytes actually land). `self.evict_slots[:n]` (the GPU
        destination cache slot per position) was already populated by base_ensure_experts
        before this call ran. So the only thing gated on I/O completion is issuing the
        per-position H2D copy -- everything else is already known.

        Positions with no NVMe dependency (host-cache hits) are copied immediately, on
        the dedicated H2D stream, before I/O is even submitted. Positions that do need a
        disk read get their copy dispatched, also on the H2D stream, the instant every
        bank of that expert has landed (`FastIoUringBatch.read_batch_grouped`'s
        per-group callback) -- so later experts' reads keep running concurrently with
        earlier experts' PCIe transfer. `copy_missing` (called once this whole layer's
        `ensure_experts` returns) just waits on the H2D stream instead of re-issuing the
        fused whole-batch kernel.
        """
        n = len(host_slots)
        self.src_indices[:n].copy_(
            torch.tensor(host_slots, dtype=torch.int32, device=self.device)
        )

        fetch_positions = {pos for pos, _, _ in fetch_needed}
        hit_positions = [i for i in range(n) if i not in fetch_positions]
        if hit_positions:
            idx = torch.tensor(hit_positions, dtype=torch.int64, device=self.device)
            with torch.cuda.stream(self._h2d_stream):
                fast_index_copy_multi_jit(
                    self._copy_dst_ptrs, self._copy_src_ptrs[layer_id], self._copy_feat_bytes,
                    self.evict_slots[idx], self.src_indices[idx], None,
                )

        jobs = []
        group_sizes = []
        group_positions = []
        bytes_this_batch = 0
        for pos, expert_id, slot in fetch_needed:
            n_before = len(jobs)
            for name in self.bank_schema:
                loc = self.bank_locations[(layer_id, name)]
                if name in self.preloaded_banks:
                    dest_slice = memoryview(self.host_banks_np[name])[
                        slot * loc.row_bytes : (slot + 1) * loc.row_bytes
                    ]
                    src_off = layer_id * (self.num_experts * loc.row_bytes) + expert_id * loc.row_bytes
                    dest_slice[:] = memoryview(self.preloaded_banks[name])[src_off : src_off + loc.row_bytes]
                else:
                    off = loc.offset_in_shard + expert_id * loc.row_bytes
                    ptr = self.host_bank_ptrs[name] + slot * loc.row_bytes
                    jobs.append((loc.fd_direct, ptr, loc.row_bytes, off))
                    bytes_this_batch += loc.row_bytes
            group_sizes.append(len(jobs) - n_before)
            group_positions.append(pos)

        n_t = len(jobs)
        if n_t == 0:
            self.qd_hist["0"] += 1
        elif n_t <= 4:
            self.qd_hist["1_4"] += 1
        elif n_t <= 8:
            self.qd_hist["5_8"] += 1
        elif n_t <= 16:
            self.qd_hist["9_16"] += 1
        else:
            self.qd_hist["gt16"] += 1

        def on_group_ready(group_idx: int) -> None:
            pos = group_positions[group_idx]
            with torch.cuda.stream(self._h2d_stream):
                fast_index_copy_multi_jit(
                    self._copy_dst_ptrs, self._copy_src_ptrs[layer_id], self._copy_feat_bytes,
                    self.evict_slots[pos : pos + 1], self.src_indices[pos : pos + 1], None,
                )

        t_io_0 = time.perf_counter()
        self.uring.read_batch_grouped(jobs, group_sizes, on_group_ready)
        self.time_nvme_read_s += (time.perf_counter() - t_io_0)
        self.bytes_nvme_read += bytes_this_batch

        self._h2d_pipeline_event.record(self._h2d_stream)
        self._pending_h2d_pipelined = True

    def _speculative_read_job(self, layer_id: int, expert_id: int, slot: int) -> None:
        """Runs on a background thread (self._speculative_pool). Touches only raw bytes
        in host_banks_np at an already-reserved slot -- never host_slot_for_expert,
        host_lru, or free_host_slots, all of which stay main-thread-only to avoid a
        cross-thread race with the normal synchronous fetch path."""
        for name in self.bank_schema:
            loc = self.bank_locations[(layer_id, name)]
            dest_slice = memoryview(self.host_banks_np[name])[
                slot * loc.row_bytes : (slot + 1) * loc.row_bytes
            ]
            if name in self.preloaded_banks:
                src_off = layer_id * (self.num_experts * loc.row_bytes) + expert_id * loc.row_bytes
                dest_slice[:] = memoryview(self.preloaded_banks[name])[src_off : src_off + loc.row_bytes]
            else:
                off = loc.offset_in_shard + expert_id * loc.row_bytes
                _pread_all(loc.fd_direct, dest_slice, off, loc.row_bytes)

    def _drain_speculative(self) -> None:
        """Finalize any speculative reads that finished since the last check into live
        cache state -- host_slot_for_expert/host_lru mutations happen here, on the main
        thread only. Called at the top of every resolve_experts_in_host_batched (cheap
        no-op when nothing is pending, i.e. always, during decode)."""
        done_keys = [k for k, fut in self._speculative_futures.items() if fut.done()]
        for key in done_keys:
            fut = self._speculative_futures.pop(key)
            slot = self._speculative_slots.pop(key)
            self._speculative_pending.discard(key)
            exc = fut.exception()
            if exc is not None:
                logger.warning(f"[Speculative Prefetch] read failed for {key}: {exc}")
                self.free_host_slots.append(slot)
                continue
            if key in self.host_slot_for_expert:
                # Already resolved via the wait-on-in-flight-future path in
                # resolve_experts_in_host_batched between when this was marked done and
                # now -- shouldn't happen (that path pops from the same dicts under the
                # same single-threaded call), but if it ever did, don't clobber a live
                # slot assignment or leak the one we just filled.
                self.free_host_slots.append(slot)
                continue
            self.host_slot_for_expert[key] = slot
            self.expert_for_host_slot[slot] = key
            self.host_lru.insert_new(key)
            if self._zero_sync_host_remap:
                self.expert_to_host_slot[key[0], key[1]] = slot

    def maybe_speculative_prefetch(self, layer_id: int) -> None:
        """Item 3 substitute (RESULTS_AND_FINDINGS.md #41): layer L+1's *exact* routed
        experts can't be known before L's own router runs -- a same-layer data
        dependency, not an implementation gap (see improvements.md's Item 3 note, and
        RESULTS_AND_FINDINGS.md #25.1 on why the framework's own prefill overlap was
        disabled for this cache). Speculatively prefetches L+1's top-K
        most-frequently-routed experts per the qwen38_routing_freq.pt calibration data
        instead, on a background thread pool, while L's own H2D copy + GEMM run on the
        GPU -- genuinely idle NVMe time, since ensure_experts (called just before this)
        already blocks until L's own reads are done. Strictly opportunistic: only claims
        slots already in free_host_slots, never evicts a resident expert for a guess.
        """
        if not self._speculative_prefetch_enabled or layer_id >= self.num_layers:
            return
        if self._speculative_pending:
            self._drain_speculative()
        layer_f = self._initial_freq[layer_id]
        k = min(self._speculative_k, self.num_experts)
        # Rank by the SAME calibration data Phase E's prewarm_cache used to choose the
        # permanently-pinned top ~64/layer -- so a plain top-K here would, every time,
        # rank almost entirely inside that already-resident band and issue nothing (this
        # was live-measured: first run issued exactly 0 speculative reads across a whole
        # request). Over-fetch the ranking and let the `key in self.host_slot_for_expert`
        # check below skip past pinned/already-resident entries into the next tier --
        # the ones actually worth speculating on.
        over_k = min(self.num_experts, k + 128)
        ranked = torch.topk(layer_f, over_k).indices.tolist()
        issued = 0
        for expert_id in ranked:
            if issued >= k:
                break
            if layer_f[expert_id] <= 0:
                continue
            key = (layer_id, expert_id)
            if key in self.host_slot_for_expert or key in self._speculative_pending:
                continue
            if not self.free_host_slots:
                break  # opportunistic only -- never evict a resident expert for a guess
            slot = self.free_host_slots.pop()
            self._speculative_pending.add(key)
            self._speculative_slots[key] = slot
            self._speculative_futures[key] = self._speculative_pool.submit(
                self._speculative_read_job, layer_id, expert_id, slot
            )
            self.stat_speculative_prefetched += 1
            issued += 1

    def resolve_experts_in_host_batched(
        self, layer_id: int, missing_experts: list[int], pipeline_h2d: bool = True,
        protect_keys: set[tuple[int, int]] | None = None,
    ) -> list[int]:
        t_start = time.perf_counter()
        host_slots = []
        fetch_needed = []  # (position, expert_id, slot)

        if self._speculative_pending:
            self._drain_speculative()

        for i, expert_id in enumerate(missing_experts):
            key = (layer_id, expert_id)

            if key not in self.host_slot_for_expert and key in self._speculative_pending:
                # A background speculative read for this exact key is already in flight
                # (Item 3, RESULTS_AND_FINDINGS.md #41) -- wait for that one read instead
                # of issuing a second, redundant NVMe fetch for the same expert.
                fut = self._speculative_futures.pop(key)
                slot = self._speculative_slots.pop(key)
                self._speculative_pending.discard(key)
                exc = fut.exception()
                if exc is not None:
                    logger.warning(f"[Speculative Prefetch] read failed for {key}: {exc}")
                    self.free_host_slots.append(slot)
                else:
                    self.host_slot_for_expert[key] = slot
                    self.expert_for_host_slot[slot] = key
                    self.host_lru.insert_new(key)
                    self.stat_speculative_hits += 1
                    if self._zero_sync_host_remap:
                        self.expert_to_host_slot[key[0], key[1]] = slot

            if key in self.host_slot_for_expert:
                slot = self.host_slot_for_expert[key]
                if key in self.pinned_experts:
                    self.stat_host_hits += 1
                    self.stat_pinned_hits += 1
                    host_slots.append(slot)
                else:
                    self.host_lru.touch(key)
                    self.stat_host_hits += 1
                    host_slots.append(slot)
                    if self._trace_host_evict:
                        self._host_access_count[key] = self._host_access_count.get(key, 0) + 1
            else:
                self.stat_nvme_reads += 1
                if self.free_host_slots:
                    slot = self.free_host_slots.pop()
                else:
                    evict_key, _ = self.host_lru.pop_lru(protect=protect_keys)
                    slot = self.host_slot_for_expert.pop(evict_key)
                    self.expert_for_host_slot[slot] = None
                    if self._zero_sync_host_remap:
                        self.expert_to_host_slot[evict_key[0], evict_key[1]] = -1
                    if self._trace_host_evict:
                        hits = self._host_access_count.pop(evict_key, 0)
                        bucket = "0" if hits == 0 else ("1" if hits == 1 else "2+")
                        self._evict_hist[bucket] += 1

                self.host_slot_for_expert[key] = slot
                self.expert_for_host_slot[slot] = key
                self.host_lru.insert_new(key)
                if self._zero_sync_host_remap:
                    self.expert_to_host_slot[key[0], key[1]] = slot
                host_slots.append(slot)
                fetch_needed.append((i, expert_id, slot))
                if self._trace_host_evict:
                    self._host_access_count[key] = 0

        t_pre_io = time.perf_counter()
        self.time_host_bookkeeping_s += (t_pre_io - t_start)

        if fetch_needed and self._pipeline_h2d and pipeline_h2d:
            self._pipelined_fetch(layer_id, host_slots, fetch_needed)
        elif fetch_needed:
            bytes_this_batch = 0
            t_io_0 = time.perf_counter()

            if self.uring is not None:
                uring_tasks = []
                for _, expert_id, slot in fetch_needed:
                    for name in self.bank_schema:
                        loc = self.bank_locations[(layer_id, name)]
                        if name in self.preloaded_banks:
                            dest_slice = memoryview(self.host_banks_np[name])[
                                slot * loc.row_bytes : (slot + 1) * loc.row_bytes
                            ]
                            src_off = layer_id * (self.num_experts * loc.row_bytes) + expert_id * loc.row_bytes
                            dest_slice[:] = memoryview(self.preloaded_banks[name])[src_off : src_off + loc.row_bytes]
                        else:
                            off = loc.offset_in_shard + expert_id * loc.row_bytes
                            ptr = self.host_bank_ptrs[name] + slot * loc.row_bytes
                            uring_tasks.append((loc.fd_direct, ptr, loc.row_bytes, off))
                            bytes_this_batch += loc.row_bytes

                n_t = len(uring_tasks)
                if n_t == 0:
                    self.qd_hist["0"] += 1
                elif n_t <= 4:
                    self.qd_hist["1_4"] += 1
                elif n_t <= 8:
                    self.qd_hist["5_8"] += 1
                elif n_t <= 16:
                    self.qd_hist["9_16"] += 1
                else:
                    self.qd_hist["gt16"] += 1

                self.uring.read_batch(uring_tasks)
            else:
                tasks = []
                for _, expert_id, slot in fetch_needed:
                    for name in self.bank_schema:
                        loc = self.bank_locations[(layer_id, name)]
                        dest_slice = memoryview(self.host_banks_np[name])[
                            slot * loc.row_bytes : (slot + 1) * loc.row_bytes
                        ]
                        if name in self.preloaded_banks:
                            src_off = layer_id * (self.num_experts * loc.row_bytes) + expert_id * loc.row_bytes
                            dest_slice[:] = memoryview(self.preloaded_banks[name])[src_off : src_off + loc.row_bytes]
                        else:
                            off = loc.offset_in_shard + expert_id * loc.row_bytes
                            tasks.append((loc.fd_direct, dest_slice, off, loc.row_bytes))
                            bytes_this_batch += loc.row_bytes

                n_t = len(tasks)
                if n_t == 0:
                    self.qd_hist["0"] += 1
                elif n_t <= 4:
                    self.qd_hist["1_4"] += 1
                elif n_t <= 8:
                    self.qd_hist["5_8"] += 1
                elif n_t <= 16:
                    self.qd_hist["9_16"] += 1
                else:
                    self.qd_hist["gt16"] += 1

                if len(tasks) == 1:
                    _read_job(tasks[0])
                elif len(tasks) > 1:
                    list(self.io_pool.map(_read_job, tasks))

            t_io_1 = time.perf_counter()
            self.time_nvme_read_s += (t_io_1 - t_io_0)
            self.bytes_nvme_read += bytes_this_batch
        else:
            self.qd_hist["0"] += 1

        return host_slots

    def ensure_experts(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        n = expert_ids.numel()
        self.stat_total_requests += n
        if getattr(self, "prefetcher", None) is not None and self.prefetcher.enabled and get_global_ctx().batch.is_decode:
            return self.prefetcher.ensure_layer_experts(layer_id, expert_ids)
        if self._pipeline_h2d:
            # Reset every call so a layer that takes the non-pipelined branch below
            # (fetch_needed empty, or pipelining not applicable) can't inherit a stale
            # True left over from a previous layer and make copy_missing skip its copy.
            self._pending_h2d_pipelined = False
        if self.collect_decode_freq and get_global_ctx().batch.is_decode:
            self.decode_freq[layer_id].scatter_add_(
                0, expert_ids.view(-1), torch.ones_like(expert_ids.view(-1), dtype=torch.int64)
            )
        if self._zero_sync_host_remap and get_global_ctx().batch.is_decode:
            # Phase 2: High-speed zero-sync device remapping for decode
            self._topk_pinned[:n].copy_(expert_ids.view(-1)[:n], non_blocking=True)
            self._topk_event.record()
            self._topk_event.synchronize()
            routed = self._topk_pinned[:n].tolist()

            protect_set = {(layer_id, e) for e in routed}

            # Pre-touch host LRU for already-resident routed experts so they move to MRU
            for e in routed:
                key = (layer_id, e)
                if key in self.host_slot_for_expert and key not in self.pinned_experts:
                    self.host_lru.touch(key)

            nvme_missing = [
                e for e in routed
                if (layer_id, e) not in self.host_slot_for_expert and (layer_id, e) not in self.gpu_pinned_experts
            ]

            if nvme_missing:
                t_io0 = time.perf_counter()
                self.resolve_experts_in_host_batched(
                    layer_id, nvme_missing, pipeline_h2d=False, protect_keys=protect_set
                )
                if self._trace_sync:
                    self.time_item_sync_with_miss_s += (time.perf_counter() - t_io0)
                    self.count_miss_layers += 1
            elif self._trace_sync:
                self.count_zero_miss_layers += 1

            # Ensure all non-gpu-pinned routed experts have their host slot registered in expert_to_host_slot
            for e in routed:
                key = (layer_id, e)
                if key in self.host_slot_for_expert:
                    self.expert_to_host_slot[layer_id, e] = self.host_slot_for_expert[key]

            # GPU-side LRU slot assignment
            t0 = time.perf_counter()
            if self.num_gpu_pinned > 0 and n <= self.num_dynamic:
                lru_ensure(
                    expert_ids,
                    self.slot_for_id.view(-1),
                    self.id_of_slot_dynamic,
                    self.usage_dynamic,
                    self.step,
                    expert_ids,
                    self.src_indices,
                    self.evict_slots,
                    self.num_indices,
                    stats=self.lru_stats[layer_id] if self.collect_stats else None,
                    id_base=layer_id * self.num_experts,
                )
            else:
                base_ensure_experts(self, layer_id, expert_ids)
            t1 = time.perf_counter()
            self.time_gpu_lru_s += (t1 - t0)

            # Device-side host slot remapping on GPU stream, clamped to >= 0
            remapped = self.expert_to_host_slot[
                layer_id, self.src_indices[:n].clamp(0, self.num_experts - 1).long()
            ].clamp(min=0)
            self.src_indices[:n].copy_(remapped)

            self._pending_src_layer = layer_id
            self._pending_whole_layer = False
            self._last_num_misses = len(nvme_missing)

            # Statistics update
            gpu_pinned_hits = sum(1 for e in routed if (layer_id, e) in self.gpu_pinned_experts)
            self.stat_gpu_hits += gpu_pinned_hits
            self.stat_host_hits += (n - len(nvme_missing) - gpu_pinned_hits)

            step = getattr(self, "_telemetry_step", 480)
            if self.stat_total_requests >= self._next_telemetry_threshold:
                self._report_telemetry(step)
            return

        t0 = time.perf_counter()
        if self.num_gpu_pinned > 0 and n <= self.num_dynamic:
            lru_ensure(
                expert_ids,
                self.slot_for_id.view(-1),
                self.id_of_slot_dynamic,
                self.usage_dynamic,
                self.step,
                expert_ids,
                self.src_indices,
                self.evict_slots,
                self.num_indices,
                stats=self.lru_stats[layer_id] if self.collect_stats else None,
                id_base=layer_id * self.num_experts,
            )
        else:
            base_ensure_experts(self, layer_id, expert_ids)
        t1 = time.perf_counter()
        self.time_gpu_lru_s += (t1 - t0)

        if self._num_indices_pinned is None:
            self._num_indices_pinned = torch.empty(
                1, dtype=self.num_indices.dtype, pin_memory=True
            )
            # Sized to num_experts (the largest n any caller can ever pass -- decode's
            # top_k is far smaller, prefill's routed-union call can be up to num_experts
            # wide), not to whichever call happens to run first. Sizing to the first
            # call's own n was correct only as long as every caller passed the same n
            # every time (true while decode's fixed top_k was the only caller); Step 5
            # added a second caller (prefill's union fetch) with a different, variable n,
            # and a buffer fixed at the first call's size would make `.copy_()` raise a
            # shape-mismatch error the next time n differed.
            self._src_indices_pinned = torch.empty(
                self.num_experts, dtype=self.src_indices.dtype, pin_memory=True
            )
            self._evict_slots_pinned = torch.empty(
                self.num_experts, dtype=self.evict_slots.dtype, pin_memory=True
            )

        if self._trace_sync:
            t_sync0 = time.perf_counter()
        # One combined async D2H transfer (issue both non-blocking, record ONE event,
        # wait ONCE) instead of the previous two separate blocking round-trips
        # (.item(), then .cpu().tolist() once num_misses was known). Always copies the
        # full n-wide src_indices slice -- cheap, and avoids the chicken-and-egg of
        # needing num_misses (not yet on host) to decide how much to copy.
        self._num_indices_pinned.copy_(self.num_indices, non_blocking=True)
        self._src_indices_pinned[:n].copy_(self.src_indices[:n], non_blocking=True)
        self._sync_event.record()
        self._sync_event.synchronize()
        if self._trace_sync:
            sync_dt = time.perf_counter() - t_sync0
            self.time_item_sync_s += sync_dt

        num_misses = int(self._num_indices_pinned.item())  # already host-resident, no wait
        self._last_num_misses = num_misses  # read by copy_missing for miss/no-miss H2D bucketing
        self.stat_gpu_hits += (n - num_misses)
        if num_misses > 0:
            missing_experts = self._src_indices_pinned[:num_misses].tolist()  # pure CPU
            if self._trace_sync:
                nvme_reads_before = self.stat_nvme_reads
            host_slots = self.resolve_experts_in_host_batched(layer_id, missing_experts)
            if self._trace_sync:
                # Classify AFTER the fact, by whether this call actually triggered a disk
                # read (GPU miss + host hit still counts as "no disk wait" -- matches
                # RESULTS_AND_FINDINGS.md #17.3's "0 disk I/O" framing, not "0 GPU miss",
                # which is a much rarer condition (all 10 routed experts GPU-resident).
                if self.stat_nvme_reads > nvme_reads_before:
                    self.time_item_sync_with_miss_s += sync_dt
                    self.count_miss_layers += 1
                else:
                    self.time_item_sync_zero_miss_s += sync_dt
                    self.count_zero_miss_layers += 1
            self.src_indices[:num_misses].copy_(
                torch.tensor(host_slots, dtype=torch.int32, device=self.device)
            )
        elif self._trace_sync:
            self.time_item_sync_zero_miss_s += sync_dt
            self.count_zero_miss_layers += 1

        self._pending_src_layer = layer_id
        self._pending_whole_layer = False

        step = getattr(self, "_telemetry_step", 480)
        if self.stat_total_requests >= self._next_telemetry_threshold:  # ~1 token
            self._report_telemetry(step)

    def _report_telemetry(self, step: int) -> None:
        self._next_telemetry_threshold = (
            (self.stat_total_requests // step) + 1
        ) * step
        gpu_hit_pct = (self.stat_gpu_hits / self.stat_total_requests * 100) if self.stat_total_requests > 0 else 0.0
        host_hit_pct = (self.stat_host_hits / self.stat_total_requests * 100) if self.stat_total_requests > 0 else 0.0
        total_hit_pct = ((self.stat_gpu_hits + self.stat_host_hits) / self.stat_total_requests * 100) if self.stat_total_requests > 0 else 0.0
        nvme_read_pct = (self.stat_nvme_reads / self.stat_total_requests * 100) if self.stat_total_requests > 0 else 0.0
        mb_read = self.bytes_nvme_read / (1024 * 1024)
        io_bw = (mb_read / self.time_nvme_read_s) if self.time_nvme_read_s > 0 else 0.0
        pinned_info = f", pinned: {self.stat_pinned_hits}" if self.pinned_experts else ""
        gpu_pinned_info = f", pinned: {len(self.gpu_pinned_experts)}" if self.gpu_pinned_experts else ""
        logger.info(
            f"[3-Tier Telemetry] Reqs: {self.stat_total_requests:6d} | "
            f"GPU Hits: {self.stat_gpu_hits} ({gpu_hit_pct:.1f}%{gpu_pinned_info}) | "
            f"Host Hits: {self.stat_host_hits} ({host_hit_pct:.1f}%{pinned_info}) | "
            f"In-Memory Hit: {total_hit_pct:.1f}% | "
            f"NVMe Reads: {self.stat_nvme_reads} ({nvme_read_pct:.1f}%)"
        )
        if self._trace_sync and self._h2d_pending:
            torch.cuda.synchronize()
            for s, e, had_miss in self._h2d_pending:
                dt = s.elapsed_time(e) / 1000.0
                self.time_h2d_copy_s += dt
                if had_miss:
                    self.time_h2d_copy_with_miss_s += dt
                    self.count_h2d_with_miss += 1
                else:
                    self.time_h2d_copy_zero_miss_s += dt
                    self.count_h2d_zero_miss += 1
            self._h2d_pending = []
        logger.info(
            f"[Latency Breakdown] NVMe I/O: {self.time_nvme_read_s:.3f}s ({mb_read:.1f} MB @ {io_bw:.0f} MB/s) | "
            f"Host Bookkeeping: {self.time_host_bookkeeping_s:.3f}s | "
            f"GPU LRU: {self.time_gpu_lru_s:.3f}s | "
            f"H2D Scatter-Gather: {self.time_h2d_copy_s:.3f}s"
        )
        logger.info(
            f"[Queue Depth Distribution] 0 misses: {self.qd_hist['0']} | "
            f"QD 1-4: {self.qd_hist['1_4']} | QD 5-8: {self.qd_hist['5_8']} | "
            f"QD 9-16: {self.qd_hist['9_16']} | QD >16: {self.qd_hist['gt16']}"
        )
        if self._trace_host_evict:
            ev = self._evict_hist
            total_ev = max(1, ev["0"] + ev["1"] + ev["2+"])
            logger.info(
                f"[Host Evict Trace] dynamic-tier evictions: {ev['0']} had 0 prior hits "
                f"({ev['0']/total_ev*100:.1f}%) | {ev['1']} had exactly 1 ({ev['1']/total_ev*100:.1f}%) | "
                f"{ev['2+']} had 2+ ({ev['2+']/total_ev*100:.1f}%, SLRU-protectable)"
            )
        if self._speculative_prefetch_enabled:
            pf = self.stat_speculative_prefetched
            hit_pct = (self.stat_speculative_hits / pf * 100) if pf else 0.0
            logger.info(
                f"[Speculative Prefetch] issued: {pf} | confirmed hits: "
                f"{self.stat_speculative_hits} ({hit_pct:.1f}% precision) | "
                f"in flight: {len(self._speculative_pending)}"
            )
        if self._trace_sync:
            z = self.count_zero_miss_layers
            m = self.count_miss_layers
            avg_zero = (self.time_item_sync_zero_miss_s / z * 1000) if z else 0.0
            avg_miss = (self.time_item_sync_with_miss_s / m * 1000) if m else 0.0
            logger.info(
                f"[Sync Trace] .item() total: {self.time_item_sync_s*1000:.1f}ms over "
                f"{z + m} layer calls ({z} no-disk-read avg {avg_zero:.3f}ms/call, "
                f"{m} disk-read avg {avg_miss:.3f}ms/call) | "
                f".cpu().tolist(): {self.time_cpu_tolist_s*1000:.1f}ms total"
            )
            hz = self.count_h2d_zero_miss
            hm = self.count_h2d_with_miss
            h2d_avg_zero = (self.time_h2d_copy_zero_miss_s / hz * 1000) if hz else 0.0
            h2d_avg_miss = (self.time_h2d_copy_with_miss_s / hm * 1000) if hm else 0.0
            logger.info(
                f"[H2D Trace] copy_missing total: {self.time_h2d_copy_s*1000:.1f}ms over "
                f"{hz + hm} calls ({hz} zero-miss avg {h2d_avg_zero:.4f}ms/call, "
                f"{hm} with-miss avg {h2d_avg_miss:.4f}ms/call)"
            )
        if self.collect_decode_freq and self.decode_freq.sum() > 0:
            self.save_decode_freq()

    def materialize_layer(self, layer_id: int) -> None:
        """Prefill path: read all experts of layer_id into host slots [0..num_experts-1]."""
        if self._trace_materialize:
            t0 = time.perf_counter()
            bytes_moved = 0
            any_preloaded = False
        for name in self.bank_schema:
            loc = self.bank_locations[(layer_id, name)]
            dest_slice = memoryview(self.host_banks_np[name])[: loc.total_bytes]
            if name in self.preloaded_banks:
                src_off = layer_id * loc.total_bytes
                dest_slice[:] = memoryview(self.preloaded_banks[name])[src_off : src_off + loc.total_bytes]
                if self._trace_materialize:
                    any_preloaded = True
            else:
                _pread_all(loc.fd, dest_slice, loc.offset_in_shard, loc.total_bytes)
            if self._trace_materialize:
                bytes_moved += loc.total_bytes
        if self._trace_materialize:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            new_tokens = get_global_ctx().batch.log_new_tokens
            topk_union = getattr(self, "_diag_topk_union", -1)
            logger.info(
                f"[Materialize Trace] layer={layer_id:2d} new_tokens={new_tokens:5d} "
                f"bytes={bytes_moved / 1e6:7.2f}MB elapsed={elapsed_ms:7.3f}ms "
                f"preloaded={any_preloaded} topk_union={topk_union:4d}/{self.num_experts}"
            )

        # Update host table for slots 0..num_experts-1 without wiping extended cache slots!
        for s in range(self.num_experts):
            old_key = self.expert_for_host_slot[s]
            if old_key is not None and old_key in self.host_slot_for_expert:
                del self.host_slot_for_expert[old_key]
                self.host_lru.discard(old_key)
                if self._zero_sync_host_remap:
                    self.expert_to_host_slot[old_key[0], old_key[1]] = -1

            key = (layer_id, s)
            # If this key is already in an extended slot (>= num_experts), free that slot
            if key in self.host_slot_for_expert:
                prev_slot = self.host_slot_for_expert[key]
                if prev_slot >= self.num_experts:
                    self.expert_for_host_slot[prev_slot] = None
                    self.free_host_slots.append(prev_slot)

            self.host_slot_for_expert[key] = s
            self.expert_for_host_slot[s] = key
            self.host_lru.insert_new(key)
            if self._zero_sync_host_remap:
                self.expert_to_host_slot[key[0], key[1]] = s

        # Filter out any slots < num_experts from free_host_slots so slots 0..287 are never handed out during decode
        self.free_host_slots = [s for s in self.free_host_slots if s >= self.num_experts]

        base_materialize_layer(self, layer_id)
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False  # use fast fused multi copy plan

    def copy_missing(self) -> None:
        layer_id = self._pending_src_layer
        assert layer_id is not None
        # Decode-only: prefill's materialize_layer also routes through this same fused
        # copy (comment above: "use fast fused multi copy plan"), but copies up to the
        # whole 512-expert layer rather than decode's up-to-10-expert miss set -- mixing
        # the two into one per-decode-token average would badly inflate it.
        trace_this_call = self._trace_sync and get_global_ctx().batch.is_decode
        if trace_this_call:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
        if self._pipeline_h2d and self._pending_h2d_pipelined:
            is_prefetch_active = (
                getattr(self, "prefetcher", None) is not None
                and self.prefetcher.enabled
                and get_global_ctx().batch.is_decode
            )
            if is_prefetch_active:
                w_s, w_e = self.prefetcher.get_wait_events(layer_id)
                w_s.record(torch.cuda.current_stream())
                torch.cuda.current_stream().wait_event(self._h2d_pipeline_event)
                w_e.record(torch.cuda.current_stream())
                self.prefetcher.record_wait_layer(layer_id)
            else:
                torch.cuda.current_stream().wait_event(self._h2d_pipeline_event)
            self._pending_h2d_pipelined = False
        else:
            if getattr(self, "_last_num_misses", 1) > 0:
                fast_index_copy_multi_jit(
                    self._copy_dst_ptrs,
                    self._copy_src_ptrs[layer_id],
                    self._copy_feat_bytes,
                    self.evict_slots,
                    self.src_indices,
                    self.num_indices,
                )
        if trace_this_call:
            end_ev.record()
            self._h2d_pending.append((start_ev, end_ev, self._last_num_misses > 0))

    def stats_summary(self) -> dict[str, Any]:
        total = max(1, self.stat_total_requests)
        return {
            "total_expert_evals": self.stat_total_requests,
            "host_hits": self.stat_host_hits,
            "nvme_reads": self.stat_nvme_reads,
            "host_hit_rate": f"{self.stat_host_hits / max(1, self.stat_host_hits + self.stat_nvme_reads) * 100:.1f}%",
        }

    def reset(self) -> None:
        super().reset()
        if self.gpu_pinned_per_layer > 0 and self.gpu_pinned_experts:
            for layer_id in range(self.num_layers):
                gpu_valid = self._gpu_pinned_list[layer_id]
                for i, e in enumerate(gpu_valid):
                    gpu_slot = self.num_dynamic + layer_id * self.gpu_pinned_per_layer + i
                    self.slot_for_id[layer_id, e] = gpu_slot
                    self.id_of_slot[gpu_slot] = layer_id * self.num_experts + e
                    self.usage[gpu_slot] = 0

    def prewarm_cache(self, freq_path: str | None = None) -> None:
        """Phase D (RESULTS_AND_FINDINGS.md #29): calibration-informed cache pre-warming.
        Pre-loads the top-K most frequently routed experts per layer into pinned host RAM
        so early decode tokens avoid cold-start NVMe misses.
        """
        if freq_path is None:
            freq_path = self._freq_path
        if not os.path.exists(freq_path):
            logger.info(f"[Prewarm] No routing frequency file found at {freq_path}, will collect during run.")
            return
        t0 = time.perf_counter()
        try:
            freq = torch.load(freq_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            logger.warning(f"[Prewarm] Failed to load {freq_path}: {exc}")
            return

        # Phase E: Calibration-informed cache pinning
        # We pin the top-K experts per layer permanently in host RAM (immune to LRU eviction).
        # We reserve at least 25% of host cache slots for dynamic LRU to handle transient tail experts.
        max_pin_per_layer = int(self.host_cache_size * 0.75) // self.num_layers
        default_pin = min(64, max_pin_per_layer)
        pin_k_cfg = int(os.getenv("FREETOKEN_PINNED_EXPERTS", str(default_pin)))
        pin_k = min(pin_k_cfg, max_pin_per_layer, self.num_experts)

        prewarmed_total = 0
        for layer_id in range(self.num_layers):
            layer_f = freq[layer_id]
            top_experts = torch.topk(layer_f, pin_k).indices.tolist()
            valid_experts = [e for e in top_experts if layer_f[e] > 0]
            if valid_experts:
                slots = self.resolve_experts_in_host_batched(layer_id, valid_experts)
                # Pin these experts and remove from dynamic LRU so they are never evicted
                for e in valid_experts:
                    key = (layer_id, e)
                    self.pinned_experts.add(key)
                    self.host_lru.discard(key)
                prewarmed_total += len(valid_experts)

        elapsed = time.perf_counter() - t0
        mb = prewarmed_total * (self.expert_bytes / (1024**2))
        bw = (mb / elapsed) if elapsed > 0 else 0.0
        dynamic_slots = len(self.free_host_slots)
        logger.info(
            f"[Prewarm] Completed: pre-warmed & PINNED {len(self.pinned_experts)} experts across {self.num_layers} layers "
            f"({mb:.1f} MB in {elapsed:.2f}s @ {bw:.0f} MB/s). Dynamic LRU slots remaining: {dynamic_slots}."
        )

        # GPU VRAM Frequency Pinning (Item 1 from Haberstroh analysis)
        gpu_pinned_count = 0
        if self.gpu_pinned_per_layer > 0:
            for layer_id in range(self.num_layers):
                layer_f = freq[layer_id]
                gpu_top_experts = torch.topk(layer_f, self.gpu_pinned_per_layer).indices.tolist()
                gpu_valid = [e for e in gpu_top_experts if layer_f[e] > 0]
                self._gpu_pinned_list[layer_id] = list(gpu_valid)
                unresolved = [e for e in gpu_valid if (layer_id, e) not in self.host_slot_for_expert]
                if unresolved:
                    self.resolve_experts_in_host_batched(layer_id, unresolved)
                    for e in unresolved:
                        key = (layer_id, e)
                        self.pinned_experts.add(key)
                        self.host_lru.discard(key)
                for i, e in enumerate(gpu_valid):
                    gpu_slot = self.num_dynamic + layer_id * self.gpu_pinned_per_layer + i
                    host_slot = self.host_slot_for_expert[(layer_id, e)]
                    for name in self.bank_schema:
                        self.bank_caches[name][gpu_slot].copy_(
                            self.host_banks[name][host_slot], non_blocking=True
                        )
                    self.slot_for_id[layer_id, e] = gpu_slot
                    self.id_of_slot[gpu_slot] = layer_id * self.num_experts + e
                    self.usage[gpu_slot] = 0
                    self.gpu_pinned_experts.add((layer_id, e))
                    gpu_pinned_count += 1
            torch.cuda.synchronize()
            gpu_mb = gpu_pinned_count * (self.expert_bytes / (1024**2))
            logger.info(
                f"[Prewarm] GPU VRAM: pinned {gpu_pinned_count} experts ({self.gpu_pinned_per_layer}/layer, {gpu_mb:.1f} MB) "
                f"into fixed slots [{self.num_dynamic}..{self.cache_size}). Dynamic GPU pool: {self.num_dynamic} slots."
            )
        # Reset counters so request-serving telemetry reflects inference hit rates cleanly
        self.stat_nvme_reads = 0
        self.stat_host_hits = 0
        self.stat_pinned_hits = 0
        self.stat_gpu_hits = 0
        self.stat_total_requests = 0
        self.bytes_nvme_read = 0
        self.time_nvme_read_s = 0.0
        self.time_host_bookkeeping_s = 0.0
        self.time_gpu_lru_s = 0.0
        self.time_h2d_copy_s = 0.0

    def save_decode_freq(self, freq_path: str | None = None) -> None:
        if freq_path is None:
            freq_path = self._freq_path
        try:
            cur = self.decode_freq.cpu()
            if self._initial_freq is not None:
                cur = cur + self._initial_freq
            torch.save(cur, freq_path)
            logger.info(f"[Routing Stats] Saved decode routing frequencies ({int(cur.sum())} total hits) to {freq_path}")
        except Exception as exc:
            logger.warning(f"[Routing Stats] Failed to save {freq_path}: {exc}")

    def close(self) -> None:
        if self.collect_decode_freq and self.decode_freq.sum() > 0:
            self.save_decode_freq()
        if hasattr(self, "uring") and self.uring is not None:
            try:
                self.uring.close()
            except Exception:
                pass
        if hasattr(self, "io_pool") and self.io_pool is not None:
            try:
                self.io_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        if hasattr(self, "_speculative_pool") and self._speculative_pool is not None:
            try:
                self._speculative_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        for fd in list(self.shard_fds.values()) + list(self.shard_fds_direct.values()):
            try:
                os.close(fd)
            except OSError:
                pass
        self.shard_fds.clear()
        self.shard_fds_direct.clear()


def make_glm5_offload_cache(*args: Any, **kwargs: Any) -> NvmeOffloadMoeCache:
    """Factory called by FreeToken engine when getattr(model, 'make_offload_moe_cache') is engaged."""
    if len(args) == 3:
        # Bound method: (self, config, device)
        _, config, device = args
    elif len(args) == 2:
        # Function: (config, device)
        config, device = args
    else:
        config = kwargs.get("config", args[0] if args else None)
        device = kwargs.get("device", args[1] if len(args) > 1 else torch.device("cuda:0"))

    mc = config.model_config
    default_gpu = max(512, mc.num_experts) if mc.num_experts > 128 else 288
    gpu_cache_size = config.moe_cache_size if config.moe_cache_size > 0 else default_gpu

    default_host = max(2048, mc.num_experts * 4) if mc.num_experts > 128 else 288
    host_cache_size = int(os.getenv("FREETOKEN_HOST_CACHE_SIZE", str(default_host)))

    _install_prefill_narrow_patch()
    _install_unified_trace_patch()
    _install_early_routing_probe()
    _install_rolling_lookahead_prefetcher()
    install_glm5_pruning_patch()

    return NvmeOffloadMoeCache(
        model_path=config.model_path,
        num_layers=mc.num_moe_layers,
        num_experts=mc.num_experts,
        cache_size=gpu_cache_size,
        device=device,
        host_cache_size=host_cache_size,
        quant_format="nvfp4",
        cache_policy=config.moe_cache_policy,
    )


_prefill_narrow_patch_installed = False


def _install_prefill_narrow_patch() -> None:
    """Monkeypatches OffloadMoELayer._prefill_routed to implement Step 5
    (RESULTS_AND_FINDINGS.md #25): fetch only this layer's routed-expert union from NVMe
    during prefill, instead of materialize_layer's unconditional full 512-expert sweep.

    Gated by FREETOKEN_PREFILL_NARROW (default off, matching this project's rollout
    convention for behavior-changing flags -- see improvements.md's FREETOKEN_LAYER_CUDA_GRAPH
    precedent). When off, this only stashes the routed-union size for materialize_layer's
    FREETOKEN_TRACE_MATERIALIZE trace line (the original diagnostic this patch grew from,
    see #24-25) and otherwise runs the unmodified vendor path.

    Mechanism, per #25.2: topk_ids (the routed expert ids for every token in this prefill
    forward, shape [num_tokens, top_k]) is already computed here, before the cache is
    touched at all. `cache.ensure_experts(layer_id, topk_ids.unique())` is the exact
    LRU/NVMe-fetch call decode already makes every step, just fed this layer's whole routed
    set instead of one token's top-k -- no new kernel, no new host bookkeeping. It rewrites
    its input in place to slot ids, but only for the deduped union; the full topk_ids tensor
    the GEMM actually needs is remapped separately via one `slot_for_id` gather (safe to do
    any time after `ensure_experts` returns, before or after `copy_missing` -- the slot
    assignment kernel writes `slot_for_id` synchronously at victim-selection time, not after
    the asynchronously-scheduled H2D bytes for that slot actually land; correctness depends
    only on `copy_missing`'s kernel launch preceding the GEMM's on the same CUDA stream,
    which the call order below already guarantees). `views=cache.bank_views()`/`n=cache.
    cache_size` (not the old `bank_views(num_experts)`/`n=num_experts`) because routed
    experts no longer sit at identity positions once fetched by slot, not by expert id.
    """
    global _prefill_narrow_patch_installed
    if _prefill_narrow_patch_installed:
        return
    if os.getenv("FREETOKEN_TRACE_MATERIALIZE", "0") != "1" and os.getenv(
        "FREETOKEN_PREFILL_NARROW", "1"
    ) != "1":
        return
    _prefill_narrow_patch_installed = True
    from freetoken.layers.moe import OffloadMoELayer

    orig = OffloadMoELayer._prefill_routed
    narrow_enabled = os.getenv("FREETOKEN_PREFILL_NARROW", "1") == "1"
    trace_enabled = os.getenv("FREETOKEN_TRACE_MATERIALIZE", "0") == "1"

    def patched(self, hidden_states, topk_weights, topk_ids):
        cache = self.offload_cache
        if cache is None or cache.prefill_overlap or not narrow_enabled:
            if cache is not None and trace_enabled:
                cache._diag_topk_union = int(topk_ids.unique().numel())
            return orig(self, hidden_states, topk_weights, topk_ids)

        layer_id = self.layer_id
        if trace_enabled:
            t_start = time.perf_counter()
            misses_before = cache.stat_nvme_reads
        union_ids = topk_ids.unique()
        union_n = int(union_ids.numel())
        if trace_enabled:
            t_union = time.perf_counter()
        cache.ensure_experts(layer_id, union_ids)  # mutates union_ids in place; unused after
        if trace_enabled:
            t_ensure = time.perf_counter()
        # Item 3 (RESULTS_AND_FINDINGS.md #41): this layer's own NVMe reads are done (the
        # ensure_experts call above blocks until they land), so the NVMe device and CPU
        # are otherwise idle for however long this layer's H2D copy + GEMM take on the
        # GPU below. Use that window to speculatively prefetch layer_id+1's calibrated
        # candidate experts in the background -- a no-op call when disabled.
        cache.maybe_speculative_prefetch(layer_id + 1)
        cache.copy_missing()
        if trace_enabled:
            t_copy = time.perf_counter()
        slot_ids = cache.slot_for_id.view(-1)[
            layer_id * cache.num_experts + topk_ids
        ].to(topk_ids.dtype)
        if trace_enabled:
            t_gather = time.perf_counter()
        out = self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            slot_ids,
            views=cache.bank_views(),
            n=cache.cache_size,
            alphas=cache.alphas_for_slots(layer_id),
            is_prefill=True,
        )
        if trace_enabled:
            t_end = time.perf_counter()
            total_ms = (t_end - t_start) * 1000
            ensure_ms = (t_ensure - t_union) * 1000
            copy_ms = (t_copy - t_ensure) * 1000
            gemm_ms = (t_end - t_gather) * 1000
            new_tokens = get_global_ctx().batch.log_new_tokens
            misses = cache.stat_nvme_reads - misses_before
            logger.info(
                f"[Prefill Narrow] layer={layer_id:2d} new_tokens={new_tokens:5d} "
                f"union={union_n:4d}/{cache.num_experts} nvme_misses={misses:4d} "
                f"ensure={ensure_ms:7.2f}ms copy={copy_ms:6.2f}ms gemm={gemm_ms:6.2f}ms "
                f"total={total_ms:7.2f}ms"
            )
        return out

    OffloadMoELayer._prefill_routed = patched


_unified_trace_patch_installed = False


def _install_unified_trace_patch() -> None:
    """Monkeypatches Qwen4ExpMoE.forward and OffloadMoELayer._decode_routed to provide
    Phase A Part 2 Redo (RESULTS_AND_FINDINGS.md #27): a single unified, single-flush-cadence
    CUDA event decomposition of decode MoE execution, avoiding the cross-tracer synchronization
    and double-counting anomalies of #22.2.
    """
    global _unified_trace_patch_installed
    if _unified_trace_patch_installed:
        return
    if os.getenv("FREETOKEN_TRACE_MOE_UNIFIED", "0") != "1":
        return
    _unified_trace_patch_installed = True

    from freetoken.core import get_global_ctx
    from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.models.qwen4_exp.moe import Qwen4ExpMoE
    from freetoken.utils import init_logger

    trace_logger = init_logger("nvme_moe_unified_trace")

    orig_moe_forward = Qwen4ExpMoE.forward
    orig_decode_routed = OffloadMoELayer._decode_routed

    _current_layer_events: dict[int, dict[str, Any]] = {}
    _pending_traces: list[dict[str, Any]] = []
    _trace_call_count = 0
    _FLUSH_CADENCE = 48  # flush every token (48 layers/token)

    def _flush_unified_trace() -> None:
        nonlocal _pending_traces, _trace_call_count
        if not _pending_traces:
            return
        torch.cuda.synchronize()
        n = len(_pending_traces)
        total_whole = 0.0
        total_preroute = 0.0
        total_topk = 0.0
        total_idle = 0.0
        total_h2d = 0.0
        total_gemm = 0.0
        total_combine = 0.0

        for tr in _pending_traces:
            preroute = tr["ev_moe_start"].elapsed_time(tr["ev_preroute_end"])
            topk = tr["ev_preroute_end"].elapsed_time(tr["ev_topk_end"])
            idle = tr["ev_topk_end"].elapsed_time(tr["ev_h2d_start"])
            h2d = tr["ev_h2d_start"].elapsed_time(tr["ev_h2d_end"])
            gemm = tr["ev_gemm_start"].elapsed_time(tr["ev_gemm_end"])
            combine = tr["ev_combine_start"].elapsed_time(tr["ev_moe_end"])
            whole = tr["ev_moe_start"].elapsed_time(tr["ev_moe_end"])

            total_preroute += preroute
            total_topk += topk
            total_idle += idle
            total_h2d += h2d
            total_gemm += gemm
            total_combine += combine
            total_whole += whole

        sum_parts = total_preroute + total_topk + total_idle + total_h2d + total_gemm + total_combine
        diff = total_whole - sum_parts
        per_tok = 48.0

        trace_logger.info(
            f"[Unified MoE Trace] {n} layer spans ({n / per_tok:.1f} tokens):\n"
            f"  Whole MoE GPU Span: {total_whole:.2f}ms ({total_whole / n:.4f}ms/layer, {total_whole / n * per_tok:.2f}ms/tok)\n"
            f"  ├── Pre-Routing (Gate+Shared): {total_preroute:.2f}ms ({total_preroute / n:.4f}ms/layer, {total_preroute / n * per_tok:.2f}ms/tok) [{total_preroute / max(1e-6, total_whole) * 100:.1f}%]\n"
            f"  ├── Router topk:               {total_topk:.2f}ms ({total_topk / n:.4f}ms/layer, {total_topk / n * per_tok:.2f}ms/tok) [{total_topk / max(1e-6, total_whole) * 100:.1f}%]\n"
            f"  ├── NVMe/Host Idle Gap:        {total_idle:.2f}ms ({total_idle / n:.4f}ms/layer, {total_idle / n * per_tok:.2f}ms/tok) [{total_idle / max(1e-6, total_whole) * 100:.1f}%]\n"
            f"  ├── H2D Scatter-Gather:        {total_h2d:.2f}ms ({total_h2d / n:.4f}ms/layer, {total_h2d / n * per_tok:.2f}ms/tok) [{total_h2d / max(1e-6, total_whole) * 100:.1f}%]\n"
            f"  ├── Routed Expert GEMM:        {total_gemm:.2f}ms ({total_gemm / n:.4f}ms/layer, {total_gemm / n * per_tok:.2f}ms/tok) [{total_gemm / max(1e-6, total_whole) * 100:.1f}%]\n"
            f"  └── Combine (Gate Mul-Add):    {total_combine:.2f}ms ({total_combine / n:.4f}ms/layer, {total_combine / n * per_tok:.2f}ms/tok) [{total_combine / max(1e-6, total_whole) * 100:.1f}%]\n"
            f"  Reconciliation: Sum of parts = {sum_parts:.2f}ms vs Whole = {total_whole:.2f}ms (Diff = {diff:+.4f}ms, {diff / max(1e-6, total_whole) * 100:+.3f}%)"
        )
        _pending_traces = []
        _trace_call_count = 0

    def patched_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        if not ctx.batch.is_decode:
            return orig_moe_forward(self, hidden_states)

        ev_moe_start = torch.cuda.Event(enable_timing=True)
        ev_preroute_end = torch.cuda.Event(enable_timing=True)
        ev_combine_start = torch.cuda.Event(enable_timing=True)
        ev_moe_end = torch.cuda.Event(enable_timing=True)

        ev_moe_start.record()

        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))

        ev_preroute_end.record()

        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)

        ev_combine_start.record()
        out = shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)
        ev_moe_end.record()

        lid = getattr(self.experts, "layer_id", None)
        events = _current_layer_events.pop(lid, None) if lid is not None else None
        if events is not None:
            events["ev_moe_start"] = ev_moe_start
            events["ev_preroute_end"] = ev_preroute_end
            events["ev_combine_start"] = ev_combine_start
            events["ev_moe_end"] = ev_moe_end
            _pending_traces.append(events)

        nonlocal _trace_call_count
        _trace_call_count += 1
        if _trace_call_count >= _FLUSH_CADENCE:
            _flush_unified_trace()

        return out

    def patched_decode_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        cache = self.offload_cache
        assert cache is not None

        ev_topk_end = torch.cuda.Event(enable_timing=True)
        ev_topk_end.record()

        cache.ensure_experts(self.layer_id, topk_ids)

        ev_h2d_start = torch.cuda.Event(enable_timing=True)
        ev_h2d_end = torch.cuda.Event(enable_timing=True)
        ev_h2d_start.record()
        cache.copy_missing()
        ev_h2d_end.record()

        ev_gemm_start = torch.cuda.Event(enable_timing=True)
        ev_gemm_end = torch.cuda.Event(enable_timing=True)
        ev_gemm_start.record()
        out = self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=False,
        )
        ev_gemm_end.record()

        _current_layer_events[self.layer_id] = {
            "ev_topk_end": ev_topk_end,
            "ev_h2d_start": ev_h2d_start,
            "ev_h2d_end": ev_h2d_end,
            "ev_gemm_start": ev_gemm_start,
            "ev_gemm_end": ev_gemm_end,
        }
        return out

    Qwen4ExpMoE.forward = patched_moe_forward
    OffloadMoELayer._decode_routed = patched_decode_routed


_early_routing_probe_installed = False


def _install_early_routing_probe() -> None:
    """Diagnostic probe for testing cross-layer predictive gating fidelity.
    Gated behind FREETOKEN_PROBE_EARLY_ROUTING=1.
    Compares early-predicted top-k for layer L+1 (computed during layer L)
    against the ground-truth top-k computed when layer L+1 actually executes.
    """
    global _early_routing_probe_installed
    if _early_routing_probe_installed:
        return
    if os.getenv("FREETOKEN_PROBE_EARLY_ROUTING", "0") != "1":
        return
    _early_routing_probe_installed = True

    from freetoken.core import Batch
    from freetoken.models.qwen4_exp.model import Qwen4ExpModel
    from freetoken.utils import init_logger

    probe_logger = init_logger("early_routing_probe")
    probe_logger.info("[Early Gating Probe] Initialized diagnostic probe on Qwen4ExpModel")

    orig_model_forward = Qwen4ExpModel.forward

    # Cumulative stats
    stats = {
        "tokens": 0,
        "cand_A_overlaps": [],  # Direct block_input: next_gate(block_input_L)
        "cand_B_overlaps": [],  # Pre-MoE hidden mixed: next_gate(next_mlp_mix(hidden))
        "cand_C_overlaps": [],  # Next Attn mixed: next_gate(next_attn_mix(hidden))
        "cand_D_overlaps": [],  # Previous token same-layer persistence
        "per_layer_A": [[] for _ in range(48)],
    }

    # Tracking previous token true top-k per layer [48]
    prev_token_topk: list[set[int] | None] = [None] * 48

    def patched_model_forward(self, input_ids: torch.Tensor, batch: Batch) -> torch.Tensor:
        if not batch.is_decode:
            return orig_model_forward(self, input_ids, batch)

        layers = self.layers.op_list
        num_layers = len(layers)

        hidden = self.embed_tokens.forward(input_ids).repeat(1, self.hc_count)
        meta = None
        if self._ple:
            from freetoken.models.qwen4_exp.ple import build_ple_metadata, commit_ngram_context
            meta = build_ple_metadata(batch, self._ple[0].args, input_ids.device)
            for ple in self._ple:
                ple.start_prefetch(batch, meta)

        pred_A = {}
        pred_B = {}
        pred_C = {}

        for l, layer in enumerate(layers):
            if layer.ple is not None:
                hidden = hidden + layer.ple.forward(hidden, batch)

            block_input, inject = layer.attn_hyper_connection.mix(hidden)
            if layer._is_linear:
                block_output = layer.linear_attn.forward(block_input)
            else:
                block_output = layer.self_attn.forward(block_input, batch)
            hidden = layer.attn_hyper_connection.combine(hidden, block_output, inject)
            block_input, inject = layer.mlp_hyper_connection.mix(hidden)

            # Compute true routing for layer l
            with torch.no_grad():
                true_logits = layer.mlp.gate.forward(block_input)
                true_topk = set(torch.topk(true_logits.view(-1), k=10).indices.tolist())

                # If layer l had predictions from layer l-1, evaluate!
                if l in pred_A:
                    ov_A = len(pred_A[l] & true_topk)
                    ov_B = len(pred_B[l] & true_topk)
                    ov_C = len(pred_C[l] & true_topk)
                    stats["cand_A_overlaps"].append(ov_A)
                    stats["cand_B_overlaps"].append(ov_B)
                    stats["cand_C_overlaps"].append(ov_C)
                    stats["per_layer_A"][l].append(ov_A)

                    if prev_token_topk[l] is not None:
                        ov_D = len(prev_token_topk[l] & true_topk)
                        stats["cand_D_overlaps"].append(ov_D)

                prev_token_topk[l] = true_topk

                # Compute predictions for layer l + 1
                if l + 1 < num_layers:
                    next_layer = layers[l + 1]
                    # Candidate A: next_gate(block_input)
                    p_A = next_layer.mlp.gate.forward(block_input)
                    pred_A[l + 1] = set(torch.topk(p_A.view(-1), k=10).indices.tolist())

                    # Candidate B: next_gate(next_mlp_mix(hidden))
                    b_B, _ = next_layer.mlp_hyper_connection.mix(hidden)
                    p_B = next_layer.mlp.gate.forward(b_B)
                    pred_B[l + 1] = set(torch.topk(p_B.view(-1), k=10).indices.tolist())

                    # Candidate C: next_gate(next_attn_mix(hidden))
                    b_C, _ = next_layer.attn_hyper_connection.mix(hidden)
                    p_C = next_layer.mlp.gate.forward(b_C)
                    pred_C[l + 1] = set(torch.topk(p_C.view(-1), k=10).indices.tolist())

            moe_out = layer.mlp.forward(block_input)
            hidden = layer.mlp_hyper_connection.combine(hidden, moe_out, inject)

        if meta is not None:
            commit_ngram_context(meta, getattr(batch, "fla_metadata", None))

        stats["tokens"] += 1
        if stats["tokens"] % 10 == 0:
            nA = len(stats["cand_A_overlaps"])
            if nA > 0:
                avg_A = sum(stats["cand_A_overlaps"]) / nA
                avg_B = sum(stats["cand_B_overlaps"]) / nA
                avg_C = sum(stats["cand_C_overlaps"]) / nA
                avg_D = sum(stats["cand_D_overlaps"]) / len(stats["cand_D_overlaps"]) if stats["cand_D_overlaps"] else 0.0
                jacc_A = avg_A / (20.0 - avg_A) * 100
                jacc_B = avg_B / (20.0 - avg_B) * 100
                jacc_C = avg_C / (20.0 - avg_C) * 100
                ge_8_pct = sum(1 for x in stats["cand_A_overlaps"] if x >= 8) / nA * 100
                ge_5_pct = sum(1 for x in stats["cand_A_overlaps"] if x >= 5) / nA * 100
                probe_logger.info(
                    f"[Early Gating Probe | {stats['tokens']} tokens ({nA} transitions)]\n"
                    f"  Cand A (Direct block_input):  Overlap = {avg_A:.2f}/10 ({avg_A*10:.1f}%) | Jaccard = {jacc_A:.1f}% | >=8/10: {ge_8_pct:.1f}% | >=5/10: {ge_5_pct:.1f}%\n"
                    f"  Cand B (Next MLP mix):        Overlap = {avg_B:.2f}/10 ({avg_B*10:.1f}%) | Jaccard = {jacc_B:.1f}%\n"
                    f"  Cand C (Next Attn mix):       Overlap = {avg_C:.2f}/10 ({avg_C*10:.1f}%) | Jaccard = {jacc_C:.1f}%\n"
                    f"  Cand D (Prev token persist):  Overlap = {avg_D:.2f}/10 ({avg_D*10:.1f}%)"
                )
            if stats["tokens"] % 50 == 0:
                early = [x for l in range(1, 16) for x in stats["per_layer_A"][l]]
                mid = [x for l in range(16, 32) for x in stats["per_layer_A"][l]]
                late = [x for l in range(32, 48) for x in stats["per_layer_A"][l]]
                e_avg = sum(early) / len(early) if early else 0
                m_avg = sum(mid) / len(mid) if mid else 0
                l_avg = sum(late) / len(late) if late else 0
                probe_logger.info(
                    f"[Early Gating Stage Breakdown]\n"
                    f"  Early Layers (L0 -> L15):  Overlap = {e_avg:.2f}/10 ({e_avg*10:.1f}%)\n"
                    f"  Mid Layers   (L16 -> L31): Overlap = {m_avg:.2f}/10 ({m_avg*10:.1f}%)\n"
                    f"  Late Layers  (L32 -> L47): Overlap = {l_avg:.2f}/10 ({l_avg*10:.1f}%)"
                )

        return self.hyper_connection_mixer.mix(hidden)[0]

    Qwen4ExpModel.forward = patched_model_forward


# --------------------------------------------------------------------------
# Rolling-Window Speculative Lookahead Prefetcher (Candidate B Architecture)
# --------------------------------------------------------------------------


class InFlightStatus:
    SPECULATIVE = 1
    DEMAND_WAITING = 2
    READY = 3


class InFlightEntry:
    def __init__(self, layer_id: int, expert_id: int, slot: int, is_speculative: bool):
        self.layer_id = layer_id
        self.expert_id = expert_id
        self.slot = slot
        self.is_speculative = is_speculative
        self.status = InFlightStatus.SPECULATIVE if is_speculative else InFlightStatus.DEMAND_WAITING
        self.created_at = time.perf_counter()
        self.completed_at: float | None = None
        self.was_promoted = False
        self.was_cqe_ready_before_demand = False
        self.was_counted_fp = False


class RollingLookaheadPrefetcher:
    """Rolling-window speculative lookahead prefetcher for 3-tier offloaded MoE.
    Speculates ONLY storage movement (NVMe -> Host RAM DMA) using Candidate B
    (intermediate post-attention residual passed to gate_{L+1}).

    Zero computation speculation: model outputs, tokens, routing decisions,
    and GEMMs remain 100% bit-exact.
    """

    def __init__(self, cache: "NvmeOffloadMoeCache") -> None:
        self.cache = cache
        self.enabled = os.getenv("FREETOKEN_SPECULATIVE_LOOKAHEAD", "0") == "1"
        self.target_qd = int(os.getenv("FREETOKEN_SPECULATIVE_TARGET_QD", "8"))
        self.trace_lookahead = os.getenv("FREETOKEN_TRACE_LOOKAHEAD", "1") == "1"
        self.in_flight_table: dict[tuple[int, int], InFlightEntry] = {}
        self.spec_k = 16
        self._pinned_pred_buf = torch.empty(
            (self.cache.num_layers, self.spec_k),
            dtype=torch.int32,
            pin_memory=True,
        )
        self._pinned_pred_np = self._pinned_pred_buf.numpy()
        self.has_staged_prediction: list[bool] = [False] * self.cache.num_layers
        self._host_slots_buf = torch.empty(self.cache.num_experts, dtype=torch.int32, pin_memory=True)
        self._host_slots_np = self._host_slots_buf.numpy()
        self._ordered_evict_buf = torch.empty(self.cache.num_experts, dtype=torch.int32, pin_memory=True)
        self._ordered_evict_np = self._ordered_evict_buf.numpy()
        self._ordered_host_buf = torch.empty(self.cache.num_experts, dtype=torch.int32, pin_memory=True)
        self._ordered_host_np = self._ordered_host_buf.numpy()

        self._h2d_events_b1 = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(self.cache.num_layers)
        ]
        self._h2d_events_stream = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(self.cache.num_layers)
        ]
        self._h2d_events_wait = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(self.cache.num_layers)
        ]
        self._b1_active_layers: list[int] = []
        self._stream_active_layers: list[int] = []
        self._stream_active_set: set[int] = set()
        self._wait_active_layers: list[int] = []
        if not hasattr(self.cache, "_h2d_stream") or self.cache._h2d_stream is None:
            self.cache._h2d_stream = torch.cuda.Stream()
        if not hasattr(self.cache, "_h2d_pipeline_event") or self.cache._h2d_pipeline_event is None:
            self.cache._h2d_pipeline_event = torch.cuda.Event()
        self.cache._pipeline_h2d = True
        self._init_metrics()
        logger.info(
            f"RollingLookaheadPrefetcher initialized: enabled={self.enabled}, "
            f"target_qd={self.target_qd}, trace={self.trace_lookahead}"
        )

    def get_b1_events(self, layer_id: int) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        return self._h2d_events_b1[layer_id]

    def record_b1_batch(self, layer_id: int, num_experts: int) -> None:
        self._b1_active_layers.append(layer_id)
        self.token_metrics["b1_submits"] += 1
        self.token_metrics["b1_experts"] += num_experts
        self.token_metrics["b1_bytes_mb"] += num_experts * 2.64

    def record_streaming_arrival(self, layer_id: int, num_experts: int = 1) -> None:
        if layer_id not in self._stream_active_set:
            self._stream_active_set.add(layer_id)
            self._stream_active_layers.append(layer_id)
            s_ev, _ = self._h2d_events_stream[layer_id]
            s_ev.record(self.cache._h2d_stream)
        _, e_ev = self._h2d_events_stream[layer_id]
        e_ev.record(self.cache._h2d_stream)
        self.token_metrics["stream_submits"] += 1
        self.token_metrics["stream_experts"] += num_experts
        self.token_metrics["stream_bytes_mb"] += num_experts * 2.64

    def get_wait_events(self, layer_id: int) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        return self._h2d_events_wait[layer_id]

    def record_wait_layer(self, layer_id: int) -> None:
        self._wait_active_layers.append(layer_id)

    def _init_metrics(self) -> None:
        self.metrics = {
            "tokens": 0,
            "demanded_misses": 0,
            "predicted_reads": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "tp_completed_before_demand": 0,
            "tp_already_completed_not_reaped": 0,
            "tp_in_flight_at_demand": 0,
            "fallback_reads": 0,
            "peak_outstanding_qd": 0,
            "sum_outstanding_qd": 0,
            "count_outstanding_qd": 0,
            "nvme_bytes_actual": 0.0,
            "nvme_bytes_speculative": 0.0,
            "exposed_nvme_ms": 0.0,
            "b1_submits": 0,
            "b1_experts": 0,
            "b1_bytes_mb": 0.0,
            "b1_h2d_ms": 0.0,
            "stream_submits": 0,
            "stream_experts": 0,
            "stream_bytes_mb": 0.0,
            "stream_h2d_ms": 0.0,
            "h2d_copy_ms": 0.0,
            "h2d_wait_ms": 0.0,
        }
        self.token_metrics = self._new_token_metrics()

    def _new_token_metrics(self) -> dict[str, Any]:
        return {
            "demanded_misses": 0,
            "predicted_reads": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "tp_completed_before_demand": 0,
            "tp_already_completed_not_reaped": 0,
            "tp_in_flight_at_demand": 0,
            "fallback_reads": 0,
            "peak_outstanding_qd": 0,
            "sum_outstanding_qd": 0,
            "count_outstanding_qd": 0,
            "nvme_bytes_actual": 0.0,
            "nvme_bytes_speculative": 0.0,
            "exposed_nvme_ms": 0.0,
            "b1_submits": 0,
            "b1_experts": 0,
            "b1_bytes_mb": 0.0,
            "b1_h2d_ms": 0.0,
            "stream_submits": 0,
            "stream_experts": 0,
            "stream_bytes_mb": 0.0,
            "stream_h2d_ms": 0.0,
            "h2d_copy_ms": 0.0,
            "h2d_wait_ms": 0.0,
        }

    def stage_prediction(self, layer_id: int, logits: torch.Tensor) -> None:
        cand_indices = torch.topk(logits, k=self.spec_k).indices.to(torch.int32)
        self._pinned_pred_buf[layer_id].copy_(cand_indices, non_blocking=True)
        self.has_staged_prediction[layer_id] = True

    def ensure_layer_experts(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        n = expert_ids.numel()
        self.cache.stat_total_requests += n
        if self.cache._pipeline_h2d:
            self.cache._pending_h2d_pipelined = False

        # GPU LRU slot assignment
        t0 = time.perf_counter()
        if self.cache.num_gpu_pinned > 0 and n <= self.cache.num_dynamic:
            lru_ensure(
                expert_ids,
                self.cache.slot_for_id.view(-1),
                self.cache.id_of_slot_dynamic,
                self.cache.usage_dynamic,
                self.cache.step,
                expert_ids,
                self.cache.src_indices,
                self.cache.evict_slots,
                self.cache.num_indices,
                stats=self.cache.lru_stats[layer_id] if self.cache.collect_stats else None,
                id_base=layer_id * self.cache.num_experts,
            )
        else:
            base_ensure_experts(self.cache, layer_id, expert_ids)
        self.cache.time_gpu_lru_s += (time.perf_counter() - t0)

        if self.cache._num_indices_pinned is None:
            self.cache._num_indices_pinned = torch.empty(1, dtype=self.cache.num_indices.dtype, pin_memory=True)
            self.cache._src_indices_pinned = torch.empty(self.cache.num_experts, dtype=self.cache.src_indices.dtype, pin_memory=True)
        if self.cache._evict_slots_pinned is None:
            self.cache._evict_slots_pinned = torch.empty(self.cache.num_experts, dtype=self.cache.evict_slots.dtype, pin_memory=True)

        self.cache._num_indices_pinned.copy_(self.cache.num_indices, non_blocking=True)
        self.cache._src_indices_pinned[:n].copy_(self.cache.src_indices[:n], non_blocking=True)
        self.cache._evict_slots_pinned[:n].copy_(self.cache.evict_slots[:n], non_blocking=True)
        self.cache._sync_event.record()
        self.cache._sync_event.synchronize()

        num_misses = int(self.cache._num_indices_pinned.item())
        self.cache._last_num_misses = num_misses
        self.cache.stat_gpu_hits += (n - num_misses)

        # 1. Non-blocking drain of CQEs in ring to identify prefetches that completed before demand
        reaped_tags = self.cache.uring.peek_completed()
        for tag in reaped_tags:
            _, t_layer, t_expert, _ = tag
            t_key = (t_layer, t_expert)
            if t_key in self.in_flight_table:
                entry = self.in_flight_table[t_key]
                entry.status = InFlightStatus.READY
                entry.completed_at = time.perf_counter()
                entry.was_cqe_ready_before_demand = True

        # Working set eviction protection
        protect_keys = {(layer_id, e) for e in expert_ids.view(-1).tolist()}
        protect_keys.update(self.in_flight_table.keys())

        resident_pairs: list[tuple[int, int]] = []
        demand_entries: list[tuple[int, int]] = []
        demand_tags: list[Any] = []
        demand_jobs = []
        required_demand_tags = set()
        tag_sizes = {}
        tag_to_expert_info = {}

        # 2. Audit Layer L demands
        if num_misses > 0:
            missing_experts = self.cache._src_indices_pinned[:num_misses].tolist()
            evict_slots_list = self.cache._evict_slots_pinned[:num_misses].tolist()
            for pos, expert_id in enumerate(missing_experts):
                evict_slot = evict_slots_list[pos]
                key = (layer_id, expert_id)
                if key in self.cache.host_slot_for_expert:
                    # In Host RAM hit
                    slot = self.cache.host_slot_for_expert[key]
                    if key in self.cache.pinned_experts:
                        self.cache.stat_host_hits += 1
                        self.cache.stat_pinned_hits += 1
                    else:
                        self.cache.host_lru.touch(key)
                        self.cache.stat_host_hits += 1
                    resident_pairs.append((evict_slot, slot))
                elif key in self.in_flight_table:
                    # TRUE POSITIVE PREFETCH MATCH
                    entry = self.in_flight_table[key]
                    slot = entry.slot
                    self.token_metrics["demanded_misses"] += 1
                    self.token_metrics["true_positives"] += 1
                    self.token_metrics["nvme_bytes_actual"] += 2.64

                    if entry.status == InFlightStatus.READY:
                        if entry.was_cqe_ready_before_demand:
                            self.token_metrics["tp_already_completed_not_reaped"] += 1
                        else:
                            self.token_metrics["tp_completed_before_demand"] += 1
                        self.cache.host_slot_for_expert[key] = slot
                        self.cache.expert_for_host_slot[slot] = key
                        self.cache.host_lru.insert_new(key)
                        del self.in_flight_table[key]
                        resident_pairs.append((evict_slot, slot))
                    else:
                        # In flight at demand -> promote in-place to demand waiting
                        self.token_metrics["tp_in_flight_at_demand"] += 1
                        entry.status = InFlightStatus.DEMAND_WAITING
                        entry.was_promoted = True
                        tag = ("spec", layer_id, expert_id, slot)
                        required_demand_tags.add(tag)
                        tag_to_expert_info[tag] = (pos, key, slot)
                        demand_entries.append((evict_slot, slot))
                        demand_tags.append(tag)
                else:
                    # FALSE NEGATIVE (fallback demand read)
                    self.token_metrics["demanded_misses"] += 1
                    self.token_metrics["false_negatives"] += 1
                    self.token_metrics["fallback_reads"] += 1
                    self.token_metrics["nvme_bytes_actual"] += 2.64
                    self.cache.stat_nvme_reads += 1

                    if self.cache.free_host_slots:
                        slot = self.cache.free_host_slots.pop()
                    else:
                        evict_key, _ = self.cache.host_lru.pop_lru(protect=protect_keys)
                        slot = self.cache.host_slot_for_expert.pop(evict_key)
                        self.cache.expert_for_host_slot[slot] = None

                    protect_keys.add(key)
                    tag = ("demand", layer_id, expert_id, slot)
                    required_demand_tags.add(tag)
                    tag_to_expert_info[tag] = (pos, key, slot)
                    expert_jobs = self.cache._build_expert_jobs(layer_id, expert_id, slot, tag)
                    demand_jobs.extend(expert_jobs)
                    tag_sizes[tag] = len(expert_jobs)
                    demand_entries.append((evict_slot, slot))
                    demand_tags.append(tag)

            # Order: resident_pairs (0..k1-1), then demand_entries (k1..k1+k2-1)
            k1 = len(resident_pairs)
            k2 = len(demand_entries)
            assert k1 + k2 == num_misses

            tag_to_entry_idx = {}
            for i, d_tag in enumerate(demand_tags):
                tag_to_entry_idx[d_tag] = k1 + i

            for i, (ev_s, h_s) in enumerate(resident_pairs):
                self._ordered_evict_np[i] = ev_s
                self._ordered_host_np[i] = h_s
            for i, (ev_s, h_s) in enumerate(demand_entries):
                self._ordered_evict_np[k1 + i] = ev_s
                self._ordered_host_np[k1 + i] = h_s

            with torch.cuda.stream(self.cache._h2d_stream):
                self.cache.evict_slots[:num_misses].copy_(
                    self._ordered_evict_buf[:num_misses], non_blocking=True
                )
                self.cache.src_indices[:num_misses].copy_(
                    self._ordered_host_buf[:num_misses], non_blocking=True
                )

                # Batch 1: Dispatch resident experts immediately before waiting on NVMe
                if k1 > 0:
                    b1_s, b1_e = self.get_b1_events(layer_id)
                    b1_s.record(self.cache._h2d_stream)
                    fast_index_copy_multi_jit(
                        self.cache._copy_dst_ptrs,
                        self.cache._copy_src_ptrs[layer_id],
                        self.cache._copy_feat_bytes,
                        self.cache.evict_slots[:k1],
                        self.cache.src_indices[:k1],
                        None,
                    )
                    b1_e.record(self.cache._h2d_stream)
                    self.record_b1_batch(layer_id, k1)
        else:
            k1 = 0
            k2 = 0
            tag_to_entry_idx = {}

        # 3. Rolling Speculative Budget & Candidate Selection for Layer L+1
        speculative_jobs = []
        next_l = layer_id + 1
        if next_l < self.cache.num_layers and self.has_staged_prediction[next_l]:
            self.has_staged_prediction[next_l] = False
            currently_outstanding = self.cache.uring.outstanding_jobs()
            demand_jobs_count = len(demand_jobs)
            banks_per_expert = 2
            desired_outstanding_jobs = self.target_qd
            avail_jobs = max(0, desired_outstanding_jobs - currently_outstanding - demand_jobs_count)
            spec_budget = avail_jobs // banks_per_expert

            if spec_budget > 0:
                ranked = self._pinned_pred_np[next_l]
                issued_spec = 0
                for cand_e_val in ranked:
                    cand_e = int(cand_e_val)
                    if issued_spec >= spec_budget:
                        break
                    cand_key = (next_l, cand_e)
                    # DUPLICATE PREVENTION:
                    if cand_key in self.cache.gpu_pinned_experts:
                        continue
                    if cand_key in self.cache.host_slot_for_expert:
                        continue
                    if cand_key in self.in_flight_table:
                        continue
                    if cand_key in protect_keys:
                        continue

                    if self.cache.free_host_slots:
                        cand_slot = self.cache.free_host_slots.pop()
                    else:
                        evict_key, _ = self.cache.host_lru.pop_lru(protect=protect_keys)
                        cand_slot = self.cache.host_slot_for_expert.pop(evict_key)
                        self.cache.expert_for_host_slot[cand_slot] = None

                    protect_keys.add(cand_key)
                    entry = InFlightEntry(next_l, cand_e, cand_slot, is_speculative=True)
                    self.in_flight_table[cand_key] = entry
                    cand_tag = ("spec", next_l, cand_e, cand_slot)
                    cand_jobs = self.cache._build_expert_jobs(next_l, cand_e, cand_slot, cand_tag)
                    speculative_jobs.extend(cand_jobs)
                    tag_sizes[cand_tag] = len(cand_jobs)
                    issued_spec += 1
                    self.token_metrics["predicted_reads"] += 1
                    self.token_metrics["nvme_bytes_speculative"] += 2.64

        # 4. Submit unified batch (demand + speculative) in single syscall
        all_jobs = demand_jobs + speculative_jobs
        if all_jobs:
            self.cache.uring.submit_tagged_requests(all_jobs, tag_sizes)

        # 5. Track Queue Depth
        cur_qd = self.cache.uring.outstanding_jobs()
        self.token_metrics["peak_outstanding_qd"] = max(self.token_metrics["peak_outstanding_qd"], cur_qd)
        self.token_metrics["sum_outstanding_qd"] += cur_qd
        self.token_metrics["count_outstanding_qd"] += 1

        # 6. Wait for required demand tags with streaming H2D dispatch
        t_wait_0 = time.perf_counter()
        if required_demand_tags:
            def on_tag_ready(tag):
                if tag in tag_to_expert_info:
                    pos, key, slot = tag_to_expert_info[tag]
                    self.cache.host_slot_for_expert[key] = slot
                    self.cache.expert_for_host_slot[slot] = key
                    self.cache.host_lru.insert_new(key)
                    self.in_flight_table.pop(key, None)
                    if tag in tag_to_entry_idx:
                        idx = tag_to_entry_idx[tag]
                        with torch.cuda.stream(self.cache._h2d_stream):
                            fast_index_copy_multi_jit(
                                self.cache._copy_dst_ptrs,
                                self.cache._copy_src_ptrs[layer_id],
                                self.cache._copy_feat_bytes,
                                self.cache.evict_slots[idx : idx + 1],
                                self.cache.src_indices[idx : idx + 1],
                                None,
                            )
                        self.record_streaming_arrival(layer_id, 1)
                else:
                    _, t_layer, t_expert, _ = tag
                    t_key = (t_layer, t_expert)
                    if t_key in self.in_flight_table:
                        entry = self.in_flight_table[t_key]
                        entry.status = InFlightStatus.READY
                        entry.completed_at = time.perf_counter()

            self.cache.uring.wait_for_required_tags(required_demand_tags, on_tag_completed=on_tag_ready)
            wait_ms = (time.perf_counter() - t_wait_0) * 1000
            self.token_metrics["exposed_nvme_ms"] += wait_ms
            self.cache.time_nvme_read_s += (wait_ms / 1000)

        # 7. Pipeline event synchronization boundary
        if num_misses > 0:
            self.cache._h2d_pipeline_event.record(self.cache._h2d_stream)
            self.cache._pending_h2d_pipelined = True
        else:
            self.cache._pending_h2d_pipelined = False

        # 8. Finalize Layer L metadata
        self.cache._pending_src_layer = layer_id
        self.cache._pending_whole_layer = False

        # 9. Finalize false positives from prior layers (layer <= layer_id)
        stale_keys = [k for k in self.in_flight_table.keys() if k[0] <= layer_id]
        for s_key in stale_keys:
            s_entry = self.in_flight_table[s_key]
            if not s_entry.was_counted_fp:
                s_entry.was_counted_fp = True
                self.token_metrics["false_positives"] += 1
            if s_entry.status == InFlightStatus.READY:
                self.cache.host_slot_for_expert[s_key] = s_entry.slot
                self.cache.expert_for_host_slot[s_entry.slot] = s_key
                self.cache.host_lru.insert_new(s_key)
                del self.in_flight_table[s_key]

    def on_token_complete(self) -> None:
        # Drain any remaining in-flight requests so the ring is clean between tokens
        if hasattr(self.cache, "uring") and self.cache.uring is not None:
            remaining_tags = set(self.cache.uring._tag_remaining.keys())
            if remaining_tags:
                self.cache.uring.wait_for_required_tags(remaining_tags)

        for key, entry in list(self.in_flight_table.items()):
            if not entry.was_counted_fp:
                entry.was_counted_fp = True
                self.token_metrics["false_positives"] += 1
            self.cache.host_slot_for_expert[key] = entry.slot
            self.cache.expert_for_host_slot[entry.slot] = key
            self.cache.host_lru.insert_new(key)
        self.in_flight_table.clear()
        self.has_staged_prediction = [False] * self.cache.num_layers
        self._stream_active_set.clear()

        # Compute H2D copy time and wait time from recorded events
        if self._b1_active_layers or self._stream_active_layers or self._wait_active_layers:
            torch.cuda.synchronize()
            b1_ms = 0.0
            for l_id in self._b1_active_layers:
                s_ev, e_ev = self._h2d_events_b1[l_id]
                b1_ms += s_ev.elapsed_time(e_ev)
            self.token_metrics["b1_h2d_ms"] = b1_ms

            stream_ms = 0.0
            for l_id in self._stream_active_layers:
                s_ev, e_ev = self._h2d_events_stream[l_id]
                stream_ms += s_ev.elapsed_time(e_ev)
            self.token_metrics["stream_h2d_ms"] = stream_ms

            self.token_metrics["h2d_copy_ms"] = b1_ms + stream_ms

            wait_ms = 0.0
            for l_id in self._wait_active_layers:
                w_s, w_e = self._h2d_events_wait[l_id]
                wait_ms += w_s.elapsed_time(w_e)
            self.token_metrics["h2d_wait_ms"] = wait_ms

            self._b1_active_layers.clear()
            self._stream_active_layers.clear()
            self._wait_active_layers.clear()

        self.metrics["tokens"] += 1
        for k in self.token_metrics:
            self.metrics[k] += self.token_metrics[k]

        if self.trace_lookahead:
            tm = self.token_metrics
            qd_mean = (tm["sum_outstanding_qd"] / tm["count_outstanding_qd"]) if tm["count_outstanding_qd"] > 0 else 0.0
            recall = (tm["true_positives"] / tm["demanded_misses"] * 100) if tm["demanded_misses"] > 0 else 0.0
            pre_hide = ((tm["tp_completed_before_demand"] + tm["tp_already_completed_not_reaped"]) / tm["true_positives"] * 100) if tm["true_positives"] > 0 else 0.0
            in_flight_hide = (tm["tp_in_flight_at_demand"] / tm["true_positives"] * 100) if tm["true_positives"] > 0 else 0.0
            removable_ms = max(0.0, 94.5 - tm["exposed_nvme_ms"])
            removable_pct = (removable_ms / 94.5 * 100)

            logger.info(
                f"[Rolling Prefetcher | Token {self.metrics['tokens']}]\n"
                f"  Demand Misses: {tm['demanded_misses']} | Pred Issued: {tm['predicted_reads']} | "
                f"TP (Hits): {tm['true_positives']} ({recall:.1f}% miss recall) | FP: {tm['false_positives']} | FN (Fallbacks): {tm['fallback_reads']}\n"
                f"  Latency Hiding: 100% Pre-Demand = {tm['tp_completed_before_demand']} ({pre_hide:.1f}%) | "
                f"Unreaped CQE = {tm['tp_already_completed_not_reaped']} | Partial In-Flight = {tm['tp_in_flight_at_demand']} ({in_flight_hide:.1f}%)\n"
                f"  Queue Depth: Target QD = {self.target_qd} | Mean Outstanding QD = {qd_mean:.1f} | Peak Outstanding QD = {tm['peak_outstanding_qd']}\n"
                f"  I/O Volume: Demand = {tm['nvme_bytes_actual']:.1f} MB | Speculative = {tm['nvme_bytes_speculative']:.1f} MB | Total DMA = {(tm['nvme_bytes_actual'] + tm['nvme_bytes_speculative']):.1f} MB | Overhead = +{(tm['nvme_bytes_speculative']/max(1e-3, tm['nvme_bytes_actual'])*100):.1f}%\n"
                f"  Streaming H2D: Batch 1 (Pre-Wait) = {tm['b1_submits']} sub ({tm['b1_experts']} exp, {tm['b1_h2d_ms']:.1f} ms) | Streaming Demand = {tm['stream_submits']} sub ({tm['stream_experts']} exp, {tm['stream_h2d_ms']:.1f} ms) | Total H2D Span = {tm['h2d_copy_ms']:.1f} ms\n"
                f"  GPU Critical Path: Compute Idle Wait = {tm['h2d_wait_ms']:.1f} ms | Exposed NVMe Latency = {tm['exposed_nvme_ms']:.1f} ms (Baseline: 94.5 ms -> Removable: -{removable_ms:.1f} ms / {removable_pct:.1f}%)"
            )
        self.token_metrics = self._new_token_metrics()


_rolling_prefetcher_installed = False


def _install_rolling_lookahead_prefetcher() -> None:
    global _rolling_prefetcher_installed
    if _rolling_prefetcher_installed:
        return
    if os.getenv("FREETOKEN_SPECULATIVE_LOOKAHEAD", "0") != "1":
        return
    _rolling_prefetcher_installed = True

    from freetoken.core import Batch
    from freetoken.models.qwen4_exp.model import Qwen4ExpModel

    logger.info("[Rolling Prefetcher] Hooking Qwen4ExpModel forward for Candidate B speculative lookahead")
    orig_forward = Qwen4ExpModel.forward

    def prefetcher_model_forward(self, input_ids: torch.Tensor, batch: Batch) -> torch.Tensor:
        if not batch.is_decode:
            return orig_forward(self, input_ids, batch)

        cache = getattr(self.layers.op_list[0].mlp.experts, "offload_cache", None)
        prefetcher = getattr(cache, "prefetcher", None) if cache else None

        layers = self.layers.op_list
        num_layers = len(layers)

        hidden = self.embed_tokens.forward(input_ids).repeat(1, self.hc_count)
        meta = None
        if self._ple:
            from freetoken.models.qwen4_exp.ple import build_ple_metadata, commit_ngram_context
            meta = build_ple_metadata(batch, self._ple[0].args, input_ids.device)
            for ple in self._ple:
                ple.start_prefetch(batch, meta)

        for l, layer in enumerate(layers):
            if layer.ple is not None:
                hidden = hidden + layer.ple.forward(hidden, batch)

            block_input, inject = layer.attn_hyper_connection.mix(hidden)
            if layer._is_linear:
                block_output = layer.linear_attn.forward(block_input)
            else:
                block_output = layer.self_attn.forward(block_input, batch)
            hidden = layer.attn_hyper_connection.combine(hidden, block_output, inject)

            # Candidate B Lookahead: predict layer L+1 right after Attention L combine
            if prefetcher is not None and prefetcher.enabled and (l + 1) < num_layers:
                next_layer = layers[l + 1]
                with torch.no_grad():
                    b_B, _ = next_layer.mlp_hyper_connection.mix(hidden)
                    p_B = next_layer.mlp.gate.forward(b_B)
                    prefetcher.stage_prediction(l + 1, p_B.view(-1))

            block_input, inject = layer.mlp_hyper_connection.mix(hidden)
            moe_out = layer.mlp.forward(block_input)
            hidden = layer.mlp_hyper_connection.combine(hidden, moe_out, inject)

        if meta is not None:
            commit_ngram_context(meta, getattr(batch, "fla_metadata", None))

        if prefetcher is not None and prefetcher.enabled:
            prefetcher.on_token_complete()

        return self.hyper_connection_mixer.mix(hidden)[0]

    Qwen4ExpModel.forward = prefetcher_model_forward


def _extract_cli_arg(args: list[str], flag: str) -> tuple[str | None, list[str]]:
    """Helper to extract a specific CLI flag and its value from argv."""
    new_args = []
    val = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == flag:
            if i + 1 < len(args):
                val = args[i + 1]
                i += 2
                continue
            else:
                i += 1
                continue
        elif arg.startswith(flag + "="):
            val = arg.split("=", 1)[1]
            i += 1
            continue
        new_args.append(arg)
        i += 1
    return val, new_args


_glm5_pruning_patch_installed = False


def install_glm5_pruning_patch() -> None:
    """Installs CLI argument interception and dynamic top-k pruning / softmax thresholding
    specifically on GLM-5 (Glm5NextSparseBlock). Completely isolated to GLM-5; zero impact on Qwen.
    """
    global _glm5_pruning_patch_installed
    if _glm5_pruning_patch_installed:
        return
    _glm5_pruning_patch_installed = True

    # 1. Patch freetoken.server.args.parse_args to intercept --topk-override and --router-min-prob
    try:
        from freetoken.server import args as ft_server_args

        orig_parse_args = ft_server_args.parse_args

        def patched_parse_args(args: list[str], run_shell: bool = False, prog: str | None = None):
            topk_str, args = _extract_cli_arg(args, "--topk-override")
            if topk_str is not None:
                os.environ["FREETOKEN_TOPK_OVERRIDE"] = topk_str
                logger.info(f"[GLM Pruning] Intercepted CLI --topk-override {topk_str}")

            min_prob_str, args = _extract_cli_arg(args, "--router-min-prob")
            if min_prob_str is not None:
                os.environ["FREETOKEN_ROUTER_MIN_PROB"] = min_prob_str
                logger.info(f"[GLM Pruning] Intercepted CLI --router-min-prob {min_prob_str}")

            return orig_parse_args(args, run_shell=run_shell, prog=prog)

        ft_server_args.parse_args = patched_parse_args
    except Exception as e:
        logger.warning(f"[GLM Pruning] Could not patch parse_args: {e}")

    # 2. Patch Glm5NextSparseBlock._route
    try:
        from freetoken.models.glm5_next.moe import Glm5NextSparseBlock

        orig_route = Glm5NextSparseBlock._route

        def patched_route(self, hidden_states: torch.Tensor):
            topk_override_env = os.getenv("FREETOKEN_TOPK_OVERRIDE", os.getenv("FREETOKEN_GLM_TOPK", None))
            router_min_prob_env = os.getenv("FREETOKEN_ROUTER_MIN_PROB", os.getenv("FREETOKEN_GLM_ROUTER_MIN_PROB", None))

            effective_topk = self.top_k
            if topk_override_env is not None:
                try:
                    ovr = int(topk_override_env)
                    if 1 <= ovr <= self.top_k:
                        effective_topk = ovr
                except ValueError:
                    pass

            min_prob = 0.0
            if router_min_prob_env is not None:
                try:
                    mp = float(router_min_prob_env)
                    if 0.0 < mp < 1.0:
                        min_prob = mp
                except ValueError:
                    pass

            # Fast path: unpruned baseline
            if effective_topk == self.top_k and min_prob <= 0.0:
                return orig_route(self, hidden_states)

            # HF computes router logits in fp32 (moe_router_dtype: float32); match exactly.
            logits = F.linear(hidden_states.float(), self.gate.weight.float())
            scores = logits.sigmoid()
            scores_for_choice = scores + self.e_score_correction_bias.float()
            if self.n_group > 1:
                scores_for_choice = self._group_limited(scores_for_choice)

            _, topk_ids = torch.topk(scores_for_choice, effective_topk, dim=-1)
            topk_weights = scores.gather(-1, topk_ids)

            if self.norm_topk_prob:
                norm_probs = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
            else:
                norm_probs = topk_weights

            if min_prob > 0.0:
                sorted_probs, sort_idx = torch.sort(norm_probs, dim=-1, descending=True)
                sorted_ids = topk_ids.gather(-1, sort_idx)

                k_per_token = (sorted_probs >= min_prob).sum(dim=-1)
                min_k = int(os.getenv("FREETOKEN_ROUTER_MIN_K", "1"))
                k_kept = max(min_k, int(k_per_token.max().item()))

                if k_kept < effective_topk:
                    topk_ids = sorted_ids[:, :k_kept]
                    sorted_probs = sorted_probs[:, :k_kept]
                    if self.norm_topk_prob:
                        topk_weights = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-20)
                    else:
                        topk_weights = sorted_probs
                else:
                    topk_ids = sorted_ids
                    topk_weights = sorted_probs
            else:
                topk_weights = norm_probs

            topk_weights = topk_weights * self.routed_scaling_factor

            if os.getenv("FREETOKEN_TRACE_ROUTING", "0") == "1":
                layer_id = getattr(getattr(self, "experts", None), "layer_id", -1)
                logger.info(
                    f"[GLM Router Trace] layer {layer_id}: routed {topk_ids.shape[1]}/{self.top_k} experts "
                    f"(effective_topk={effective_topk}, min_prob={min_prob})"
                )

            return topk_weights.to(torch.float32).contiguous(), topk_ids.to(torch.int32).contiguous()

        Glm5NextSparseBlock._route = patched_route
        logger.info(
            f"[GLM Pruning Patch] Successfully installed on Glm5NextSparseBlock "
            f"(topk_override={os.getenv('FREETOKEN_TOPK_OVERRIDE')}, min_prob={os.getenv('FREETOKEN_ROUTER_MIN_PROB')})"
        )
    except Exception as e:
        logger.warning(f"[GLM Pruning Patch] Failed to patch Glm5NextSparseBlock: {e}")


