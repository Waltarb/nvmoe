#!/usr/bin/env python3
"""NVMe read-shape / queue-depth / O_DIRECT sweep against the REAL model shards.

Answers one question the production telemetry can't: is the ~2.4 GB/s the decode path
sustains (`[Latency Breakdown]`'s NVMe I/O line) the drive's wall, or an artifact of how
this cache issues reads?

Two independent variables the rest of this repo never separated:

1. **Buffered vs O_DIRECT.** Despite "Direct I/O" appearing throughout
   RESULTS_AND_FINDINGS.md, neither `nvme_offload_cache.py:588` nor `bench.py:147` ever
   passes `os.O_DIRECT` -- both open `O_RDONLY` and evict the page cache with
   `posix_fadvise(DONTNEED)` before timing. That's a cold *buffered* read: correct for
   "is it on disk", but it still pays a kernel-page-cache -> userspace memcpy per byte,
   and pushes ~300 MB/token of expert weights through a page cache on a machine sized to
   keep ~10 GiB free for the desktop.
2. **Read shape.** §32's 3.10 GB/s came from a full-layer sweep (512 contiguous experts,
   1.35 GB in one go). Decode issues ~10 scattered experts from one layer at a time. Those
   are different workloads and were never compared head to head, so the "2.4 vs 3.1 gap"
   may be shape, not a reclaimable inefficiency.

Reads real weight bytes only; never writes to the model directory, never loads the model,
never touches a running server. Every measurement is preceded by
`posix_fadvise(..., POSIX_FADV_DONTNEED)` over the exact ranges about to be read -- this
repo's standing rule, since a prior run's page cache silently inflated results before.
"""
from __future__ import annotations

import argparse
import bisect
import ctypes
import json
import mmap
import os
import random
import statistics
import time

MODEL_DIR = os.path.expanduser("~/.freetoken/models/qwen3.8-flash-next-nvfp4-ftw")

# --------------------------------------------------------------------------
# Minimal io_uring bindings (standalone -- this harness stays independent of
# nvme_offload_cache.py, per this repo's one-bench-one-file convention)
# --------------------------------------------------------------------------
_RING_STRUCT_SIZE = 512


class _Cqe(ctypes.Structure):
    _fields_ = [("user_data", ctypes.c_uint64), ("res", ctypes.c_int32), ("flags", ctypes.c_uint32)]


class Uring:
    def __init__(self, depth: int = 256):
        self.lib = ctypes.CDLL("liburing-ffi.so.2")
        self.lib.io_uring_queue_init.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
        self.lib.io_uring_queue_init.restype = ctypes.c_int
        self.lib.io_uring_queue_exit.argtypes = [ctypes.c_void_p]
        self.lib.io_uring_get_sqe.argtypes = [ctypes.c_void_p]
        self.lib.io_uring_get_sqe.restype = ctypes.c_void_p
        self.lib.io_uring_prep_read.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint64
        ]
        self.lib.io_uring_sqe_set_data64.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.lib.io_uring_submit_and_wait.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        self.lib.io_uring_submit_and_wait.restype = ctypes.c_int
        self.lib.io_uring_wait_cqe.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.io_uring_wait_cqe.restype = ctypes.c_int
        self.lib.io_uring_cqe_seen.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._buf = ctypes.create_string_buffer(_RING_STRUCT_SIZE)
        self.ring = ctypes.addressof(self._buf)
        rc = self.lib.io_uring_queue_init(depth, self.ring, 0)
        if rc < 0:
            raise RuntimeError(f"io_uring_queue_init: {os.strerror(-rc)}")
        self.cqe_ptr = ctypes.c_void_p()

    def read_batch(self, jobs, max_inflight: int) -> None:
        """jobs: [(fd, ptr, nbytes, offset)]. Keeps at most `max_inflight` in flight, so
        this sweeps true queue depth rather than just batch size."""
        n = len(jobs)
        submitted = reaped = 0
        while reaped < n:
            while submitted < n and (submitted - reaped) < max_inflight:
                fd, ptr, nb, off = jobs[submitted]
                sqe = self.lib.io_uring_get_sqe(self.ring)
                if not sqe:
                    break
                self.lib.io_uring_prep_read(sqe, fd, ctypes.c_void_p(ptr), nb, off)
                self.lib.io_uring_sqe_set_data64(sqe, submitted)
                submitted += 1
            self.lib.io_uring_submit_and_wait(self.ring, 1)
            while reaped < submitted:
                rc = self.lib.io_uring_wait_cqe(self.ring, ctypes.byref(self.cqe_ptr))
                if rc < 0:
                    raise RuntimeError(f"wait_cqe: {os.strerror(-rc)}")
                cqe = ctypes.cast(self.cqe_ptr, ctypes.POINTER(_Cqe)).contents
                res = cqe.res
                self.lib.io_uring_cqe_seen(self.ring, self.cqe_ptr)
                if res < 0:
                    raise RuntimeError(f"read failed: {os.strerror(-res)}")
                reaped += 1
                if (submitted - reaped) < max_inflight and submitted < n:
                    break

    def close(self) -> None:
        self.lib.io_uring_queue_exit(self.ring)


# --------------------------------------------------------------------------
# Manifest -> expert bank locations (mirrors nvme_offload_cache._init_nvme_manifest)
# --------------------------------------------------------------------------
class Bank:
    __slots__ = ("file", "off", "row_bytes", "total_bytes")

    def __init__(self, file, off, row_bytes, total_bytes):
        self.file, self.off, self.row_bytes, self.total_bytes = file, off, row_bytes, total_bytes


def load_banks(model_dir: str):
    with open(os.path.join(model_dir, "freetoken_weight.json"), encoding="utf-8") as f:
        man = json.load(f)
    shards = sorted(man["shards"], key=lambda s: s["global_off"])
    offs = [s["global_off"] for s in shards]
    out: dict[tuple[int, str], Bank] = {}
    names: set[str] = set()
    for t in man["tensors"]:
        if t["kind"] != "experts_bank" or "#L" not in t["name"]:
            continue
        bank_name, lid = t["name"].split("#L")
        lid = int(lid)
        i = bisect.bisect_right(offs, t["global_off"]) - 1
        shard = shards[i]
        rows = t["shape"][0]
        out[(lid, bank_name)] = Bank(
            shard["file"], t["global_off"] - shard["global_off"], t["nbytes"] // rows, t["nbytes"]
        )
        names.add(bank_name)
    return out, sorted(names)


def drop_cache(fd: int, ranges) -> None:
    """Evict exactly the ranges about to be read (repo standing rule: an unmeasured cold
    read is a page-cache hit waiting to inflate the number)."""
    for off, nb in ranges:
        os.posix_fadvise(fd, off, nb, os.POSIX_FADV_DONTNEED)


def run_case(banks, bank_names, model_dir, *, shape, qd, direct, layers, experts_per_layer,
             seed, big_banks_only=True):
    rnd = random.Random(seed)
    flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
    fds: dict[str, int] = {}

    use_banks = bank_names
    if big_banks_only:
        # Match the production read path: banks preloaded wholesale into RAM
        # (gate_up_global/down_global scale banks) are memcpy'd, never read per-expert.
        use_banks = [b for b in bank_names if not b.endswith("_global")]

    jobs, ranges_by_fd, bufs = [], {}, []
    total = 0
    for lid in layers:
        if shape == "scattered":
            eids = [rnd.randrange(0, 512) for _ in range(experts_per_layer)]
        else:  # full-layer sequential sweep, as in RESULTS_AND_FINDINGS.md §32
            eids = None
        for bn in use_banks:
            bk = banks[(lid, bn)]
            if bk.file not in fds:
                fds[bk.file] = os.open(os.path.join(model_dir, bk.file), flags)
                ranges_by_fd[bk.file] = []
            fd = fds[bk.file]
            if eids is None:
                off, nb = bk.off, bk.total_bytes
                if direct and (off % 4096 or nb % 4096):
                    continue
                buf = mmap.mmap(-1, nb)  # page-aligned, required for O_DIRECT
                bufs.append(buf)
                jobs.append((fd, _addr(buf), nb, off))
                ranges_by_fd[bk.file].append((off, nb))
                total += nb
            else:
                for e in eids:
                    off, nb = bk.off + e * bk.row_bytes, bk.row_bytes
                    if direct and (off % 4096 or nb % 4096):
                        continue
                    buf = mmap.mmap(-1, nb)
                    bufs.append(buf)
                    jobs.append((fd, _addr(buf), nb, off))
                    ranges_by_fd[bk.file].append((off, nb))
                    total += nb

    if not jobs:
        for fd in fds.values():
            os.close(fd)
        return None

    for fname, rs in ranges_by_fd.items():
        drop_cache(fds[fname], rs)

    ring = Uring(depth=max(256, qd * 2))
    t0 = time.perf_counter()
    ring.read_batch(jobs, max_inflight=qd)
    dt = time.perf_counter() - t0
    ring.close()

    for fd in fds.values():
        os.close(fd)
    for b in bufs:
        b.close()
    return total, dt, len(jobs)


def _addr(m: mmap.mmap) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(m))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--qd", default="4,8,16,32,64,128")
    ap.add_argument("--experts-per-layer", type=int, default=10, help="decode routes top-10")
    ap.add_argument("--layers", type=int, default=12, help="layers per scattered trial")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    banks, names = load_banks(args.model_dir)
    nlayers = 1 + max(l for l, _ in banks)
    per_expert = sum(banks[(0, b)].row_bytes for b in names if not b.endswith("_global"))
    print(f"model: {args.model_dir}")
    print(f"banks: {names}")
    print(f"layers: {nlayers} | per-expert NVMe bytes (non-global banks): {per_expert/1e6:.3f} MB")
    a = banks[(0, [b for b in names if not b.endswith('_global')][0])]
    print(f"alignment: first bank off%4096={a.off % 4096}, row_bytes%4096={a.row_bytes % 4096}")
    print()

    qds = [int(x) for x in args.qd.split(",")]
    print(f"{'shape':<11} {'mode':<9} {'QD':>4} {'reqs':>6} {'MB':>8} {'ms':>8} {'GB/s':>7}")
    print("-" * 60)

    for shape in ("scattered", "sequential"):
        for direct in (False, True):
            for qd in qds:
                if shape == "sequential" and qd > 32:
                    continue
                rates, ms_all = [], []
                for r in range(args.repeats):
                    if shape == "scattered":
                        layers = [ (args.seed + r * 7 + i * 3) % nlayers for i in range(args.layers) ]
                    else:
                        layers = [(args.seed + r) % nlayers]
                    res = run_case(
                        banks, names, args.model_dir, shape=shape, qd=qd, direct=direct,
                        layers=layers, experts_per_layer=args.experts_per_layer,
                        seed=args.seed + r,
                    )
                    if res is None:
                        break
                    tot, dt, nreq = res
                    rates.append(tot / dt / 1e9)
                    ms_all.append(dt * 1e3)
                if not rates:
                    print(f"{shape:<11} {'O_DIRECT' if direct else 'buffered':<9} {qd:>4}  (skipped: alignment)")
                    continue
                print(f"{shape:<11} {'O_DIRECT' if direct else 'buffered':<9} {qd:>4} "
                      f"{nreq:>6} {tot/1e6:>8.1f} {statistics.median(ms_all):>8.1f} "
                      f"{statistics.median(rates):>7.2f}")
        print()


if __name__ == "__main__":
    main()
