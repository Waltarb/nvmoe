#!/usr/bin/env python3
"""
NVMe-tier MoE expert cache feasibility benchmark.

Simulates a 3-tier (VRAM / RAM / NVMe) LRU cache for Mixture-of-Experts
expert weights, replaying a synthetic Zipf-skewed routing trace against the
REAL on-disk expert bytes of an already-downloaded FreeToken model
(Qwen3.6-35B-A3B-NVFP4). This does NOT touch or modify FreeToken itself,
does NOT run the model, and never writes to the model directory -- it only
reads real weight bytes to get honest NVMe latency numbers, and times a
synthetic cache-replacement simulation against them.

See README.md in this directory for the design rationale.
"""
import argparse
import bisect
import json
import os
import statistics
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    from iouring_ffi import IoUring, IoUringError
except OSError:
    IoUring = None
    IoUringError = RuntimeError

DEFAULT_MODEL_DIR = os.path.expanduser(
    "~/.freetoken/models/Qwen3.6-35B-A3B-NVFP4"
)


# --------------------------------------------------------------------------
# Manifest parsing
# --------------------------------------------------------------------------

class ExpertTensorLoc:
    """Where one (layer, tensor-kind) row-stacked tensor lives on disk."""

    __slots__ = ("fd", "tensor_off_in_shard", "row_bytes", "num_rows")

    def __init__(self, fd, tensor_off_in_shard, row_bytes, num_rows):
        self.fd = fd
        self.tensor_off_in_shard = tensor_off_in_shard
        self.row_bytes = row_bytes
        self.num_rows = num_rows

    def offset_for(self, expert_id):
        return self.tensor_off_in_shard + expert_id * self.row_bytes

    def read_expert(self, expert_id):
        return os.pread(self.fd, self.row_bytes, self.offset_for(expert_id))

    def drop_from_cache(self, expert_id):
        off = self.offset_for(expert_id)
        try:
            os.posix_fadvise(self.fd, off, self.row_bytes, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass  # best-effort; not fatal if unsupported


# --------------------------------------------------------------------------
# I/O backends -- pluggable so the cache simulation doesn't care whether a
# "read this expert's bytes" job is served by an OS thread doing a blocking
# pread(), or by an io_uring submission reaped on the main thread.
# --------------------------------------------------------------------------

class ThreadPoolBackend:
    name = "threadpool"

    def __init__(self, locs, workers):
        self.locs = locs
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def submit(self, layer, expert_id):
        loc_gu = self.locs[(layer, "gu")]
        loc_dn = self.locs[(layer, "down")]

        def job():
            loc_gu.read_expert(expert_id)
            loc_dn.read_expert(expert_id)
            loc_gu.drop_from_cache(expert_id)
            loc_dn.drop_from_cache(expert_id)

        return self.pool.submit(job)

    def wait(self, handle):
        handle.result()

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


class IoUringBackend:
    name = "iouring"

    def __init__(self, locs, queue_depth):
        if IoUring is None:
            raise RuntimeError(
                "liburing-ffi.so not loadable on this system -- "
                "install liburing (pacman -S liburing) to use --io-backend iouring"
            )
        self.locs = locs
        # Two reads (gate_up + down) per in-flight expert job, so give the
        # ring headroom for that.
        self.ring = IoUring(queue_depth=max(8, queue_depth * 2))

    def submit(self, layer, expert_id):
        loc_gu = self.locs[(layer, "gu")]
        loc_dn = self.locs[(layer, "down")]
        u_gu = self.ring.submit_read(
            loc_gu.fd, loc_gu.row_bytes, loc_gu.offset_for(expert_id)
        )
        u_dn = self.ring.submit_read(
            loc_dn.fd, loc_dn.row_bytes, loc_dn.offset_for(expert_id)
        )
        return (u_gu, u_dn, loc_gu, loc_dn, expert_id)

    def wait(self, handle):
        u_gu, u_dn, loc_gu, loc_dn, expert_id = handle
        self.ring.wait(u_gu)
        self.ring.wait(u_dn)
        loc_gu.drop_from_cache(expert_id)
        loc_dn.drop_from_cache(expert_id)

    def close(self):
        self.ring.close()


def load_manifest(model_dir):
    with open(os.path.join(model_dir, "freetoken_weight.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    return manifest, config


def open_shard_fds(model_dir, manifest):
    fds = {}
    for shard in manifest["shards"]:
        path = os.path.join(model_dir, shard["file"])
        fds[shard["file"]] = os.open(path, os.O_RDONLY)
    return fds


def find_shard(shards_sorted_offsets, shards, global_off):
    i = bisect.bisect_right(shards_sorted_offsets, global_off) - 1
    shard = shards[i]
    assert shard["global_off"] <= global_off < shard["global_off"] + shard["nbytes"], (
        f"offset {global_off} not within any shard"
    )
    return shard


def build_expert_locations(model_dir, manifest, config, fds):
    """Returns dict[(layer, kind)] -> ExpertTensorLoc for kind in (gu, down)."""
    tc = config.get("text_config", config)
    num_layers = tc["num_hidden_layers"]
    num_experts = tc["num_experts"]

    tensors_by_name = {t["name"]: t for t in manifest["tensors"]}
    shards = sorted(manifest["shards"], key=lambda s: s["global_off"])
    shard_offsets = [s["global_off"] for s in shards]

    locs = {}
    for layer in range(num_layers):
        for kind, prefix in (("gu", "gate_up_packed"), ("down", "down_packed")):
            name = f"{prefix}#L{layer:05d}"
            t = tensors_by_name.get(name)
            if t is None:
                raise KeyError(f"manifest missing tensor {name!r}")
            shard = find_shard(shard_offsets, shards, t["global_off"])
            off_in_shard = t["global_off"] - shard["global_off"]
            num_rows = t["shape"][0]
            row_bytes = t["nbytes"] // num_rows
            assert num_rows == num_experts, (layer, kind, num_rows, num_experts)
            locs[(layer, kind)] = ExpertTensorLoc(
                fds[shard["file"]], off_in_shard, row_bytes, num_rows
            )
    return locs, num_layers, num_experts


# --------------------------------------------------------------------------
# PCIe H2D bandwidth calibration (real, small transfers -- doesn't need a
# large persistent VRAM allocation, so it works fine even while FreeToken
# has most of the GPU's VRAM in use)
# --------------------------------------------------------------------------

def calibrate_h2d_bandwidth_bytes_per_sec(transfer_size_bytes):
    """Calibrate at the SAME transfer size the simulation actually moves
    (one expert's bytes, ~1.5MiB here), not an arbitrary size. H2D bandwidth
    is strongly size-dependent below ~128MB (fixed per-call/kernel-launch
    overhead dominates): measured on this machine, 1.5MiB/4MiB/16MiB all land
    around ~3.2GB/s, while 512MiB-1GiB batched transfers reach ~12GB/s. Using
    a large-transfer number here would understate the real cost of RAM-tier
    promotions, which move one expert (small) at a time."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        size = transfer_size_bytes
        host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            dev = host.to("cuda", non_blocking=False)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            del dev
        torch.cuda.empty_cache()
        median_t = statistics.median(times)
        return size / median_t
    except RuntimeError as e:
        print(f"  (H2D calibration failed: {e}; falling back to fixed estimate)",
              file=sys.stderr)
        return None


# --------------------------------------------------------------------------
# Synthetic routing trace
# --------------------------------------------------------------------------

def make_zipf_probs(num_experts, skew, rng):
    ranks = np.arange(1, num_experts + 1)
    weights = ranks.astype(np.float64) ** (-skew)
    weights /= weights.sum()
    return weights


def make_trace(num_layers, num_experts, top_k, steps, skew, seed, temporal_corr=0.0):
    """temporal_corr in [0,1]: probability that a given (step, layer)'s
    selection is simply repeated from the previous step rather than freshly
    drawn. 0.0 = i.i.d. Zipf per step (zero temporal correlation -- a
    stationary distribution has no session-level locality beyond what a
    fixed popularity skew gives you for free). Real routing plausibly has
    session-level working-set locality (a given prompt/context keeps hitting
    a related subset of experts) that i.i.d. draws can't represent; this is
    a deliberately simple proxy for that, not a captured real trace."""
    rng = np.random.default_rng(seed)
    base_probs = make_zipf_probs(num_experts, skew, rng)
    # Each layer gets its own random assignment of "which physical expert
    # is rank 1/2/3/..." so hot experts differ per layer, like real routing.
    layer_perms = [rng.permutation(num_experts) for _ in range(num_layers)]
    layer_probs = [base_probs[np.argsort(perm)] for perm in layer_perms]

    trace = np.empty((steps, num_layers, top_k), dtype=np.int32)
    for L in range(num_layers):
        p = layer_probs[L]
        trace[0, L] = rng.choice(num_experts, size=top_k, replace=False, p=p)
        for t in range(1, steps):
            if temporal_corr > 0.0 and rng.random() < temporal_corr:
                trace[t, L] = trace[t - 1, L]
            else:
                trace[t, L] = rng.choice(num_experts, size=top_k, replace=False, p=p)
    return trace


# --------------------------------------------------------------------------
# LRU tier helpers
# --------------------------------------------------------------------------

def lru_touch(od: OrderedDict, key):
    if key in od:
        od.move_to_end(key)
        return True
    return False


def lru_insert(od: OrderedDict, key, capacity):
    """Insert key as most-recently-used; return evicted key or None."""
    if key in od:
        od.move_to_end(key)
        return None
    od[key] = True
    if len(od) > capacity:
        evicted, _ = od.popitem(last=False)
        return evicted
    return None


# --------------------------------------------------------------------------
# Simulation core
# --------------------------------------------------------------------------

def run_one_config(
    locs, num_layers, num_experts, trace, top_k,
    vram_cap, ram_cap, compute_budget_s, h2d_bw_bytes_s, backend,
    prefetch_mode="one-layer-ideal",
):
    steps = trace.shape[0]
    vram = [OrderedDict() for _ in range(num_layers)]
    ram = [OrderedDict() for _ in range(num_layers)]

    hits_vram = hits_ram = misses_nvme = 0
    prefetched_hits = 0
    token_latencies_ms = []
    stall_count = 0

    pending_prefetch = {}  # (layer, expert_id) -> backend-specific handle

    def row_bytes_total(layer):
        return locs[(layer, "gu")].row_bytes + locs[(layer, "down")].row_bytes

    for t in range(steps):
        if prefetch_mode == "next-token-realistic" and t + 1 < steps:
            # Realistic (non-cheating) prefetch: use THIS token's own
            # just-determined routing as the prediction for the NEXT
            # token's routing, at every layer, all issued up front -- so
            # the fetch has this whole token's compute time (~all layers'
            # budget, not just one layer's) to land before it's needed.
            # This only uses information already available when token t
            # starts being processed, not ground truth about token t+1.
            for L in range(num_layers):
                for e in trace[t, L]:
                    e = int(e)
                    if e in vram[L] or e in ram[L]:
                        continue
                    key = (L, e)
                    if key not in pending_prefetch:
                        pending_prefetch[key] = backend.submit(L, e)

        token_time_s = 0.0
        for L in range(num_layers):
            nonlocal_extra = 0.0
            needed = trace[t, L]

            cold_needed = []
            for e in needed:
                e = int(e)
                if lru_touch(vram[L], e):
                    hits_vram += 1
                    continue
                if lru_touch(ram[L], e):
                    hits_ram += 1
                    # promote ram -> vram
                    evicted = lru_insert(vram[L], e, vram_cap)
                    if evicted is not None:
                        lru_insert(ram[L], evicted, ram_cap)
                    nonlocal_extra = max(
                        nonlocal_extra, row_bytes_total(L) / h2d_bw_bytes_s
                    )
                    continue
                # miss both tiers -- either a pending prefetch or cold
                key = (L, e)
                handle = pending_prefetch.pop(key, None)
                if handle is not None:
                    w0 = time.perf_counter()
                    backend.wait(handle)
                    wait = time.perf_counter() - w0
                    prefetched_hits += 1
                    nonlocal_extra = max(nonlocal_extra, wait)
                    lru_insert(ram[L], e, ram_cap)
                else:
                    cold_needed.append(e)

            if cold_needed:
                handles = [backend.submit(L, e) for e in cold_needed]
                t0 = time.perf_counter()
                for h in handles:
                    backend.wait(h)
                batch_time = time.perf_counter() - t0
                misses_nvme += len(cold_needed)
                nonlocal_extra = max(nonlocal_extra, batch_time)
                for e in cold_needed:
                    lru_insert(ram[L], e, ram_cap)
                    # evicted from ram tier is simply dropped (goes "cold")

            layer_time = compute_budget_s + nonlocal_extra
            if nonlocal_extra > compute_budget_s:
                # extra time alone exceeds the whole normal compute
                # budget -- a materially visible hitch, not just noise
                stall_count += 1
            token_time_s += layer_time

            if prefetch_mode == "one-layer-ideal":
                # Idealized one-layer-ahead prefetch: peek at ground-truth
                # future routing (upper bound on what a real predictor could
                # achieve) and kick off reads for the next layer's likely
                # misses while this layer's "compute" happens.
                next_L, next_t = (L + 1, t) if L + 1 < num_layers else (0, t + 1)
                if next_t < steps:
                    for e in trace[next_t, next_L]:
                        e = int(e)
                        if e in vram[next_L] or e in ram[next_L]:
                            continue
                        key = (next_L, e)
                        if key not in pending_prefetch:
                            pending_prefetch[key] = backend.submit(next_L, e)

            time.sleep(compute_budget_s)

        token_latencies_ms.append(token_time_s * 1000.0)

    # prefetched_hits and misses_nvme are both NVMe-tier accesses --
    # they differ only in whether the fetch was hidden by prefetch (fast)
    # or hit the token synchronously (a cold read).
    total_nvme = prefetched_hits + misses_nvme
    total_lookups = hits_vram + hits_ram + total_nvme
    lat = np.array(token_latencies_ms)
    return {
        "vram_cap": vram_cap,
        "ram_cap": ram_cap,
        "hit_rate_vram": hits_vram / total_lookups,
        "hit_rate_ram": hits_ram / total_lookups,
        "hit_rate_nvme": total_nvme / total_lookups,
        "prefetch_effectiveness": (
            prefetched_hits / total_nvme if total_nvme else float("nan")
        ),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "max_ms": float(lat.max()),
        "mean_ms": float(lat.mean()),
        "tok_s": 1000.0 / lat.mean(),
        "stall_fraction": stall_count / (steps * num_layers),
    }


def parse_sweep(spec):
    out = []
    for pair in spec.split(","):
        v, r = pair.split(":")
        out.append((int(v), int(r)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--steps", type=int, default=500,
                     help="number of simulated decode tokens")
    ap.add_argument("--skew", type=float, default=1.2,
                     help="Zipf skew for expert popularity (higher = more skewed)")
    ap.add_argument("--tok-budget-ms", type=float, default=None,
                     help="per-layer compute budget in ms "
                          "(default: derived from live FreeToken tok/s if reachable, "
                          "else 0.514ms = 1000/48.6/40)")
    ap.add_argument("--workers", type=int, default=8,
                     help="thread pool size for concurrent NVMe reads "
                          "(--io-backend threadpool only)")
    ap.add_argument("--io-backend", choices=["threadpool", "iouring"],
                     default="threadpool",
                     help="how NVMe reads are issued/reaped")
    ap.add_argument("--queue-depth", type=int, default=192,
                     help="io_uring submission queue depth "
                          "(--io-backend iouring only; next-token-realistic "
                          "prefetch can submit up to top_k*num_layers reads "
                          "at once, so this needs headroom)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--prefetch-mode",
        choices=["none", "one-layer-ideal", "next-token-realistic"],
        default="one-layer-ideal",
        help="none: no prefetch. one-layer-ideal: peek ground-truth next-"
             "layer routing (idealized upper bound, ~0.5ms window). "
             "next-token-realistic: use this token's OWN routing as the "
             "prediction for the next token's, issued for all layers up "
             "front (realistic, no cheating, ~20ms window).",
    )
    ap.add_argument(
        "--temporal-corr", type=float, default=0.0,
        help="probability a (step,layer) selection repeats the previous "
             "step's instead of a fresh i.i.d. Zipf draw (0=no session "
             "locality beyond the fixed skew; try e.g. 0.8)",
    )
    ap.add_argument(
        "--sweep", default="0:0,4:16,16:64,32:128",
        help="comma-separated vram_experts_per_layer:ram_experts_per_layer configs",
    )
    args = ap.parse_args()

    print(f"Loading manifest from {args.model_dir} ...")
    manifest, config = load_manifest(args.model_dir)
    fds = open_shard_fds(args.model_dir, manifest)
    locs, num_layers, num_experts = build_expert_locations(
        args.model_dir, manifest, config, fds
    )
    top_k = config.get("text_config", config)["num_experts_per_tok"]
    print(f"  layers={num_layers} num_experts={num_experts} top_k={top_k}")
    expert_bytes = locs[(0, "gu")].row_bytes + locs[(0, "down")].row_bytes
    print(f"  bytes/expert (gate_up+down only) = {expert_bytes:,}")

    tok_budget_ms = args.tok_budget_ms
    if tok_budget_ms is None:
        tok_budget_ms = 1000.0 / 48.6 / num_layers
        print(f"  using default per-layer compute budget: {tok_budget_ms:.4f} ms "
              f"(derived from 48.6 tok/s live measurement / {num_layers} layers)")
    compute_budget_s = tok_budget_ms / 1000.0

    print(f"Calibrating H2D (pinned RAM -> GPU) bandwidth at the real "
          f"per-expert transfer size ({expert_bytes:,} bytes) ...")
    h2d_bw = calibrate_h2d_bandwidth_bytes_per_sec(expert_bytes)
    if h2d_bw is None:
        h2d_bw = 3.2e9  # measured small-transfer (~1.5MiB) figure on this laptop;
        # NOTE: large-batch H2D transfers (128MB+) reach ~12GB/s on this
        # machine, but that's not what a single expert promotion moves.
        print(f"  no CUDA available/usable -- using fallback estimate "
              f"{h2d_bw/1e9:.1f} GB/s (measured small-transfer figure, "
              f"not a large-batch number)")
    else:
        print(f"  measured: {h2d_bw/1e9:.2f} GB/s")

    print(f"Generating synthetic Zipf(skew={args.skew}, "
          f"temporal_corr={args.temporal_corr}) routing trace "
          f"for {args.steps} tokens ...")
    trace = make_trace(
        num_layers, num_experts, top_k, args.steps, args.skew, args.seed,
        temporal_corr=args.temporal_corr,
    )

    sweep = parse_sweep(args.sweep)
    print(f"I/O backend: {args.io_backend}  |  prefetch mode: {args.prefetch_mode}")
    print()
    header = (
        f"{'vram':>5} {'ram':>5} {'hit_vram':>9} {'hit_ram':>8} {'hit_nvme':>9} "
        f"{'pfx_eff':>8} {'p50ms':>8} {'p95ms':>8} {'p99ms':>8} {'maxms':>9} "
        f"{'tok/s':>7} {'stall%':>7}"
    )
    print(header)
    print("-" * len(header))
    for vram_cap, ram_cap in sweep:
        if args.io_backend == "iouring":
            backend = IoUringBackend(locs, args.queue_depth)
        else:
            backend = ThreadPoolBackend(locs, args.workers)
        try:
            r = run_one_config(
                locs, num_layers, num_experts, trace, top_k,
                vram_cap, ram_cap, compute_budget_s, h2d_bw, backend,
                prefetch_mode=args.prefetch_mode,
            )
        finally:
            backend.close()
        print(
            f"{r['vram_cap']:>5} {r['ram_cap']:>5} "
            f"{r['hit_rate_vram']*100:>8.1f}% {r['hit_rate_ram']*100:>7.1f}% "
            f"{r['hit_rate_nvme']*100:>8.1f}% {r['prefetch_effectiveness']*100:>7.1f}% "
            f"{r['p50_ms']:>8.2f} {r['p95_ms']:>8.2f} {r['p99_ms']:>8.2f} "
            f"{r['max_ms']:>9.2f} {r['tok_s']:>7.2f} {r['stall_fraction']*100:>6.1f}%"
        )

    for fd in fds.values():
        os.close(fd)


if __name__ == "__main__":
    main()
