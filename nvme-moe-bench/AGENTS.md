# AGENTS.md

Instructions for any coding agent working in this repository. This is the
single source of truth — `CLAUDE.md` is only a pointer here, so edit this file
rather than duplicating guidance into a tool-specific one.

## What this is

A 3-tier (GPU VRAM / pinned host RAM / NVMe) MoE expert-offload cache for
FreeToken, plus the benchmarks and research notes that produced it.
`nvme_offload_cache.py` is the real, deployed cache implementation
(`NvmeOffloadMoeCache`); `bench.py` and friends are standalone
feasibility/measurement harnesses that read real weight bytes off disk
without touching the live model server. No CI, no
conventional test suite — correctness is established by live measurement
against the real server and real NVMe shards, recorded in the docs.

## Environment

- Run all FreeToken-aware Python (including this cache) under
  `~/.freetoken/venv/bin/python3` — the only local interpreter with a
  CUDA-enabled torch build.
- `~/.freetoken/venv/lib/python3.12/site-packages/sitecustomize.py` is the
  integration hook: with `FREETOKEN_NVME_TIER=1` set, it monkeypatches
  `make_offload_moe_cache` onto FreeToken's model classes to point at this
  repo's `nvme_offload_cache.py`. It lives in site-packages but is
  effectively project code, not vendor internals.
- `ft_serve_qwen38_background.sh` (port 8000, max throughput) /
  `ft_serve_qwen38_coexist.sh` (port 8000, desktop-friendly) / `ft_serve_glm5.sh`
  (port 1919) are the production launchers. All pass `--disable-moe-prefill-overlap`
  deliberately — the framework's built-in prefill-overlap machinery is
  structurally incompatible with this NVMe-backed cache. Don't remove that flag
  without reading why first (`archive/RESULTS_AND_FINDINGS.md` §25.1, if you
  have the local archive — it is not part of the public repository).
- Vendor source for FreeToken itself:
  `~/.freetoken/venv/lib/python3.12/site-packages/freetoken/`. Read it before
  assuming how the framework behaves rather than reasoning from memory or
  general MoE-serving intuition.

## Documentation

- **`performance.md`** — performance benchmarks on reference hardware, breakdown of token latency, and cross-platform hardware projection models.
- **`README.md`** — public project overview, architecture, quickstart, and serving guides.
- **`archive/RESULTS_AND_FINDINGS.md`** — the historical 42-section development lab notebook and experimental records. Local archive only; excluded from the public repository via `.gitignore`, so treat any reference to it as unavailable to outside readers.

## Operating principles

**Verify before trusting a number, including your own.** This project's
history has repeated instances of a plausible assumption turning out wrong
once measured: a "cold" NVMe read that was secretly a page-cache hit from an
earlier run, an H2D bandwidth number that varied 4x with transfer size, a
peak-bandwidth figure that didn't transfer to a different read shape than
the one it was measured on. When benchmarking disk I/O in this repo, evict
the page cache first (`posix_fadvise(fd, off, len, os.POSIX_FADV_DONTNEED)`)
— the model shards are large enough that a prior server run's page cache
will silently inflate results otherwise. Treat an unmeasured number as wrong
until checked against the real machine or the real vendor source.

**Two A/B gotchas specific to this cache**, both learned the hard way. First,
**freeze `qwen38_routing_freq.pt` between runs** — restore a snapshot and set
`FREETOKEN_COLLECT_ROUTING=0`. It is rewritten on every telemetry flush, so
back-to-back runs otherwise compare *different pinned sets*; its distribution
has already drifted far enough that the historical coverage figures no longer
hold. Second, **generate with `temperature=0`** when comparing cache configs:
at temp>0 each run produces different text, routes to different experts, and
its wall-clock and bytes-read numbers are not comparable. Note also that
changing cache *sizing* legitimately changes GPU slot assignment order, hence
MoE accumulation order, hence bit-level output — so a byte-identical output
check is a valid correctness gate only across configs with identical sizing.

**If you flag something, fix it.** If you're already reading code closely
enough to notice a bug, a dead path, a misleading comment, or a stale doc
claim, fix it in the same pass instead of leaving a note for later — unless
it's genuinely out of scope or needs the user's judgment call. An unfixed
flag just becomes something to re-discover next time.

**Toggleable instrumentation, not permanent overhead, not ad hoc prints.**
`nvme_offload_cache.py`'s pattern: an `os.getenv("FREETOKEN_TRACE_X", "0") ==
"1"` flag cached once in `__init__`, checked before any per-call tracing
work, default off, left in the tree after its immediate question is answered
so it's reusable later (`FREETOKEN_TRACE_SYNC`, `FREETOKEN_TRACE_MATERIALIZE`
are examples). Follow this shape for new instrumentation.

**Confirm the mechanism before building against it.** Several roadmap items
in this project's history were designed or built against a plausible-looking
cause before it was actually confirmed, and had to be redone once measured.
When you see a symptom (a regression, a flat latency curve, a low hit rate),
measure the suspected mechanism directly before designing a fix for it.

**Live-server measurement is the normal test method here.** Starting
`ft_serve_qwen38_background.sh` in the background, sending it real requests, and reading
its logs is how this project validates changes — there's no mock harness.
Stop the server when done (`pkill -f "freetoken.cli serve"`), and delete
one-off diagnostic log files (`server_*_trace.log`) once their findings are
captured in the docs — the numbered doc sections are the durable record, raw
logs are not.

## Key files

| File | Role |
|---|---|
| `nvme_offload_cache.py` | The real, deployed `NvmeOffloadMoeCache` — 3-tier cache logic, all `FREETOKEN_TRACE_*` instrumentation |
| `performance.md` | Comprehensive performance benchmarks and hardware projection model |
| `archive/RESULTS_AND_FINDINGS.md` | Full numbered measurement/design research log (local only, not published) |
| `bench.py`, `bench_read_shape.py`, `bench_agentic_session.py` | Standalone measurement harnesses, each independent |
| `ft_serve_qwen38_background.sh`, `ft_serve_glm5.sh` | Production server launchers |
| `install_hook.sh`, `sitecustomize.py`, `patches/` | FreeToken integration hook and vendor patches |
| `iouring_ffi.py` | ctypes `liburing` wrapper used by `io_uring` backends |
