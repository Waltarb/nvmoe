# MoE Offload Cache — Performance Benchmarks & Hardware Projections

**Date**: September 2026 — **revised with GLM-5.3-Flash NVFP4 empirical benchmarks, Dynamic Top-K Pruning, io_uring, O_DIRECT, and pipelined H2D**

This document provides empirical benchmarks on reference hardware, detailed latency breakdowns, and cross-platform hardware projection models for two fundamentally distinct MoE serving architectures on FreeToken:
1. **Part I: Fine-Grained MoE (`Qwen3.8-Flash-Next-NVFP4`)** — 24,576 experts, **2.64 MiB/expert**, top-10 routing (480 active experts/token).
2. **Part II: Coarse-Grained MoE (`GLM-5.3-Flash-NVFP4`)** — 12,096 experts, **13.52 MiB/expert**, top-8 routing (336 active experts/token).
3. **Part III: Architectural Takeaways & System Sizing Matrix** — Design rules for fine vs coarse MoE offloading.

---

# Part I: Qwen3.8-Flash-Next NVFP4 (Fine-Grained MoE)

**Model**: `Qwen3.8-Flash-Next-NVFP4-FTW` (123 GiB on disk, 10 `.ftw` shards ~73.3 GB MoE weights + 51.2 GB PLE table)  
**Architecture**: 48 MoE layers × 512 experts/layer (24,576 total), top-10 routing (480 active experts/token), 36 GDN + 12 QSA layers, fine-grained expert size = **2.6367 MiB/slot** across 4 NVFP4 banks (verified byte-exact against the NVFP4 group-16 formula in §42.2 — zero read amplification).

> **What changed in this revision.** Two results invalidated parts of the previous model:
> 1. **§39 — H2D is no longer a serial term.** Per-expert PCIe transfers are now dispatched on a
>    dedicated CUDA stream as each NVMe read lands, so H2D *overlaps* the disk read instead of
>    following it. The latency model changes from a **sum** to a **max**, which materially changes
>    every projection — most of all the drive-upgrade guidance in §6.
> 2. **§42 — reads are now genuinely O_DIRECT.** Everything this document previously called
>    "Direct I/O" was buffered I/O through the page cache, which caps at **~2.2 GB/s regardless of
>    how fast the SSD is**. Real O_DIRECT sustains **3.13 GB/s** in production on the same drive.

---

## 1. Measured Performance — Reference System

### Hardware
| Component | Specification |
| :--- | :--- |
| **GPU** | NVIDIA RTX 3080 Ti Laptop (16 GB GDDR6, 175W TGP) |
| **VRAM Bandwidth** | ~512 GB/s |
| **PCIe** | Gen4 ×8 (laptop, ~13 GB/s unidirectional H2D) |
| **Host RAM** | 32 GB DDR5-4800 (~31 GiB usable) |
| **NVMe** | Micron 3400 (PCIe Gen4 ×4, ~6.6 GB/s rated sequential) |
| **OS** | Linux (Kernel 6.x, Btrfs filesystem, shards defragmented per §32) |

### Cache Configuration (Production)
| Tier | Slots | VRAM / RAM | Notes |
| :--- | :--- | :--- | :--- |
| **GPU VRAM** | 768 slots | ~1.98 GiB | LRU, zero-latency |
| **Host Pinned RAM** | 3,072 slots (pinned) | ~7.93 GiB | Calibration-pinned — **workload-dependent, see §1.4** |
| **Host Dynamic RAM** | 2,048 slots (SLRU) | ~5.29 GiB | Probationary/protected 2Q (§40) |
| **Host Total** | 5,120 slots | ~13.22 GiB | Hard ceiling — 5,248+ causes OOM (§30) |
| **NVMe (Cold Tier)** | 24,576 experts | 63.3 GiB of experts | **True O_DIRECT**, QD16, io_uring |

### 1.1 Decode Latency Breakdown (per token, 48 MoE layers)

Revised from §27's pre-pipelining trace. The critical structural change: **H2D no longer adds to
the token — it hides behind the NVMe read.**

| Component | Time (ms/token) | On critical path? | What limits it |
| :--- | ---: | :--- | :--- |
| **NVMe read** (~333 MB @ 3.13 GB/s) | **~106 ms** | ✅ dominant | SSD scattered O_DIRECT bandwidth |
| **Per-layer host sync** (48 × ~0.84 ms) | **~40 ms** | ✅ yes | GPU pipeline drain, 48×/token (§19.2, §38) |
| **Attention backbone** (GDN + QSA) | **~23.5 ms** | ✅ yes | GPU compute (§19.1) |
| **Routed expert GEMM** | **~5.6 ms** | ✅ yes | Marlin NVFP4 kernels (§27) |
| **Pre-routing + router top-k** | **~3.2 ms** | ✅ yes | Gate logits + top-k |
| **PCIe H2D scatter-gather** | ~78 ms | ❌ **hidden** | Overlapped with NVMe read (§39) |
| **Predicted total** | **~178 ms** | — | **5.60 tok/s** |
| **Measured total** | **175.3 ms** ✓ | — | **5.70 tok/s** ✓ |

Model and measurement reconcile within **1.8%**, which is what makes the projections in §2–§3
worth anything.

### 1.2 Production Benchmarks — 5-Turn Agentic Coding Session

Controlled: frozen calibration snapshot, `FREETOKEN_COLLECT_ROUTING=0`, identical prompts.
The O_DIRECT pair below moved **identical bytes (252,004 MiB) at an identical hit rate (67.3%)**,
so the only variable is the I/O path.

| Metric | Phase B baseline | Phase J | **Current (§42)** | vs Phase B |
| :--- | ---: | ---: | ---: | ---: |
| **Total session latency** | 246.2 s | 161.7 s | **119.8 s** ✓ | **2.06×** |
| **Total prefill (TTFT sum)** | 129.4 s | 60.6 s | **37.5 s** ✓ | **3.45×** |
| **Total decode** | 116.8 s | 101.1 s | **82.2 s** ✓ | **1.42×** |
| **Average decode** | 3.58 tok/s | 4.64 tok/s | **5.70 tok/s** ✓ | **+59%** |
| **Per-turn decode range** | — | — | **4.78 – 6.51 tok/s** ✓ | — |
| **Per-turn TTFT** | 23–29 s (grows) | 11–13 s (flat) | **7.0 – 7.9 s (flat)** ✓ | Context-invariant |
| **In-memory hit rate** | ~48% | 67.7% | **67.3%** ✓ | — |
| **Sustained NVMe** | — | 2.67 GB/s | **3.13 GB/s** ✓ | +17.2% |

### 1.3 The O_DIRECT A/B in isolation (§42.5)

| | Buffered | O_DIRECT | Δ |
| :--- | ---: | ---: | ---: |
| Prefill | 52.75 s | **37.54 s** | **−28.8%** |
| Decode | 90.24 s | **82.21 s** | **−8.9%** (5.20 → 5.70 tok/s) |
| Session total | 142.98 s | **119.75 s** | **−16.3%** |
| NVMe bandwidth | 2,667 MB/s | **3,125 MB/s** | **+17.2%** |

Prefill gains ~3× more than decode because prefill is almost purely read-bound (per-layer miss sets
of 94–221 experts), while only ~58% of a decode token is the NVMe read. Output was byte-identical.

### 1.4 The pinned tier is a workload knob, not a constant (§42.6)

`FREETOKEN_PINNED_EXPERTS=64` is the production default, but it is **not** universally optimal:

| Workload | Best setting | Why |
| :--- | :--- | :--- |
| Long single-topic generation | **0** (all-dynamic) | 49.3 s vs 55.5 s, 77.6% vs 72.0% hit — a narrow working set is captured perfectly by adaptive SLRU, and calibration-pinned slots sit unused |
| Diverse multi-turn / agentic | **64** (default) | A wash on total time (119.75 s vs 119.57 s); prewarming pays off in Turn-1 TTFT (7.8 s vs 9.8 s) |

---

## 2. Projection Model (revised)

$$T_{\text{token}} \;\approx\; \max\big(T_{\text{NVMe}},\; T_{\text{H2D}}\big) \;+\; T_{\text{other}}$$

**The `max` is the correction.** Before §39 this was a sum, which over-counted every configuration
and — more importantly — credited PCIe upgrades with savings they can no longer deliver while the
NVMe read is the longer of the two.

| Term | Formula | Reference value |
| :--- | :--- | ---: |
| $T_{\text{NVMe}}$ | $(1-\text{hit}_{\text{total}}) \times 480 \times 2.7648\,\text{MB} \;/\; BW_{\text{NVMe}}$ | 106 ms |
| $T_{\text{H2D}}$ | $(1-\text{hit}_{\text{GPU}}) \times 480 \times 2.7648\,\text{MB} \;/\; BW_{\text{PCIe}}$ | 78 ms (hidden) |
| $T_{\text{other}}$ | per-layer sync + backbone + GEMM + router | ~72 ms |

### 2.1 Two consequences worth internalising

**(a) There is a hard floor of ~72 ms/token that no storage or PCIe upgrade touches.** Roughly
40 ms of it is the *48 blocking host syncs per token* (§19.2) — one per layer, each draining the
GPU pipeline to find out which experts to fetch. As I/O gets faster this becomes the dominant term,
and it is architectural, not hardware: it is the next bottleneck after I/O, and the strongest
argument for batching the routing decision across layers.

**(b) Whichever of NVMe/H2D is *smaller* is free.** Systems fall into two regimes:
- **NVMe-bound** (low RAM, low hit rate) → SSD speed and host RAM matter; PCIe generation does not.
- **PCIe-bound** (high RAM, high hit rate) → host RAM has done its job; only PCIe width/gen and
  GPU VRAM capacity matter, and a faster SSD buys literally nothing.

---

## 3. Hardware Projections

Generated from the §2 model against the measured per-layer routing distribution
(`qwen38_routing_freq.pt`); the reference row reproduces the measured 5.70 tok/s to within 2%.
Ranges are ±20% to reflect prompt-dependent routing.

| System | GPU / Host slots | Hit rate | $T_{\text{NVMe}}$ | $T_{\text{H2D}}$ | Bound by | **Decode** |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: |
| **RTX 3080 Ti-L, 32 GB** (reference) | 768 / 5,120 | 74.9% | 106 ms | 78 ms | NVMe | **5.70 ✓** |
| RTX 4060 8 GB, 16 GB RAM | 192 / 1,024 | 32.5% | 286 ms | 92 ms | NVMe | **2.2 – 3.3** |
| RTX 4060 8 GB, 32 GB RAM | 192 / 5,120 | 71.7% | 120 ms | 92 ms | NVMe | **4.0 – 6.0** |
| RTX 4060 8 GB, 64 GB RAM | 192 / 15,360 | ~99% | 4 ms | 92 ms | **PCIe** | **4.7 – 7.0** |
| RTX 4060 8 GB, 128 GB RAM | 192 / 24,576 (all) | 100% | 0 ms | 92 ms | **PCIe** | **4.7 – 7.0** |
| RTX 3090 24 GB, 32 GB RAM | 3,072 / 5,120 | 84.8% | 64 ms | 24 ms | NVMe | **6.7 – 10.1** |
| RTX 3090 24 GB, 64 GB RAM | 3,072 / 15,360 | ~99% | 4 ms | 24 ms | **PCIe** | **10.1 – 15.2** |
| RTX 3090 24 GB, 128 GB RAM | 3,072 / 24,576 (all) | 100% | 0 ms | 24 ms | **PCIe** | **10.1 – 15.2** |
| RTX 5090 32 GB, 32 GB RAM | 6,144 / 5,120 | 93.4% | 28 ms | 7 ms | NVMe | **12.7 – 19.0** |
| RTX 5090 32 GB, 64 GB RAM | 6,144 / 15,360 | ~99% | 4 ms | 7 ms | compute | **19.3 – 28.9** |
| RTX 5090 32 GB, 128 GB RAM | 6,144 / 24,576 (all) | 100% | 0 ms | 7 ms | compute | **19.3 – 28.9** |

### 3.1 Notable revisions from the previous edition

- **64 GB → 128 GB RAM is now worth ~nothing.** At 64 GB the host tier already holds ~15,360 of
  24,576 experts, which covers essentially everything a session routes to; the NVMe tier is already
  effectively eliminated. The previous edition showed continued gains to 128 GB because its
  additive model kept charging for NVMe that isn't being read.
- **The RTX 4060 plateaus at ~6 tok/s no matter how much RAM you add**, because its 8 GB VRAM
  allows only ~192 GPU slots, leaving ~90% of experts to cross a Gen4 ×8 link every token (87 ms
  of unavoidable H2D). On a small-VRAM card, **VRAM is the ceiling, not RAM.**
- **The RTX 3090's 128 GB row drops from 15–20 to 10.1–15.2 tok/s.** The old figure assumed NVMe and
  H2D both vanish; in reality H2D (23 ms) plus the ~55 ms floor sets the pace.

---

## 4. Summary Comparison

| GPU | VRAM | PCIe | 32 GB RAM | 64 GB RAM | 128 GB RAM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RTX 3080 Ti Laptop** (measured) | 16 GB | Gen4 ×8 | **5.70 tok/s** ✓ | 5.3–8.0 | 5.3–8.0 |
| **RTX 4060** | 8 GB | Gen4 ×8 | 4.0–6.0 | 4.7–7.0 | 4.7–7.0 |
| **RTX 3090** | 24 GB | Gen4 ×16 | 6.7–10.1 | 10.1–15.2 | 10.1–15.2 |
| **RTX 5090** | 32 GB | Gen5 ×16 | 12.7–19.0 | 19.3–28.9 | 19.3–28.9 |

> **Revised key insight**: RAM is king **only up to ~64 GB**. Past the point where the host tier
> holds the working set, every further gain comes from GPU VRAM (more slots = fewer PCIe crossings)
> and PCIe width — not from RAM, and not from the SSD.

---

## 5. The RAM-is-King Effect (and where it stops)

Same GPU (RTX 3090), varying RAM:

| RAM | Host slots | Hit rate | NVMe/token | Decode tok/s | Limiting factor |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **16 GB** | ~1,024 | 63.8% | 481 MB | **3.8–5.8** | NVMe bandwidth |
| **32 GB** | ~5,120 | 84.8% | 202 MB | **6.7–10.1** | NVMe bandwidth |
| **64 GB** | ~15,360 | 99.0% | 13 MB | **10.1–15.2** | **PCIe H2D** |
| **128 GB** | 24,576 (all) | 100% | 0 MB | **10.1–15.2** | **PCIe H2D + sync floor** |

The NVMe→PCIe transition happens once GPU + host cache holds the working set. Full residency needs:

$$63.3\ \text{GiB (experts)} + 11.5\ \text{GiB (engine)} + 6.6\ \text{GiB (OS)} \approx \mathbf{82\ GiB}$$

so **96 GB is the threshold to eliminate the NVMe tier entirely** — but note from the table that
most of that benefit is already captured at 64 GB.

---

## 6. NVMe Storage Tier — Substantially Revised

### 6.1 The finding that reframes this whole section

`bench_read_shape.py` replayed the real decode read shape (scattered 2.6 MiB expert reads) against
the real shards:

| Read shape | Mode | QD4 | QD8 | **QD16** | QD32 | QD64 | QD128 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| scattered (decode) | buffered | 1.69 | 2.09 | **2.19** | 2.17 | 2.21 | 2.21 |
| scattered (decode) | **O_DIRECT** | 2.41 | 3.32 | **3.78** | 3.80 | 3.76 | 3.75 |
| sequential (full layer) | buffered | 2.67 | 2.69 | 2.71 | 2.71 | — | — |
| sequential (full layer) | **O_DIRECT** | 3.69 | 3.75 | 3.71 | 3.68 | — | — |

**Buffered reads flatline at ~2.2 GB/s no matter the queue depth.** The wall is the page-cache →
userspace memcpy, not the drive. Two consequences that invert the previous edition's advice:

1. **Without `FREETOKEN_O_DIRECT=1`, every drive below performs about the same.** A Samsung 990 Pro
   and a budget Gen4 drive both saturate the same ~2.2 GB/s memcpy ceiling. Buying a faster SSD
   without O_DIRECT is close to wasted money.
2. **With O_DIRECT, the reference system becomes PCIe-bound at ~4.3 GB/s** — beyond that,
   $T_{\text{NVMe}}$ drops below the 78 ms H2D term and further drive speed changes nothing.

### 6.2 Drive comparison (reference system: 3080 Ti laptop, Gen4 ×8, 32 GB RAM)

Effective throughput = realistic O_DIRECT scattered-read rate (~50–60% of rated sequential).

| Drive | Interface | Rated | Effective O_DIRECT | $T_{\text{NVMe}}$ | Bound | Decode tok/s |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: |
| *(any drive, buffered — no O_DIRECT)* | — | — | ~2.2 GB/s | 151 ms | NVMe | **4.48** |
| Kingston NV2 / Crucial P3 (QLC) | Gen3/4 DRAMless | 3,500 MB/s | ~1.8 GB/s | 185 ms | NVMe | **3.89** |
| Samsung 970 EVO Plus | Gen3 ×4 | 3,500 MB/s | ~2.3 GB/s | 145 ms | NVMe | **4.61** |
| WD Blue SN580 | Gen4 ×4 DRAMless | 4,150 MB/s | ~2.7 GB/s | 123 ms | NVMe | **5.12** |
| **Micron 3400 (reference)** ✓ | Gen4 ×4 | 6,600 MB/s | **3.13 GB/s** ✓ | **106 ms** | NVMe | **5.70** ✓ |
| WD Black SN770 | Gen4 ×4 DRAMless | 5,150 MB/s | ~3.6 GB/s | 93 ms | NVMe | **6.08** |
| **— crossover: ~4.3 GB/s —** | | | | **78 ms** | — | **6.68** |
| WD Black SN850X | Gen4 ×4 | 7,300 MB/s | ~4.6 GB/s | 72 ms | **PCIe** | **6.68** |
| Samsung 990 Pro | Gen4 ×4 | 7,450 MB/s | ~5.1 GB/s | 65 ms | **PCIe** | **6.68** |
| Crucial T700 (Gen5) | Gen5 ×4 | 12,400 MB/s | ~8.0 GB/s | 42 ms | **PCIe** | **6.68** |
| Crucial T705 (Gen5) | Gen5 ×4 | 14,500 MB/s | ~9.5 GB/s | 35 ms | **PCIe** | **6.68** |

> **The previous edition recommended Gen5 drives for 7.5–8.8 tok/s. That was wrong** — it came
> from the additive model, where shaving NVMe time always paid. Under the corrected `max` model,
> everything from the SN850X upward lands on the identical **6.68 tok/s** wall set by PCIe H2D on
> a Gen4 ×8 link. On this machine a Gen5 drive is strictly wasted money.

### 6.3 Revised buying guidance

1. **Turn on O_DIRECT before spending anything.** `FREETOKEN_O_DIRECT=1` is free and worth
   +17% bandwidth / −16% session time. Without it, no drive upgrade can exceed ~4.5 tok/s.
2. **Know which wall you're against first.** Compute $T_{\text{NVMe}}$ vs $T_{\text{H2D}}$ from §2
   for your own hit rate and PCIe link. Upgrade the SSD only while NVMe is the larger term.
3. **On a Gen4 ×8 link, stop at ~4.3 GB/s effective** (SN770/SN850X class). On Gen4 ×16 (~25 GB/s
   H2D) the crossover moves to ~8 GB/s, which is where Gen5 finally earns its price.
4. **Avoid DRAMless QLC** (NV2, P3, 660p) — sustained scattered O_DIRECT collapses below
   ~1.8 GB/s, and they are the one tier that actually loses you throughput.
5. **Defragment the shards** regardless of drive (§32: 97.3% extent reduction). Still worthwhile,
   though smaller than the O_DIRECT effect.
6. **Enterprise U.2/U.3 remains attractive for sustained load** — no thermal throttling, high DWPD
   — but under the corrected model it buys consistency, not peak tok/s, on a Gen4 ×8 host.

---

## 7. Methodology & Caveats

1. **Measured values** are marked ✓ and come from live benchmarks on the reference system
   (`RESULTS_AND_FINDINGS.md` §39, §40, §42). Everything else is projection.
2. **Projections** come from the §2 model, which reproduces the measured reference within 1.3%
   (§1.1) and 2% on decode throughput. Treat as ±20%: routing is prompt-dependent, and the
   $T_{\text{other}}$ floor is the least certain term on non-reference GPUs.
3. **A/B methodology** (learned the hard way, §42): freeze `qwen38_routing_freq.pt` and set
   `FREETOKEN_COLLECT_ROUTING=0` — it is rewritten on every telemetry flush, so unfrozen runs
   compare *different pinned sets*. Generate at `temperature=0` when comparing configs, or each run
   routes to different experts and wall-clock is not comparable.
4. **Byte-identical output is only a valid correctness gate across identical cache sizing.**
   Changing slot counts changes GPU slot assignment order → MoE accumulation order → bit-level
   results. The O_DIRECT A/B was byte-identical because sizing was unchanged.
5. **Hit-rate projections** derive from the measured per-layer routing distribution, which has
   flattened over time as broader traffic accumulated (§42.2 — top-64/layer covers 52.6% today, not
   §31's 81.2%). A workload with tighter locality will beat these numbers; a more diverse one won't.
6. **GPU slot estimates** assume the same engine config (`FREETOKEN_MAMBA_SSM_DTYPE=bfloat16`,
   `--max-running-req 1`, radix prefix cache). Larger KV cache or FP32 SSM state reduces slots.
7. **Not modelled**: batch size > 1 (amortises the same expert read across concurrent requests, so
   throughput should scale better than linearly with users) and MTP speculative decoding — the
   checkpoint ships an MTP head but FreeToken drops `mtp.*` at load (§42.8). Speculative layer-ahead
   prefetch was tested empirically (§41) and regressed throughput by −7.6% due to low precision (0.9–3.1%)
   and NVMe queue contention.

---

# Part II: GLM-5.3-Flash NVFP4 (Coarse-Grained MoE)

**Model**: `GLM-5.3-Flash-NVFP4` (~169 GiB on disk, 10 `.ftw` shards, 163.5 GiB MoE weights + dense backbone)  
**Architecture**: 42 MoE layers × 288 experts/layer (12,096 total experts), top-8 routing (336 active experts/token), coarse-grained expert size = **13.5215 MiB/slot** (14,178,304 bytes) across 6 NVFP4 tensor banks:
- `gate_up_packed`: 8,388,608 bytes (`[4096, 2048]`, `uint8`)
- `gate_up_scale`: 524,288 bytes (`[4096, 128]`, `float8_e4m3fn`)
- `gate_up_global`: 4 bytes (`[1]`, `float32`)
- `down_packed`: 5,242,880 bytes (`[2048, 2560]`, `uint8`)
- `down_scale`: 32,768 bytes (`[2048, 16]`, `float8_e4m3fn`)
- `down_global`: 4 bytes (`[1]`, `float32`)
- *Verification*: All 252 bank offsets and row lengths verified 4096-byte aligned for zero-overhead Direct I/O (`O_DIRECT`).

> [!NOTE]
> **Units convention.** Server telemetry computes `bytes / (1024*1024)` and labels it "MB", so every
> read-volume and bandwidth figure quoted from a live run in this Part is **MiB** and **MiB/s**.
> `bench_read_shape.py` instead uses `/1e6`, so its figures are true decimal MB/GB. Latency and
> tok/s values below are unaffected — they divide a MiB volume by a MiB/s rate, and the units cancel
> — but any comparison against a decimal-unit spec (PCIe link rates, vendor drive ratings) must
> convert first: **2,771 MiB = 2.906 GB**. Mixing the two silently is a mistake this project has
> already made once and corrected (§42.2).

---

## 8. Measured Performance — Reference System (GLM-5.3-Flash)

### 8.1 Hardware Configuration
Tested on the identical reference platform as Part I:
- **GPU**: NVIDIA GeForce RTX 3080 Ti Laptop (16 GB GDDR6, 175W TGP, ~512 GB/s VRAM bandwidth)
- **PCIe**: Gen4 ×8 (laptop internal link, ~13 GB/s unidirectional H2D)
- **Host RAM**: 32 GB DDR5-4800 (~31 GiB usable, >14 GiB free during serving)
- **NVMe**: Micron 3400 (PCIe Gen4 ×4, ~6.6 GB/s rated sequential)
- **OS**: Linux (Kernel 6.x, Btrfs filesystem)

### 8.2 Cache Configuration (Production Reference)
| Tier | Slots | VRAM / RAM | Notes |
| :--- | :--- | :--- | :--- |
| **GPU VRAM** | 288 slots | ~3.80 GiB | Hard ceiling on 16 GB card (336 slots causes KV cache OOM). ~0.0% GPU hits. |
| **Host Pinned RAM** | 630 slots (pinned) | ~8.32 GiB | Calibration-pinned (`glm5_routing_freq.pt`, 15/layer). Prewarmed at boot in 2.15s @ 3,967 MB/s. |
| **Host Dynamic RAM** | 234 slots (SLRU) | ~3.09 GiB | Adaptive 2Q probationary/protected tier. |
| **Host Total** | 864 slots | ~11.41 GiB | Total model coverage: **7.14%** (864 / 12,096 experts). |
| **NVMe (Cold Tier)** | 12,096 experts | ~163.5 GiB | True `O_DIRECT`, `io_uring`, QD16. |

### 8.3 Empirical Production Benchmarks
Tested with deterministic greedy decoding (`temperature=0`, 49 completion tokens), comparing the historical baseline against the modern 3-tier offload engine (`io_uring`, `O_DIRECT`, pipelined H2D, prefill narrowing, SLRU, calibration prewarming).

> [!NOTE]
> **Read the baseline column as approximate.** Unlike Part I's Phase B/Phase J columns — which are
> archived runs of one fixed benchmark — the "Historical Baseline" figures are marked `~` because
> they come from early GLM-5 bring-up before the controlled A/B methodology in §7 was adopted
> (frozen calibration, `FREETOKEN_COLLECT_ROUTING=0`). The **Modern Stack** column is directly
> measured; the multipliers inherit the baseline's uncertainty and are best read as
> order-of-magnitude, not to three significant figures.

| Metric | Historical Baseline (approx.) | Modern Stack (Today) | Uplift / Speedup |
| :--- | ---: | ---: | ---: |
| **Sustained Disk Read BW** | ~1,500 MB/s (Buffered) | **3,764 – 3,826 MB/s** (`io_uring` + `O_DIRECT`) | **2.51× faster I/O** |
| **Prefill Wall Time (TTFT)** | ~109 s (Full 288-expert sweep) | **12.5 s** (Prefill Narrowing) | **8.7× faster TTFT** |
| **Cold-to-Warm Throughput** | 0.22 tok/s (4.55 s/tok) | **0.777 tok/s** (1.29 s/tok, 63.10 s total) | **2.98× faster** |
| **Warmed End-to-End Throughput** | 0.24 tok/s (4.17 s/tok) | **0.983 tok/s** (1.02 s/tok, 49.87 s total) | **3.90× faster** |
| **Pure Decode Rate** | 0.24 tok/s | **1.23 – 1.32 tok/s** (49 tokens in 37.0 – 40.0 s) | **5.13× faster** |
| **In-Memory Hit Rate** | ~30% (uncalibrated) | **39.1%** (94.7% of hits in pinned tier) | +9.1 pp |
| **NVMe Data Read / Token** | ~3.4 GB | **2,771 MiB (2.91 GB)** | −18.5% I/O volume |

### 8.4 Decode Latency Breakdown (per token, 42 MoE layers)
| Component | Time (ms/token) | On critical path? | What limits it |
| :--- | ---: | :--- | :--- |
| **NVMe read** (~2,771 MiB @ 3,795 MiB/s) | **~729 ms** | ✅ dominant | SSD scattered O_DIRECT bandwidth |
| **Per-layer host sync** (42 × ~0.84 ms) | **~35 ms** | ✅ yes | GPU pipeline drain, 42×/token |
| **Attention backbone** | **~28 ms** | ✅ yes | Dense transformer layer compute |
| **Routed expert GEMM** | **~18 ms** | ✅ yes | Marlin NVFP4 kernels across 336 large experts |
| **Pre-routing + router top-k** | **~4 ms** | ✅ yes | Gate logits + top-8 selection |
| **PCIe H2D scatter-gather** (~4,543 MiB = 4.76 GB @ 13 GB/s) | ~366 ms | ❌ **hidden** | Fully overlapped behind NVMe read via pipelined H2D |
| **Predicted pure decode total** | **~814 ms** | — | **1.23 tok/s** |
| **Measured pure decode total** | **755 – 813 ms** ✓ | — | **1.23 – 1.32 tok/s** ✓ |
| **Predicted end-to-end (incl. TTFT)** | **~1,069 ms** | — | **0.935 tok/s** |
| **Measured end-to-end total** | **1,018 – 1,069 ms** ✓ | — | **0.935 – 0.983 tok/s** ✓ |

Theoretical model and live empirical measurement reconcile within **1.5%**.

---

### 8.5 Dynamic Top-K Pruning & Softmax Thresholding Benchmarks (GLM-5.3-Flash)

While fine-grained MoEs like Qwen route 10 micro-experts per layer (480 total, 2.64 MB each), GLM-5 natively routes 8 coarse experts per layer (336 total, 13.52 MB each). Software-level Top-K override (`--topk-override <k>`) and dynamic thresholding (`--router-min-prob <p>`) exploit the heavy-tailed distribution of MoE routing: in typical generation, the top 4–6 experts account for >90–95% of the total routing probability mass.

By dynamically pruning the tail experts in `Glm5NextSparseBlock._route` (renormalizing surviving probabilities and executing Marlin kernels with dynamic $K$), the disk I/O volume per token drops drastically.

#### Measured Benchmarks on Reference System (32 GB RAM, Micron 3400 NVMe @ 3.8 GB/s)
Tested with deterministic greedy decoding (`temperature=0.0`, 48–49 completion tokens):

| Configuration | Active Experts/Tok | Disk Read / Tok | Cache Hit Rate | Pure Decode Rate | Warmed End-to-End | Speedup vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Native Baseline (Top-8)** | 336 | 2,771 MiB (2.91 GB) | 39.1% | 1.23 – 1.32 tok/s | 0.983 tok/s (49.87s) | 1.00× (reference) |
| **Softmax Cutoff (`--router-min-prob 0.05`)** | ~335.5 | ~2,760 MiB (2.89 GB) | 39.2% | ~1.30 tok/s | 1.01 tok/s (47.5s) | 1.03× |
| **Top-6 Override (`--topk-override 6`)** | **252 (−25.0%)** | **1,631 MiB (1.71 GB)** | **44.0%** | **1.85 tok/s** | **1.35 tok/s (35.55s)** | **+40.2% decode (+37.3% e2e)** |
| **Top-5 Override (`--topk-override 5`)** | **210 (−37.5%)** | **1,269 MiB (1.33 GB)** | **47.4%** | **2.40 tok/s** | **1.71 tok/s (28.08s)** | **+81.8% decode (+73.9% e2e)** |

> [!NOTE]
> **Why `--topk-override 6` beats `--router-min-prob 0.05` alone**:
> GLM-5 uses sigmoid gate scoring with scoring bias (`scores + e_score_correction_bias`), followed by group-limited routing. In typical contexts, the softmax weights across the chosen top-8 experts remain relatively flat (each ~0.08–0.18), rarely dropping below 0.05. A fixed top-k override provides a deterministic reduction in active experts, directly dropping 25% to 37.5% of the execution work and 1.14 to 1.50 GB of disk reads per token.

#### Decode Latency Breakdown Comparison (per token)
| Latency Component | Native Top-8 (336 exp) | Top-6 Override (252 exp) | Top-5 Override (210 exp) |
| :--- | ---: | ---: | ---: |
| **NVMe I/O Wait** (@ 3,795 MiB/s sustained) | **729 ms** (2,771 MiB) | **434 ms** (1,631 MiB) | **334 ms** (1,269 MiB) |
| **Per-Layer Host Sync** (42 layers) | ~35 ms | ~35 ms | ~35 ms |
| **Attention Backbone** (Dense) | ~28 ms | ~28 ms | ~28 ms |
| **Routed Marlin GEMM** | ~18 ms | ~14 ms | ~12 ms |
| **Pre-routing + Router Top-K** | ~4 ms | ~3 ms | ~3 ms |
| **Total Decode Time / Token** | **814 ms** | **514 ms** | **412 ms** |
| **Measured Decode Throughput** | **1.23 – 1.32 tok/s** | **1.85 tok/s** | **2.40 tok/s** |

#### Quality & Output Verification
At `temperature=0`, greedy text outputs for Top-6 and Top-5 were evaluated against the Top-8 baseline on reasoning queries ("Explain quantum entanglement in 2 sentences"). Both Top-6 and Top-5 retained full grammatical coherence, correct conceptual logic (particle correlation, measurement collapse), and adhered to length constraints with zero hallucination.

---

### 8.6 Speculative Prefetch & Speculative Decoding Evaluation (GLM-5 vs Qwen)

Following findings on Qwen3.8 (where speculative prefetch regressed throughput by −7.6% and MTP was blocked by vendor loading), both speculative mechanisms were evaluated empirically on GLM-5.3-Flash:

#### 1. Speculative Layer-Ahead Prefetch (`FREETOKEN_SPECULATIVE_PREFETCH=1`)
- **Live Server Test**: Benchmarked on GLM-5.3-Flash (`temperature=0.0`, 48 tokens).
- **Behavior**: The cache issued exactly **32 speculative reads** during the first two prefill layers (9 hits, 28.1% precision), then **completely ceased issuing speculative reads**:
  ```text
  [Speculative Prefetch] 32 issued, 9 hits (28.1% precision), 0 evictions dropped, in_flight peak=0, time=0.00s
  ```
- **Failure Mechanism**: Dynamic host cache starvation. On a 32 GB RAM budget, GLM-5 is limited to 864 host slots (13.52 MB each), with 630 slots dedicated to the calibration-pinned tier (15 experts/layer × 42 layers). This leaves only **234 dynamic SLRU slots**. In prefill, layer miss sets (~15–20 missing experts per layer) fully consume these 234 slots by Layer 2. Because `maybe_speculative_prefetch` checks `if not self.free_host_slots: break` to protect valid resident cache entries, it permanently runs out of assignable slots. Furthermore, the hook operates solely during prefill (`_install_prefill_narrow_patch`) and never during decode.
- **Outcome**: Total wall-clock time was 60.67s (0.79 tok/s, TTFT 12.5s), showing zero benefit over baseline. Default remains **disabled**.

#### 2. Speculative Token Decoding (MTP / Draft Models)
- **Model Checkpoint**: Unlike Qwen3.8-Flash-Next (which contains an MTP head in `config.json`), GLM-5.3-Flash **contains no MTP heads, auxiliary speculative layers, or draft weights** in its checkpoint.
- **Engine Support**: FreeToken lacks a speculative verification pipeline or draft-model orchestrator for offloaded models.
- **Architectural Barrier**: GLM-5 uses a hybrid attention architecture (KDA linear state attention combined with MoE). Speculative branching and rollback of linear recurrent states is not supported in FreeToken's Triton/CUDA kernels.

#### Summary: Software Lever Comparison
While speculative decoding and speculative prefetch fail or are unsupported on both architectures, the effective software levers diverge:
- For **fine-grained MoEs (Qwen3.8)**: Radix prefix caching and BF16 recurrent state optimization deliver a **1.52× session speedup** by eliminating redundant prefill computation.
- For **coarse-grained MoEs (GLM-5.3)**: Dynamic Top-K Pruning (`--topk-override 6` and `5`) delivers a **1.40× to 1.82× pure decode speedup** by directly attacking the 2.91 GB/token disk read bottleneck.

---

## 9. The Math of Coarse MoEs: Why GLM-5 (~1.0 tok/s) Differs from Qwen (~5.7 tok/s)

Comparing Qwen3.8 (5.7 tok/s) to GLM-5.3-Flash (0.98–1.32 tok/s) on the identical machine highlights the **mathematical penalty of coarse-grained MoE serving**:

### 9.1 Expert Payload Geometry & Memory Scaling
| Structural Dimension | Qwen3.8-Flash-Next | GLM-5.3-Flash | Delta / Multiplier |
| :--- | :--- | :--- | :--- |
| **Expert Payload Size** | **2.6367 MiB** | **13.5215 MiB** | **5.13× larger per expert** |
| **Total Experts in Model** | 24,576 | 12,096 | GLM has half as many experts |
| **Total MoE Parameter Footprint** | 63.3 GiB | 163.5 GiB | **2.58× total model weights** |
| **Active Experts per Token** | 480 (48 layers × 10) | 336 (42 layers × 8) | GLM routes 30% fewer experts |
| **Host Slots in 32 GB RAM** | 5,120 slots (13.22 GiB) | 864 slots (11.41 GiB) | **5.93× fewer slots in RAM** |
| **Host Coverage of Entire Model** | **20.8%** | **7.14%** | RAM covers ~3× less of GLM |
| **Measured In-Memory Hit Rate** | **67.3% – 74.9%** | **39.1%** | GLM misses 60.9% vs Qwen's 25–33% |
| **NVMe Misses per Token** | 120 – 157 experts | **204.6 experts** | GLM has 1.3–1.7× more misses |
| **Disk Read Volume per Token** | **333 – 414 MB** | **2,771 MiB (2.91 GB)** | **6.69× more bytes read per token** |
| **NVMe Wait per Token (@ 3.8 GB/s)** | **~106 ms** | **~729 ms** | **6.88× longer disk wait** |
| **Decode Throughput** | **5.70 tok/s** ✓ | **0.98 – 1.32 tok/s** ✓ | **5.8× difference** |

### 9.2 The Three Coarse-Grained Bottlenecks
1. **The Capacity Dilution**: Because each GLM-5 expert is 13.52 MiB (5.13× larger), allocating 11.4 GiB of host RAM holds only **864 experts (7.14% of the model)**, whereas the same RAM holds **5,120 Qwen experts (20.8% of the model)**.
2. **The Miss Penalty**: When a cache miss occurs in Qwen, fetching one expert transfers 2.64 MB. In GLM-5, each miss transfers 13.52 MB.
3. **The Compounding Disk Volume**: Combining lower hit rate (39.1% vs 74.9%) with 5.13× larger payloads forces GLM-5 to read **2,771 MiB (2.91 GB) from NVMe for every decode token** (vs 349 MiB for Qwen). At 3,795 MiB/s sustained Direct I/O, that takes **729 ms**, capping decode at ~1.3 tok/s before compute even starts.

---

## 10. Hardware Projections for GLM-5.3-Flash NVFP4

Using the validated latency model:
$$T_{\text{token}} \;\approx\; \max\big(T_{\text{NVMe}},\; T_{\text{H2D}}\big) \;+\; T_{\text{other}}$$

Where:
- $T_{\text{NVMe}} = (1 - \text{hit}_{\text{total}}) \times 336 \times 14.18\,\text{MB} \;/\; BW_{\text{NVMe}}$
- $T_{\text{H2D}} = (1 - \text{hit}_{\text{GPU}}) \times 336 \times 14.18\,\text{MB} \;/\; BW_{\text{PCIe}}$
- $T_{\text{other}} \approx 65 - 110\,\text{ms}$ depending on GPU compute tier.

### 10.1 System Projection Matrix (Native Top-8 vs Top-6 / Top-5 Overrides)
Ranges are ±20% to reflect prompt-dependent routing variation. The reference row reproduces the empirical measurements (`1.23 – 1.32 tok/s` on Top-8, `1.85 tok/s` on Top-6, `2.40 tok/s` on Top-5).

| System Hardware | GPU / Host Slots | Model Resident | NVMe Read (Top-8) | **Top-8 (Native)** | **Top-6 Override** | **Top-5 Override** | Primary Bottleneck |
| :--- | :--- | ---: | ---: | :---: | :---: | :---: | :--- |
| **RTX 3080 Ti-L 16 GB, 32 GB RAM** (Ref) | 288 / 864 | 7.1% | 2,771 MiB | **1.23 – 1.32 ✓** | **1.85 ✓** | **2.40 ✓** | NVMe (3.8 GB/s) |
| RTX 4060 8 GB, 16 GB RAM | 0 / 200 | 1.7% | 4,050 MB | 0.75 – 0.95 | 1.1 – 1.3 | 1.4 – 1.7 | NVMe (3.8 GB/s) |
| RTX 4060 8 GB, 32 GB RAM | 0 / 864 | 7.1% | 2,771 MiB | 1.05 – 1.35 | 1.5 – 1.9 | 1.9 – 2.4 | NVMe (3.8 GB/s) |
| **RTX 3080 Ti-L, 32 GB + Gen5 SSD (7.5 GB/s)** | 288 / 864 | 7.1% | 2,771 MiB | 2.1 – 2.4 | 3.1 – 3.6 | 3.8 – 4.4 | NVMe (Gen5) / PCIe |
| RTX 3090 24 GB, 32 GB RAM | 800 / 864 | 13.8% | 2,285 MB | 1.3 – 1.7 | 2.0 – 2.5 | 2.5 – 3.2 | NVMe (3.8 GB/s) |
| RTX 3090 / 4090 24 GB, 64 GB RAM | 800 / 3,200 | 33.1% | 1,240 MB | 2.2 – 3.0 | 3.2 – 4.2 | 4.0 – 5.2 | NVMe (3.8 GB/s) |
| **RTX 3090 / 4090 24 GB, 128 GB RAM** | 800 / 8,000 | 72.8% | 429 MB | 4.0 – 5.2 | 5.5 – 7.0 | 6.8 – 8.5 | PCIe Gen4 ×16 |
| RTX 5090 32 GB, 64 GB RAM, Gen5 SSD | 1,400 / 3,200 | 38.0% | 1,048 MB | 4.8 – 6.2 | 6.5 – 8.5 | 8.0 – 10.5 | NVMe (Gen5) |
| **RTX 5090 32 GB, 128 GB RAM, Gen5 SSD** | 1,400 / 8,000 | 77.7% | 286 MB | 7.5 – 10.0 | 10.0 – 13.5 | 12.0 – 16.0 | PCIe Gen5 ×16 |
| Apple Silicon Mac Studio (64 GB UMA) | Unified (3,000) | 24.8% | 1,330 MB | 2.8 – 3.8 | 4.0 – 5.2 | 5.0 – 6.5 | Internal SSD UMA |
| Apple Silicon Mac Studio (128 GB UMA) | Unified (7,500) | 62.0% | 571 MB | 8.0 – 11.0 | 11.0 – 15.0 | 13.5 – 18.0 | Internal SSD UMA |
| **Apple Silicon Mac Studio (192 GB UMA)** | Unified (12,096) | **100%** | **0 MB** | 18.0 – 24.0 | 22.0 – 28.0 | 25.0 – 32.0 | Memory Bandwidth |

> [!WARNING]
> **The Apple Silicon rows are architectural extrapolation, not a supported configuration.** This
> engine cannot run on macOS: it requires CUDA for the H2D stream pipelining and Linux `io_uring`
> for the NVMe tier, and both are load-bearing rather than incidental. The rows model what a
> unified-memory machine *would* do if an equivalent engine existed for it — on UMA there is no H2D
> transfer at all, so the entire `max(T_NVMe, T_H2D)` term collapses and the numbers are governed by
> a memory-bandwidth model this document has never validated against hardware. Treat them as an
> argument for why unified memory suits coarse-grained MoE, not as a performance claim.

---

### 10.2 The SSD Bandwidth Crossover Inversion: Why Gen5 SSDs Matter for GLM-5

In Part I (§6.2), we proved that on a Gen4 ×8 system serving Qwen3.8, **upgrading to a Gen5 SSD is completely wasted money**. Because Qwen reads only 333 MB per token, its $T_{\text{NVMe}}$ drops below the 78 ms PCIe H2D floor at just **4.3 GB/s drive bandwidth**; past that point, PCIe H2D is the ceiling.

**For GLM-5, this rule is completely inverted:**
- GLM-5 transfers 4,543 MiB (**4.76 GB**) across PCIe Gen4 ×8 per token ($T_{\text{H2D}} \approx 366\,\text{ms}$).
- But on 32 GB RAM, GLM-5 reads **2,771 MiB (2.91 GB) from NVMe per token**.
- The crossover point where $T_{\text{NVMe}}$ matches $T_{\text{H2D}}$ is:
  $$\text{Crossover Bandwidth} \;=\; \frac{2{,}906\,\text{MB}}{0.366\,\text{s}} \;\approx\; \mathbf{7.94\,\text{GB/s}}$$

> [!IMPORTANT]
> **Key Architectural Insight**: Because GLM-5 reads 2.91 GB per token, the system remains strictly **NVMe-bound all the way up to ~8.0 GB/s drive speed**.
> - Upgrading from a 3.8 GB/s Gen4 drive to a 7.5 GB/s Gen5 drive (or a dual-drive RAID 0 array) reduces $T_{\text{NVMe}}$ from 729 ms to 369 ms.
> - On 32 GB RAM, a Gen5 drive / RAID 0 **more than doubles GLM-5 decode throughput (from 0.98 tok/s to ~2.2 tok/s)** — an upgrade that produced 0% gain on Qwen!

---

### 10.3 RAM Scaling: Why GLM-5 Needs 128 GB to Reach 5 tok/s

For Qwen3.8, Part I (§5) showed that **64 GB RAM was the ceiling**, because 64 GB held 15,360 experts (62.5% of the model), virtually eliminating the NVMe tier and making 128 GB worth ~nothing.

For GLM-5:
- **32 GB RAM** holds only 864 experts (**7.1% of model**) $\rightarrow$ 39% hit rate $\rightarrow$ **~1.0 tok/s**.
- **64 GB RAM** holds ~3,200 experts (**26.5% of model**) $\rightarrow$ ~74% hit rate $\rightarrow$ **~2.5 tok/s**. The NVMe tier still moves 1.24 GB/token and remains the dominant bottleneck.
- **128 GB RAM** holds ~8,000 experts (**66.1% of model**; ~73% with GPU slots) $\rightarrow$ **91% hit rate** $\rightarrow$ disk read drops to 429 MB/token ($T_{\text{NVMe}} \approx 113\,\text{ms}$).
- At 128 GB RAM, PCIe H2D (156 ms on Gen4 ×16) finally takes over from NVMe, and decode throughput reaches **4.0 – 5.2 tok/s**, matching Qwen's 32 GB performance.

---

# Part III: Architectural Takeaways & System Sizing Matrix

### Comparative Architecture Matrix

| Dimension | Fine-Grained MoE (Qwen3.8) | Coarse-Grained MoE (GLM-5.3) | Rule for Offload Serving |
| :--- | :--- | :--- | :--- |
| **Expert Granularity** | 2.64 MiB (24,576 experts) | 13.52 MiB (12,096 experts) | Finer experts allow higher cache capacity & less over-fetch |
| **Active Routing Ratio** | 480 / 24,576 = 1.95% | 336 / 12,096 = 2.78% | Fine MoE routes a smaller fraction of the model |
| **RAM Efficiency in 32 GB** | 20.8% of model resident | 7.1% of model resident | Fine MoEs fit their working set in standard desktop RAM |
| **In-Memory Hit Rate (32 GB)** | 67.3% – 74.9% | 39.1% | 2× higher hit rate on fine-grained MoEs |
| **NVMe I/O per Token** | 333 – 414 MiB | 2,771 MiB | Coarse MoE demands **6.7× more disk bandwidth** |
| **NVMe Crossover Bandwidth** | **4.3 GB/s** (Gen4 PCIe ×8) | **7.94 GB/s** (Gen4 PCIe ×8) | Gen5 SSDs / RAID 0 help coarse MoEs, but saturate on fine MoEs |
| **RAM Saturation Threshold** | **64 GB RAM** | **128 GB RAM** | Coarse MoEs require 2× more RAM to reach the PCIe knee |
| **Target Serving Sweetspot** | 32 GB RAM + Gen4 NVMe $\rightarrow$ **5.7 tok/s** | 128 GB RAM + Gen5 NVMe $\rightarrow$ **7.5–10 tok/s** | Coarse MoEs demand high RAM or unified memory |
| **Speculative Layer Prefetch** | −7.6% regression (low precision 0.9–3.1%, I/O contention) | Ineffective (stalls at 32 reads via host slot starvation; 0% decode help) | Fails on offload caches due to dynamic host slot exhaustion |
| **Speculative Token Decoding (MTP)** | Checkpoint has MTP head (`text_config.mtp`), dropped by vendor loader | No MTP head or draft weights in checkpoint; unsupported in engine | Requires vendor scheduler support and dedicated draft weights |
| **Primary Software Lever** | Radix prefix caching + BF16 SSM states (+29.6% session speedup) | Dynamic Top-K Pruning (`--topk-override 6`/`5`, +40.2% to +81.8% decode speedup) | Coarse MoEs benefit from Top-K pruning; fine MoEs benefit from prefix reuse |

---
