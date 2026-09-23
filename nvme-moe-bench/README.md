# nvmoe: 3-Tier NVMe MoE Expert Offload Engine for FreeToken

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Linux](https://img.shields.io/badge/OS-Linux-orange.svg)](https://kernel.org)
[![io_uring](https://img.shields.io/badge/io__uring-enabled-green.svg)](https://github.com/axboe/liburing)

A high-performance **3-tier (GPU VRAM / Host RAM / NVMe)** Mixture-of-Experts (MoE) offload engine for [FreeToken](https://github.com/FlashML-org/FreeToken-Web).

Run MoE models whose weights are several times larger than your VRAM *and* RAM combined — such as **Qwen3.8-Flash-Next** (24,576 experts, **123 GiB on disk**) and **GLM-5.3-Flash** — on consumer hardware (e.g. a laptop RTX 3080 Ti with 16 GB VRAM and 32 GB RAM) by paging expert weights off NVMe on demand, with true Direct I/O, asynchronous PCIe pipelining, and zero-copy `io_uring` batching.

---

## Headline Results (Live Measurement on Reference Hardware)

Tested on **NVIDIA RTX 3080 Ti Laptop (16 GB VRAM, PCIe Gen4 x8)**, **32 GB DDR5 RAM**, **Micron 3400 PCIe Gen4 NVMe**, serving **Qwen3.8-Flash-Next-NVFP4-FTW** (123 GiB on disk, 48 layers × 512 experts = 24,576 total experts, top-10 routing):

| Metric | Phase B Baseline (Buffered/Sync) | Current Engine (O_DIRECT + Pipelining) | Improvement |
| :--- | :---: | :---: | :---: |
| **Sustained Decode Throughput** | 3.58 tok/s | **5.70 tok/s** (up to 6.51 tok/s) | **+59% faster** |
| **5-Turn Agentic Session Time** | 246.2 s | **119.8 s** | **2.06× speedup** (-2.1 min) |
| **Total Prefill Time (TTFT sum)** | 129.4 s | **37.5 s** | **3.45× speedup** |
| **Turn-by-Turn TTFT** | 23–29 s (grows with context) | **7.0–7.9 s (flat)** | **Context-invariant** |
| **Sustained NVMe Bandwidth** | 2.67 GB/s (buffered, Phase J) | **3.13 GB/s** (O_DIRECT QD16) | **+17% bandwidth** |
| **In-Memory Cache Hit Rate** | ~48% | **67.3%** | +19.3 percentage points |
| **Desktop Headroom** | Unpredictable (page cache churn) | **>10.2 GiB Free RAM** | Zero UI stutter / swap |

*The bandwidth row compares against Phase J (the most recent buffered-I/O configuration) rather than Phase B, for which no sustained-bandwidth figure was recorded. All other rows are Phase B.*

*Full measurement data, per-kernel traces, and hardware projection models across RTX 4060, 3090, and 5090 are available in [`performance.md`](performance.md).*

---

## Architectural Highlights

```
  ┌─────────────────────────────────────────────────────────────┐
  │                       GPU VRAM Tier                         │
  │      768 slots (~1.98 GiB) • Zero-latency active cache      │
  └──────────────────────────────▲──────────────────────────────┘
                                 │
              Asynchronous PCIe  │  Dedicated CUDA Stream
               H2D Pipelining    │  (Overlapped with NVMe)
                                 │
  ┌──────────────────────────────┴──────────────────────────────┐
  │                    Host Pinned RAM Tier                     │
  │  3,072 Calibration-Pinned Slots  +  2,048 Dynamic SLRU (2Q) │
  │        (~7.93 GiB)                        (~5.29 GiB)       │
  └──────────────────────────────▲──────────────────────────────┘
                                 │
              True Direct I/O    │  Zero-Copy liburing Batching
              O_DIRECT QD16      │  Single-Syscall SQE Submission
                                 │
  ┌──────────────────────────────┴──────────────────────────────┐
  │                       NVMe Cold Tier                        │
  │     24,576 experts (63.3 GiB MoE + 51.2 GiB PLE Table)      │
  │          Micron 3400 PCIe Gen4 SSD • Defragmented           │
  └─────────────────────────────────────────────────────────────┘
```

1. **True `O_DIRECT` Direct I/O**:
   Bypasses the OS page cache completely for per-expert weight reads, eliminating expensive memory copies and avoiding kernel lock contention. In isolation ([`bench_read_shape.py`](bench_read_shape.py)) scattered expert reads go from a **2.19 GB/s buffered ceiling — flat at every queue depth, because the wall is the page-cache memcpy, not the drive — to 3.78 GB/s**; in production the engine sustains **3.13 GB/s**.
2. **Zero-Copy `io_uring` Kernel Batching**:
   Replaces multi-threaded pool contention with a single-threaded kernel submission ring using a custom ctypes FFI wrapper ([`iouring_ffi.py`](iouring_ffi.py)). Dispatches batch expert requests directly into pinned memory without thread wakeups.
3. **Asynchronous PCIe H2D Stream Pipelining**:
   As each individual expert read completes from NVMe, its Host-to-Device PCIe transfer is launched immediately on a dedicated CUDA stream. The ~78 ms/token PCIe transfer is completely overlapped behind the NVMe disk read rather than executed serially.
4. **Segmented LRU (SLRU / 2Q) Host Cache**:
   Partitions dynamic host RAM into probationary and protected segments, preventing one-off transient token misses from evicting established working-set experts. Premature eviction of experts that had already earned 2+ hits falls from 7.2% to 2.4% of evictions; net effect is a modest **+1.0 pp hit rate / −2.7% NVMe reads**.
5. **Prefill Narrowing to Routed Unions**:
   Prefill scans only the union of experts actually routed by prompt tokens (typically 44–71% of the layer pool) rather than the entire 512-expert layer, reducing prefill disk I/O by 1.63×.
6. **Calibration-Informed Cache Pinning & Prewarming**:
   Pre-populates the hottest ~64 experts per layer into pinned RAM during a 4-second boot sweep ([`qwen38_routing_freq.pt`](qwen38_routing_freq.pt)), providing zero-wait hits for >52% of activations.

---

## Quickstart & Installation

### Prerequisites
- **Linux** (Kernel 5.10+ recommended with `io_uring` support)
- **`liburing`**: Ensure `liburing-ffi.so.2` is installed:
  ```bash
  # Arch / Manjaro:
  sudo pacman -S liburing
  # Ubuntu / Debian:
  sudo apt-get install liburing-dev
  ```
- **Python 3.10+** with CUDA-enabled PyTorch
- **FreeToken** installed in your environment
- **Model weights**, downloaded separately — `Qwen3.8-Flash-Next-NVFP4-FTW` is **123 GiB on disk**
  (10 `.ftw` shards). Point `MODEL_PATH` at the directory containing them; the launchers default to
  `~/.freetoken/models/qwen3.8-flash-next-nvfp4-ftw`.
- **NVMe storage** with free space for the weights. A SATA SSD or hard disk will not keep up — the
  engine's throughput is a direct function of scattered-read bandwidth (see
  [`performance.md`](performance.md) §6).

### 1. Install Hook and Patches
Run the automated installer to link the integration hook into your active Python / FreeToken environment and apply the required Qwen3.8 / PLE disk patches:

```bash
git clone https://github.com/waltarb/nvmoe.git
cd nvmoe
./install_hook.sh
```

*(Alternatively, you can manually apply [`patches/freetoken-0.1.2.patch`](patches/freetoken-0.1.2.patch) to your `site-packages/freetoken` directory and copy [`sitecustomize.py`](sitecustomize.py) into your `site-packages`.)*

The installed hook needs to locate `nvme_offload_cache.py` at runtime. It finds it automatically
when the server is launched from the repository directory; from anywhere else, point it at the
checkout:

```bash
export NVMOE_DIR=/path/to/nvmoe
```

If the hook can't import the cache it prints a single `Warning: Failed to install
FREETOKEN_NVME_TIER hook` line to stderr and FreeToken then starts **without** the NVMe tier —
worth checking for in the startup log, since the server otherwise appears to run normally.

### 2. Launch the Model Server

Launch FreeToken using one of the production profiles:

```bash
# Background Profile (Max throughput: 5,120 host slots, 768 GPU slots, Radix cache):
./ft_serve_qwen38_background.sh

# Coexist Profile (Desktop friendly: caps host cache to 3,072 slots, leaving ~6 GiB free RAM):
./ft_serve_qwen38_coexist.sh

# GLM-5.3-Flash Profile:
./ft_serve_glm5.sh
```

All launchers accept standard FreeToken / OpenAI API arguments (e.g. `--port 8000`).

---

## Configuration & Environment Variables

`Default` is the value the **code** falls back to when the variable is unset. Several performance
features default to off and are switched on by the production launchers — the `Launcher` column
shows what `ft_serve_qwen38_background.sh` sets. If you invoke `freetoken.cli serve` yourself
rather than through a launcher, you get the `Default` column.

| Variable | Default | Launcher | Description |
| :--- | :---: | :---: | :--- |
| `FREETOKEN_NVME_TIER` | `0` | `1` | Set to `1` to activate the 3-tier NVMe cache. |
| `FREETOKEN_IO_BACKEND` | `iouring` | `iouring` | I/O engine: `iouring` (recommended) or `threadpool`. |
| `FREETOKEN_O_DIRECT` | `0` | `1` | Enable true `O_DIRECT` reads, bypassing Linux page cache. Falls back to buffered automatically if alignment requirements aren't met. |
| `FREETOKEN_PIPELINE_H2D`| `0` | `1` | Asynchronously pipeline H2D PCIe copies with NVMe reads. |
| `FREETOKEN_SLRU_HOST_TIER`| `0` | `1` | Use 2Q / Segmented LRU for dynamic host RAM cache. |
| `FREETOKEN_HOST_CACHE_SIZE`| *auto* | `5120` | Total slots in host pinned RAM (~2.64 MiB / slot). |
| `FREETOKEN_PINNED_EXPERTS` | *auto* | `64` | Experts per layer permanently pinned from calibration. Workload-dependent — see [`performance.md`](performance.md) §1.4. |
| `FREETOKEN_PREWARM_CACHE` | `0` | `1` | Preload pinned experts at startup (takes ~4s). |
| `FREETOKEN_PREFILL_NARROW`| `1` | `1` | Narrow prefill reads to the routed expert union. |
| `FREETOKEN_IO_WORKERS` | `16` | `16` | Thread pool queue depth / io_uring queue capacity. |
| `FREETOKEN_COLLECT_ROUTING` | `1` | `1` | Accumulate live routing frequencies into `qwen38_routing_freq.pt`. **Rewrites the shipped calibration file in place** — set to `0` to keep it pristine, and always set it to `0` when A/B testing cache configs. |
| `FREETOKEN_SPECULATIVE_PREFETCH` | `0` | — | Calibration-seeded layer-ahead prefetch. Implemented and correctness-verified, but measured ineffective; ships off. |

---

## Benchmarks & Evaluation Tools

The repository contains standalone benchmarking suites that measure disk I/O, cache replacement, and live server performance without mock data:

- **`bench_agentic_session.py`**: Simulates a 5-turn tool-using coding conversation against the live server, measuring turn-by-turn TTFT and decode throughput.
- **`bench_read_shape.py`**: Benchmarks buffered vs. `O_DIRECT` throughput and queue-depth scaling directly against real `.ftw` model shards.
- **`bench.py`**: Standalone 3-tier cache simulator evaluating prefetch strategies and Zipf-skewed routing traces against real disk bytes.
- **`bench_controlled.py`**: Temperature=0 deterministic single-prompt decode benchmark.
- **`bench_qwen38_stream.py`**: Real-time streaming latency and tokens/sec measurement.

Example running the standalone read-shape benchmark (sweeps buffered vs. `O_DIRECT` at the given
queue depths, evicting the page cache before every measurement):
```bash
python3 bench_read_shape.py --qd 4,8,16,32 --repeats 5
```

Raw benchmark JSON is deliberately not committed — it is specific to one drive, one GPU and one
cache configuration, and doesn't transfer. [`performance.md`](performance.md) is the durable record.

### Sample output

[`generated_website/index.html`](generated_website/) is a self-contained dashboard written by the
model *while running on this engine* — a qualitative check alongside the throughput numbers, showing
coherent long-form generation on a machine holding roughly a quarter of the model's weights in
memory. Offloading changes when weights arrive, not what the model computes.

---

## Documentation Index

- **[`performance.md`](performance.md)**: Executive performance summary, token latency breakdowns, and hardware scaling models (PCIe width, NVMe generations, RAM scaling).
- **[`AGENTS.md`](AGENTS.md)**: Architecture invariants, environment setup, and operating guidelines for AI coding assistants.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
