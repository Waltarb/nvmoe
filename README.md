# NVMoE for llama.cpp & nvmoe-bench

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/CUDA-12%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Linux io_uring](https://img.shields.io/badge/io__uring-enabled-red.svg)](https://kernel.dk/io_uring.pdf)
[![TypeScript](https://img.shields.io/badge/TypeScript-Vitest-blue.svg)](https://vitest.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**NVMoE** is a high-performance, 3-tier hierarchical caching engine for [`llama.cpp`](https://github.com/ggml-org/llama.cpp) that enables multi-hundred-billion parameter Mixture-of-Experts (MoE) LLMs to run on consumer GPUs (e.g., an NVIDIA RTX 3080 Ti Laptop GPU with 16 GB VRAM and 32 GB RAM) at full GPU speeds.

This repository unifies:
1. **The NVMoE Inference Engine (`moe_cache_probe`)**: Combines direct Linux **`io_uring` NVMe streaming**, **pinned Host RAM segmented caching (2Q/SLRU)**, and **GPU VRAM dynamic LRU paging** via a single-callback graph interception (`eval_cb`) with zero host weight math.
2. **`nvmoe-bench` (Hard Core Agentic SWE Benchmark)**: A multi-turn, token-budgeted TypeScript/Node software engineering benchmark suite with visible specs and hidden assertions, designed so frontier models (Gemini Flash, Claude, GPT-4o) do **not** saturate at 100/100.
3. **Real-Time Live Monitor**: An interactive web dashboard (`http://localhost:8085`) and terminal TUI (`./scripts/live_dashboard.py --cli`) showing live token streaming, instantaneous decode speed (tok/s), GPU VRAM footprint, and hardware thermals.

---

## 🏆 Benchmark Scoreboard (Hard Core Agentic SWE Suite)

Evaluated across the 10-task Hard Core SWE Suite (Tasks `06`–`15`: Concurrency, Segmented LRU, WebSocket Frame Codecs, MVCC Snapshot Isolation, Raft Consensus, AST Optimizers, Token Bucket, Schema Validation, Diff Engines, and Typed Event Buses):

```
$ cd bench && pnpm compare
┌─────────┬─────────────────────────┬──────┬──────────┬───────────┬──────────┬────────┬──────────────────┬───────────────┬───────────┬────────────┬───────┬───────────┬──────────┬─────────────────┬────────┐
│ (index) │ config                  │ runs │ score100 │ hiddenPct │ passRate │ solved │ medWallSolvedMin │ testsPer1kOut │ outTokens │ decodeTokS │ ttftS │ formatErr │ applyErr │ budgetOrTimeout │ errors │
├─────────┼─────────────────────────┼──────┼──────────┼───────────┼──────────┼────────┼──────────────────┼───────────────┼───────────┼────────────┼───────┼───────────┼──────────┼─────────────────┼────────┤
│ 0       │ 'qwen38-nvfp4'          │ 10   │ 80       │ '98.4%'   │ '80%'    │ '8/10' │ 9.1              │ 3.51          │ 17097     │ 5.37       │ 210.2 │ 0         │ 0        │ 0               │ 0      │
│ 1       │ 'gemini-3.8-flash-high' │ 10   │ 70       │ '91.8%'   │ '70%'    │ '7/10' │ 0.1              │ 6.48          │ 8640      │ 113        │ 0.4   │ 0         │ 0        │ 0               │ 0      │
│ 2       │ 'qwen38-iq2'            │ 10   │ 40       │ '37.7%'   │ '40%'    │ '4/10' │ 1.6              │ 0.97          │ 23610     │ 30.3       │ 19.7  │ 5         │ 0        │ 5               │ 0      │
│ 3       │ 'glm53-flash-iq2xxs'    │ 5    │ 100      │ '100%'    │ '100%'   │ '5/5'  │ 8.1              │ 20.03         │ 2296      │ 1.76       │ 263.4 │ 0         │ 0        │ 0               │ 0      │
│ 4       │ 'qwen38-flash-nvfp4'    │ 5    │ 100      │ '100%'    │ '100%'   │ '5/5'  │ 3.2              │ 10.36         │ 4439      │ 6.89       │ 62.6  │ 0         │ 0        │ 0               │ 0      │
└─────────┴─────────────────────────┴──────┴──────────┴───────────┴──────────┴────────┴──────────────────┴───────────────┴───────────┴────────────┴───────┴───────────┴──────────┴─────────────────┴────────┘
```

### Side-by-Side Model Comparison:
| Model Configuration | Provider / Runtime | Hard Core Score | Solved Rate | Hidden Assertion Rate | Steady Decode | Prompt TTFT | VRAM Usage |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen 3.8 Flash NVFP4** | **Local NVMoE Engine** | **80.0 / 100** | **8 / 10 (80%)** | **98.4%** (60/61) | **5.37 tok/s** | 210.2s | **12.47 GiB** (< 14 GiB) |
| **Gemini Flash 3.8 (High)** | Google Cloud API | **70.0 / 100** | 7 / 10 (70%) | 91.8% (56/61) | 113.0 tok/s | **0.4s** | Cloud API |
| **Qwen 3.8 Flash IQ2** | **Local NVMoE Engine** | **40.0 / 100** | 4 / 10 (40%) | 37.7% (23/61) | **30.30 tok/s** | **19.7s** | **12.64 GiB** (< 14 GiB) |
| **GLM-5.3-Flash IQ2_XXS** | **Local NVMoE Engine** | *Smoke: 100/100* | 5 / 5 (100%) | 100% (46/46) | 1.76 – 8.9 tok/s | 263.4s | **12.13 GiB** (< 14 GiB) |

> [!NOTE]
> **Key Benchmark Insight**: When evaluated on realistic, unconstrained time budgets (~1.5 hours across 10 tasks), **Qwen 3.8 Flash NVFP4** solves complex multi-file systems tasks that cloud frontier models fail on (e.g. 3-way merge line-drift patching in Task 14 and strictly typed wildcard event buses in Task 15).

---

## 🚀 Supported Models & Download Guide

NVMoE supports both official FP4 quantizations and ultra-compact community 2-bit quantizations:

| Model | Format / Shards | Size on Disk | Architecture | Target Speed | Hugging Face Source |
|---|---|---|---|---|---|
| **Qwen3.8-Flash-Next-NVFP4** | Single GGUF (`qwen3.8-flash-next-nvfp4.gguf`) | **72.5 GiB** | 48L, 512 experts (top-10), 24.5k total | 5.4 – 8.0 tok/s | [FlashML / FreeToken](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| **Qwen3.8-Flash-Next (UD-Q2_K_XL)** | 3 GGUF Shards (`Qwen3.8-Flash-Next-UD-Q2_K_XL-*.gguf`) | **36.2 GiB** | 48L, 512 experts (top-10), 2-bit PLE | **24.0 – 36.3 tok/s** | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| **GLM-5.3-Flash (UD-IQ2_XXS)** | 4 GGUF Shards (`GLM-5.3-Flash-UD-IQ2_XXS-*.gguf`) | **44.1 GiB** | 47L (42 MoE), 288 experts (top-8) | 1.8 – 8.9 tok/s | [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) |
| **Qwen3.6-35B-A3B (Q4_K_M)** | Single GGUF | **22.8 GiB** | 40L, 256 experts (top-8) | **45.0 – 58.4 tok/s** | [Qwen/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-GGUF) |

### How to Download Models

Install the Hugging Face CLI:
```bash
pip install huggingface_hub
```

#### Option 1: Qwen 3.8 Flash Next Unsloth 2-Bit (High Throughput, 30+ tok/s)
```bash
mkdir -p models/qwen-3.8-flash-unsloth/UD-Q2_K_XL
huggingface-cli download unsloth/Qwen3.8-Flash-Next-GGUF \
  --include "UD-Q2_K_XL/*" \
  --local-dir models/qwen-3.8-flash-unsloth/
```
*Repack interleaved blocks for GGML kernels:*
```bash
./repack_unsloth_interleaved models/qwen-3.8-flash-unsloth/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf
```

#### Option 2: GLM-5.3-Flash Unsloth 2-Bit (Direct I/O Multi-Shard)
```bash
mkdir -p models/glm-5.3-flash-iq2xxs/UD-IQ2_XXS
huggingface-cli download unsloth/GLM-5.3-Flash-GGUF \
  --include "UD-IQ2_XXS/*" \
  --local-dir models/glm-5.3-flash-iq2xxs/
```

#### Option 3: Qwen 3.8 Flash Next NVFP4 (NVIDIA FP4 Micro-Scales)
If starting from FreeToken `.ftw` checkpoints:
```bash
python3 scripts/convert_ftw_to_gguf.py \
  --indir /path/to/qwen3.8-flash-next-nvfp4-ftw \
  --outfile models/qwen3.8-flash-next-nvfp4.gguf

./scripts/repack_nvfp4_gguf models/qwen3.8-flash-next-nvfp4.gguf
```

---

## ⚙️ Core Architecture

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 NVMoE Cache Architecture               │
                  └────────────────────────────────────────────────────────┘

     ┌──────────────────────┐        cudaStreamWaitEvent        ┌──────────────────────┐
     │   CUDA Compute       │ ◄──────────────────────────────── │  Pipelined H2D Copy  │
     │   (Main Stream)      │                                   │   (h2d_stream)       │
     └──────────┬───────────┘                                   └──────────▲───────────┘
                │ Intercepts GGML_OP_GET_ROWS                              │
                │ on ffn_moe_weights-%d                                    │
                ▼                                                          │ cudaMemcpyAsync
     ┌──────────────────────────────────────┐                   ┌──────────┴───────────┐
     │        Tier 1: GPU VRAM Cache        │                   │ Tier 2: Pinned Host  │
     │   Reduced Tensor [NVMOE_CACHE_SIZE]  │ ◄─── Cache Miss ─ │       RAM Cache      │
     │   (Top-K Pinned + Dynamic LRU)       │                   │    (Segmented LRU)   │
     └──────────────────────────────────────┘                   └──────────▲───────────┘
                                                                           │
                                                                           │ io_uring Batch
                                                                           │ O_DIRECT (4KB aligned)
                                                                ┌──────────┴───────────┐
                                                                │ Tier 3: NVMe Storage │
                                                                │  Direct GGUF Expert  │
                                                                │       Payloads       │
                                                                └──────────────────────┘
```

1. **1-Callback Unification (`eval_cb`)**: Intercepts `GGML_OP_GET_ROWS` on `ffn_moe_weights-%d` right before CUDA graph execution. It remaps router expert IDs to preallocated GPU cache slots in-place inside `selected_experts` using zero host weight math.
2. **Dynamic Tensor Shrinking (`create_tensor_reduced`)**: Replaces full `[..., n_expert]` tensors with `[..., NVMOE_CACHE_SIZE]` in GPU memory, allowing 120+ GB MoE models to run within 12–14 GB VRAM.
3. **Non-Blocking Asynchronous CUDA Pipelining**: Host-to-Device copies run on a dedicated `h2d_stream`. Hardware synchronization uses `cudaEventRecord` and `cudaStreamWaitEvent(cudaStreamPerThread, st->h2d_event, 0)`, eliminating CPU stalls in the token generation loop.
4. **Linux `io_uring` + `O_DIRECT` Direct I/O**: Direct I/O with 4KB sector alignment completely bypasses the Linux page cache, avoiding memory copies and kernel lock contention at hardware NVMe speeds (3.13+ GB/s).
5. **Segmented LRU (2Q) Host Cache**: Pinned host memory (`cudaHostAlloc`) partitioned into probationary and protected segments, prewarmed at boot using empirical routing frequencies (`freq_qwen38.bin`, `freq_glm53.bin`).
6. **Cost-Aware Dynamic Pruning**: Skips loading low-mass tail experts (`NVMOE_PRUNE_NVME_THRESH=0.08`), drastically cutting NVMe reads during multi-token generation.

---

## 🔒 Hardware Invariants & Safety Constraints

When running on consumer workstations or laptops (e.g. 16 GB RTX 3080 Ti Laptop, 32 GB RAM):

| Resource | Available | Strict Hard Cap | Reason |
|---|---|---|---|
| **GPU VRAM** | 16 GB | **< 14.0 GiB** | Prevents CUDA out-of-memory driver resets. |
| **Host RAM** | 31 GiB DDR5 | **< 20.0 GiB** | Prevents triggering Linux OOM Killer. |
| **Context Length** | Model default: 256k | **Explicit `-c 4096` or `-c 2048`** | Default 256k allocates 30+ GiB KV cache at startup! |
| **CUDA Graphs** | Enabled by default | **`GGML_CUDA_DISABLE_GRAPHS=1`** | Disables CUDA graph capture conflicts with dynamic H2D updates. |

---

## 🛠️ Quickstart: Building & Running

### 1. Prerequisites
- **Linux x86_64** (Kernel >= 5.10 with `io_uring` support)
- **CUDA Toolkit >= 12.0** (`/opt/cuda` or `/usr/local/cuda`)
- **System Packages**:
  ```bash
  # Ubuntu / Debian:
  sudo apt-get install -y build-essential cmake liburing-dev
  # Arch / Manjaro:
  sudo pacman -S base-devel cmake liburing
  ```
- **Node.js >= 20 & pnpm** (for the benchmark suite and live dashboard):
  ```bash
  npm install -g pnpm
  ```

### 2. Automated Build
```bash
./setup.sh
```

### 3. Launching Model Servers (OpenAI-Compatible HTTP)

We provide preconfigured, hardware-safe launch scripts that enforce VRAM (< 14 GiB) and Host RAM (< 20 GiB) invariants:

#### Launch Qwen 3.8 Flash Next NVFP4 (80% Benchmark Winner)
```bash
./scripts/run_nvfp4_server.sh
```

#### Launch Qwen 3.8 Flash Next Unsloth 2-Bit (30+ tok/s Speed Champion)
```bash
./scripts/run_qwen38_iq2_server.sh
```

#### Launch GLM-5.3-Flash Unsloth 2-Bit
```bash
./scripts/run_glm53_server.sh
```

All servers start an OpenAI-compatible HTTP server on `http://localhost:8080` with:
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` (full streaming support via SSE)

---

## 📊 Running `nvmoe-bench` (Agentic Coding Benchmark)

`nvmoe-bench` tests model reasoning on real multi-file TypeScript repositories with compiler checks and hidden test suites.

```bash
cd bench
pnpm install

# Test local NVMoE server on the Hard Core Suite (Tasks 06 to 15):
BENCH_ENDPOINT=http://localhost:8080/v1 \
BENCH_MODEL=local \
pnpm bench --config qwen38-nvfp4 --tier core --noTimeout

# Run only the Smoke Tier (Tasks 01 to 05):
pnpm bench --config my-model --tier smoke

# Benchmark external models (Ollama, vLLM, Groq, DeepSeek):
BENCH_ENDPOINT=http://localhost:11434/v1 \
BENCH_MODEL="qwen2.5-coder:32b" \
pnpm bench --config ollama-qwen32b --tier core

# View aggregated scoreboard:
pnpm compare
```

### Eliminating Artificial Timeouts (`--noTimeout`)
> [!IMPORTANT]
> A timeout in a benchmark measures the impatience of the harness, not the capability of the model!
> Pass `--noTimeout` or set `BENCH_TIMEOUT_SEC=1800` when running slow local models (2–5 tok/s). This allows models to run their internal reasoning and complete full multi-turn solutions over 1–2 hours without premature aborts.

---

## 📈 Real-Time Monitoring Dashboard

Monitor token output, decode speed, and hardware invariants in real time while benchmarks run:

```bash
# Start background monitor daemon:
./scripts/live_dashboard.py
```

- **Web Dashboard**: Open [http://localhost:8085](http://localhost:8085) for an interactive dark-mode dashboard with live streaming output, instantaneous tok/s gauge, and GPU utilization/thermals.
- **Terminal TUI**: Run `./scripts/live_dashboard.py --cli` in any terminal window.
- **Live Stream Log**: Direct token stream is mirrored to `bench/.live_stream.txt`.

---

## 🔧 Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `NVMOE_CACHE_SIZE` | `32` | Number of expert slots allocated per layer in **GPU VRAM**. |
| `NVMOE_GPU_PINNED_EXPERTS` | `0` | Number of top routed experts permanently locked in GPU VRAM (slots `0..K-1`). |
| `NVMOE_HOST_CACHE_SIZE` | `64` | Total expert slots allocated per layer in **Pinned Host RAM**. |
| `NVMOE_PINNED_EXPERTS` | `0` | Number of top routed experts permanently pinned in Host RAM. |
| `NVMOE_FREQ_PATH` | `""` | Path to empirical routing frequency binary (`models/freq_qwen38.bin`, `freq_glm53.bin`). |
| `NVMOE_PRUNE_NVME_THRESH` | `0.0` | Probability mass threshold for dynamic expert pruning. |
| `NVMOE_PRUNE_MIN_KEEP` | `2` | Minimum number of routed experts to evaluate per token. |
| `GGML_CUDA_DISABLE_GRAPHS`| `0` | Must be `1` to avoid CUDA graph capture conflicts with dynamic H2D transfers. |
| `BENCH_TIMEOUT_SEC` | `1800` | Harness wall-clock timeout per task in seconds (set `0` or pass `--noTimeout` to disable). |
| `BENCH_OUTPUT_BUDGET` | `4000` | Maximum completion token budget per task. |

---

## 📂 Repository Structure

```
nvmoe-llamacpp/
├── setup.sh                     # Automated build and compilation script
├── README.md                    # Authoritative unified documentation & benchmark guide
├── AGENTS.md                    # Technical & safety invariants guide for AI developers
├── HANDOVER.md                  # Comprehensive architectural log & historical milestones
├── bench/                       # nvmoe-bench agentic software engineering benchmark
│   ├── tasks/                   # Tasks 01–05 (Smoke) & 06–15 (Hard Core)
│   ├── harness/                 # TypeScript benchmark runner, SEARCH/REPLACE parser, Vitest runner
│   ├── results/                 # Raw benchmark result JSONLs and comparative logs
│   └── HARDCORE_BENCHMARK.md    # Detailed benchmark specifications, scoring formulas & rubrics
├── models/
│   ├── freq_qwen38.bin          # Empirical routing distributions for Qwen3.8 (48L x 512E)
│   └── freq_glm53.bin           # Empirical routing distributions for GLM-5.3 (42L x 288E)
├── scripts/
│   ├── moe_cache_probe.cpp      # Authoritative 3-tier cache inference engine & HTTP server
│   ├── run_nvfp4_server.sh      # Launch script for Qwen 3.8 Flash NVFP4
│   ├── run_qwen38_iq2_server.sh # Launch script for Qwen 3.8 Flash Unsloth 2-Bit
│   ├── run_glm53_server.sh      # Launch script for GLM-5.3-Flash Unsloth 2-Bit
│   ├── run_all_benchmarks.sh    # Autonomous multi-model benchmark runner
│   ├── live_dashboard.py        # Real-time web & terminal monitoring dashboard (port 8085)
│   ├── llama_cpp_nvmoe.patch    # Upstream patch for ggml-org/llama.cpp
│   ├── repack_unsloth_interleaved.cpp # In-place tensor reorganizer for Unsloth GGUFs
│   └── repack_nvfp4_gguf.cpp    # In-place NVFP4 block reorganizer
└── llama.cpp/                   # Git submodule pointing to upstream ggml-org/llama.cpp
```

---

## 📄 License
MIT License. Modifications to `llama.cpp` retain the upstream MIT license.
