# NVMoE for llama.cpp

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/CUDA-12%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Linux io_uring](https://img.shields.io/badge/io__uring-enabled-red.svg)](https://kernel.dk/io_uring.pdf)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**NVMoE for llama.cpp** is a high-performance, 3-tier hierarchical caching engine that enables multi-hundred-billion parameter Mixture-of-Experts (MoE) LLMs to run on consumer GPUs (such as an NVIDIA RTX 3080 Ti Laptop GPU with 16 GB VRAM) at full GPU speeds.

By combining direct Linux **`io_uring` NVMe streaming**, **pinned Host RAM segmented caching**, and **GPU VRAM dynamic LRU paging**, NVMoE achieves state-of-the-art inference speeds while respecting strict hardware constraints (Host RAM < 20 GiB, VRAM < 14 GiB).

---

## Performance Milestones

### 1. Qwen3.8-Flash-Next (NVFP4, 48 Layers, 512 Experts)
Evaluated on **NVIDIA RTX 3080 Ti Laptop GPU (16 GB VRAM)**, PCIe Gen4 NVMe SSD, Host RAM 31 GiB DDR5:

| Configuration | Decode Throughput | In-Memory Hit Rate | GPU VRAM Hit Rate | Peak VRAM |
|---|---|---|---|---|
| **FreeToken Baseline** | 4.64 – 5.70 tok/s | ~80% | N/A | ~13.5 GiB |
| **NVMoE Cold Start (16 GPU slots)** | 1.36 tok/s | 32.1% | 19.4% | ~9.8 GiB |
| **NVMoE Optimized (48 GPU slots / 128 Host slots)** | **5.04 tok/s** | **82.6%** | **59.2%** | **~12.6 GiB** |

*Verified coherent, fluent output on factual and reasoning tasks at `temperature=0`.*

### 2. Qwen3.6-35B-A3B (Q4_K_M, 40 Layers, 256 Experts)
| Stage | Optimization Technique | Sustained Decode | Speedup vs Baseline |
|---|---|---|---|
| **Baseline** | Synchronous `pread` per miss (O_DIRECT) | 1.14 tok/s | 1.0x |
| **Phase 6** | `io_uring` + Pinned Host RAM + Async CUDA | 12.70 tok/s | 11.1x |
| **Phase 7** | **1-Callback Unification** (Zero host weight math) | 14.68 tok/s | 12.9x |
| **Phase 7+** | Host Prewarming Pinning (192 experts/layer) | 32.02 tok/s | 28.1x |
| **Phase 8** | **GPU Tier LRU + Direct VRAM Boot Prewarming** (`CACHE_SIZE=96`) | **56.13 tok/s** | **49.2x** |
| **Phase 8 Peak** | **GPU Tier LRU + Direct VRAM Boot Prewarming** (`CACHE_SIZE=128`) | **58.36 tok/s** | **51.2x** |
| **Code Benchmark** | 128-token Python coding benchmark | **45.79 tok/s** | **40.1x** |

---

## Core Architecture

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

1. **Single-Callback Graph Interception (1-Callback Unification)**:
   Instead of trapping each individual GEMM or evaluating expert weights on CPU, NVMoE intercepts `GGML_OP_GET_ROWS` on `ffn_moe_weights-%d` right before CUDA graph execution. It maps incoming router expert IDs to preallocated GPU cache slots in-place inside `selected_experts` using zero host weight math.
2. **Dynamic Tensor Shrinking (`create_tensor_reduced`)**:
   Overrides `llama-model-loader.cpp` and `llama-model.cpp` to allocate only `NVMOE_CACHE_SIZE` expert slots in VRAM rather than the full `num_experts` (e.g. 48 slots instead of 512). This fits models that require 80+ GB VRAM into 12–14 GB VRAM.
3. **Non-Blocking Asynchronous CUDA Pipelining**:
   Transfers between Host RAM and GPU VRAM occur on a dedicated `h2d_stream`. Synchronization is performed entirely in GPU hardware via `cudaEventRecord` and `cudaStreamWaitEvent(cudaStreamPerThread, st->h2d_event, 0)`, eliminating CPU stalls in the token generation loop.
4. **Linux `io_uring` + `O_DIRECT` NVMe Engine**:
   Asynchronous batch expert reader using `io_uring_submit_and_wait` with 4KB sector alignment, bypassing the Linux kernel page cache for zero-copy transfers at hardware NVMe speeds (1.57+ GB/s).
5. **Dual-Tier Prewarming & Empirical Frequency Pinning**:
   Loads the most frequently visited experts (derived from empirical offline routing profiles, e.g. `freq_qwen38.bin`) into pinned host RAM and GPU VRAM at engine boot.

---

## Quickstart

### Prerequisites
- **OS**: Linux x86_64 (Kernel >= 5.10 with `io_uring` support)
- **Toolchain**: GCC / G++ >= 11 (supporting C++17), CMake >= 3.20
- **CUDA**: NVIDIA CUDA Toolkit >= 12.0 (`/opt/cuda` or `/usr/local/cuda`)
- **Libraries**: `liburing-dev` (`sudo apt install liburing-dev`)

### Automated Setup
Run the automated build script:
```bash
./setup.sh
```

### Manual Build Steps
1. **Initialize Submodule and Apply Patch**:
   ```bash
   git submodule update --init --recursive
   cd llama.cpp
   git apply ../scripts/llama_cpp_nvmoe.patch
   ```

2. **Build `llama.cpp` with CUDA**:
   ```bash
   mkdir -p build && cd build
   cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
   cmake --build . --config Release -j$(nproc) --target llama llama-common ggml ggml-base ggml-cuda ggml-cpu
   cd ../..
   ```

3. **Compile `moe_cache_probe`**:
   ```bash
   g++ -std=c++17 -O3 \
       -I llama.cpp/include -I llama.cpp/common -I llama.cpp/ggml/include -I llama.cpp/src -I /opt/cuda/include \
       scripts/moe_cache_probe.cpp \
       -L llama.cpp/build/bin -L /opt/cuda/lib64 \
       -lllama -lllama-common -lggml -lggml-base -lcudart -luring \
       -Wl,-rpath,'$ORIGIN/llama.cpp/build/bin' \
       -Wl,-rpath,/home/waltarb/nvmoe-llamacpp/llama.cpp/build/bin \
       -Wl,-rpath,/opt/cuda/lib64 \
       -o moe_cache_probe
   ```

---

## Running Inference

### Running Qwen3.8-Flash-Next (NVFP4, 512 Experts)
```bash
GGML_CUDA_DISABLE_GRAPHS=1 \
NVMOE_CACHE_SIZE=48 \
NVMOE_GPU_PINNED_EXPERTS=24 \
NVMOE_HOST_CACHE_SIZE=128 \
NVMOE_PINNED_EXPERTS=64 \
NVMOE_FREQ_PATH=models/freq_qwen38.bin \
./moe_cache_probe \
  -m models/qwen3.8-flash-next-nvfp4.gguf \
  -p "<|im_start|>user\nExplain quantum entanglement in simple terms.<|im_end|>\n<|im_start|>assistant\n" \
  -c 2048 -n 128 --temp 0
```

### Running Qwen3.6-35B-A3B (Q4_K_M, 256 Experts)
```bash
GGML_CUDA_DISABLE_GRAPHS=1 \
NVMOE_CACHE_SIZE=128 \
NVMOE_PINNED_EXPERTS=192 \
NVMOE_FREQ_PATH=/tmp/qwen_freq.bin \
./moe_cache_probe \
  -m /path/to/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -ngl 999 -c 512 -p "The capital of France is" -n 64 --temp 0
```

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `NVMOE_CACHE_SIZE` | `32` | Number of expert slots allocated per layer in **GPU VRAM**. Must be $< N_{experts}$. |
| `NVMOE_GPU_PINNED_EXPERTS` | `0` | Number of top routed experts permanently pinned in GPU VRAM (slots `0..K-1`). |
| `NVMOE_HOST_CACHE_SIZE` | `64` | Total number of expert slots allocated per layer in **Pinned Host RAM**. |
| `NVMOE_PINNED_EXPERTS` | `0` | Number of top routed experts permanently pinned in Host RAM. |
| `NVMOE_FREQ_PATH` | `""` | Path to empirical routing frequency binary (`uint32_t` matrix `[n_layers, n_experts]`). |
| `GGML_CUDA_DISABLE_GRAPHS`| `0` | Recommended set to `1` to avoid CUDA graph capture conflicts with dynamic H2D transfers. |
| `NVMOE_DEBUG_LOGITS` | `0` | Set to `1` to print top-5 token logits per generation step. |
| `NVMOE_DEBUG_TOKENS` | `0` | Set to `1` to inspect hex tokens and byte streams. |
| `NVMOE_VERIFY_READBACK` | `0` | Set to `1` to verify GPU tensor contents against host memory copies. |

---

## Critical Safety Constraints

When running on resource-constrained consumer GPUs (such as 16 GB Laptop GPUs with 32 GB RAM):
- **VRAM Hard Cap**: $< \mathbf{14\text{ GiB}}$ (Prevents CUDA Out-Of-Memory segmentation faults).
- **Host RAM Hard Cap**: $< \mathbf{20\text{ GiB}}$ (Prevents Linux OOM Killer invocation).
- **Always Specify `-c <context_size>`**: Default llama.cpp context allocation for Qwen3.8 is 256k tokens, which immediately allocates >30 GiB of KV cache and crashes the host. Always pass `-c 2048` or similar reasonable context limit.

---

## Repository Structure

```
nvmoe-llamacpp/
├── setup.sh                     # Automated build and setup script
├── README.md                    # Project overview, benchmarks, and usage guide
├── AGENTS.md                    # Technical and developer guide for AI agents
├── HANDOVER.md                  # Comprehensive architectural log & historical milestones
├── plan.md                      # Detailed technical roadmap across development phases
├── models/
│   ├── freq_qwen38.bin          # Empirical routing distribution for Qwen3.8 (192 KB)
│   └── qwen3.8-flash-next-...   # (Excluded from git) Quantized GGUF weights
├── scripts/
│   ├── moe_cache_probe.cpp      # Authoritative 3-tier cache inference engine
│   ├── llama_cpp_nvmoe.patch    # Clean patch for upstream llama.cpp
│   ├── convert_ftw_to_gguf.py   # FreeToken (.ftw) to GGUF converter
│   ├── repack_nvfp4_gguf.cpp    # In-place NVFP4 block reorganizer
│   ├── expert_reader_batch.cpp  # liburing async NVMe reader implementation
│   ├── segmented_host_lru.hpp   # Pinned host RAM segmented LRU cache header
│   └── bench_expert_io.cpp      # Standalone NVMe IO throughput benchmark
└── llama.cpp/                   # Git submodule pointing to ggml-org/llama.cpp
```

---

## License
MIT License. Modifications to `llama.cpp` retain the upstream MIT license.
