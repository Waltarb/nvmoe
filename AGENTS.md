# AGENTS.md — AI Developer & Agent Architecture Guide

This document is the authoritative guide for AI coding agents and engineers working on the **NVMoE for llama.cpp** codebase. It outlines the architectural invariants, hardware safety rules, design decisions, and common failure modes.

---

## 1. Safety Rules & Hardware Invariants

When working in this repository on the host machine, you **must adhere to these constraints at all times**:

| Resource | Available | Strict Hard Cap | Reason |
|---|---|---|---|
| **GPU VRAM** | 16 GB (RTX 3080 Ti Laptop) | **< 14 GiB** | Avoid CUDA out-of-memory errors & driver resets. |
| **Host RAM** | 31 GiB DDR5 | **< 20 GiB** | Avoid triggering the Linux OOM Killer. |

### Mandatory Execution Rules
1. **Always pass an explicit context limit (`-c 2048` or `-c 512`)**:
   MoE architectures such as Qwen3.8 have default training context lengths of 256k tokens. If `-c` is omitted, `llama.cpp` attempts to allocate 30+ GiB for the KV cache at startup, immediately causing host OOM.
2. **Never run FreeToken's offline batch loader `LLM()`**:
   FreeToken's batch mode preallocates massive buffers and will crash the system. Only reference offline weight files (`.ftw`) or use conversion scripts.
3. **Always set `GGML_CUDA_DISABLE_GRAPHS=1`**:
   CUDA graph capture can conflict with dynamic in-place H2D weight updates unless static graph capture nodes are specifically redesigned. Always run inference with graphs disabled.

---

## 2. Core Architecture & Mechanisms

### 2.1 The 3-Tier Hierarchy
- **Tier 1: GPU VRAM Cache**:
  - Allocated via `create_tensor_reduced()` with size `NVMOE_CACHE_SIZE` (e.g. 48 slots).
  - Slots `0 .. NVMOE_GPU_PINNED_EXPERTS - 1` are permanently locked with the model's top routed experts.
  - Remaining slots act as a dynamic LRU cache managed per-layer.
- **Tier 2: Pinned Host RAM Cache**:
  - Pinned CPU host memory (`cudaHostAlloc`) organized into a Segmented LRU (`scripts/segmented_host_lru.hpp`).
  - Slots `0 .. NVMOE_PINNED_EXPERTS - 1` are prewarmed at engine boot from NVMe via batch `io_uring`.
- **Tier 3: NVMe Direct Storage**:
  - Expert weights reside inside the GGUF file on NVMe SSD.
  - Read via `liburing` (`io_uring_submit_and_wait`) with `O_DIRECT`.
  - Windows are rounded to 4096-byte page boundaries to satisfy direct I/O alignment requirements.

### 2.2 The 1-Callback Unification
The entire cache engine hooks into `llama.cpp` using a single evaluation callback (`eval_cb`) set via `llama_set_eval_callback`:
```cpp
llama_set_eval_callback(ctx, eval_callback, &cache_state);
```

#### Critical Invariant: Tensor Filtering
In `eval_cb`, the callback **must strictly filter** for the exact MoE weight tensor:
```cpp
if (t->op != GGML_OP_GET_ROWS) return true;
if (strncmp(t->name, "ffn_moe_weights-", 16) != 0) return true;
```
**Why**: Downstream reshaped view tensors (or fused tensors) can trigger `eval_cb` with null or secondary source operands (`t->src[1] == nullptr`). Attempting to dereference router inputs on non-weight tensors causes immediate segmentation faults.

### 2.3 Non-Blocking Pipelined Asynchronous CUDA Synchronization
To achieve high decode token throughput (up to 8.0 tok/s steady-state), host-to-device transfers must completely overlap with NVMe reads without blocking the CPU thread:
- **Instant Host Hit Transfers**: Any expert that hits the pinned host RAM cache is dispatched immediately via `cudaMemcpyAsync` on `st->h2d_stream` *before* the layer's NVMe read queue is submitted.
- **CQE-Triggered H2D Transfers**: As each pending NVMe expert's blocks land via `io_uring_wait_cqe`, its H2D copy is enqueued to `st->h2d_stream` immediately, overlapping any subsequent disk reads.
- **Zero-Block Hardware Synchronization**:
  A CUDA event is recorded immediately after all copies:
  ```cpp
  cudaEventRecord(st->h2d_event, st->h2d_stream);
  ```
  The main computation stream waits on the event inside the CUDA driver:
  ```cpp
  cudaStreamWaitEvent(cudaStreamPerThread, st->h2d_event, 0);
  ```
- **Never call `cudaStreamSynchronize()`** inside the hot token generation loop!

### 2.4 Reduced Tensor Allocation (`create_tensor_reduced`)
Located in `llama.cpp/src/llama-model-loader.cpp` and `llama.cpp/src/llama-model.cpp`:
- Replaces full `[..., n_expert]` tensor shapes with `[..., cache_size]`.
- Does **not** insert the tensor into `ml.weights_map` (preventing `load_all_data()` from attempting to copy the full weight payload from disk).
- Allocates memory through `select_weight_buft` and `ctx_map`, ensuring proper `ggml_backend_buffer_t` registration so the graph scheduler recognizes the buffer.
- Only applied when the buffer type is a true GPU device buffer (`buft == ggml_backend_dev_buffer_type(buft_dev)`).

### 2.5 NVFP4 Super-Block Structure
For Qwen3.8-Flash-Next-NVFP4:
- Weights are quantized using NVIDIA's FP4 format: 4-bit E2M1 floating point values packed 2 per byte.
- Each 36-byte block contains:
  - 4 bytes: `ue4m3` micro-scales.
  - 32 bytes: 64 FP4 quantized weight values.
- In-place repacking is performed by `scripts/repack_nvfp4_gguf.cpp` to ensure memory layouts align with GGML `block_nvfp4` kernels.

---

## 3. Repository Structure & Key Files

| Path | Description |
|---|---|
| `scripts/moe_cache_probe.cpp` | **Authoritative runtime engine**. Contains 3-tier cache, `eval_cb`, async CUDA streaming, and token generation loop. |
| `scripts/llama_cpp_nvmoe.patch` | **Authoritative upstream patch** for `llama.cpp`. |
| `scripts/segmented_host_lru.hpp` | Standalone segmented LRU implementation for host pinned RAM. |
| `scripts/convert_ftw_to_gguf.py` | Converts FreeToken `.ftw` checkpoints into GGUF format. |
| `scripts/repack_nvfp4_gguf.cpp` | Repacks NVFP4 tensor blocks in-place inside GGUF files. |
| `models/freq_qwen38.bin` | Flat binary matrix (`48 layers x 512 experts`, `uint32_t`) of empirical routing frequencies. |
| `HANDOVER.md` | Detailed architectural logs and historical progression across all phases. |
| `plan.md` | Original technical roadmap and phase breakdown. |

---

## 4. Build & Verification Commands

### Quick Rebuild of `moe_cache_probe`
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

### Standard Verification Run (Qwen3.8-Flash-Next NVFP4)
```bash
GGML_CUDA_DISABLE_GRAPHS=1 \
NVMOE_CACHE_SIZE=48 \
NVMOE_GPU_PINNED_EXPERTS=24 \
NVMOE_HOST_CACHE_SIZE=128 \
NVMOE_PINNED_EXPERTS=64 \
NVMOE_FREQ_PATH=models/freq_qwen38.bin \
./moe_cache_probe \
  -m models/qwen3.8-flash-next-nvfp4.gguf \
  -p "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n" \
  -c 2048 -n 50 --temp 0
```
Expected output:
- Target decode throughput: **> 5.0 tok/s**.
- In-memory hit rate: **> 80%**.
- Output text: Coherent generation ending with `"The capital of France is **Paris**."`

### Standard Verification Run (GLM-5.3-Flash-UD-IQ2_XXS)
```bash
GGML_CUDA_DISABLE_GRAPHS=1 \
NVMOE_CACHE_SIZE=24 \
NVMOE_GPU_PINNED_EXPERTS=6 \
NVMOE_HOST_CACHE_SIZE=56 \
NVMOE_PINNED_EXPERTS=8 \
NVMOE_PRUNE_NVME_THRESH=0.28 \
NVMOE_PRUNE_MIN_KEEP=2 \
NVMOE_PRUNE_MIN_MASS=0.60 \
NVMOE_FREQ_PATH=models/freq_glm53.bin \
./moe_cache_probe \
  -m models/glm-5.3-flash-iq2xxs/UD-IQ2_XXS/GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf \
  -p "The capital of France is" \
  -c 512 -n 50 --temp 0
```
Expected output:
- Target decode throughput: **> 8.0 tok/s** (steady-state ~8.9 tok/s, min latency < 65 ms / ~16 tok/s peak).
- In-memory hit rate: **> 90%** (up to 93.4%).
- Pruned NVMe tail reads: **> 5,700 skipped per 50 tokens**.
- Output text: Coherent, rich generation describing Paris and France.

### Running the Benchmark Suite (`nvmoe-bench`)
```bash
cd bench
BENCH_ENDPOINT=http://localhost:8080/v1 BENCH_MODEL=local pnpm bench --config glm53-flash-iq2xxs --tier smoke
pnpm compare
```

---

## 5. Next Planned Milestones

When extending this repository, focus on these priorities:
1. **Speculative Prefetch (Layer $l+1$ Lookahead)**:
   Predict top-$k$ expert candidates for Layer $l+1$ while Layer $l$'s GEMMs are running on the GPU, achieving full overlap and pushing decode throughput higher.
2. **Upstream PR Packaging**:
   Clean up the changes in `scripts/llama_cpp_nvmoe.patch` to match `llama.cpp` coding guidelines for upstream submission.
3. **Multi-Turn KV Cache Management**:
   Enhance context window sliding / prefix reuse in server daemon mode for extended multi-turn dialogs.

