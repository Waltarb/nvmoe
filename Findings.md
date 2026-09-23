# NVMoE Performance Optimization Findings & Empirical Log

**Target Model**: `Qwen3.8-Flash-Next-NVFP4` (48 layers, 512 experts/layer, top-10 routing)  
**System Hardware**: NVIDIA GeForce RTX 3080 Ti Laptop GPU (16 GB VRAM, 175W TGP), 32 GB DDR5-4800, PCIe Gen4 x4 Micron 3400 NVMe SSD, Linux 6.x.  
**Safety Hard Limits**: Host RAM < 20 GiB, GPU VRAM < 14 GiB. Mandatory `-c 2048`.  
**Benchmark Prompt**: Controlled 150-token decode (`temperature=0`):  
`"<|im_start|>user\nCount from 1 to 200, one number per line, no other text.<|im_end|>\n<|im_start|>assistant\n"`

---

## Performance Progression Summary

| Milestone / Experiment | Tag | Decode (tok/s) | Steady-State (tok/s) | TTFT (s) | 150-tok Time (s) | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **FreeToken Reference Average** | — | 5.70 | 5.70 | ~7.5s | ~26.3s | Reference |
| **FreeToken Pipelined Peak (§39.3)** | — | 6.57 | 6.57 | ~7.0s | ~22.8s | Reference |
| **NVMoE Baseline (Cold/Sequential)** | — | 4.81 | 4.81 | 4.57s | 30.98s | Superseded |
| **Exp 1: Pipelined H2D Overlap** | — | 6.78 | 6.76 | 3.15s | 21.99s | **Success (+41.0%)** |
| **Exp 2: Non-blocking Writeback + 52-slot tuning** | `v1-baseline-7.18toks` | **7.18** | **7.18 (peak 8.37)** | **3.13s** | **20.75s** | **Success (+49.3%)** |

---

## Detailed Experiments & Findings

### Experiment 1: NVMe $\leftrightarrow$ PCIe H2D Pipeline Overlap
- **Hypothesis**: The sequential execution of `io_uring_submit_and_wait` followed by whole-layer `cudaMemcpyAsync` serialized ~33 ms of PCIe latency behind disk reads. Enqueuing Host Hits immediately before disk submission and CQE-triggering H2D copies as NVMe blocks land will hide PCIe transfer time behind NVMe I/O.
- **Implementation**:
  - Identified GPU and Host slots upfront.
  - Dispatched Host Hit transfers on `st->h2d_stream` *prior* to `io_uring_submit`.
  - Dispatched individual expert H2D transfers inside the `io_uring_wait_cqe` loop immediately as each expert's gate_up and down blocks land.
- **Results**:
  - TTFT improved from 4.57 s to 3.15 s (**+45.2%**).
  - 150-token decode throughput increased from 4.81 tok/s to **6.78 tok/s** (**+41.0%**).
  - 50-token decode reached **7.88 tok/s** (8.00 tok/s steady).
- **Verdict**: **Massive Win**. Shipped.

### Experiment 2: Non-blocking Async Selected Experts Writeback & High-Priority Stream
- **Hypothesis**: `ggml_backend_tensor_set` on `selected_experts` called `ggml_backend_cuda_buffer_set_tensor`, which called `cudaStreamSynchronize(cudaStreamPerThread)` 48 times per token. Replacing this with an asynchronous `cudaMemcpyAsync` from pinned host memory avoids 48 blocking CPU-GPU syncs per token. Additionally, giving `h2d_stream` highest CUDA priority (`cudaStreamCreateWithPriority`) ensures PCIe DMA transfers preempt lower-priority GPU tasks. Tuning GPU cache to 52 slots (28 pinned) increases GPU VRAM hit rate while staying well below the 14 GiB cap (~13.1 GiB).
- **Results**:
  - 150-token decode throughput reached **7.18 tok/s** (139.3 ms/tok).
  - Peak 50-token steady-state decode reached **8.37 tok/s** (119.4 ms/tok, min 75.4 ms).
  - Total 150-token decode time cut to **20.75 s** (over 10 seconds faster than original baseline).
- **Verdict**: **Massive Win**. Tagged `v1-baseline-7.18toks`.


### Experiment 3: Scaling GPU Slots to 56 & GPU Pinning to 32
- **Hypothesis**: Expanding GPU cache to 56 slots and pinning the top-32 empirical experts (which capture 33.4% of historical routing mass) would push decode throughput higher.
- **Results**:
  - GPU VRAM Hit Rate rose to **66.9%** (up from 65.0%).
  - In-Memory Hit Rate **dropped from 63.1% to 54.9%**.
  - Decode throughput decreased from 8.25 tok/s to **7.61 tok/s**.
- **Analysis**:
  Confirms FreeToken's finding (§1.4 in `performance.md`): static calibration pinning has diminishing returns. Pinning 32 static slots deprives the dynamic LRU tier of elasticity. For specific prompts, static slots sit unused while active prompt-specific experts are forced to evict and read from NVMe repeatedly.
- **Verdict**: **Rejected**. Optimal GPU cache configuration is `NVMOE_CACHE_SIZE=52`, `NVMOE_GPU_PINNED_EXPERTS=24-28`.

