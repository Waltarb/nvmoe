#!/usr/bin/env bash
set -e

export GGML_CUDA_DISABLE_GRAPHS=1
export NVMOE_CACHE_SIZE=32
export NVMOE_GPU_PINNED_EXPERTS=16
export NVMOE_HOST_CACHE_SIZE=64
export NVMOE_PINNED_EXPERTS=32
export NVMOE_PRUNE_NVME_THRESH=0.08
export NVMOE_PRUNE_MIN_KEEP=4
export NVMOE_PRUNE_MIN_MASS=0.85
export NVMOE_FREQ_PATH=models/freq_qwen38.bin

exec ./moe_cache_probe --server --port 8080 \
  -m models/qwen3.8-flash-next-nvfp4.gguf \
  -c 4096 -b 512 --temp 0
