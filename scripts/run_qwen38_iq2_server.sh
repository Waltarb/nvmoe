#!/usr/bin/env bash
set -e

export GGML_CUDA_DISABLE_GRAPHS=1
export NVMOE_CACHE_SIZE=108
export NVMOE_GPU_PINNED_EXPERTS=64
export NVMOE_HOST_CACHE_SIZE=168
export NVMOE_PINNED_EXPERTS=120
export NVMOE_PRUNE_NVME_THRESH=0.45
export NVMOE_PRUNE_HOST_THRESH=0.04
export NVMOE_PRUNE_MIN_KEEP=2
export NVMOE_PRUNE_MIN_MASS=0.45
export NVMOE_FREQ_PATH=models/freq_qwen38.bin

exec ./moe_cache_probe --server --port 8080 \
  -m models/qwen-3.8-flash-unsloth/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf \
  -c 4096 -b 512 --temp 0
