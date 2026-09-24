#!/usr/bin/env bash
set -e

export GGML_CUDA_DISABLE_GRAPHS=1
export NVMOE_CACHE_SIZE=20
export NVMOE_GPU_PINNED_EXPERTS=6
export NVMOE_HOST_CACHE_SIZE=40
export NVMOE_PINNED_EXPERTS=8
export NVMOE_PRUNE_NVME_THRESH=0.28
export NVMOE_PRUNE_MIN_KEEP=2
export NVMOE_PRUNE_MIN_MASS=0.60
export NVMOE_FREQ_PATH=models/freq_glm53.bin

exec ./moe_cache_probe --server --port 8080 \
  -m models/glm-5.3-flash-iq2xxs/UD-IQ2_XXS/GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf \
  -c 4096 -b 512 --temp 0
