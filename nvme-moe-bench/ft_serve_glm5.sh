#!/usr/bin/env bash
# Launcher for GLM-5.3-Flash NVFP4 with 3-Tier NVMe Expert Offloading
set -e

export FREETOKEN_NVME_TIER=1
export FREETOKEN_PREFILL_NARROW=1
export FREETOKEN_GLM5_ATTN_FP8=1
export FREETOKEN_GLM5_MLP_FP8=1
export FREETOKEN_HOST_CACHE_SIZE=${FREETOKEN_HOST_CACHE_SIZE:-864}
export FREETOKEN_ROUTING_FREQ_PATH="${FREETOKEN_ROUTING_FREQ_PATH:-$HOME/nvme-moe-bench/glm5_routing_freq.pt}"
export FREETOKEN_O_DIRECT=${FREETOKEN_O_DIRECT:-1}
export FREETOKEN_IO_BACKEND=${FREETOKEN_IO_BACKEND:-iouring}
export FREETOKEN_PIPELINE_H2D=${FREETOKEN_PIPELINE_H2D:-1}
export FREETOKEN_SLRU_HOST_TIER=${FREETOKEN_SLRU_HOST_TIER:-1}
export FREETOKEN_PREWARM_CACHE=${FREETOKEN_PREWARM_CACHE:-1}

# Dynamic Top-K Pruning / Softmax Thresholding (GLM-5 specific)
# E.g. export FREETOKEN_TOPK_OVERRIDE=6 or pass --topk-override 6
# E.g. export FREETOKEN_ROUTER_MIN_PROB=0.05 or pass --router-min-prob 0.05
export FREETOKEN_TOPK_OVERRIDE=${FREETOKEN_TOPK_OVERRIDE:-}
export FREETOKEN_ROUTER_MIN_PROB=${FREETOKEN_ROUTER_MIN_PROB:-}

# Protect desktop responsiveness
export OMP_NUM_THREADS=4
export TORCH_NUM_THREADS=4

VENV_DIR="${FREETOKEN_VENV:-$HOME/.freetoken/venv}"
if [ -f "$VENV_DIR/bin/activate" ]; then
  source "$VENV_DIR/bin/activate"
fi

MODEL_DIR="${MODEL_DIR:-$HOME/.freetoken/models/glm-5.3-flash-nvfp4-ftw}"
FT_BIN="${FT_BIN:-$VENV_DIR/bin/ft}"

echo "=== Launching GLM-5.3-Flash NVFP4 with 3-Tier NVMe Cache ==="
echo "Model: $MODEL_DIR"
echo "Host Cache Slots: $FREETOKEN_HOST_CACHE_SIZE (approx 11.41 GiB pinned RAM)"
echo "GPU Cache Slots: 288 (approx 3.80 GiB VRAM)"
echo "CUDA Graphs: Disabled (--cuda-graph-max-bs 0 for per-step Python offload)"
echo "CPU Threads capped at: $OMP_NUM_THREADS"

exec taskset -c 2-19 nice -n 19 ionice -c 2 -n 7 "$FT_BIN" serve \
  --model "$MODEL_DIR" \
  --moe-backend offload \
  --moe-cache-size 288 \
  --max-running-req 1 \
  --cache-type naive \
  --memory-ratio 0.92 \
  --cuda-graph-max-bs 0 \
  --disable-moe-prefill-overlap \
  --port 1919 \
  "$@"
