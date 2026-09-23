#!/usr/bin/env bash
# Production Launcher: Qwen3.8-Flash-Next (Background Profile)
# Tuned for maximum background throughput: reserves Core 0 (threads 0,1), uses 5120 host slots
# (3072 calibration-pinned + 2048 dynamic LRU), 768 GPU slots, Radix prefix cache, BF16 SSM states,
# and QD16 I/O (4.64 tok/s sustained decode, 1.52x faster agentic sessions), safely leaving >= 10.2 GiB available RAM.
set -e

VENV_DIR="${FREETOKEN_VENV:-$HOME/.freetoken/venv}"
if [ -f "$VENV_DIR/bin/activate" ]; then
  source "$VENV_DIR/bin/activate"
fi

export FREETOKEN_NVME_TIER=1
export FREETOKEN_PREFILL_NARROW=1
export FREETOKEN_HOST_CACHE_SIZE=5120
export FREETOKEN_PREWARM_CACHE=${FREETOKEN_PREWARM_CACHE:-1}
export FREETOKEN_PINNED_EXPERTS=${FREETOKEN_PINNED_EXPERTS:-64}
export FREETOKEN_IO_WORKERS=${FREETOKEN_IO_WORKERS:-16}
export FREETOKEN_IO_BACKEND=${FREETOKEN_IO_BACKEND:-iouring}
export FREETOKEN_MAMBA_SSM_DTYPE=${FREETOKEN_MAMBA_SSM_DTYPE:-bfloat16}
export FREETOKEN_PIPELINE_H2D=${FREETOKEN_PIPELINE_H2D:-1}
export FREETOKEN_O_DIRECT=${FREETOKEN_O_DIRECT:-1}
export FREETOKEN_SLRU_HOST_TIER=${FREETOKEN_SLRU_HOST_TIER:-1}
export FREETOKEN_GPU_PINNED_PER_LAYER=${FREETOKEN_GPU_PINNED_PER_LAYER:-6}
export FREETOKEN_COLLECT_ROUTING=${FREETOKEN_COLLECT_ROUTING:-0}
export NVME_MOE_BENCH_DIR="${NVME_MOE_BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

MODEL_PATH="${MODEL_PATH:-$HOME/.freetoken/models/qwen3.8-flash-next-nvfp4-ftw}"
PORT="${PORT:-8000}"

# Process and resource isolation: reserve Core 0 (threads 0,1) for OS/desktop via taskset,
# lowest priority (nice 19), best-effort/idle I/O, 4 CPU compute threads
taskset -c 2-19 nice -n 19 ionice -c 2 -n 7 \
  env OMP_NUM_THREADS=4 TORCH_NUM_THREADS=4 \
  python3 -m freetoken.cli serve \
    --model-path "$MODEL_PATH" \
    --moe-cache-size "${MOE_CACHE_SIZE:-768}" \
    --disable-moe-prefill-overlap \
    --ple-backend disk \
    --max-running-req 1 \
    --cache-type radix \
    --enable-special-token-ckpt \
    --enable-cache-report \
    --cuda-graph-max-bs 0 \
    --memory-ratio 0.88 \
    --port "$PORT" \
    "$@"
