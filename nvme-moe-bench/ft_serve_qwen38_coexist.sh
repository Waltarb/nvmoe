#!/usr/bin/env bash
# Production Launcher: Qwen3.8-Flash-Next (Coexist Profile)
# Tuned for desktop multitasking: reserves Core 0 (threads 0,1), caps host cache to 3072 slots,
# leaving ~6.0 GiB free physical RAM for desktop browsers, editors, and window manager.
set -e

VENV_DIR="${FREETOKEN_VENV:-$HOME/.freetoken/venv}"
if [ -f "$VENV_DIR/bin/activate" ]; then
  source "$VENV_DIR/bin/activate"
fi

export FREETOKEN_NVME_TIER=1
export FREETOKEN_PREFILL_NARROW=1
export FREETOKEN_HOST_CACHE_SIZE=3072
export FREETOKEN_PREWARM_CACHE=${FREETOKEN_PREWARM_CACHE:-1}
export FREETOKEN_PINNED_EXPERTS=${FREETOKEN_PINNED_EXPERTS:-48}
export FREETOKEN_IO_WORKERS=${FREETOKEN_IO_WORKERS:-16}
export FREETOKEN_PIPELINE_H2D=${FREETOKEN_PIPELINE_H2D:-1}
export FREETOKEN_O_DIRECT=${FREETOKEN_O_DIRECT:-1}
export FREETOKEN_SLRU_HOST_TIER=${FREETOKEN_SLRU_HOST_TIER:-1}

MODEL_PATH="${MODEL_PATH:-$HOME/.freetoken/models/qwen3.8-flash-next-nvfp4-ftw}"
PORT="${PORT:-8000}"

# Process and resource isolation: reserve Core 0 (threads 0,1) for OS/desktop via taskset,
# lowest priority (nice 19), best-effort/idle I/O, 4 CPU compute threads
taskset -c 2-19 nice -n 19 ionice -c 2 -n 7 \
  env OMP_NUM_THREADS=4 TORCH_NUM_THREADS=4 \
  python3 -m freetoken.cli serve \
    --model-path "$MODEL_PATH" \
    --moe-cache-size 512 \
    --disable-moe-prefill-overlap \
    --ple-backend disk \
    --max-running-req 1 \
    --cache-type naive \
    --cuda-graph-max-bs 0 \
    --memory-ratio 0.88 \
    --port "$PORT" \
    "$@"
