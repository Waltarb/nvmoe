#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

stop_server() {
  echo "[orchestrator] Stopping any running server..."
  pkill -9 -f "moe_cache_probe --server" || true
  fuser -k 8080/tcp 2>/dev/null || true
  sleep 4
}

wait_for_server() {
  echo "[orchestrator] Waiting for server on http://localhost:8080/health..."
  for i in {1..60}; do
    if curl -s http://localhost:8080/health | grep -q '"status":"ok"'; then
      echo "[orchestrator] Server is online and ready!"
      return 0
    fi
    sleep 5
  done
  echo "[orchestrator] Server failed to start in 300s!"
  exit 1
}

run_suite() {
  local model_name="$1"
  local server_script="$2"
  local tier="${3:-core}"
  local budget="${4:-4000}"

  echo "=========================================================="
  echo " [orchestrator] Starting evaluation: $model_name"
  echo " Tier: $tier | Output Budget: $budget | No Timeout"
  echo "=========================================================="

  stop_server

  echo "[orchestrator] Launching server: $server_script..."
  "$server_script" > "server_${model_name}.log" 2>&1 &
  local s_pid=$!

  wait_for_server

  echo "[orchestrator] Running benchmark harness for $model_name..."
  cd "$DIR/bench"
  BENCH_ENDPOINT=http://localhost:8080/v1 \
  BENCH_MODEL=local \
  pnpm bench --config "$model_name" --tier "$tier" --noTimeout --budget "$budget" || true

  cd "$DIR"
  stop_server
  echo "[orchestrator] Completed $model_name!"
  sleep 5
}

# Allow passing single model: ./scripts/run_all_benchmarks.sh <model>
TARGET="${1:-all}"

if [ "$TARGET" = "all" ] || [ "$TARGET" = "nvfp4" ]; then
  run_suite "qwen38-nvfp4" "$DIR/scripts/run_nvfp4_server.sh" "core" "4000"
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "glm53" ]; then
  run_suite "glm53-flash-iq2xxs" "$DIR/scripts/run_glm53_server.sh" "core" "4000"
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "qwen38-iq2" ]; then
  run_suite "qwen38-iq2" "$DIR/scripts/run_qwen38_iq2_server.sh" "core" "4000"
fi

echo "=========================================================="
echo " [orchestrator] All evaluations finished. Final scoreboard:"
echo "=========================================================="
cd "$DIR/bench"
pnpm compare
