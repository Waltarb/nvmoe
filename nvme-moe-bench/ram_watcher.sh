#!/usr/bin/env bash
# OOM Prevention Daemon: monitors MemAvailable every 500ms.
# If MemAvailable falls below 3500 MB (3.5 GiB safety threshold),
# it instantly kills all freetoken server and benchmark processes to prevent system crash.

MIN_AVAILABLE_MB=3500
LOG_FILE="/tmp/ram_watcher.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] RAM Watcher started (Threshold: < ${MIN_AVAILABLE_MB} MB MemAvailable)" | tee -a "$LOG_FILE"

while true; do
  # Read MemAvailable directly from /proc/meminfo (in kB)
  avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if [ -n "$avail_kb" ]; then
    avail_mb=$((avail_kb / 1024))
    if [ "$avail_mb" -lt "$MIN_AVAILABLE_MB" ]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] CRITICAL: MemAvailable dropped to ${avail_mb} MB (< ${MIN_AVAILABLE_MB} MB)! Killing freetoken to prevent OOM crash!" | tee -a "$LOG_FILE"
      pkill -9 -f "freetoken" || true
      pkill -9 -f "bench_ceiling" || true
      rm -f /tmp/ft_artificial_ratio.txt /tmp/ft_force_pinned_pct.txt /tmp/ft_reset_stats.txt
      sleep 5
    fi
  fi
  sleep 0.5
done
