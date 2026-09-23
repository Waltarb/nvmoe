#!/bin/bash
# Simple safety watcher: logs WSL free -h every 2s so a runaway test can be caught early.
while true; do
    echo "=== $(date +%H:%M:%S) ==="
    free -h | sed -n '2p'
    sleep 2
done
