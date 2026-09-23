#!/bin/bash
set -u
cd ~/models
TARGET=20419565568
FILE=Qwen3.6-35B-A3B-Q4_K_M.gguf
URL='https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-Q4_K_M.gguf'

get_size() {
  stat -c%s "$FILE" 2>/dev/null || echo 0
}

while [ "$(get_size)" -lt "$TARGET" ]; do
  curl -L --http1.1 --retry-all-errors --retry 5 --retry-delay 3 -o "$FILE" --continue-at - "$URL"
  echo "progress: $(get_size) / $TARGET"
done
echo DOWNLOAD_COMPLETE
