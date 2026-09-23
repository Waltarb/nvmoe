#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=== Setting up NVMoE for llama.cpp ==="

# 1. Check prerequisites
command -v g++ >/dev/null 2>&1 || { echo "Error: g++ not found. Please install build-essential."; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo "Error: cmake not found."; exit 1; }
command -v nvcc >/dev/null 2>&1 || { echo "Warning: nvcc not in PATH. Assuming CUDA libraries are in /opt/cuda or /usr/local/cuda."; }

CUDA_PATH="/opt/cuda"
if [ ! -d "${CUDA_PATH}" ]; then
    if [ -d "/usr/local/cuda" ]; then
        CUDA_PATH="/usr/local/cuda"
    fi
fi
echo "[+] Using CUDA Path: ${CUDA_PATH}"

# 2. Check / Init Submodule
if [ ! -f "llama.cpp/CMakeLists.txt" ]; then
    echo "[+] Initializing llama.cpp git submodule..."
    git submodule update --init --recursive
fi

# 3. Apply NVMoE Patch to llama.cpp
echo "[+] Applying scripts/llama_cpp_nvmoe.patch..."
cd "${SCRIPT_DIR}/llama.cpp"
if git diff --quiet; then
    echo "[+] Applying patch..."
    git apply "${SCRIPT_DIR}/scripts/llama_cpp_nvmoe.patch" || echo "[!] Patch may already be applied or partially applied."
else
    echo "[+] llama.cpp already has working changes. Skipping patch application."
fi

# 4. Build llama.cpp with CUDA support
echo "[+] Building llama.cpp with CUDA support..."
mkdir -p build
cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j"$(nproc)" --target llama llama-common ggml ggml-base ggml-cuda ggml-cpu
cd "${SCRIPT_DIR}"

# 5. Build moe_cache_probe and helper binaries
echo "[+] Compiling moe_cache_probe..."
g++ -std=c++17 -O3 \
    -I llama.cpp/include \
    -I llama.cpp/common \
    -I llama.cpp/ggml/include \
    -I llama.cpp/src \
    -I "${CUDA_PATH}/include" \
    scripts/moe_cache_probe.cpp \
    -L llama.cpp/build/bin \
    -L "${CUDA_PATH}/lib64" \
    -lllama -lllama-common -lggml -lggml-base -lcudart -luring \
    -Wl,-rpath,'$ORIGIN/llama.cpp/build/bin' \
    -Wl,-rpath,"${SCRIPT_DIR}/llama.cpp/build/bin" \
    -Wl,-rpath,"${CUDA_PATH}/lib64" \
    -o moe_cache_probe

echo "[+] Compiling repack_nvfp4_gguf..."
g++ -std=c++17 -O3 \
    -I llama.cpp/ggml/include \
    scripts/repack_nvfp4_gguf.cpp \
    -L llama.cpp/build/bin \
    -lggml -lggml-base \
    -Wl,-rpath,'$ORIGIN/llama.cpp/build/bin' \
    -Wl,-rpath,"${SCRIPT_DIR}/llama.cpp/build/bin" \
    -fopenmp \
    -o scripts/repack_nvfp4_gguf

echo "=== Build Complete! ==="
echo "You can now run:"
echo "  NVMOE_CACHE_SIZE=48 NVMOE_GPU_PINNED_EXPERTS=24 NVMOE_HOST_CACHE_SIZE=128 NVMOE_PINNED_EXPERTS=64 NVMOE_FREQ_PATH=models/freq_qwen38.bin ./moe_cache_probe -m models/qwen3.8-flash-next-nvfp4.gguf -p '<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n' -c 2048 -n 50 --temp 0"
