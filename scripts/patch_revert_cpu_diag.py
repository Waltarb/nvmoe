path = '/root/projects/llama.cpp/src/llama-model-loader.cpp'
with open(path) as f:
    content = f.read()

old = '''    bool is_real_gpu_buft = buft_dev
        && ggml_backend_dev_type(buft_dev) == GGML_BACKEND_DEVICE_TYPE_GPU
        && buft == ggml_backend_dev_buffer_type(buft_dev);
    // TEMP DIAGNOSTIC ONLY: also allow the default CPU buft when NVMOE_ALLOW_CPU_CACHE is
    // set, so the cache-fill mechanism can be isolation-tested on pure-CPU compute (no CUDA
    // graphs involved) with --load-mode none. Remove once the CUDA-graph-interaction theory
    // is confirmed or ruled out.
    bool is_real_cpu_buft = buft_dev
        && ggml_backend_dev_type(buft_dev) == GGML_BACKEND_DEVICE_TYPE_CPU
        && buft == ggml_backend_dev_buffer_type(buft_dev)
        && getenv("NVMOE_ALLOW_CPU_CACHE") != nullptr;
    if (!is_real_gpu_buft && !is_real_cpu_buft) {
        return nullptr;
    }'''

new = '''    bool is_real_gpu_buft = buft_dev
        && ggml_backend_dev_type(buft_dev) == GGML_BACKEND_DEVICE_TYPE_GPU
        && buft == ggml_backend_dev_buffer_type(buft_dev);
    if (!is_real_gpu_buft) {
        return nullptr;
    }'''

assert old in content
content = content.replace(old, new)
with open(path, 'w') as f:
    f.write(content)
print('reverted CPU diagnostic hook')
