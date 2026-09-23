path = '/root/projects/llama.cpp/src/llama-model-loader.cpp'
with open(path) as f:
    content = f.read()

old = '''    ggml_backend_dev_t buft_dev = ggml_backend_buft_get_device(buft);
    fprintf(stderr, "[nvmoe-debug] create_tensor_reduced %s: buft=%s dev=%s dev_type=%d\\n",
            tn.str().c_str(), ggml_backend_buft_name(buft),
            buft_dev ? ggml_backend_dev_name(buft_dev) : "(null)",
            buft_dev ? (int) ggml_backend_dev_type(buft_dev) : -1);
    if (!buft_dev || ggml_backend_dev_type(buft_dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return nullptr;
    }'''

new = '''    ggml_backend_dev_t buft_dev = ggml_backend_buft_get_device(buft);
    // NOTE: checking ggml_backend_dev_type(buft_dev) == GPU is not enough -- CPU-placed
    // layers can still resolve to the "CUDA_Host" buffer type (pinned host memory used
    // to speed up H2D transfer), which reports its device as the CUDA device even though
    // compute for that layer still runs on the CPU. The real VRAM buffer type is always
    // exactly a GPU device's own default/primary buft, so require that instead.
    bool is_real_gpu_buft = buft_dev
        && ggml_backend_dev_type(buft_dev) == GGML_BACKEND_DEVICE_TYPE_GPU
        && buft == ggml_backend_dev_buffer_type(buft_dev);
    if (!is_real_gpu_buft) {
        return nullptr;
    }'''

assert old in content
content = content.replace(old, new)
with open(path, 'w') as f:
    f.write(content)
print('gpu check fixed')
