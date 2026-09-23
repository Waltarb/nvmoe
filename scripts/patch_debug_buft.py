path = '/root/projects/llama.cpp/src/llama-model-loader.cpp'
with open(path) as f:
    content = f.read()

old = '''    ggml_backend_dev_t buft_dev = ggml_backend_buft_get_device(buft);
    if (!buft_dev || ggml_backend_dev_type(buft_dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return nullptr;
    }'''

new = '''    ggml_backend_dev_t buft_dev = ggml_backend_buft_get_device(buft);
    fprintf(stderr, "[nvmoe-debug] create_tensor_reduced %s: buft=%s dev=%s dev_type=%d\\n",
            tn.str().c_str(), ggml_backend_buft_name(buft),
            buft_dev ? ggml_backend_dev_name(buft_dev) : "(null)",
            buft_dev ? (int) ggml_backend_dev_type(buft_dev) : -1);
    if (!buft_dev || ggml_backend_dev_type(buft_dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return nullptr;
    }'''

assert old in content
content = content.replace(old, new)
with open(path, 'w') as f:
    f.write(content)
print('debug print added')
