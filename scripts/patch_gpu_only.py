path = '/root/projects/llama.cpp/src/llama-model-loader.cpp'
with open(path) as f:
    content = f.read()

old = '''    const llm_tensor_info & info = llm_tensor_info_for(tn.tensor);
    ggml_backend_buffer_type_t buft = select_weight_buft(hparams, &t_meta, info.op, buft_list_layer);
    if (!buft) {
        throw std::runtime_error(format("failed to find a compatible buffer type for reduced tensor %s", tn.str().c_str()));
    }

    const ctx_key key { buft, false };'''

new = '''    const llm_tensor_info & info = llm_tensor_info_for(tn.tensor);
    ggml_backend_buffer_type_t buft = select_weight_buft(hparams, &t_meta, info.op, buft_list_layer);
    if (!buft) {
        throw std::runtime_error(format("failed to find a compatible buffer type for reduced tensor %s", tn.str().c_str()));
    }

    // Shrinking only matters for the GPU tier (VRAM is the scarce resource); a CPU-backed
    // tensor for this layer would, under mmap load mode, get its ->data pointer set via a
    // completely different path (ggml_backend_dev_buffer_from_host_ptr wrapping the mmap,
    // driven by weights_map lookups) that this reduced tensor is deliberately not part of --
    // it would silently stay null. So: only take the reduced path when the tensor actually
    // landed on a GPU device; let the caller fall back to the normal, full-size create_tensor
    // for CPU-placed layers.
    ggml_backend_dev_t buft_dev = ggml_backend_buft_get_device(buft);
    if (!buft_dev || ggml_backend_dev_type(buft_dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return nullptr;
    }

    const ctx_key key { buft, false };'''

assert old in content
content = content.replace(old, new)
with open(path, 'w') as f:
    f.write(content)
print('gpu-only guard patched')
