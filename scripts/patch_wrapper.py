import re

# --- llama-model.h: declare the wrapper ---
h_path = '/root/projects/llama.cpp/src/llama-model.h'
with open(h_path) as f:
    h = f.read()

old_h = '''    // convenience overload of create_tensor that doesn't require llama_model_loader
    ggml_tensor * create_tensor(const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags);
'''
assert old_h in h
new_h = old_h + '''
    // nvmoe: like create_tensor, but allocates with the last dimension overridden to
    // reduced_last_dim (a cache_size) instead of the real on-disk expert count -- see
    // llama_model_loader::create_tensor_reduced for what this does and does not do.
    ggml_tensor * create_tensor_reduced(const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne_real, int64_t reduced_last_dim);
'''
h = h.replace(old_h, new_h)
with open(h_path, 'w') as f:
    f.write(h)

# --- llama-model.cpp: implement the wrapper ---
cpp_path = '/root/projects/llama.cpp/src/llama-model.cpp'
with open(cpp_path) as f:
    cpp = f.read()

old_cpp = '''ggml_tensor * llama_model_base::create_tensor(const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    GGML_ASSERT(ml != nullptr);
    return create_tensor(*ml, tn, ne, flags);
}
'''
assert old_cpp in cpp
new_cpp = old_cpp + '''
ggml_tensor * llama_model_base::create_tensor_reduced(const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne_real, int64_t reduced_last_dim) {
    GGML_ASSERT(ml != nullptr);
    GGML_ASSERT(tn.bid != -1);
    const buft_list_t * buft_list_layer = pimpl->dev_layer.at(tn.bid).buft_list;
    return ml->create_tensor_reduced(hparams, buft_list_layer, tn, ne_real, reduced_last_dim);
}
'''
cpp = cpp.replace(old_cpp, new_cpp)
with open(cpp_path, 'w') as f:
    f.write(cpp)

print('wrapper patched')
