h_path = '/root/projects/llama.cpp/include/llama.h'
with open(h_path) as f:
    h = f.read()

old = '''    LLAMA_API const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
'''
assert old in h
new = old + '''
    // nvmoe: look up a model weight tensor by its exact GGUF name (e.g. "blk.0.ffn_gate_exps.weight").
    // Returns NULL if not found. Used by the expert cache to read/write GPU-resident weight
    // buffers directly (ggml_backend_tensor_get/set) for cache fills and slot-id remaps.
    LLAMA_API struct ggml_tensor * llama_model_get_tensor(const struct llama_model * model, const char * name);
'''
h = h.replace(old, new)
with open(h_path, 'w') as f:
    f.write(h)

cpp_path = '/root/projects/llama.cpp/src/llama-model.cpp'
with open(cpp_path) as f:
    cpp = f.read()

marker = 'std::string llama_model::arch_name() const {'
assert marker in cpp
new_impl = '''struct ggml_tensor * llama_model_get_tensor(const struct llama_model * model, const char * name) {
    return const_cast<struct ggml_tensor *>(model->get_tensor(name));
}

'''
cpp = cpp.replace(marker, new_impl + marker)
with open(cpp_path, 'w') as f:
    f.write(cpp)

print('llama_model_get_tensor API added')
