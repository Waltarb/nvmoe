path = '/root/projects/llama.cpp/src/models/qwen35moe.cpp'
with open(path) as f:
    content = f.read()

old = '''        const int cache_size = nvmoe_cache_size();
        if (cache_size > 0 && cache_size < n_expert) {
            layer.ffn_down_exps = create_tensor_reduced(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il),
                    { n_ff_exp, n_embd, n_expert }, cache_size);
            layer.ffn_gate_exps = create_tensor_reduced(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", il),
                    { n_embd, n_ff_exp, n_expert }, cache_size);
            layer.ffn_up_exps = create_tensor_reduced(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", il),
                    { n_embd, n_ff_exp, n_expert }, cache_size);
        } else {
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
            create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);
        }
'''

new = '''        const int cache_size = nvmoe_cache_size();
        // create_tensor_reduced() only succeeds for GPU-placed layers (see its comment);
        // it returns nullptr for CPU-placed layers, where shrinking doesn't help anyway
        // (VRAM is the scarce resource, not host RAM) -- fall back to the normal,
        // full-size tensor in that case.
        layer.ffn_down_exps = cache_size > 0 && cache_size < n_expert
            ? create_tensor_reduced(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, cache_size)
            : nullptr;
        if (!layer.ffn_down_exps) {
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        }

        layer.ffn_gate_exps = cache_size > 0 && cache_size < n_expert
            ? create_tensor_reduced(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", il), { n_embd, n_ff_exp, n_expert }, cache_size)
            : nullptr;
        layer.ffn_up_exps = cache_size > 0 && cache_size < n_expert
            ? create_tensor_reduced(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", il), { n_embd, n_ff_exp, n_expert }, cache_size)
            : nullptr;
        if (!layer.ffn_gate_exps || !layer.ffn_up_exps) {
            create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);
        }
'''

assert old in content
content = content.replace(old, new)
with open(path, 'w') as f:
    f.write(content)
print('fallback patched')
