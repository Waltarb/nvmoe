path = '/root/projects/llama.cpp/src/llama-model-loader.cpp'
with open(path) as f:
    content = f.read()

marker = 'void llama_model_loader::done_getting_tensors(bool partial) const {'
assert marker in content

new_method = '''struct ggml_tensor * llama_model_loader::create_tensor_reduced(
        const llama_hparams & hparams, const buft_list_t * buft_list_layer,
        const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne_real, int64_t reduced_last_dim) {
    const struct ggml_tensor * cur = check_tensor_dims(tn.str(), ne_real, /*required=*/true, /*allow_reshape=*/false);
    if (cur == NULL) {
        return NULL;
    }

    ggml_tensor t_meta = *cur;
    // only the last (expert-count) dimension shrinks; nb[0..2] don't depend on ne[2] itself,
    // only nb[3] (= nb[2]*ne[2], used by ggml_nbytes) needs recomputing.
    t_meta.ne[2] = reduced_last_dim;
    t_meta.nb[3] = t_meta.nb[2] * t_meta.ne[2];

    const llm_tensor_info & info = llm_tensor_info_for(tn.tensor);
    ggml_backend_buffer_type_t buft = select_weight_buft(hparams, &t_meta, info.op, buft_list_layer);
    if (!buft) {
        throw std::runtime_error(format("failed to find a compatible buffer type for reduced tensor %s", tn.str().c_str()));
    }

    const ctx_key key { buft, false };
    ggml_context * ctx;
    auto it = ctx_map.find(key);
    if (it == ctx_map.end()) {
        int max_n_tensors = n_tensors;
        max_n_tensors += 1;
        max_n_tensors += hparams.n_layer()*2;
        if (files.empty()) {
            max_n_tensors += hparams.n_layer()*256;
        }
        const size_t ctx_size = ggml_tensor_overhead()*max_n_tensors;

        ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
        if (!ctx) {
            throw std::runtime_error(format("failed to create ggml context"));
        }
        ctx_map.emplace(key, ctx);
    } else {
        ctx = it->second.get();
    }

    struct ggml_tensor * tensor = ggml_dup_tensor(ctx, &t_meta);
    ggml_set_name(tensor, tn.str().c_str());
    n_created++;
    // deliberately NOT inserted into weights_map: load_all_data() must never try to
    // memcpy the real (larger) on-disk tensor's bytes into this smaller allocation.
    return tensor;
}

'''

content = content.replace(marker, new_method + marker)
with open(path, 'w') as f:
    f.write(content)
print('cpp patched')
