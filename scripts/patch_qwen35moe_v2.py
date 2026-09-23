path = '/root/projects/llama.cpp/src/models/qwen35moe.cpp'
with open(path) as f:
    content = f.read()

old_helper_block = '''#include "models.h"
#include "llama-memory-recurrent.h"

#include <cstdlib>

// --- nvmoe Phase 3 prototype -------------------------------------------------
// When NVMOE_CACHE_SIZE is set, allocate the routed-expert weight tensors with a
// reduced expert dimension (cache_size instead of the model's real n_expert) so
// the cache's GPU tier only ever holds cache_size experts' worth of VRAM per bank,
// instead of all of them. Bypasses llama_model_loader::create_tensor entirely for
// these tensors (which cannot allocate less than the GGUF's real tensor size, see
// TENSOR_ALLOW_RESHAPE's byte-count assert), so load_all_data() never tries to fill
// them from the file -- they start as zeroed placeholders and are populated at
// runtime by the cache logic (later phases). The backing ggml_context is
// intentionally leaked for the process lifetime, matching the model's own lifetime.
static ggml_tensor * nvmoe_alloc_reduced_expert_tensor(
        ggml_type type, int64_t n_embd_, int64_t n_ff_, int64_t n_expert_real,
        const char * name) {
    static int cache_size = -1;
    if (cache_size < 0) {
        const char * env = getenv("NVMOE_CACHE_SIZE");
        cache_size = env ? atoi(env) : 0;
    }
    if (cache_size <= 0 || cache_size >= n_expert_real) {
        return nullptr; // not enabled, or no reduction requested -- caller uses normal path
    }

    // Generous overestimate (2 bytes/element) rather than exact quant-block math --
    // this is just a memory pool size, slight overallocation is harmless.
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() + (size_t) n_embd_ * n_ff_ * cache_size * 2 + 4096,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params); // leaked intentionally, see comment above
    ggml_tensor * t = ggml_new_tensor_3d(ctx, type, n_embd_, n_ff_, cache_size);
    ggml_set_name(t, name);
    memset(t->data, 0, ggml_nbytes(t));
    return t;
}
'''

new_helper_block = '''#include "models.h"
#include "llama-memory-recurrent.h"

#include <cstdlib>

// --- nvmoe Phase 3 prototype -------------------------------------------------
// When NVMOE_CACHE_SIZE is set, allocate the routed-expert weight tensors via
// llama_model_base::create_tensor_reduced() with a reduced expert dimension
// (cache_size instead of the model's real n_expert), so the cache's GPU tier only
// ever holds cache_size experts' worth of VRAM per bank. That helper goes through
// the model's normal buft/device selection and gets folded into the same bulk
// backend-buffer allocation as every other weight (see llama_model_loader::
// create_tensor_reduced) -- unlike an ad hoc ggml_context, this gives the tensor a
// real ggml_backend_buffer_t on the correct device. It is NOT registered in
// weights_map, so load_all_data() never tries to fill it from the file -- it starts
// as an uninitialized backend allocation and is populated at runtime by the cache
// logic (later phases).
static int nvmoe_cache_size() {
    static int cache_size = -1;
    if (cache_size < 0) {
        const char * env = getenv("NVMOE_CACHE_SIZE");
        cache_size = env ? atoi(env) : 0;
    }
    return cache_size;
}
'''

assert old_helper_block in content
content = content.replace(old_helper_block, new_helper_block)

old_calls = '''        layer.ffn_down_exps = nvmoe_alloc_reduced_expert_tensor(GGML_TYPE_Q4_K, n_ff_exp, n_embd, n_expert,
                tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il).str().c_str());
        if (!layer.ffn_down_exps) {
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        }

        layer.ffn_gate_exps = nvmoe_alloc_reduced_expert_tensor(GGML_TYPE_Q4_K, n_embd, n_ff_exp, n_expert,
                tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", il).str().c_str());
        layer.ffn_up_exps = nvmoe_alloc_reduced_expert_tensor(GGML_TYPE_Q4_K, n_embd, n_ff_exp, n_expert,
                tn(LLM_TENSOR_FFN_UP_EXPS, "weight", il).str().c_str());
        if (!layer.ffn_gate_exps || !layer.ffn_up_exps) {
            create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);
        }
'''

new_calls = '''        const int cache_size = nvmoe_cache_size();
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

assert old_calls in content
content = content.replace(old_calls, new_calls)

with open(path, 'w') as f:
    f.write(content)
print('qwen35moe.cpp patched')
