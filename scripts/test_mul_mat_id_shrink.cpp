// Isolated correctness test, no llama.cpp model loading at all: does ggml_mul_mat_id
// on the CUDA backend produce the SAME output when (a) run against a full-size expert
// tensor with real ids, vs (b) run against a shrunk expert tensor holding the exact same
// per-expert data at different (remapped) slot indices, with ids rewritten accordingly?
// This isolates the core nvmoe Phase 3 mechanism from every model/quantization/graph
// complexity that's been ruled out so far (CUDA graphs, gate/up fusion, self-eviction,
// uninitialized memory, write/readback correctness) but not yet from mul_mat_id itself.
#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

static ggml_backend_t g_backend = nullptr;

static ggml_tensor * make_input(ggml_context * ctx, const char * name, int64_t ne0, int64_t ne1, int64_t ne2) {
    ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_set_name(t, name);
    ggml_set_input(t);
    return t;
}

// Runs as = mul_mat_id(weights, b, ids) on the CUDA backend and returns dst's data.
// `weights_f32` is quantized to Q4_K before upload, matching the real model's expert
// tensor type -- everything else (b, ids, dst) stays F32 like the real MoE graph.
static std::vector<float> run(
        const std::vector<float> & weights_f32, int64_t n_embd, int64_t n_ff, int64_t n_as,
        const std::vector<float> & b_data, int64_t n_expert_used, int64_t n_tokens,
        const std::vector<int32_t> & ids_data) {
    ggml_init_params params = { /*.mem_size=*/ 16*1024*1024, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, n_embd, n_ff, n_as);
    ggml_set_name(weights, "weights");
    ggml_set_input(weights);
    ggml_tensor * b       = make_input(ctx, "b", n_embd, n_expert_used, n_tokens);
    ggml_tensor * ids     = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, n_tokens);
    ggml_set_name(ids, "ids");
    ggml_set_input(ids);

    ggml_tensor * dst = ggml_mul_mat_id(ctx, weights, b, ids);
    ggml_set_output(dst);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);

    std::vector<uint8_t> quantized(ggml_nbytes(weights));
    ggml_quantize_chunk(GGML_TYPE_Q4_K, weights_f32.data(), quantized.data(), 0, n_ff * n_as, n_embd, nullptr);
    ggml_backend_tensor_set(weights, quantized.data(), 0, quantized.size());
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));
    ggml_backend_tensor_set(ids, ids_data.data(), 0, ids_data.size() * sizeof(int32_t));

    ggml_backend_graph_compute(g_backend, gf);

    std::vector<float> out(ggml_nelements(dst));
    ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

int main() {
    ggml_backend_load_all();
    g_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!g_backend) {
        fprintf(stderr, "no CUDA backend available\n");
        return 1;
    }
    fprintf(stderr, "backend: %s\n", ggml_backend_name(g_backend));

    // Exact real-model dimensions: Qwen3.6-35B-A3B has n_embd=2048, expert_feed_forward_length=512,
    // n_expert=256, n_expert_used=8 (top-8 routing), decode batch n_tokens=1.
    const int64_t n_embd = 2048, n_ff = 512, n_expert_used = 8, n_tokens = 1;
    const int64_t n_as_full = 256; // real expert count
    const int64_t n_as_shrunk = 32; // our "cache_size"

    // Full tensor: expert e's block is entirely filled with value (e+1)*100.
    std::vector<float> full_weights(n_embd * n_ff * n_as_full);
    for (int64_t e = 0; e < n_as_full; e++) {
        for (int64_t i = 0; i < n_embd * n_ff; i++) {
            full_weights[e * n_embd * n_ff + i] = (float) (e + 1) * 100.0f;
        }
    }

    std::vector<float> b_data(n_embd * n_expert_used * n_tokens);
    for (size_t i = 0; i < b_data.size(); i++) {
        b_data[i] = 1.0f + 0.1f * (float) i; // arbitrary, non-trivial input activations
    }

    // Real ids: this token's real top-8 routing (arbitrary, spread across the 0-255 range
    // like a real router would produce).
    std::vector<int32_t> real_ids = { 229, 108, 5, 2, 194, 63, 150, 17 };

    auto full_out = run(full_weights, n_embd, n_ff, n_as_full, b_data, n_expert_used, n_tokens, real_ids);

    // Shrunk tensor: 32 slots. Put each real expert's data into slot == its position in
    // real_ids (0..7); slots 8..31 stay zeroed/unused -- not referenced by any remapped id.
    std::vector<float> shrunk_weights(n_embd * n_ff * n_as_shrunk, 0.0f);
    std::vector<int32_t> remapped_ids(real_ids.size());
    for (size_t i = 0; i < real_ids.size(); i++) {
        memcpy(&shrunk_weights[i * n_embd * n_ff], &full_weights[(size_t) real_ids[i] * n_embd * n_ff], n_embd * n_ff * sizeof(float));
        remapped_ids[i] = (int32_t) i;
    }

    auto shrunk_out = run(shrunk_weights, n_embd, n_ff, n_as_shrunk, b_data, n_expert_used, n_tokens, remapped_ids);

    bool match = full_out.size() == shrunk_out.size();
    int64_t first_mismatch_row = -1;
    if (match) {
        for (size_t i = 0; i < full_out.size(); i++) {
            float denom = std::max(std::fabs(full_out[i]), 1.0f);
            if (std::fabs(full_out[i] - shrunk_out[i]) / denom > 1e-3f) {
                match = false;
                first_mismatch_row = (int64_t) (i / n_ff);
                break;
            }
        }
    }
    printf("out size=%zu, per-expert-row size=%lld\n", full_out.size(), (long long) n_ff);
    for (size_t e = 0; e < real_ids.size(); e++) {
        printf("  expert idx %zu (real_id=%d): full[0]=%.4f shrunk[0]=%.4f\n",
                e, real_ids[e], full_out[e * n_ff], shrunk_out[e * n_ff]);
    }
    printf("MATCH: %s%s\n", match ? "YES" : "NO",
            match ? "" : (" (first mismatch at expert row " + std::to_string(first_mismatch_row) + ")").c_str());

    ggml_backend_free(g_backend);
    return match ? 0 : 2;
}
