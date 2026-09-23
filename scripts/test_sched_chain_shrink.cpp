// Isolated correctness test #2, no llama.cpp model loading:
// Chains TWO shrunk ggml_mul_mat_id calls back-to-back under ggml_backend_sched,
// with the full 3-touch eval callback pattern matching moe_cache_probe.cpp:
//   1. snapshot probs (probs-38, probs-39)
//   2. read + remap + write non-contiguous ids (ids-38, ids-39, view with full stride)
//   3. overwrite get_rows weights (weights-38, weights-39) and use them to scale dst
//
// Tests both single-token (decode) and multi-token (prefill, n_tokens > 1) with
// non-contiguous row stride on the ids tensor to guard against stride regressions.
#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <string>
#include <unordered_map>
#include <algorithm>

static ggml_backend_t g_backend = nullptr;

struct layer_remap {
    std::unordered_map<int32_t, int32_t> slot_of_real;
    std::vector<float>   last_probs;
    std::vector<int32_t> last_real_ids;
    int64_t n_expert_used = 8;
    int64_t n_tokens      = 1;
    int64_t n_as_full     = 256;
    int hits_probs = 0;
    int hits_ids = 0;
    int hits_weights = 0;
};

struct cb_state {
    layer_remap l38;
    layer_remap l39;
};

static bool g_cb_disabled = false; // NVMOE_TEST_NO_CB control run

static bool eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    if (g_cb_disabled) {
        return false;
    }
    auto * st = static_cast<cb_state *>(user_data);

    const bool is_probs38   = strcmp(t->name, "probs-38") == 0;
    const bool is_probs39   = strcmp(t->name, "probs-39") == 0;
    const bool is_ids38     = strcmp(t->name, "ids-38") == 0;
    const bool is_ids39     = strcmp(t->name, "ids-39") == 0;
    const bool is_weights38 = strcmp(t->name, "weights-38") == 0;
    const bool is_weights39 = strcmp(t->name, "weights-39") == 0;

    if (!is_probs38 && !is_probs39 && !is_ids38 && !is_ids39 && !is_weights38 && !is_weights39) {
        return false;
    }
    if (ask) {
        return true;
    }

    // Touch 1: probs snapshot (read-only)
    if (is_probs38 || is_probs39) {
        layer_remap & lr = is_probs38 ? st->l38 : st->l39;
        const int64_t n = ggml_nelements(t);
        lr.last_probs.resize(n);
        ggml_backend_tensor_get(t, lr.last_probs.data(), 0, n * sizeof(float));
        lr.hits_probs++;
        return true;
    }

    // Touch 2: ids remap (row-by-row respecting t->nb[1] for non-contiguous views)
    if (is_ids38 || is_ids39) {
        layer_remap & lr = is_ids38 ? st->l38 : st->l39;
        const int64_t n_expert_used = t->ne[0];
        const int64_t n_tokens = t->ne[1];
        const int64_t n = n_expert_used * n_tokens;
        std::vector<int32_t> ids(n);

        for (int64_t row = 0; row < n_tokens; row++) {
            ggml_backend_tensor_get(t, ids.data() + row * n_expert_used, row * t->nb[1], n_expert_used * sizeof(int32_t));
        }

        lr.last_real_ids = ids;
        lr.n_expert_used = n_expert_used;
        lr.n_tokens      = n_tokens;

        for (auto & id : ids) {
            auto it = lr.slot_of_real.find(id);
            if (it == lr.slot_of_real.end()) {
                fprintf(stderr, "[BUG] %s: real id %d has no slot mapping\n", t->name, id);
                continue;
            }
            id = it->second;
        }

        for (int64_t row = 0; row < n_tokens; row++) {
            ggml_backend_tensor_set(t, ids.data() + row * n_expert_used, row * t->nb[1], n_expert_used * sizeof(int32_t));
        }

        lr.hits_ids++;
        return true;
    }

    // Touch 3: weights fix (recomputing correct gating values from probs snapshot + saved real ids)
    if (is_weights38 || is_weights39) {
        layer_remap & lr = is_weights38 ? st->l38 : st->l39;
        const int64_t n = lr.n_expert_used * lr.n_tokens;
        std::vector<float> correct(n);
        for (int64_t token = 0; token < lr.n_tokens; token++) {
            for (int64_t exp = 0; exp < lr.n_expert_used; exp++) {
                int64_t idx = token * lr.n_expert_used + exp;
                int32_t real_id = lr.last_real_ids[idx];
                correct[idx] = lr.last_probs[token * lr.n_as_full + real_id];
            }
        }
        ggml_backend_tensor_set(t, correct.data(), 0, n * sizeof(float));
        lr.hits_weights++;
        return true;
    }

    return true;
}

int main() {
    g_cb_disabled = getenv("NVMOE_TEST_NO_CB") != nullptr;
    if (g_cb_disabled) {
        fprintf(stderr, "*** CONTROL RUN: eval_cb disabled, uploading pre-remapped slot ids directly ***\n");
    }
    ggml_backend_load_all();
    g_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!g_backend) {
        fprintf(stderr, "no CUDA backend available\n");
        return 1;
    }
    fprintf(stderr, "backend: %s\n", ggml_backend_name(g_backend));

    // Dimensions: n_embd=2048, n_ff=512, n_as_full=256, n_as_shrunk=32, n_expert_used=8.
    // Multi-token n_tokens=4 to thoroughly test prefill non-contiguous row stride.
    const int64_t n_embd = 2048, n_ff = 512, n_expert_used = 8, n_tokens = 4;
    const int64_t n_as_full = 256, n_as_shrunk = 32;

    std::vector<int32_t> real_ids38_pool = { 229, 108, 5, 2, 194, 63, 150, 17, 33, 44, 55, 66, 77, 88, 99, 111 };
    std::vector<int32_t> real_ids39_pool = { 11, 240, 88, 199, 3, 77, 160, 45, 12, 23, 34, 45, 56, 67, 78, 89 };

    cb_state st;
    st.l38.n_as_full = n_as_full;
    st.l39.n_as_full = n_as_full;
    for (size_t i = 0; i < real_ids38_pool.size(); i++) st.l38.slot_of_real[real_ids38_pool[i]] = (int32_t) i;
    for (size_t i = 0; i < real_ids39_pool.size(); i++) st.l39.slot_of_real[real_ids39_pool[i]] = (int32_t) i;

    auto make_shrunk_weights = [&](const std::vector<int32_t> & pool, float base) {
        std::vector<float> w(n_embd * n_ff * n_as_shrunk, 0.0f);
        for (size_t slot = 0; slot < pool.size(); slot++) {
            float val = base + 0.001f * (float) pool[slot];
            for (int64_t i = 0; i < n_embd * n_ff; i++) {
                w[slot * n_embd * n_ff + i] = val;
            }
        }
        return w;
    };
    std::vector<float> weights1_f32 = make_shrunk_weights(real_ids38_pool, 1.0f);
    std::vector<float> weights2_f32 = make_shrunk_weights(real_ids39_pool, 2.0f);

    std::vector<float> b1_data(n_embd * n_expert_used * n_tokens);
    for (size_t i = 0; i < b1_data.size(); i++) b1_data[i] = 0.0001f * (float) (i % 50);

    // Initial simulated probability distributions
    std::vector<float> probs1_data(n_as_full * n_tokens, 0.001f);
    std::vector<float> probs2_data(n_as_full * n_tokens, 0.001f);
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t e = 0; e < n_expert_used; e++) {
            int32_t id1 = real_ids38_pool[(t * 2 + e) % real_ids38_pool.size()];
            int32_t id2 = real_ids39_pool[(t * 2 + e) % real_ids39_pool.size()];
            probs1_data[t * n_as_full + id1] = 0.1f + 0.01f * (float) e;
            probs2_data[t * n_as_full + id2] = 0.1f + 0.01f * (float) e;
        }
    }

    // Full argsort table (shape [n_as_full, n_tokens]), with top-k placed in the first n_expert_used columns
    std::vector<int32_t> argsort1_data(n_as_full * n_tokens, 0);
    std::vector<int32_t> argsort2_data(n_as_full * n_tokens, 0);
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t e = 0; e < n_expert_used; e++) {
            int32_t id1 = real_ids38_pool[(t * 2 + e) % real_ids38_pool.size()];
            int32_t id2 = real_ids39_pool[(t * 2 + e) % real_ids39_pool.size()];
            argsort1_data[t * n_as_full + e] = g_cb_disabled ? st.l38.slot_of_real[id1] : id1;
            argsort2_data[t * n_as_full + e] = g_cb_disabled ? st.l39.slot_of_real[id2] : id2;
        }
    }

    // --- Build graph ---
    ggml_init_params params = { /*.mem_size=*/ 64*1024*1024, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    ggml_context * ctx = ggml_init(params);

    // Layer 38
    ggml_tensor * weights1 = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, n_embd, n_ff, n_as_shrunk);
    ggml_set_name(weights1, "weights1"); ggml_set_input(weights1);
    ggml_tensor * b1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_expert_used, n_tokens);
    ggml_set_name(b1, "b1"); ggml_set_input(b1);

    // Touch 1: probs node
    ggml_tensor * probs1_raw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_as_full, n_tokens);
    ggml_set_name(probs1_raw, "probs1_raw"); ggml_set_input(probs1_raw);
    ggml_tensor * probs1 = ggml_cont(ctx, probs1_raw);
    ggml_set_name(probs1, "probs-38");

    // Touch 2: ids view from full argsort (exercises non-contiguous stride nb[1] = n_as_full * 4)
    ggml_tensor * argsort1_raw = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_as_full, n_tokens);
    ggml_set_name(argsort1_raw, "argsort1_raw"); ggml_set_input(argsort1_raw);
    ggml_tensor * ids1 = ggml_view_2d(ctx, argsort1_raw, n_expert_used, n_tokens, argsort1_raw->nb[1], 0);
    ggml_set_name(ids1, "ids-38");

    // Touch 3: weights node from get_rows
    ggml_tensor * probs1_3d = ggml_reshape_3d(ctx, probs1, 1, n_as_full, n_tokens);
    ggml_tensor * weights1_node = ggml_get_rows(ctx, probs1_3d, ids1);
    ggml_set_name(weights1_node, "weights-38");

    ggml_tensor * dst1 = ggml_mul_mat_id(ctx, weights1, b1, ids1); // [n_ff, n_expert_used, n_tokens]
    ggml_set_name(dst1, "dst1");

    // Scale dst1 by weights1_node
    ggml_tensor * dst1_weighted = ggml_mul(ctx, dst1, weights1_node);
    ggml_set_name(dst1_weighted, "dst1_weighted");

    // Connector between layers
    ggml_tensor * dst1_flat = ggml_reshape_2d(ctx, dst1_weighted, n_ff * n_expert_used, n_tokens);
    static_assert(4096 >= 2048, "connector view needs n_ff*n_expert_used >= n_embd");
    ggml_tensor * hidden_view = ggml_view_2d(ctx, dst1_flat, n_embd, n_tokens, dst1_flat->nb[1], 0);
    ggml_tensor * hidden = ggml_cont(ctx, hidden_view);
    ggml_set_name(hidden, "hidden");
    ggml_tensor * hidden3d = ggml_reshape_3d(ctx, hidden, n_embd, 1, n_tokens);
    ggml_tensor * b2_template = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_expert_used, n_tokens);
    ggml_tensor * b2 = ggml_repeat(ctx, hidden3d, b2_template);

    // Layer 39
    ggml_tensor * weights2 = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, n_embd, n_ff, n_as_shrunk);
    ggml_set_name(weights2, "weights2"); ggml_set_input(weights2);

    ggml_tensor * probs2_raw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_as_full, n_tokens);
    ggml_set_name(probs2_raw, "probs2_raw"); ggml_set_input(probs2_raw);
    ggml_tensor * probs2 = ggml_cont(ctx, probs2_raw);
    ggml_set_name(probs2, "probs-39");

    ggml_tensor * argsort2_raw = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_as_full, n_tokens);
    ggml_set_name(argsort2_raw, "argsort2_raw"); ggml_set_input(argsort2_raw);
    ggml_tensor * ids2 = ggml_view_2d(ctx, argsort2_raw, n_expert_used, n_tokens, argsort2_raw->nb[1], 0);
    ggml_set_name(ids2, "ids-39");

    ggml_tensor * probs2_3d = ggml_reshape_3d(ctx, probs2, 1, n_as_full, n_tokens);
    ggml_tensor * weights2_node = ggml_get_rows(ctx, probs2_3d, ids2);
    ggml_set_name(weights2_node, "weights-39");

    ggml_tensor * dst2 = ggml_mul_mat_id(ctx, weights2, b2, ids2); // [n_ff, n_expert_used, n_tokens]
    ggml_set_name(dst2, "dst2");

    ggml_tensor * dst2_weighted = ggml_mul(ctx, dst2, weights2_node);
    ggml_set_name(dst2_weighted, "dst2_weighted");
    ggml_set_output(dst2_weighted);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst2_weighted);

    // --- sched ---
    ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_cpu_set_n_threads(backend_cpu, 4);
    ggml_backend_t backends[2] = { g_backend, backend_cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    ggml_backend_sched_set_eval_callback(sched, eval_cb, &st);

    ggml_set_output(dst1_flat);
    ggml_tensor * all_nodes[] = {
        weights1, b1, probs1_raw, probs1, argsort1_raw, ids1, probs1_3d, weights1_node, dst1, dst1_weighted,
        dst1_flat, hidden, hidden3d, b2, weights2, probs2_raw, probs2, argsort2_raw, ids2, probs2_3d,
        weights2_node, dst2, dst2_weighted
    };
    for (ggml_tensor * t : all_nodes) {
        ggml_backend_sched_set_tensor_backend(sched, t, g_backend);
    }

    if (!ggml_backend_sched_reserve(sched, gf)) {
        fprintf(stderr, "sched_reserve failed\n");
        return 1;
    }
    for (ggml_tensor * t : all_nodes) {
        ggml_backend_sched_set_tensor_backend(sched, t, g_backend);
    }
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        fprintf(stderr, "sched_alloc_graph failed\n");
        return 1;
    }

    // Upload data
    std::vector<uint8_t> q1(ggml_nbytes(weights1)), q2(ggml_nbytes(weights2));
    ggml_quantize_chunk(GGML_TYPE_Q4_K, weights1_f32.data(), q1.data(), 0, n_ff * n_as_shrunk, n_embd, nullptr);
    ggml_quantize_chunk(GGML_TYPE_Q4_K, weights2_f32.data(), q2.data(), 0, n_ff * n_as_shrunk, n_embd, nullptr);
    ggml_backend_tensor_set(weights1, q1.data(), 0, q1.size());
    ggml_backend_tensor_set(weights2, q2.data(), 0, q2.size());
    ggml_backend_tensor_set(b1, b1_data.data(), 0, b1_data.size() * sizeof(float));
    ggml_backend_tensor_set(probs1_raw, probs1_data.data(), 0, probs1_data.size() * sizeof(float));
    ggml_backend_tensor_set(probs2_raw, probs2_data.data(), 0, probs2_data.size() * sizeof(float));
    ggml_backend_tensor_set(argsort1_raw, argsort1_data.data(), 0, argsort1_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(argsort2_raw, argsort2_data.data(), 0, argsort2_data.size() * sizeof(int32_t));

    ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
    fprintf(stderr, "compute status: %d\n", (int) status);

    std::vector<float> out(ggml_nelements(dst2_weighted));
    ggml_backend_tensor_get(dst2_weighted, out.data(), 0, out.size() * sizeof(float));

    int nan_count = 0;
    for (float v : out) if (std::isnan(v)) nan_count++;
    printf("dst2_weighted elements: %zu, NaN count: %d\n", out.size(), nan_count);
    printf("dst2_weighted sample: [0]=%.6f [%zu]=%.6f [%zu]=%.6f\n",
           out[0], out.size()/2, out[out.size()/2], out.size()-1, out.back());
    printf("l38 hits: probs=%d ids=%d weights=%d | l39 hits: probs=%d ids=%d weights=%d\n",
           st.l38.hits_probs, st.l38.hits_ids, st.l38.hits_weights,
           st.l39.hits_probs, st.l39.hits_ids, st.l39.hits_weights);
    printf("RESULT: %s\n", nan_count > 0 ? "NAN_REPRODUCED" : "CLEAN");

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(g_backend);
    return nan_count > 0 ? 2 : 0;
}
