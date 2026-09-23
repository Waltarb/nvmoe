// Phase 1 probe: register a ggml_backend_sched eval-callback that stops only at the
// MoE router's "ffn_moe_topk-<layer>" tensor, reads back the selected expert ids, and
// times generation with vs without the callback registered, to confirm:
//   1. we can intercept the right tensor for this architecture (qwen35moe)
//   2. doing so does not regress throughput on the rest of the graph
#include "common.h"
#include "arg.h"
#include "sampling.h"
#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

struct probe_state {
    bool     active = false;
    long     hits = 0;
    long     tokens_read = 0;
    int      cache_size = 0; // nvmoe Phase 3 validation: >0 means remap real expert ids
                              // (0..n_expert_real) into cache-slot ids (0..cache_size)
                              // before letting mul_mat_id run against the shrunk tensor.
};

static bool eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<probe_state *>(user_data);
    if (!st->active) {
        return false;
    }
    if (strncmp(t->name, "ffn_moe_topk-", 13) != 0) {
        return false;
    }
    if (ask) {
        return true;
    }
    // t is now computed and synchronized: read back the selected expert ids
    const int64_t n = ggml_nelements(t);
    std::vector<int32_t> ids(n);
    ggml_backend_tensor_get(t, ids.data(), 0, n * sizeof(int32_t));
    st->hits++;
    st->tokens_read += n;
    if (st->hits <= 3) {
        fprintf(stderr, "[probe] %s: n_elements=%lld first_ids=[%d,%d,%d,...]\n",
                t->name, (long long) n, ids[0], n > 1 ? ids[1] : -1, n > 2 ? ids[2] : -1);
    }
    if (st->cache_size > 0) {
        // Placeholder remap for allocation-plumbing validation only: real cache logic
        // (later phase) replaces this with an actual slot table (LRU/pin lookup +
        // on-demand NVMe fill), not a naive modulo -- this just proves that writing
        // remapped, in-range ids back before resume lets mul_mat_id run without a
        // shrunk-tensor out-of-bounds access.
        for (auto & id : ids) {
            id = id % st->cache_size;
        }
        ggml_backend_tensor_set(t, ids.data(), 0, n * sizeof(int32_t));
        if (st->hits <= 3) {
            fprintf(stderr, "[probe] remapped to cache slots: [%d,%d,%d,...]\n",
                    ids[0], n > 1 ? ids[1] : -1, n > 2 ? ids[2] : -1);
        }
    }
    return true;
}

static double run_generation(common_params & params, bool with_callback, probe_state & st) {
    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return -1.0;
    }

    llama_context_params cparams = common_context_params_to_llama(params);
    st.active = with_callback || st.cache_size > 0; // remap is mandatory whenever the
                                                     // GPU tensor is shrunk -- real ids
                                                     // would run mul_mat_id out of bounds
    st.hits = 0;
    st.tokens_read = 0;
    if (st.active) {
        cparams.cb_eval = eval_cb;
        cparams.cb_eval_user_data = &st;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return -1.0;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = common_tokenize(vocab, params.prompt, true, true);

    common_sampler * smpl = common_sampler_init(model, params.sampling);

    auto t_start = std::chrono::steady_clock::now();

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    int n_decoded = 0;
    std::string out_text;

    while (n_decoded < params.n_predict) {
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed\n");
            break;
        }
        llama_token new_token = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, new_token, true);
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }
        out_text += common_token_to_piece(ctx, new_token);
        n_decoded++;
        batch = llama_batch_get_one(&new_token, 1);
    }

    auto t_end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t_end - t_start).count();

    fprintf(stderr, "[probe] with_callback=%d generated=%d tokens in %.2fs (%.2f tok/s), callback_hits=%ld, ids_read=%ld\n",
            with_callback, n_decoded, secs, n_decoded / secs, st.hits, st.tokens_read);
    fprintf(stderr, "[probe] output: %s\n", out_text.c_str());

    common_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return secs;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    probe_state st;
    const char * cache_env = getenv("NVMOE_CACHE_SIZE");
    st.cache_size = cache_env ? atoi(cache_env) : 0;

    if (st.cache_size > 0) {
        // Baseline (no callback) is not safe to run here: the GPU tensor is allocated
        // at cache_size experts, so real (unremapped) router ids would index it out of
        // bounds. Just run once, with the mandatory remap, to validate the pipeline.
        fprintf(stderr, "=== nvmoe cache_size=%d: single run WITH id-remap callback ===\n", st.cache_size);
        double t_cb = run_generation(params, true, st);
        fprintf(stderr, "\n=== summary ===\n");
        fprintf(stderr, "cache_size=%d run: %.2fs\n", st.cache_size, t_cb);
        return t_cb < 0 ? 1 : 0;
    }

    fprintf(stderr, "=== run 1: WITHOUT eval callback (baseline) ===\n");
    double t_base = run_generation(params, false, st);

    fprintf(stderr, "\n=== run 2: WITH eval callback on ffn_moe_topk-* ===\n");
    double t_cb = run_generation(params, true, st);

    fprintf(stderr, "\n=== summary ===\n");
    fprintf(stderr, "baseline: %.2fs, with_callback: %.2fs, delta: %.1f%%\n",
            t_base, t_cb, 100.0 * (t_cb - t_base) / t_base);

    return 0;
}
