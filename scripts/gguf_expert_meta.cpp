// Phase 2, step 1: parse MoE expert-tensor metadata directly out of a GGUF file,
// with no llama.cpp model loading involved -- just gguf.h + a no_alloc ggml_context.
// Computes the absolute file offset and per-expert byte stride (row_bytes) for the
// ffn_gate_exps / ffn_up_exps / ffn_down_exps tensors of every layer (this GGUF stores
// them as three separate on-disk tensors; llama.cpp fuses gate+up into one ggml_tensor
// only after loading), and checks whether offsets/strides are 4096-byte aligned
// (required for O_DIRECT in Phase 2 step 2).
#include "gguf.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <string>

struct expert_bank_info {
    std::string name;
    size_t      abs_offset;   // absolute byte offset of expert 0 within the file
    size_t      row_bytes;    // bytes per single expert (nb[2])
    int64_t     n_expert;     // ne[last dim]
    ggml_type   type;
};

static bool describe_tensor(struct gguf_context * gctx, struct ggml_context * ctx,
                             const char * name, expert_bank_info & out) {
    int64_t tid = gguf_find_tensor(gctx, name);
    if (tid < 0) {
        return false;
    }
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "found in gguf but not in ggml_context: %s\n", name);
        return false;
    }
    size_t data_offset   = gguf_get_data_offset(gctx);
    size_t tensor_offset = gguf_get_tensor_offset(gctx, tid);

    out.name       = name;
    out.abs_offset = data_offset + tensor_offset;
    out.row_bytes  = t->nb[2]; // stride to advance one expert along the last dim
    out.n_expert   = t->ne[2];
    out.type       = t->type;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [n_layers]\n", argv[0]);
        return 1;
    }
    const char * path = argv[1];
    int n_layers = argc > 2 ? atoi(argv[2]) : 40;

    struct ggml_context * ctx = nullptr;
    struct gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ &ctx };
    struct gguf_context * gctx = gguf_init_from_file(path, params);
    if (!gctx) {
        fprintf(stderr, "failed to open %s\n", path);
        return 1;
    }

    printf("gguf data section starts at offset %zu\n", gguf_get_data_offset(gctx));
    printf("alignment = %zu\n\n", gguf_get_alignment(gctx));
    fflush(stdout);

    int ok_layers = 0;
    int misaligned_offset = 0;
    int misaligned_row    = 0;
    int banks_checked     = 0;

    static const char * suffixes[3] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" };

    for (int il = 0; il < n_layers; il++) {
        expert_bank_info banks[3];
        bool all_ok = true;
        for (int b = 0; b < 3; b++) {
            char name[128];
            snprintf(name, sizeof(name), "blk.%d.%s.weight", il, suffixes[b]);
            if (!describe_tensor(gctx, ctx, name, banks[b])) {
                all_ok = false;
            }
        }

        if (!all_ok) {
            fprintf(stderr, "layer %d: one or more expert tensors missing\n", il);
            continue;
        }
        ok_layers++;

        for (auto & b : banks) {
            banks_checked++;
            bool off_aligned = (b.abs_offset % 4096) == 0;
            bool row_aligned = (b.row_bytes  % 4096) == 0;
            if (!off_aligned) misaligned_offset++;
            if (!row_aligned) misaligned_row++;
            if (il < 3 || !off_aligned || !row_aligned) {
                printf("layer %2d %-20s off=%12zu (align:%s) row_bytes=%8zu (align:%s) n_expert=%lld type=%s\n",
                       il, b.name.c_str(), b.abs_offset, off_aligned ? "OK" : "NO",
                       b.row_bytes, row_aligned ? "OK" : "NO", (long long) b.n_expert,
                       ggml_type_name(b.type));
            }
        }
        fflush(stdout);
    }

    printf("\nsummary: %d/%d layers fully described, %d misaligned offsets, %d misaligned row_bytes (out of %d banks checked)\n",
           ok_layers, n_layers, misaligned_offset, misaligned_row, banks_checked);

    ggml_free(ctx);
    gguf_free(gctx);
    return 0;
}
