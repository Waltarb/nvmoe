#include "gguf.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>

int main(int argc, char ** argv) {
    struct ggml_context * ctx = nullptr;
    struct gguf_init_params params = { true, &ctx };
    struct gguf_context * gctx = gguf_init_from_file(argv[1], params);
    if (!gctx) { fprintf(stderr, "open failed\n"); return 1; }
    const char * filter = argc > 2 ? argv[2] : "";
    int64_t n = gguf_get_n_tensors(gctx);
    for (int64_t i = 0; i < n; i++) {
        const char * name = gguf_get_tensor_name(gctx, i);
        if (!filter[0] || strstr(name, filter)) {
            printf("%s\n", name);
        }
    }
    return 0;
}
