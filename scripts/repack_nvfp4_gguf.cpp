#include "gguf.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <chrono>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [--layer <il>]\n", argv[0]);
        return 1;
    }
    const char * path = argv[1];
    int target_layer = -1;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            target_layer = atoi(argv[++i]);
        }
    }

    struct ggml_context * ctx = nullptr;
    struct gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ &ctx };
    struct gguf_context * gctx = gguf_init_from_file(path, params);
    if (!gctx) {
        fprintf(stderr, "failed to open %s\n", path);
        return 1;
    }

    int fd = open(path, O_RDWR);
    if (fd < 0) {
        perror("open O_RDWR");
        return 1;
    }

    size_t page_size = sysconf(_SC_PAGESIZE);
    size_t data_offset = gguf_get_data_offset(gctx);
    int64_t n_tensors = gguf_get_n_tensors(gctx);

    printf("GGUF opened: %s (data_offset=%zu, n_tensors=%lld)\n", path, data_offset, (long long)n_tensors);
    if (target_layer >= 0) {
        printf("Targeting single layer: %d\n", target_layer);
    } else {
        printf("Targeting ALL NVFP4 tensors across model\n");
    }

    size_t total_repacked_bytes = 0;
    int repacked_tensors = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (int64_t tid = 0; tid < n_tensors; ++tid) {
        const char * name = gguf_get_tensor_name(gctx, tid);
        enum ggml_type type = gguf_get_tensor_type(gctx, tid);
        if (type != GGML_TYPE_NVFP4) {
            continue;
        }

        if (target_layer >= 0) {
            char prefix[32];
            snprintf(prefix, sizeof(prefix), "blk.%d.", target_layer);
            if (strncmp(name, prefix, strlen(prefix)) != 0) {
                continue;
            }
        }

        size_t tensor_offset = gguf_get_tensor_offset(gctx, tid);
        size_t abs_offset = data_offset + tensor_offset;

        ggml_tensor * t = ggml_get_tensor(ctx, name);
        if (!t) {
            fprintf(stderr, "warning: tensor %s not in ggml_context\n", name);
            continue;
        }
        size_t tensor_bytes = ggml_nbytes(t);
        if (tensor_bytes % 36 != 0) {
            fprintf(stderr, "error: tensor %s size %zu not multiple of 36\n", name, tensor_bytes);
            continue;
        }

        size_t mmap_offset = abs_offset & ~(page_size - 1);
        size_t mmap_len = (abs_offset + tensor_bytes) - mmap_offset;

        uint8_t * ptr = (uint8_t *) mmap(NULL, mmap_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mmap_offset);
        if (ptr == MAP_FAILED) {
            perror("mmap");
            return 1;
        }

        uint8_t * data = ptr + (abs_offset - mmap_offset);
        size_t n_blocks = tensor_bytes / 36;

        #pragma omp parallel for schedule(static)
        for (size_t b = 0; b < n_blocks; ++b) {
            uint8_t * blk = data + b * 36;
            // bytes 0..3 are scales (UE4M3) -- keep untouched
            // bytes 4..35 are 4 sub-blocks of 8 bytes each
            for (int k = 0; k < 4; ++k) {
                uint8_t * p = blk + 4 + k * 8;
                uint8_t b0 = p[0], b1 = p[1], b2 = p[2], b3 = p[3];
                uint8_t b4 = p[4], b5 = p[5], b6 = p[6], b7 = p[7];

                p[0] = (b0 & 0x0F) | ((b4 & 0x0F) << 4);
                p[1] = (b0 >> 4)   | ((b4 >> 4) << 4);
                p[2] = (b1 & 0x0F) | ((b5 & 0x0F) << 4);
                p[3] = (b1 >> 4)   | ((b5 >> 4) << 4);
                p[4] = (b2 & 0x0F) | ((b6 & 0x0F) << 4);
                p[5] = (b2 >> 4)   | ((b6 >> 4) << 4);
                p[6] = (b3 & 0x0F) | ((b7 & 0x0F) << 4);
                p[7] = (b3 >> 4)   | ((b7 >> 4) << 4);
            }
        }

        msync(ptr, mmap_len, MS_SYNC);
        munmap(ptr, mmap_len);

        total_repacked_bytes += tensor_bytes;
        repacked_tensors++;
        printf("[repack] %-35s: %8.2f MB repacked\n", name, tensor_bytes / (1024.0 * 1024.0));
    }

    close(fd);
    gguf_free(gctx);
    ggml_free(ctx);

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    printf("Finished: %d tensors, %.2f GB repacked in %.2f seconds (%.2f GB/s)\n",
           repacked_tensors, total_repacked_bytes / (1024.0 * 1024.0 * 1024.0), sec,
           (total_repacked_bytes / (1024.0 * 1024.0 * 1024.0)) / sec);

    return 0;
}
