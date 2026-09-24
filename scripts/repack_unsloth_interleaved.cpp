#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#pragma pack(push, 1)
struct layer_meta {
    uint64_t file_offset;     // offset of expert 0 for this layer
    uint32_t expert_size;     // total 4096-aligned bytes per expert
    uint32_t gate_row_bytes;  // gate bytes
    uint32_t up_row_bytes;    // up bytes
    uint32_t down_row_bytes;  // down bytes
    uint32_t pad_bytes;       // padding bytes
};

struct nvmoe_repack_header {
    uint64_t magic;         // 0x4e564d4f4552504b ("NVMOERPK")
    uint32_t version;       // 1
    uint32_t n_layers;      // 48
    uint32_t n_experts;     // 512
    uint32_t reserved;      // 0
    layer_meta layers[64];
    uint8_t padding[4096 - (24 + 64 * sizeof(layer_meta))];
};
#pragma pack(pop)

static_assert(sizeof(nvmoe_repack_header) == 4096, "Header must be exactly 4096 bytes");

struct shard_info {
    std::string path;
    int fd = -1;
    struct gguf_context * gctx = nullptr;
    struct ggml_context * meta_ctx = nullptr;
    size_t data_offset = 0;
};

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <first_shard.gguf> [output_interleaved.bin]\n", argv[0]);
        return 1;
    }

    const char * first_shard_path = argv[1];
    std::string out_path = (argc >= 3) ? argv[2] : "models/qwen-3.8-flash-unsloth/UD-Q2_K_XL/interleaved_experts.bin";

    fprintf(stderr, "=== NVMoE Contiguous Interleaved Expert Repacker ===\n");
    fprintf(stderr, "Input first shard: %s\n", first_shard_path);
    fprintf(stderr, "Output interleaved file: %s\n", out_path.c_str());

    // 1. Detect shards
    struct ggml_context * init_meta = nullptr;
    struct gguf_init_params init_gp = { /*.no_alloc =*/ true, /*.ctx =*/ &init_meta };
    struct gguf_context * init_gctx = gguf_init_from_file(first_shard_path, init_gp);
    if (!init_gctx) {
        fprintf(stderr, "FATAL: gguf_init_from_file failed for %s\n", first_shard_path);
        return 1;
    }

    int32_t split_count = 1;
    int kid_sc = gguf_find_key(init_gctx, "split.count");
    if (kid_sc >= 0) {
        split_count = gguf_get_val_u16(init_gctx, kid_sc);
    }
    int32_t split_no = 0;
    int kid_sn = gguf_find_key(init_gctx, "split.no");
    if (kid_sn >= 0) {
        split_no = gguf_get_val_u16(init_gctx, kid_sn);
    }

    std::vector<std::string> shard_paths;
    if (split_count > 1) {
        char prefix[1024];
        int ret = llama_split_prefix(prefix, sizeof(prefix), first_shard_path, split_no, split_count);
        if (ret <= 0) {
            fprintf(stderr, "FATAL: llama_split_prefix failed\n");
            return 1;
        }
        for (int i = 0; i < split_count; i++) {
            char spath[1024];
            llama_split_path(spath, sizeof(spath), prefix, i, split_count);
            shard_paths.push_back(spath);
        }
    } else {
        shard_paths.push_back(first_shard_path);
    }

    ggml_free(init_meta);
    gguf_free(init_gctx);

    std::vector<shard_info> shards(shard_paths.size());
    for (size_t i = 0; i < shard_paths.size(); i++) {
        shards[i].path = shard_paths[i];
        shards[i].fd = open(shards[i].path.c_str(), O_RDONLY);
        if (shards[i].fd < 0) {
            fprintf(stderr, "FATAL: failed to open shard %s\n", shards[i].path.c_str());
            return 1;
        }
        struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &shards[i].meta_ctx };
        shards[i].gctx = gguf_init_from_file(shards[i].path.c_str(), gp);
        if (!shards[i].gctx) {
            fprintf(stderr, "FATAL: gguf_init_from_file failed for %s\n", shards[i].path.c_str());
            return 1;
        }
        shards[i].data_offset = gguf_get_data_offset(shards[i].gctx);
        fprintf(stderr, "Loaded shard %zu: %s (%d tensors, data_offset=%zu)\n",
                i, shards[i].path.c_str(), gguf_get_n_tensors(shards[i].gctx), shards[i].data_offset);
    }

    const int n_layers = 48;
    const int n_experts = 512;

    nvmoe_repack_header hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = 0x4e564d4f4552504bULL; // "NVMOERPK"
    hdr.version = 1;
    hdr.n_layers = n_layers;
    hdr.n_experts = n_experts;

    auto find_tensor_in_shards = [&](const char * name, shard_info *& found_sh, int64_t & tid, ggml_tensor *& t) -> bool {
        for (auto & sh : shards) {
            tid = gguf_find_tensor(sh.gctx, name);
            if (tid >= 0) {
                t = ggml_get_tensor(sh.meta_ctx, name);
                if (t) {
                    found_sh = &sh;
                    return true;
                }
            }
        }
        return false;
    };

    uint64_t cur_file_offset = 4096; // expert 0 of layer 0 starts after 4096-byte header

    for (int il = 0; il < n_layers; il++) {
        char gname[128], uname[128], dname[128];
        snprintf(gname, sizeof(gname), "blk.%d.ffn_gate_exps.weight", il);
        snprintf(uname, sizeof(uname), "blk.%d.ffn_up_exps.weight",   il);
        snprintf(dname, sizeof(dname), "blk.%d.ffn_down_exps.weight", il);

        shard_info * sh_g = nullptr, * sh_u = nullptr, * sh_d = nullptr;
        int64_t tid_g = -1, tid_u = -1, tid_d = -1;
        ggml_tensor * t_g = nullptr, * t_u = nullptr, * t_d = nullptr;

        if (!find_tensor_in_shards(gname, sh_g, tid_g, t_g) ||
            !find_tensor_in_shards(uname, sh_u, tid_u, t_u) ||
            !find_tensor_in_shards(dname, sh_d, tid_d, t_d)) {
            fprintf(stderr, "FATAL: could not find all MoE tensors for layer %d\n", il);
            return 1;
        }

        uint32_t gate_row_bytes = (uint32_t) (ggml_nbytes(t_g) / n_experts);
        uint32_t up_row_bytes   = (uint32_t) (ggml_nbytes(t_u) / n_experts);
        uint32_t down_row_bytes = (uint32_t) (ggml_nbytes(t_d) / n_experts);

        uint32_t raw_bytes = gate_row_bytes + up_row_bytes + down_row_bytes;
        uint32_t pad_bytes = (4096 - (raw_bytes % 4096)) % 4096;
        uint32_t expert_size = raw_bytes + pad_bytes;

        hdr.layers[il].file_offset    = cur_file_offset;
        hdr.layers[il].expert_size    = expert_size;
        hdr.layers[il].gate_row_bytes = gate_row_bytes;
        hdr.layers[il].up_row_bytes   = up_row_bytes;
        hdr.layers[il].down_row_bytes = down_row_bytes;
        hdr.layers[il].pad_bytes      = pad_bytes;

        fprintf(stderr, "Layer %2d: offset=%12lu expert_size=%u (gate=%u up=%u down=%u pad=%u)\n",
                il, cur_file_offset, expert_size, gate_row_bytes, up_row_bytes, down_row_bytes, pad_bytes);

        cur_file_offset += (uint64_t) n_experts * expert_size;
    }

    fprintf(stderr, "Total output size will be: %.2f GiB (%lu bytes)\n",
            (double) cur_file_offset / (1024.0 * 1024.0 * 1024.0), cur_file_offset);

    int out_fd = open(out_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out_fd < 0) {
        fprintf(stderr, "FATAL: failed to create output file %s\n", out_path.c_str());
        return 1;
    }

    // Write header
    if (write(out_fd, &hdr, sizeof(hdr)) != sizeof(hdr)) {
        fprintf(stderr, "FATAL: failed to write header\n");
        return 1;
    }

    auto t_start = std::chrono::steady_clock::now();

    // 2. Interleave and write layers
    for (int il = 0; il < n_layers; il++) {
        auto t_lstart = std::chrono::steady_clock::now();
        char gname[128], uname[128], dname[128];
        snprintf(gname, sizeof(gname), "blk.%d.ffn_gate_exps.weight", il);
        snprintf(uname, sizeof(uname), "blk.%d.ffn_up_exps.weight",   il);
        snprintf(dname, sizeof(dname), "blk.%d.ffn_down_exps.weight", il);

        shard_info * sh_g = nullptr, * sh_u = nullptr, * sh_d = nullptr;
        int64_t tid_g = -1, tid_u = -1, tid_d = -1;
        ggml_tensor * t_g = nullptr, * t_u = nullptr, * t_d = nullptr;

        find_tensor_in_shards(gname, sh_g, tid_g, t_g);
        find_tensor_in_shards(uname, sh_u, tid_u, t_u);
        find_tensor_in_shards(dname, sh_d, tid_d, t_d);

        size_t off_g = sh_g->data_offset + gguf_get_tensor_offset(sh_g->gctx, tid_g);
        size_t off_u = sh_u->data_offset + gguf_get_tensor_offset(sh_u->gctx, tid_u);
        size_t off_d = sh_d->data_offset + gguf_get_tensor_offset(sh_d->gctx, tid_d);

        size_t bytes_g = ggml_nbytes(t_g);
        size_t bytes_u = ggml_nbytes(t_u);
        size_t bytes_d = ggml_nbytes(t_d);

        std::vector<uint8_t> buf_g(bytes_g);
        std::vector<uint8_t> buf_u(bytes_u);
        std::vector<uint8_t> buf_d(bytes_d);

        // Read all 3 tensors
        ssize_t rg = pread(sh_g->fd, buf_g.data(), bytes_g, off_g);
        ssize_t ru = pread(sh_u->fd, buf_u.data(), bytes_u, off_u);
        ssize_t rd = pread(sh_d->fd, buf_d.data(), bytes_d, off_d);

        if (rg != (ssize_t) bytes_g || ru != (ssize_t) bytes_u || rd != (ssize_t) bytes_d) {
            fprintf(stderr, "FATAL: pread failed on layer %d\n", il);
            return 1;
        }

        const auto & meta = hdr.layers[il];
        size_t layer_out_bytes = (size_t) n_experts * meta.expert_size;
        std::vector<uint8_t> layer_out(layer_out_bytes, 0);

        for (int e = 0; e < n_experts; e++) {
            uint8_t * dst = layer_out.data() + (size_t) e * meta.expert_size;
            const uint8_t * src_g = buf_g.data() + (size_t) e * meta.gate_row_bytes;
            const uint8_t * src_u = buf_u.data() + (size_t) e * meta.up_row_bytes;
            const uint8_t * src_d = buf_d.data() + (size_t) e * meta.down_row_bytes;

            memcpy(dst,                                src_g, meta.gate_row_bytes);
            memcpy(dst + meta.gate_row_bytes,          src_u, meta.up_row_bytes);
            memcpy(dst + meta.gate_row_bytes + meta.up_row_bytes, src_d, meta.down_row_bytes);
            // padding is already 0
        }

        // Write layer output
        size_t written = 0;
        while (written < layer_out_bytes) {
            ssize_t w = write(out_fd, layer_out.data() + written, layer_out_bytes - written);
            if (w <= 0) {
                fprintf(stderr, "FATAL: write failed on layer %d\n", il);
                return 1;
            }
            written += w;
        }

        auto t_lend = std::chrono::steady_clock::now();
        double lsec = std::chrono::duration<double>(t_lend - t_lstart).count();
        double mb_s = ((double) layer_out_bytes / (1024.0 * 1024.0)) / lsec;
        fprintf(stderr, "[repack] Layer %2d / %d completed: %.1f MB written in %.2f s (%.1f MB/s)\n",
                il + 1, n_layers, (double) layer_out_bytes / (1024.0 * 1024.0), lsec, mb_s);
    }

    fprintf(stderr, "Flushing to disk (fdatasync)...\n");
    fdatasync(out_fd);
    close(out_fd);

    for (auto & sh : shards) {
        if (sh.fd >= 0) close(sh.fd);
        if (sh.meta_ctx) ggml_free(sh.meta_ctx);
        if (sh.gctx) gguf_free(sh.gctx);
    }

    auto t_end = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();
    fprintf(stderr, "=== Repacking Complete! Written in %.2f seconds ===\n", total_sec);

    return 0;
}
