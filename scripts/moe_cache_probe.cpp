// Phase 3+4: three-tier expert cache (GPU slots <- host-RAM SLRU <- NVMe), on top of
// the validated shrunk-tensor allocation (create_tensor_reduced) and id-remap mechanism
// (moe_probe.cpp). GPU tier: round-robin-evicted slot table per layer, filled from the
// host tier (never straight from NVMe -- that shortcut was only for the earlier Phase 3
// crash/plumbing validation). Host tier: nvmoe's SegmentedHostLru port
// (segmented_host_lru.hpp) -- probation/protected 2Q policy, promote-on-2nd-hit, drain
// probation before protected -- backing a plain byte-array slot pool per bank per layer.
// A host miss fetches the real expert's gate/up/down rows straight off the GGUF file via
// the Phase 2-verified O_DIRECT aligned read.
//
// NOTE: end-to-end numerical output remains an open, separately-tracked issue (see the
// plan doc's Phase 3 section) even for the simpler GPU<->NVMe-only version of this same
// fill mechanism -- exhaustively isolated there (not CUDA graphs, not gate/up fusion, not
// self-eviction, not uninitialized memory, writes/reads independently verified correct).
// This host-tier layer is orthogonal infrastructure (a byte-cache in front of NVMe) and
// doesn't depend on that bug being fixed first; its own correctness (right bytes reaching
// the right GPU slot) is verified the same way -- readback diffed against source.
#include "common.h"
#include "arg.h"
#include "sampling.h"
#include "llama.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "segmented_host_lru.hpp"

#include <liburing.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <chrono>
#include <vector>
#include <unordered_set>
#include <fcntl.h>
#include <unistd.h>
#include <cmath>
#include <algorithm>

static constexpr size_t ALIGN = 4096;
static inline size_t align_down(size_t x) { return x - (x % ALIGN); }
static inline size_t align_up(size_t x)   { return ((x + ALIGN - 1) / ALIGN) * ALIGN; }

struct read_job {
    size_t offset;
    size_t len;
    uint8_t * out;
};

// Verified in Phase 2 (scripts/expert_reader.cpp): GGUF's own tensor alignment (32B
// default) does not guarantee O_DIRECT's 4096B requirement, so round the read window
// out to the nearest boundaries and slice the exact bytes needed out of it.
static bool read_odirect(int fd_direct, size_t logical_offset, size_t len, uint8_t * out) {
    size_t aligned_off = align_down(logical_offset);
    size_t front_pad   = logical_offset - aligned_off;
    size_t aligned_len = align_up(front_pad + len);

    uint8_t * buf = nullptr;
    if (posix_memalign((void **) &buf, ALIGN, aligned_len) != 0) {
        return false;
    }
    ssize_t n = pread(fd_direct, buf, aligned_len, aligned_off);
    bool ok = (n >= 0 && (size_t) n >= front_pad + len);
    if (ok) {
        memcpy(out, buf + front_pad, len);
    }
    free(buf);
    return ok;
}

struct uring_reader {
    io_uring ring;
    bool initialized = false;
    std::vector<char *> staging_buffers;
    size_t max_batch = 128;
    size_t staging_buf_size = 0;

    void init(size_t max_b, size_t row_bytes) {
        max_batch = max_b;
        staging_buf_size = align_up(row_bytes + 8192);
        if (io_uring_queue_init(max_batch, &ring, 0) == 0) {
            initialized = true;
            staging_buffers.resize(max_batch);
            for (size_t i = 0; i < max_batch; i++) {
                if (posix_memalign((void **) &staging_buffers[i], ALIGN, staging_buf_size) != 0) {
                    initialized = false;
                    break;
                }
            }
        }
    }

    ~uring_reader() {
        if (initialized) {
            for (char * buf : staging_buffers) free(buf);
            io_uring_queue_exit(&ring);
        }
    }
};

// One routed-expert weight bank (gate_exps, up_exps, or down_exps) for one layer:
// where its real per-expert rows live on disk, and the GPU-resident (shrunk) tensor
// they get copied into.
struct expert_bank {
    ggml_tensor * gpu_tensor = nullptr; // shape [.., .., cache_size] if shrunk, else full
    size_t        base_offset = 0;      // file offset of real expert 0's row
    size_t        row_bytes   = 0;      // bytes per real expert (== nb[2] of the real tensor)
};

// Host-RAM staging pool for one bank (gate/up/down) of one layer: a flat array of
// `host_cache_size` slots, each `row_bytes` long, holding real expert weight bytes.
// Uses pinned CUDA host memory (cudaHostAlloc) so GPU transfers achieve full PCIe line rate.
struct host_bank_pool {
    uint8_t * data = nullptr;
    size_t total_bytes = 0;
    size_t row_bytes = 0;
    bool is_pinned = false;

    void init(size_t rows, size_t r_bytes) {
        row_bytes = r_bytes;
        total_bytes = rows * row_bytes;
        if (total_bytes == 0) return;
        cudaError_t err = cudaHostAlloc((void **) &data, total_bytes, cudaHostAllocDefault);
        if (err == cudaSuccess) {
            is_pinned = true;
        } else {
            posix_memalign((void **) &data, 4096, total_bytes);
            is_pinned = false;
        }
    }

    ~host_bank_pool() {
        if (data) {
            if (is_pinned) cudaFreeHost(data);
            else free(data);
            data = nullptr;
        }
    }

    host_bank_pool() = default;
    host_bank_pool(const host_bank_pool &) = delete;
    host_bank_pool & operator=(const host_bank_pool &) = delete;
    host_bank_pool(host_bank_pool && o) noexcept {
        data = o.data; o.data = nullptr;
        total_bytes = o.total_bytes;
        row_bytes = o.row_bytes;
        is_pinned = o.is_pinned;
    }
    host_bank_pool & operator=(host_bank_pool && o) noexcept {
        if (this != &o) {
            if (data) { if (is_pinned) cudaFreeHost(data); else free(data); }
            data = o.data; o.data = nullptr;
            total_bytes = o.total_bytes;
            row_bytes = o.row_bytes;
            is_pinned = o.is_pinned;
        }
        return *this;
    }

    uint8_t * slot_ptr(int32_t slot) { return data + (size_t) slot * row_bytes; }
};

struct layer_cache {
    int64_t n_expert_real = 0;
    int64_t cache_size    = 0; // == gpu_tensor->ne[2] if shrunk; == n_expert_real if not
    bool fused_gate_up = false;
    expert_bank gate_up;
    expert_bank gate, up, down;
    expert_bank gate_up_scale;
    expert_bank down_scale;
    std::vector<float> host_gate_up_scale; // [n_expert_real]
    std::vector<float> host_down_scale;    // [n_expert_real]

    std::vector<int32_t> slot_of_real; // [n_expert_real], -1 = not resident (GPU tier)
    std::vector<int32_t> real_in_slot; // [cache_size],    -1 = empty slot   (GPU tier)
    std::vector<uint32_t> slot_last_used; // [cache_size], step when slot was last accessed
    uint32_t current_step = 0;
    int next_evict = 0;

    // Host tier: nvmoe's SegmentedHostLru policy over `host_cache_size` shared slots.
    int64_t host_cache_size = 0;
    SegmentedHostLru<int32_t> host_lru;
    std::vector<int32_t> host_slot_of_real; // [n_expert_real],    -1 = not resident
    std::vector<int32_t> host_real_in_slot; // [host_cache_size],  -1 = empty slot
    int32_t host_next_free = 0;             // slots < this have been used at least once
    host_bank_pool host_gate_up;
    host_bank_pool host_gate, host_up, host_down;

    // Phase 4 calibration: per-expert decode-hit counters (every time this layer needed
    // this real expert, hit or miss), and the set of experts prewarm-pinned into the host
    // tier at startup (discarded from host_lru so pop_lru() never touches them -- same
    // "pin by removing from LRU tracking, keep the slot mapping" trick as nvmoe).
    std::vector<int64_t> access_count; // [n_expert_real]
    std::unordered_set<int32_t> pinned;
    int32_t gpu_pinned_k = 0; // top-K experts permanently pinned in GPU VRAM slots 0..gpu_pinned_k-1
    bool shrunk() const { return cache_size > 0 && cache_size < n_expert_real; }
};

struct cache_state {
    bool     active = false;
    bool     is_decode_phase = false;
    int      cache_size_cfg = 0;
    int      host_cache_size_cfg = 0;
    int      fd_direct = -1;
    long     hits = 0, misses = 0, callback_hits = 0;
    long     host_hits = 0, host_misses = 0;
    long     decode_hits = 0, decode_misses = 0;
    long     decode_host_hits = 0, decode_host_misses = 0;
    long     decode_pruned_nvme = 0;
    float    prune_nvme_thresh = 0.0f;
    int      prune_min_keep = 6;
    float    prune_min_mass = 0.95f;
    std::vector<layer_cache> layers; // indexed by layer id parsed from tensor name

    uring_reader uring;
    cudaStream_t h2d_stream = nullptr;
    cudaEvent_t  h2d_event  = nullptr;
    int32_t *    pinned_ids = nullptr;
    float *      pinned_weights = nullptr;

    cache_state() {
        int least_pri = 0, greatest_pri = 0;
        cudaDeviceGetStreamPriorityRange(&least_pri, &greatest_pri);
        cudaStreamCreateWithPriority(&h2d_stream, cudaStreamNonBlocking, greatest_pri);
        cudaEventCreateWithFlags(&h2d_event, cudaEventDisableTiming);
        cudaHostAlloc((void **)&pinned_ids, 4096 * sizeof(int32_t), cudaHostAllocDefault);
        cudaHostAlloc((void **)&pinned_weights, 4096 * sizeof(float), cudaHostAllocDefault);
    }
    ~cache_state() {
        if (pinned_weights) {
            cudaFreeHost(pinned_weights);
            pinned_weights = nullptr;
        }
        if (pinned_ids) {
            cudaFreeHost(pinned_ids);
            pinned_ids = nullptr;
        }
        if (h2d_event) {
            cudaEventDestroy(h2d_event);
            h2d_event = nullptr;
        }
        if (h2d_stream) {
            cudaStreamDestroy(h2d_stream);
            h2d_stream = nullptr;
        }
        if (fd_direct >= 0) {
            close(fd_direct);
            fd_direct = -1;
        }
    }
};

static int parse_layer_id(const char * name) {
    // name looks like "ffn_moe_topk-<N>"
    const char * dash = strrchr(name, '-');
    return dash ? atoi(dash + 1) : -1;
}

static int g_verify_count = 0;

static bool read_jobs_batch(cache_state & st, const std::vector<read_job> & jobs) {
    if (jobs.empty()) return true;
    if (!st.uring.initialized) {
        for (const auto & j : jobs) {
            if (!read_odirect(st.fd_direct, j.offset, j.len, j.out)) return false;
        }
        return true;
    }

    size_t total = jobs.size();
    for (size_t start = 0; start < total; start += st.uring.max_batch) {
        size_t batch_size = std::min(total - start, st.uring.max_batch);
        for (size_t i = 0; i < batch_size; i++) {
            const auto & j = jobs[start + i];
            size_t aligned_off = align_down(j.offset);
            size_t front_pad   = j.offset - aligned_off;
            size_t aligned_len  = align_up(front_pad + j.len);

            io_uring_sqe * sqe = io_uring_get_sqe(&st.uring.ring);
            io_uring_prep_read(sqe, st.fd_direct, st.uring.staging_buffers[i], aligned_len, aligned_off);
            io_uring_sqe_set_data64(sqe, i);
        }

        int ret = io_uring_submit_and_wait(&st.uring.ring, batch_size);
        if (ret < 0) return false;

        for (size_t i = 0; i < batch_size; i++) {
            io_uring_cqe * cqe = nullptr;
            int wret = io_uring_wait_cqe(&st.uring.ring, &cqe);
            if (wret < 0 || !cqe) return false;
            uint64_t idx = io_uring_cqe_get_data64(cqe);
            int res = cqe->res;
            io_uring_cqe_seen(&st.uring.ring, cqe);

            const auto & j = jobs[start + idx];
            size_t aligned_off = align_down(j.offset);
            size_t front_pad   = j.offset - aligned_off;
            if (res < 0 || (size_t) res < front_pad + j.len) return false;
            memcpy(j.out, st.uring.staging_buffers[idx] + front_pad, j.len);
        }
    }
    return true;
}

static void gpu_fill_from_host_async(cache_state & st, layer_cache & lc, expert_bank & bank, host_bank_pool & host_pool, int32_t gpu_slot, int32_t host_slot, int32_t real_id) {
    if (host_slot < 0 || (size_t) host_slot >= (size_t) lc.host_cache_size) {
        fprintf(stderr, "[cache] FATAL: invalid host_slot=%d for real_id=%d (host_cache_size=%lld)\n",
                host_slot, real_id, (long long) lc.host_cache_size);
        abort();
    }
    if (gpu_slot < 0 || (size_t) gpu_slot >= (size_t) lc.cache_size) {
        fprintf(stderr, "[cache] FATAL: invalid gpu_slot=%d for real_id=%d (cache_size=%lld)\n",
                gpu_slot, real_id, (long long) lc.cache_size);
        abort();
    }
    uint8_t * src = host_pool.slot_ptr(host_slot);
    cudaMemcpyAsync((char *) bank.gpu_tensor->data + (size_t) gpu_slot * bank.row_bytes,
                    src, bank.row_bytes, cudaMemcpyHostToDevice, st.h2d_stream);

    static const bool verify_readback = getenv("NVMOE_VERIFY_READBACK") != nullptr;
    if (verify_readback && g_verify_count < 5) {
        cudaStreamSynchronize(st.h2d_stream);
        std::vector<uint8_t> readback(bank.row_bytes);
        cudaMemcpy(readback.data(), (char *) bank.gpu_tensor->data + (size_t) gpu_slot * bank.row_bytes,
                   bank.row_bytes, cudaMemcpyDeviceToHost);
        bool match = memcmp(readback.data(), src, bank.row_bytes) == 0;
        fprintf(stderr, "[cache-verify] %s real_id=%d gpu_slot=%d host_slot=%d row_bytes=%zu readback_match=%s (pinned=%s)\n",
                ggml_get_name(bank.gpu_tensor), real_id, gpu_slot, host_slot, bank.row_bytes, match ? "YES" : "NO",
                host_pool.is_pinned ? "YES" : "NO");
        g_verify_count++;
    }
}

static std::unordered_set<std::string> g_seen_names;

static void print_tensor_stats(const char * label, ggml_tensor * t) {
    if (!t) return;
    int64_t n = ggml_nelements(t);
    int64_t sample_n = std::min<int64_t>(n, 2560);
    std::vector<float> f_data(sample_n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, f_data.data(), 0, sample_n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h_data(sample_n);
        ggml_backend_tensor_get(t, h_data.data(), 0, sample_n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < sample_n; i++) f_data[i] = ggml_fp16_to_fp32(h_data[i]);
    } else if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> b_data(sample_n);
        ggml_backend_tensor_get(t, b_data.data(), 0, sample_n * sizeof(ggml_bf16_t));
        for (int64_t i = 0; i < sample_n; i++) f_data[i] = ggml_bf16_to_fp32(b_data[i]);
    } else {
        fprintf(stderr, "[debug-t] %s: type=%s not float\n", label, ggml_type_name(t->type));
        return;
    }
    float mn = f_data[0], mx = f_data[0], sum = 0.0f;
    for (float v : f_data) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
    }
    float avg = sum / f_data.size();
    fprintf(stderr, "[debug-t] %-20s ne=[%ld,%ld,%ld] type=%s min=%.4f max=%.4f avg=%.4f | [0..4]: %.4f %.4f %.4f %.4f %.4f\n",
            label, (long)t->ne[0], (long)t->ne[1], (long)t->ne[2], ggml_type_name(t->type),
            mn, mx, avg, f_data[0], f_data[1], f_data[2], f_data[3], f_data[4]);
}

static bool eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<cache_state *>(user_data);
    if (!st->active) {
        return false;
    }
    if (getenv("NVMOE_LOG_ALL_NAMES") && ask) {
        // Strip the trailing "-<layer>" suffix so we get one entry per distinct node kind.
        std::string name(t->name);
        auto dash = name.rfind('-');
        std::string base = (dash != std::string::npos && dash > 0) ? name.substr(0, dash) : name;
        if (g_seen_names.insert(base).second) {
            fprintf(stderr, "[all-names] %s (example: %s)\n", base.c_str(), t->name);
        }
    }

    bool is_debug_node = false;
    if (getenv("NVMOE_DEBUG_LAYER0")) {
        const char * debug_prefixes[] = {
            "hc_init", "hc_norm-0", "hc_mixed-0", "q_conv-0", "k_conv-0", "v_conv-0",
            "linear_attn_out-0", "hc_attn_combine-0", "ffn_moe_logits-0",
            "ffn_shexp-0", "ffn_moe_out-0", "hc_ffn_combine-0"
        };
        for (const char * p : debug_prefixes) {
            if (strcmp(t->name, p) == 0) {
                is_debug_node = true;
                break;
            }
        }
    }
    if (is_debug_node) {
        if (ask) return true;
        print_tensor_stats(t->name, t);
        return true;
    }

    if (t->op != GGML_OP_GET_ROWS || strncmp(t->name, "ffn_moe_weights-", 16) != 0) {
        return false;
    }
    const int il = atoi(t->name + 16);
    if (il < 0 || (size_t) il >= st->layers.size()) {
        return false;
    }
    char expected_weights[64];
    snprintf(expected_weights, sizeof(expected_weights), "ffn_moe_weights-%d", il);
    if (strcmp(t->name, expected_weights) != 0) {
        return false;
    }
    if (ask) {
        return true;
    }
    if (il == 0 && getenv("NVMOE_DEBUG_LAYER0")) {
        print_tensor_stats(t->name, t);
    }

    layer_cache & lc = st->layers[il];
    if (!lc.shrunk()) {
        return true; // full-size tensor for this layer: real ids already valid
    }

    // Single-Callback Optimization:
    // ffn_moe_weights has already executed ggml_get_rows natively on GPU using the real
    // expert IDs in selected_experts (t->src[1]). Gating weights in t are ALREADY
    // numerically correct!
    // We now read selected_experts, resolve cache misses into GPU slots, and overwrite
    // selected_experts in-place with GPU slot IDs just before downstream mul_mat_id runs.
    ggml_tensor * selected_experts = t->src[1];
    if (!selected_experts) {
        fprintf(stderr, "[cache] FATAL: t->src[1] is null for tensor '%s' (op=%s, ask=%d, src[0]=%s) on layer %d\n",
                t->name, ggml_op_name(t->op), ask, (t->src[0] ? t->src[0]->name : "null"), il);
        abort();
    }

    const int64_t n_expert_used = selected_experts->ne[0];
    const int64_t n_tokens = selected_experts->ne[1];
    const int64_t n = n_expert_used * n_tokens;
    std::vector<int32_t> ids(n);

    // selected_experts is a view of argsort (shape [n_expert, n_tokens]), so selected_experts->nb[1]
    // is the full row stride (e.g. 256*4 = 1024 bytes), NOT contiguous (8*4 = 32 bytes).
    for (int64_t row = 0; row < n_tokens; row++) {
        ggml_backend_tensor_get(selected_experts, ids.data() + row * n_expert_used, row * selected_experts->nb[1], n_expert_used * sizeof(int32_t));
    }

    // Dynamic Opportunistic Zero-IO Expert Pruning (Decode Phase Only)
    if (st->is_decode_phase && n_tokens == 1 && st->prune_nvme_thresh > 0.0f && t->data && st->pinned_weights) {
        cudaMemcpy(st->pinned_weights, t->data, n_expert_used * sizeof(float), cudaMemcpyDeviceToHost);

        float total_weight = 0.0f;
        for (int64_t i = 0; i < n_expert_used; i++) {
            total_weight += st->pinned_weights[i];
        }

        if (total_weight > 1e-6f) {
            int32_t resident_real = -1;
            for (int s = 0; s < lc.cache_size; s++) {
                if (lc.real_in_slot[s] != -1) {
                    resident_real = lc.real_in_slot[s];
                    break;
                }
            }

            if (resident_real != -1) {
                float remaining_mass = 1.0f;
                int kept_count = n_expert_used;
                bool weights_modified = false;

                for (int64_t k = n_expert_used - 1; k >= 0; k--) {
                    if (kept_count <= st->prune_min_keep) {
                        break;
                    }
                    int32_t real_id = ids[k];
                    bool in_gpu  = (lc.slot_of_real[real_id] != -1);
                    bool in_host = (lc.host_slot_of_real[real_id] != -1);

                    // If neither in GPU nor in Host, fetching requires NVMe disk I/O
                    if (!in_gpu && !in_host) {
                        float rel_w = st->pinned_weights[k] / total_weight;
                        if (rel_w < st->prune_nvme_thresh && (remaining_mass - rel_w) >= st->prune_min_mass) {
                            st->pinned_weights[k] = 0.0f;
                            ids[k] = resident_real;
                            remaining_mass -= rel_w;
                            kept_count--;
                            weights_modified = true;
                            st->decode_pruned_nvme++;
                        }
                    }
                }

                if (weights_modified) {
                    cudaMemcpyAsync(t->data, st->pinned_weights, n_expert_used * sizeof(float),
                                    cudaMemcpyHostToDevice, cudaStreamPerThread);
                }
            }
        }
    }

    // Fetch every distinct real expert id used by this layer's routing before remapping
    std::vector<int32_t> unique_ids;
    unique_ids.reserve(n);
    for (int32_t id : ids) {
        if (std::find(unique_ids.begin(), unique_ids.end(), id) == unique_ids.end()) {
            unique_ids.push_back(id);
        }
    }

    // --- Step 1: Analyze GPU and Host residency for all unique active experts ---
    lc.current_step++;
    bool any_gpu_fill = false;

    struct host_hit_item {
        int32_t real_id;
        int32_t gpu_slot;
        int32_t host_slot;
    };
    std::vector<host_hit_item> host_hits_to_transfer;

    struct nvme_expert_task {
        int32_t real_id;
        int32_t gpu_slot;
        int32_t host_slot;
        int completed_banks;
        int total_banks;
    };
    std::vector<nvme_expert_task> nvme_tasks;
    nvme_tasks.reserve(unique_ids.size());

    std::vector<int32_t> gpu_miss_ids;

    for (int32_t real_id : unique_ids) {
        int32_t slot = lc.slot_of_real[real_id];
        if (slot != -1) {
            st->hits++;
            if (st->is_decode_phase) st->decode_hits++;
            lc.slot_last_used[slot] = lc.current_step;
            lc.access_count[real_id]++;
            if (lc.host_slot_of_real[real_id] != -1) {
                lc.host_lru.touch(real_id);
            }
        } else {
            st->misses++;
            if (st->is_decode_phase) st->decode_misses++;
            gpu_miss_ids.push_back(real_id);
        }
    }

    // Allocate GPU slots for GPU misses and identify Host Hit vs NVMe Miss
    for (int32_t real_id : gpu_miss_ids) {
        int chosen_slot = -1;
        uint32_t oldest_step = UINT32_MAX;
        int start_slot = (st->is_decode_phase && lc.gpu_pinned_k > 0) ? lc.gpu_pinned_k : 0;
        for (int s = start_slot; s < lc.cache_size; s++) {
            int32_t occupant = lc.real_in_slot[s];
            if (occupant == -1) {
                chosen_slot = s;
                break;
            }
            if (std::find(unique_ids.begin(), unique_ids.end(), occupant) == unique_ids.end()) {
                if (lc.slot_last_used[s] < oldest_step) {
                    oldest_step = lc.slot_last_used[s];
                    chosen_slot = s;
                }
            }
        }
        if (chosen_slot == -1) {
            for (int s = 0; s < lc.cache_size; s++) {
                int32_t occupant = lc.real_in_slot[s];
                if (occupant == -1) {
                    chosen_slot = s;
                    break;
                }
                if (std::find(unique_ids.begin(), unique_ids.end(), occupant) == unique_ids.end()) {
                    if (lc.slot_last_used[s] < oldest_step) {
                        oldest_step = lc.slot_last_used[s];
                        chosen_slot = s;
                    }
                }
            }
        }
        if (chosen_slot == -1) {
            fprintf(stderr, "[FATAL] active_ids (%zu) exceeds cache_size (%lld)\n",
                    unique_ids.size(), (long long) lc.cache_size);
            abort();
        }
        int32_t slot = chosen_slot;
        lc.slot_last_used[slot] = lc.current_step;

        int32_t evicted_real = lc.real_in_slot[slot];
        if (evicted_real != -1) {
            lc.slot_of_real[evicted_real] = -1;
        }
        lc.real_in_slot[slot] = real_id;
        lc.slot_of_real[real_id] = slot;

        // Check host tier residency
        lc.access_count[real_id]++;
        int32_t h_slot = lc.host_slot_of_real[real_id];
        if (h_slot != -1) {
            st->host_hits++;
            if (st->is_decode_phase) st->decode_host_hits++;
            lc.host_lru.touch(real_id);
            host_hits_to_transfer.push_back({ real_id, slot, h_slot });
        } else {
            st->host_misses++;
            if (st->is_decode_phase) st->decode_host_misses++;
            int32_t new_slot = -1;
            if (lc.host_next_free < lc.host_cache_size) {
                new_slot = lc.host_next_free++;
            } else {
                auto not_in_active = [&](int32_t id) {
                    return std::find(unique_ids.begin(), unique_ids.end(), id) == unique_ids.end();
                };
                int32_t victim_real = lc.host_lru.pop_lru_matching(not_in_active);
                new_slot = lc.host_slot_of_real[victim_real];
                lc.host_slot_of_real[victim_real] = -1;
                lc.host_real_in_slot[new_slot] = -1;
            }
            lc.host_real_in_slot[new_slot] = real_id;
            lc.host_slot_of_real[real_id] = new_slot;
            lc.host_lru.insert_new(real_id);

            nvme_tasks.push_back({ real_id, slot, new_slot, 0, lc.fused_gate_up ? 2 : 3 });
        }
    }

    // --- Step 2: Immediately dispatch H2D transfers for Host Hits on dedicated stream ---
    // (This streams over PCIe concurrently with disk reads below!)
    for (const auto & hit : host_hits_to_transfer) {
        if (lc.fused_gate_up) {
            gpu_fill_from_host_async(*st, lc, lc.gate_up, lc.host_gate_up, hit.gpu_slot, hit.host_slot, hit.real_id);
            gpu_fill_from_host_async(*st, lc, lc.down,    lc.host_down,    hit.gpu_slot, hit.host_slot, hit.real_id);
            if (lc.gate_up_scale.gpu_tensor && !lc.host_gate_up_scale.empty()) {
                cudaMemcpyAsync((char *) lc.gate_up_scale.gpu_tensor->data + (size_t) hit.gpu_slot * sizeof(float),
                                &lc.host_gate_up_scale[hit.real_id], sizeof(float), cudaMemcpyHostToDevice, st->h2d_stream);
            }
            if (lc.down_scale.gpu_tensor && !lc.host_down_scale.empty()) {
                cudaMemcpyAsync((char *) lc.down_scale.gpu_tensor->data + (size_t) hit.gpu_slot * sizeof(float),
                                &lc.host_down_scale[hit.real_id], sizeof(float), cudaMemcpyHostToDevice, st->h2d_stream);
            }
        } else {
            gpu_fill_from_host_async(*st, lc, lc.gate, lc.host_gate, hit.gpu_slot, hit.host_slot, hit.real_id);
            gpu_fill_from_host_async(*st, lc, lc.up,   lc.host_up,   hit.gpu_slot, hit.host_slot, hit.real_id);
            gpu_fill_from_host_async(*st, lc, lc.down, lc.host_down, hit.gpu_slot, hit.host_slot, hit.real_id);
        }
        any_gpu_fill = true;
    }

    // --- Step 3: Pipelined NVMe reads + instant H2D dispatch per completed expert ---
    if (!nvme_tasks.empty()) {
        struct nvme_subjob {
            size_t offset;
            size_t len;
            uint8_t * out;
            size_t staging_idx;
            nvme_expert_task * task;
        };
        std::vector<nvme_subjob> subjobs;
        subjobs.reserve(nvme_tasks.size() * (lc.fused_gate_up ? 2 : 3));

        for (auto & t : nvme_tasks) {
            if (lc.fused_gate_up) {
                subjobs.push_back({ lc.gate_up.base_offset + (size_t) t.real_id * lc.gate_up.row_bytes, lc.gate_up.row_bytes, lc.host_gate_up.slot_ptr(t.host_slot), 0, &t });
                subjobs.push_back({ lc.down.base_offset    + (size_t) t.real_id * lc.down.row_bytes,    lc.down.row_bytes,    lc.host_down.slot_ptr(t.host_slot),    0, &t });
            } else {
                subjobs.push_back({ lc.gate.base_offset + (size_t) t.real_id * lc.gate.row_bytes, lc.gate.row_bytes, lc.host_gate.slot_ptr(t.host_slot), 0, &t });
                subjobs.push_back({ lc.up.base_offset   + (size_t) t.real_id * lc.up.row_bytes,   lc.up.row_bytes,   lc.host_up.slot_ptr(t.host_slot),   0, &t });
                subjobs.push_back({ lc.down.base_offset + (size_t) t.real_id * lc.down.row_bytes, lc.down.row_bytes, lc.host_down.slot_ptr(t.host_slot), 0, &t });
            }
        }

        auto dispatch_expert_to_gpu = [&](nvme_expert_task * t) {
            if (lc.fused_gate_up) {
                gpu_fill_from_host_async(*st, lc, lc.gate_up, lc.host_gate_up, t->gpu_slot, t->host_slot, t->real_id);
                gpu_fill_from_host_async(*st, lc, lc.down,    lc.host_down,    t->gpu_slot, t->host_slot, t->real_id);
                if (lc.gate_up_scale.gpu_tensor && !lc.host_gate_up_scale.empty()) {
                    cudaMemcpyAsync((char *) lc.gate_up_scale.gpu_tensor->data + (size_t) t->gpu_slot * sizeof(float),
                                    &lc.host_gate_up_scale[t->real_id], sizeof(float), cudaMemcpyHostToDevice, st->h2d_stream);
                }
                if (lc.down_scale.gpu_tensor && !lc.host_down_scale.empty()) {
                    cudaMemcpyAsync((char *) lc.down_scale.gpu_tensor->data + (size_t) t->gpu_slot * sizeof(float),
                                    &lc.host_down_scale[t->real_id], sizeof(float), cudaMemcpyHostToDevice, st->h2d_stream);
                }
            } else {
                gpu_fill_from_host_async(*st, lc, lc.gate, lc.host_gate, t->gpu_slot, t->host_slot, t->real_id);
                gpu_fill_from_host_async(*st, lc, lc.up,   lc.host_up,   t->gpu_slot, t->host_slot, t->real_id);
                gpu_fill_from_host_async(*st, lc, lc.down, lc.host_down, t->gpu_slot, t->host_slot, t->real_id);
            }
            any_gpu_fill = true;
        };

        if (!st->uring.initialized) {
            for (auto & sj : subjobs) {
                if (!read_odirect(st->fd_direct, sj.offset, sj.len, sj.out)) {
                    fprintf(stderr, "[cache] FATAL: read_odirect failed\n");
                    abort();
                }
                sj.task->completed_banks++;
                if (sj.task->completed_banks == sj.task->total_banks) {
                    dispatch_expert_to_gpu(sj.task);
                }
            }
        } else {
            size_t total = subjobs.size();
            for (size_t start = 0; start < total; start += st->uring.max_batch) {
                size_t batch_size = std::min(total - start, st->uring.max_batch);
                for (size_t i = 0; i < batch_size; i++) {
                    auto & sj = subjobs[start + i];
                    sj.staging_idx = i;
                    size_t aligned_off = align_down(sj.offset);
                    size_t front_pad   = sj.offset - aligned_off;
                    size_t aligned_len  = align_up(front_pad + sj.len);

                    io_uring_sqe * sqe = io_uring_get_sqe(&st->uring.ring);
                    io_uring_prep_read(sqe, st->fd_direct, st->uring.staging_buffers[i], aligned_len, aligned_off);
                    io_uring_sqe_set_data64(sqe, (uint64_t)&sj);
                }

                int ret = io_uring_submit(&st->uring.ring);
                if (ret < 0) {
                    fprintf(stderr, "[cache] FATAL: io_uring_submit failed: %d\n", ret);
                    abort();
                }

                size_t completed = 0;
                while (completed < batch_size) {
                    io_uring_cqe * cqe = nullptr;
                    int wret = io_uring_wait_cqe(&st->uring.ring, &cqe);
                    if (wret < 0 || !cqe) {
                        fprintf(stderr, "[cache] FATAL: io_uring_wait_cqe failed: %d\n", wret);
                        abort();
                    }
                    auto * sj = (nvme_subjob *) io_uring_cqe_get_data64(cqe);
                    int res = cqe->res;
                    io_uring_cqe_seen(&st->uring.ring, cqe);
                    completed++;

                    size_t aligned_off = align_down(sj->offset);
                    size_t front_pad   = sj->offset - aligned_off;
                    if (res < 0 || (size_t) res < front_pad + sj->len) {
                        fprintf(stderr, "[cache] FATAL: read error in io_uring: res=%d, expected=%zu\n", res, front_pad + sj->len);
                        abort();
                    }
                    memcpy(sj->out, st->uring.staging_buffers[sj->staging_idx] + front_pad, sj->len);
                    sj->task->completed_banks++;
                    if (sj->task->completed_banks == sj->task->total_banks) {
                        dispatch_expert_to_gpu(sj->task);
                    }
                }
            }
        }
    }

    // Remap real IDs to GPU slot IDs
    for (auto & id : ids) {
        int32_t orig_id = id;
        if (orig_id < 0 || orig_id >= lc.n_expert_real) {
            fprintf(stderr, "[BUG] invalid real_id=%d on layer %d (n_expert_real=%lld, n_tokens=%lld, n_expert_used=%lld)\n",
                    orig_id, il, (long long) lc.n_expert_real, (long long) n_tokens, (long long) n_expert_used);
            id = 0;
            continue;
        }
        id = lc.slot_of_real[orig_id];
        if (id < 0) {
            fprintf(stderr, "[BUG] self-eviction detected: real_id=%d resolved to slot=%d on layer %d (unique_ids=%zu, cache_size=%lld, n_tokens=%lld)\n",
                    orig_id, id, il, unique_ids.size(), (long long) lc.cache_size, (long long) n_tokens);
            id = 0;
        }
    }

    if (st->pinned_ids && selected_experts->data) {
        char * dev_dst = (char *) selected_experts->data;
        for (int64_t row = 0; row < n_tokens; row++) {
            memcpy(st->pinned_ids + row * n_expert_used, ids.data() + row * n_expert_used, n_expert_used * sizeof(int32_t));
            cudaMemcpyAsync(dev_dst + row * selected_experts->nb[1],
                            st->pinned_ids + row * n_expert_used,
                            n_expert_used * sizeof(int32_t),
                            cudaMemcpyHostToDevice,
                            cudaStreamPerThread);
        }
    } else {
        for (int64_t row = 0; row < n_tokens; row++) {
            ggml_backend_tensor_set(selected_experts, ids.data() + row * n_expert_used, row * selected_experts->nb[1], n_expert_used * sizeof(int32_t));
        }
    }

    if (any_gpu_fill) {
        cudaEventRecord(st->h2d_event, st->h2d_stream);
        cudaStreamWaitEvent(cudaStreamPerThread, st->h2d_event, 0);
    }

    st->callback_hits++;
    return true;
}

// Parse the GGUF file's per-layer expert-bank offsets/strides (same approach as
// scripts/gguf_expert_meta.cpp, Phase 2 step 1 -- verified against this exact model).
static bool describe_bank(struct gguf_context * gctx, struct ggml_context * meta_ctx, int il, const char * suffix, expert_bank & out) {
    char name[128];
    snprintf(name, sizeof(name), "blk.%d.%s.weight", il, suffix);
    int64_t tid = gguf_find_tensor(gctx, name);
    if (tid < 0) {
        return false;
    }
    ggml_tensor * t = ggml_get_tensor(meta_ctx, name);
    if (!t) {
        return false;
    }
    out.base_offset = gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, tid);
    out.row_bytes   = t->nb[2];
    if (il == 0 && getenv("NVMOE_DEBUG_LAYER0")) {
        fprintf(stderr, "[alignment] %s base_offset=%zu (%%4096=%zu) row_bytes=%zu (%%4096=%zu)\n",
                name, out.base_offset, out.base_offset % 4096, out.row_bytes, out.row_bytes % 4096);
    }
    return true;
}

static void setup_cache(cache_state & st, llama_model * model, const char * gguf_path, int n_layer) {
    st.fd_direct = open(gguf_path, O_RDONLY | O_DIRECT);
    if (st.fd_direct < 0) {
        fprintf(stderr, "[cache] FATAL: open(O_DIRECT) failed for %s: %s\n", gguf_path, strerror(errno));
        abort();
    }

    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta_ctx };
    struct gguf_context * gctx = gguf_init_from_file(gguf_path, gp);
    if (!gctx) {
        fprintf(stderr, "[cache] FATAL: gguf_init_from_file failed for %s\n", gguf_path);
        abort();
    }

    st.layers.resize(n_layer);
    size_t max_row_bytes = 0;
    for (int il = 0; il < n_layer; il++) {
        layer_cache & lc = st.layers[il];

        char gate_inp_name[64];
        snprintf(gate_inp_name, sizeof(gate_inp_name), "blk.%d.ffn_gate_inp.weight", il);
        ggml_tensor * gate_inp = llama_model_get_tensor(model, gate_inp_name);
        if (!gate_inp) {
            continue; // MTP / non-MoE layer
        }
        lc.n_expert_real = gate_inp->ne[1];

        char fused_name[64];
        snprintf(fused_name, sizeof(fused_name), "blk.%d.ffn_gate_up_exps.weight", il);
        lc.gate_up.gpu_tensor = llama_model_get_tensor(model, fused_name);

        if (lc.gate_up.gpu_tensor) {
            lc.fused_gate_up = true;
            char down_name[64];
            snprintf(down_name, sizeof(down_name), "blk.%d.ffn_down_exps.weight", il);
            lc.down.gpu_tensor = llama_model_get_tensor(model, down_name);
            if (!lc.down.gpu_tensor) continue;

            lc.cache_size = lc.gate_up.gpu_tensor->ne[2];
            if (!lc.shrunk()) continue;

            if (!describe_bank(gctx, meta_ctx, il, "ffn_gate_up_exps", lc.gate_up) ||
                !describe_bank(gctx, meta_ctx, il, "ffn_down_exps",    lc.down)) {
                fprintf(stderr, "[cache] FATAL: could not describe fused expert banks for layer %d\n", il);
                abort();
            }

            max_row_bytes = std::max({ max_row_bytes, lc.gate_up.row_bytes, lc.down.row_bytes });

            // NVFP4 scale tensors:
            char gu_scale_name[64], dn_scale_name[64];
            snprintf(gu_scale_name, sizeof(gu_scale_name), "blk.%d.ffn_gate_up_exps.scale", il);
            snprintf(dn_scale_name, sizeof(dn_scale_name), "blk.%d.ffn_down_exps.scale", il);
            lc.gate_up_scale.gpu_tensor = llama_model_get_tensor(model, gu_scale_name);
            lc.down_scale.gpu_tensor    = llama_model_get_tensor(model, dn_scale_name);

            int64_t tid_gu = gguf_find_tensor(gctx, gu_scale_name);
            if (tid_gu >= 0) {
                size_t off = gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, tid_gu);
                lc.host_gate_up_scale.resize(lc.n_expert_real);
                read_odirect(st.fd_direct, off, lc.n_expert_real * sizeof(float), (uint8_t *) lc.host_gate_up_scale.data());
            }
            int64_t tid_dn = gguf_find_tensor(gctx, dn_scale_name);
            if (tid_dn >= 0) {
                size_t off = gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, tid_dn);
                lc.host_down_scale.resize(lc.n_expert_real);
                read_odirect(st.fd_direct, off, lc.n_expert_real * sizeof(float), (uint8_t *) lc.host_down_scale.data());
            }
            if (lc.gate_up_scale.gpu_tensor && lc.down_scale.gpu_tensor) {
                fprintf(stderr, "[cache] layer %d loaded NVFP4 scales (gpu_tensor size=%lld, host_scales=%zu, sample: gu=%.6f, dn=%.6f)\n",
                        il, (long long) lc.gate_up_scale.gpu_tensor->ne[0], lc.host_gate_up_scale.size(),
                        lc.host_gate_up_scale.empty() ? 0.0f : lc.host_gate_up_scale[0],
                        lc.host_down_scale.empty() ? 0.0f : lc.host_down_scale[0]);
            }
        } else {
            lc.fused_gate_up = false;
            char gate_name[64], up_name[64], down_name[64];
            snprintf(gate_name, sizeof(gate_name), "blk.%d.ffn_gate_exps.weight", il);
            snprintf(up_name,   sizeof(up_name),   "blk.%d.ffn_up_exps.weight",   il);
            snprintf(down_name, sizeof(down_name), "blk.%d.ffn_down_exps.weight", il);

            lc.gate.gpu_tensor = llama_model_get_tensor(model, gate_name);
            lc.up.gpu_tensor   = llama_model_get_tensor(model, up_name);
            lc.down.gpu_tensor = llama_model_get_tensor(model, down_name);
            if (!lc.gate.gpu_tensor || !lc.up.gpu_tensor || !lc.down.gpu_tensor) {
                continue;
            }
            lc.cache_size = lc.gate.gpu_tensor->ne[2];
            if (!lc.shrunk()) continue;

            if (!describe_bank(gctx, meta_ctx, il, "ffn_gate_exps", lc.gate) ||
                !describe_bank(gctx, meta_ctx, il, "ffn_up_exps",   lc.up)   ||
                !describe_bank(gctx, meta_ctx, il, "ffn_down_exps", lc.down)) {
                fprintf(stderr, "[cache] FATAL: could not describe expert banks for layer %d\n", il);
                abort();
            }

            max_row_bytes = std::max({ max_row_bytes, lc.gate.row_bytes, lc.up.row_bytes, lc.down.row_bytes });
        }

        lc.slot_of_real.assign(lc.n_expert_real, -1);
        lc.real_in_slot.assign(lc.cache_size, -1);
        lc.slot_last_used.assign(lc.cache_size, 0);

        // Zero-initialize
        std::vector<expert_bank *> banks;
        if (lc.fused_gate_up) {
            banks = { &lc.gate_up, &lc.down };
        } else {
            banks = { &lc.gate, &lc.up, &lc.down };
        }
        for (expert_bank * bank : banks) {
            std::vector<uint8_t> zeros(ggml_nbytes(bank->gpu_tensor), 0);
            ggml_backend_tensor_set(bank->gpu_tensor, zeros.data(), 0, zeros.size());
        }
        if (lc.gate_up_scale.gpu_tensor) {
            std::vector<uint8_t> zeros(ggml_nbytes(lc.gate_up_scale.gpu_tensor), 0);
            ggml_backend_tensor_set(lc.gate_up_scale.gpu_tensor, zeros.data(), 0, zeros.size());
        }
        if (lc.down_scale.gpu_tensor) {
            std::vector<uint8_t> zeros(ggml_nbytes(lc.down_scale.gpu_tensor), 0);
            ggml_backend_tensor_set(lc.down_scale.gpu_tensor, zeros.data(), 0, zeros.size());
        }

        // Host tier setup
        lc.host_cache_size = std::min<int64_t>(st.host_cache_size_cfg, lc.n_expert_real);
        lc.host_lru.set_capacity_hint(lc.host_cache_size);
        lc.host_slot_of_real.assign(lc.n_expert_real, -1);
        lc.host_real_in_slot.assign(lc.host_cache_size, -1);
        if (lc.fused_gate_up) {
            lc.host_gate_up.init(lc.host_cache_size, lc.gate_up.row_bytes);
            lc.host_down.init(lc.host_cache_size, lc.down.row_bytes);
        } else {
            lc.host_gate.init(lc.host_cache_size, lc.gate.row_bytes);
            lc.host_up.init(lc.host_cache_size, lc.up.row_bytes);
            lc.host_down.init(lc.host_cache_size, lc.down.row_bytes);
        }

        lc.access_count.assign(lc.n_expert_real, 0);

        if (lc.fused_gate_up) {
            fprintf(stderr, "[cache] layer %d: GPU cache_size=%lld, host cache_size=%lld (real n_expert=%lld), row_bytes gate_up=%zu down=%zu (pinned=%s)\n",
                    il, (long long) lc.cache_size, (long long) lc.host_cache_size, (long long) lc.n_expert_real,
                    lc.gate_up.row_bytes, lc.down.row_bytes, lc.host_gate_up.is_pinned ? "YES" : "NO");
        } else {
            fprintf(stderr, "[cache] layer %d: GPU cache_size=%lld, host cache_size=%lld (real n_expert=%lld), row_bytes gate=%zu up=%zu down=%zu (pinned=%s)\n",
                    il, (long long) lc.cache_size, (long long) lc.host_cache_size, (long long) lc.n_expert_real,
                    lc.gate.row_bytes, lc.up.row_bytes, lc.down.row_bytes, lc.host_gate.is_pinned ? "YES" : "NO");
        }
    }

    st.uring.init(128, max_row_bytes > 0 ? max_row_bytes : 589824);

    ggml_free(meta_ctx);
    gguf_free(gctx);
}

// Cross-check our GGUF-offset/row_bytes math against the model's OWN loaded (full,
// unshrunk) tensor data for a CPU-placed layer, to rule out a bank/offset mismatch in
// our own reader (as opposed to a self-consistency check against another read of the
// same file, which Phase 2 already covered).
static void cross_check_against_loaded_tensor(cache_state & st, llama_model * model, const char * gguf_path, int il, int expert_id) {
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params gp = { true, &meta_ctx };
    struct gguf_context * gctx = gguf_init_from_file(gguf_path, gp);

    expert_bank bank;
    if (!describe_bank(gctx, meta_ctx, il, "ffn_gate_exps", bank)) {
        fprintf(stderr, "[cross-check] FATAL: could not describe bank\n");
        abort();
    }

    std::vector<uint8_t> from_file(bank.row_bytes);
    read_odirect(st.fd_direct, bank.base_offset + (size_t) expert_id * bank.row_bytes, bank.row_bytes, from_file.data());

    char name[64];
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", il);
    ggml_tensor * loaded = llama_model_get_tensor(model, name);
    std::vector<uint8_t> from_model(bank.row_bytes);
    ggml_backend_tensor_get(loaded, from_model.data(), (size_t) expert_id * bank.row_bytes, bank.row_bytes);

    bool match = memcmp(from_file.data(), from_model.data(), bank.row_bytes) == 0;
    fprintf(stderr, "[cross-check] layer %d expert %d: our_odirect_read == model_loaded_tensor: %s\n",
            il, expert_id, match ? "YES" : "NO");

    ggml_free(meta_ctx);
    gguf_free(gctx);
}

// Phase 4 calibration persistence: a trivial flat binary format (not nvmoe's torch.save,
// no need for cross-language compat here) -- per-layer: n_expert_real int64 counts.
static void save_decode_freq(cache_state & st, const char * path) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[calib] WARNING: failed to open %s for writing\n", path);
        return;
    }
    int64_t total = 0;
    for (auto & lc : st.layers) {
        if (lc.access_count.empty()) {
            continue;
        }
        fwrite(lc.access_count.data(), sizeof(int64_t), lc.access_count.size(), f);
        for (int64_t c : lc.access_count) {
            total += c;
        }
    }
    fclose(f);
    fprintf(stderr, "[calib] saved decode routing frequencies (%lld total hits) to %s\n", (long long) total, path);
}

static bool load_decode_freq(cache_state & st, const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    for (auto & lc : st.layers) {
        if (lc.access_count.empty()) {
            continue;
        }
        std::vector<int64_t> freq(lc.n_expert_real);
        size_t n = fread(freq.data(), sizeof(int64_t), freq.size(), f);
        if (n != freq.size()) {
            fprintf(stderr, "[calib] WARNING: %s is truncated/mismatched, ignoring\n", path);
            fclose(f);
            return false;
        }
        lc.access_count = std::move(freq); // seed with prior run's frequencies
    }
    fclose(f);
    return true;
}

// Phase 4, Phase D/E of nvmoe (prewarm_cache): pin the top-K most frequently routed
// experts per layer into the host tier before generation starts, so early decode tokens
// avoid cold-start NVMe misses. Reserves at least 25% of host slots for the dynamic LRU
// so pinning can't starve eviction candidates entirely -- same formula as nvmoe.
static void prewarm_cache(cache_state & st) {
    int64_t prewarmed_total = 0;
    std::vector<read_job> prewarm_jobs;

    for (auto & lc : st.layers) {
        if (!lc.shrunk() || lc.access_count.empty()) {
            continue;
        }
        int64_t host_fill_k = std::min<int64_t>(lc.host_cache_size, lc.n_expert_real);
        int64_t max_pin_per_layer = (lc.host_cache_size * 3 / 4);
        int64_t default_pin = std::min<int64_t>(64, max_pin_per_layer);
        const char * pin_env = getenv("NVMOE_PINNED_EXPERTS");
        int64_t pin_k_cfg = pin_env ? atoll(pin_env) : default_pin;
        int64_t pin_k = std::min({ pin_k_cfg, max_pin_per_layer, lc.n_expert_real });

        const char * gpu_pin_env = getenv("NVMOE_GPU_PINNED_EXPERTS");
        int64_t default_gpu_pin = std::min<int64_t>(16, lc.cache_size / 2);
        lc.gpu_pinned_k = gpu_pin_env ? atoi(gpu_pin_env) : default_gpu_pin;
        if (lc.gpu_pinned_k > lc.cache_size - 10) lc.gpu_pinned_k = std::max<int32_t>(0, (int32_t)lc.cache_size - 10);

        std::vector<int32_t> ids_by_freq(lc.n_expert_real);
        for (int32_t i = 0; i < (int32_t) lc.n_expert_real; i++) {
            ids_by_freq[i] = i;
        }
        std::partial_sort(ids_by_freq.begin(), ids_by_freq.begin() + host_fill_k, ids_by_freq.end(),
                [&](int32_t a, int32_t b) { return lc.access_count[a] > lc.access_count[b]; });

        int64_t pinned_this_layer = 0;
        for (int64_t k = 0; k < host_fill_k; k++) {
            int32_t real_id = ids_by_freq[k];
            if (lc.access_count[real_id] <= 0) {
                break; // sorted descending -- rest are all zero too
            }
            int32_t slot = lc.host_next_free++;
            lc.host_real_in_slot[slot] = real_id;
            lc.host_slot_of_real[real_id] = slot;
            if (k < pin_k) {
                lc.pinned.insert(real_id);
                pinned_this_layer++;
            } else {
                lc.host_lru.insert_new(real_id);
            }

            if (lc.fused_gate_up) {
                prewarm_jobs.push_back({ lc.gate_up.base_offset + (size_t) real_id * lc.gate_up.row_bytes, lc.gate_up.row_bytes, lc.host_gate_up.slot_ptr(slot) });
                prewarm_jobs.push_back({ lc.down.base_offset    + (size_t) real_id * lc.down.row_bytes,    lc.down.row_bytes,    lc.host_down.slot_ptr(slot) });
            } else {
                prewarm_jobs.push_back({ lc.gate.base_offset + (size_t) real_id * lc.gate.row_bytes, lc.gate.row_bytes, lc.host_gate.slot_ptr(slot) });
                prewarm_jobs.push_back({ lc.up.base_offset   + (size_t) real_id * lc.up.row_bytes,   lc.up.row_bytes,   lc.host_up.slot_ptr(slot)   });
                prewarm_jobs.push_back({ lc.down.base_offset + (size_t) real_id * lc.down.row_bytes, lc.down.row_bytes, lc.host_down.slot_ptr(slot) });
            }
        }
        prewarmed_total += pinned_this_layer;
    }

    if (!prewarm_jobs.empty()) {
        fprintf(stderr, "[calib] prewarm: reading %zu expert banks via io_uring batch...\n", prewarm_jobs.size());
        auto t0 = std::chrono::steady_clock::now();
        if (!read_jobs_batch(st, prewarm_jobs)) {
            fprintf(stderr, "[calib] FATAL: prewarm read_jobs_batch failed\n");
            abort();
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        size_t total_job_bytes = 0;
        for (const auto & j : prewarm_jobs) total_job_bytes += j.len;
        double mb = total_job_bytes / (1024.0 * 1024.0);
        fprintf(stderr, "[calib] prewarm: pinned %lld experts across %zu layers (%.2f ms, %.1f MB/s)\n",
                (long long) prewarmed_total, st.layers.size(), ms, mb / (ms / 1000.0));
    } else {
        fprintf(stderr, "[calib] prewarm: pinned %lld experts across %zu layers\n", (long long) prewarmed_total, st.layers.size());
    }

    int64_t gpu_prewarmed_total = 0;
    for (auto & lc : st.layers) {
        if (!lc.shrunk()) continue;
        int64_t n_fill = std::min<int64_t>(lc.cache_size, lc.host_next_free);
        for (int32_t gpu_slot = 0; gpu_slot < n_fill; gpu_slot++) {
            int32_t real_id = lc.host_real_in_slot[gpu_slot];
            if (real_id < 0) continue;
            lc.real_in_slot[gpu_slot] = real_id;
            lc.slot_of_real[real_id] = gpu_slot;
            lc.slot_last_used[gpu_slot] = 0;

            if (lc.fused_gate_up) {
                cudaMemcpyAsync((char *) lc.gate_up.gpu_tensor->data + (size_t) gpu_slot * lc.gate_up.row_bytes,
                                lc.host_gate_up.slot_ptr(gpu_slot), lc.gate_up.row_bytes, cudaMemcpyHostToDevice, st.h2d_stream);
                cudaMemcpyAsync((char *) lc.down.gpu_tensor->data + (size_t) gpu_slot * lc.down.row_bytes,
                                lc.host_down.slot_ptr(gpu_slot), lc.down.row_bytes, cudaMemcpyHostToDevice, st.h2d_stream);
                if (lc.gate_up_scale.gpu_tensor && !lc.host_gate_up_scale.empty()) {
                    cudaMemcpyAsync((char *) lc.gate_up_scale.gpu_tensor->data + (size_t) gpu_slot * sizeof(float),
                                    &lc.host_gate_up_scale[real_id], sizeof(float), cudaMemcpyHostToDevice, st.h2d_stream);
                }
                if (lc.down_scale.gpu_tensor && !lc.host_down_scale.empty()) {
                    cudaMemcpyAsync((char *) lc.down_scale.gpu_tensor->data + (size_t) gpu_slot * sizeof(float),
                                    &lc.host_down_scale[real_id], sizeof(float), cudaMemcpyHostToDevice, st.h2d_stream);
                }
            } else {
                cudaMemcpyAsync((char *) lc.gate.gpu_tensor->data + (size_t) gpu_slot * lc.gate.row_bytes,
                                lc.host_gate.slot_ptr(gpu_slot), lc.gate.row_bytes, cudaMemcpyHostToDevice, st.h2d_stream);
                cudaMemcpyAsync((char *) lc.up.gpu_tensor->data + (size_t) gpu_slot * lc.up.row_bytes,
                                lc.host_up.slot_ptr(gpu_slot), lc.up.row_bytes, cudaMemcpyHostToDevice, st.h2d_stream);
                cudaMemcpyAsync((char *) lc.down.gpu_tensor->data + (size_t) gpu_slot * lc.down.row_bytes,
                                lc.host_down.slot_ptr(gpu_slot), lc.down.row_bytes, cudaMemcpyHostToDevice, st.h2d_stream);
            }
            gpu_prewarmed_total++;
        }
    }
    if (gpu_prewarmed_total > 0) {
        cudaStreamSynchronize(st.h2d_stream);
        fprintf(stderr, "[calib] prewarm: loaded %lld hot experts directly into GPU VRAM\n", (long long) gpu_prewarmed_total);
    }

    // Reset stats so inference telemetry reflects steady-state hit rates, not the
    // prewarm sweep's own fills (matches nvmoe's counter reset after prewarm_cache).
    st.hits = st.misses = st.host_hits = st.host_misses = 0;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    cache_state st;
    const char * cache_env = getenv("NVMOE_CACHE_SIZE");
    st.cache_size_cfg = cache_env ? atoi(cache_env) : 0;
    if (st.cache_size_cfg <= 0) {
        fprintf(stderr, "NVMOE_CACHE_SIZE must be set (>0) for moe_cache_probe\n");
        return 1;
    }
    const char * host_cache_env = getenv("NVMOE_HOST_CACHE_SIZE");
    st.host_cache_size_cfg = host_cache_env ? atoi(host_cache_env) : st.cache_size_cfg * 4;

    const char * prune_env = getenv("NVMOE_PRUNE_NVME_THRESH");
    st.prune_nvme_thresh = prune_env ? atof(prune_env) : 0.0f;
    const char * prune_keep_env = getenv("NVMOE_PRUNE_MIN_KEEP");
    if (prune_keep_env) st.prune_min_keep = atoi(prune_keep_env);
    const char * prune_mass_env = getenv("NVMOE_PRUNE_MIN_MASS");
    if (prune_mass_env) st.prune_min_mass = atof(prune_mass_env);
    if (st.prune_nvme_thresh > 0.0f) {
        fprintf(stderr, "[prune] Zero-IO Tail Pruning enabled: thresh=%.4f (min_keep=%d, min_mass=%.3f)\n",
                st.prune_nvme_thresh, st.prune_min_keep, st.prune_min_mass);
    }

    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    setup_cache(st, model, params.model.path.c_str(), llama_model_n_layer(model));

    const char * freq_path = getenv("NVMOE_FREQ_PATH");
    if (freq_path && load_decode_freq(st, freq_path)) {
        fprintf(stderr, "[calib] loaded prior routing frequencies from %s\n", freq_path);
        prewarm_cache(st);
    } else if (freq_path) {
        fprintf(stderr, "[calib] no routing frequency file at %s yet, will collect during this run\n", freq_path);
    }

    st.active = true;

    llama_context_params cparams = common_context_params_to_llama(params);
    cparams.cb_eval = eval_cb;
    cparams.cb_eval_user_data = &st;

    int32_t min_cache = 0;
    for (const auto & lc : st.layers) {
        if (lc.shrunk()) {
            if (min_cache == 0 || lc.cache_size < min_cache) {
                min_cache = lc.cache_size;
            }
        }
    }
    if (min_cache > 0) {
        // With top-10 routing, a batch of N tokens can require up to min(512, N*10) unique experts.
        // Clamp n_ubatch so the worst-case active set comfortably fits within GPU cache_size.
        int32_t safe_ubatch = std::max(1, (min_cache - 4) / 10);
        if (cparams.n_ubatch > safe_ubatch) {
            fprintf(stderr, "[cache] auto-clamping n_ubatch from %d to %d (cache_size=%d, top-10 routing)\n",
                    cparams.n_ubatch, safe_ubatch, min_cache);
            cparams.n_ubatch = safe_ubatch;
        }
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = common_tokenize(vocab, params.prompt, true, true);
    fprintf(stderr, "[prompt] %zu tokens: ", tokens.size());
    for (auto t : tokens) fprintf(stderr, "%d ", t);
    fprintf(stderr, "\n");
    common_sampler * smpl = common_sampler_init(model, params.sampling);

    auto t_start = std::chrono::steady_clock::now();
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    st.is_decode_phase = false;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "prompt decode failed\n");
        return 1;
    }

    auto t_prompt_done = std::chrono::steady_clock::now();
    double ttft = std::chrono::duration<double>(t_prompt_done - t_start).count();

    // Switch to decode phase and reset steady-state metrics
    st.is_decode_phase = true;
    st.decode_hits = st.decode_misses = st.decode_host_hits = st.decode_host_misses = 0;

    llama_token new_token = common_sampler_sample(smpl, ctx, -1);
    common_sampler_accept(smpl, new_token, true);

    std::string out_text;
    int n_decoded = 0;
    if (!llama_vocab_is_eog(vocab, new_token)) {
        std::string piece = common_token_to_piece(ctx, new_token);
        printf("%s", piece.c_str());
        fflush(stdout);
        out_text += piece;
        n_decoded = 1;
        batch = llama_batch_get_one(&new_token, 1);
    }

    auto t_decode_start = std::chrono::steady_clock::now();
    auto t_last_token = t_decode_start;
    std::vector<double> token_times;

    while (n_decoded < params.n_predict) {
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed\n");
            break;
        }
        const bool debug_logits = getenv("NVMOE_DEBUG_LOGITS") != nullptr;
        const bool debug_tokens = getenv("NVMOE_DEBUG_TOKENS") != nullptr;
        if (debug_logits) {
            float * logits = llama_get_logits_ith(ctx, -1);
            int n_vocab = llama_vocab_n_tokens(vocab);
            std::vector<std::pair<float, int>> top_k;
            int nan_count = 0;
            for (int v = 0; v < n_vocab; v++) {
                if (std::isnan(logits[v])) {
                    nan_count++;
                } else {
                    if (top_k.size() < 5) {
                        top_k.push_back({ logits[v], v });
                        std::push_heap(top_k.begin(), top_k.end(), std::greater<std::pair<float, int>>());
                    } else if (logits[v] > top_k.front().first) {
                        std::pop_heap(top_k.begin(), top_k.end(), std::greater<std::pair<float, int>>());
                        top_k.back() = { logits[v], v };
                        std::push_heap(top_k.begin(), top_k.end(), std::greater<std::pair<float, int>>());
                    }
                }
            }
            std::sort(top_k.begin(), top_k.end(), std::greater<std::pair<float, int>>());
            fprintf(stderr, "[debug-logits] step=%d nan_count=%d target(11751 ' Paris')=%.3f top5:\n",
                    n_decoded, nan_count, (11751 < n_vocab ? logits[11751] : -999.0f));
            for (size_t k = 0; k < top_k.size(); k++) {
                std::string piece = common_token_to_piece(ctx, top_k[k].second);
                fprintf(stderr, "  #%zu: id=%d (logit=%.3f) '%s'\n", k, top_k[k].second, top_k[k].first, piece.c_str());
            }
        }

        llama_token next_token = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, next_token, true);
        if (llama_vocab_is_eog(vocab, next_token)) {
            break;
        }
        std::string piece = common_token_to_piece(ctx, next_token);
        if (debug_tokens) {
            fprintf(stderr, "[token] id=%d piece=%s piece_hex=", next_token, piece.c_str());
            for (unsigned char c : piece) fprintf(stderr, "%02x ", c);
            fprintf(stderr, "\n");
        } else {
            printf("%s", piece.c_str());
            fflush(stdout);
        }
        out_text += piece;
        n_decoded++;
        auto t_cur_token = std::chrono::steady_clock::now();
        token_times.push_back(std::chrono::duration<double>(t_cur_token - t_last_token).count());
        t_last_token = t_cur_token;
        batch = llama_batch_get_one(&next_token, 1);
    }

    auto t_decode_end = std::chrono::steady_clock::now();
    double decode_secs = std::chrono::duration<double>(t_decode_end - t_decode_start).count();
    double total_secs = std::chrono::duration<double>(t_decode_end - t_start).count();
    double decode_tok_per_sec = (n_decoded > 1 && decode_secs > 0) ? (n_decoded - 1) / decode_secs : 0.0;

    long total_refs = st.decode_hits + st.decode_misses;
    double gpu_hit_rate = (total_refs > 0) ? (100.0 * st.decode_hits / total_refs) : 0.0;
    long total_host_lookups = st.decode_host_hits + st.decode_host_misses;
    double host_hit_rate = (total_host_lookups > 0) ?
        (100.0 * st.decode_host_hits / total_host_lookups) : 0.0;
    double nvme_miss_rate = (total_host_lookups > 0) ?
        (100.0 * st.decode_host_misses / total_host_lookups) : 0.0;

    fprintf(stderr, "\n\n========================= NVMoE Performance Report =========================\n");
    fprintf(stderr, "Prompt Tokens:        %zu\n", tokens.size());
    fprintf(stderr, "Generated Tokens:     %d\n", n_decoded);
    fprintf(stderr, "TTFT (Prefill):       %.3f s (%.2f tok/s)\n", ttft, tokens.size() / (ttft > 0 ? ttft : 1.0));
    fprintf(stderr, "Decode Time:          %.3f s\n", decode_secs);
    fprintf(stderr, "Decode Throughput:    %.2f tok/s\n", decode_tok_per_sec);
    if (token_times.size() > 5) {
        double steady_sum = 0.0;
        double min_ms = 1e9, max_ms = 0.0;
        size_t steady_count = token_times.size() - 5;
        for (size_t i = 5; i < token_times.size(); i++) {
            double ms = token_times[i] * 1000.0;
            steady_sum += ms;
            min_ms = std::min(min_ms, ms);
            max_ms = std::max(max_ms, ms);
        }
        double steady_avg_ms = steady_sum / steady_count;
        double steady_tok_s = 1000.0 / steady_avg_ms;
        fprintf(stderr, "Steady-State Decode:  %.2f tok/s (avg=%.1f ms/tok, min=%.1f ms, max=%.1f ms, n=%zu)\n",
                steady_tok_s, steady_avg_ms, min_ms, max_ms, steady_count);
    }
    fprintf(stderr, "Total End-to-End:     %.3f s (%.2f tok/s)\n", total_secs, (tokens.size() + n_decoded) / total_secs);
    fprintf(stderr, "----------------------------------------------------------------------------\n");
    fprintf(stderr, "Decode GPU VRAM Hits: %ld / %ld (%.1f%%, 0 ms PCIe transfer)\n", st.decode_hits, total_refs, gpu_hit_rate);
    fprintf(stderr, "Decode In-Memory Hits:%ld / %ld (%.1f%% RAM hit, %.1f%% NVMe read)\n",
            st.decode_host_hits, total_host_lookups, host_hit_rate, nvme_miss_rate);
    if (st.decode_pruned_nvme > 0) {
        fprintf(stderr, "Decode NVMe Pruned:   %ld tail misses skipped (Zero-IO dynamic pruning)\n", st.decode_pruned_nvme);
    }
    fprintf(stderr, "Cumulative Stats:     callback_hits=%ld gpu_hits=%ld gpu_misses=%ld host_hits=%ld host_misses=%ld\n",
            st.callback_hits, st.hits, st.misses, st.host_hits, st.host_misses);
    fprintf(stderr, "============================================================================\n");
    fprintf(stderr, "[cache] output: %s\n", out_text.c_str());

    if (freq_path) {
        save_decode_freq(st, freq_path);
    }

    common_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    close(st.fd_direct);
    return 0;
}
