#include <liburing.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <random>

static constexpr size_t ALIGN = 4096;
static inline size_t align_down(size_t x) { return x - (x % ALIGN); }
static inline size_t align_up(size_t x)   { return ((x + ALIGN - 1) / ALIGN) * ALIGN; }

struct read_req {
    size_t offset;
    size_t len;
    uint8_t * out;
};

// 1. Naive O_DIRECT pread with per-read posix_memalign & free (current implementation)
static bool read_odirect_naive(int fd, size_t offset, size_t len, uint8_t * out) {
    size_t aligned_off = align_down(offset);
    size_t front_pad   = offset - aligned_off;
    size_t aligned_len  = align_up(front_pad + len);

    char * buf = nullptr;
    if (posix_memalign((void **) &buf, ALIGN, aligned_len) != 0) return false;
    ssize_t n = pread(fd, buf, aligned_len, aligned_off);
    bool ok = (n >= 0 && (size_t) n >= front_pad + len);
    if (ok) memcpy(out, buf + front_pad, len);
    free(buf);
    return ok;
}

// 2. Optimized synchronous O_DIRECT pread with preallocated staging buffer
static bool read_odirect_cached(int fd, size_t offset, size_t len, uint8_t * out, char * staging_buf) {
    size_t aligned_off = align_down(offset);
    size_t front_pad   = offset - aligned_off;
    size_t aligned_len  = align_up(front_pad + len);

    ssize_t n = pread(fd, staging_buf, aligned_len, aligned_off);
    bool ok = (n >= 0 && (size_t) n >= front_pad + len);
    if (ok) memcpy(out, staging_buf + front_pad, len);
    return ok;
}

// 3. Buffered pread
static bool read_buffered(int fd, size_t offset, size_t len, uint8_t * out) {
    ssize_t n = pread(fd, out, len, offset);
    return (n == (ssize_t) len);
}

// 4. Batched io_uring O_DIRECT with preallocated staging buffers
struct uring_batch_ctx {
    io_uring ring;
    std::vector<char *> staging_buffers;
    size_t max_batch = 64;
    size_t max_aligned_len = 0;

    uring_batch_ctx(size_t max_b, size_t max_len) : max_batch(max_b), max_aligned_len(max_len) {
        io_uring_queue_init(max_batch, &ring, 0);
        staging_buffers.resize(max_batch);
        for (size_t i = 0; i < max_batch; i++) {
            posix_memalign((void **) &staging_buffers[i], ALIGN, max_aligned_len);
        }
    }

    ~uring_batch_ctx() {
        for (char * buf : staging_buffers) free(buf);
        io_uring_queue_exit(&ring);
    }

    bool execute(int fd, const std::vector<read_req> & reqs) {
        if (reqs.empty()) return true;
        size_t n = std::min(reqs.size(), max_batch);

        for (size_t i = 0; i < n; i++) {
            const auto & r = reqs[i];
            size_t aligned_off = align_down(r.offset);
            size_t front_pad   = r.offset - aligned_off;
            size_t aligned_len  = align_up(front_pad + r.len);

            io_uring_sqe * sqe = io_uring_get_sqe(&ring);
            io_uring_prep_read(sqe, fd, staging_buffers[i], aligned_len, aligned_off);
            io_uring_sqe_set_data64(sqe, i);
        }

        int ret = io_uring_submit_and_wait(&ring, n);
        if (ret < 0) return false;

        bool all_ok = true;
        for (size_t i = 0; i < n; i++) {
            io_uring_cqe * cqe = nullptr;
            int wret = io_uring_wait_cqe(&ring, &cqe);
            if (wret < 0 || !cqe) {
                all_ok = false;
                continue;
            }
            uint64_t idx = io_uring_cqe_get_data64(cqe);
            int res = cqe->res;
            io_uring_cqe_seen(&ring, cqe);

            const auto & r = reqs[idx];
            size_t aligned_off = align_down(r.offset);
            size_t front_pad   = r.offset - aligned_off;
            if (res < 0 || (size_t) res < front_pad + r.len) {
                all_ok = false;
                continue;
            }
            memcpy(r.out, staging_buffers[idx] + front_pad, r.len);
        }
        return all_ok;
    }
};

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "/home/waltarb/models/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    size_t base_offset = argc > 2 ? strtoull(argv[2], nullptr, 10) : 740986368ULL;
    size_t row_bytes = 589824; // 576 KiB
    int n_rounds = 20;
    int batch_size = 24; // 8 experts per round (simulating 1 layer decode miss)

    int fd_direct = open(path, O_RDONLY | O_DIRECT);
    int fd_buffered = open(path, O_RDONLY);
    if (fd_direct < 0 || fd_buffered < 0) {
        fprintf(stderr, "Failed to open %s\n", path);
        return 1;
    }

    printf("=== NVMe Expert IO Benchmark ===\n");
    printf("Model: %s\n", path);
    printf("Row size: %zu bytes (%.2f KiB)\n", row_bytes, row_bytes / 1024.0);
    printf("Batch size per step: %d experts (%.2f MiB)\n", batch_size, (batch_size * row_bytes) / (1024.0 * 1024.0));
    printf("Rounds: %d (Total data: %.2f MiB)\n\n", n_rounds, (n_rounds * batch_size * row_bytes) / (1024.0 * 1024.0));

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);

    std::vector<std::vector<read_req>> rounds(n_rounds);
    std::vector<std::vector<uint8_t>> out_buffers(n_rounds * batch_size, std::vector<uint8_t>(row_bytes));

    for (int r = 0; r < n_rounds; r++) {
        for (int b = 0; b < batch_size; b++) {
            int exp = dist(rng);
            size_t off = base_offset + (size_t) exp * row_bytes;
            rounds[r].push_back({off, row_bytes, out_buffers[r * batch_size + b].data()});
        }
    }

    // Benchmark 1: Naive O_DIRECT (posix_memalign + pread + free per read)
    {
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < n_rounds; r++) {
            for (const auto & req : rounds[r]) {
                read_odirect_naive(fd_direct, req.offset, req.len, req.out);
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double mb = (n_rounds * batch_size * row_bytes) / (1024.0 * 1024.0);
        printf("[1] Naive O_DIRECT (alloc+pread+free): %.2f ms total, %.2f ms/batch, %.1f MB/s (%.0f IOPs)\n",
               ms, ms / n_rounds, (mb / (ms / 1000.0)), (n_rounds * batch_size) / (ms / 1000.0));
    }

    // Benchmark 2: Cached-buffer O_DIRECT (no per-read allocation)
    {
        char * staging = nullptr;
        posix_memalign((void **) &staging, ALIGN, align_up(row_bytes + ALIGN));
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < n_rounds; r++) {
            for (const auto & req : rounds[r]) {
                read_odirect_cached(fd_direct, req.offset, req.len, req.out, staging);
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        free(staging);
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double mb = (n_rounds * batch_size * row_bytes) / (1024.0 * 1024.0);
        printf("[2] Cached-buffer O_DIRECT (pread):    %.2f ms total, %.2f ms/batch, %.1f MB/s (%.0f IOPs)\n",
               ms, ms / n_rounds, (mb / (ms / 1000.0)), (n_rounds * batch_size) / (ms / 1000.0));
    }

    // Benchmark 3: io_uring O_DIRECT Batch (parallel asynchronous NVMe read submission)
    {
        uring_batch_ctx uctx(batch_size, align_up(row_bytes + ALIGN));
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < n_rounds; r++) {
            uctx.execute(fd_direct, rounds[r]);
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double mb = (n_rounds * batch_size * row_bytes) / (1024.0 * 1024.0);
        printf("[3] io_uring O_DIRECT (batch=%d):       %.2f ms total, %.2f ms/batch, %.1f MB/s (%.0f IOPs)\n",
               batch_size, ms, ms / n_rounds, (mb / (ms / 1000.0)), (n_rounds * batch_size) / (ms / 1000.0));
    }

    // Benchmark 4: Buffered pread (with OS page-cache)
    {
        // First drop OS page cache for this file if possible via fadvise
        posix_fadvise(fd_buffered, 0, 0, POSIX_FADV_DONTNEED);
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < n_rounds; r++) {
            for (const auto & req : rounds[r]) {
                read_buffered(fd_buffered, req.offset, req.len, req.out);
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double mb = (n_rounds * batch_size * row_bytes) / (1024.0 * 1024.0);
        printf("[4] Buffered pread (cold/fadvise):     %.2f ms total, %.2f ms/batch, %.1f MB/s (%.0f IOPs)\n",
               ms, ms / n_rounds, (mb / (ms / 1000.0)), (n_rounds * batch_size) / (ms / 1000.0));
    }

    close(fd_direct);
    close(fd_buffered);
    return 0;
}
