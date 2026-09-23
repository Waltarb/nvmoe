// Phase 2, step 3: batch multiple expert reads into ONE io_uring submission, built on
// the aligned-window technique from expert_reader.cpp (already verified byte-correct).
// Mirrors nvmoe's iouring_ffi.py FastIoUringBatch.read_batch, ported to native liburing.
#include <liburing.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cerrno>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

static constexpr size_t ALIGN = 4096;
static inline size_t align_down(size_t x) { return x - (x % ALIGN); }
static inline size_t align_up(size_t x)   { return ((x + ALIGN - 1) / ALIGN) * ALIGN; }

struct expert_job {
    size_t logical_offset;
    size_t len;
    uint8_t * out;          // caller-owned destination buffer, `len` bytes
    // filled in by prepare():
    uint8_t * aligned_buf = nullptr;
    size_t    front_pad   = 0;
};

// Submits all jobs in ONE io_uring_submit_and_wait call, reaps all CQEs, then copies
// each job's needed sub-range out of its aligned staging buffer into job.out.
static bool read_batch_odirect(int fd_direct, std::vector<expert_job> & jobs) {
    const unsigned n = jobs.size();
    if (n == 0) return true;

    io_uring ring;
    if (io_uring_queue_init(std::max(8u, n), &ring, 0) < 0) {
        fprintf(stderr, "io_uring_queue_init failed: %s\n", strerror(errno));
        return false;
    }

    for (unsigned i = 0; i < n; i++) {
        expert_job & j = jobs[i];
        size_t aligned_off = align_down(j.logical_offset);
        j.front_pad         = j.logical_offset - aligned_off;
        size_t aligned_len  = align_up(j.front_pad + j.len);

        if (posix_memalign((void **) &j.aligned_buf, ALIGN, aligned_len) != 0) {
            fprintf(stderr, "posix_memalign failed for job %u\n", i);
            io_uring_queue_exit(&ring);
            return false;
        }

        io_uring_sqe * sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, fd_direct, j.aligned_buf, aligned_len, aligned_off);
        io_uring_sqe_set_data64(sqe, i);
    }

    int rc = io_uring_submit_and_wait(&ring, n);
    if (rc < 0) {
        fprintf(stderr, "io_uring_submit_and_wait failed: %s\n", strerror(-rc));
        io_uring_queue_exit(&ring);
        return false;
    }

    bool all_ok = true;
    for (unsigned reaped = 0; reaped < n; reaped++) {
        io_uring_cqe * cqe = nullptr;
        int wrc = io_uring_wait_cqe(&ring, &cqe);
        if (wrc < 0 || !cqe) {
            fprintf(stderr, "io_uring_wait_cqe failed: %s\n", strerror(-wrc));
            all_ok = false;
            continue;
        }
        uint64_t idx = io_uring_cqe_get_data64(cqe);
        int res = cqe->res;
        io_uring_cqe_seen(&ring, cqe);

        expert_job & j = jobs[idx];
        if (res < 0 || (size_t) res < j.front_pad + j.len) {
            fprintf(stderr, "job %llu short/failed read: res=%d\n", (unsigned long long) idx, res);
            all_ok = false;
            continue;
        }
        memcpy(j.out, j.aligned_buf + j.front_pad, j.len);
    }

    for (auto & j : jobs) {
        free(j.aligned_buf);
    }
    io_uring_queue_exit(&ring);
    return all_ok;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <file> <base_offset> <row_bytes> [expert_id...]\n", argv[0]);
        return 1;
    }
    const char * path  = argv[1];
    size_t base_offset = strtoull(argv[2], nullptr, 10);
    size_t row_bytes   = strtoull(argv[3], nullptr, 10);

    int fd_direct   = open(path, O_RDONLY | O_DIRECT);
    int fd_buffered = open(path, O_RDONLY);
    if (fd_direct < 0 || fd_buffered < 0) {
        fprintf(stderr, "open failed: %s\n", strerror(errno));
        return 1;
    }

    std::vector<int> expert_ids;
    if (argc > 4) {
        for (int i = 4; i < argc; i++) expert_ids.push_back(atoi(argv[i]));
    } else {
        expert_ids = {0, 1, 2, 63, 64, 127, 128, 200, 255};
    }

    std::vector<std::vector<uint8_t>> outbufs(expert_ids.size(), std::vector<uint8_t>(row_bytes));
    std::vector<expert_job> jobs;
    for (size_t i = 0; i < expert_ids.size(); i++) {
        jobs.push_back({ base_offset + (size_t) expert_ids[i] * row_bytes, row_bytes, outbufs[i].data() });
    }

    bool ok = read_batch_odirect(fd_direct, jobs);
    printf("read_batch_odirect: %zu jobs, all_ok=%d\n", jobs.size(), ok);

    int mismatches = 0;
    for (size_t i = 0; i < expert_ids.size(); i++) {
        std::vector<uint8_t> ref(row_bytes);
        ssize_t n = pread(fd_buffered, ref.data(), row_bytes, base_offset + (size_t) expert_ids[i] * row_bytes);
        bool match = (n == (ssize_t) row_bytes) && memcmp(ref.data(), outbufs[i].data(), row_bytes) == 0;
        if (!match) mismatches++;
        printf("expert %3d: match_vs_buffered=%s\n", expert_ids[i], match ? "YES" : "NO");
    }
    printf("summary: %d mismatches out of %zu experts (single-syscall batch of %zu)\n",
           mismatches, expert_ids.size(), jobs.size());

    close(fd_direct);
    close(fd_buffered);
    return mismatches == 0 ? 0 : 2;
}
