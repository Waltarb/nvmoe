// Phase 2, step 2: O_DIRECT aligned read of a single expert's bytes out of a GGUF file,
// verified against a plain buffered read of the same logical range for correctness.
// This is the core primitive the NVMe cold-tier will use: base_offset + expert_id*row_bytes
// is usually NOT 4096-aligned in a stock GGUF, so we round the read window out to the
// nearest 4096 boundaries and slice the exact bytes we need out of the aligned buffer.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <vector>
#include <cerrno>

static constexpr size_t ALIGN = 4096;

static inline size_t align_down(size_t x) { return x - (x % ALIGN); }
static inline size_t align_up(size_t x)   { return ((x + ALIGN - 1) / ALIGN) * ALIGN; }

// Reads [logical_offset, logical_offset+len) via O_DIRECT, handling misaligned
// logical_offset/len by reading a padded aligned window and slicing it out.
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

static bool read_buffered(int fd_buffered, size_t logical_offset, size_t len, uint8_t * out) {
    ssize_t n = pread(fd_buffered, out, len, logical_offset);
    return n == (ssize_t) len;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <file> <base_offset> <row_bytes> [expert_id...]\n", argv[0]);
        return 1;
    }
    const char * path       = argv[1];
    size_t base_offset      = strtoull(argv[2], nullptr, 10);
    size_t row_bytes        = strtoull(argv[3], nullptr, 10);

    int fd_buffered = open(path, O_RDONLY);
    int fd_direct   = open(path, O_RDONLY | O_DIRECT);
    if (fd_buffered < 0 || fd_direct < 0) {
        fprintf(stderr, "open failed (buffered=%d direct=%d): %s\n", fd_buffered, fd_direct, strerror(errno));
        return 1;
    }

    std::vector<uint8_t> a(row_bytes), b(row_bytes);

    int n_tests = argc - 4;
    if (n_tests <= 0) {
        // default: test experts 0, 1, 127, 255
        int defaults[] = {0, 1, 127, 255};
        for (int e : defaults) {
            size_t off = base_offset + (size_t) e * row_bytes;
            bool ok_d = read_odirect(fd_direct, off, row_bytes, a.data());
            bool ok_b = read_buffered(fd_buffered, off, row_bytes, b.data());
            bool match = ok_d && ok_b && memcmp(a.data(), b.data(), row_bytes) == 0;
            printf("expert %3d: offset=%zu odirect_ok=%d buffered_ok=%d bytes_match=%s\n",
                   e, off, ok_d, ok_b, match ? "YES" : "NO");
        }
    } else {
        for (int i = 0; i < n_tests; i++) {
            int e = atoi(argv[4 + i]);
            size_t off = base_offset + (size_t) e * row_bytes;
            bool ok_d = read_odirect(fd_direct, off, row_bytes, a.data());
            bool ok_b = read_buffered(fd_buffered, off, row_bytes, b.data());
            bool match = ok_d && ok_b && memcmp(a.data(), b.data(), row_bytes) == 0;
            printf("expert %3d: offset=%zu odirect_ok=%d buffered_ok=%d bytes_match=%s\n",
                   e, off, ok_d, ok_b, match ? "YES" : "NO");
        }
    }

    close(fd_buffered);
    close(fd_direct);
    return 0;
}
