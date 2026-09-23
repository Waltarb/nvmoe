"""
Minimal ctypes wrapper around liburing-ffi.so for one purpose: submit async
reads and reap their completions, single-threaded, no dedicated OS thread
per in-flight I/O (unlike the ThreadPoolExecutor path in bench.py).

Uses liburing-ffi.so specifically because normal liburing.h helper functions
(io_uring_get_sqe, io_uring_prep_read, etc.) are `static inline` in the
public header and not exported as real symbols in plain liburing.so --
liburing-ffi.so exists precisely to give FFI consumers (like this) real
exported symbols for them. Confirmed present via:
  pacman -Ql liburing | grep ffi
  nm -D /usr/lib/liburing-ffi.so.2 | grep -E "queue_init|prep_read|wait_cqe"

We treat `struct io_uring` as an opaque blob (we never read/write its fields
ourselves -- only the library touches it) and only define the one ABI-stable
struct we do need to read directly: `struct io_uring_cqe` (see
/usr/include/liburing/io_uring.h), which is just
    __u64 user_data; __s32 res; __u32 flags;
"""
import ctypes
import itertools
import os

_lib = ctypes.CDLL("liburing-ffi.so.2")

# Real struct io_uring is small (two io_uring_sq/io_uring_cq sub-structs of
# pointers/unsigneds plus a few flags) -- comfortably under 256 bytes across
# liburing versions. We over-allocate to 512 bytes of zeroed, malloc-aligned
# memory and never touch it ourselves; only the library's functions do.
_RING_STRUCT_SIZE = 512


class _Cqe(ctypes.Structure):
    _fields_ = [
        ("user_data", ctypes.c_uint64),
        ("res", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
    ]


_lib.io_uring_queue_init.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
_lib.io_uring_queue_init.restype = ctypes.c_int

_lib.io_uring_queue_exit.argtypes = [ctypes.c_void_p]
_lib.io_uring_queue_exit.restype = None

_lib.io_uring_get_sqe.argtypes = [ctypes.c_void_p]
_lib.io_uring_get_sqe.restype = ctypes.c_void_p

_lib.io_uring_prep_read.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint64,
]
_lib.io_uring_prep_read.restype = None

_lib.io_uring_sqe_set_data64.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
_lib.io_uring_sqe_set_data64.restype = None

_lib.io_uring_submit.argtypes = [ctypes.c_void_p]
_lib.io_uring_submit.restype = ctypes.c_int

_lib.io_uring_wait_cqe.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
_lib.io_uring_wait_cqe.restype = ctypes.c_int

_lib.io_uring_peek_cqe.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
_lib.io_uring_peek_cqe.restype = ctypes.c_int

_lib.io_uring_cqe_seen.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_lib.io_uring_cqe_seen.restype = None


class IoUringError(RuntimeError):
    pass


class IoUring:
    """One ring. Single-threaded use only: submit and reap from the same
    thread (that's all bench.py needs -- there's no separate submitter and
    reaper thread here, so no locking is required)."""

    def __init__(self, queue_depth=64):
        self._ring_buf = ctypes.create_string_buffer(_RING_STRUCT_SIZE)
        self._ring = ctypes.addressof(self._ring_buf)
        rc = _lib.io_uring_queue_init(queue_depth, self._ring, 0)
        if rc < 0:
            raise IoUringError(f"io_uring_queue_init failed: {os.strerror(-rc)}")
        self._next_id = itertools.count(1)
        self._pending_bufs = {}  # user_data -> (ctypes buffer, nbytes)
        self._completed = {}     # user_data -> res

    def submit_read(self, fd, nbytes, offset):
        """Enqueue a read; returns a user_data token to wait on later."""
        sqe = _lib.io_uring_get_sqe(self._ring)
        if not sqe:
            # SQ full -- flush what's queued and retry once.
            _lib.io_uring_submit(self._ring)
            sqe = _lib.io_uring_get_sqe(self._ring)
            if not sqe:
                raise IoUringError("submission queue full even after flush")
        buf = ctypes.create_string_buffer(nbytes)
        udata = next(self._next_id)
        _lib.io_uring_prep_read(sqe, fd, ctypes.addressof(buf), nbytes, offset)
        _lib.io_uring_sqe_set_data64(sqe, udata)
        self._pending_bufs[udata] = buf
        rc = _lib.io_uring_submit(self._ring)
        if rc < 0:
            raise IoUringError(f"io_uring_submit failed: {os.strerror(-rc)}")
        return udata

    def _reap_one(self, blocking):
        cqe_ptr = ctypes.c_void_p()
        fn = _lib.io_uring_wait_cqe if blocking else _lib.io_uring_peek_cqe
        rc = fn(self._ring, ctypes.byref(cqe_ptr))
        if rc < 0:
            if not blocking and rc == -11:  # -EAGAIN: nothing ready
                return False
            raise IoUringError(f"io_uring_{'wait' if blocking else 'peek'}_cqe "
                                f"failed: {os.strerror(-rc)}")
        cqe = ctypes.cast(cqe_ptr, ctypes.POINTER(_Cqe)).contents
        self._completed[cqe.user_data] = cqe.res
        _lib.io_uring_cqe_seen(self._ring, cqe_ptr)
        return True

    def wait(self, udata):
        """Block until the given request completes; returns bytes read
        (raises on a negative errno result) and frees its buffer."""
        while udata not in self._completed:
            self._reap_one(blocking=True)
        res = self._completed.pop(udata)
        self._pending_bufs.pop(udata, None)
        if res < 0:
            raise IoUringError(f"read failed: {os.strerror(-res)}")
        return res

    def close(self):
        _lib.io_uring_queue_exit(self._ring)


def selftest():
    """Round-trip a real read against this file itself."""
    ring = IoUring(queue_depth=8)
    try:
        fd = os.open(__file__, os.O_RDONLY)
        try:
            size = min(4096, os.fstat(fd).st_size)
            udata = ring.submit_read(fd, size, 0)
            n = ring.wait(udata)
            assert n == size, (n, size)
            print(f"iouring_ffi selftest OK: read {n} bytes via io_uring")
        finally:
            os.close(fd)
    finally:
        ring.close()


if __name__ == "__main__":
    selftest()
