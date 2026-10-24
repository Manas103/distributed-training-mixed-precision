"""Generates one synthetic tick stream and writes it to disk in two
layouts from the identical underlying data, so bench_tick_pipeline.py's
naive-vs-optimized comparison isolates the storage layout and copy path,
not a difference in the data itself.

Column layout (what the optimized path reads): a small header, then each
field's float32 array written fully contiguous, back to back, so a window
read of one field is a single contiguous memory region a memory-mapped
NumPy array can slice with zero copies.

Row layout (what the naive path reads): a small header, then one
interleaved record per tick (price, size, side, dt), the layout a simple
append-one-record-at-a-time tick logger would actually produce. Reading a
window out of this format means seeking to each record and unpacking its
fields one at a time -- real, measurable Python-level cost, not a sleep().
"""
import struct

import numpy as np

FIELDS = ("price", "size", "side", "dt")
N_FIELDS = len(FIELDS)
HEADER_BYTES = 8  # one little-endian int64: n_ticks


def generate_ticks(n_ticks, seed=0):
    """A synthetic tick stream: price is a bounded random walk, size and
    side are iid, dt is an exponential inter-arrival gap. Not calibrated to
    any real venue; see README "Honest framing"."""
    rng = np.random.default_rng(seed)
    price = (100.0 + np.cumsum(rng.normal(0, 0.02, size=n_ticks))).astype(np.float32)
    size = rng.integers(1, 500, size=n_ticks).astype(np.float32)
    side = rng.integers(0, 2, size=n_ticks).astype(np.float32)
    dt = rng.exponential(0.01, size=n_ticks).astype(np.float32)
    return price, size, side, dt


def write_column_capture(path, price, size, side, dt):
    n = len(price)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", n))
        for col in (price, size, side, dt):
            f.write(np.ascontiguousarray(col, dtype=np.float32).tobytes())


def write_row_capture(path, price, size, side, dt):
    n = len(price)
    rows = np.stack([price, size, side, dt], axis=1).astype(np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", n))
        f.write(np.ascontiguousarray(rows).tobytes())


def read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<q", f.read(HEADER_BYTES))
    return n


class ColumnTickStore:
    """Memory-mapped column reader. `window(start, length)` is a zero-copy
    view into the mmap; nothing is read from disk until a caller actually
    touches those bytes, and the OS page cache serves repeated windows over
    the same region for free."""

    def __init__(self, path):
        self.n = read_header(path)
        self.mm = np.memmap(
            path, dtype=np.float32, mode="r", offset=HEADER_BYTES, shape=(N_FIELDS, self.n)
        )

    def window(self, start, length):
        return self.mm[:, start : start + length]


def read_window_row_naive(handle, start, length):
    """Plain Python seek + read + struct.unpack, one record at a time -- no
    NumPy vectorization. `handle` is an already-open binary file object.
    Returns a (length, N_FIELDS) list of float tuples."""
    record_bytes = N_FIELDS * 4
    handle.seek(HEADER_BYTES + start * record_bytes)
    raw = handle.read(length * record_bytes)
    fmt = "<" + "f" * N_FIELDS
    out = [struct.unpack_from(fmt, raw, i * record_bytes) for i in range(length)]
    return out
