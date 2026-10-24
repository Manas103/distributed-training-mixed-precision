"""Two Dataset implementations over the identical underlying tick stream
(see tick_capture.py), differing only in storage layout and read path:

- `MmapColumnTickDataset`: memory-mapped column file, NumPy-vectorized
  window slicing.
- `NaiveRowTickDataset`: row-oriented file, per-record Python struct
  unpacking.

Both must return bit-identical samples for the same index -- that is the
reference-oracle property `tests/test_tick_dataset.py` checks, the same
discipline the rest of this portfolio applies to a fast path vs a
deliberately slow, obviously-correct one.
"""
import struct

import numpy as np
import torch
from torch.utils.data import Dataset

from tick_capture import ColumnTickStore, read_header, read_window_row_naive

WINDOW = 512


def _label(price_at_window_end, price_before_it):
    return 1 if price_at_window_end > price_before_it else 0


class MmapColumnTickDataset(Dataset):
    def __init__(self, path, window=WINDOW):
        self.store = ColumnTickStore(path)
        self.window = window

    def __len__(self):
        return max(0, self.store.n - self.window - 1)

    def __getitem__(self, idx):
        w = self.store.window(idx, self.window)  # (4, window) view, zero-copy
        x = torch.from_numpy(np.ascontiguousarray(w))
        price_row = w[0]
        y = _label(self.store.mm[0, idx + self.window], price_row[-1])
        return x, y


class NaiveRowTickDataset(Dataset):
    def __init__(self, path, window=WINDOW):
        self.path = path
        self.n = read_header(path)
        self.window = window
        self._handle = None

    def __len__(self):
        return max(0, self.n - self.window - 1)

    def _ensure_open(self):
        if self._handle is None:
            self._handle = open(self.path, "rb")
        return self._handle

    def __getitem__(self, idx):
        handle = self._ensure_open()
        rows = read_window_row_naive(handle, idx, self.window)
        # rows: window-length list of (price, size, side, dt) tuples.
        # Transposing in pure Python (no NumPy) is itself part of the
        # naive path's real cost.
        cols = [[0.0] * self.window for _ in range(4)]
        for t, rec in enumerate(rows):
            for c in range(4):
                cols[c][t] = rec[c]
        x = torch.tensor(cols, dtype=torch.float32)

        handle.seek(8 + (idx + self.window) * 4 * 4)
        next_price = struct.unpack("<f", handle.read(4))[0]
        y = _label(next_price, cols[0][-1])
        return x, y

    def __del__(self):
        if self._handle is not None:
            self._handle.close()
