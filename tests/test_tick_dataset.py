"""Reference-oracle diff for the tick-capture extension: the naive
row-oriented reader (plain Python struct unpacking) and the memory-mapped
column reader must return bit-identical samples and labels for the same
underlying data, since they are two reads of the same tick stream, not
two different datasets. Also checks dataset length boundaries and the
background prefetcher's batch ordering. Standalone script, matching this
repo's existing tests/ convention (no pytest dependency).
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tick_capture import generate_ticks, write_column_capture, write_row_capture  # noqa: E402
from tick_data import MmapColumnTickDataset, NaiveRowTickDataset, WINDOW  # noqa: E402


def _build_captures(tmp_dir, n_ticks=5000, seed=123):
    price, size, side, dt = generate_ticks(n_ticks, seed=seed)
    col_path = Path(tmp_dir) / "col.bin"
    row_path = Path(tmp_dir) / "row.bin"
    write_column_capture(col_path, price, size, side, dt)
    write_row_capture(row_path, price, size, side, dt)
    return col_path, row_path, (price, size, side, dt)


def test_lengths_agree(tmp_dir):
    col_path, row_path, (price, *_rest) = _build_captures(tmp_dir)
    col_ds = MmapColumnTickDataset(col_path, window=WINDOW)
    row_ds = NaiveRowTickDataset(row_path, window=WINDOW)
    assert len(col_ds) == len(row_ds)
    assert len(col_ds) == len(price) - WINDOW - 1
    print(f"OK: lengths agree ({len(col_ds)})")


def test_samples_match_exactly(tmp_dir):
    col_path, row_path, _ = _build_captures(tmp_dir)
    col_ds = MmapColumnTickDataset(col_path, window=WINDOW)
    row_ds = NaiveRowTickDataset(row_path, window=WINDOW)
    for idx in (0, 1, 17, 500, 4486):
        assert idx < len(col_ds)
        x_col, y_col = col_ds[idx]
        x_row, y_row = row_ds[idx]
        assert torch.equal(x_col, x_row), f"sample mismatch at idx={idx}"
        assert y_col == y_row, f"label mismatch at idx={idx}"
    print("OK: naive row reader and mmap column reader agree exactly at 5 indices")


def test_samples_match_against_raw_arrays(tmp_dir):
    col_path, _row_path, (price, size, side, dt) = _build_captures(tmp_dir)
    ds = MmapColumnTickDataset(col_path, window=WINDOW)
    idx = 1234
    x, y = ds[idx]
    expected = np.stack(
        [price[idx : idx + WINDOW], size[idx : idx + WINDOW],
         side[idx : idx + WINDOW], dt[idx : idx + WINDOW]]
    )
    assert np.allclose(x.numpy(), expected)
    expected_label = 1 if price[idx + WINDOW] > price[idx + WINDOW - 1] else 0
    assert y == expected_label
    print("OK: dataset sample matches the raw generator arrays directly")


def test_last_valid_index_does_not_read_past_eof(tmp_dir):
    col_path, row_path, _ = _build_captures(tmp_dir)
    col_ds = MmapColumnTickDataset(col_path, window=WINDOW)
    row_ds = NaiveRowTickDataset(row_path, window=WINDOW)
    last = len(col_ds) - 1
    x_col, y_col = col_ds[last]
    x_row, y_row = row_ds[last]
    assert x_col.shape == (4, WINDOW)
    assert torch.equal(x_col, x_row)
    assert y_col == y_row
    print(f"OK: last valid index ({last}) reads exactly, no EOF overrun")


def test_background_prefetcher_preserves_order(tmp_dir):
    if not torch.cuda.is_available():
        print("SKIP: background prefetcher test requires a CUDA GPU")
        return
    from tick_prefetcher import BackgroundPrefetcher

    col_path, _row_path, _ = _build_captures(tmp_dir)
    ds = MmapColumnTickDataset(col_path, window=WINDOW)
    batch_size = 8
    prefetcher = BackgroundPrefetcher(ds, batch_size, device="cuda", queue_size=2)
    try:
        xb0, _ = prefetcher.next()
        xb1, _ = prefetcher.next()
        expected0 = torch.stack([ds[i][0] for i in range(batch_size)]).cuda()
        expected1 = torch.stack([ds[i][0] for i in range(batch_size, 2 * batch_size)]).cuda()
        assert torch.allclose(xb0, expected0)
        assert torch.allclose(xb1, expected1)
        print("OK: background prefetcher delivers batches in dataset order")
    finally:
        prefetcher.stop()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_lengths_agree(tmp_dir)
        test_samples_match_exactly(tmp_dir)
        test_samples_match_against_raw_arrays(tmp_dir)
        test_last_valid_index_does_not_read_past_eof(tmp_dir)
        test_background_prefetcher_preserves_order(tmp_dir)
    print("\nALL PASS: test_tick_dataset.py")
