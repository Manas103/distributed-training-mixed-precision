"""Reproduces and then fixes a trading-specific input-path bottleneck: a
naive row-oriented tick reader (per-record Python struct unpacking, no
memory mapping, blocking host-to-device copy) versus a memory-mapped
column-oriented reader (NumPy-vectorized window slicing) whose batches are
read and copied to the GPU on a background thread overlapped with compute
on a side CUDA stream. Both configurations run single-process
(num_workers=0, no DataLoader worker pool) so the measured difference is
attributable only to the storage layout and the copy-overlap mechanism,
not to the worker-process fix bench_dataloader.py already measured.

Usage: python bench/bench_tick_pipeline.py [n_ticks] [n_steps]
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tick_capture import generate_ticks, write_column_capture, write_row_capture  # noqa: E402
from tick_data import MmapColumnTickDataset, NaiveRowTickDataset, WINDOW  # noqa: E402
from tick_model import TickCNN  # noqa: E402
from tick_prefetcher import BackgroundPrefetcher  # noqa: E402

BATCH_SIZE = 128


def build_captures(tmp_dir, n_ticks, seed=0):
    price, size, side, dt = generate_ticks(n_ticks, seed=seed)
    col_path = tmp_dir / "ticks_column.bin"
    row_path = tmp_dir / "ticks_row.bin"
    write_column_capture(col_path, price, size, side, dt)
    write_row_capture(row_path, price, size, side, dt)
    return col_path, row_path


def run_naive(row_path, device, n_steps, warmup_steps):
    dataset = NaiveRowTickDataset(row_path, window=WINDOW)
    model = TickCNN().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    n = len(dataset)
    idx = 0

    def one_step():
        nonlocal idx
        xs, ys = [], []
        for _ in range(BATCH_SIZE):
            x, y = dataset[idx % n]
            idx += 1
            xs.append(x)
            ys.append(y)
        xb = torch.stack(xs)
        yb = torch.tensor(ys, dtype=torch.long)
        xb = xb.to(device)  # blocking: no pin_memory, no non_blocking
        yb = yb.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()

    for _ in range(warmup_steps):
        one_step()
    torch.cuda.synchronize()

    start = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n_steps):
            one_step()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    samples_per_sec = (n_steps * BATCH_SIZE) / elapsed
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=12)
    return samples_per_sec, table


def run_optimized(col_path, device, n_steps, warmup_steps):
    dataset = MmapColumnTickDataset(col_path, window=WINDOW)
    model = TickCNN().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    prefetcher = BackgroundPrefetcher(dataset, BATCH_SIZE, device, queue_size=2)

    def one_step():
        xb, yb = prefetcher.next()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()

    try:
        for _ in range(warmup_steps):
            one_step()
        torch.cuda.synchronize()

        start = time.time()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(n_steps):
                one_step()
        torch.cuda.synchronize()
        elapsed = time.time() - start
    finally:
        prefetcher.stop()

    samples_per_sec = (n_steps * BATCH_SIZE) / elapsed
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=12)
    return samples_per_sec, table


def main():
    assert torch.cuda.is_available(), "this benchmark requires a CUDA GPU"
    n_ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    device = "cuda"

    tmp_dir = Path(__file__).resolve().parent.parent / "docs" / "_tick_capture_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    col_path, row_path = build_captures(tmp_dir, n_ticks)

    print(f"generated {n_ticks} ticks, window={WINDOW}, batch_size={BATCH_SIZE}")

    naive_thr, naive_table = run_naive(row_path, device, n_steps, warmup_steps=5)
    print(f"\n===== NAIVE (row-oriented, blocking copy) =====")
    print(f"throughput: {naive_thr:.1f} samples/s")
    print(naive_table)

    opt_thr, opt_table = run_optimized(col_path, device, n_steps, warmup_steps=5)
    print(f"\n===== OPTIMIZED (mmap column, overlapped copy) =====")
    print(f"throughput: {opt_thr:.1f} samples/s")
    print(opt_table)

    print(f"\n===== summary =====")
    print(f"naive:     {naive_thr:.1f} samples/s")
    print(f"optimized: {opt_thr:.1f} samples/s")
    print(f"speedup: {opt_thr / naive_thr:.2f}x")

    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    (docs_dir / "tick_profiler_before.txt").write_text(str(naive_table), encoding="utf-8")
    (docs_dir / "tick_profiler_after.txt").write_text(str(opt_table), encoding="utf-8")

    for p in (col_path, row_path):
        p.unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
