"""Reproduces and then fixes a real host-side data-loading bottleneck,
using torch.profiler to prove it both ways.

"slow" config: num_workers=0, pin_memory=False -- every __getitem__'s real
CPU-side preprocessing cost (see SyntheticImageDataset's cpu_cost="high",
a hand-rolled multi-pass NumPy blur) runs on the main process, serialized
with the GPU step, so the GPU sits idle waiting on the next batch.

"fast" config: num_workers>0, pin_memory=True, persistent_workers=True --
preprocessing moves to worker processes that run ahead of the GPU, and
pinned memory makes the host->device copy async, so the GPU stays fed.

Usage: python bench/bench_dataloader.py
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import SyntheticImageDataset  # noqa: E402
from model import SmallCNN  # noqa: E402


def build_loader(num_workers, pin_memory, batch_size=128, n_samples=6000):
    ds = SyntheticImageDataset(n_samples, seed=0, cpu_cost="high", blur_passes=6)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=num_workers > 0, drop_last=True,
    )


def run_config(name, num_workers, pin_memory, profile_steps=40, warmup_steps=5, device="cuda"):
    loader = build_loader(num_workers, pin_memory)
    model = SmallCNN().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    it = iter(loader)

    def one_step():
        nonlocal it
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x = x.to(device, non_blocking=pin_memory)
        y = y.to(device, non_blocking=pin_memory)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

    for _ in range(warmup_steps):
        one_step()
    torch.cuda.synchronize()

    start = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(profile_steps):
            one_step()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    samples_per_sec = (profile_steps * 128) / elapsed
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=12)

    print(f"\n===== {name}: num_workers={num_workers} pin_memory={pin_memory} =====")
    print(f"wall time for {profile_steps} steps: {elapsed:.2f}s  "
          f"throughput: {samples_per_sec:.1f} samples/s")
    print(table)
    return samples_per_sec, table


def main():
    assert torch.cuda.is_available(), "this benchmark requires a CUDA GPU"
    slow_thr, slow_table = run_config("SLOW (before fix)", num_workers=0, pin_memory=False)
    fast_thr, fast_table = run_config("FAST (after fix)", num_workers=4, pin_memory=True)

    print(f"\n===== summary =====")
    print(f"slow (num_workers=0, pin_memory=False): {slow_thr:.1f} samples/s")
    print(f"fast (num_workers=4, pin_memory=True):  {fast_thr:.1f} samples/s")
    print(f"speedup: {fast_thr / slow_thr:.2f}x")

    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "profiler_before.txt").write_text(str(slow_table), encoding="utf-8")
    (docs_dir / "profiler_after.txt").write_text(str(fast_table), encoding="utf-8")
    print(f"\nwrote {docs_dir / 'profiler_before.txt'} and profiler_after.txt")


if __name__ == "__main__":
    main()
