"""Single-process, single-GPU fp32-vs-mixed-precision step-time comparison.
Deliberately not distributed: DDP process-count overhead would confound a
clean apples-to-apples AMP measurement, so this benchmark isolates it.

Usage: python bench/bench_amp.py
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import SyntheticImageDataset  # noqa: E402
from model import SmallCNN  # noqa: E402


def run(amp_enabled, steps=60, warmup=10, batch_size=256, device="cuda"):
    torch.manual_seed(0)
    model = SmallCNN().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loss_fn = nn.CrossEntropyLoss()

    ds = SyntheticImageDataset(batch_size * (steps + warmup), seed=0, cpu_cost="low")
    x_all = torch.stack([ds[i][0] for i in range(batch_size)]).to(device)
    y_all = torch.randint(0, 10, (batch_size,), device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            loss = loss_fn(model(x_all), y_all)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    peak_mem_before = torch.cuda.max_memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    start = time.time()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    peak_mem = torch.cuda.max_memory_allocated(device)
    step_time_ms = (elapsed / steps) * 1000
    throughput = (steps * batch_size) / elapsed
    return step_time_ms, throughput, peak_mem


def main():
    assert torch.cuda.is_available(), "this benchmark requires a CUDA GPU"
    print(f"device: {torch.cuda.get_device_name(0)}")

    fp32_ms, fp32_thr, fp32_mem = run(amp_enabled=False)
    print(f"fp32:  step_time={fp32_ms:.3f} ms  throughput={fp32_thr:.1f} samples/s  "
          f"peak_mem={fp32_mem / 1e6:.1f} MB")

    amp_ms, amp_thr, amp_mem = run(amp_enabled=True)
    print(f"amp:   step_time={amp_ms:.3f} ms  throughput={amp_thr:.1f} samples/s  "
          f"peak_mem={amp_mem / 1e6:.1f} MB")

    speedup = fp32_ms / amp_ms
    mem_reduction = 1 - (amp_mem / fp32_mem)
    print(f"\nspeedup: {speedup:.2f}x   peak-memory reduction: {mem_reduction * 100:.1f}%")


if __name__ == "__main__":
    main()
