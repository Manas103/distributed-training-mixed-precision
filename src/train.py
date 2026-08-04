"""Distributed data-parallel training with mixed precision and gradient
accumulation. Launch with torchrun, e.g.:

    torchrun --standalone --nproc_per_node=2 src/train.py --epochs 3 --amp

See README.md for the full command list and real measured results.
"""
import argparse
import time

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import SyntheticImageDataset
from model import SmallCNN
from utils import setup_distributed, cleanup_distributed, is_main


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128, help="per-rank micro-batch size")
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--noise-std", type=float, default=3.0)
    p.add_argument("--train-samples", type=int, default=20000)
    p.add_argument("--val-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    return p.parse_args()


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / max(total, 1)


def main():
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed(backend=args.backend)
    torch.manual_seed(args.seed + rank)

    train_ds = SyntheticImageDataset(args.train_samples, seed=args.seed, noise_std=args.noise_std, cpu_cost="low")
    val_ds = SyntheticImageDataset(args.val_samples, seed=args.seed + 1000, noise_std=args.noise_std, cpu_cost="low")

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)

    model = SmallCNN().to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    # DDP.forward() does a collective buffer broadcast, so it must be
    # called from every rank in lockstep. Evaluation below only runs on
    # rank 0 (an early test caught this: calling the DDP-wrapped model
    # from a single rank hangs/breaks the process group), so eval uses the
    # underlying module directly, bypassing DDP's collective forward.
    eval_model = model.module if world_size > 1 else model

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()

    if is_main(rank):
        print(f"[rank {rank}] world_size={world_size} device={device} amp={args.amp} "
              f"grad_accum_steps={args.grad_accum_steps} batch_size={args.batch_size}")

    step = 0
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        running_loss = 0.0
        opt.zero_grad(set_to_none=True)

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            sync = ((i + 1) % args.grad_accum_steps == 0)
            ctx = model.no_sync() if (world_size > 1 and not sync) else _null_ctx()
            with ctx:
                with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                    logits = model(x)
                    loss = loss_fn(logits, y) / args.grad_accum_steps
                scaler.scale(loss).backward()

            if sync:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            running_loss += loss.item() * args.grad_accum_steps
            step += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(train_loader)

        if is_main(rank):
            acc = evaluate(eval_model, val_loader, device)
            samples_per_sec = (len(train_loader) * args.batch_size * world_size) / epoch_time
            print(f"epoch {epoch}: loss={avg_loss:.4f} val_acc={acc:.4f} "
                  f"epoch_time={epoch_time:.2f}s throughput={samples_per_sec:.1f} samples/s")

    cleanup_distributed()


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
