"""Confirms DistributedDataParallel's gradient all-reduce actually
synchronizes parameters across ranks: two ranks see disjoint data shards,
take one optimizer step each, and must end up with identical weights.
Spawns its own processes (gloo backend -- see README on why this project
uses gloo rather than NCCL on this single-GPU machine) so it can run as a
plain `python tests/test_ddp_sync.py`, no torchrun needed."""
import os
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import SmallCNN  # noqa: E402


def test_ddp_gradient_sync_matches_across_ranks():
    world_size = 2
    tmp_dir = Path(__file__).resolve().parent / "_tmp_ddp_sync"
    tmp_dir.mkdir(exist_ok=True)
    rank0_path = tmp_dir / "rank0_state.pt"
    rank1_path = tmp_dir / "rank1_state.pt"

    procs = []
    for rank, path in [(0, rank0_path), (1, rank1_path)]:
        p = mp.Process(target=_worker_and_save, args=(rank, world_size, str(path)))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        assert p.exitcode == 0, f"worker process failed with exit code {p.exitcode}"

    state0 = torch.load(rank0_path, weights_only=True)
    state1 = torch.load(rank1_path, weights_only=True)
    assert state0.keys() == state1.keys()
    max_diff = 0.0
    for k in state0:
        diff = (state0[k] - state1[k]).abs().max().item()
        max_diff = max(max_diff, diff)
    assert max_diff < 1e-6, f"post-step parameters diverged across ranks: max diff {max_diff}"
    print(f"OK: post-step parameters identical across 2 ranks (max diff {max_diff:.3e}), "
          "confirming DDP's gradient all-reduce actually synchronized both ranks "
          "despite each training on a disjoint data shard.")

    for path in (rank0_path, rank1_path):
        path.unlink(missing_ok=True)
    tmp_dir.rmdir()


def _worker_and_save(rank, world_size, save_path):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29601")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    torch.manual_seed(123)
    model = SmallCNN(num_classes=4, width=8)
    model = DDP(model)

    torch.manual_seed(100 + rank)
    x = torch.randn(4, 3, 16, 16)
    y = torch.randint(0, 4, (4,))

    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    opt.zero_grad(set_to_none=True)
    loss_fn(model(x), y).backward()
    opt.step()

    # Only the learnable parameters are covered by DDP's gradient
    # all-reduce guarantee. BatchNorm's running_mean/running_var buffers
    # are broadcast from rank 0 once at the *start* of forward but then
    # each rank updates its own copy locally from its own batch during
    # that forward pass -- so buffers legitimately diverge across ranks
    # that see different data (confirmed: comparing the full state_dict
    # here failed with max diff ~1e-2, entirely from BN buffers; comparing
    # parameters only, both ranks match to float precision).
    state = {k: v.clone() for k, v in model.module.named_parameters()}
    torch.save(state, save_path)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    test_ddp_gradient_sync_matches_across_ranks()
