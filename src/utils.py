import os
import torch
import torch.distributed as dist


def dist_env():
    """Reads the rank/world-size env vars torchrun sets."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def setup_distributed(backend="nccl"):
    rank, local_rank, world_size = dist_env()
    if world_size > 1:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    # Only one physical GPU is present on this machine, so every rank maps
    # onto device 0 -- see README "Hardware limitations". NCCL 2.21 hard-
    # rejects two ranks sharing one physical device ("Duplicate GPU
    # detected", confirmed by testing -- there is no documented override),
    # so this project defaults --backend to nccl (the correct choice on
    # real multi-GPU hardware) but the README-documented single-GPU runs
    # use --backend gloo, which has no such restriction and still drives a
    # genuine multi-process collective / DDP gradient-sync code path.
    device_count = torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank % device_count}") if device_count else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, local_rank, world_size, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0
