# Distributed Training Pipeline with Mixed Precision

A multi-process, data-parallel PyTorch training pipeline: `DistributedDataParallel`
driven by real NCCL/Gloo collectives, automatic mixed precision, gradient
accumulation, and a `torch.profiler`-driven investigation that found (and fixed) a
genuine host-side data-loading bottleneck. Every number in this README came from an
actual run on the GPU described below -- nothing here is simulated or hand-waved.

## Hardware / environment this was built and measured on

- 1x NVIDIA GeForce RTX 4070 SUPER (12 GB), driver-reported CUDA 13.3, running
  under WSL2 Ubuntu 22.04 with real GPU passthrough (`nvidia-smi` works natively
  inside the WSL guest).
- PyTorch 2.6.0+cu124, CUDA 12.4 toolkit.
- 16 logical CPU cores available to the dataloader workers.

**Only one physical GPU is available on this machine.** See
[Hardware limitations](#hardware-limitations-read-this-first) below for exactly
what that does and does not mean for the claims in this README.

## What's actually implemented

- **`DistributedDataParallel`** wrapping the model, launched with `torchrun`
  across multiple OS processes (`src/train.py`).
- **`DistributedSampler`** so each rank trains on a disjoint shard of the dataset.
- **Real collective communication** for gradient synchronization -- see
  [NCCL vs. Gloo](#nccl-vs-gloo-a-real-finding) for what backend is actually used
  here and why.
- **Mixed precision** via `torch.amp.autocast` + `GradScaler` (`--amp`).
- **Gradient accumulation** (`--grad-accum-steps N`), using `model.no_sync()` to
  skip the DDP all-reduce on non-boundary micro-steps (the standard, and only
  correct, way to combine grad-accum with DDP -- otherwise every micro-step pays
  a full all-reduce for no reason).
- **A profiler-driven bottleneck fix** (`bench/bench_dataloader.py`) -- the
  headline engineering story of this project, see below.

## Quickstart

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 2-process DDP training run, mixed precision + gradient accumulation
cd src
torchrun --standalone --nproc_per_node=2 train.py \
    --epochs 5 --train-samples 20000 --val-samples 2000 \
    --batch-size 128 --grad-accum-steps 2 \
    --num-workers 4 --pin-memory --amp --backend gloo

# benchmarks
python bench/bench_amp.py
python bench/bench_dataloader.py

# correctness tests
python tests/test_grad_accum.py
python tests/test_ddp_sync.py
```

## The model and data

`src/model.py` is a small conv-BN-ReLU CNN (4 stages, adaptive-pool head).
`src/data.py` is a synthetic image classification dataset: each class owns a
fixed random 32x32x3 "template" (shared across the train/val split via a common
`template_seed`), and each sample is that template plus per-sample Gaussian
noise. It's not real image data, but it's a genuine, non-trivial classification
task -- an early smoke test caught a real bug here: giving train and val
different template seeds made the task unlearnable (val accuracy stuck at
random-chance 10% while train loss collapsed to ~0, i.e. the model was
memorizing patterns that meant nothing on val). After fixing the shared-template
bug, a real training run looks like this (2 DDP ranks, AMP on, grad-accum=2,
20k train / 2k val samples):

```
[rank 0] world_size=2 device=cuda:0 amp=True grad_accum_steps=2 batch_size=128
epoch 0: loss=1.6129 val_acc=0.6390 epoch_time=2.33s throughput=8573.9 samples/s
epoch 1: loss=0.1501 val_acc=0.9925 epoch_time=1.34s throughput=14877.5 samples/s
epoch 2: loss=0.0117 val_acc=0.9975 epoch_time=1.32s throughput=15174.5 samples/s
epoch 3: loss=0.0047 val_acc=0.9995 epoch_time=1.33s throughput=15033.3 samples/s
epoch 4: loss=0.0033 val_acc=0.9995 epoch_time=1.32s throughput=15033.3 samples/s
```

## NCCL vs. Gloo: a real finding

The task description for this project called for NCCL specifically, and
`src/train.py --backend nccl` is the default -- that's the correct backend on
real multi-GPU hardware, and is what this code is written for. On this
single-GPU machine, though, running 2 ranks with NCCL fails outright:

```
DESKTOP-IKITC19:747:762 [0] init.cc:943 NCCL WARN Duplicate GPU detected : rank 1 and rank 0 both on CUDA device 1000
torch.distributed.DistBackendError: NCCL error in: .../NCCLUtils.hpp:268, invalid usage
ncclInvalidUsage: This usually reflects invalid usage of NCCL library.
Last error: Duplicate GPU detected : rank 0 and rank 1 both on CUDA device 1000
```

NCCL 2.21 (the version pulled in by this PyTorch build) hard-rejects two ranks
mapping to the same physical device -- there is no documented environment
variable to disable this check; it exists specifically to prevent the topology
assumptions NCCL's ring/tree algorithms make from being silently violated. That
means genuine multi-process NCCL execution requires 2+ physical GPUs, which this
machine doesn't have.

All runs and numbers in this README therefore use `--backend gloo` instead. Gloo
has no such restriction and still drives the *entire* real distributed code
path -- `init_process_group`, `DistributedSampler`, `DDP`'s gradient all-reduce,
`no_sync()` for gradient accumulation -- across genuinely separate OS processes.
It just uses Gloo's collective implementation instead of NCCL's to move the
gradients. `--backend` is a plain CLI flag in `src/train.py`; on a machine with
2+ GPUs, `--backend nccl` is expected to work unmodified.

## Correctness tests

- **`tests/test_grad_accum.py`**: 4 micro-batches of size 2 with gradient
  accumulation must produce the same gradients (within float tolerance) as one
  full batch of size 8. BatchNorm normalizes against the *current* batch's
  statistics in train mode, which would make micro- and full-batch runs
  genuinely different for a reason unrelated to grad-accum correctness, so both
  models run in `eval()` (frozen BN stats) to isolate exactly what's being
  tested. Result: max abs gradient diff `1.49e-08`.
- **`tests/test_ddp_sync.py`**: spawns 2 real processes (Gloo), each training on
  a *different* random data shard, and checks that after one optimizer step
  their parameters are identical -- proof the DDP all-reduce actually
  synchronized gradients across processes rather than each rank training in
  isolation. First version of this test compared full `state_dict()` (including
  BatchNorm's `running_mean`/`running_var` buffers) and failed with a `1.1e-2`
  max diff -- a real, expected result: those buffers are only broadcast from
  rank 0 once at the start of `forward()`, then each rank updates its own local
  copy from its own data, so they're supposed to diverge. Comparing only the
  learnable parameters (the ones DDP's backward hook actually all-reduces) gives
  `0.0` max diff.

## Mixed precision: measured speedup

`bench/bench_amp.py` runs identical single-GPU training steps with and without
`torch.amp.autocast`/`GradScaler`, back to back, same seed, same data:

```
device: NVIDIA GeForce RTX 4070 SUPER
fp32:  step_time=24.560 ms  throughput=10423.5 samples/s  peak_mem=927.6 MB
amp:   step_time=13.021 ms  throughput=19660.9 samples/s  peak_mem=366.4 MB

speedup: 1.89x   peak-memory reduction: 60.5%
```

## The centerpiece: finding and fixing a real data-loading bottleneck

`bench/bench_dataloader.py` gives `SyntheticImageDataset` a real, non-trivial
per-sample CPU cost (`cpu_cost="high"`): a hand-rolled 6-pass box blur plus a
nonlinear color-correction pass, all pure NumPy, no `time.sleep()` anywhere --
this is meant to stand in for a realistic image-augmentation/decode pipeline.

**Before** (`num_workers=0`, `pin_memory=False` -- everything runs serialized on
the main process, so the GPU sits idle waiting for each batch):

```
wall time for 40 steps: 2.87s   throughput: 1782.0 samples/s
```

```
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
enumerate(DataLoader)#_SingleProcessDataLoaderIter._...        91.77%        2.493s        92.50%        2.513s      62.827ms            40
Self CPU time total: 2.717s
```

92.5% of total CPU time is spent inside the `DataLoader` iterator -- almost the
entire step is the main process blocked on `__getitem__`, not the GPU doing
useful work. Full table in `docs/profiler_before.txt`.

**After** (`num_workers=4`, `pin_memory=True`, `persistent_workers=True` --
preprocessing moves to worker processes that run ahead of the GPU, and the
host->device copy becomes async):

```
wall time for 40 steps: 0.82s   throughput: 6221.5 samples/s
```

Full table in `docs/profiler_after.txt`. **Measured speedup: 3.49x**, from one
`DataLoader` configuration change -- no model or algorithm change at all.

## Repository layout

```
src/
  model.py     small CNN (4 conv-BN-ReLU stages + adaptive pool head)
  data.py      synthetic seeded dataset, shared-template train/val split,
               optional real CPU-cost preprocessing
  train.py     DDP + AMP + gradient-accumulation training loop (torchrun entry point)
  utils.py     process-group setup/teardown helpers
bench/
  bench_amp.py          fp32 vs. mixed-precision step-time/memory comparison
  bench_dataloader.py   dataloader before/after profiler comparison
tests/
  test_grad_accum.py    grad-accum vs. full-batch gradient equivalence
  test_ddp_sync.py      DDP gradient all-reduce correctness (2 spawned processes)
docs/
  profiler_before.txt, profiler_after.txt   full torch.profiler tables
```

## Hardware limitations (read this first)

- **Single physical GPU.** Every DDP run in this README uses 2+ OS processes
  that all target `cuda:0` -- the standard way to exercise real multi-process
  distributed training code on a single-GPU dev box. This genuinely runs
  `init_process_group`, `DistributedSampler`, and DDP's real gradient
  all-reduce across separate processes, and `test_ddp_sync.py` proves the
  synchronization is real. It does **not** demonstrate multi-*device* compute
  scaling (2 GPUs training faster than 1) -- that claim isn't made anywhere in
  this README, because it wasn't measured.
- **NCCL doesn't run at all on this hardware** for the reason documented above
  (its own duplicate-GPU safety check) -- all measured numbers use the Gloo
  backend. `--backend nccl` is the default and is expected to work on real
  multi-GPU hardware, but that was not available to test here.
- All throughput/speedup numbers are specific to the RTX 4070 SUPER described
  above and this small CNN/synthetic-data setup; they are not general claims
  about mixed precision or data-loading optimization on other hardware or
  models.
