"""Confirms gradient accumulation over N micro-batches produces (within
floating-point tolerance) the same gradients as one full-batch step. CPU
only, fixed seeds, no distributed process group needed."""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import SmallCNN  # noqa: E402


def make_model():
    torch.manual_seed(0)
    return SmallCNN(num_classes=4, width=8)


def test_grad_accum_matches_full_batch():
    torch.manual_seed(0)
    x = torch.randn(8, 3, 16, 16)
    y = torch.randint(0, 4, (8,))
    loss_fn = nn.CrossEntropyLoss()

    # BatchNorm normalizes against the *current* batch's statistics in
    # train() mode, so a full batch of 8 and four micro-batches of 2 are
    # not mathematically equivalent there by construction -- that's a real
    # BatchNorm property, not a grad-accumulation bug. eval() freezes BN to
    # its (identical, freshly-initialized) running stats on both models,
    # which isolates exactly what this test checks: that accumulating
    # scaled per-micro-batch gradients equals one full-batch gradient.
    full_model = make_model().eval()
    full_model.zero_grad(set_to_none=True)
    loss_full = loss_fn(full_model(x), y)
    loss_full.backward()
    full_grads = [p.grad.clone() for p in full_model.parameters()]

    accum_model = make_model().eval()
    accum_model.zero_grad(set_to_none=True)
    micro_batches = 4
    step = x.shape[0] // micro_batches
    for i in range(micro_batches):
        xb = x[i * step:(i + 1) * step]
        yb = y[i * step:(i + 1) * step]
        loss = loss_fn(accum_model(xb), yb) / micro_batches
        loss.backward()
    accum_grads = [p.grad.clone() for p in accum_model.parameters()]

    assert len(full_grads) == len(accum_grads)
    max_abs_diff = 0.0
    for g_full, g_accum in zip(full_grads, accum_grads):
        diff = (g_full - g_accum).abs().max().item()
        max_abs_diff = max(max_abs_diff, diff)
        assert torch.allclose(g_full, g_accum, atol=1e-5, rtol=1e-4), (
            f"gradient mismatch: max abs diff {diff}"
        )
    print(f"OK: grad-accum matches full-batch gradients, max abs diff = {max_abs_diff:.3e}")


if __name__ == "__main__":
    test_grad_accum_matches_full_batch()
