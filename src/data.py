"""Synthetic, seeded image-classification dataset.

Each class owns a fixed random "template" image generated once at dataset
construction time; a sample is that template plus per-sample Gaussian noise.
A CNN can genuinely learn to separate the classes from this (it is not pure
noise), which lets training curves and accuracy numbers in this project be
real measurements rather than props.

The `cpu_cost` knob controls how much host-side work `__getitem__` does
before returning a sample. "high" runs a small hand-rolled multi-pass
Gaussian blur in pure NumPy -- real, non-trivial CPU work (not a sleep())
that stands in for a realistic augmentation/decode pipeline and is used to
reproduce a genuine data-loading bottleneck in bench/bench_dataloader.py.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def _box_blur_pass(img, radius=2):
    # Separable box blur implemented by hand (no scipy/cv2 dependency) --
    # this is real floating point work, sized so a handful of passes take
    # a measurable fraction of a millisecond per image.
    padded = np.pad(img, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    window = 2 * radius + 1
    csum = np.cumsum(padded, axis=0)
    zero_row = np.zeros((1,) + csum.shape[1:], dtype=img.dtype)
    csum = np.concatenate([zero_row, csum], axis=0)
    horiz = (csum[window:] - csum[:-window]) / window
    csum2 = np.cumsum(horiz, axis=1)
    zero_col = np.zeros((csum2.shape[0], 1, csum2.shape[2]), dtype=img.dtype)
    csum2 = np.concatenate([zero_col, csum2], axis=1)
    out = (csum2[:, window:] - csum2[:, :-window]) / window
    return out.astype(img.dtype)


class SyntheticImageDataset(Dataset):
    def __init__(self, n_samples, num_classes=10, image_size=32, channels=3,
                 seed=0, template_seed=1234, noise_std=0.6, cpu_cost="low", blur_passes=6):
        """`template_seed` fixes the per-class visual patterns shared by
        every split; `seed` only controls this split's labels/noise. Train
        and val must share `template_seed` so the classes mean the same
        thing in both splits -- otherwise the task is unlearnable, which is
        exactly the bug an early smoke test here caught (val stuck at
        chance accuracy while train loss went to ~0)."""
        self.n_samples = n_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.channels = channels
        self.noise_std = noise_std
        self.cpu_cost = cpu_cost
        self.blur_passes = blur_passes

        template_rng = np.random.default_rng(template_seed)
        self.templates = template_rng.standard_normal(
            (num_classes, image_size, image_size, channels)
        ).astype(np.float32)

        label_rng = np.random.default_rng(seed + 1)
        self.labels = label_rng.integers(0, num_classes, size=n_samples).astype(np.int64)

        # per-sample seed so __getitem__ is deterministic and reproducible
        # across worker processes (each worker gets a copy of this array).
        self.sample_seeds = label_rng.integers(0, 2**31 - 1, size=n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        label = int(self.labels[idx])
        rng = np.random.default_rng(self.sample_seeds[idx])
        noise = rng.standard_normal(
            (self.image_size, self.image_size, self.channels)
        ).astype(np.float32) * self.noise_std
        img = self.templates[label] + noise

        if self.cpu_cost == "high":
            for _ in range(self.blur_passes):
                img = _box_blur_pass(img, radius=2)
            # a second, unrelated pass of per-pixel nonlinear "color
            # correction" work, again real elementwise float math.
            img = np.tanh(img * 1.3) + 0.05 * np.sin(img * 7.0)

        img = (img - img.mean()) / (img.std() + 1e-6)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
        return torch.from_numpy(img.copy()), label
