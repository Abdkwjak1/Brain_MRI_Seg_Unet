"""Procedural brain-MRI phantoms.

Used by tests, benchmarks and the demo so everything can run without the real (non-redistributable)
LGG dataset. Phantoms are *not* a clinical substitute: they only exercise the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def make_phantom(
    size: int = 256, rng: np.random.Generator | None = None, with_tumor: bool | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(image uint8 HxWx3, mask uint8 HxW in {0,255})``."""
    rng = rng or np.random.default_rng()
    if with_tumor is None:
        with_tumor = rng.random() > 0.35
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size

    # head: skull ring + brain parenchyma
    cy, cx = 0.5 + rng.normal(0, 0.02, 2)
    ry, rx = rng.uniform(0.34, 0.40), rng.uniform(0.28, 0.34)
    r = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
    brain = np.clip(1.0 - r, 0, 1) ** 0.3 * (r < 1)
    skull = np.exp(-(((r - 1.08) / 0.05) ** 2)) * 0.8

    texture = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), size / 40)
    base = 0.35 * brain + skull + 0.08 * texture * (r < 1)

    mask = np.zeros((size, size), np.float32)
    if with_tumor:
        for _ in range(int(rng.integers(1, 3))):
            ty = cy + rng.uniform(-0.6, 0.6) * ry
            tx = cx + rng.uniform(-0.6, 0.6) * rx
            a, b = rng.uniform(0.04, 0.12), rng.uniform(0.04, 0.12)
            ang = rng.uniform(0, np.pi)
            dy, dx = yy - ty, xx - tx
            u = dx * np.cos(ang) + dy * np.sin(ang)
            v = -dx * np.sin(ang) + dy * np.cos(ang)
            mask = np.maximum(mask, (((u / a) ** 2 + (v / b) ** 2) < 1).astype(np.float32))
        mask *= (r < 0.95).astype(np.float32)

    soft = cv2.GaussianBlur(mask, (0, 0), 1.5)
    chans = [
        base + 0.15 * soft + rng.normal(0, 0.02, base.shape),  # "pre-contrast"
        base + 0.55 * soft + rng.normal(0, 0.02, base.shape),  # "FLAIR": tumour is bright
        base + 0.30 * soft + rng.normal(0, 0.02, base.shape),  # "post-contrast"
    ]
    image = (np.clip(np.stack(chans, axis=-1), 0, 1) * 255).astype(np.uint8)
    return image, (mask * 255).astype(np.uint8)


def make_batch(n: int, size: int = 256, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``(images uint8 NxHxWx3, masks uint8 NxHxWx1 in {0,255})``."""
    rng = np.random.default_rng(seed)
    pairs = [make_phantom(size, rng) for _ in range(n)]
    images = np.stack([p[0] for p in pairs])
    masks = np.stack([p[1] for p in pairs])[..., None]
    return images, masks


def write_dataset(
    root: str | Path, n_patients: int = 12, slices_per_patient: int = 8, size: int = 256, seed: int = 0
) -> Path:
    """Write phantoms in the same on-disk layout as the Kaggle LGG dataset."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    for p in range(n_patients):
        pid = f"TCGA_XX_{1000 + p}_{20000101 + p}"
        folder = root / pid
        folder.mkdir(parents=True, exist_ok=True)
        for s in range(1, slices_per_patient + 1):
            image, mask = make_phantom(size, rng)
            # cv2 writes BGR; the LGG tifs are read through cv2 too, so round-trip stays consistent
            cv2.imwrite(str(folder / f"{pid}_{s}.tif"), image)
            cv2.imwrite(str(folder / f"{pid}_{s}_mask.tif"), mask)
    return root
