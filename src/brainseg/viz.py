"""Plotting helpers: mask overlays, training curves and prediction grids."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 40, 40),
    alpha: float = 0.45,
    contour: bool = True,
) -> np.ndarray:
    """Blend a binary mask over an RGB uint8 image; draws a crisp outline for readability."""
    out = image.copy()
    m = mask.astype(bool)
    if m.any():
        tint = np.zeros_like(out)
        tint[:] = color
        out[m] = (out[m] * (1 - alpha) + tint[m] * alpha).astype(np.uint8)
        if contour:
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cs, -1, color, 1)
    return out


def plot_history(history: pd.DataFrame, path: str | Path) -> None:
    """Loss / Dice / IoU curves for train vs validation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, title) in zip(
        axes, [("loss", "Loss"), ("dice_coefficient", "Dice"), ("iou_score", "IoU")], strict=True
    ):
        if key in history:
            ax.plot(history[key], label="train", lw=2)
        if f"val_{key}" in history:
            ax.plot(history[f"val_{key}"], label="validation", lw=2)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_predictions(
    images: np.ndarray,
    masks: np.ndarray,
    probs: np.ndarray,
    path: str | Path,
    n: int = 6,
    threshold: float = 0.5,
) -> None:
    """Grid: MRI | ground truth | prediction overlay, for the first ``n`` slices."""
    n = min(n, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for i in range(n):
        gt = masks[i, ..., 0] > 0.5
        pr = probs[i, ..., 0] > threshold
        axes[i, 0].imshow(images[i])
        axes[i, 1].imshow(overlay(images[i], gt, (40, 220, 40)))
        axes[i, 2].imshow(overlay(images[i], pr))
        for ax, title in zip(axes[i], ["MRI", "Ground truth", "Prediction"], strict=True):
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
