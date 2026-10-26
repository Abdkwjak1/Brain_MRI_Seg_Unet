"""Differentiable losses / training metrics for binary segmentation.

Dice/IoU/Tversky are computed over the whole batch (flattened), not per slice. A per-slice Dice scores an
empty-vs-empty slice as 1, and since ~65 % of LGG slices are tumour-free the optimiser can collapse to
"predict nothing" (observed: validation Dice stuck at 0.63, the all-background baseline). The flattened form
has no such shortcut. Per-slice behaviour is measured separately in :mod:`brainseg.metrics`.
"""

from __future__ import annotations

import keras
from keras import ops

SMOOTH = 1.0
PKG = "brainseg"


def _flat(y_true, y_pred):
    y_true = ops.cast(y_true, y_pred.dtype)
    return ops.reshape(y_true, (-1,)), ops.reshape(y_pred, (-1,))


@keras.saving.register_keras_serializable(package=PKG)
def dice_coefficient(y_true, y_pred, smooth: float = SMOOTH):
    t, p = _flat(y_true, y_pred)
    inter = ops.sum(t * p)
    return (2.0 * inter + smooth) / (ops.sum(t) + ops.sum(p) + smooth)


@keras.saving.register_keras_serializable(package=PKG)
def iou_score(y_true, y_pred, smooth: float = SMOOTH):
    t, p = _flat(y_true, y_pred)
    inter = ops.sum(t * p)
    union = ops.sum(t) + ops.sum(p) - inter
    return (inter + smooth) / (union + smooth)


@keras.saving.register_keras_serializable(package=PKG)
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


@keras.saving.register_keras_serializable(package=PKG)
def bce_dice_loss(y_true, y_pred):
    """Binary cross-entropy (stable gradients early on) + Dice (optimises the overlap metric)."""
    bce = ops.mean(keras.losses.binary_crossentropy(ops.cast(y_true, y_pred.dtype), y_pred))
    return bce + dice_loss(y_true, y_pred)


@keras.saving.register_keras_serializable(package=PKG)
def focal_tversky_loss(y_true, y_pred, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75):
    """Penalises false negatives more than false positives (alpha > beta); good for small tumours."""
    t, p = _flat(y_true, y_pred)
    tp = ops.sum(t * p)
    fn = ops.sum(t * (1 - p))
    fp = ops.sum((1 - t) * p)
    tversky = (tp + SMOOTH) / (tp + alpha * fn + beta * fp + SMOOTH)
    return ops.power(1.0 - tversky, gamma)


LOSSES = {
    "dice": dice_loss,
    "bce_dice": bce_dice_loss,
    "focal_tversky": focal_tversky_loss,
}

CUSTOM_OBJECTS = {
    "dice_coefficient": dice_coefficient,
    "iou_score": iou_score,
    "dice_loss": dice_loss,
    "bce_dice_loss": bce_dice_loss,
    "focal_tversky_loss": focal_tversky_loss,
}


def get_loss(name: str):
    try:
        return LOSSES[name]
    except KeyError:
        raise ValueError(f"Unknown loss '{name}'. Choose from {sorted(LOSSES)}") from None
