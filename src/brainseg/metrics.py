"""Hard (thresholded) NumPy evaluation metrics, reported per slice, per patient and overall."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _binarise(a: np.ndarray, threshold: float | None) -> np.ndarray:
    a = np.asarray(a)
    return (a > threshold if threshold is not None else a > 0.5).reshape(len(a), -1)


def slice_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, smooth: float = 1e-7
) -> pd.DataFrame:
    """Per-slice Dice / IoU / precision / recall. Empty-vs-empty slices score a perfect 1."""
    t, p = _binarise(y_true, 0.5), _binarise(y_prob, threshold)
    tp = (t & p).sum(1).astype(np.float64)
    fp = (~t & p).sum(1)
    fn = (t & ~p).sum(1)
    empty_both = (t.sum(1) == 0) & (p.sum(1) == 0)
    dice = np.where(empty_both, 1.0, (2 * tp + smooth) / (2 * tp + fp + fn + smooth))
    iou = np.where(empty_both, 1.0, (tp + smooth) / (tp + fp + fn + smooth))
    precision = np.where(p.sum(1) == 0, np.where(t.sum(1) == 0, 1.0, 0.0), tp / np.maximum(tp + fp, 1))
    recall = np.where(t.sum(1) == 0, np.where(p.sum(1) == 0, 1.0, 0.0), tp / np.maximum(tp + fn, 1))
    return pd.DataFrame(
        {
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "has_tumor": t.sum(1) > 0,
            "pred_tumor": p.sum(1) > 0,
        }
    )


def summarize(per_slice: pd.DataFrame, patients: pd.Series | None = None) -> dict[str, float]:
    """Headline numbers.

    * ``dice`` / ``iou``: mean over *all* slices (empty-vs-empty counts as 1).
    * ``dice_tumor_slices``: mean over slices that truly contain tumour (the harder, more honest number).
    * ``false_positive_rate_empty``: share of tumour-free slices where the model hallucinated tumour.
    * ``dice_patient_macro``: mean of per-patient mean Dice (each patient weighs equally).
    """
    tumor = per_slice[per_slice["has_tumor"]]
    empty = per_slice[~per_slice["has_tumor"]]
    out = {
        "n_slices": int(len(per_slice)),
        "dice": float(per_slice["dice"].mean()),
        "iou": float(per_slice["iou"].mean()),
        "precision": float(per_slice["precision"].mean()),
        "recall": float(per_slice["recall"].mean()),
        "dice_tumor_slices": float(tumor["dice"].mean()) if len(tumor) else float("nan"),
        "iou_tumor_slices": float(tumor["iou"].mean()) if len(tumor) else float("nan"),
        "false_positive_rate_empty": float(empty["pred_tumor"].mean()) if len(empty) else float("nan"),
        "detection_recall": float(tumor["pred_tumor"].mean()) if len(tumor) else float("nan"),
    }
    if patients is not None:
        out["dice_patient_macro"] = float(
            per_slice.assign(patient=patients.to_numpy()).groupby("patient")["dice"].mean().mean()
        )
        out["n_patients"] = int(patients.nunique())
    return out
