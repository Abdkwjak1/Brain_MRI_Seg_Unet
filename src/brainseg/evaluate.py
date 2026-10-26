"""Evaluate a trained model on the held-out, patient-disjoint test split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import find_pairs, load_arrays, patient_split
from .metrics import slice_metrics, summarize
from .predict import Segmenter


def evaluate_arrays(
    segmenter: Segmenter,
    images: np.ndarray,
    masks: np.ndarray,
    patients: pd.Series | None = None,
    batch_size: int = 16,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Return ``(summary, per_slice_df, probabilities)`` for uint8 images / {0,1} masks."""
    x = images.astype(np.float32) / 255.0
    t0 = time.perf_counter()
    probs = segmenter.predict_batch(x, batch_size)
    elapsed = time.perf_counter() - t0
    per_slice = slice_metrics(masks, probs, segmenter.threshold)
    summary = summarize(per_slice, patients)
    summary["inference_ms_per_slice"] = 1000 * elapsed / len(x)
    return summary, per_slice, probs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained U-Net on the test split.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=None, help="config used for training (for the split)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="write metrics JSON + per-slice CSV here")
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.data_dir:
        cfg.data.root = args.data_dir
    df = find_pairs(cfg.data.root)
    _, _, test_df = patient_split(df, cfg.data.val_fraction, cfg.data.test_fraction, cfg.data.seed)
    seg = Segmenter.from_path(args.model, args.threshold)
    images, masks = load_arrays(test_df, seg.size)
    summary, per_slice, _ = evaluate_arrays(seg, images, masks, test_df["patient"])
    print(json.dumps(summary, indent=2))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "test_metrics.json").write_text(json.dumps(summary, indent=2))
        per_slice.assign(image=test_df["image"].to_numpy(), patient=test_df["patient"].to_numpy()).to_csv(
            out / "test_per_slice.csv", index=False
        )


if __name__ == "__main__":
    main()
