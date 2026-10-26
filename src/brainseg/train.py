"""Training entry point: ``python -m brainseg.train --config configs/default.yaml``."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from .config import Config
from .data import find_pairs, load_arrays, make_dataset, patient_split
from .evaluate import evaluate_arrays
from .losses import CUSTOM_OBJECTS, dice_coefficient, get_loss, iou_score
from .model import build_unet
from .predict import Segmenter
from .viz import plot_history, plot_predictions


@dataclass
class Splits:
    """In-memory data for one experiment: ``name -> (images uint8, masks {0,1}, patients)``."""

    train: tuple[np.ndarray, np.ndarray]
    val: tuple[np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray, pd.Series | None]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def configure_runtime(mixed_precision: bool) -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        with contextlib.suppress(RuntimeError):  # raises if the runtime is already initialised
            tf.config.experimental.set_memory_growth(gpu, True)
    keras.mixed_precision.set_global_policy("mixed_float16" if mixed_precision else "float32")


def load_splits(cfg: Config) -> Splits:
    df = find_pairs(cfg.data.root)
    tr, va, te = patient_split(df, cfg.data.val_fraction, cfg.data.test_fraction, cfg.data.seed)
    size = cfg.data.image_size
    print(
        f"patients  train/val/test = {tr.patient.nunique()}/{va.patient.nunique()}/{te.patient.nunique()}"
        f"   slices = {len(tr)}/{len(va)}/{len(te)}"
    )
    return Splits(load_arrays(tr, size), load_arrays(va, size), (*load_arrays(te, size), te["patient"]))


def make_optimizer(cfg: Config, steps_per_epoch: int) -> keras.optimizers.Optimizer:
    lr = cfg.train.learning_rate
    if cfg.train.lr_schedule == "cosine":
        lr = keras.optimizers.schedules.CosineDecay(
            lr, decay_steps=max(1, cfg.train.epochs * steps_per_epoch), alpha=0.01
        )
    elif cfg.train.lr_schedule not in ("plateau", "constant"):
        raise ValueError(f"Unknown lr_schedule '{cfg.train.lr_schedule}'")
    return keras.optimizers.Adam(lr)


def compile_model(cfg: Config, steps_per_epoch: int = 100) -> keras.Model:
    s = cfg.data.image_size
    model = build_unet(
        (s, s, 3), cfg.model.base_filters, cfg.model.depth, cfg.model.dropout, cfg.model.batch_norm
    )
    model.compile(
        optimizer=make_optimizer(cfg, steps_per_epoch),
        loss=get_loss(cfg.train.loss),
        metrics=[dice_coefficient, iou_score],
        jit_compile=False,  # XLA raises peak memory; keep 4 GB GPUs predictable
    )
    return model


def run_training(cfg: Config, out_dir: str | Path, splits: Splits | None = None) -> dict:
    """Train, then evaluate the *best* checkpoint on the untouched test split."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.train.seed)
    configure_runtime(cfg.train.mixed_precision)
    cfg.save(out / "config.yaml")

    splits = splits or load_splits(cfg)
    train_ds = make_dataset(*splits.train, cfg.train.batch_size, cfg.data.augment, True, cfg.data.seed)
    val_ds = make_dataset(*splits.val, cfg.train.batch_size)

    model = compile_model(cfg, math.ceil(len(splits.train[0]) / cfg.train.batch_size))
    best = out / "best.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            best, monitor="val_dice_coefficient", mode="max", save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_dice_coefficient",
            mode="max",
            patience=cfg.train.early_stopping_patience,
            restore_best_weights=True,
        ),
        keras.callbacks.CSVLogger(out / "history.csv"),
    ]
    if cfg.train.lr_schedule == "plateau":
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=cfg.train.lr_patience, min_lr=1e-6
            )
        )
    t0 = time.perf_counter()
    model.fit(train_ds, validation_data=val_ds, epochs=cfg.train.epochs, callbacks=callbacks, verbose=2)
    train_seconds = time.perf_counter() - t0

    history = pd.read_csv(out / "history.csv")
    plot_history(history, out / "training_curves.png")

    seg = Segmenter(keras.models.load_model(best, custom_objects=CUSTOM_OBJECTS, compile=False))
    images, masks, patients = splits.test
    summary, per_slice, probs = evaluate_arrays(seg, images, masks, patients, cfg.train.batch_size)
    summary.update(
        train_seconds=train_seconds,
        epochs_run=int(len(history)),
        params=int(model.count_params()),
        best_val_dice=float(history["val_dice_coefficient"].max()),
    )
    (out / "test_metrics.json").write_text(json.dumps(summary, indent=2))
    per_slice.to_csv(out / "test_per_slice.csv", index=False)
    # show tumour-bearing slices first so the grid is informative
    order = np.argsort(~per_slice["has_tumor"].to_numpy(), kind="stable")[:6]
    plot_predictions(images[order], masks[order], probs[order], out / "test_predictions.png")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train the brain-tumour U-Net.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-dir", default=None, help="override data.root")
    ap.add_argument("--out", default=None, help="output dir (default runs/<timestamp>)")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    if args.data_dir:
        cfg.data.root = args.data_dir
    if args.epochs:
        cfg.train.epochs = args.epochs
    out = args.out or f"runs/{time.strftime('%Y%m%d-%H%M%S')}"
    print(json.dumps(run_training(cfg, out), indent=2))


if __name__ == "__main__":
    main()
