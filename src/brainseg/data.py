"""Dataset discovery, leakage-free splitting and the ``tf.data`` input pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

MASK_SUFFIX = "_mask"


def patient_id(path: str | Path) -> str:
    """``TCGA_CS_4941_19960909/…`` -> ``TCGA_CS_4941`` (the folder encodes patient + study date)."""
    return "_".join(Path(path).parent.name.split("_")[:3])


def find_pairs(root: str | Path) -> pd.DataFrame:
    """Locate every ``(image, mask)`` pair under ``root`` -> DataFrame[image, mask, patient, has_tumor]."""
    root = Path(root)
    masks = sorted(root.rglob(f"*{MASK_SUFFIX}.tif"))
    if not masks:
        raise FileNotFoundError(f"No '*{MASK_SUFFIX}.tif' files under {root}. See README → 'Get the data'.")
    rows = []
    for m in masks:
        img = m.with_name(m.name.replace(f"{MASK_SUFFIX}.tif", ".tif"))
        if img.exists():
            rows.append((str(img), str(m), patient_id(m)))
    df = pd.DataFrame(rows, columns=["image", "mask", "patient"])
    if df.empty:
        raise FileNotFoundError(f"Found masks under {root} but no matching images.")
    return df


def patient_split(
    df: pd.DataFrame, val_fraction: float = 0.15, test_fraction: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by *patient* so neighbouring slices of one scan never straddle train/val/test."""
    if not 0 < val_fraction + test_fraction < 1:
        raise ValueError("val_fraction + test_fraction must be in (0, 1)")
    patients = np.array(sorted(df["patient"].unique()))
    if len(patients) < 3:
        raise ValueError(f"Need at least 3 patients to split, got {len(patients)}")
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n_test = max(1, round(len(patients) * test_fraction))
    n_val = max(1, round(len(patients) * val_fraction))
    test_p, val_p = patients[:n_test], patients[n_test : n_test + n_val]
    train_p = patients[n_test + n_val :]
    pick = lambda ps: df[df["patient"].isin(ps)].reset_index(drop=True)  # noqa: E731
    return pick(train_p), pick(val_p), pick(test_p)


def load_image(path: str, size: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def load_mask(path: str, size: int) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise OSError(f"Could not read mask: {path}")
    if m.shape != (size, size):
        m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)[..., None]


def load_arrays(df: pd.DataFrame, size: int = 256, workers: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Read a split into RAM: images uint8 (N,S,S,3), masks uint8 (N,S,S,1) in {0,1}."""
    with ThreadPoolExecutor(workers) as pool:
        images = list(pool.map(lambda p: load_image(p, size), df["image"]))
        masks = list(pool.map(lambda p: load_mask(p, size), df["mask"]))
    return np.stack(images), np.stack(masks)


def _augment(image: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Joint geometric augmentation (image+mask share one transform) and photometric jitter."""
    size = tf.shape(image)[0]
    if tf.random.uniform(()) < 0.5:
        image, mask = tf.image.flip_left_right(image), tf.image.flip_left_right(mask)

    # small random affine: rotation (±~11°), translation (±5%), zoom (±5%)
    angle = tf.random.uniform((), -0.2, 0.2)
    zoom = tf.random.uniform((), 0.95, 1.05)
    shift = tf.random.uniform((2,), -0.05, 0.05) * tf.cast(size, tf.float32)
    c, s = tf.cos(angle) / zoom, tf.sin(angle) / zoom
    half = tf.cast(size, tf.float32) / 2.0
    transform = tf.stack(
        [c, -s, half - c * half + s * half + shift[0], s, c, half - s * half - c * half + shift[1], 0.0, 0.0]
    )[None]

    def warp(x: tf.Tensor, interp: str) -> tf.Tensor:
        out = tf.raw_ops.ImageProjectiveTransformV3(
            images=x[None],
            transforms=transform,
            output_shape=tf.shape(x)[:2],
            interpolation=interp,
            fill_mode="NEAREST",
            fill_value=0.0,
        )
        return out[0]

    image, mask = warp(image, "BILINEAR"), warp(mask, "NEAREST")
    image = image * tf.random.uniform((), 0.9, 1.1) + tf.random.uniform((), -0.05, 0.05)
    return tf.clip_by_value(image, 0.0, 1.0), mask


def make_dataset(
    images: np.ndarray,
    masks: np.ndarray,
    batch_size: int = 16,
    augment: bool = False,
    shuffle: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Batched ``(float32 image in [0,1], float32 mask in {0,1})`` pipeline."""
    ds = tf.data.Dataset.from_tensor_slices((images, masks))
    if shuffle:
        ds = ds.shuffle(min(len(images), 2048), seed=seed, reshuffle_each_iteration=True)

    def prep(img, msk):
        img = tf.cast(img, tf.float32) / 255.0
        msk = tf.cast(msk, tf.float32)
        return _augment(img, msk) if augment else (img, msk)

    ds = ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
