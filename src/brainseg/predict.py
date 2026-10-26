"""Inference: load a trained model and segment arbitrary-size MRI slices."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import keras
import numpy as np

from .losses import CUSTOM_OBJECTS
from .viz import overlay


@dataclass(frozen=True)
class SegmentationResult:
    probability: np.ndarray  # float32 (H, W) at the *original* resolution
    mask: np.ndarray  # uint8 (H, W) in {0, 1}
    tumor_pixels: int
    tumor_fraction: float
    max_probability: float

    @property
    def has_tumor(self) -> bool:
        return self.tumor_pixels > 0


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Coerce grayscale / RGBA / 16-bit / float images to RGB uint8."""
    img = np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim != 3 or img.shape[2] not in (1, 3, 4):
        raise ValueError(f"Unsupported image shape {img.shape}")
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    img = img[..., :3]
    if img.dtype != np.uint8:
        img = img.astype(np.float64)
        lo, hi = float(img.min()), float(img.max())
        img = ((img - lo) / (hi - lo) * 255.0) if hi > lo else np.zeros_like(img)
        img = img.astype(np.uint8)
    return img


class Segmenter:
    """Thin, testable wrapper around a Keras segmentation model."""

    def __init__(self, model: keras.Model, threshold: float = 0.5):
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        self.model = model
        self.threshold = threshold
        self.size = int(model.input_shape[1])

    @classmethod
    def from_path(cls, path: str | Path, threshold: float = 0.5) -> Segmenter:
        model = keras.models.load_model(str(path), custom_objects=CUSTOM_OBJECTS, compile=False)
        return cls(model, threshold)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        img = to_rgb_uint8(image)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0

    def predict(self, image: np.ndarray, threshold: float | None = None) -> SegmentationResult:
        thr = self.threshold if threshold is None else threshold
        h, w = np.asarray(image).shape[:2]
        prob = self.model.predict(self.preprocess(image)[None], verbose=0)[0, ..., 0]
        prob = cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (prob > thr).astype(np.uint8)
        return SegmentationResult(
            probability=prob,
            mask=mask,
            tumor_pixels=int(mask.sum()),
            tumor_fraction=float(mask.mean()),
            max_probability=float(prob.max()),
        )

    def predict_batch(self, images: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """``float32 (N,S,S,3)`` in [0,1] -> probabilities ``(N,S,S,1)``."""
        return self.model.predict(images, batch_size=batch_size, verbose=0)


def read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"Could not read image: {path}")
    if img.ndim == 3 and img.shape[2] >= 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    return img


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Segment one or more MRI slices.")
    ap.add_argument("images", nargs="+", help="image files (tif/png/jpg)")
    ap.add_argument("--model", required=True, help="path to a trained .keras model")
    ap.add_argument("--out", default="predictions", help="output directory")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    seg = Segmenter.from_path(args.model, args.threshold)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in args.images:
        img = to_rgb_uint8(read_image(p))
        res = seg.predict(img)
        stem = Path(p).stem
        cv2.imwrite(str(out / f"{stem}_mask.png"), res.mask * 255)
        cv2.imwrite(str(out / f"{stem}_overlay.png"), cv2.cvtColor(overlay(img, res.mask), cv2.COLOR_RGB2BGR))
        print(
            f"{p}: tumour={'yes' if res.has_tumor else 'no'} "
            f"area={res.tumor_fraction:.2%} max_p={res.max_probability:.2f}"
        )


if __name__ == "__main__":
    main()
