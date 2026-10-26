"""Performance budgets. Deliberately generous so they catch regressions (10x), not noise."""

import time

import numpy as np
import pytest

from brainseg.data import make_dataset
from brainseg.model import build_unet
from brainseg.predict import Segmenter
from brainseg.synthetic import make_batch

from .conftest import TINY

pytestmark = pytest.mark.perf


def test_input_pipeline_throughput():
    x, m = make_batch(256, 64, seed=0)
    ds = make_dataset(x, (m > 127).astype(np.uint8), 32, augment=True, shuffle=True)
    for _ in ds:  # warm-up epoch (graph tracing)
        pass
    t0 = time.perf_counter()
    for _ in ds:
        pass
    assert 256 / (time.perf_counter() - t0) > 200, "augmented pipeline slower than 200 slices/s"


def test_single_slice_latency_budget(tiny_model_path):
    seg = Segmenter.from_path(tiny_model_path)
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    seg.predict(img)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        seg.predict(img)
        times.append(time.perf_counter() - t0)
    assert np.median(times) < 0.25, f"median latency {np.median(times) * 1e3:.0f} ms"


def test_model_build_time_budget():
    t0 = time.perf_counter()
    build_unet(**TINY)
    assert time.perf_counter() - t0 < 5.0


def test_full_size_model_fits_memory_budget():
    """The default config must stay small enough for a 4 GB laptop GPU."""
    model = build_unet((256, 256, 3), 32, 4, 0.1)
    assert model.count_params() * 4 / 2**20 < 100  # < 100 MB of float32 weights
