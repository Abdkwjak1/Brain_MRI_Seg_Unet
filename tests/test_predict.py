import numpy as np
import pytest

from brainseg.predict import Segmenter, to_rgb_uint8
from brainseg.viz import overlay


@pytest.fixture()
def seg(tiny_model_path):
    return Segmenter.from_path(tiny_model_path)


@pytest.mark.parametrize(
    "arr",
    [
        np.zeros((10, 12), np.uint8),
        np.zeros((10, 12, 1), np.uint8),
        np.zeros((10, 12, 4), np.uint8),
        np.random.rand(10, 12, 3),
        (np.random.rand(10, 12) * 65535).astype(np.uint16),
    ],
)
def test_to_rgb_uint8_normalises_any_input(arr):
    out = to_rgb_uint8(arr)
    assert out.shape == (10, 12, 3) and out.dtype == np.uint8


def test_to_rgb_rejects_bad_shapes():
    with pytest.raises(ValueError):
        to_rgb_uint8(np.zeros((4, 4, 2)))
    with pytest.raises(ValueError):
        to_rgb_uint8(np.zeros(5))


def test_constant_image_does_not_divide_by_zero():
    assert to_rgb_uint8(np.full((4, 4), 7.0)).max() == 0


@pytest.mark.parametrize("shape", [(64, 64), (100, 80), (200, 300)])
def test_prediction_is_at_original_resolution(seg, shape):
    res = seg.predict(np.random.randint(0, 255, (*shape, 3), dtype=np.uint8))
    assert res.probability.shape == shape and res.mask.shape == shape
    assert set(np.unique(res.mask)) <= {0, 1}
    assert res.probability.min() >= 0.0 and res.probability.max() <= 1.0
    assert res.tumor_pixels == res.mask.sum()


def test_threshold_extremes(seg):
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    assert (
        seg.predict(img, threshold=0.999999).tumor_pixels <= seg.predict(img, threshold=0.000001).tumor_pixels
    )
    assert seg.predict(img, threshold=0.000001).has_tumor


def test_invalid_threshold(tiny_model_path):
    with pytest.raises(ValueError):
        Segmenter.from_path(tiny_model_path, threshold=1.5)


def test_overlay_only_touches_mask_region():
    img = np.full((16, 16, 3), 100, np.uint8)
    m = np.zeros((16, 16), np.uint8)
    m[4:8, 4:8] = 1
    out = overlay(img, m, contour=False)
    assert np.array_equal(out[m == 0], img[m == 0]) and not np.array_equal(out[m == 1], img[m == 1])
    assert np.array_equal(overlay(img, np.zeros_like(m)), img)
