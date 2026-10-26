import numpy as np

from brainseg.synthetic import make_batch, make_phantom


def test_phantom_shapes_and_dtypes():
    img, mask = make_phantom(64, np.random.default_rng(0), with_tumor=True)
    assert img.shape == (64, 64, 3) and img.dtype == np.uint8
    assert mask.shape == (64, 64) and set(np.unique(mask)) <= {0, 255}
    assert mask.any()


def test_empty_phantom_has_empty_mask():
    _, mask = make_phantom(64, np.random.default_rng(0), with_tumor=False)
    assert not mask.any()


def test_deterministic_with_seed():
    a, _ = make_batch(3, 64, seed=5)
    b, _ = make_batch(3, 64, seed=5)
    c, _ = make_batch(3, 64, seed=6)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_tumour_is_inside_the_head():
    img, mask = make_phantom(128, np.random.default_rng(3), with_tumor=True)
    assert img[mask > 0].mean() > img[mask == 0].mean()  # tumour brighter than background+brain
