import numpy as np
import pytest

from brainseg.losses import (
    bce_dice_loss,
    dice_coefficient,
    dice_loss,
    focal_tversky_loss,
    get_loss,
    iou_score,
)
from brainseg.metrics import slice_metrics


def _arr(a):
    return np.asarray(a, dtype="float32")[None, ..., None] if np.ndim(a) == 2 else np.asarray(a, "float32")


def test_perfect_prediction():
    m = np.zeros((2, 8, 8, 1), "float32")
    m[:, 2:6, 2:6] = 1
    assert float(dice_coefficient(m, m)) == pytest.approx(1.0)
    assert float(iou_score(m, m)) == pytest.approx(1.0)
    assert float(dice_loss(m, m)) == pytest.approx(0.0, abs=1e-6)


def test_disjoint_prediction_scores_near_zero():
    a = np.zeros((1, 8, 8, 1), "float32")
    a[:, :4] = 1
    b = 1 - a
    assert float(dice_coefficient(a, b)) < 0.05
    assert float(iou_score(a, b)) < 0.05


def test_empty_vs_empty_is_perfect():
    z = np.zeros((1, 8, 8, 1), "float32")
    assert float(dice_coefficient(z, z)) == pytest.approx(1.0)


def test_all_background_prediction_is_not_rewarded():
    """Regression: per-slice Dice let the model collapse to 'predict nothing' (Dice 0.63 plateau)."""
    t = np.zeros((10, 16, 16, 1), "float32")
    t[:3, 4:8, 4:8] = 1  # 3 of 10 slices contain tumour
    nothing = np.zeros_like(t)
    assert float(dice_coefficient(t, nothing)) < 0.1
    assert float(dice_loss(t, nothing)) > 0.9


def test_matches_numpy_metric():
    """Global Dice equals the per-slice NumPy Dice when there is a single slice."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        t = (rng.random((1, 16, 16, 1)) > 0.7).astype("float32")
        p = (rng.random((1, 16, 16, 1)) > 0.6).astype("float32")
        assert float(dice_coefficient(t, p, smooth=1e-7)) == pytest.approx(
            slice_metrics(t, p).dice[0], abs=1e-4
        )


@pytest.mark.parametrize("loss", [dice_loss, bce_dice_loss, focal_tversky_loss])
def test_losses_are_monotone_in_quality(loss):
    t = np.zeros((1, 16, 16, 1), "float32")
    t[:, 4:12, 4:12] = 1
    good = np.clip(t * 0.9 + 0.05, 0, 1)
    bad = np.clip((1 - t) * 0.9 + 0.05, 0, 1)
    assert float(loss(t, good)) < float(loss(t, bad))


def test_registry():
    assert get_loss("bce_dice") is bce_dice_loss
    with pytest.raises(ValueError, match="Unknown loss"):
        get_loss("nope")
