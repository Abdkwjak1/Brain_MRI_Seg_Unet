import numpy as np
import pandas as pd
import pytest

from brainseg.metrics import slice_metrics, summarize


def _masks():
    t = np.zeros((4, 8, 8, 1))
    t[0, :4] = 1  # tumour
    t[1, :2, :2] = 1  # tumour
    p = np.zeros_like(t)
    p[0, :4] = 1  # perfect
    p[2, 0, 0] = 1  # false positive on empty slice
    return t, p  # slice 1: missed, slice 3: correct-empty


def test_slice_metrics_known_values():
    df = slice_metrics(*_masks())
    assert df.dice.round(3).tolist() == [1.0, 0.0, 0.0, 1.0]
    assert df.has_tumor.tolist() == [True, True, False, False]
    assert df.recall.tolist() == [1.0, 0.0, 0.0, 1.0]
    assert df.precision.tolist() == [1.0, 0.0, 0.0, 1.0]


def test_summary_headlines():
    t, p = _masks()
    df = slice_metrics(t, p)
    s = summarize(df, pd.Series(["a", "a", "b", "b"]))
    assert s["dice"] == pytest.approx(0.5)
    assert s["dice_tumor_slices"] == pytest.approx(0.5)
    assert s["false_positive_rate_empty"] == pytest.approx(0.5)
    assert s["detection_recall"] == pytest.approx(0.5)
    assert s["dice_patient_macro"] == pytest.approx(0.5)
    assert s["n_patients"] == 2


def test_threshold_matters():
    t = np.ones((1, 4, 4, 1))
    p = np.full((1, 4, 4, 1), 0.4)
    assert slice_metrics(t, p, 0.5).dice[0] < 0.01
    assert slice_metrics(t, p, 0.3).dice[0] > 0.99


def test_summary_handles_no_tumour_slices():
    z = np.zeros((2, 4, 4, 1))
    s = summarize(slice_metrics(z, z))
    assert s["dice"] == 1.0 and np.isnan(s["dice_tumor_slices"])
