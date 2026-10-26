"""Sanity checks against the real LGG dataset; skipped when it is not downloaded."""

from pathlib import Path

import pytest

from brainseg.data import find_pairs, load_arrays, patient_split

ROOT = Path(__file__).resolve().parent.parent / "Dataset" / "lgg-mri-segmentation"
pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(not ROOT.exists(), reason="run `make data` to download the dataset"),
]


@pytest.fixture(scope="module")
def df():
    return find_pairs(ROOT)


def test_dataset_inventory(df):
    assert len(df) == 3929 and df["patient"].nunique() == 110


def test_real_split_has_no_patient_leakage(df):
    tr, va, te = patient_split(df, 0.15, 0.15, 42)
    assert not set(tr.patient) & set(va.patient)
    assert not set(tr.patient) & set(te.patient)
    assert not set(va.patient) & set(te.patient)


def test_real_slices_load_correctly(df):
    x, m = load_arrays(df.sample(32, random_state=0), 256)
    assert x.shape == (32, 256, 256, 3) and set(m.ravel().tolist()) <= {0, 1}
    assert x.std() > 10  # not blank
