import numpy as np
import pandas as pd
import pytest

from brainseg.data import (
    find_pairs,
    load_arrays,
    make_dataset,
    patient_id,
    patient_split,
)


def test_patient_id_from_folder():
    assert patient_id("a/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1.tif") == "TCGA_CS_4941"


def test_find_pairs(dataset_root):
    df = find_pairs(dataset_root)
    assert len(df) == 12 * 8 and df["patient"].nunique() == 12
    assert all("_mask" not in p for p in df["image"])


def test_find_pairs_missing_dir_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Get the data"):
        find_pairs(tmp_path)


def test_split_is_patient_disjoint_and_exhaustive(dataset_root):
    df = find_pairs(dataset_root)
    tr, va, te = patient_split(df, 0.2, 0.2, seed=1)
    sets = [set(x["patient"]) for x in (tr, va, te)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert len(tr) + len(va) + len(te) == len(df)
    assert all(len(s) >= 1 for s in sets)


def test_split_is_reproducible_and_seed_sensitive(dataset_root):
    df = find_pairs(dataset_root)
    a = patient_split(df, seed=1)[2]["patient"].unique()
    b = patient_split(df, seed=1)[2]["patient"].unique()
    c = patient_split(df, seed=2)[2]["patient"].unique()
    assert list(a) == list(b) and set(a) != set(c)


def test_split_validates_inputs():
    df = pd.DataFrame({"patient": ["a", "a", "b"], "image": "x", "mask": "y"})
    with pytest.raises(ValueError):
        patient_split(df)
    with pytest.raises(ValueError):
        patient_split(df.assign(patient=list("abc")), 0.6, 0.6)


def test_load_arrays_shapes_and_binary_masks(dataset_root):
    df = find_pairs(dataset_root).head(10)
    x, y = load_arrays(df, 32)  # also exercises resizing
    assert x.shape == (10, 32, 32, 3) and x.dtype == np.uint8
    assert y.shape == (10, 32, 32, 1) and set(np.unique(y)) <= {0, 1}


def test_dataset_ranges_and_batching(phantoms):
    x, m = phantoms
    ds = make_dataset(x, (m > 127).astype(np.uint8), batch_size=5)
    xb, yb = next(iter(ds))
    assert xb.shape == (5, 64, 64, 3) and yb.shape == (5, 64, 64, 1)
    assert float(xb.numpy().min()) >= 0.0 and float(xb.numpy().max()) <= 1.0
    assert set(np.unique(yb.numpy())) <= {0.0, 1.0}


def test_normalisation_is_applied(phantoms):
    """Regression: the original generator yielded raw 0-255 images."""
    x, m = phantoms
    xb, _ = next(iter(make_dataset(x, m, 4)))
    assert xb.numpy().max() <= 1.0


def test_augmentation_keeps_image_and_mask_aligned():
    """Encode the mask in the image; after joint augmentation they must still coincide."""
    rng = np.random.default_rng(0)
    m = np.zeros((16, 64, 64, 1), np.uint8)
    for i in range(16):
        y, x = rng.integers(16, 40, 2)
        m[i, y : y + 16, x : x + 16] = 1
    img = np.repeat(m * 255, 3, axis=-1).astype(np.uint8)
    xb, yb = next(iter(make_dataset(img, m, 16, augment=True, seed=0)))
    agree = ((xb.numpy()[..., :1] > 0.5) == (yb.numpy() > 0.5)).mean()
    assert agree > 0.97
    assert not np.array_equal(xb.numpy()[..., :1] > 0.5, m > 0)  # something actually moved


def test_augmentation_is_stochastic_but_masks_stay_binary(phantoms):
    x, m = phantoms
    ds = make_dataset(x, (m > 127).astype(np.uint8), 24, augment=True)
    a, ma = next(iter(ds))
    b, _ = next(iter(ds))
    assert not np.allclose(a.numpy(), b.numpy())
    assert set(np.unique(ma.numpy())) <= {0.0, 1.0}
    assert a.numpy().min() >= 0.0 and a.numpy().max() <= 1.0


def test_unreadable_files_raise_clear_errors(tmp_path):
    from brainseg.data import load_image, load_mask

    bad = tmp_path / "nope.tif"
    with pytest.raises(OSError, match="Could not read image"):
        load_image(str(bad), 32)
    with pytest.raises(OSError, match="Could not read mask"):
        load_mask(str(bad), 32)


def test_orphan_masks_are_ignored(tmp_path):
    import cv2

    folder = tmp_path / "TCGA_XX_1_20000101"
    folder.mkdir()
    cv2.imwrite(str(folder / "TCGA_XX_1_20000101_1_mask.tif"), np.zeros((8, 8), np.uint8))
    with pytest.raises(FileNotFoundError, match="no matching images"):
        find_pairs(tmp_path)
