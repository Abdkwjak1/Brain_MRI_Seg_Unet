import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
import pytest  # noqa: E402

from brainseg.model import build_unet  # noqa: E402
from brainseg.synthetic import make_batch, write_dataset  # noqa: E402

TINY = dict(input_shape=(64, 64, 3), base_filters=8, depth=3, dropout=0.0)


@pytest.fixture(scope="session")
def dataset_root(tmp_path_factory):
    """On-disk synthetic dataset in the Kaggle LGG layout (12 patients x 8 slices, 64 px)."""
    return write_dataset(tmp_path_factory.mktemp("lgg"), n_patients=12, slices_per_patient=8, size=64)


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory):
    model = build_unet(**TINY)
    path = tmp_path_factory.mktemp("model") / "tiny.keras"
    model.save(path)
    return path


@pytest.fixture(scope="session")
def phantoms():
    return make_batch(24, size=64, seed=1)


@pytest.fixture(autouse=True)
def _clear_keras():
    yield
    keras.backend.clear_session()
