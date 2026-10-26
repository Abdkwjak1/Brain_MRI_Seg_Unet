import keras
import numpy as np
import pytest

from brainseg.losses import CUSTOM_OBJECTS
from brainseg.model import build_unet

from .conftest import TINY


def test_output_shape_and_range():
    model = build_unet(**TINY)
    y = model.predict(np.random.rand(2, 64, 64, 3).astype("float32"), verbose=0)
    assert y.shape == (2, 64, 64, 1)
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_classic_unet_parameter_count():
    """base=64/depth=4 without dropout is the ~31M-parameter reference network."""
    model = build_unet(base_filters=64, depth=4, dropout=0.0, batch_norm=False)
    assert 30.5e6 < model.count_params() < 31.5e6


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_depth_changes_capacity(depth):
    a = build_unet((64, 64, 3), 8, depth).count_params()
    b = build_unet((64, 64, 3), 8, depth + 1).count_params()
    assert b > a


def test_has_skip_connections():
    names = [layer.name for layer in build_unet(**TINY).layers]
    assert sum(n.startswith("skip") for n in names) == TINY["depth"]


@pytest.mark.parametrize(
    "kwargs", [dict(input_shape=(100, 100, 3), depth=4), dict(depth=0), dict(base_filters=0)]
)
def test_invalid_configs_raise(kwargs):
    with pytest.raises(ValueError):
        build_unet(**{**TINY, **kwargs})


def test_output_is_float32_under_mixed_precision():
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        model = build_unet(**TINY)
        y = model(np.zeros((1, 64, 64, 3), "float32"))
        assert y.dtype.name == "float32"
    finally:
        keras.mixed_precision.set_global_policy("float32")


def test_save_load_roundtrip_is_exact(tmp_path):
    model = build_unet(**TINY)
    x = np.random.rand(2, 64, 64, 3).astype("float32")
    model.save(tmp_path / "m.keras")
    loaded = keras.models.load_model(tmp_path / "m.keras", custom_objects=CUSTOM_OBJECTS, compile=False)
    np.testing.assert_allclose(model.predict(x, verbose=0), loaded.predict(x, verbose=0), atol=1e-6)


def test_gradients_flow_to_every_weight():
    import tensorflow as tf

    model = build_unet(**TINY)
    x = tf.random.uniform((2, 64, 64, 3))
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(model(x, training=True))
    grads = tape.gradient(loss, model.trainable_variables)
    assert all(g is not None for g in grads)
