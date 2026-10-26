"""U-Net (Ronneberger et al., 2015) with batch-norm, optional dropout and a configurable width/depth."""

from __future__ import annotations

import keras
from keras import layers


def _conv_block(x, filters: int, batch_norm: bool, dropout: float, name: str):
    for i in (1, 2):
        x = layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=not batch_norm,
            kernel_initializer="he_normal",
            name=f"{name}_conv{i}",
        )(x)
        if batch_norm:
            x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.Activation("relu", name=f"{name}_relu{i}")(x)
    if dropout > 0:
        x = layers.SpatialDropout2D(dropout, name=f"{name}_drop")(x)
    return x


def build_unet(
    input_shape: tuple[int, int, int] = (256, 256, 3),
    base_filters: int = 32,
    depth: int = 4,
    dropout: float = 0.1,
    batch_norm: bool = True,
    name: str = "unet",
) -> keras.Model:
    """Build a U-Net producing a per-pixel tumour probability (sigmoid, float32).

    ``depth`` is the number of down-sampling steps; the spatial size must be divisible by
    ``2**depth``. ``base_filters=64, depth=4`` is the original ~31M-parameter network.
    """
    h, w, _ = input_shape
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if h % 2**depth or w % 2**depth:
        raise ValueError(f"input size {h}x{w} must be divisible by 2**depth ({2**depth})")
    if base_filters < 1:
        raise ValueError("base_filters must be >= 1")

    inputs = keras.Input(input_shape, name="image")
    x, skips = inputs, []
    for level in range(depth):  # encoder
        x = _conv_block(x, base_filters * 2**level, batch_norm, 0.0, f"enc{level + 1}")
        skips.append(x)
        x = layers.MaxPooling2D(2, name=f"pool{level + 1}")(x)

    x = _conv_block(x, base_filters * 2**depth, batch_norm, dropout, "bottleneck")

    for level in reversed(range(depth)):  # decoder
        filters = base_filters * 2**level
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same", name=f"up{level + 1}")(x)
        x = layers.Concatenate(name=f"skip{level + 1}")([x, skips[level]])
        x = _conv_block(x, filters, batch_norm, dropout if level == 0 else 0.0, f"dec{level + 1}")

    # float32 output keeps the loss numerically stable under mixed precision
    outputs = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", name="mask")(x)
    return keras.Model(inputs, outputs, name=name)
