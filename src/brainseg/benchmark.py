"""Latency / throughput / size benchmark of a U-Net configuration on this machine."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import keras
import numpy as np
import tensorflow as tf

from .model import build_unet


def _time_calls(fn, repeats: int, warmup: int = 3) -> list[float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def benchmark_model(
    size: int = 256,
    base_filters: int = 32,
    depth: int = 4,
    batch_size: int = 8,
    repeats: int = 20,
    train_step: bool = True,
    train_batch_size: int = 4,
) -> dict:
    model = build_unet((size, size, 3), base_filters, depth)
    x1 = np.random.rand(1, size, size, 3).astype("float32")
    xb = np.random.rand(batch_size, size, size, 3).astype("float32")
    predict_fn = tf.function(lambda t: model(t, training=False))

    lat = _time_calls(lambda: predict_fn(x1).numpy(), repeats)
    thr = _time_calls(lambda: predict_fn(xb).numpy(), max(5, repeats // 2))
    out = {
        "config": {"size": size, "base_filters": base_filters, "depth": depth, "batch_size": batch_size},
        "params": int(model.count_params()),
        "weights_mb": round(model.count_params() * 4 / 2**20, 1),
        "latency_ms_batch1": {
            "mean": 1000 * statistics.mean(lat),
            "p50": 1000 * float(np.percentile(lat, 50)),
            "p95": 1000 * float(np.percentile(lat, 95)),
        },
        "throughput_slices_per_s": batch_size / statistics.mean(thr),
        "device": (tf.config.list_physical_devices("GPU") or ["CPU"])[0].name
        if tf.config.list_physical_devices("GPU")
        else "CPU",
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
    }
    if train_step:
        model.compile(optimizer="adam", loss="binary_crossentropy")
        xt = xb[:train_batch_size]
        y = (np.random.rand(len(xt), size, size, 1) > 0.9).astype("float32")
        try:
            step = _time_calls(lambda: model.train_on_batch(xt, y), max(5, repeats // 2))
            out["train_ms_per_batch"] = 1000 * statistics.mean(step)
            out["train_slices_per_s"] = len(xt) / statistics.mean(step)
        except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as exc:
            out["train_error"] = f"{type(exc).__name__} (out of memory at batch {len(xt)})"
    keras.backend.clear_session()
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Benchmark U-Net inference/training speed.")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--filters", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8, help="inference throughput batch")
    ap.add_argument("--train-batch-size", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--json", default=None, help="write results to this file")
    args = ap.parse_args(argv)

    results = [
        benchmark_model(
            args.size, f, args.depth, args.batch_size, args.repeats, not args.no_train, args.train_batch_size
        )
        for f in args.filters
    ]
    print(f"{'filters':>7} {'params':>10} {'b1 p50 ms':>10} {'b1 p95 ms':>10} {'sl/s':>7} {'train sl/s':>11}")
    for r in results:
        print(
            f"{r['config']['base_filters']:>7} {r['params']:>10,} {r['latency_ms_batch1']['p50']:>10.1f} "
            f"{r['latency_ms_batch1']['p95']:>10.1f} {r['throughput_slices_per_s']:>9.1f} "
            f"{r.get('train_slices_per_s', float('nan')):>10.1f}"
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
