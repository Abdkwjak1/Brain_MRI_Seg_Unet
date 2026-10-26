"""Train a small U-Net on synthetic phantoms -> runs/demo/best.keras (for the app & screenshots).

This model has never seen real MRI: it demonstrates the pipeline end to end, nothing clinical.
"""

import argparse

from brainseg.config import Config
from brainseg.synthetic import make_batch
from brainseg.train import Splits, run_training


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()

    cfg = Config()
    cfg.data.image_size = 128
    cfg.model.base_filters, cfg.model.depth = 16, 4
    cfg.train.epochs, cfg.train.batch_size = args.epochs, 16

    def split(n, seed):
        x, m = make_batch(n, 128, seed)
        return x, (m > 127).astype("uint8")

    tx, tm = split(800, 1)
    vx, vm = split(150, 2)
    ex, em = split(150, 3)
    summary = run_training(cfg, args.out, Splits((tx, tm), (vx, vm), (ex, em, None)))
    print({k: round(v, 4) for k, v in summary.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
