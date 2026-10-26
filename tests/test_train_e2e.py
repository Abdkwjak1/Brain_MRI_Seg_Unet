"""End-to-end: synthetic data on disk -> train -> checkpoint -> evaluate -> predict."""

import json

import numpy as np
import pytest

from brainseg.config import Config
from brainseg.data import find_pairs, load_arrays, patient_split
from brainseg.evaluate import evaluate_arrays
from brainseg.predict import Segmenter
from brainseg.synthetic import write_dataset
from brainseg.train import run_training


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = write_dataset(
        tmp_path_factory.mktemp("e2e_data"), n_patients=20, slices_per_patient=12, size=64, seed=3
    )
    cfg = Config.from_yaml("configs/smoke.yaml")
    cfg.data.root = str(root)
    cfg.train.epochs = 12
    out = tmp_path_factory.mktemp("run")
    summary = run_training(cfg, out)
    return cfg, out, summary


@pytest.mark.slow
def test_artifacts_written(trained):
    _, out, _ = trained
    for name in (
        "best.keras",
        "config.yaml",
        "history.csv",
        "training_curves.png",
        "test_metrics.json",
        "test_per_slice.csv",
        "test_predictions.png",
    ):
        assert (out / name).exists(), name


@pytest.mark.slow
def test_model_actually_learns(trained):
    """Learning check: a tiny U-Net on phantoms must clearly beat the all-background baseline."""
    _, _, s = trained
    assert s["dice_tumor_slices"] > 0.5, s
    assert s["detection_recall"] > 0.8, s
    assert s["false_positive_rate_empty"] < 0.5, s


@pytest.mark.slow
def test_metrics_file_is_valid_json(trained):
    _, out, s = trained
    on_disk = json.loads((out / "test_metrics.json").read_text())
    assert on_disk["n_slices"] == s["n_slices"] and on_disk["params"] > 0


@pytest.mark.slow
def test_reloaded_model_reproduces_test_score(trained):
    cfg, out, s = trained
    _, _, te = patient_split(
        find_pairs(cfg.data.root), cfg.data.val_fraction, cfg.data.test_fraction, cfg.data.seed
    )
    images, masks = load_arrays(te, cfg.data.image_size)
    seg = Segmenter.from_path(out / "best.keras")
    summary, _, _ = evaluate_arrays(seg, images, masks, te["patient"])
    assert summary["dice"] == pytest.approx(s["dice"], abs=1e-3)
    assert np.isfinite(summary["inference_ms_per_slice"])


@pytest.mark.slow
def test_evaluate_cli(trained, tmp_path, capsys):
    from brainseg import evaluate

    _, out, s = trained
    evaluate.main(
        ["--model", str(out / "best.keras"), "--config", str(out / "config.yaml"), "--out", str(tmp_path)]
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["dice"] == pytest.approx(s["dice"], abs=1e-3)
    assert (tmp_path / "test_metrics.json").exists() and (tmp_path / "test_per_slice.csv").exists()


@pytest.mark.slow
def test_train_cli_runs_with_overrides(trained, tmp_path, capsys):
    from brainseg import train

    cfg, _, _ = trained
    train.main(
        [
            "--config",
            "configs/smoke.yaml",
            "--data-dir",
            cfg.data.root,
            "--epochs",
            "1",
            "--out",
            str(tmp_path / "cli_run"),
        ]
    )
    assert (tmp_path / "cli_run" / "best.keras").exists()


def test_lr_schedules():
    from brainseg.train import make_optimizer

    cfg = Config()
    base = cfg.train.learning_rate
    cfg.train.lr_schedule = "cosine"
    opt = make_optimizer(cfg, 100)
    assert float(opt.learning_rate) == pytest.approx(base)
    opt.iterations.assign(cfg.train.epochs * 100)
    assert float(opt.learning_rate) < 0.02 * base  # decayed to ~1% by the end
    cfg.train.lr_schedule = "constant"
    opt = make_optimizer(cfg, 100)
    opt.iterations.assign(500)
    assert float(opt.learning_rate) == pytest.approx(base)
    cfg.train.lr_schedule = "bogus"
    with pytest.raises(ValueError, match="lr_schedule"):
        make_optimizer(cfg, 100)
