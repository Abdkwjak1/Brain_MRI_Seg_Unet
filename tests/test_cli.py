import json

import cv2
import numpy as np

from brainseg import benchmark, predict
from brainseg.synthetic import make_phantom


def test_predict_cli_writes_outputs(tmp_path, tiny_model_path, capsys):
    img, _ = make_phantom(64, np.random.default_rng(0), with_tumor=True)
    p = tmp_path / "slice.png"
    cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    predict.main([str(p), "--model", str(tiny_model_path), "--out", str(tmp_path / "o")])
    assert (tmp_path / "o" / "slice_mask.png").exists()
    assert (tmp_path / "o" / "slice_overlay.png").exists()
    assert "tumour=" in capsys.readouterr().out


def test_benchmark_cli_json(tmp_path):
    out = tmp_path / "b.json"
    benchmark.main(
        [
            "--size",
            "64",
            "--filters",
            "4",
            "--depth",
            "2",
            "--batch-size",
            "2",
            "--repeats",
            "3",
            "--json",
            str(out),
        ]
    )
    r = json.loads(out.read_text())[0]
    assert r["params"] > 0 and r["latency_ms_batch1"]["p50"] > 0 and r["train_slices_per_s"] > 0
