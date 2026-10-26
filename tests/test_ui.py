"""Streamlit UI tests via the headless AppTest harness."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.ui

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


@pytest.fixture()
def no_models(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAINSEG_MODEL_DIRS", str(tmp_path / "none"))


@pytest.fixture()
def with_model(monkeypatch, tiny_model_path):
    monkeypatch.setenv("BRAINSEG_MODEL_DIRS", str(tiny_model_path.parent))


def run():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def test_starts_without_model_and_guides_user(no_models):
    at = run()
    assert not at.exception
    assert any("No trained model found" in i.value for i in at.info)


def test_sidebar_controls_present(no_models):
    at = run()
    assert at.title[0].value == "Brain MRI tumour segmentation"
    assert [s.key for s in at.sidebar.slider] == ["threshold", "alpha"]
    assert at.sidebar.radio[0].options[:2] == ["Synthetic phantom", "Upload image"]
    assert any("Not a medical device" in w.value for w in at.sidebar.warning)


def test_segments_phantom_and_reports_metrics(with_model):
    at = run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert {
        "Tumour detected",
        "Tumour area",
        "Peak confidence",
        "Dice vs ground truth",
        "IoU vs ground truth",
    } <= set(labels)
    dice = float(next(m for m in at.metric if m.label == "Dice vs ground truth").value)
    assert 0.0 <= dice <= 1.0


def test_controls_trigger_recompute(with_model):
    at = run()
    at.sidebar.slider(key="threshold").set_value(0.95).run()
    assert not at.exception
    at.number_input(key="seed").set_value(11).run()
    assert not at.exception and len(at.metric) >= 3


def test_upload_mode_waits_for_file(with_model):
    at = run()
    at.sidebar.radio(key="source").set_value("Upload image").run()
    assert not at.exception
    assert any("Upload an image" in i.value for i in at.info)


def test_bad_model_path_is_reported_not_crashed(no_models):
    at = run()
    at.sidebar.text_input(key="model_path").set_value("/nonexistent/model.keras").run()
    assert not at.exception
    assert any("Could not load model" in e.value for e in at.error)
