import pytest

from brainseg.config import Config


def test_defaults_roundtrip(tmp_path):
    cfg = Config()
    cfg.model.base_filters = 16
    cfg.save(tmp_path / "c.yaml")
    assert Config.from_yaml(tmp_path / "c.yaml") == cfg


def test_partial_yaml_keeps_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  epochs: 3\n")
    cfg = Config.from_yaml(p)
    assert cfg.train.epochs == 3
    assert cfg.model.depth == Config().model.depth


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  epocs: 3\n")
    with pytest.raises(ValueError, match="epocs"):
        Config.from_yaml(p)


def test_shipped_configs_load():
    for name in ("default", "smoke"):
        Config.from_yaml(f"configs/{name}.yaml")
