"""Typed experiment configuration, loadable from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "Dataset/lgg-mri-segmentation"
    image_size: int = 256
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    augment: bool = True
    seed: int = 42


@dataclass
class ModelConfig:
    base_filters: int = 16
    depth: int = 4
    dropout: float = 0.1
    batch_norm: bool = True


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 1e-3
    loss: str = "bce_dice"  # bce_dice | dice | focal_tversky
    lr_schedule: str = "cosine"  # cosine | plateau | constant
    mixed_precision: bool = False
    early_stopping_patience: int = 15
    lr_patience: int = 5
    seed: int = 42


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> Config:
        return _build(cls, raw or {})

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))


def _build(cls: type, raw: dict[str, Any]):
    """Instantiate nested dataclasses from a dict, rejecting unknown keys."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in raw.items():
        ftype = known[name].default_factory if known[name].default_factory is not None else None
        if ftype is not None and is_dataclass(ftype):
            kwargs[name] = _build(ftype, value or {})
        else:
            kwargs[name] = value
    return cls(**kwargs)
