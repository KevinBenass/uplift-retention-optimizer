"""Typed configuration loaded from YAML with dotted-key overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


class DataConfig(BaseModel):
    dataset: Literal["x5", "hillstrom", "synthetic"] = "x5"
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    synthetic_clients: int = 60_000
    split: dict[str, float] = {"train": 0.6, "valid": 0.2, "test": 0.2}
    seed: int = 42


class Economics(BaseModel):
    gross_margin: float = 0.25
    contact_cost: float = 0.10
    discount_value: float = 3.0
    redemption_rate: float = 0.6


class ModelsConfig(BaseModel):
    learners: list[str]
    lgbm: dict[str, Any]


class EvalConfig(BaseModel):
    budget_fraction: float = 0.3
    bootstrap: int = 500
    n_bins: int = 10
    results_dir: Path = Path("results")
    artifacts_dir: Path = Path("artifacts")


class Settings(BaseModel):
    data: DataConfig
    economics: dict[str, Economics]
    models: ModelsConfig
    evaluation: EvalConfig

    @property
    def econ(self) -> Economics:
        return self.economics[self.data.dataset]

    def resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else ROOT / p


def load_settings(
    path: Path | str | None = None, overrides: dict[str, Any] | None = None
) -> Settings:
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        if leaf not in node:
            raise KeyError(f"Unknown config key: {dotted}")
        node[leaf] = copy.deepcopy(value)
    return Settings.model_validate(raw)
