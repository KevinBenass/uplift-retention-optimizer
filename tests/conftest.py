from __future__ import annotations

import pytest

from uplift_optimizer.config import load_settings
from uplift_optimizer.data.synthetic import write_x5_like

FAST_LGBM = {
    "n_estimators": 60,
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_child_samples": 40,
    "verbose": -1,
}


@pytest.fixture(scope="session")
def sim_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("uplift")
    write_x5_like(root / "raw" / "synthetic", n_clients=12000, seed=3)
    return root


@pytest.fixture(scope="session")
def settings(sim_dir):
    return load_settings(
        overrides={
            "data.dataset": "synthetic",
            "data.raw_dir": str(sim_dir / "raw"),
            "data.processed_dir": str(sim_dir / "processed"),
            "evaluation.artifacts_dir": str(sim_dir / "artifacts"),
            "evaluation.results_dir": str(sim_dir / "results"),
            "evaluation.bootstrap": 50,
            "models.lgbm": FAST_LGBM,
        }
    )
