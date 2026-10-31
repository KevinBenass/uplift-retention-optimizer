import numpy as np
import pandas as pd
import pytest
from conftest import FAST_LGBM
from scipy.stats import spearmanr

from uplift_optimizer.data.features import build_x5, feature_columns, split
from uplift_optimizer.models.learners import LEARNERS, make_learner
from uplift_optimizer.pipeline import prepare, processed_dir, run, save_results
from uplift_optimizer.report import END, START, build
from uplift_optimizer.scoring import Scorer


@pytest.fixture(scope="module")
def features(sim_dir):
    df = build_x5(sim_dir / "raw" / "synthetic")
    truth = pd.read_parquet(sim_dir / "raw" / "synthetic" / "truth.parquet")
    return df.merge(truth[["client_id", "tau"]], on="client_id")


def test_feature_table_contract(features):
    assert features["client_id"].is_unique
    assert set(features["treatment"].unique()) == {0, 1}
    assert features["age"].dropna().between(14, 95).all()  # noisy ages cleaned
    for col in [
        "recency_days",
        "n_tx",
        "avg_basket",
        "tx_trend",
        "share_tx_points_spent",
        "gender_F",
    ]:
        assert col in features
    assert (features["revenue"] == features["target"] * features["basket_value"]).all()


def test_split_is_stratified(features):
    parts = split(features, {"train": 0.6, "valid": 0.2, "test": 0.2}, seed=0)
    assert sum(len(p) for p in parts.values()) == len(features)
    rates = [p["treatment"].mean() for p in parts.values()]
    assert max(rates) - min(rates) < 0.01


@pytest.mark.parametrize("name", sorted(LEARNERS))
def test_learners_recover_true_uplift_ranking(features, name):
    feats = [c for c in feature_columns(features) if c != "tau"]
    train, test = features.iloc[:9000], features.iloc[9000:]
    model = make_learner(name, FAST_LGBM).fit(
        train[feats], train["treatment"].to_numpy(), train["target"].to_numpy()
    )
    pred = model.predict(test[feats])
    assert np.isfinite(pred).all()
    assert spearmanr(pred, test["tau"]).statistic > 0.05  # small sample: just clearly positive


def test_end_to_end(settings, tmp_path):
    sizes = prepare(settings)
    assert sizes["train"] > sizes["test"] > 0
    result = run(settings)
    out = tmp_path / "results"
    save_results(result, out)

    assert {"S-learner", "Random", "Churn-risk model"} <= set(result.model_table["model"])
    assert "Contact nobody" in set(result.policy_table["policy"])
    nobody = result.policy_table.set_index("policy").loc["Contact nobody"]
    assert nobody.profit_per_1000 == 0
    assert result.summary["best_model"] in set(result.model_table["model"])
    assert len(result.calibration) == settings.evaluation.n_bins

    readme = tmp_path / "README.md"
    readme.write_text(f"x\n{START}\nold\n{END}\n")
    memo = tmp_path / "memo.md"
    body = build(out, readme=readme, memo=memo)
    assert "Ranking quality" in body
    assert (out / "profit.png").exists() and memo.read_text().startswith("# Memo")
    assert "old" in readme.read_text()  # synthetic results never overwrite the README

    scorer = Scorer(
        settings.resolve(settings.evaluation.artifacts_dir) / "synthetic" / "model.joblib"
    )
    test = pd.read_parquet(processed_dir(settings) / "test.parquet")
    targets = scorer.target_list(test)
    assert len(targets) <= round(settings.evaluation.budget_fraction * len(test))
    assert targets["expected_incremental_profit"].is_monotonic_decreasing
    assert (targets["expected_incremental_profit"] > 0).all()


def test_api_scores(settings, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import uplift_optimizer.api as api

    path = settings.resolve(settings.evaluation.artifacts_dir) / "synthetic" / "model.joblib"
    if not path.exists():
        pytest.skip("end-to-end test did not produce a model")
    monkeypatch.setattr(api, "scorer", lambda: Scorer(path))
    client = TestClient(api.app)
    resp = client.post(
        "/score",
        json={
            "customers": [
                {
                    "client_id": "a",
                    "basket_value": 500,
                    "features": {"recency_days": 40, "tx_30d": 1, "share_tx_points_spent": 0.6},
                },
                {"client_id": "b", "features": {}},
            ]
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [d["client_id"] for d in body] == ["a", "b"]
    assert all(isinstance(d["contact"], bool) for d in body)
