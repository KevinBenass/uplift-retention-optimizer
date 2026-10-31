"""End-to-end: prepare data, fit models, select on validation, evaluate on test."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from scipy.stats import spearmanr

from uplift_optimizer.config import Settings
from uplift_optimizer.data.download import download
from uplift_optimizer.data.features import (
    build_hillstrom,
    build_x5,
    feature_columns,
    load_split,
    save_splits,
    split,
)
from uplift_optimizer.data.synthetic import write_x5_like
from uplift_optimizer.evaluate.curves import (
    auuc,
    decile_table,
    qini_coefficient,
    qini_curve,
    uplift_at_k,
)
from uplift_optimizer.evaluate.policy import (
    Outcomes,
    bootstrap_policy_table,
    budget_policy,
    expected_incremental_profit,
    profit_curve,
    top_fraction,
)
from uplift_optimizer.explain import segment_profile, uplift_drivers
from uplift_optimizer.models.baselines import ChurnRiskTargeting, RandomTargeting, ResponseTargeting
from uplift_optimizer.models.learners import make_learner

console = Console()


def raw_dir(s: Settings) -> Path:
    return s.resolve(s.data.raw_dir) / s.data.dataset


def processed_dir(s: Settings) -> Path:
    return s.resolve(s.data.processed_dir) / s.data.dataset


def prepare(s: Settings) -> dict[str, int]:
    """Download (or simulate), engineer features and write stratified splits."""
    rdir = raw_dir(s)
    if s.data.dataset == "synthetic":
        if not (rdir / "purchases.csv.gz").exists():
            console.print(f"Simulating {s.data.synthetic_clients:,} customers")
            write_x5_like(rdir, s.data.synthetic_clients, s.data.seed)
        df = build_x5(rdir)
    elif s.data.dataset == "x5":
        download("x5", s.resolve(s.data.raw_dir))
        console.print("Aggregating purchase history with DuckDB (a few minutes)…")
        df = build_x5(rdir)
    else:
        download("hillstrom", s.resolve(s.data.raw_dir))
        df = build_hillstrom(rdir)
    if (rdir / "truth.parquet").exists():
        df = df.merge(
            pd.read_parquet(rdir / "truth.parquet")[["client_id", "tau"]].rename(
                columns={"tau": "_true_tau"}
            ),
            on="client_id",
        )
    splits = split(df, s.data.split, s.data.seed)
    save_splits(splits, processed_dir(s))
    return {k: len(v) for k, v in splits.items()}


@dataclass
class RunResult:
    model_table: pd.DataFrame
    policy_table: pd.DataFrame
    profit_curves: pd.DataFrame
    qini_curves: pd.DataFrame
    calibration: pd.DataFrame
    sensitivity: pd.DataFrame
    drivers: pd.DataFrame
    profile: pd.DataFrame
    summary: dict = field(default_factory=dict)


def _xy(df: pd.DataFrame, features: list[str]):
    return df[features], df["treatment"].to_numpy(), df["target"].to_numpy()


def run(s: Settings) -> RunResult:
    pdir = processed_dir(s)
    train, valid, test = (load_split(pdir, n) for n in ("train", "valid", "test"))
    features = [c for c in feature_columns(train) if not c.startswith("_")]
    Xtr, ttr, ytr = _xy(train, features)
    Xva, tva, yva = _xy(valid, features)
    Xte, tte, yte = _xy(test, features)
    params = s.models.lgbm
    econ = s.econ
    budget = s.evaluation.budget_fraction
    has_truth = "_true_tau" in test.columns

    # ---------------------------------------------------------------- fit
    models = {}
    for name in s.models.learners:
        t0 = time.perf_counter()
        models[name] = make_learner(name, params).fit(Xtr, ttr, ytr)
        console.print(f"fit {models[name].name:<22} {time.perf_counter() - t0:5.1f}s")
    baselines = {
        "random": RandomTargeting(seed=s.data.seed),
        "response": ResponseTargeting(params).fit(Xtr, ttr, ytr),
        "churn_risk": ChurnRiskTargeting(params).fit(Xtr, ttr, ytr),
    }
    response_model = baselines["response"]  # P(buy | contacted, x), used to price the offer

    # ---------------------------------------------------------------- select on validation
    valid_qini = {n: qini_coefficient(m.predict(Xva), tva, yva) for n, m in models.items()}
    best_name = max(valid_qini, key=valid_qini.get)
    best = models[best_name]
    console.print(f"[bold]Selected {best.name}[/] (validation Qini {valid_qini[best_name]:.3f})")

    # ---------------------------------------------------------------- test metrics
    rows, qini_rows, preds = [], [], {}
    for name, m in {**models, **baselines}.items():
        score = m.predict(Xte)
        preds[name] = score
        row = {
            "model": m.name,
            "kind": "uplift" if name in models else "baseline",
            "qini_valid": valid_qini.get(name, np.nan),
            "qini_test": qini_coefficient(score, tte, yte),
            "auuc_test": auuc(score, tte, yte),
            "uplift@10%": uplift_at_k(score, tte, yte, 0.1),
            "uplift@30%": uplift_at_k(score, tte, yte, 0.3),
        }
        if has_truth and name in models:
            true_tau = test["_true_tau"].to_numpy()
            row["pehe"] = float(np.sqrt(np.mean((score - true_tau) ** 2)))
            row["spearman_true_tau"] = float(spearmanr(score, true_tau).statistic)
        rows.append(row)
        frac, q = qini_curve(score, tte, yte)
        qini_rows.append(pd.DataFrame({"model": m.name, "fraction": frac, "qini": q}))
    model_table = pd.DataFrame(rows)

    # ---------------------------------------------------------------- policies in currency
    basket = test["basket_value"].to_numpy()
    p_treated = response_model.predict(Xte)
    uplift_best = preds[best_name]
    profit_score = expected_incremental_profit(uplift_best, p_treated, basket, econ)
    o = Outcomes(t=tte, y=yte, revenue=test["revenue"].to_numpy())
    policies = {
        "Contact nobody": np.zeros(len(test)),
        "Contact everybody": np.ones(len(test)),
        f"Random {budget:.0%}": top_fraction(preds["random"], budget),
        f"Response model, top {budget:.0%}": top_fraction(preds["response"], budget),
        f"Churn-risk model, top {budget:.0%}": top_fraction(preds["churn_risk"], budget),
        f"{best.name} uplift, top {budget:.0%}": top_fraction(uplift_best, budget),
        f"{best.name} expected profit > 0 (≤ {budget:.0%})": budget_policy(profit_score, budget),
    }
    if has_truth:
        true_score = expected_incremental_profit(
            test["_true_tau"].to_numpy(), p_treated, basket, econ
        )
        policies[f"Oracle (true uplift) ≤ {budget:.0%}"] = budget_policy(true_score, budget)
    policy_table = bootstrap_policy_table(policies, o, econ, s.evaluation.bootstrap, s.data.seed)

    curves = []
    for label, score in [
        ("Response model", preds["response"]),
        ("Churn-risk model", preds["churn_risk"]),
        ("Random", preds["random"]),
        (f"{best.name} (uplift)", uplift_best),
        (f"{best.name} (expected profit)", profit_score),
    ]:
        curves.append(profit_curve(score, o, econ).assign(ranking=label))
    profit_curves = pd.concat(curves, ignore_index=True)

    # ---------------------------------------------------------------- sensitivity
    sens = []
    for d_mult in (0.5, 1.0, 2.0):
        for c_mult in (0.5, 1.0, 2.0):
            e2 = econ.model_copy(
                update={
                    "discount_value": econ.discount_value * d_mult,
                    "contact_cost": econ.contact_cost * c_mult,
                }
            )
            score = expected_incremental_profit(uplift_best, p_treated, basket, e2)
            curve = profit_curve(score, o, e2)
            best_row = curve.loc[curve["incremental_profit_per_customer"].idxmax()]
            unconstrained = budget_policy(score, 1.0)
            sens.append(
                {
                    "offer_cost": e2.discount_value,
                    "contact_cost": e2.contact_cost,
                    "share_with_positive_expected_profit": unconstrained.mean(),
                    "best_fraction_on_test": best_row["fraction"],
                    "profit_per_1000_at_best": best_row["incremental_profit_per_customer"] * 1000,
                }
            )
    sensitivity = pd.DataFrame(sens)

    # ---------------------------------------------------------------- calibration, drivers
    calibration = decile_table(uplift_best, tte, yte, s.evaluation.n_bins)
    drivers, fidelity, _ = uplift_drivers(Xte, uplift_best)
    profile = segment_profile(test, uplift_best, drivers["feature"].head(6).tolist())

    summary = {
        "dataset": s.data.dataset,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "treatment_share": float(ttr.mean()),
        "ate_test": float(yte[tte == 1].mean() - yte[tte == 0].mean()),
        "response_rate_control": float(yte[tte == 0].mean()),
        "best_model": best.name,
        "best_model_key": best_name,
        "budget_fraction": budget,
        "surrogate_r2": fidelity,
        "economics": econ.model_dump(),
        "features": features,
    }

    # ---------------------------------------------------------------- persist model
    adir = s.resolve(s.evaluation.artifacts_dir) / s.data.dataset
    adir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "uplift": best,
            "response": response_model,
            "features": features,
            "economics": econ.model_dump(),
            "budget_fraction": budget,
            "basket_median": float(train["basket_value"].median()),
        },
        adir / "model.joblib",
    )
    (adir / "summary.json").write_text(json.dumps(summary, indent=2))

    return RunResult(
        model_table,
        policy_table,
        profit_curves,
        pd.concat(qini_rows),
        calibration,
        sensitivity,
        drivers,
        profile,
        summary,
    )


def save_results(r: RunResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "model_table",
        "policy_table",
        "profit_curves",
        "qini_curves",
        "calibration",
        "sensitivity",
        "drivers",
        "profile",
    ):
        getattr(r, name).to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(r.summary, indent=2))
