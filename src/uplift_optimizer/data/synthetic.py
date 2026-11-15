"""A simulator that writes data in the exact X5 RetailHero schema, with known true uplift.

Why it exists
1. Method validation. On real data the individual treatment effect is never observed, so
   uplift models can only be checked indirectly (Qini, policy value). Here the true effect
   is known, so we can verify that the evaluation itself ranks models correctly.
2. Tests and CI run the full pipeline (DuckDB features -> models -> policy) in seconds.

Customer archetypes are the classic four, driven by latent traits:
  sure things      loyal, frequent shoppers: buy anyway; the offer only costs a discount
  persuadables     lapsing, price-sensitive (they redeem points): the offer brings them back
  lost causes      long-inactive: low purchase probability either way
  sleeping dogs    loyal customers who never use promotions: contact slightly annoys them
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REF_DATE = pd.Timestamp("2019-03-18")
HISTORY_DAYS = 120


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def simulate(n_clients: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (uplift_train, clients, purchases, truth) DataFrames in X5 format."""
    rng = np.random.default_rng(seed)
    ids = np.array([f"{i:010x}" for i in rng.choice(16**10, n_clients, replace=False)])

    # ---- latent traits
    weekly_visits = rng.gamma(shape=1.6, scale=1.1, size=n_clients)          # loyalty
    lapsing = rng.random(n_clients) < 0.30                                   # recent drop-off
    price_sens = rng.beta(2, 3, n_clients)                                   # promo affinity
    basket_mu = rng.lognormal(mean=6.1, sigma=0.45, size=n_clients)         # ~450 RUB
    n_stores = rng.integers(1, 4, n_clients)

    # ---- purchase history (transaction level, then product lines)
    rate_early = weekly_visits / 7
    rate_late = np.where(lapsing, rate_early * rng.uniform(0.05, 0.4, n_clients), rate_early)
    n_early = rng.poisson(rate_early * (HISTORY_DAYS - 45))
    n_late = rng.poisson(rate_late * 45)
    n_tx = n_early + n_late
    owner = np.repeat(np.arange(n_clients), n_tx)
    is_late = np.concatenate([np.r_[np.zeros(a, bool), np.ones(b, bool)] for a, b in zip(n_early, n_late, strict=True)]) if n_tx.sum() else np.zeros(0, bool)
    days_ago = np.where(is_late, rng.uniform(0, 45, len(owner)), rng.uniform(45, HISTORY_DAYS, len(owner)))
    ts = REF_DATE - pd.to_timedelta(days_ago, unit="D") + pd.to_timedelta(rng.integers(8 * 3600, 22 * 3600, len(owner)), unit="s")
    basket = np.round(basket_mu[owner] * rng.lognormal(0, 0.35, len(owner)), 2)
    spend_points = rng.random(len(owner)) < price_sens[owner] * 0.5
    express = rng.random(len(owner)) < 0.15 + 0.3 * price_sens[owner]
    store = owner * 7 % 400 + rng.integers(0, n_stores[owner])
    lines = rng.integers(1, 6, len(owner))

    tx = pd.DataFrame(
        {
            "client_id": ids[owner],
            "transaction_id": [f"t{i:09x}" for i in range(len(owner))],
            "transaction_datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "regular_points_received": np.round(basket * 0.01, 1),
            "express_points_received": np.where(express, np.round(basket * 0.02, 1), 0.0),
            "regular_points_spent": np.where(spend_points, -np.round(basket * rng.uniform(0.02, 0.1, len(owner)), 1), 0.0),
            "express_points_spent": np.where(spend_points & express, -5.0, 0.0),
            "purchase_sum": basket,
            "store_id": [f"s{s:05d}" for s in store],
            "n_lines": lines,
        }
    )
    purchases = tx.loc[tx.index.repeat(tx.pop("n_lines"))].reset_index(drop=True)
    purchases["product_id"] = [f"p{p:05d}" for p in rng.integers(0, 5000, len(purchases))]
    purchases["product_quantity"] = rng.integers(1, 4, len(purchases)).astype(float)
    purchases["trn_sum_from_iss"] = np.round(purchases["purchase_sum"] / 3, 2)
    purchases["trn_sum_from_red"] = 0.0

    # ---- clients table
    issue_ago = rng.uniform(150, 1500, n_clients)
    redeemer = rng.random(n_clients) < np.clip(price_sens * 1.3, 0, 1)
    age = np.round(rng.normal(46, 13, n_clients)).astype(int)
    noise = rng.random(n_clients) < 0.01
    age[noise] = rng.choice([-7, 1, 118, 1901], noise.sum())
    clients = pd.DataFrame(
        {
            "client_id": ids,
            "first_issue_date": (REF_DATE - pd.to_timedelta(issue_ago, unit="D")).strftime("%Y-%m-%d %H:%M:%S"),
            "first_redeem_date": np.where(
                redeemer,
                (REF_DATE - pd.to_timedelta(issue_ago * rng.uniform(0, 0.9, n_clients), unit="D")).strftime("%Y-%m-%d %H:%M:%S"),
                None,
            ),
            "age": age,
            "gender": rng.choice(["F", "M", "U"], n_clients, p=[0.45, 0.25, 0.30]),
        }
    )

    # ---- potential outcomes over the 30-day post-campaign window
    recent_rate = rate_late * 30
    base_logit = -1.6 + 1.25 * np.log1p(recent_rate) + 0.3 * (weekly_visits > 2)
    p0 = _sigmoid(base_logit)
    persuadable = lapsing & (price_sens > 0.35) & (weekly_visits > 0.4)
    sleeping_dog = (weekly_visits > 2.5) & (price_sens < 0.2)
    tau_logit = (
        1.1 * persuadable * price_sens * 2
        + 0.25 * (~lapsing) * price_sens
        - 0.6 * sleeping_dog
    )
    p1 = _sigmoid(base_logit + tau_logit)
    treatment = (rng.random(n_clients) < 0.5).astype(int)
    target = (rng.random(n_clients) < np.where(treatment == 1, p1, p0)).astype(int)

    train = pd.DataFrame({"client_id": ids, "treatment_flg": treatment, "target": target})
    archetype = np.select(
        [persuadable, sleeping_dog, p0 > 0.7, p0 < 0.25], ["persuadable", "sleeping_dog", "sure_thing", "lost_cause"], "other"
    )
    truth = pd.DataFrame({"client_id": ids, "p0": p0, "p1": p1, "tau": p1 - p0, "archetype": archetype})
    return train, clients, purchases, truth


def write_x5_like(raw_dir: Path, n_clients: int, seed: int = 0) -> None:
    """Write simulated data as gzip CSVs with X5 file names, plus the ground truth."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    train, clients, purchases, truth = simulate(n_clients, seed)
    train.to_csv(raw_dir / "uplift_train.csv.gz", index=False)
    clients.to_csv(raw_dir / "clients.csv.gz", index=False)
    purchases.to_csv(raw_dir / "purchases.csv.gz", index=False)
    truth.to_parquet(raw_dir / "truth.parquet", index=False)
