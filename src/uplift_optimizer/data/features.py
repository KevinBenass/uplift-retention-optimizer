"""Customer-level features.

X5: 45M product-level purchase rows are aggregated in DuckDB (out-of-core, a few minutes on
a laptop) into one row per customer: RFM, trend, loyalty-points behavior, shopping habits.
All history precedes the campaign, so there is no leakage from the outcome period.

Every dataset is normalized to the same contract:
    client_id, treatment (0/1), target (0/1), revenue (post-period revenue, if known),
    basket_value (expected basket for a purchase), features...
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ID_COLS = ["client_id", "treatment", "target", "revenue", "basket_value"]

X5_SQL = """
WITH p AS (
    SELECT * FROM read_csv_auto('{purchases}', header=true)
),
ref AS (SELECT max(CAST(transaction_datetime AS TIMESTAMP)) AS ref_ts FROM p),
tx AS (
    SELECT client_id, transaction_id,
           any_value(CAST(transaction_datetime AS TIMESTAMP)) AS ts,
           any_value(purchase_sum)               AS basket,
           any_value(regular_points_received)    AS reg_rcv,
           any_value(express_points_received)    AS exp_rcv,
           any_value(regular_points_spent)       AS reg_spent,
           any_value(express_points_spent)       AS exp_spent,
           any_value(store_id)                   AS store_id,
           count(*)                              AS n_lines,
           sum(product_quantity)                 AS qty
    FROM p GROUP BY 1, 2
),
agg AS (
    SELECT tx.client_id, ref.ref_ts,
        count(*)                                                    AS n_tx,
        sum(basket)                                                 AS total_spend,
        avg(basket)                                                 AS avg_basket,
        coalesce(stddev_samp(basket), 0)                            AS std_basket,
        max(basket)                                                 AS max_basket,
        date_diff('day', max(ts), ref_ts)                AS recency_days,
        date_diff('day', min(ts), ref_ts)                AS history_days,
        count(DISTINCT store_id)                                    AS n_stores,
        avg(n_lines)                                                AS avg_lines,
        avg(qty)                                                    AS avg_qty,
        sum(CASE WHEN ts >= ref_ts - INTERVAL 30 DAY THEN 1 ELSE 0 END)     AS tx_30d,
        sum(CASE WHEN ts <  ref_ts - INTERVAL 30 DAY
                  AND ts >= ref_ts - INTERVAL 60 DAY THEN 1 ELSE 0 END)     AS tx_30_60d,
        sum(CASE WHEN ts >= ref_ts - INTERVAL 30 DAY THEN basket ELSE 0 END) AS spend_30d,
        sum(reg_rcv)                                                AS reg_points_rcv,
        sum(exp_rcv)                                                AS exp_points_rcv,
        -sum(reg_spent)                                             AS reg_points_spent,
        -sum(exp_spent)                                             AS exp_points_spent,
        avg(CASE WHEN reg_spent < 0 OR exp_spent < 0 THEN 1 ELSE 0 END) AS share_tx_points_spent,
        avg(CASE WHEN exp_rcv > 0 THEN 1 ELSE 0 END)                AS share_tx_express,
        avg(hour(ts))                                               AS avg_hour,
        avg(CASE WHEN dayofweek(ts) IN (0, 6) THEN 1 ELSE 0 END)    AS share_weekend
    FROM tx CROSS JOIN ref GROUP BY tx.client_id, ref.ref_ts
),
products AS (
    SELECT client_id, count(DISTINCT product_id) AS n_products FROM p GROUP BY 1
)
SELECT u.client_id,
       u.treatment_flg AS treatment,
       u.target,
       c.age, c.gender,
       date_diff('day', CAST(c.first_issue_date AS TIMESTAMP), (SELECT ref_ts FROM ref))  AS days_since_issue,
       CASE WHEN c.first_redeem_date IS NULL THEN 0 ELSE 1 END                           AS has_redeemed,
       date_diff('day', CAST(c.first_issue_date AS TIMESTAMP),
                 CAST(c.first_redeem_date AS TIMESTAMP))                                AS days_issue_to_redeem,
       a.* EXCLUDE (client_id, ref_ts),
       pr.n_products
FROM read_csv_auto('{train}', header=true) u
LEFT JOIN read_csv_auto('{clients}', header=true) c USING (client_id)
LEFT JOIN agg a USING (client_id)
LEFT JOIN products pr USING (client_id)
"""


def build_x5(raw_dir: Path, threads: int | None = None) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    if threads:
        con.execute(f"PRAGMA threads={threads}")
    df = con.execute(
        X5_SQL.format(
            purchases=raw_dir / "purchases.csv.gz",
            train=raw_dir / "uplift_train.csv.gz",
            clients=raw_dir / "clients.csv.gz",
        )
    ).df()
    return finalize_x5(df)


def finalize_x5(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[(df["age"] < 14) | (df["age"] > 95), "age"] = np.nan  # known data-entry noise
    for g in ("F", "M", "U"):
        df[f"gender_{g}"] = (df["gender"] == g).astype(int)
    df = df.drop(columns=["gender"])
    df["tx_trend"] = (df["tx_30d"] + 1) / (df["tx_30_60d"] + 1)
    df["days_between_tx"] = df["history_days"] / df["n_tx"].clip(lower=1)
    df["points_burn_ratio"] = (df["reg_points_spent"] + df["exp_points_spent"]) / (
        df["reg_points_rcv"] + df["exp_points_rcv"] + 1
    )
    df["basket_value"] = df["avg_basket"].fillna(df["avg_basket"].median())
    # X5 does not report post-campaign spend, so revenue = purchase x expected basket.
    df["revenue"] = df["target"] * df["basket_value"]
    return _order(df)


def build_hillstrom(raw_dir: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_dir / "hillstrom.csv.gz")
    df = pd.DataFrame(
        {
            "client_id": np.arange(len(raw)),
            "treatment": (raw["segment"] != "No E-Mail").astype(int),
            "target": raw["conversion"].astype(int),
            "revenue": raw["spend"].astype(float),
        }
    )
    conv = raw.loc[raw["spend"] > 0, "spend"]
    df["basket_value"] = conv.mean()
    feats = raw[["recency", "history", "mens", "womens", "newbie"]].copy()
    feats["history_segment"] = raw["history_segment"].str.extract(r"^(\d)").astype(int)
    feats = pd.concat([feats, pd.get_dummies(raw[["zip_code", "channel"]], dtype=int)], axis=1)
    return _order(pd.concat([df, feats], axis=1))


def _order(df: pd.DataFrame) -> pd.DataFrame:
    return df[ID_COLS + [c for c in df.columns if c not in ID_COLS]]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS]


def split(df: pd.DataFrame, fractions: dict[str, float], seed: int) -> dict[str, pd.DataFrame]:
    """Split stratified on treatment x target, so every split is itself a valid RCT."""
    rng = np.random.default_rng(seed)
    strata = df["treatment"].astype(str) + df["target"].astype(str)
    assignment = np.empty(len(df), dtype=object)
    names = list(fractions)
    cuts = np.cumsum([fractions[n] for n in names])
    for _, idx in df.groupby(strata).indices.items():
        idx = rng.permutation(idx)
        bounds = (cuts * len(idx)).round().astype(int)
        start = 0
        for name, end in zip(names, bounds, strict=True):
            assignment[idx[start:end]] = name
            start = end
    return {n: df[assignment == n].reset_index(drop=True) for n in names}


def save_splits(splits: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(out_dir / f"{name}.parquet", index=False)


def load_split(out_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(out_dir / f"{name}.parquet")
