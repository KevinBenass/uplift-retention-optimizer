"""Score customers with a trained bundle: uplift, offer cost, decision. Used by CLI and API."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from uplift_optimizer.config import Economics
from uplift_optimizer.evaluate.policy import expected_incremental_profit


class Scorer:
    def __init__(self, bundle_path: Path) -> None:
        b = joblib.load(bundle_path)
        self.uplift_model = b["uplift"]
        self.response_model = b["response"]
        self.features: list[str] = b["features"]
        self.econ = Economics(**b["economics"])
        self.budget_fraction: float = b["budget_fraction"]
        self.basket_median: float = b["basket_median"]

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.reindex(columns=self.features).astype(float)
        uplift = self.uplift_model.predict(X)
        p_treated = self.response_model.predict(X)
        basket = (
            df["basket_value"].astype(float).fillna(self.basket_median).to_numpy()
            if "basket_value" in df
            else np.full(len(df), self.basket_median)
        )
        profit = expected_incremental_profit(uplift, p_treated, basket, self.econ)
        out = pd.DataFrame(
            {
                "uplift": uplift,
                "p_buy_if_contacted": p_treated,
                "expected_incremental_profit": profit,
            },
            index=df.index,
        )
        if "client_id" in df:
            out.insert(0, "client_id", df["client_id"].to_numpy())
        out["contact"] = profit > 0
        return out

    def target_list(self, df: pd.DataFrame) -> pd.DataFrame:
        """Positive expected profit, best first, capped at the budget fraction of the base."""
        scored = self.score(df).sort_values("expected_incremental_profit", ascending=False)
        cap = round(self.budget_fraction * len(df))
        return scored[scored["contact"]].head(cap)
