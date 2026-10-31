"""Scoring service. Run: `make api` (UPLIFT_DATASET selects the trained bundle)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from uplift_optimizer.config import load_settings
from uplift_optimizer.scoring import Scorer

app = FastAPI(title="Uplift Retention Optimizer", version="0.1.0")


@lru_cache(maxsize=1)
def scorer() -> Scorer:
    s = load_settings()
    dataset = os.environ.get("UPLIFT_DATASET", s.data.dataset)
    path = s.resolve(s.evaluation.artifacts_dir) / dataset / "model.joblib"
    if not path.exists():
        raise HTTPException(503, f"No trained model at {path}. Run `make train`.")
    return Scorer(path)


class Customer(BaseModel):
    client_id: str | None = None
    basket_value: float | None = None
    features: dict[str, Any] = Field(default_factory=dict)


class ScoreRequest(BaseModel):
    customers: list[Customer] = Field(..., min_length=1, max_length=10_000)


class Decision(BaseModel):
    client_id: str | None
    uplift: float
    p_buy_if_contacted: float
    expected_incremental_profit: float
    contact: bool


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model")
def model_info() -> dict[str, Any]:
    sc = scorer()
    return {
        "model": sc.uplift_model.name,
        "features": sc.features,
        "economics": sc.econ.model_dump(),
        "budget_fraction": sc.budget_fraction,
    }


@app.post("/score", response_model=list[Decision])
def score(req: ScoreRequest) -> list[Decision]:
    df = pd.DataFrame(
        [
            {"client_id": c.client_id, "basket_value": c.basket_value, **c.features}
            for c in req.customers
        ]
    )
    out = scorer().score(df)
    return [
        Decision(**{**r, "client_id": r.get("client_id")}) for r in out.to_dict(orient="records")
    ]
