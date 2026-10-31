"""Ranking metrics for uplift models, computed on randomized holdout data.

Individual effects are never observed, so uplift models are judged on groups: sort
customers by predicted uplift and compare treated vs control outcomes among the top k%.

  uplift curve   cumulative incremental responders when targeting the top k%:
                 (rate_T(k) - rate_C(k)) * n(k)
  Qini curve     treated responders minus control responders rescaled to the treated
                 group size: Y_T(k) - Y_C(k) * N_T(k) / N_C(k)
  Qini coef.     area between the model's Qini curve and random targeting, normalized
                 by the same area for the best ordering the observed data allows
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _cumulative(score: np.ndarray, t: np.ndarray, y: np.ndarray, n_points: int = 101):
    order = np.argsort(-score, kind="stable")
    t, y = t[order], y[order]
    n = len(t)
    cut = np.unique(np.linspace(0, n, n_points).round().astype(int))
    nt = np.r_[0, np.cumsum(t)][cut]
    nc = np.r_[0, np.cumsum(1 - t)][cut]
    yt = np.r_[0, np.cumsum(y * t)][cut]
    yc = np.r_[0, np.cumsum(y * (1 - t))][cut]
    return cut / n, nt, nc, yt, yc


def qini_curve(score, t, y, n_points: int = 101) -> tuple[np.ndarray, np.ndarray]:
    frac, nt, nc, yt, yc = _cumulative(np.asarray(score), np.asarray(t), np.asarray(y), n_points)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = yt - np.where(nc > 0, yc * nt / nc, 0.0)
    return frac, q


def uplift_curve(score, t, y, n_points: int = 101) -> tuple[np.ndarray, np.ndarray]:
    frac, nt, nc, yt, yc = _cumulative(np.asarray(score), np.asarray(t), np.asarray(y), n_points)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_diff = np.where(
            (nt > 0) & (nc > 0), yt / np.maximum(nt, 1) - yc / np.maximum(nc, 1), 0.0
        )
    return frac, rate_diff * (nt + nc)


def _area_over_random(frac: np.ndarray, curve: np.ndarray) -> float:
    random_line = frac * curve[-1]
    return float(np.trapezoid(curve - random_line, frac))


def qini_coefficient(score, t, y) -> float:
    t, y = np.asarray(t), np.asarray(y)
    model = _area_over_random(*qini_curve(score, t, y))
    # Best observable ordering: treated responders first, control responders last.
    perfect = _area_over_random(*qini_curve(y * (2 * t - 1) + 0.5 * t, t, y))
    return model / perfect if perfect else 0.0


def auuc(score, t, y) -> float:
    """Area under the uplift curve above random targeting, per customer."""
    frac, curve = uplift_curve(score, t, y)
    return _area_over_random(frac, curve) / len(t)


def uplift_at_k(score, t, y, k: float) -> float:
    score, t, y = np.asarray(score), np.asarray(t), np.asarray(y)
    top = np.argsort(-score, kind="stable")[: max(1, round(k * len(score)))]
    tt, yy = t[top], y[top]
    if tt.sum() == 0 or (1 - tt).sum() == 0:
        return float("nan")
    return float(yy[tt == 1].mean() - yy[tt == 0].mean())


def decile_table(pred_uplift, t, y, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed uplift by score bin: the uplift model's calibration check."""
    df = pd.DataFrame({"pred": pred_uplift, "t": t, "y": y})
    df["bin"] = pd.qcut(df["pred"].rank(method="first", ascending=False), n_bins, labels=False) + 1
    rows = []
    for b, g in df.groupby("bin"):
        yt, yc = g.loc[g.t == 1, "y"], g.loc[g.t == 0, "y"]
        obs = yt.mean() - yc.mean()
        se = np.sqrt(yt.var(ddof=1) / max(len(yt), 1) + yc.var(ddof=1) / max(len(yc), 1))
        rows.append(
            {
                "bin": int(b),
                "n": len(g),
                "predicted_uplift": g["pred"].mean(),
                "observed_uplift": obs,
                "ci_low": obs - 1.96 * se,
                "ci_high": obs + 1.96 * se,
                "response_treated": yt.mean(),
                "response_control": yc.mean(),
            }
        )
    return pd.DataFrame(rows)
