"""Figures, the README results block, and the one-page memo for the marketing lead."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from uplift_optimizer.config import ROOT

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
INK, MUTED, GRID = "#1f2937", "#9ca3af", "#e5e7eb"
ACCENT, ACCENT2 = "#2563eb", "#059669"
BASELINE_COLORS = {"Random": MUTED, "Response model": "#d97706", "Churn-risk model": "#dc2626"}


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)
    ax.set_axisbelow(True)


def fig_qini(qini: pd.DataFrame, best: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model, g in qini.groupby("model", sort=False):
        color = ACCENT if model == best else BASELINE_COLORS.get(model, "#93c5fd")
        lw = 2.4 if model == best else 1.3
        ax.plot(g["fraction"] * 100, g["qini"], color=color, lw=lw, label=model)
    last = qini.groupby("model")["qini"].last().mean()
    ax.plot([0, 100], [0, last], color=MUTED, ls="--", lw=1)
    _style(
        ax,
        "Qini curves on the test set",
        "% of customers contacted (ranked by model)",
        "Incremental buyers",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_profit(curves: pd.DataFrame, budget: float, currency: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for ranking, g in curves.groupby("ranking", sort=False):
        base = ranking.split(" (")[0]
        color = BASELINE_COLORS.get(base, ACCENT2 if "profit" in ranking else ACCENT)
        lw = 2.4 if "profit" in ranking else 1.5
        ax.plot(
            g["fraction"] * 100,
            g["incremental_profit_per_customer"] * 1000,
            color=color,
            lw=lw,
            label=ranking,
        )
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(budget * 100, color=MUTED, ls=":", lw=1.2)
    ax.text(budget * 100 + 1, ax.get_ylim()[1] * 0.92, "budget cap", color=MUTED, fontsize=8)
    _style(
        ax,
        "Incremental profit by targeting depth",
        "% of customers contacted",
        f"Incremental profit per 1,000 customers ({currency})",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_calibration(cal: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = cal["bin"]
    ax.bar(x - 0.2, cal["predicted_uplift"] * 100, width=0.4, color=ACCENT, label="Predicted")
    ax.bar(
        x + 0.2, cal["observed_uplift"] * 100, width=0.4, color="#93c5fd", label="Observed (test)"
    )
    ax.errorbar(
        x + 0.2,
        cal["observed_uplift"] * 100,
        yerr=[
            (cal["observed_uplift"] - cal["ci_low"]) * 100,
            (cal["ci_high"] - cal["observed_uplift"]) * 100,
        ],
        fmt="none",
        ecolor=INK,
        lw=0.8,
        capsize=2,
    )
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x)
    _style(
        ax,
        "Uplift by predicted-uplift decile",
        "Decile (1 = highest predicted uplift)",
        "Uplift (percentage points)",
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_drivers(drivers: pd.DataFrame, path: Path) -> None:
    d = drivers.iloc[::-1]
    colors = [ACCENT2 if v > 0 else "#dc2626" for v in d["direction"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(d["feature"], d["mean_abs_shap"] * 100, color=colors)
    _style(ax, "What drives predicted uplift", "Mean |SHAP| (percentage points of uplift)", "")
    ax.text(
        0.99,
        0.02,
        "green: higher value → more uplift\nred: higher value → less uplift",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _md(df: pd.DataFrame, floatfmt: dict[str, str]) -> str:
    out = df.copy()
    for col, fmt in floatfmt.items():
        if col in out:
            out[col] = out[col].map(lambda v, fmt=fmt: "" if pd.isna(v) else fmt.format(v))
    return out.to_markdown(index=False, disable_numparse=True)


CURRENCY = {"x5": "RUB", "hillstrom": "USD", "synthetic": "RUB"}


def build(results_dir: Path, readme: Path | None = None, memo: Path | None = None) -> str:
    s = json.loads((results_dir / "summary.json").read_text())
    ds, best, budget = s["dataset"], s["best_model"], s["budget_fraction"]
    cur = CURRENCY.get(ds, "")
    models = pd.read_csv(results_dir / "model_table.csv")
    policies = pd.read_csv(results_dir / "policy_table.csv")
    curves = pd.read_csv(results_dir / "profit_curves.csv")
    qini = pd.read_csv(results_dir / "qini_curves.csv")
    cal = pd.read_csv(results_dir / "calibration.csv")
    sens = pd.read_csv(results_dir / "sensitivity.csv")
    drivers = pd.read_csv(results_dir / "drivers.csv")
    profile = pd.read_csv(results_dir / "profile.csv")

    fig_qini(qini, best, results_dir / "qini.png")
    fig_profit(curves, budget, cur, results_dir / "profit.png")
    fig_calibration(cal, results_dir / "calibration.png")
    fig_drivers(drivers, results_dir / "drivers.png")

    rel = results_dir.relative_to(ROOT) if results_dir.is_relative_to(ROOT) else results_dir
    pol = policies.copy()
    pol["95% CI"] = pol.apply(lambda r: f"[{r.ci_low:,.0f}, {r.ci_high:,.0f}]", axis=1)
    pol = pol[["policy", "contacted", "uplift_in_targeted", "profit_per_1000", "95% CI"]]
    mcols = [
        c
        for c in [
            "model",
            "kind",
            "qini_valid",
            "qini_test",
            "uplift@10%",
            "uplift@30%",
            "spearman_true_tau",
            "pehe",
        ]
        if c in models
    ]

    body = "\n".join(
        [
            f"**Dataset:** `{ds}` · train {s['n_train']:,} / valid {s['n_valid']:,} / test {s['n_test']:,} customers · "
            f"treated share {s['treatment_share']:.0%} · average effect of the campaign on test: "
            f"{s['ate_test'] * 100:+.2f} pp on a {s['response_rate_control']:.1%} base purchase rate.",
            "",
            "#### Does targeting by uplift make money? (test set, per 1,000 customers in the base)",
            "",
            _md(
                pol,
                {
                    "contacted": "{:.0%}",
                    "uplift_in_targeted": "{:+.3f}",
                    "profit_per_1000": "{:,.0f}",
                },
            ),
            "",
            f"![Profit curves]({rel}/profit.png)",
            "",
            "#### Ranking quality",
            f"Model selected on validation Qini: **{best}**. Higher Qini = better ranking of who the offer moves.",
            "",
            _md(models[mcols], {c: "{:.3f}" for c in mcols if c not in ("model", "kind")}),
            "",
            f"![Qini curves]({rel}/qini.png)",
            "",
            "#### Is the predicted uplift trustworthy?",
            "",
            f"![Calibration]({rel}/calibration.png)",
            "",
            "#### Who are the persuadables?",
            f"Surrogate fidelity R² = {s['surrogate_r2']:.2f}.",
            "",
            f"![Drivers]({rel}/drivers.png)",
            "",
            "Median profile of the customers the model ranks highest vs lowest:",
            "",
            _md(profile, {c: "{:,.2f}" for c in profile.columns if c != "feature"}),
            "",
            "#### Sensitivity to campaign economics",
            "",
            _md(
                sens,
                {
                    "share_with_positive_expected_profit": "{:.0%}",
                    "best_fraction_on_test": "{:.0%}",
                    "profit_per_1000_at_best": "{:,.0f}",
                },
            ),
        ]
    )
    (results_dir / "REPORT.md").write_text("# Results\n\n" + body.replace(f"{rel}/", "") + "\n")

    # The README shows real-data results; the simulation study lives in its own report.
    readme = readme or ROOT / "README.md"
    if readme.exists() and ds != "synthetic":
        text = readme.read_text()
        if START in text:
            readme.write_text(
                re.sub(
                    re.escape(START) + ".*?" + re.escape(END),
                    lambda _: f"{START}\n{body}\n{END}",
                    text,
                    flags=re.DOTALL,
                )
            )

    memo = memo or ROOT / "docs" / f"memo_{ds}.md"
    memo.parent.mkdir(parents=True, exist_ok=True)
    memo.write_text(write_memo(s, policies, drivers, sens, cur))
    return body


def write_memo(
    s: dict, policies: pd.DataFrame, drivers: pd.DataFrame, sens: pd.DataFrame, cur: str
) -> str:
    def row(prefix: str) -> pd.Series:
        return policies[policies["policy"].str.startswith(prefix)].iloc[0]

    best = s["best_model"]
    rec = row(f"{best} expected profit")
    everyone = row("Contact everybody")
    churn = row("Churn-risk model")
    response = row("Response model")
    up = [f for f, d in zip(drivers["feature"], drivers["direction"], strict=True) if d > 0][:3]
    down = [f for f, d in zip(drivers["feature"], drivers["direction"], strict=True) if d < 0][:3]
    base = 100_000
    gain = (rec.profit_per_1000 - churn.profit_per_1000) * base / 1000
    lines = [
        "# Memo: who should receive the retention offer",
        "",
        f"**To:** Head of CRM · **Data:** randomized campaign, `{s['dataset']}` · **Model:** {best}",
        "",
        "## Recommendation",
        f"Send the offer to **{rec.contacted:.0%} of customers**, chosen by expected incremental profit, "
        f"instead of the current churn-risk list. On held-out data this earns "
        f"**{rec.profit_per_1000:,.0f} {cur} per 1,000 customers** (95% CI {rec.ci_low:,.0f} to {rec.ci_high:,.0f}), "
        f"versus {churn.profit_per_1000:,.0f} {cur} for churn-risk targeting at the same budget cap "
        f"and {everyone.profit_per_1000:,.0f} {cur} for contacting everybody. "
        f"For a base of {base:,} customers that is about **{gain:,.0f} {cur} per campaign** over the current approach.",
        "",
        "## Why the usual lists earn less",
        f"- **Contacting everyone** pays the offer to customers who would have bought anyway: "
        f"{everyone.profit_per_1000:,.0f} {cur} per 1,000.",
        f"- **Churn-risk targeting** picks people who are leaving but often do not respond to an offer "
        f"(uplift among them: {churn.uplift_in_targeted * 100:+.1f} pp).",
        f"- **Response-model targeting** picks loyal buyers, the most expensive 'sure things' "
        f"(uplift among them: {response.uplift_in_targeted * 100:+.1f} pp).",
        f"- **Uplift targeting** picks customers whose behavior the offer changes "
        f"(uplift among them: {rec.uplift_in_targeted * 100:+.1f} pp).",
        "",
        "## Who responds",
        f"Predicted uplift is higher for customers with higher {', '.join(up) or 'n/a'} and lower {', '.join(down) or 'n/a'}. "
        "See the driver chart in the README for the full picture.",
        "",
        "## How confident are we",
        "- Profit is measured on customers the model never saw, from a randomized test, so it is a causal estimate.",
        f"- Assumptions: margin {s['economics']['gross_margin']:.0%}, contact cost {s['economics']['contact_cost']} {cur}, "
        f"offer cost {s['economics']['discount_value']} {cur}, redemption {s['economics']['redemption_rate']:.0%}. "
        f"If the offer cost doubles, the profitable share falls to "
        f"{sens.loc[(sens.offer_cost == sens.offer_cost.max()) & (sens.contact_cost == s['economics']['contact_cost']), 'share_with_positive_expected_profit'].iloc[0]:.0%}.",
        "",
        "## Next steps",
        "1. Run the next campaign as champion/challenger: 80% uplift targeting, 20% current list, plus a 10% no-contact holdout.",
        "2. Refresh the model monthly and monitor the uplift-decile chart for drift.",
        "3. Test a cheaper offer for customers whose uplift is positive but whose expected profit is negative at the current offer cost.",
        "",
    ]
    return "\n".join(lines)
