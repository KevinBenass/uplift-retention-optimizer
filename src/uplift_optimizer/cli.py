"""Command line entry point: `uplift --help`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console

from uplift_optimizer.config import Settings, load_settings

app = typer.Typer(add_completion=False, help="Uplift modeling and budget-constrained targeting.")
console = Console()
DatasetOpt = Annotated[str, typer.Option("--dataset", "-d", help="x5 | hillstrom | synthetic")]


def _settings(dataset: str, **overrides) -> Settings:
    return load_settings(overrides={"data.dataset": dataset, **overrides})


def _results_dir(s: Settings) -> Path:
    return s.resolve(s.evaluation.results_dir) / s.data.dataset


@app.command()
def prepare(dataset: DatasetOpt = "x5") -> None:
    """Download (or simulate) data, engineer features, write stratified splits."""
    from uplift_optimizer.pipeline import prepare as _prepare

    console.print(_prepare(_settings(dataset)))


@app.command()
def train(dataset: DatasetOpt = "x5", bootstrap: int | None = None) -> None:
    """Fit all learners and baselines, select on validation, evaluate on test, write report."""
    from uplift_optimizer.pipeline import run, save_results
    from uplift_optimizer.report import build

    overrides = {"evaluation.bootstrap": bootstrap} if bootstrap else {}
    s = _settings(dataset, **overrides)
    result = run(s)
    out = _results_dir(s)
    save_results(result, out)
    build(out)
    console.print(result.policy_table.round(3).to_string(index=False))
    console.print(f"[green]Results in {out}, README updated, memo in docs/memo_{dataset}.md")


@app.command()
def report(dataset: DatasetOpt = "x5") -> None:
    """Rebuild figures, README block and memo from saved results."""
    from uplift_optimizer.report import build

    build(_results_dir(_settings(dataset)))


@app.command()
def score(
    input_path: Path,
    output_path: Path,
    dataset: DatasetOpt = "x5",
    target_list: Annotated[bool, typer.Option(help="Only output customers to contact")] = True,
) -> None:
    """Score a customer feature file (parquet or csv) and write the contact list."""
    from uplift_optimizer.scoring import Scorer

    s = _settings(dataset)
    scorer = Scorer(s.resolve(s.evaluation.artifacts_dir) / dataset / "model.joblib")
    df = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    out = scorer.target_list(df) if target_list else scorer.score(df)
    out.to_csv(output_path, index=False)
    console.print(f"[green]{len(out):,} rows -> {output_path}")


if __name__ == "__main__":
    app()
