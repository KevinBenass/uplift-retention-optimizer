"""Fetch public uplift datasets (the mirrors used by scikit-uplift).

X5 RetailHero: 200k loyalty-card customers of a Russian grocery chain, randomly split
into an SMS-campaign group and a control group; target = purchase after the campaign.
Purchase history: ~45M product-level rows (≈ 4 GB uncompressed), handled with DuckDB.

Hillstrom (MineThatData, 2008): 64k e-mail customers randomized into men's e-mail,
women's e-mail or no e-mail; outcomes are visit, conversion and spend.
"""

from __future__ import annotations

from pathlib import Path

import httpx
from rich.progress import Progress

SOURCES: dict[str, dict[str, str]] = {
    "x5": {
        "uplift_train.csv.gz": "https://sklift.s3.eu-west-2.amazonaws.com/uplift_train.csv.gz",
        "clients.csv.gz": "https://sklift.s3.eu-west-2.amazonaws.com/clients.csv.gz",
        "purchases.csv.gz": "https://sklift.s3.eu-west-2.amazonaws.com/purchases.csv.gz",
    },
    "hillstrom": {
        "hillstrom.csv.gz": "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz",
    },
}


def download(dataset: str, raw_dir: Path) -> list[Path]:
    if dataset not in SOURCES:
        raise ValueError(f"No download for dataset {dataset!r}")
    dest_dir = raw_dir / dataset
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with Progress() as progress:
        for name, url in SOURCES[dataset].items():
            dest = dest_dir / name
            paths.append(dest)
            if dest.exists():
                continue
            tmp = dest.with_suffix(".part")
            with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
                r.raise_for_status()
                task = progress.add_task(name, total=int(r.headers.get("content-length", 0)) or None)
                with tmp.open("wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
                        progress.advance(task, len(chunk))
            tmp.rename(dest)
    return paths
