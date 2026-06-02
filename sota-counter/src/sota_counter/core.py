from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import (
    AAAI_VALID_TRACKS,
    ACL_MAIN_STATUSES,
    CONFERENCES,
    REJECTED_STATUSES,
    YEARS,
)

SOTA_PATTERN = re.compile(
    "|".join(
        [
            r"\bstate(?:[-‐‑–—]|\s)?of(?:[-‐‑–—]|\s)?the(?:[-‐‑–—]|\s)?art(?:s)?\b",
            r"\bsota\b",
            r"\bs(?:\.|\s)?o(?:\.|\s)?t(?:\.|\s)?a\.?\b",
        ]
    ),
    re.IGNORECASE,
)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().casefold()


def git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def load_papers(paperlists_dir: Path, conf_slug: str, year: int) -> list[dict[str, Any]]:
    path = paperlists_dir / conf_slug / f"{conf_slug}{year}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Paper Copilot JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, found {type(data).__name__}")
    return data


def filter_papers(papers: list[dict[str, Any]], conf_slug: str) -> list[dict[str, Any]]:
    rejected = {_clean_text(status) for status in REJECTED_STATUSES}
    filtered = [paper for paper in papers if _clean_text(paper.get("status")) not in rejected]

    if conf_slug == "aaai":
        valid_tracks = {_clean_text(track) for track in AAAI_VALID_TRACKS}
        return [
            paper
            for paper in filtered
            if _clean_text(paper.get("track")) in valid_tracks
        ]

    if conf_slug == "acl":
        main_statuses = {_clean_text(status) for status in ACL_MAIN_STATUSES}
        return [paper for paper in filtered if _clean_text(paper.get("status")) in main_statuses]

    return filtered


def has_sota_mention(paper: dict[str, Any]) -> bool:
    abstract = paper.get("abstract") or ""
    return bool(SOTA_PATTERN.search(str(abstract)))


def compute_counts(paperlists_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for conference in CONFERENCES:
        for year in YEARS:
            papers = load_papers(paperlists_dir, conference.slug, year)
            accepted = filter_papers(papers, conference.slug)
            sota = sum(1 for paper in accepted if has_sota_mention(paper))
            total = len(accepted)
            rows.append(
                {
                    "conference": conference.label,
                    "year": year,
                    "accepted": total,
                    "sota": sota,
                    "non_sota": total - sota,
                    "ratio": (sota / total * 100) if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def pivot_counts(df: pd.DataFrame, value: str) -> pd.DataFrame:
    labels = [conference.label for conference in CONFERENCES]
    return (
        df.pivot(index="conference", columns="year", values=value)
        .reindex(labels)
        .reindex(columns=list(YEARS))
    )


def rounded_ratio_table(df: pd.DataFrame) -> pd.DataFrame:
    return pivot_counts(df, "ratio").round(2)
