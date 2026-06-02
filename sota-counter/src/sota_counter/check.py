from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    CONFERENCES,
    REFERENCE_PAPERLISTS_COMMIT,
    REFERENCE_ACCEPTED,
    REFERENCE_RATIOS,
    REFERENCE_SOTA,
    YEARS,
)
from .core import git_commit, pivot_counts, rounded_ratio_table


def _reference_differences(
    actual: pd.DataFrame,
    expected: dict[str, dict[int, int | float]],
    label: str,
) -> list[str]:
    differences: list[str] = []
    for conference in [conf.label for conf in CONFERENCES]:
        for year in YEARS:
            actual_value = actual.loc[conference, year]
            expected_value = expected[conference][year]
            if actual_value != expected_value:
                differences.append(
                    f"{label} mismatch for {conference} {year}: "
                    f"actual={actual_value}, expected={expected_value}"
                )
    return differences


def check_results(
    df: pd.DataFrame,
    out_dir: Path,
    paperlists_dir: Path,
) -> dict[str, Any]:
    accepted = pivot_counts(df, "accepted").astype(int)
    sota = pivot_counts(df, "sota").astype(int)
    ratios = rounded_ratio_table(df)

    issues: list[str] = []
    reference_differences: list[str] = []
    reference_differences.extend(_reference_differences(accepted, REFERENCE_ACCEPTED, "accepted"))
    reference_differences.extend(_reference_differences(sota, REFERENCE_SOTA, "sota"))
    reference_differences.extend(_reference_differences(ratios, REFERENCE_RATIOS, "ratio"))

    table_paths = {
        "accepted": out_dir / "tables" / "accepted_counts.csv",
        "sota": out_dir / "tables" / "sota_counts.csv",
        "ratio": out_dir / "tables" / "sota_ratios.csv",
        "tex": out_dir / "tables" / "appendix_a_tables.tex",
    }
    expected_tables = {
        "accepted": accepted,
        "sota": sota,
        "ratio": ratios,
    }

    for name, path in table_paths.items():
        if not path.exists() or path.stat().st_size == 0:
            issues.append(f"{path.name} is missing or empty.")
            continue
        if name == "tex":
            continue
        loaded = pd.read_csv(path, index_col=0)
        loaded.columns = [int(col) for col in loaded.columns]
        expected = expected_tables[name]
        if name == "ratio":
            table_match = np.allclose(
                loaded.to_numpy(dtype=float),
                expected.to_numpy(dtype=float),
                rtol=0,
                atol=1e-12,
            )
        else:
            table_match = loaded.astype(int).equals(expected.astype(int))
        if not table_match:
            issues.append(f"{path.name} does not match computed table.")

    figure_input = out_dir / "figure1" / "figure1_input.csv"
    trend_png = out_dir / "figure1" / "trend.png"
    trend_pdf = out_dir / "figure1" / "trend.pdf"

    if not figure_input.exists() or figure_input.stat().st_size == 0:
        issues.append("Figure 1 input CSV is missing or empty.")
    else:
        figure_df = pd.read_csv(figure_input)
        expected_cols = {"conference", "year", "accepted", "sota", "non_sota", "ratio"}
        if not expected_cols.issubset(figure_df.columns):
            issues.append("Figure 1 input CSV is missing required columns.")
        else:
            columns = ["conference", "year", "accepted", "sota", "non_sota", "ratio"]
            merged = df[columns].sort_values(["conference", "year"]).reset_index(drop=True)
            loaded = figure_df[columns].sort_values(["conference", "year"]).reset_index(drop=True)
            exact_columns = ["conference", "year", "accepted", "sota", "non_sota"]
            counts_match = merged[exact_columns].equals(loaded[exact_columns])
            ratios_match = np.allclose(
                merged["ratio"].to_numpy(),
                loaded["ratio"].to_numpy(),
                rtol=0,
                atol=1e-12,
            )
            if not counts_match or not ratios_match:
                issues.append("Figure 1 input CSV does not match computed table.")

    for path in (trend_png, trend_pdf):
        if not path.exists() or path.stat().st_size == 0:
            issues.append(f"{path.name} is missing or empty.")

    commit = git_commit(paperlists_dir)
    return {
        "ok": not issues,
        "data_commit": commit,
        "reference_commit": REFERENCE_PAPERLISTS_COMMIT,
        "matches_reference": not reference_differences,
        "issues": issues,
    }
