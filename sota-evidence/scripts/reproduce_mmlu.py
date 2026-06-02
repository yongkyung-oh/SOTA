#!/usr/bin/env python3
"""Reproduce the MMLU column of Table 1 from arXiv:2605.17273.

The script collects the public HELM MMLU leaderboard snapshot, compares it with
the included reference CSV, runs the documented preprocessing and diagnostics,
and checks the reported MMLU summary values.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import traceback
import urllib.request
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = SCRIPT_DIR.parent
DEFAULT_ARTIFACTS_DIR = RELEASE_ROOT / "artifacts" / "mmlu"
MANIFEST_PATH = RELEASE_ROOT / "manifests" / "mmlu.json"

BENCHMARK_NAME = "MMLU"
EPSILON = 1e-10
TAU_W = 0.60
TAU_D = 0.20
TAU_B = 0.20
TOP_K_VALUES = [10, 20, 30, 40, 50, 60, 70, 80]

SUBJECT_LABEL_OVERRIDES = {
    "High School Computer Science - EM": "High School Computer Scienc...",
    "High School European History - EM": "High School European Histor...",
    "High School Government And Politics - EM": "High School Government And ...",
    "High School Macroeconomics - EM": "High School Macroeconomics ...",
    "High School Microeconomics - EM": "High School Microeconomics ...",
    "High School Us History - EM": "High School US History - EM",
}


@dataclass
class Paths:
    artifacts: Path
    source: Path
    source_helm: Path
    processed: Path
    results: Path
    reports: Path


def load_manifest() -> Dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text())


def build_paths(artifacts_dir: Path) -> Paths:
    return Paths(
        artifacts=artifacts_dir,
        source=artifacts_dir / "source",
        source_helm=artifacts_dir / "source" / "helm",
        processed=artifacts_dir / "processed",
        results=artifacts_dir / "results",
        reports=artifacts_dir / "reports",
    )


def ensure_dirs(paths: Paths) -> None:
    for path in [
        paths.artifacts,
        paths.source,
        paths.source_helm,
        paths.processed,
        paths.results,
        paths.reports,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def release_path(rel_path: str) -> Path:
    return RELEASE_ROOT / rel_path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(RELEASE_ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def fetch_bytes(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "mmlu-reproducibility/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_to_file(url: str, dest: Path, timeout: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(fetch_bytes(url, timeout=timeout))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def parse_config(config_text: str, manifest: Dict[str, Any]) -> Tuple[str, str]:
    release = manifest["official_source"]["release"]
    base_url = manifest["official_source"]["benchmark_output_base_url"]
    release_match = re.search(r'window\.RELEASE\s*=\s*"([^"]+)"', config_text)
    base_match = re.search(r'window\.BENCHMARK_OUTPUT_BASE_URL\s*=\s*"([^"]+)"', config_text)
    if release_match:
        release = release_match.group(1)
    if base_match:
        base_url = base_match.group(1)
    return release, base_url


def url_join(base: str, *parts: str) -> str:
    base = base.rstrip("/")
    tail = "/".join(part.strip("/") for part in parts)
    return f"{base}/{tail}"


def normalized_subject_label(title: str) -> Tuple[str, str]:
    if title == "Massive Multitask Language Understanding (MMLU) All Subjects":
        return "MMLU All Subjects - EM", "MMLU All Subjects - EM"
    if not title.startswith("subject: "):
        raise ValueError(f"Unexpected MMLU table title: {title}")

    subject = title.replace("subject: ", "", 1).replace("_", " ").title()
    full_label = f"{subject} - EM"
    return full_label, SUBJECT_LABEL_OVERRIDES.get(full_label, full_label)


def decimal_half_up(value: Any, digits: int = 3) -> float:
    if pd.isna(value):
        return math.nan
    quant = Decimal("1").scaleb(-digits)
    return float(Decimal(str(float(value))).quantize(quant, rounding=ROUND_HALF_UP))


def round_numeric_half_up(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    rounded = df.copy()
    numeric_cols = [col for col in rounded.columns if col != "Model"]
    for col in numeric_cols:
        rounded[col] = rounded[col].map(lambda value: decimal_half_up(value, digits))
    return rounded


def reconstruct_leaderboard_from_group_json(
    group_json_path: Path,
    reference_csv_path: Optional[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, str]]]:
    tables = read_json(group_json_path)
    if not isinstance(tables, list) or len(tables) < 2:
        raise ValueError(f"Expected a list of MMLU group tables, got {type(tables)}")

    columns = ["Model"]
    label_map: List[Dict[str, str]] = []
    values_by_model: Dict[str, Dict[str, Any]] = {}
    model_order: List[str] = []

    for table in tables:
        title = table.get("title", "")
        full_label, normalized_label = normalized_subject_label(title)
        columns.append(normalized_label)
        label_map.append(
            {
                "title": title,
                "official_label": full_label,
                "normalized_label": normalized_label,
            }
        )

        rows = table.get("rows", [])
        for row in rows:
            model = row[0].get("value")
            score = row[1].get("value")
            if model is None:
                continue
            values_by_model.setdefault(model, {})[normalized_label] = score
            if model not in model_order:
                model_order.append(model)

    full_precision = pd.DataFrame(
        [{"Model": model, **values_by_model[model]} for model in model_order]
    )
    full_precision = full_precision[columns]
    full_precision = full_precision.sort_values("MMLU All Subjects - EM", ascending=False)
    full_precision = full_precision.reset_index(drop=True)

    if reference_csv_path is not None and reference_csv_path.exists():
        reference_columns = list(pd.read_csv(reference_csv_path, nrows=0).columns)
        missing = [col for col in reference_columns if col not in full_precision.columns]
        if missing:
            raise ValueError(f"Official HELM data is missing reference columns: {missing}")
        full_precision = full_precision[reference_columns]

    normalized = round_numeric_half_up(full_precision, digits=3)
    return full_precision, normalized, label_map


def copy_reference_snapshot(paths: Paths, reason: str) -> Dict[str, Any]:
    manifest = load_manifest()
    source_rel = manifest["reference_snapshot"]["path"]
    source_path = release_path(source_rel)
    if not source_path.exists():
        raise FileNotFoundError(f"Reference snapshot not found: {source_path}")

    dest = paths.source / "helm_mmlu_leaderboard.csv"
    full_dest = paths.source / "helm_mmlu_leaderboard_full_precision.csv"
    shutil.copy2(source_path, dest)
    shutil.copy2(source_path, full_dest)

    payload = {
        "benchmark": BENCHMARK_NAME,
        "collection_mode": "reference_snapshot",
        "reason": reason,
        "source_path": display_path(source_path),
        "normalized_csv": display_path(dest),
        "full_precision_csv": display_path(full_dest),
        "official_source_url": manifest["official_source"]["project_url"],
    }
    write_json(paths.source / "source_manifest.json", payload)
    write_json(paths.source / "subject_label_map.json", {"subjects": []})
    return payload


def collect(args: argparse.Namespace) -> Dict[str, Any]:
    paths = build_paths(args.artifacts_dir)
    ensure_dirs(paths)
    manifest = load_manifest()

    if args.offline:
        return copy_reference_snapshot(paths, "offline mode requested")

    config_url = manifest["official_source"]["config_url"]
    release = args.release or manifest["official_source"]["release"]
    base_url = manifest["official_source"]["benchmark_output_base_url"]

    try:
        if args.source_url:
            fetch_to_file(args.source_url, paths.source / "helm_mmlu_leaderboard.csv")
            df = pd.read_csv(paths.source / "helm_mmlu_leaderboard.csv")
            df.to_csv(paths.source / "helm_mmlu_leaderboard_full_precision.csv", index=False)
            payload = {
                "benchmark": BENCHMARK_NAME,
                "collection_mode": "direct_csv_url",
                "source_url": args.source_url,
                "normalized_csv": display_path(paths.source / "helm_mmlu_leaderboard.csv"),
                "full_precision_csv": display_path(paths.source / "helm_mmlu_leaderboard_full_precision.csv"),
            }
            write_json(paths.source / "source_manifest.json", payload)
            return payload

        config_text = fetch_bytes(config_url).decode("utf-8")
        (paths.source_helm / "config.js").write_text(config_text)
        detected_release, detected_base_url = parse_config(config_text, manifest)
        release = args.release or detected_release
        base_url = detected_base_url

        release_prefix = f"releases/{release}"
        files = {
            "summary.json": url_join(base_url, release_prefix, "summary.json"),
            "groups.json": url_join(base_url, release_prefix, "groups.json"),
            "schema.json": url_join(base_url, release_prefix, "schema.json"),
            "group_mmlu.json": url_join(base_url, release_prefix, "groups", "mmlu.json"),
        }
        for filename, url in files.items():
            fetch_to_file(url, paths.source_helm / filename)

        reference_csv = release_path(manifest["reference_snapshot"]["path"])
        full_precision, normalized, label_map = reconstruct_leaderboard_from_group_json(
            paths.source_helm / "group_mmlu.json",
            reference_csv,
        )
        full_csv = paths.source / "helm_mmlu_leaderboard_full_precision.csv"
        normalized_csv = paths.source / "helm_mmlu_leaderboard.csv"
        full_precision.to_csv(full_csv, index=False)
        normalized.to_csv(normalized_csv, index=False, float_format="%.3f")
        write_json(paths.source / "subject_label_map.json", {"subjects": label_map})

        summary = read_json(paths.source_helm / "summary.json")
        payload = {
            "benchmark": BENCHMARK_NAME,
            "collection_mode": "official_helm_group_json",
            "config_url": config_url,
            "benchmark_output_base_url": base_url,
            "release": release,
            "snapshot_date": summary.get("date"),
            "official_files": {
                key: display_path(paths.source_helm / key)
                for key in files
            },
            "normalized_csv": display_path(normalized_csv),
            "full_precision_csv": display_path(full_csv),
            "normalization": "numeric values rounded to 3 decimals with ROUND_HALF_UP; subject headers aligned with the included reference CSV",
            "shape": list(normalized.shape),
            "n_models": int(normalized.shape[0]),
            "n_score_columns": int(normalized.shape[1] - 1),
        }
        write_json(paths.source / "source_manifest.json", payload)
        return payload
    except Exception as exc:
        if args.strict_collection:
            raise
        return copy_reference_snapshot(paths, f"official collection failed: {exc}")


def compare_dataframes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key_col: str = "Model",
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "columns_equal": list(left.columns) == list(right.columns),
        "models_equal": False,
        "numeric_max_abs_delta": None,
        "numeric_mean_abs_delta": None,
        "numeric_nonzero_count": None,
        "exact_numeric_match": False,
    }

    if key_col not in left.columns or key_col not in right.columns:
        result["error"] = f"Missing key column {key_col}"
        return result

    common_columns = [col for col in right.columns if col in left.columns and col != key_col]
    left_aligned = left.set_index(key_col)
    right_aligned = right.set_index(key_col)
    common_models = left_aligned.index.intersection(right_aligned.index)
    result["models_equal"] = list(left_aligned.index) == list(right_aligned.index)
    result["common_models"] = int(len(common_models))
    result["common_numeric_columns"] = int(len(common_columns))

    if len(common_models) == 0 or len(common_columns) == 0:
        result["error"] = "No common models or numeric columns"
        return result

    left_num = left_aligned.loc[common_models, common_columns].apply(pd.to_numeric)
    right_num = right_aligned.loc[common_models, common_columns].apply(pd.to_numeric)
    delta = (left_num - right_num).abs()
    max_delta = float(delta.max().max())
    result["numeric_max_abs_delta"] = max_delta
    result["numeric_mean_abs_delta"] = float(delta.stack().mean())
    result["numeric_nonzero_count"] = int((delta > 1e-12).sum().sum())
    result["exact_numeric_match"] = max_delta <= 1e-12
    result["top_deltas"] = [
        {"model": model, "column": column, "abs_delta": float(value)}
        for (model, column), value in delta.stack().sort_values(ascending=False).head(10).items()
    ]
    return result


def verify_data(args: argparse.Namespace) -> Dict[str, Any]:
    paths = build_paths(args.artifacts_dir)
    ensure_dirs(paths)
    manifest = load_manifest()

    source_csv = paths.source / "helm_mmlu_leaderboard.csv"
    full_csv = paths.source / "helm_mmlu_leaderboard_full_precision.csv"
    reference_csv = release_path(manifest["reference_snapshot"]["path"])

    if not source_csv.exists():
        raise FileNotFoundError(f"Run collect first: {source_csv}")
    if not reference_csv.exists():
        raise FileNotFoundError(f"Reference data not found: {reference_csv}")

    source = pd.read_csv(source_csv)
    reference = pd.read_csv(reference_csv)
    normalized_cmp = compare_dataframes(source, reference)

    full_precision_cmp: Optional[Dict[str, Any]] = None
    if full_csv.exists():
        full_precision = pd.read_csv(full_csv)
        full_precision_cmp = compare_dataframes(full_precision, reference)

    report = {
        "benchmark": BENCHMARK_NAME,
        "source_csv": display_path(source_csv),
        "reference_csv": display_path(reference_csv),
        "normalized_vs_reference": normalized_cmp,
        "full_precision_vs_reference": full_precision_cmp,
        "verdict": "PASS" if normalized_cmp["columns_equal"] and normalized_cmp["models_equal"] and normalized_cmp["exact_numeric_match"] else "CHECK",
        "note": "Full-precision HELM values differ from the reference CSV only by display/export rounding when max_abs_delta <= 0.0005.",
    }
    write_json(paths.reports / "data_comparison.json", report)
    return report


def load_source_leaderboard(paths: Paths) -> pd.DataFrame:
    source_csv = paths.source / "helm_mmlu_leaderboard.csv"
    if not source_csv.exists():
        raise FileNotFoundError(f"Run collect first: {source_csv}")
    return pd.read_csv(source_csv, na_values=["-", "nan", "NaN", ""])


def preprocess(args: argparse.Namespace) -> Dict[str, Any]:
    paths = build_paths(args.artifacts_dir)
    ensure_dirs(paths)

    raw = load_source_leaderboard(paths)
    meta_cols = ["Model", "MMLU All Subjects - EM"]
    missing_meta = [col for col in meta_cols if col not in raw.columns]
    if missing_meta:
        raise ValueError(f"Missing required columns: {missing_meta}")

    score_cols = [col for col in raw.columns if col not in meta_cols]
    scores = raw.set_index("Model")[score_cols].apply(pd.to_numeric)
    full_matrix = scores.T

    mean_scores = full_matrix.mean(axis=0).sort_values(ascending=False)
    top20_models = mean_scores.head(20).index.tolist()
    top20 = full_matrix[top20_models]

    full_matrix.to_csv(paths.processed / "MMLU_full.csv")
    top20.to_csv(paths.processed / "MMLU_top20.csv")

    for top_k in TOP_K_VALUES:
        selected_models = mean_scores.head(top_k).index.tolist()
        top_k_matrix = full_matrix[selected_models]
        top_k_matrix.to_csv(paths.processed / f"MMLU_top{top_k}.csv")
        top_k_meta = {
            "name": "HELM MMLU Leaderboard",
            "direction": "higher_better",
            "metric": "EM (Exact Match)",
            "n_models_total": int(scores.shape[0]),
            "n_models_selected": int(len(selected_models)),
            "n_datasets_total": int(len(score_cols)),
            "n_datasets_selected": int(len(top_k_matrix)),
            "models": selected_models,
            "datasets": list(top_k_matrix.index),
            "source": "helm_mmlu_leaderboard.csv",
            "top_k": top_k,
        }
        write_json(paths.processed / f"MMLU_top{top_k}_meta.json", top_k_meta)

    meta = {
        "name": "HELM MMLU Leaderboard",
        "domain": "NLP/LLM",
        "direction": "higher_better",
        "metric": "EM (Exact Match)",
        "n_models_total": int(scores.shape[0]),
        "n_models_selected": int(len(top20_models)),
        "n_datasets_total": int(len(score_cols)),
        "n_datasets_selected": int(len(top20)),
        "models": top20_models,
        "datasets": list(top20.index),
        "source": "helm_mmlu_leaderboard.csv",
    }
    write_json(paths.processed / "MMLU_meta.json", meta)

    return {
        "full_shape": list(full_matrix.shape),
        "top20_shape": list(top20.shape),
        "top20_models": top20_models,
        "processed_csv": display_path(paths.processed / "MMLU_top20.csv"),
        "meta_json": display_path(paths.processed / "MMLU_meta.json"),
    }


def generate_model_pairs(df: pd.DataFrame, direction: str) -> List[Tuple[str, str]]:
    mean_scores = df.mean(axis=0)
    pairs = []
    for model_a, model_b in combinations(df.columns, 2):
        mean_a = mean_scores[model_a]
        mean_b = mean_scores[model_b]
        if direction == "lower_better":
            winner = model_a if mean_a < mean_b else model_b
        else:
            winner = model_a if mean_a > mean_b else model_b
        loser = model_b if winner == model_a else model_a
        pairs.append((winner, loser))
    return pairs


def compute_l1(df: pd.DataFrame, direction: str, paths: Paths) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    mean_scores = df.mean(axis=0).sort_values(ascending=(direction == "lower_better"))
    l1_results = pd.DataFrame(
        {
            "rank": range(1, len(mean_scores) + 1),
            "model": mean_scores.index,
            "mean_score": mean_scores.values,
        }
    )
    friedman_stat, friedman_p = stats.friedmanchisquare(*[df[col] for col in df.columns])
    winner = mean_scores.index[0]
    runner_up = mean_scores.index[1]
    score_gap = abs(mean_scores.iloc[0] - mean_scores.iloc[1])
    stats_payload = {
        "winner": winner,
        "runner_up": runner_up,
        "winner_mean": float(mean_scores.iloc[0]),
        "runner_up_mean": float(mean_scores.iloc[1]),
        "score_gap": float(score_gap),
        "friedman_stat": float(friedman_stat),
        "friedman_p": float(friedman_p),
    }
    l1_results.to_csv(paths.results / "L1_ranking_MMLU.csv", index=False)
    return l1_results, stats_payload


def compute_l2(df: pd.DataFrame, model_pairs: List[Tuple[str, str]], direction: str, paths: Paths) -> pd.DataFrame:
    k = len(df.index)
    rows = []
    for winner, loser in model_pairs:
        scores_w = df[winner].values
        scores_l = df[loser].values
        if direction == "lower_better":
            wins = int(np.sum(scores_w < scores_l))
            losses = int(np.sum(scores_w > scores_l))
        else:
            wins = int(np.sum(scores_w > scores_l))
            losses = int(np.sum(scores_w < scores_l))
        ties = int(k - wins - losses)
        rows.append(
            {
                "winner": winner,
                "loser": loser,
                "win_rate": wins / k,
                "neg_rate": losses / k,
                "tie_rate": ties / k,
                "wins": wins,
                "losses": losses,
                "ties": ties,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(paths.results / "L2_Consistency_MMLU.csv", index=False)
    return out


def compute_l3(
    df: pd.DataFrame,
    model_pairs: List[Tuple[str, str]],
    direction: str,
    paths: Paths,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    for winner, loser in model_pairs:
        scores_w = df[winner].values
        scores_l = df[loser].values
        if direction == "lower_better":
            diff = scores_l - scores_w
        else:
            diff = scores_w - scores_l

        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))
        cohens_d = mean_diff / (std_diff + EPSILON)
        try:
            _, wilcoxon_p = stats.wilcoxon(scores_w, scores_l)
        except ValueError:
            wilcoxon_p = 1.0

        bootstrap_means = []
        for _ in range(1000):
            idx = rng.choice(len(diff), size=len(diff), replace=True)
            bootstrap_means.append(float(np.mean(diff[idx])))
        ci_lower = float(np.percentile(bootstrap_means, 2.5))
        ci_upper = float(np.percentile(bootstrap_means, 97.5))

        rows.append(
            {
                "winner": winner,
                "loser": loser,
                "cohens_d": cohens_d,
                "wilcoxon_p": float(wilcoxon_p),
                "ci_width": ci_upper - ci_lower,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "mean_diff": mean_diff,
                "std_diff": std_diff,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(paths.results / "L3_Magnitude_MMLU.csv", index=False)
    return out


def compute_l4(df: pd.DataFrame, model_pairs: List[Tuple[str, str]], direction: str, paths: Paths) -> pd.DataFrame:
    k = len(df.index)
    rows = []
    for winner, loser in model_pairs:
        scores_w = df[winner]
        scores_l = df[loser]
        datasets = df.index.tolist()

        loo_flips = 0
        influential_datasets = []
        for dataset in datasets:
            subset = [item for item in datasets if item != dataset]
            mean_w_loo = scores_w[subset].mean()
            mean_l_loo = scores_l[subset].mean()
            if direction == "lower_better":
                loo_winner_is_w = mean_w_loo < mean_l_loo
            else:
                loo_winner_is_w = mean_w_loo > mean_l_loo
            if not loo_winner_is_w:
                loo_flips += 1
                influential_datasets.append(dataset)

        if direction == "lower_better":
            advantage = scores_l - scores_w
        else:
            advantage = scores_w - scores_l

        sorted_datasets = advantage.sort_values(ascending=False).index.tolist()
        bp = k
        for i in range(1, k + 1):
            keep = sorted_datasets[i:]
            if len(keep) == 0:
                break
            mean_w_bp = scores_w[keep].mean()
            mean_l_bp = scores_l[keep].mean()
            if direction == "lower_better":
                bp_winner_is_w = mean_w_bp < mean_l_bp
            else:
                bp_winner_is_w = mean_w_bp > mean_l_bp
            if not bp_winner_is_w:
                bp = i
                break

        rows.append(
            {
                "winner": winner,
                "loser": loser,
                "bp_ratio": bp / k,
                "breakdown_point": bp,
                "loo_flips": loo_flips,
                "loo_flip_ratio": loo_flips / k,
                "influential_datasets": influential_datasets,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(paths.results / "L4_Stability_MMLU.csv", index=False)
    return out


def compute_pass_rates(summary_df: pd.DataFrame, tau_w: float, tau_d: float, tau_b: float) -> Dict[str, Any]:
    l2_pass = summary_df["win_rate"] > tau_w
    l3_pass = summary_df["cohens_d"] > tau_d
    l4_pass = summary_df["bp_ratio"] > tau_b
    joint_pass = l2_pass & l3_pass & l4_pass
    return {
        "L2": float(l2_pass.mean()),
        "L3": float(l3_pass.mean()),
        "L4": float(l4_pass.mean()),
        "Joint Rate": float(joint_pass.mean()),
        "l2_pass": l2_pass,
        "l3_pass": l3_pass,
        "l4_pass": l4_pass,
        "sota_pass": joint_pass,
    }


def build_paper_summary(
    meta: Dict[str, Any],
    l1_stats: Dict[str, Any],
    l2_df: pd.DataFrame,
    l3_df: pd.DataFrame,
    l4_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    rates: Dict[str, Any],
    paths: Paths,
) -> pd.DataFrame:
    l2_fail_rate = 1 - rates["L2"]
    l3_fail_rate = 1 - rates["L3"]
    l4_fail_rate = 1 - rates["L4"]
    joint_fail_rate = 1 - rates["Joint Rate"]
    fail_rates = {
        "Consistency (L2)": l2_fail_rate,
        "Magnitude (L3)": l3_fail_rate,
        "Stability (L4)": l4_fail_rate,
    }
    bottleneck = max(fail_rates, key=fail_rates.get)

    l2_fail_mask = ~(summary_df["win_rate"] > TAU_W)
    l3_fail_mask = ~(summary_df["cohens_d"] > TAU_D)
    l4_fail_mask = ~(summary_df["bp_ratio"] > TAU_B)

    n_l2_only = int((l2_fail_mask & ~l3_fail_mask & ~l4_fail_mask).sum())
    n_l3_only = int((~l2_fail_mask & l3_fail_mask & ~l4_fail_mask).sum())
    n_l4_only = int((~l2_fail_mask & ~l3_fail_mask & l4_fail_mask).sum())
    n_multi_fail = int(((l2_fail_mask.astype(int) + l3_fail_mask.astype(int) + l4_fail_mask.astype(int)) == 2).sum())
    n_all_fail = int((l2_fail_mask & l3_fail_mask & l4_fail_mask).sum())

    failure_breakdown = pd.DataFrame(
        [
            {"category": "Fail L2 only", "count": n_l2_only, "rate": n_l2_only / len(summary_df)},
            {"category": "Fail L3 only", "count": n_l3_only, "rate": n_l3_only / len(summary_df)},
            {"category": "Fail L4 only", "count": n_l4_only, "rate": n_l4_only / len(summary_df)},
            {"category": "Fail multiple levels", "count": n_multi_fail, "rate": n_multi_fail / len(summary_df)},
            {"category": "Fail all levels", "count": n_all_fail, "rate": n_all_fail / len(summary_df)},
        ]
    )
    failure_breakdown.to_csv(paths.results / "failure_breakdown_MMLU.csv", index=False)

    rows = [
        ("Benchmark", meta["name"], "-"),
        ("Domain", meta.get("domain", "N/A"), "-"),
        ("Source", meta.get("source", "N/A"), "-"),
        ("Models (total / selected)", f"{meta.get('n_models_total')} / {meta.get('n_models_selected')}", "-"),
        ("Datasets (total / selected)", f"{meta.get('n_datasets_total')} / {meta.get('n_datasets_selected')}", "-"),
        ("Model pairs", f"{len(summary_df)}", "-"),
        ("Direction", meta["direction"], "-"),
        ("Metric name", meta["metric"], "-"),
        ("---", "---", "---"),
        ("Thresholds", "---", "---"),
        ("tau_w (L2: Win Rate)", f"{TAU_W}", "-"),
        ("tau_d (L3: Cohen's d)", f"{TAU_D}", "-"),
        ("tau_b (L4: BP Ratio)", f"{TAU_B}", "-"),
        ("---", "---", "---"),
        ("L1: Winner", l1_stats["winner"][:40], "-"),
        ("L1: Runner-up", l1_stats["runner_up"][:40], "-"),
        ("L1: Winner score", f"{l1_stats['winner_mean']:.4f}", "-"),
        ("L1: Runner-up score", f"{l1_stats['runner_up_mean']:.4f}", "-"),
        ("L1: Score Gap (1st-2nd)", f"{l1_stats['score_gap']:.4f}", "-"),
        ("L1: Score Gap (%)", f"{l1_stats['score_gap'] / l1_stats['runner_up_mean'] * 100:.2f}%", "-"),
        ("L1: Friedman p", f"{l1_stats['friedman_p']:.2e}", "-"),
        ("---", "---", "---"),
        ("L2: Win Rate (mean)", f"{l2_df['win_rate'].mean():.4f}", f"{l2_df['win_rate'].std():.4f}"),
        ("L2: Neg Rate (mean)", f"{l2_df['neg_rate'].mean():.4f}", f"{l2_df['neg_rate'].std():.4f}"),
        ("L2: Tie Rate (mean)", f"{l2_df['tie_rate'].mean():.4f}", f"{l2_df['tie_rate'].std():.4f}"),
        ("L2 Pass Rate", f"{rates['L2']:.2%}", "-"),
        ("L2 Fail Rate", f"{l2_fail_rate:.2%}", "-"),
        ("---", "---", "---"),
        ("L3: Cohen's d (mean)", f"{l3_df['cohens_d'].mean():.4f}", f"{l3_df['cohens_d'].std():.4f}"),
        ("L3: Wilcoxon p<0.05 rate", f"{(l3_df['wilcoxon_p'] < 0.05).mean():.2%}", "-"),
        ("L3: CI width (mean)", f"{l3_df['ci_width'].mean():.4f}", f"{l3_df['ci_width'].std():.4f}"),
        ("L3 Pass Rate", f"{rates['L3']:.2%}", "-"),
        ("L3 Fail Rate", f"{l3_fail_rate:.2%}", "-"),
        ("---", "---", "---"),
        ("L4: BP Ratio (mean)", f"{l4_df['bp_ratio'].mean():.4f}", f"{l4_df['bp_ratio'].std():.4f}"),
        ("L4: LOO flips (mean)", f"{l4_df['loo_flips'].mean():.2f}", f"{l4_df['loo_flips'].std():.2f}"),
        ("L4: LOO flip ratio", f"{(l4_df['loo_flips'] > 0).mean():.2%}", "-"),
        ("L4 Pass Rate", f"{rates['L4']:.2%}", "-"),
        ("L4 Fail Rate", f"{l4_fail_rate:.2%}", "-"),
        ("---", "---", "---"),
        ("Joint Pass Rate", f"{rates['Joint Rate']:.2%}", "-"),
        ("Fragility Rate", f"{joint_fail_rate:.2%}", "-"),
        ("Bottleneck", bottleneck, "-"),
        ("---", "---", "---"),
        ("Fail L2 only", f"{n_l2_only} ({n_l2_only / len(summary_df):.1%})", "-"),
        ("Fail L3 only", f"{n_l3_only} ({n_l3_only / len(summary_df):.1%})", "-"),
        ("Fail L4 only", f"{n_l4_only} ({n_l4_only / len(summary_df):.1%})", "-"),
        ("Fail multiple levels", f"{n_multi_fail} ({n_multi_fail / len(summary_df):.1%})", "-"),
        ("Fail all levels", f"{n_all_fail} ({n_all_fail / len(summary_df):.1%})", "-"),
    ]
    paper_summary = pd.DataFrame(rows, columns=["Metric", "Value", "Std"])
    paper_summary.to_csv(paths.results / "paper_summary_MMLU.csv", index=False)
    return paper_summary


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    paths = build_paths(args.artifacts_dir)
    ensure_dirs(paths)

    top20_csv = paths.processed / "MMLU_top20.csv"
    meta_json = paths.processed / "MMLU_meta.json"
    if not top20_csv.exists() or not meta_json.exists():
        raise FileNotFoundError("Run preprocess first.")

    df = pd.read_csv(top20_csv, index_col=0)
    meta = read_json(meta_json)
    direction = meta["direction"]
    model_pairs = generate_model_pairs(df, direction)

    _, l1_stats = compute_l1(df, direction, paths)
    l2_df = compute_l2(df, model_pairs, direction, paths)
    l3_df = compute_l3(df, model_pairs, direction, paths, args.bootstrap_seed)
    l4_df = compute_l4(df, model_pairs, direction, paths)

    summary_df = l2_df[["winner", "loser", "win_rate", "neg_rate"]].copy()
    summary_df = summary_df.merge(
        l3_df[["winner", "loser", "cohens_d", "wilcoxon_p", "ci_width"]],
        on=["winner", "loser"],
    )
    summary_df = summary_df.merge(
        l4_df[["winner", "loser", "bp_ratio", "loo_flips", "loo_flip_ratio"]],
        on=["winner", "loser"],
    )
    summary_df.to_csv(paths.results / "summary_MMLU.csv", index=False)

    rates = compute_pass_rates(summary_df, TAU_W, TAU_D, TAU_B)
    build_paper_summary(meta, l1_stats, l2_df, l3_df, l4_df, summary_df, rates, paths)

    return {
        "n_pairs": int(len(summary_df)),
        "winner": l1_stats["winner"],
        "runner_up": l1_stats["runner_up"],
        "score_gap": l1_stats["score_gap"],
        "score_gap_percent": l1_stats["score_gap"] / l1_stats["runner_up_mean"] * 100,
        "wilcoxon_p_lt_0_05_percent": float((l3_df["wilcoxon_p"] < 0.05).mean() * 100),
        "cohens_d_mean": float(l3_df["cohens_d"].mean()),
        "cohens_d_std": float(l3_df["cohens_d"].std()),
        "magnitude_violation_percent": float((~(summary_df["cohens_d"] > TAU_D)).mean() * 100),
        "win_rate_mean_percent": float(l2_df["win_rate"].mean() * 100),
        "win_rate_std_percent": float(l2_df["win_rate"].std() * 100),
        "consistency_violation_percent": float((~(summary_df["win_rate"] > TAU_W)).mean() * 100),
        "breakpoint_rate_mean_percent": float(l4_df["bp_ratio"].mean() * 100),
        "breakpoint_rate_std_percent": float(l4_df["bp_ratio"].std() * 100),
        "stability_violation_percent": float((~(summary_df["bp_ratio"] > TAU_B)).mean() * 100),
        "fragility_rate_percent": float(
            ((~(summary_df["win_rate"] > TAU_W)) | (~(summary_df["cohens_d"] > TAU_D)) | (~(summary_df["bp_ratio"] > TAU_B))).mean() * 100
        ),
        "summary_csv": display_path(paths.results / "summary_MMLU.csv"),
        "paper_summary_csv": display_path(paths.results / "paper_summary_MMLU.csv"),
    }


def parse_percent(value: str) -> float:
    return float(value.replace("\\%", "").replace("%", "").strip())


def parse_mean_std(value: str) -> Tuple[float, Optional[float]]:
    match = re.match(r"\s*([0-9.]+)\s*(?:\(([0-9.]+)\))?\s*", value)
    if not match:
        raise ValueError(f"Cannot parse mean/std value: {value}")
    mean = float(match.group(1))
    std = float(match.group(2)) if match.group(2) is not None else None
    return mean, std


def first_table_cell(row_line: str) -> str:
    cells = [cell.strip() for cell in row_line.split("&")]
    if len(cells) < 2:
        raise ValueError(f"Cannot parse table row: {row_line}")
    cell = cells[1].split("\\\\")[0].strip()
    return cell


def parse_paper_table1_mmlu() -> Dict[str, Any]:
    manifest = load_manifest()
    expected = dict(manifest["paper_table1_mmlu"])
    expected["source"] = "manifests/mmlu.json"
    return expected


def compare_metric(name: str, computed: float, expected: float, tolerance: float) -> Dict[str, Any]:
    delta = computed - expected
    return {
        "metric": name,
        "computed": computed,
        "expected": expected,
        "delta": delta,
        "tolerance": tolerance,
        "status": "PASS" if abs(delta) <= tolerance else "FAIL",
    }


def load_computed_metrics(paths: Paths) -> Dict[str, Any]:
    summary_csv = paths.results / "summary_MMLU.csv"
    paper_summary_csv = paths.results / "paper_summary_MMLU.csv"
    l1_csv = paths.results / "L1_ranking_MMLU.csv"
    if not summary_csv.exists() or not paper_summary_csv.exists() or not l1_csv.exists():
        raise FileNotFoundError("Run analyze first.")

    summary = pd.read_csv(summary_csv)
    paper_summary = pd.read_csv(paper_summary_csv)
    paper_lookup = dict(zip(paper_summary["Metric"], paper_summary["Value"]))
    paper_std_lookup = dict(zip(paper_summary["Metric"], paper_summary["Std"]))

    def value_float(metric: str, percent: bool = False) -> float:
        value = str(paper_lookup[metric])
        if percent:
            return parse_percent(value)
        return float(value)

    def std_float(metric: str, percent_scale: bool = False) -> float:
        value = str(paper_std_lookup[metric])
        out = float(value)
        return out * 100 if percent_scale else out

    return {
        "model_pairs": int(len(summary)),
        "score_gap": value_float("L1: Score Gap (1st-2nd)"),
        "score_gap_percent": value_float("L1: Score Gap (%)", percent=True),
        "wilcoxon_p_lt_0_05_percent": value_float("L3: Wilcoxon p<0.05 rate", percent=True),
        "cohens_d_mean": value_float("L3: Cohen's d (mean)"),
        "cohens_d_std": std_float("L3: Cohen's d (mean)"),
        "magnitude_violation_percent": value_float("L3 Fail Rate", percent=True),
        "win_rate_mean_percent": value_float("L2: Win Rate (mean)") * 100,
        "win_rate_std_percent": std_float("L2: Win Rate (mean)", percent_scale=True),
        "consistency_violation_percent": value_float("L2 Fail Rate", percent=True),
        "breakpoint_rate_mean_percent": value_float("L4: BP Ratio (mean)") * 100,
        "breakpoint_rate_std_percent": std_float("L4: BP Ratio (mean)", percent_scale=True),
        "stability_violation_percent": value_float("L4 Fail Rate", percent=True),
        "fragility_rate_percent": value_float("Fragility Rate", percent=True),
    }


def verify_paper(args: argparse.Namespace) -> Dict[str, Any]:
    paths = build_paths(args.artifacts_dir)
    ensure_dirs(paths)
    expected = parse_paper_table1_mmlu()
    computed = load_computed_metrics(paths)

    checks = [
        compare_metric("model_pairs", computed["model_pairs"], expected["model_pairs"], 0),
        compare_metric("score_gap", computed["score_gap"], expected["score_gap"], 0.00005),
        compare_metric("score_gap_percent", computed["score_gap_percent"], expected["score_gap_percent"], 0.005),
        compare_metric("wilcoxon_p_lt_0_05_percent", computed["wilcoxon_p_lt_0_05_percent"], expected["wilcoxon_p_lt_0_05_percent"], 0.01),
        compare_metric("cohens_d_mean", computed["cohens_d_mean"], expected["cohens_d_mean"], 0.005),
        compare_metric("cohens_d_std", computed["cohens_d_std"], expected["cohens_d_std"], 0.005),
        compare_metric("magnitude_violation_percent", computed["magnitude_violation_percent"], expected["magnitude_violation_percent"], 0.01),
        compare_metric("win_rate_mean_percent", computed["win_rate_mean_percent"], expected["win_rate_mean_percent"], 0.01),
        compare_metric("win_rate_std_percent", computed["win_rate_std_percent"], expected["win_rate_std_percent"], 0.01),
        compare_metric("consistency_violation_percent", computed["consistency_violation_percent"], expected["consistency_violation_percent"], 0.01),
        compare_metric("breakpoint_rate_mean_percent", computed["breakpoint_rate_mean_percent"], expected["breakpoint_rate_mean_percent"], 0.01),
        compare_metric("breakpoint_rate_std_percent", computed["breakpoint_rate_std_percent"], expected["breakpoint_rate_std_percent"], 0.01),
        compare_metric("stability_violation_percent", computed["stability_violation_percent"], expected["stability_violation_percent"], 0.01),
        compare_metric("fragility_rate_percent", computed["fragility_rate_percent"], expected["fragility_rate_percent"], 0.01),
    ]
    verdict = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    report = {
        "benchmark": BENCHMARK_NAME,
        "paper_source": expected.get("source", "manifests/mmlu.json"),
        "verdict": verdict,
        "checks": checks,
        "computed": computed,
        "expected": expected,
    }
    write_json(paths.reports / "paper_verification.json", report)
    write_reproduction_report(paths, report)
    return report


def write_reproduction_report(paths: Paths, paper_report: Dict[str, Any]) -> None:
    source_manifest_path = paths.source / "source_manifest.json"
    data_comparison_path = paths.reports / "data_comparison.json"
    source_manifest = read_json(source_manifest_path) if source_manifest_path.exists() else {}
    data_comparison = read_json(data_comparison_path) if data_comparison_path.exists() else {}

    lines = [
        "# MMLU Reproduction Report",
        "",
        f"- Benchmark: {BENCHMARK_NAME}",
        f"- Collection mode: {source_manifest.get('collection_mode', 'unknown')}",
        f"- Release: {source_manifest.get('release', 'unknown')}",
        f"- Snapshot date: {source_manifest.get('snapshot_date', 'unknown')}",
        f"- Data comparison verdict: {data_comparison.get('verdict', 'unknown')}",
        f"- Paper verification verdict: {paper_report.get('verdict', 'unknown')}",
        "",
        "## Data Comparison",
        "",
    ]
    normalized = data_comparison.get("normalized_vs_reference", {})
    full = data_comparison.get("full_precision_vs_reference", {}) or {}
    lines.extend(
        [
            f"- Normalized source columns equal reference: {normalized.get('columns_equal')}",
            f"- Normalized source models equal reference: {normalized.get('models_equal')}",
            f"- Normalized source max abs delta: {normalized.get('numeric_max_abs_delta')}",
            f"- Full-precision official max abs delta vs reference: {full.get('numeric_max_abs_delta')}",
            "",
            "## Paper Table 1 Checks",
            "",
            "| Metric | Computed | Paper | Delta | Status |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for check in paper_report.get("checks", []):
        lines.append(
            f"| {check['metric']} | {check['computed']:.6g} | {check['expected']:.6g} | {check['delta']:.6g} | {check['status']} |"
        )
    lines.append("")
    lines.append("Failure masks use `fail = ~(metric > threshold)`, equivalent to `metric <= threshold`.")
    (paths.reports / "reproduction_report.md").write_text("\n".join(lines) + "\n")


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    collected = collect(args)
    data_report = verify_data(args)
    prep_report = preprocess(args)
    analysis_report = analyze(args)
    paper_report = verify_paper(args)
    return {
        "collect": collected,
        "verify_data": data_report,
        "preprocess": prep_report,
        "analyze": analysis_report,
        "verify_paper": paper_report,
    }


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory for generated artifacts.",
    )
    parser.add_argument(
        "--release",
        default=None,
        help="HELM release override. Defaults to the release from config.js or manifest.",
    )
    parser.add_argument(
        "--source-url",
        default=None,
        help="Optional direct CSV URL. The default collector uses HELM group JSON.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip network collection and use data/MMLU/helm_mmlu_leaderboard.csv as the source snapshot.",
    )
    parser.add_argument(
        "--strict-collection",
        action="store_true",
        help="Fail instead of using the included reference snapshot when official collection fails.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260110,
        help="Seed for deterministic bootstrap CI widths.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["collect", "verify-data", "preprocess", "analyze", "verify-paper", "all"]:
        subparser = subparsers.add_parser(name)
        add_common_options(subparser)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "collect":
            result = collect(args)
        elif args.command == "verify-data":
            result = verify_data(args)
        elif args.command == "preprocess":
            result = preprocess(args)
        elif args.command == "analyze":
            result = analyze(args)
        elif args.command == "verify-paper":
            result = verify_paper(args)
        elif args.command == "all":
            result = run_all(args)
        else:
            parser.error(f"Unknown command: {args.command}")
            return 2
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command in {"verify-paper", "all"}:
        report = result if args.command == "verify-paper" else result["verify_paper"]
        return 0 if report.get("verdict") == "PASS" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
