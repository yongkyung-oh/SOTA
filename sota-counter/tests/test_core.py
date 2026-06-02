from __future__ import annotations

import json
import os
from pathlib import Path

from sota_counter.cli import main
from sota_counter.core import SOTA_PATTERN, filter_papers


ROOT = Path(__file__).resolve().parents[1]
PAPERLISTS_DIR = Path(os.environ.get("PAPERLISTS_DIR", ROOT / "data" / "paperlists"))


def test_sota_pattern_matches_expected_forms() -> None:
    positives = [
        "achieves state-of-the-art performance",
        "state of the art results",
        "SOTA accuracy",
        "SotA model",
        "state‑of‑the‑art",
        "S.O.T.A. benchmark",
    ]
    negatives = ["our method achieves the best", "", "state art"]

    for text in positives:
        assert SOTA_PATTERN.search(text), text
    for text in negatives:
        assert not SOTA_PATTERN.search(text), text


def test_conference_specific_filters() -> None:
    papers = [
        {"status": "Technical", "track": "main"},
        {"status": " Technical ", "track": " AAAI Special Track "},
        {"status": "Technical", "track": "other"},
        {"status": " reject ", "track": "main"},
    ]
    assert len(filter_papers(papers, "aaai")) == 2

    acl_papers = [
        {"status": "Long"},
        {"status": " short "},
        {"status": "Findings"},
        {"status": "Industry"},
    ]
    assert len(filter_papers(acl_papers, "acl")) == 2


def test_default_data_path_is_inside_release_folder() -> None:
    assert PAPERLISTS_DIR == ROOT / "data" / "paperlists"


def test_cli_writes_public_outputs_for_current_data(tmp_path: Path) -> None:
    paperlists = tmp_path / "paperlists"
    for slug in ("icml", "iclr", "nips", "aaai", "cvpr", "acl"):
        conf_dir = paperlists / slug
        conf_dir.mkdir(parents=True)
        for year in range(2021, 2026):
            record = {
                "status": "Poster",
                "track": "main",
                "abstract": "A simple baseline.",
            }
            if slug == "aaai":
                record["status"] = "Technical"
            if slug == "acl":
                record["status"] = "Long"
            (conf_dir / f"{slug}{year}.json").write_text(
                json.dumps([record]),
                encoding="utf-8",
            )

    out_dir = tmp_path / "results"
    code = main(["reproduce", "--paperlists-dir", str(paperlists), "--out-dir", str(out_dir)])
    assert code == 0

    expected_files = [
        out_dir / "tables" / "accepted_counts.csv",
        out_dir / "tables" / "sota_counts.csv",
        out_dir / "tables" / "sota_ratios.csv",
        out_dir / "tables" / "appendix_a_tables.tex",
        out_dir / "figure1" / "figure1_input.csv",
        out_dir / "figure1" / "trend.png",
        out_dir / "figure1" / "trend.pdf",
        out_dir / "check.json",
    ]
    for path in expected_files:
        assert path.exists()
        assert path.stat().st_size > 0

    check = json.loads((out_dir / "check.json").read_text(encoding="utf-8"))
    assert set(check) == {"ok", "data_commit", "reference_commit", "matches_reference", "issues"}
    assert check["ok"] is True
    assert check["matches_reference"] is False
