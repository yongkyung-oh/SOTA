from __future__ import annotations

import argparse
from pathlib import Path

from .constants import REFERENCE_PAPERLISTS_COMMIT
from .core import compute_counts
from .output import (
    write_check,
    write_figure1,
    write_tables,
)
from .check import check_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sota-counter",
        description="Reproduce Figure 1 and Appendix A SOTA mention analysis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    reproduce = subparsers.add_parser("reproduce", help="Generate and check results.")
    reproduce.add_argument("--paperlists-dir", type=Path, default=Path("data/paperlists"))
    reproduce.add_argument("--out-dir", type=Path, default=Path("results"))

    return parser


def reproduce(args: argparse.Namespace) -> int:
    paperlists_dir = args.paperlists_dir
    out_dir = args.out_dir

    df = compute_counts(paperlists_dir)
    write_tables(df, out_dir)
    write_figure1(df, out_dir)
    report = check_results(df, out_dir, paperlists_dir)
    write_check(report, out_dir)

    if not report["ok"]:
        print("Result check failed.")
        for issue in report["issues"]:
            print(f"- {issue}")
        return 1

    print(f"Results written to {out_dir}")
    print(f"Data version: {report.get('data_commit')}")
    print(f"Reference version: {REFERENCE_PAPERLISTS_COMMIT}")
    print(f"Reference tables: {'match' if report['matches_reference'] else 'changed'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "reproduce":
        return reproduce(args)

    parser.error(f"Unknown command: {args.command}")
    return 2
