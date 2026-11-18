from __future__ import annotations

import argparse
from pathlib import Path

from .constants import REFERENCE_PAPERLISTS_COMMIT
from .core import compute_counts, git_commit
from .output import write_figure1, write_tables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sota-counter",
        description="Reproduce Figure 1 and Appendix A SOTA mention analysis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    reproduce = subparsers.add_parser("reproduce", help="Generate the paper tables and Figure 1.")
    reproduce.add_argument("--paperlists-dir", type=Path, default=Path("data/paperlists"))
    reproduce.add_argument("--out-dir", type=Path, required=True,
                           help="Output directory outside the source tree.")

    return parser


def reproduce(args: argparse.Namespace) -> int:
    paperlists_dir = args.paperlists_dir
    out_dir = args.out_dir.expanduser().resolve()
    package_root = Path(__file__).resolve().parents[2]
    source_root = next(
        (parent for parent in (package_root, *package_root.parents)
         if (parent / ".git").exists()),
        package_root.parent,
    )
    if out_dir == source_root or source_root in out_dir.parents:
        raise ValueError("Choose an output directory outside the source tree.")

    df = compute_counts(paperlists_dir)
    write_tables(df, out_dir)
    write_figure1(df, out_dir)

    print(f"Results written to {out_dir}")
    print(f"Data version: {git_commit(paperlists_dir)}")
    print(f"Reference version: {REFERENCE_PAPERLISTS_COMMIT}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "reproduce":
        return reproduce(args)

    parser.error(f"Unknown command: {args.command}")
    return 2
