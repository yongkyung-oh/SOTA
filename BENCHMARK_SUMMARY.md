# Benchmark Summary Tables

CSV versions of the paper's benchmark summary table (Table 1). Raw benchmark data are not included. Part of the [SOTA](README.md) repository; the CSV files live in [`benchmark-summary/`](benchmark-summary/).

| Version | Cut-off | File |
|---|---:|---|
| Paper snapshot | Dec 31, 2025 | [`benchmark_summary_snapshot.csv`](benchmark-summary/benchmark_summary_snapshot.csv) |
| Live refresh | May 31, 2026 | [`benchmark_summary_live.csv`](benchmark-summary/benchmark_summary_live.csv) |

## Scope

- `benchmark_summary_snapshot.csv` follows the paper Table 1 source: `paper/arxiv_2605_17273/source/tables/benchmark_summary.tex`.
- `benchmark_summary_live.csv` keeps the same table layout and recomputes the affected columns using refreshed public leaderboard data.
- LiveBench uses the detected `2026-01-08` release.

## Changes

| Benchmark | Snapshot | Live | Effect |
|---|---:|---:|---|
| LiveBench | 20 (58) models, 21 (21) datasets | 16 (48) models, 23 (23) datasets | Fragility: 44.21% -> 50.83% |
| Open ASR | 20 (39) models, 8 (8) datasets | 20 (90) models, 7 (7) datasets | Fragility: 45.79% -> 43.68% |
| VBench | 20 (68) certified models | 20 (72) certified models | Fragility unchanged at 53.68% |
| Open VLM | 20 (284) raw rows | 20 (285) raw rows | Diagnostic values unchanged |
| MMLU, TabArena, TSFM | No table-level change | No table-level change | Diagnostic values unchanged |

## Notes

- Open ASR drops `Tedlium` in the live source.
- `RTFx` is excluded from Open ASR diagnostics. It is an inference-speed field, not a WER task.
- Snapshot and live tables are compared as observed. Live data are not forced onto the paper snapshot model/task subset.
- Excluded artifacts: raw data, upstream dumps, JavaScript bundles, PDFs, and SSE responses.
