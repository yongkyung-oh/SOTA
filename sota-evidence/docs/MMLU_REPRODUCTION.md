# MMLU Reproduction

## Source Data

Offline mode uses the included HELM MMLU `v1.13.0` snapshot. Online mode
uses the release advertised by HELM; pass `--release v1.13.0` to select it explicitly.

- Project page: `https://crfm.stanford.edu/helm/mmlu/`
- Snapshot date: `2025-01-10`
- Main table JSON: `releases/v1.13.0/groups/mmlu.json`

The HELM JSON contains full-precision values. The included reference CSV stores
the same leaderboard in the three-decimal format used for the table check.

## Preprocessing

1. Read the HELM MMLU leaderboard.
2. Use the 57 subject-level exact-match columns.
3. Arrange data as rows = subjects and columns = models.
4. Rank models by mean score across the 57 subjects.
5. Select the top 20 models.
6. Analyze all 190 model pairs.

## Diagnostics

For each pair, the winner is the model with the higher mean score.

- Consistency: win rate across 57 subjects.
- Magnitude: paired Cohen's d on winner-minus-loser differences.
- Stability: breakdown-point ratio, computed by removing the winner's most favorable subjects until the mean-score ordering changes.
- Fragility: a pair fails at least one of the three checks.

The thresholds are:

| Check | Pass Rule | Failure Rule |
|---|---|---|
| Consistency | win rate > 0.60 | win rate <= 0.60 |
| Magnitude | Cohen's d > 0.20 | Cohen's d <= 0.20 |
| Stability | BP ratio > 0.20 | BP ratio <= 0.20 |

## Output Files

```bash
python scripts/reproduce_mmlu.py all --artifacts-dir "$HOME/sota-output/mmlu"
```

Use `--artifacts-dir` to choose an output directory outside the source checkout.
The command above writes:

- `$HOME/sota-output/mmlu/source/`: collected HELM files and normalized CSV.
- `$HOME/sota-output/mmlu/processed/`: full and top-k matrices.
- `$HOME/sota-output/mmlu/results/`: pairwise diagnostics and paper-summary CSV.
