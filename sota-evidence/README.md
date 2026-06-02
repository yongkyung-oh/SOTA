<div align="center">

# SOTA Evidence Reproducibility

**Reproduce the MMLU column of Table 1** in
*Position: State-of-the-Art Claims Require State-of-the-Art Evidence*

Recomputes the paper's MMLU fragility numbers from the public
[HELM MMLU leaderboard](https://crfm.stanford.edu/helm/mmlu/) — no model inference,
just the published scores.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Reproduction](https://img.shields.io/badge/reproduction-Table%201%20(MMLU)-B22222)
![Source](https://img.shields.io/badge/source-HELM%20MMLU%20v1.13.0-555)

</div>

---

> **Part of the [SOTA](../) reproducibility package** for *Position: State-of-the-Art Claims Require State-of-the-Art Evidence* ([arXiv:2605.17273](https://arxiv.org/abs/2605.17273)).

**Companion packages**

| Package | Reproduces |
|---|---|
| [`benchmark-fragility`](https://github.com/yongkyung-oh/benchmark-fragility) | reusable fragility metrics — Cohen's *d* · win rate · breakdown-point |
| **`sota-evidence`** _(this package)_ | Table 1 — MMLU fragility |
| [`sota-counter`](../sota-counter/) | Figure 1 + Appendix A — SOTA mentions |

---

## 🚀 Quickstart

> [!NOTE]
> Requires **Python 3.10**. Offline mode needs no network — it uses the
> leaderboard snapshot included in the package.

```bash
conda env create -f environment.yml
conda activate sota-evidence
python scripts/reproduce_mmlu.py all --offline
```

The run reproduces the paper's MMLU numbers and prints `PASS` when they match.

<details>
<summary>📦 Other ways to run</summary>

```bash
# pip instead of conda
pip install -r requirements.txt
python scripts/reproduce_mmlu.py all --offline

# online: download the live HELM data and check it against the snapshot
python scripts/reproduce_mmlu.py all
```

Outputs (tables and reports) are written under `artifacts/` and are not tracked
by Git.

</details>

---

## 📋 Expected result

The check reproduces the paper's MMLU row (top 20 models, 190 pairs):

| Quantity | Expected |
|---|---:|
| Model pairs | 190 |
| Win rate | 69.88 (16.10) |
| Consistency violation | 33.68% |
| Cohen's d | 0.53 (0.35) |
| Magnitude violation | 23.16% |
| Breakdown-point rate | 39.45 (29.24) |
| Stability violation | 35.26% |
| **Fragility rate** | **39.47%** |

> *Fragility is the share of model pairs where the higher-mean model fails at
> least one of three tests: consistent wins, a meaningful effect size, or
> stability to removing favorable subjects.*

> [!IMPORTANT]
> **NumPy version and exact reproduction.** The package runs on any recent NumPy,
> but the breakdown-point rate matches the paper's **39.45** exactly only on
> **NumPy 1.x**. In two of the 190 pairs, the breakdown step compares two subset
> means that are *exactly tied*; NumPy's summation order decides the tie, and
> NumPy ≥ 2.0 resolves it the other way, giving **39.43** (fragility itself is
> unchanged at 39.47%). To reproduce the published table value, use `numpy<2`
> (e.g. the included `environment.yml` with NumPy 1.25).

---

## 🔖 Data version

> [!TIP]
> The package is pinned to the exact snapshot used in the paper, so the numbers
> match out of the box.

The MMLU scores come from HELM release **`v1.13.0`** (**2025-01-10**) — 79 models
across 57 subjects.

```text
https://crfm.stanford.edu/helm/mmlu/  →  releases/v1.13.0/groups/mmlu.json
```

---

## 🎯 Scope

This package covers **MMLU only** — it reproduces the numbers behind the paper's
MMLU row, not every benchmark or figure. The full method (data, selection,
thresholds) is in [`docs/MMLU_REPRODUCTION.md`](docs/MMLU_REPRODUCTION.md).

---

## 📄 Citation

```bibtex
@misc{oh_sota_2026,
  title        = {Position: {State}-of-the-{Art} {Claims} {Require} {State}-of-the-{Art} {Evidence}},
  author       = {Oh, YongKyung},
  year         = 2026,
  publisher    = {arXiv},
  doi          = {10.48550/arXiv.2605.17273}
}
```
