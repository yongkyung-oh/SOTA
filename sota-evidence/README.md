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
python scripts/reproduce_mmlu.py all --offline --artifacts-dir "$HOME/sota-output/mmlu"
```

The command computes the MMLU statistics.

<details>
<summary>📦 Other ways to run</summary>

```bash
# pip instead of conda
pip install -r requirements.txt
python scripts/reproduce_mmlu.py all --offline --artifacts-dir "$HOME/sota-output/mmlu"

# online: download the HELM data
python scripts/reproduce_mmlu.py all --artifacts-dir "$HOME/sota-output/mmlu"
```

Use `--artifacts-dir` to choose an output directory outside the source checkout.
The commands above write tables under `$HOME/sota-output/mmlu/results/`.

</details>

---

## 🔖 Data version

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
@inproceedings{oh_sota_2026,
  author      = {Oh, YongKyung},
  title       = {Position: {State}-of-the-{Art} {Claims} {Require} {State}-of-the-{Art} {Evidence}},
  booktitle   = {Proceedings of the 43rd {International} {Conference} on {Machine} {Learning}},
  series      = {Proceedings of Machine Learning Research},
  volume      = 306,
  pages       = {171583--171601},
  publisher   = {PMLR},
  year        = 2026,
  editor      = {Zhang, Tong and Dudik, Miroslav and Jaggi, Martin and Agarwal, Alekh and Li, Sharon and Schuurmans, Dale and Zhu, Jerry and Berkenkamp, Felix and Dong, Hanze and Bietti, Alberto},
  url         = {https://proceedings.mlr.press/v306/oh26d.html}
}
```

```bibtex
@misc{oh_sota_2026_arxiv,
  author      = {Oh, YongKyung},
  title       = {Position: {State}-of-the-{Art} {Claims} {Require} {State}-of-the-{Art} {Evidence}},
  publisher   = {arXiv},
  year        = 2026,
  doi         = {10.48550/arXiv.2605.17273},
  url         = {https://arxiv.org/abs/2605.17273}
}
```
