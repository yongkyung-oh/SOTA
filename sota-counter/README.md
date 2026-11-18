<div align="center">

# SOTA Mention Reproduction

**Reproduce Figure 1 and Appendix A** of
*Position: State-of-the-Art Claims Require State-of-the-Art Evidence*

Counts accepted papers whose abstracts mention state-of-the-art performance —
using [Paper Copilot](https://github.com/papercopilot/paperlists) metadata — and writes the paper's figure and tables.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/reproduction-Figure%201%20%2B%20Appendix%20A-B22222)
![Data](https://img.shields.io/badge/source-Paper%20Copilot-555)

</div>

---

> **Part of the [SOTA](../) reproducibility package** for *Position: State-of-the-Art Claims Require State-of-the-Art Evidence* ([arXiv:2605.17273](https://arxiv.org/abs/2605.17273)).

**Companion packages**

| Package | Reproduces |
|---|---|
| [`benchmark-fragility`](https://github.com/yongkyung-oh/benchmark-fragility) | reusable fragility metrics — Cohen's *d* · win rate · breakdown-point |
| [`sota-evidence`](../sota-evidence/) | Table 1 — MMLU fragility |
| **`sota-counter`** _(this package)_ | Figure 1 + Appendix A — SOTA mentions |

---

## 🚀 Quickstart

> [!NOTE]
> Requires **Python 3.10**. Use the included `environment.yml` for a ready-made
> conda environment.

```bash
conda env create -f environment.yml
conda activate sota-counter
python -m pip install -e .
mkdir -p data
git clone https://github.com/papercopilot/paperlists data/paperlists
python -m sota_counter reproduce --out-dir "$HOME/sota-output/counter"
```

Point at a different Paper Copilot checkout if you already have one:

```bash
python -m sota_counter reproduce --paperlists-dir path/to/paperlists --out-dir "$HOME/sota-output/counter"
```

<details>
<summary>📦 What it generates</summary>

Choose an output directory outside the source checkout with `--out-dir`.

```text
$HOME/sota-output/counter/
├── figure1/
│   ├── trend.png            # Figure 1
│   ├── trend.pdf
│   └── figure1_input.csv
└── tables/
    ├── accepted_counts.csv
    ├── sota_counts.csv
    ├── sota_ratios.csv
    └── appendix_a_tables.tex   # Appendix A
```

</details>

---

## 🔖 Data version

Use the reference commit below for the paper tables.

The paper tables were generated from Paper Copilot commit:

```text
29c55620f95f7965b7d2772fab08f35e08dacc8f
```

```bash
git -C data/paperlists checkout 29c55620f95f7965b7d2772fab08f35e08dacc8f
```

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
