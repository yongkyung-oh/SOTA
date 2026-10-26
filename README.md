# Position: State-of-the-Art Claims Require State-of-the-Art Evidence

**TL;DR:** A position paper arguing that a marginal gain in mean score signals a top average rank, not genuine superiority. Across ten cross-domain benchmarks, in over half of top-model comparisons at least one commonly assumed property of superiority fails to hold, with reported gains often driven by outlier datasets rather than consistent improvement. This repository is the home of the paper and its reproducibility code.

- **Author:** YongKyung Oh
- **Venue:** ICML 2026 (position paper), PMLR 306, pp. 171583–171601
- **Paper:** [PMLR v306](https://proceedings.mlr.press/v306/oh26d.html) · arXiv: [arXiv:2605.17273](https://arxiv.org/abs/2605.17273) · arXiv DOI: [10.48550/arXiv.2605.17273](https://doi.org/10.48550/arXiv.2605.17273)
- **Keywords:** Benchmarking, Evaluation, Statistical Methods, Reproducibility

## Overview

State-of-the-art (SOTA) claims in machine learning are frequently asserted from small differences in average metrics. This work re-examines what such gains actually imply, showing that a higher mean rank does not guarantee robust, consistent, or statistically meaningful superiority across datasets. Superiority is assessed through three properties of a top-model comparison — **magnitude** (Cohen's *d*), **consistency** (win rate across tasks), and **stability** (breakdown-point) — and a pair is *fragile* when it fails at least one. The repository collects the analysis, benchmark protocols, and reproducibility code behind these findings.

> **Status:** Under active development. Code, benchmark protocols, and analysis scripts are being released progressively.

## Reproducibility packages

Each package is self-contained and regenerates part of the paper from public data:

| Package | Reproduces | Source data |
|---|---|---|
| [`sota-counter`](sota-counter/) | Figure 1 + Appendix A — SOTA-mention counts across major AI venues (2021–2025) | [Paper Copilot](https://github.com/papercopilot/paperlists) |
| [`sota-evidence`](sota-evidence/) | Table 1 (MMLU) — fragility of top-model comparisons | [HELM MMLU](https://crfm.stanford.edu/helm/mmlu/) v1.13.0 |
| [`benchmark-fragility`](https://github.com/yongkyung-oh/benchmark-fragility) | the fragility metrics as a reusable library (Cohen's *d* · win rate · breakdown-point) | — |

[`benchmark-fragility`](https://github.com/yongkyung-oh/benchmark-fragility) is maintained as a standalone repository and packages the metrics used throughout the paper; [`sota-evidence`](sota-evidence/) reproduces the MMLU table with the same diagnostics. The benchmark summary tables — the paper snapshot (Dec 31, 2025) and a live refresh (May 31, 2026) — are documented in [`BENCHMARK_SUMMARY.md`](BENCHMARK_SUMMARY.md), with the CSVs in [`benchmark-summary/`](benchmark-summary/).

## 📄 Citation

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff). Please cite the
official PMLR version; the arXiv entry is listed separately.

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

## License

Released under the [MIT License](LICENSE).
