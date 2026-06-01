# Position: State-of-the-Art Claims Require State-of-the-Art Evidence

**TL;DR:** A position paper arguing that a marginal gain in mean score signals a top average rank, not genuine superiority. Across ten cross-domain benchmarks, in over half of top-model comparisons at least one commonly assumed property of superiority fails to hold, with reported gains often driven by outlier datasets rather than consistent improvement. This repository hosts the accompanying analysis and benchmarking code.

- **Author:** YongKyung Oh
- **Venue:** ICML 2026 (position paper, in press)
- **Paper:** [arXiv:2605.17273](https://arxiv.org/abs/2605.17273) · arXiv DOI: [10.48550/arXiv.2605.17273](https://doi.org/10.48550/arXiv.2605.17273)
- **Keywords:** state-of-the-art evaluation, benchmarking, statistical significance, machine learning evaluation

## Overview

State-of-the-art (SOTA) claims in machine learning are frequently asserted from small differences in average metrics. This work re-examines what such gains actually imply, showing that a higher mean rank does not guarantee robust, consistent, or statistically meaningful superiority across datasets. The repository collects the code, benchmark protocols, and analysis scripts used to support these findings.

> **Status:** Under active development. Code, benchmark protocols, and analysis scripts are being released progressively.

## Citation

This paper is accepted at ICML 2026 (Position Paper Track) and is in press (no proceedings DOI or page numbers assigned yet). Until the official proceedings appear, the arXiv version ([arXiv:2605.17273](https://arxiv.org/abs/2605.17273)) is available. Machine-readable metadata is available in [`CITATION.cff`](CITATION.cff).

```bibtex
@inproceedings{oh2026sota,
  title     = {Position: State-of-the-Art Claims Require State-of-the-Art Evidence},
  author    = {Oh, YongKyung},
  booktitle = {Proceedings of the Forty-third International Conference on Machine Learning (ICML), Position Paper Track},
  year      = {2026},
  note      = {Position paper, in press},
  eprint    = {2605.17273},
  archivePrefix = {arXiv},
  doi       = {10.48550/arXiv.2605.17273},
  url       = {https://arxiv.org/abs/2605.17273}
}
```

## License

Released under the [MIT License](LICENSE).
