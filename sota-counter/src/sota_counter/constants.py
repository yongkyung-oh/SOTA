from __future__ import annotations

from dataclasses import dataclass

REFERENCE_PAPERLISTS_COMMIT = "29c55620f95f7965b7d2772fab08f35e08dacc8f"

YEARS = (2021, 2022, 2023, 2024, 2025)


@dataclass(frozen=True)
class Conference:
    slug: str
    label: str


CONFERENCES = (
    Conference("icml", "ICML"),
    Conference("iclr", "ICLR"),
    Conference("nips", "NeurIPS"),
    Conference("aaai", "AAAI"),
    Conference("cvpr", "CVPR"),
    Conference("acl", "ACL"),
)

REJECTED_STATUSES = {
    "Reject",
    "Rejected",
    "Withdraw",
    "Withdrawn",
    "Desk Reject",
    "Desk-Reject",
    "DeskRejected",
    "NeurIPS 2023 Conference Withdrawn Submission",
    None,
}

AAAI_VALID_TRACKS = {
    "main",
    "aaai special track",
    "aaai technical track focus area",
}

ACL_MAIN_STATUSES = {
    "Long",
    "Short",
    "Main",
    "Long Main",
    "Short Main",
}

REFERENCE_ACCEPTED = {
    "ICML": {2021: 1183, 2022: 1233, 2023: 1865, 2024: 2610, 2025: 3342},
    "ICLR": {2021: 860, 2022: 1095, 2023: 1575, 2024: 2260, 2025: 3704},
    "NeurIPS": {2021: 2508, 2022: 2901, 2023: 3584, 2024: 4562, 2025: 5812},
    "AAAI": {2021: 1654, 2022: 1319, 2023: 1720, 2024: 2501, 2025: 3182},
    "CVPR": {2021: 1661, 2022: 2062, 2023: 2359, 2024: 2716, 2025: 2871},
    "ACL": {2021: 710, 2022: 700, 2023: 1075, 2024: 940, 2025: 1699},
}

REFERENCE_SOTA = {
    "ICML": {2021: 264, 2022: 262, 2023: 386, 2024: 566, 2025: 783},
    "ICLR": {2021: 246, 2022: 318, 2023: 476, 2024: 629, 2025: 903},
    "NeurIPS": {2021: 627, 2022: 716, 2023: 895, 2024: 1143, 2025: 1611},
    "AAAI": {2021: 609, 2022: 515, 2023: 666, 2024: 1000, 2025: 1158},
    "CVPR": {2021: 885, 2022: 1068, 2023: 1090, 2024: 1147, 2025: 1184},
    "ACL": {2021: 232, 2022: 211, 2023: 299, 2024: 193, 2025: 335},
}

REFERENCE_RATIOS = {
    "ICML": {2021: 22.32, 2022: 21.25, 2023: 20.70, 2024: 21.69, 2025: 23.43},
    "ICLR": {2021: 28.60, 2022: 29.04, 2023: 30.22, 2024: 27.83, 2025: 24.38},
    "NeurIPS": {2021: 25.00, 2022: 24.68, 2023: 24.97, 2024: 25.05, 2025: 27.72},
    "AAAI": {2021: 36.82, 2022: 39.04, 2023: 38.72, 2024: 39.98, 2025: 36.39},
    "CVPR": {2021: 53.28, 2022: 51.79, 2023: 46.21, 2024: 42.23, 2025: 41.24},
    "ACL": {2021: 32.68, 2022: 30.14, 2023: 27.81, 2024: 20.53, 2025: 19.72},
}
