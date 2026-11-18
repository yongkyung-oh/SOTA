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
