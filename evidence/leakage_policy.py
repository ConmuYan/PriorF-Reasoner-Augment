"""Shared leakage-policy constants for student-visible Evidence Cards."""

from __future__ import annotations

from typing import Final

LEAKAGE_POLICY_VERSION: Final[str] = "evidence_leakage_policy/v1"
NEIGHBOR_LABEL_POLICY: Final[str] = "removed_from_student_visible"
EVIDENCE_CARD_PROJECTION: Final[str] = "student_safe_v1"
INTERNAL_EVIDENCE_CARD_PROJECTION: Final[str] = "internal_full_v1"

STUDENT_VISIBLE_FORBIDDEN_FIELDS: Final[tuple[str, ...]] = (
    "labeled_neighbors",
    "positive_neighbors",
    "negative_neighbors",
    "unlabeled_neighbors",
)
STUDENT_VISIBLE_FORBIDDEN_FIELD_PATHS: Final[tuple[str, ...]] = tuple(
    f"neighbor_summary.{field}" for field in STUDENT_VISIBLE_FORBIDDEN_FIELDS
)
STUDENT_VISIBLE_FORBIDDEN_PHRASES: Final[tuple[str, ...]] = (
    "positive neighbor",
    "negative neighbor",
    "labeled neighbor",
    "fraud neighbor",
    "benign neighbor",
    "neighbor label",
    "neighbor fraud ratio",
    "neighbor positive ratio",
)

TEACHER_PROB_MASKED: Final[bool] = True
TEACHER_LOGIT_MASKED: Final[bool] = True
NEIGHBOR_LABEL_COUNTS_VISIBLE: Final[bool] = False
FORMAL_SAFE_RESULT: Final[bool] = True

FORMAL_LEAKAGE_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "leakage_policy_version",
    "neighbor_label_policy",
    "evidence_card_projection",
    "student_visible_forbidden_fields",
    "teacher_prob_masked",
    "teacher_logit_masked",
    "neighbor_label_counts_visible",
    "formal_safe_result",
    "prompt_audit_path",
    "prompt_audit_hash",
)
HEX64_PATTERN: Final[str] = r"^[0-9a-fA-F]{64}$"


def formal_leakage_provenance_fields(*, prompt_audit_path: str, prompt_audit_hash: str) -> dict[str, object]:
    """Return explicit fail-closed leakage/projection fields for formal reports."""

    return {
        "leakage_policy_version": LEAKAGE_POLICY_VERSION,
        "neighbor_label_policy": NEIGHBOR_LABEL_POLICY,
        "evidence_card_projection": EVIDENCE_CARD_PROJECTION,
        "student_visible_forbidden_fields": STUDENT_VISIBLE_FORBIDDEN_FIELD_PATHS,
        "teacher_prob_masked": TEACHER_PROB_MASKED,
        "teacher_logit_masked": TEACHER_LOGIT_MASKED,
        "neighbor_label_counts_visible": NEIGHBOR_LABEL_COUNTS_VISIBLE,
        "formal_safe_result": FORMAL_SAFE_RESULT,
        "prompt_audit_path": str(prompt_audit_path),
        "prompt_audit_hash": str(prompt_audit_hash),
    }


def validate_formal_leakage_payload(payload: object, *, context: str = "formal report") -> None:
    """Fail closed on missing or non-canonical formal leakage/projection fields."""

    import re
    from collections.abc import Mapping

    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be a mapping")
    missing = [field for field in FORMAL_LEAKAGE_REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"{context} missing required leakage fields: {', '.join(missing)}")
    expected = {
        "leakage_policy_version": LEAKAGE_POLICY_VERSION,
        "neighbor_label_policy": NEIGHBOR_LABEL_POLICY,
        "evidence_card_projection": EVIDENCE_CARD_PROJECTION,
        "student_visible_forbidden_fields": STUDENT_VISIBLE_FORBIDDEN_FIELD_PATHS,
        "teacher_prob_masked": TEACHER_PROB_MASKED,
        "teacher_logit_masked": TEACHER_LOGIT_MASKED,
        "neighbor_label_counts_visible": NEIGHBOR_LABEL_COUNTS_VISIBLE,
        "formal_safe_result": FORMAL_SAFE_RESULT,
    }
    for field, value in expected.items():
        observed = payload[field]
        if field == "student_visible_forbidden_fields":
            observed = tuple(observed)
        if observed != value:
            raise ValueError(f"{context}.{field} must be {value!r}; got {payload[field]!r}")
    prompt_audit_path = payload["prompt_audit_path"]
    if not isinstance(prompt_audit_path, str) or not prompt_audit_path:
        raise ValueError(f"{context}.prompt_audit_path must be a non-empty string")
    prompt_audit_hash = payload["prompt_audit_hash"]
    if not isinstance(prompt_audit_hash, str) or re.fullmatch(HEX64_PATTERN, prompt_audit_hash) is None:
        raise ValueError(f"{context}.prompt_audit_hash must be a 64-hex sha256")
