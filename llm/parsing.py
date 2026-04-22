"""Generation-output parsing helpers.

This module keeps the formal strict parser in ``evidence.output_schema`` as the
only headline path, while providing a separate normalized parser for diagnostic
use in formal generation evaluation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Final

from evidence.output_schema import PredLabel, StrictOutput

__all__ = ("NormalizedParseError", "extract_json_object", "parse_normalized_output")

_JSON_FENCE_RE: Final[re.Pattern[str]] = re.compile(
    r"```(?:json)?\s*(?P<body>.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)
_FIELD_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "rationale": ("rationale", "reasoning", "analysis"),
    "evidence": ("evidence", "evidences", "supporting_evidence"),
    "pattern_hint": ("pattern_hint", "patternhint", "pattern_hints", "pattern", "hint"),
    "label": ("label", "pred_label", "prediction", "predicted_label"),
    "score": ("score", "confidence", "prob", "probability"),
}
_LABEL_ALIASES: Final[dict[str, PredLabel]] = {
    "fraud": PredLabel.FRAUD,
    "positive": PredLabel.FRAUD,
    "1": PredLabel.FRAUD,
    "benign": PredLabel.BENIGN,
    "negative": PredLabel.BENIGN,
    "0": PredLabel.BENIGN,
}


class NormalizedParseError(ValueError):
    """Raised when the diagnostic normalization path cannot recover a payload."""


def extract_json_object(text: str) -> str:
    """Extract the first JSON object from raw model text.

    The diagnostic path accepts fenced JSON or a larger wrapper string that
    contains a single JSON object. It does not attempt multi-object recovery.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    stripped = text.strip()
    if not stripped:
        raise NormalizedParseError("generation output is empty")

    fenced_match = _JSON_FENCE_RE.fullmatch(stripped)
    if fenced_match is not None:
        stripped = fenced_match.group("body").strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        candidate = stripped[index : index + end].strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate

    raise NormalizedParseError("could not extract a JSON object from generation output")


def parse_normalized_output(text: str) -> StrictOutput:
    """Parse a diagnostic normalized output and coerce it into ``StrictOutput``.

    This function is intentionally separate from the formal strict parser.
    It may recover fenced JSON, common field aliases, single-string evidence,
    and case/whitespace label variants, but its output is for diagnostic use
    only and must not be used as the headline parse-success metric.
    """

    payload = json.loads(extract_json_object(text))
    if not isinstance(payload, Mapping):
        raise NormalizedParseError("normalized parser requires a JSON object payload")

    normalized_payload = {
        "rationale": _normalize_text_field(_pick_required(payload, "rationale"), field_name="rationale"),
        "evidence": _normalize_evidence(_pick_required(payload, "evidence")),
        "pattern_hint": _normalize_text_field(_pick_required(payload, "pattern_hint"), field_name="pattern_hint"),
        "label": _normalize_label(_pick_required(payload, "label")),
        "score": _normalize_score(_pick_required(payload, "score")),
    }
    return StrictOutput.model_validate(normalized_payload)


def _pick_required(payload: Mapping[str, object], canonical_field: str) -> object:
    for alias in _FIELD_ALIASES[canonical_field]:
        for key, value in payload.items():
            if _normalize_key(key) == alias:
                return value
    raise NormalizedParseError(f"missing required field: {canonical_field}")


def _normalize_key(key: object) -> str:
    if not isinstance(key, str):
        raise NormalizedParseError("JSON object keys must be strings")
    return key.strip().lower().replace("-", "_").replace(" ", "")


def _normalize_text_field(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise NormalizedParseError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise NormalizedParseError(f"{field_name} must be non-empty")
    return normalized


def _normalize_evidence(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise NormalizedParseError("evidence must be non-empty")
        return (normalized,)
    if isinstance(value, (list, tuple)):
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise NormalizedParseError("evidence items must be strings")
            normalized = item.strip()
            if not normalized:
                raise NormalizedParseError("evidence items must be non-empty")
            items.append(normalized)
        if not items:
            raise NormalizedParseError("evidence must contain at least one item")
        return tuple(items)
    raise NormalizedParseError("evidence must be a string or a list of strings")


def _normalize_label(value: object) -> PredLabel:
    if isinstance(value, PredLabel):
        return value
    if not isinstance(value, str):
        raise NormalizedParseError("label must be a string")
    normalized = value.strip().lower()
    try:
        return _LABEL_ALIASES[normalized]
    except KeyError as exc:
        raise NormalizedParseError(f"unsupported normalized label: {value!r}") from exc


def _normalize_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise NormalizedParseError("score must be numeric") from exc
    return score
