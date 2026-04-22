"""Strict formal output schema for PriorF-Reasoner generation.

Task 3 scope only: this module defines the schema contract and deterministic
JSON serialization/parsing.  It intentionally does not implement generation,
normalization, alias parsing, retry parsing, model inference, or tokenization.
"""

from __future__ import annotations

import json
from enum import Enum
from math import isfinite
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictStr, field_validator

RATIONALE_MAX_CHARS: Final[int] = 2_000
EVIDENCE_MAX_ITEMS: Final[int] = 8
EVIDENCE_ITEM_MAX_CHARS: Final[int] = 500
PATTERN_HINT_MAX_CHARS: Final[int] = 500
CANONICAL_OUTPUT_ORDER: Final[tuple[str, str, str, str, str]] = (
    "rationale",
    "evidence",
    "pattern_hint",
    "label",
    "score",
)

_STRICT_MODEL_CONFIG: Final[ConfigDict] = ConfigDict(
    extra="forbid",
    frozen=True,
    populate_by_name=False,
    str_to_lower=False,
    str_to_upper=False,
    str_strip_whitespace=False,
)


class PredLabel(str, Enum):
    """Formal prediction label vocabulary; no aliases or case folding."""

    FRAUD = "fraud"
    BENIGN = "benign"


class StrictOutput(BaseModel):
    """Formal schema-constrained model output.

    Field order is semantically irrelevant to JSON, but canonical serialization
    is byte-stable and keeps the mandated rationale -> evidence -> pattern_hint
    -> label -> score order for generation targets.
    """

    model_config = _STRICT_MODEL_CONFIG

    rationale: StrictStr = Field(min_length=1, max_length=RATIONALE_MAX_CHARS)
    evidence: tuple[StrictStr, ...] = Field(min_length=1, max_length=EVIDENCE_MAX_ITEMS)
    pattern_hint: StrictStr = Field(min_length=1, max_length=PATTERN_HINT_MAX_CHARS)
    label: PredLabel
    score: StrictFloat = Field(ge=0.0, le=1.0)

    @field_validator("evidence")
    @classmethod
    def _evidence_items_must_be_non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            if not isinstance(item, str):
                raise TypeError("evidence items must be strings")
            if len(item) < 1:
                raise ValueError("evidence items must be non-empty")
            if len(item) > EVIDENCE_ITEM_MAX_CHARS:
                raise ValueError(f"evidence items must be <= {EVIDENCE_ITEM_MAX_CHARS} characters")
        return value

    @field_validator("score")
    @classmethod
    def _score_must_be_finite(cls, value: float) -> float:
        if not isfinite(float(value)):
            raise ValueError("score must be finite")
        return value


def canonical_serialize(strict_output: StrictOutput) -> str:
    """Serialize a StrictOutput as deterministic JSON in canonical field order."""

    output = StrictOutput.model_validate(strict_output)
    payload = {
        "rationale": output.rationale,
        "evidence": list(output.evidence),
        "pattern_hint": output.pattern_hint,
        "label": output.label.value,
        "score": output.score,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def parse_strict(text: str) -> StrictOutput:
    """Parse exactly one JSON object with no aliases, extraction, or retries."""

    payload = json.loads(text)
    return StrictOutput.model_validate(payload)
