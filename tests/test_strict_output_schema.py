from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from evidence.output_schema import PredLabel, StrictOutput, canonical_serialize, parse_strict
from llm.parsing import parse_normalized_output


def _strict_output(**overrides) -> StrictOutput:
    payload = {
        "rationale": "Reason from structural evidence first.",
        "evidence": ("Teacher confidence is high.", "Neighborhood pattern is consistent."),
        "pattern_hint": "high structural agreement",
        "label": PredLabel.FRAUD,
        "score": 0.93,
    }
    payload.update(overrides)
    return StrictOutput.model_validate(payload)


def test_strict_parser_still_rejects_alias_payloads_used_by_normalized_parser():
    alias_payload = {
        "Rationale": "diagnostic only",
        "Evidence": "teacher evidence",
        "patternHint": "aliased hint",
        "pred_label": "BENIGN",
        "confidence": 0.75,
    }
    with pytest.raises(ValidationError):
        parse_strict(json.dumps(alias_payload))


def test_normalized_parser_accepts_wrapped_alias_payload_as_diagnostic_only():
    normalized = parse_normalized_output(
        '```json\n'
        '{"Rationale":"diagnostic only","Evidence":"teacher evidence",'
        '"patternHint":"aliased hint","pred_label":"BENIGN","confidence":0.75}'
        '\n```'
    )

    assert normalized == _strict_output(
        rationale="diagnostic only",
        evidence=("teacher evidence",),
        pattern_hint="aliased hint",
        label=PredLabel.BENIGN,
        score=0.75,
    )


def test_strict_round_trip_contract_is_unchanged():
    output = _strict_output(label=PredLabel.BENIGN, score=0.11)
    assert parse_strict(canonical_serialize(output)) == output
