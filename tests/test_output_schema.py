from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

from evidence.output_schema import PredLabel, StrictOutput, canonical_serialize, parse_strict


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


@pytest.mark.parametrize("bad_label", ["spam", "Fraud", "FRAUD", "benign "])
def test_label_must_be_exact_enum(bad_label):
    payload = _strict_output().model_dump(mode="json")
    payload["label"] = bad_label
    with pytest.raises(ValidationError):
        StrictOutput.model_validate(payload)


@pytest.mark.parametrize("bad_score", [-0.1, 1.1, math.nan, math.inf, -math.inf, "0.5"])
def test_score_bounds_finite_and_numeric(bad_score):
    payload = _strict_output().model_dump(mode="json")
    payload["score"] = bad_score
    with pytest.raises(ValidationError):
        StrictOutput.model_validate(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rationale", ""),
        ("pattern_hint", ""),
        ("evidence", ()),
        ("evidence", ("",)),
    ],
)
def test_required_text_fields_are_non_empty(field, value):
    payload = _strict_output().model_dump(mode="json")
    payload[field] = value
    with pytest.raises(ValidationError):
        StrictOutput.model_validate(payload)


def test_canonical_serialize_is_byte_exact_and_round_trips():
    output = _strict_output()
    serialized = canonical_serialize(output)
    assert serialized == (
        '{"rationale":"Reason from structural evidence first.",'
        '"evidence":["Teacher confidence is high.","Neighborhood pattern is consistent."],'
        '"pattern_hint":"high structural agreement",'
        '"label":"fraud",'
        '"score":0.93}'
    )
    assert list(json.loads(serialized).keys()) == ["rationale", "evidence", "pattern_hint", "label", "score"]
    assert parse_strict(serialized) == output


def test_parse_strict_accepts_only_exact_schema_json():
    assert parse_strict(canonical_serialize(_strict_output(label=PredLabel.BENIGN, score=0.11))).label == PredLabel.BENIGN

    extra_payload = json.loads(canonical_serialize(_strict_output()))
    extra_payload["extra"] = "forbidden"
    with pytest.raises(ValidationError):
        parse_strict(json.dumps(extra_payload))

    alias_payload = json.loads(canonical_serialize(_strict_output()))
    alias_payload["pred_label"] = alias_payload.pop("label")
    with pytest.raises(ValidationError):
        parse_strict(json.dumps(alias_payload))

    case_payload = json.loads(canonical_serialize(_strict_output()))
    case_payload["Rationale"] = case_payload.pop("rationale")
    with pytest.raises(ValidationError):
        parse_strict(json.dumps(case_payload))

    with pytest.raises(json.JSONDecodeError):
        parse_strict("not-json")


def test_no_forgiving_parser_exports_exist():
    import evidence.output_schema as output_schema

    assert not hasattr(output_schema, "parse_normalized")
    assert not hasattr(output_schema, "parse_forgiving")
    assert not hasattr(output_schema, "parse_with_alias")
