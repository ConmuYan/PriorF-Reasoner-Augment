from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from priorf_teacher.schema import GraphRegime
from schemas.gate_manifest import GateManifest, load_gate_manifest
from scripts.gate_check import gate_check, main


def _manifest_payload(**overrides):
    payload = {
        "schema_version": "gate_manifest/v1",
        "graph_regime": GraphRegime.TRANSDUCTIVE_STANDARD.value,
        "commit": "a" * 40,
        "generated_at": datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc).isoformat(),
        "config_fingerprint": "cfg-123",
        "data_manifest_hash": "b" * 64,
        "data_validation_pass": True,
        "teacher_baseline_pass": True,
        "subset_head_gate_pass": True,
        "validation_eval_parity_pass": True,
        "student_contribution_pass": True,
        "strict_schema_parse_pass": True,
        "smoke_pipeline_pass": True,
        "teacher_prob_ablation_pass": True,
        "population_contract_pass": True,
    }
    payload.update(overrides)
    return payload


def _write_manifest(tmp_path: Path, **overrides) -> Path:
    path = tmp_path / "gate_manifest.json"
    path.write_text(json.dumps(_manifest_payload(**overrides)), encoding="utf-8")
    return path


def test_gate_manifest_round_trips_and_gate_check_passes(tmp_path: Path):
    path = _write_manifest(tmp_path)

    manifest = load_gate_manifest(path)

    assert manifest.schema_version == "gate_manifest/v1"
    assert manifest.graph_regime == GraphRegime.TRANSDUCTIVE_STANDARD
    assert gate_check(manifest_path=path) == manifest
    assert main(["--manifest-path", str(path)]) == 0


@pytest.mark.parametrize(
    "field_name",
    [
        "data_validation_pass",
        "teacher_baseline_pass",
        "subset_head_gate_pass",
        "validation_eval_parity_pass",
        "student_contribution_pass",
        "strict_schema_parse_pass",
        "smoke_pipeline_pass",
        "teacher_prob_ablation_pass",
        "population_contract_pass",
    ],
)
def test_gate_check_fails_closed_when_any_required_gate_is_false(tmp_path: Path, field_name: str):
    path = _write_manifest(tmp_path, **{field_name: False})

    with pytest.raises(ValueError, match=field_name):
        gate_check(manifest_path=path)
    assert main(["--manifest-path", str(path)]) == 1


def test_gate_manifest_requires_all_required_fields(tmp_path: Path):
    payload = _manifest_payload()
    del payload["config_fingerprint"]
    path = tmp_path / "gate_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_gate_manifest(path)
    assert main(["--manifest-path", str(path)]) == 1


@pytest.mark.parametrize(
    "timestamp",
    [
        datetime(2026, 4, 22, 12, 0),
        datetime(2026, 4, 22, 12, 0, tzinfo=timezone(timedelta(hours=8))).isoformat(),
    ],
)
def test_gate_manifest_generated_at_must_be_utc(timestamp):
    payload = _manifest_payload(generated_at=timestamp)

    with pytest.raises(ValidationError):
        GateManifest.model_validate(payload)
