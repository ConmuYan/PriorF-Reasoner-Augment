"""Formal gate-manifest contract for fail-closed launcher gating."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from priorf_teacher.schema import GraphRegime

__all__ = ("GateManifest", "load_gate_manifest")

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_GATE_MANIFEST_SCHEMA_VERSION = "gate_manifest/v1"
HEX40_PATTERN = r"^[0-9a-fA-F]{40}$"
HEX64_PATTERN = r"^[0-9a-fA-F]{64}$"


class GateManifest(BaseModel):
    """Explicit executable gate manifest required before formal launch."""

    model_config = _STRICT_MODEL_CONFIG

    schema_version: Literal["gate_manifest/v1"] = _GATE_MANIFEST_SCHEMA_VERSION
    graph_regime: GraphRegime
    commit: str = Field(pattern=HEX40_PATTERN)
    generated_at: datetime
    config_fingerprint: str = Field(min_length=1)
    data_manifest_hash: str = Field(pattern=HEX64_PATTERN)
    data_validation_pass: bool
    teacher_baseline_pass: bool
    subset_head_gate_pass: bool
    validation_eval_parity_pass: bool
    student_contribution_pass: bool
    strict_schema_parse_pass: bool
    smoke_pipeline_pass: bool
    teacher_prob_ablation_pass: bool
    population_contract_pass: bool

    @field_validator("generated_at")
    @classmethod
    def _generated_at_must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware UTC")
        if value.utcoffset() != timedelta(0):
            raise ValueError("generated_at tzinfo must be UTC")
        return value


def load_gate_manifest(path: str | Path) -> GateManifest:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"gate manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return GateManifest.model_validate(payload)
