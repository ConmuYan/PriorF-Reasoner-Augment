"""Formal gate-manifest contract for fail-closed launcher gating."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

from evidence.leakage_policy import (
    EVIDENCE_CARD_PROJECTION,
    FORMAL_SAFE_RESULT,
    LEAKAGE_POLICY_VERSION,
    NEIGHBOR_LABEL_COUNTS_VISIBLE,
    NEIGHBOR_LABEL_POLICY,
    STUDENT_VISIBLE_FORBIDDEN_FIELD_PATHS,
    TEACHER_LOGIT_MASKED,
    TEACHER_PROB_MASKED,
)
from priorf_teacher.schema import DatasetName, GraphRegime

__all__ = ("GateManifest", "load_gate_manifest")

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_GATE_MANIFEST_SCHEMA_VERSION = "gate_manifest/v1"
HEX40_PATTERN = r"^[0-9a-fA-F]{40}$"
HEX64_PATTERN = r"^[0-9a-fA-F]{64}$"


class GateArtifactReference(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    path: StrictStr = Field(min_length=1)
    exists: bool
    sha256: str | None = Field(default=None, pattern=HEX64_PATTERN)

    @model_validator(mode="after")
    def _hash_presence_must_match_exists(self) -> "GateArtifactReference":
        if self.exists and self.sha256 is None:
            raise ValueError("artifact sha256 is required when exists is true")
        if not self.exists and self.sha256 is not None:
            raise ValueError("artifact sha256 must be omitted when exists is false")
        return self


class GateManifestProvenance(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    run_id: StrictStr = Field(min_length=1)
    data_manifest_path: StrictStr = Field(min_length=1)
    teacher_baseline_report: GateArtifactReference
    head_only_report: GateArtifactReference
    fusion_report: GateArtifactReference
    gen_only_report: GateArtifactReference
    faithfulness_report: GateArtifactReference
    prompt_audit_path: StrictStr = Field(min_length=1)
    prompt_audit_hash: str = Field(pattern=HEX64_PATTERN)
    generator_command: StrictStr = Field(min_length=1)
    generator_git_commit: StrictStr = Field(pattern=HEX40_PATTERN)
    generator_git_dirty: bool


class GateManifest(BaseModel):
    """Explicit executable gate manifest required before formal launch."""

    model_config = _STRICT_MODEL_CONFIG

    schema_version: Literal["gate_manifest/v1"] = _GATE_MANIFEST_SCHEMA_VERSION
    dataset_name: DatasetName
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
    leakage_audit_pass: bool
    leakage_policy_version: Literal["evidence_leakage_policy/v1"]
    neighbor_label_policy: Literal["removed_from_student_visible"]
    evidence_card_projection: Literal["student_safe_v1"]
    student_visible_forbidden_fields: tuple[str, ...]
    teacher_prob_masked: Literal[True]
    teacher_logit_masked: Literal[True]
    neighbor_label_counts_visible: Literal[False]
    formal_safe_result: Literal[True]
    provenance: GateManifestProvenance

    @field_validator("generated_at")
    @classmethod
    def _generated_at_must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware UTC")
        if value.utcoffset() != timedelta(0):
            raise ValueError("generated_at tzinfo must be UTC")
        return value

    @model_validator(mode="after")
    def _leakage_policy_must_match_canonical(self) -> "GateManifest":
        if self.leakage_policy_version != LEAKAGE_POLICY_VERSION:
            raise ValueError("unexpected leakage_policy_version")
        if self.neighbor_label_policy != NEIGHBOR_LABEL_POLICY:
            raise ValueError("unexpected neighbor_label_policy")
        if self.evidence_card_projection != EVIDENCE_CARD_PROJECTION:
            raise ValueError("unexpected evidence_card_projection")
        if self.student_visible_forbidden_fields != STUDENT_VISIBLE_FORBIDDEN_FIELD_PATHS:
            raise ValueError("student_visible_forbidden_fields must match leakage policy")
        if self.teacher_prob_masked is not TEACHER_PROB_MASKED:
            raise ValueError("teacher_prob_masked must be true")
        if self.teacher_logit_masked is not TEACHER_LOGIT_MASKED:
            raise ValueError("teacher_logit_masked must be true")
        if self.neighbor_label_counts_visible is not NEIGHBOR_LABEL_COUNTS_VISIBLE:
            raise ValueError("neighbor_label_counts_visible must be false")
        if self.formal_safe_result is not FORMAL_SAFE_RESULT:
            raise ValueError("formal_safe_result must be true")
        if self.provenance.generator_git_dirty:
            raise ValueError("generator_git_dirty must be false for clean gate manifests")
        return self


def load_gate_manifest(path: str | Path) -> GateManifest:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"gate manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return GateManifest.model_validate(payload)
