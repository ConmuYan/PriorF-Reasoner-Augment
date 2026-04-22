"""Fail-closed teacher export and baseline-gate contracts.

Task 2 intentionally defines only teacher-side IO contracts and gate reports.
It does not run teacher inference, student training, fusion, faithfulness,
Evidence Card construction, or diagnostic logic.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

HEX40_PATTERN = r"^[0-9a-fA-F]{40}$"
HEX64_PATTERN = r"^[0-9a-fA-F]{64}$"
TEACHER_EXPORT_SCHEMA_VERSION = "teacher_export/v1"


class GraphRegime(str, Enum):
    """Allowed graph regimes for teacher exports and reports."""

    TRANSDUCTIVE_STANDARD = "transductive_standard"
    INDUCTIVE_MASKED = "inductive_masked"


class PopulationName(str, Enum):
    """Allowed population names; ambiguous `test` is intentionally absent."""

    TRAIN = "train"
    VALIDATION = "validation"
    UNUSED_HOLDOUT = "unused_holdout"
    DIAGNOSTIC_HOLDOUT = "diagnostic_holdout"
    FINAL_TEST = "final_test"


class DatasetName(str, Enum):
    """Supported benchmark datasets."""

    AMAZON = "amazon"
    YELPCHI = "yelpchi"


class MetricName(str, Enum):
    """Teacher baseline gate metrics."""

    AUROC = "auroc"
    AUPRC = "auprc"
    F1_MACRO = "f1_macro"


def _ensure_non_empty(value: str, *, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _ensure_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError("datetime tzinfo must be UTC")
    return value


def _ensure_finite(value: float, *, field_name: str) -> float:
    if not isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite")
    return value


class TeacherProvenance(BaseModel):
    """Required provenance for teacher export artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code_git_sha: str = Field(pattern=HEX40_PATTERN)
    teacher_checkpoint_path: str = Field(min_length=1)
    teacher_checkpoint_sha256: str = Field(pattern=HEX64_PATTERN)
    data_manifest_path: str = Field(min_length=1)
    data_manifest_sha256: str = Field(pattern=HEX64_PATTERN)
    export_timestamp_utc: datetime
    random_seed: int = Field(ge=0)
    graph_regime: GraphRegime

    @field_validator("code_git_sha", "teacher_checkpoint_path", "teacher_checkpoint_sha256", "data_manifest_path", "data_manifest_sha256")
    @classmethod
    def _strings_must_not_be_blank(cls, value: str) -> str:
        return _ensure_non_empty(value, field_name="provenance string field")

    @field_validator("export_timestamp_utc")
    @classmethod
    def _timestamp_must_be_utc(cls, value: datetime) -> datetime:
        return _ensure_utc_datetime(value)


class RelationProfile(BaseModel):
    """Pinned relation-profile fields; no free-form relation dictionaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_relations: int = Field(ge=0)
    active_relations: int = Field(ge=0)
    max_relation_neighbor_count: int = Field(ge=0)
    mean_relation_neighbor_count: float = Field(ge=0.0)
    max_relation_discrepancy: float = Field(ge=0.0)
    mean_relation_discrepancy: float = Field(ge=0.0)

    @field_validator(
        "mean_relation_neighbor_count",
        "max_relation_discrepancy",
        "mean_relation_discrepancy",
    )
    @classmethod
    def _floats_must_be_finite(cls, value: float) -> float:
        return _ensure_finite(value, field_name="relation profile float")

    @model_validator(mode="after")
    def _active_relations_cannot_exceed_total(self) -> "RelationProfile":
        if self.active_relations > self.total_relations:
            raise ValueError("active_relations must be <= total_relations")
        return self


class NeighborSummary(BaseModel):
    """Pinned neighborhood-summary fields; no free-form neighbor dictionaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_neighbors: int = Field(ge=0)
    labeled_neighbors: int = Field(ge=0)
    positive_neighbors: int = Field(ge=0)
    negative_neighbors: int = Field(ge=0)
    unlabeled_neighbors: int = Field(ge=0)

    @model_validator(mode="after")
    def _neighbor_counts_must_balance(self) -> "NeighborSummary":
        if self.labeled_neighbors > self.total_neighbors:
            raise ValueError("labeled_neighbors must be <= total_neighbors")
        if self.positive_neighbors > self.labeled_neighbors:
            raise ValueError("positive_neighbors must be <= labeled_neighbors")
        if self.negative_neighbors > self.labeled_neighbors:
            raise ValueError("negative_neighbors must be <= labeled_neighbors")
        if self.unlabeled_neighbors > self.total_neighbors:
            raise ValueError("unlabeled_neighbors must be <= total_neighbors")
        if self.labeled_neighbors != self.positive_neighbors + self.negative_neighbors:
            raise ValueError("labeled_neighbors must equal positive_neighbors + negative_neighbors")
        if self.total_neighbors != self.labeled_neighbors + self.unlabeled_neighbors:
            raise ValueError("total_neighbors must equal labeled_neighbors + unlabeled_neighbors")
        return self


class TeacherExportRecord(BaseModel):
    """One row in a validated teacher export artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: DatasetName
    teacher_model_name: str = Field(min_length=1)
    teacher_checkpoint: str = Field(min_length=1)
    population_name: PopulationName
    node_id: int = Field(ge=0)
    ground_truth_label: Literal[0, 1]
    teacher_prob: float = Field(ge=0.0, le=1.0)
    teacher_logit: float
    hsd: float
    hsd_quantile: float = Field(ge=0.0, le=1.0)
    asda_switch: bool
    mlp_logit: float
    gnn_logit: float
    branch_gap: float
    relation_profile: RelationProfile
    neighbor_summary: NeighborSummary
    high_hsd_flag: bool
    graph_regime: GraphRegime

    @field_validator("teacher_model_name", "teacher_checkpoint")
    @classmethod
    def _strings_must_not_be_blank(cls, value: str) -> str:
        return _ensure_non_empty(value, field_name="teacher export string field")

    @field_validator("teacher_prob", "teacher_logit", "hsd", "hsd_quantile", "mlp_logit", "gnn_logit", "branch_gap")
    @classmethod
    def _numeric_fields_must_be_finite(cls, value: float) -> float:
        return _ensure_finite(value, field_name="teacher export numeric field")


class TeacherExportManifest(BaseModel):
    """Manifest paired with exactly one teacher export artifact population."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: DatasetName
    population_name: PopulationName
    graph_regime: GraphRegime
    row_count: int = Field(ge=1)
    node_ids_hash: str = Field(pattern=HEX64_PATTERN)
    split_values: tuple[str, ...]
    contains_tuning_rows: bool
    contains_final_test_rows: bool
    provenance: TeacherProvenance
    schema_version: str = TEACHER_EXPORT_SCHEMA_VERSION

    @field_validator("node_ids_hash", "schema_version")
    @classmethod
    def _strings_must_not_be_blank(cls, value: str) -> str:
        return _ensure_non_empty(value, field_name="teacher export manifest string field")

    @field_validator("split_values")
    @classmethod
    def _split_values_must_be_explicit(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("split_values must not be empty")
        if any(not str(item).strip() for item in value):
            raise ValueError("split_values must not contain blank values")
        return tuple(str(item) for item in value)

    @model_validator(mode="after")
    def _manifest_cross_fields_must_match(self) -> "TeacherExportManifest":
        if self.graph_regime != self.provenance.graph_regime:
            raise ValueError("manifest graph_regime must match provenance.graph_regime")
        if self.population_name == PopulationName.UNUSED_HOLDOUT and self.contains_final_test_rows:
            raise ValueError("unused_holdout must not be marked as containing final_test rows")
        if self.population_name == PopulationName.FINAL_TEST and self.contains_tuning_rows:
            raise ValueError("final_test must not contain tuning rows")
        return self


class TeacherBaselineReport(BaseModel):
    """Validated report emitted by the teacher baseline gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: DatasetName
    teacher_model_name: str = Field(min_length=1)
    teacher_checkpoint_sha256: str = Field(pattern=HEX64_PATTERN)
    graph_regime: GraphRegime
    population_name: PopulationName
    metric_name: MetricName
    metric_value: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    passed: bool
    data_manifest_sha256: str = Field(pattern=HEX64_PATTERN)
    code_git_sha: str = Field(pattern=HEX40_PATTERN)
    export_timestamp_utc: datetime

    @field_validator("teacher_model_name", "teacher_checkpoint_sha256", "data_manifest_sha256", "code_git_sha")
    @classmethod
    def _strings_must_not_be_blank(cls, value: str) -> str:
        return _ensure_non_empty(value, field_name="teacher baseline report string field")

    @field_validator("metric_value", "threshold")
    @classmethod
    def _metric_numbers_must_be_finite(cls, value: float) -> float:
        return _ensure_finite(value, field_name="teacher baseline metric field")

    @field_validator("export_timestamp_utc")
    @classmethod
    def _timestamp_must_be_utc(cls, value: datetime) -> datetime:
        return _ensure_utc_datetime(value)

    @model_validator(mode="after")
    def _baseline_report_must_be_self_consistent(self) -> "TeacherBaselineReport":
        if self.population_name != PopulationName.VALIDATION:
            raise ValueError("TeacherBaselineReport population_name must be validation")
        expected_passed = self.metric_value >= self.threshold
        if self.passed != expected_passed:
            raise ValueError("passed must equal metric_value >= threshold")
        return self
