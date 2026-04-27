"""Evidence Card schema and schema-preserving ablation contracts.

Task 3 scope only: typed Evidence Card construction from an already-validated
teacher record and data manifest.  This module intentionally does not read
teacher export parquet, access assets/outputs, run teacher forward passes, or
perform model inference.
"""

from __future__ import annotations

from enum import Enum
from math import isfinite
from typing import Final, Iterable, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator, model_validator

from evidence.leakage_policy import EVIDENCE_CARD_PROJECTION, INTERNAL_EVIDENCE_CARD_PROJECTION
from priorf_teacher.schema import DatasetName, GraphRegime, NeighborSummary, PopulationName, RelationProfile, TeacherExportRecord

CANONICAL_SCHEMA_HINT_ORDER: Final[tuple[str, str, str, str, str]] = (
    "rationale",
    "evidence",
    "pattern_hint",
    "label",
    "score",
)
TASK_INSTRUCTION_MAX_CHARS: Final[int] = 2_000
SCHEMA_VERSION: Final[str] = "evidence_card/v1"

_STRICT_MODEL_CONFIG: Final[ConfigDict] = ConfigDict(
    extra="forbid",
    frozen=True,
    populate_by_name=False,
    str_to_lower=False,
    str_to_upper=False,
    str_strip_whitespace=False,
)

SchemaHintField = Literal["rationale", "evidence", "pattern_hint", "label", "score"]


@runtime_checkable
class DataManifestLike(Protocol):
    dataset_name: str
    graph_regime: str
    num_nodes: int


class TaskInstruction(BaseModel):
    """Instruction text plus the required output-key order hint."""

    model_config = _STRICT_MODEL_CONFIG

    text: StrictStr = Field(min_length=1, max_length=TASK_INSTRUCTION_MAX_CHARS)
    schema_hint_order: tuple[SchemaHintField, ...]

    @model_validator(mode="after")
    def _schema_hint_order_must_be_canonical(self) -> "TaskInstruction":
        if self.schema_hint_order != CANONICAL_SCHEMA_HINT_ORDER:
            raise ValueError("schema_hint_order must equal canonical output order")
        return self


class TeacherSummary(BaseModel):
    """Typed teacher-side scalar summary.  Masked fields are represented by None."""

    model_config = _STRICT_MODEL_CONFIG

    teacher_prob: StrictFloat | None
    teacher_logit: StrictFloat | None
    hsd: StrictFloat | None
    hsd_quantile: StrictFloat | None
    asda_switch: StrictBool | None
    mlp_logit: StrictFloat | None
    gnn_logit: StrictFloat | None
    branch_gap: StrictFloat | None
    high_hsd_flag: StrictBool | None

    @field_validator("teacher_prob", "hsd_quantile")
    @classmethod
    def _unit_interval_fields(cls, value: float | None) -> float | None:
        if value is None:
            return None
        _ensure_finite(value, field_name="teacher summary probability field")
        if value < 0.0 or value > 1.0:
            raise ValueError("teacher_prob and hsd_quantile must be within [0.0, 1.0]")
        return value

    @field_validator("teacher_logit", "hsd", "mlp_logit", "gnn_logit", "branch_gap")
    @classmethod
    def _float_fields_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None:
            _ensure_finite(value, field_name="teacher summary numeric field")
        return value


class DiscrepancySummary(BaseModel):
    """Pinned discrepancy summary; deliberately no dict[str, Any] escape hatch."""

    model_config = _STRICT_MODEL_CONFIG

    branch_gap_abs: StrictFloat | None = Field(ge=0.0)
    teacher_mlp_agreement: StrictBool | None
    teacher_gnn_agreement: StrictBool | None
    discrepancy_severity: Literal["low", "medium", "high"] | None
    route_hint: Literal["mlp_dominant", "gnn_dominant", "balanced"] | None

    @field_validator("branch_gap_abs")
    @classmethod
    def _branch_gap_abs_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None:
            _ensure_finite(value, field_name="branch_gap_abs")
        return value


class StudentVisibleNeighborSummary(BaseModel):
    """Student-safe neighborhood summary with no label-derived neighbor counts."""

    model_config = _STRICT_MODEL_CONFIG

    total_neighbors: StrictInt = Field(ge=0)


class EvidenceAblationMask(str, Enum):
    """Explicit field-level ablation targets; no free-form mask strings."""

    TEACHER_SUMMARY_TEACHER_PROB = "teacher_summary.teacher_prob"
    TEACHER_SUMMARY_TEACHER_LOGIT = "teacher_summary.teacher_logit"
    TEACHER_SUMMARY_HSD = "teacher_summary.hsd"
    TEACHER_SUMMARY_HSD_QUANTILE = "teacher_summary.hsd_quantile"
    TEACHER_SUMMARY_ASDA_SWITCH = "teacher_summary.asda_switch"
    TEACHER_SUMMARY_MLP_LOGIT = "teacher_summary.mlp_logit"
    TEACHER_SUMMARY_GNN_LOGIT = "teacher_summary.gnn_logit"
    TEACHER_SUMMARY_BRANCH_GAP = "teacher_summary.branch_gap"
    TEACHER_SUMMARY_HIGH_HSD_FLAG = "teacher_summary.high_hsd_flag"
    DISCREPANCY_SUMMARY_BRANCH_GAP_ABS = "discrepancy_summary.branch_gap_abs"
    DISCREPANCY_SUMMARY_TEACHER_MLP_AGREEMENT = "discrepancy_summary.teacher_mlp_agreement"
    DISCREPANCY_SUMMARY_TEACHER_GNN_AGREEMENT = "discrepancy_summary.teacher_gnn_agreement"
    DISCREPANCY_SUMMARY_DISCREPANCY_SEVERITY = "discrepancy_summary.discrepancy_severity"
    DISCREPANCY_SUMMARY_ROUTE_HINT = "discrepancy_summary.route_hint"
    RELATION_PROFILE_TOTAL_RELATIONS = "relation_profile.total_relations"
    RELATION_PROFILE_ACTIVE_RELATIONS = "relation_profile.active_relations"
    RELATION_PROFILE_MAX_RELATION_NEIGHBOR_COUNT = "relation_profile.max_relation_neighbor_count"
    RELATION_PROFILE_MEAN_RELATION_NEIGHBOR_COUNT = "relation_profile.mean_relation_neighbor_count"
    RELATION_PROFILE_MAX_RELATION_DISCREPANCY = "relation_profile.max_relation_discrepancy"
    RELATION_PROFILE_MEAN_RELATION_DISCREPANCY = "relation_profile.mean_relation_discrepancy"
    NEIGHBOR_SUMMARY_TOTAL_NEIGHBORS = "neighbor_summary.total_neighbors"
    NEIGHBOR_SUMMARY_LABELED_NEIGHBORS = "neighbor_summary.labeled_neighbors"
    NEIGHBOR_SUMMARY_POSITIVE_NEIGHBORS = "neighbor_summary.positive_neighbors"
    NEIGHBOR_SUMMARY_NEGATIVE_NEIGHBORS = "neighbor_summary.negative_neighbors"
    NEIGHBOR_SUMMARY_UNLABELED_NEIGHBORS = "neighbor_summary.unlabeled_neighbors"


STUDENT_PROMPT_ABLATION_MASK: Final[frozenset[EvidenceAblationMask]] = frozenset(
    {
        EvidenceAblationMask.TEACHER_SUMMARY_TEACHER_PROB,
        EvidenceAblationMask.TEACHER_SUMMARY_TEACHER_LOGIT,
    }
)


class EvidenceCard(BaseModel):
    """One schema-validated, non-leaking structural evidence card."""

    model_config = _STRICT_MODEL_CONFIG

    dataset_name: DatasetName
    population_name: PopulationName
    graph_regime: GraphRegime
    node_id: StrictInt = Field(ge=0)
    teacher_summary: TeacherSummary
    discrepancy_summary: DiscrepancySummary
    relation_profile: RelationProfile
    neighbor_summary: NeighborSummary | StudentVisibleNeighborSummary
    task_instruction: TaskInstruction
    ablation_mask: frozenset[EvidenceAblationMask] = Field(default_factory=frozenset)
    schema_version: Literal["evidence_card/v1"] = SCHEMA_VERSION
    evidence_card_projection: Literal["internal_full_v1", "student_safe_v1"] = INTERNAL_EVIDENCE_CARD_PROJECTION

    @model_validator(mode="after")
    def _masked_nullable_fields_must_match(self) -> "EvidenceCard":
        for mask in self.ablation_mask:
            section, field_name = mask.value.split(".", 1)
            if section == "teacher_summary":
                value = getattr(self.teacher_summary, field_name)
                if value is not None:
                    raise ValueError(f"masked field {mask.value} must be None")
            elif section == "discrepancy_summary":
                value = getattr(self.discrepancy_summary, field_name)
                if value is not None:
                    raise ValueError(f"masked field {mask.value} must be None")

        for field_name in TeacherSummary.model_fields:
            mask = EvidenceAblationMask(f"teacher_summary.{field_name}")
            if mask not in self.ablation_mask and getattr(self.teacher_summary, field_name) is None:
                raise ValueError(f"unmasked field teacher_summary.{field_name} must be non-None")
        for field_name in DiscrepancySummary.model_fields:
            mask = EvidenceAblationMask(f"discrepancy_summary.{field_name}")
            if mask not in self.ablation_mask and getattr(self.discrepancy_summary, field_name) is None:
                raise ValueError(f"unmasked field discrepancy_summary.{field_name} must be non-None")
        return self


def build_evidence_card(
    *,
    teacher_record: TeacherExportRecord,
    data_manifest: DataManifestLike,
    ablation_mask: Iterable[EvidenceAblationMask | str] = frozenset(),
) -> EvidenceCard:
    """Build an EvidenceCard from validated Task 2 artifacts without label leakage."""

    validated_record = TeacherExportRecord.model_validate(teacher_record)
    validated_mask = _validate_ablation_mask(ablation_mask)

    if validated_record.dataset_name.value != str(data_manifest.dataset_name):
        raise ValueError("teacher_record.dataset_name must match data_manifest.dataset_name")
    if validated_record.graph_regime.value != str(data_manifest.graph_regime):
        raise ValueError("teacher_record.graph_regime must match data_manifest.graph_regime")
    if not _node_id_exists(validated_record.node_id, data_manifest):
        raise ValueError("teacher_record.node_id must exist in data_manifest.node_ids")

    teacher_summary = TeacherSummary(
        teacher_prob=_maybe_mask("teacher_summary.teacher_prob", validated_record.teacher_prob, validated_mask),
        teacher_logit=_maybe_mask("teacher_summary.teacher_logit", validated_record.teacher_logit, validated_mask),
        hsd=_maybe_mask("teacher_summary.hsd", validated_record.hsd, validated_mask),
        hsd_quantile=_maybe_mask("teacher_summary.hsd_quantile", validated_record.hsd_quantile, validated_mask),
        asda_switch=_maybe_mask("teacher_summary.asda_switch", validated_record.asda_switch, validated_mask),
        mlp_logit=_maybe_mask("teacher_summary.mlp_logit", validated_record.mlp_logit, validated_mask),
        gnn_logit=_maybe_mask("teacher_summary.gnn_logit", validated_record.gnn_logit, validated_mask),
        branch_gap=_maybe_mask("teacher_summary.branch_gap", validated_record.branch_gap, validated_mask),
        high_hsd_flag=_maybe_mask("teacher_summary.high_hsd_flag", validated_record.high_hsd_flag, validated_mask),
    )
    discrepancy_summary = DiscrepancySummary(
        branch_gap_abs=_maybe_mask("discrepancy_summary.branch_gap_abs", abs(validated_record.branch_gap), validated_mask),
        teacher_mlp_agreement=_maybe_mask(
            "discrepancy_summary.teacher_mlp_agreement",
            _same_side(validated_record.teacher_logit, validated_record.mlp_logit),
            validated_mask,
        ),
        teacher_gnn_agreement=_maybe_mask(
            "discrepancy_summary.teacher_gnn_agreement",
            _same_side(validated_record.teacher_logit, validated_record.gnn_logit),
            validated_mask,
        ),
        discrepancy_severity=_maybe_mask(
            "discrepancy_summary.discrepancy_severity",
            _severity(abs(validated_record.branch_gap)),
            validated_mask,
        ),
        route_hint=_maybe_mask(
            "discrepancy_summary.route_hint",
            _route_hint(validated_record.mlp_logit, validated_record.gnn_logit),
            validated_mask,
        ),
    )
    return EvidenceCard(
        dataset_name=validated_record.dataset_name,
        population_name=validated_record.population_name,
        graph_regime=validated_record.graph_regime,
        node_id=validated_record.node_id,
        teacher_summary=teacher_summary,
        discrepancy_summary=discrepancy_summary,
        relation_profile=validated_record.relation_profile,
        neighbor_summary=validated_record.neighbor_summary,
        task_instruction=TaskInstruction(
            text="Use the structural Evidence Card under the declared graph regime to produce the strict JSON output.",
            schema_hint_order=CANONICAL_SCHEMA_HINT_ORDER,
        ),
        ablation_mask=validated_mask,
        evidence_card_projection=INTERNAL_EVIDENCE_CARD_PROJECTION,
    )


def build_student_evidence_card(
    teacher_record: TeacherExportRecord,
    data_manifest: DataManifestLike,
) -> EvidenceCard:
    """Build the student-visible Evidence Card without direct teacher-score shortcuts."""

    card = build_evidence_card(
        teacher_record=teacher_record,
        data_manifest=data_manifest,
        ablation_mask=STUDENT_PROMPT_ABLATION_MASK,
    )
    return card.model_copy(
        update={
            "neighbor_summary": StudentVisibleNeighborSummary(
                total_neighbors=card.neighbor_summary.total_neighbors,
            ),
            "evidence_card_projection": EVIDENCE_CARD_PROJECTION,
        }
    )


def _validate_ablation_mask(mask: Iterable[EvidenceAblationMask | str]) -> frozenset[EvidenceAblationMask]:
    return frozenset(EvidenceAblationMask(item) for item in mask)


def _maybe_mask(mask_value: str, value, mask: frozenset[EvidenceAblationMask]):
    return None if EvidenceAblationMask(mask_value) in mask else value


def _node_id_exists(node_id: int, data_manifest: DataManifestLike) -> bool:
    node_ids = getattr(data_manifest, "node_ids", None)
    if node_ids is not None:
        return int(node_id) in {int(item) for item in node_ids}
    num_nodes = getattr(data_manifest, "num_nodes", None)
    return isinstance(num_nodes, int) and 0 <= int(node_id) < num_nodes


def _same_side(left: float, right: float) -> bool:
    return (left >= 0.0 and right >= 0.0) or (left < 0.0 and right < 0.0)


def _severity(branch_gap_abs: float) -> Literal["low", "medium", "high"]:
    if branch_gap_abs < 0.5:
        return "low"
    if branch_gap_abs < 1.5:
        return "medium"
    return "high"


def _route_hint(mlp_logit: float, gnn_logit: float) -> Literal["mlp_dominant", "gnn_dominant", "balanced"]:
    gap = abs(mlp_logit - gnn_logit)
    if gap < 0.25:
        return "balanced"
    return "mlp_dominant" if mlp_logit > gnn_logit else "gnn_dominant"


def _ensure_finite(value: float, *, field_name: str) -> None:
    if not isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite")
