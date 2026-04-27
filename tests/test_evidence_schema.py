from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from evidence.evidence_schema import (
    CANONICAL_SCHEMA_HINT_ORDER,
    DiscrepancySummary,
    EvidenceAblationMask,
    EvidenceCard,
    STUDENT_PROMPT_ABLATION_MASK,
    TaskInstruction,
    TeacherSummary,
    build_evidence_card,
    build_student_evidence_card,
)
from graph_data.manifests import DataArtifact, DataManifest, PopulationMetadata
from priorf_teacher.schema import DatasetName, GraphRegime, NeighborSummary, PopulationName, RelationProfile, TeacherExportRecord


def _relation_profile() -> RelationProfile:
    return RelationProfile(
        total_relations=3,
        active_relations=2,
        max_relation_neighbor_count=5,
        mean_relation_neighbor_count=1.5,
        max_relation_discrepancy=0.4,
        mean_relation_discrepancy=0.2,
    )


def _neighbor_summary() -> NeighborSummary:
    return NeighborSummary(
        total_neighbors=4,
        labeled_neighbors=3,
        positive_neighbors=1,
        negative_neighbors=2,
        unlabeled_neighbors=1,
    )


def _teacher_record(**overrides) -> TeacherExportRecord:
    payload = {
        "dataset_name": DatasetName.AMAZON,
        "teacher_model_name": "PriorF-GNN",
        "teacher_checkpoint": "checkpoints/priorf.pt",
        "population_name": PopulationName.VALIDATION,
        "node_id": 1,
        "ground_truth_label": 1,
        "teacher_prob": 0.92,
        "teacher_logit": 2.1,
        "hsd": 0.3,
        "hsd_quantile": 0.8,
        "asda_switch": True,
        "mlp_logit": 1.7,
        "gnn_logit": 2.2,
        "branch_gap": 0.5,
        "relation_profile": _relation_profile(),
        "neighbor_summary": _neighbor_summary(),
        "high_hsd_flag": True,
        "graph_regime": GraphRegime.TRANSDUCTIVE_STANDARD,
    }
    payload.update(overrides)
    return TeacherExportRecord.model_validate(payload)


def _manifest(**overrides) -> DataManifest:
    payload = {
        "dataset_name": "amazon",
        "graph_regime": "transductive_standard",
        "feature_dim": 25,
        "relation_count": 3,
        "num_nodes": 3,
        "populations": (
            PopulationMetadata(
                population_name="validation",
                split_values=("validation",),
                node_ids_hash="a" * 64,
                contains_tuning_rows=True,
                contains_final_test_rows=False,
            ),
        ),
        "artifacts": (DataArtifact(kind="source_mat", path="amazon.mat", sha256="0" * 64),),
    }
    payload.update(overrides)
    return DataManifest.model_validate(payload)


def _card(**overrides) -> EvidenceCard:
    return build_evidence_card(teacher_record=_teacher_record(), data_manifest=_manifest(), **overrides)


def test_build_evidence_card_constructs_without_label_leakage():
    card = _card()
    dumped = card.model_dump(mode="json")
    assert dumped["schema_version"] == "evidence_card/v1"
    assert dumped["dataset_name"] == "amazon"
    assert "ground_truth_label" not in dumped
    assert "label" not in dumped
    assert "ground_truth_label" not in str(dumped)


def test_missing_required_and_extra_fields_are_rejected():
    payload = _card().model_dump(mode="json")
    del payload["teacher_summary"]
    with pytest.raises(ValidationError):
        EvidenceCard.model_validate(payload)

    payload = _card().model_dump(mode="json")
    payload["extra"] = "forbidden"
    with pytest.raises(ValidationError):
        EvidenceCard.model_validate(payload)


@pytest.mark.parametrize("field,bad_value", [("graph_regime", "test"), ("population_name", "test"), ("dataset_name", "etsy")])
def test_invalid_contract_enums_are_rejected(field, bad_value):
    payload = _card().model_dump(mode="json")
    payload[field] = bad_value
    with pytest.raises(ValidationError):
        EvidenceCard.model_validate(payload)


@pytest.mark.parametrize("forbidden_key", ["label", "ground_truth_label"])
def test_label_and_ground_truth_label_are_forbidden(forbidden_key):
    payload = _card().model_dump(mode="json")
    payload[forbidden_key] = 1
    with pytest.raises(ValidationError):
        EvidenceCard.model_validate(payload)


def test_task_instruction_requires_canonical_order():
    with pytest.raises(ValidationError):
        TaskInstruction(text="x", schema_hint_order=("label", "score"))
    assert TaskInstruction(text="x", schema_hint_order=CANONICAL_SCHEMA_HINT_ORDER).schema_hint_order == CANONICAL_SCHEMA_HINT_ORDER


def test_ablation_mask_sets_nullable_values_to_none_and_requires_unmasked_values():
    mask = frozenset({EvidenceAblationMask.TEACHER_SUMMARY_TEACHER_PROB})
    card = build_evidence_card(teacher_record=_teacher_record(), data_manifest=_manifest(), ablation_mask=mask)
    assert card.teacher_summary.teacher_prob is None
    assert card.teacher_summary.teacher_logit is not None

    payload = card.model_dump(mode="json")
    payload["teacher_summary"]["teacher_prob"] = 0.92
    with pytest.raises(ValidationError, match="masked field"):
        EvidenceCard.model_validate(payload)

    payload = _card().model_dump(mode="json")
    payload["teacher_summary"]["teacher_prob"] = None
    with pytest.raises(ValidationError, match="unmasked field"):
        EvidenceCard.model_validate(payload)


def test_student_evidence_card_masks_direct_teacher_score_shortcuts():
    card = build_student_evidence_card(teacher_record=_teacher_record(), data_manifest=_manifest())

    assert card.ablation_mask == STUDENT_PROMPT_ABLATION_MASK
    assert card.evidence_card_projection == "student_safe_v1"
    assert card.teacher_summary.teacher_prob is None
    assert card.teacher_summary.teacher_logit is None
    assert card.teacher_summary.hsd_quantile is not None
    assert card.teacher_summary.mlp_logit is not None
    assert card.teacher_summary.gnn_logit is not None
    dumped = card.model_dump(mode="json")
    assert dumped["neighbor_summary"] == {"total_neighbors": 4}
    for forbidden in ("labeled_neighbors", "positive_neighbors", "negative_neighbors", "unlabeled_neighbors"):
        assert forbidden not in dumped["neighbor_summary"]


def test_relation_and_neighbor_ablation_targets_are_explicit_enum_values():
    assert EvidenceAblationMask.RELATION_PROFILE_TOTAL_RELATIONS.value == "relation_profile.total_relations"
    assert EvidenceAblationMask.NEIGHBOR_SUMMARY_TOTAL_NEIGHBORS.value == "neighbor_summary.total_neighbors"
    with pytest.raises(ValueError):
        EvidenceAblationMask("teacher_summary.unknown")


def test_build_evidence_card_rejects_manifest_mismatch_and_missing_node():
    with pytest.raises(ValueError, match="dataset_name"):
        build_evidence_card(teacher_record=_teacher_record(), data_manifest=_manifest(dataset_name="yelpchi", feature_dim=32))

    with pytest.raises(ValueError, match="graph_regime"):
        build_evidence_card(teacher_record=_teacher_record(), data_manifest=_manifest(graph_regime="inductive_masked"))

    with pytest.raises(ValueError, match="node_id"):
        build_evidence_card(teacher_record=_teacher_record(node_id=99), data_manifest=_manifest())

    explicit_manifest = SimpleNamespace(dataset_name="amazon", graph_regime="transductive_standard", num_nodes=10, node_ids=(3, 7))
    with pytest.raises(ValueError, match="node_id"):
        build_evidence_card(teacher_record=_teacher_record(node_id=1), data_manifest=explicit_manifest)


def test_summary_schemas_reject_bad_values_and_extras():
    with pytest.raises(ValidationError):
        TeacherSummary.model_validate(
            {
                "teacher_prob": 1.2,
                "teacher_logit": 2.0,
                "hsd": 0.3,
                "hsd_quantile": 0.8,
                "asda_switch": True,
                "mlp_logit": 1.0,
                "gnn_logit": 2.0,
                "branch_gap": 1.0,
                "high_hsd_flag": True,
            }
        )
    with pytest.raises(ValidationError):
        DiscrepancySummary.model_validate(
            {
                "branch_gap_abs": 0.1,
                "teacher_mlp_agreement": True,
                "teacher_gnn_agreement": False,
                "discrepancy_severity": "extreme",
                "route_hint": "balanced",
            }
        )
    with pytest.raises(ValidationError):
        DiscrepancySummary.model_validate(
            {
                "branch_gap_abs": 0.1,
                "teacher_mlp_agreement": True,
                "teacher_gnn_agreement": False,
                "discrepancy_severity": "low",
                "route_hint": "balanced",
                "freeform": {},
            }
        )
