"""Fail-closed tests for the teacher-probability ablation audit (Task 11)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import sklearn.metrics
from pydantic import ValidationError

from eval.head_scoring import CheckpointProvenance, ScorerReport
from evidence.ablations import (
    TEACHER_PROB_MASK,
    TeacherProbAblationAudit,
    ablate_teacher_prob,
    run_teacher_prob_ablation_audit,
)
from evidence.evidence_schema import EvidenceAblationMask, build_evidence_card
from evidence.prompt_builder import ThinkingMode
from evidence.leakage_policy import formal_leakage_provenance_fields
from graph_data.manifests import DataArtifact, DataManifest, PopulationMetadata
from priorf_teacher.schema import (
    DatasetName,
    GraphRegime,
    NeighborSummary,
    PopulationName,
    RelationProfile,
    TeacherExportRecord,
)



def _score_head_audit_kwargs() -> dict[str, str]:
    return {
        "prompt_audit_path": "outputs/tests/prompt_audit.json",
        "prompt_audit_hash": "a" * 64,
    }


def _formal_report_provenance_kwargs() -> dict:
    return formal_leakage_provenance_fields(**_score_head_audit_kwargs())

CHECKPOINT = CheckpointProvenance(
    path="outputs/gated/ckpt-step-8.safetensors",
    step=8,
    content_hash="a" * 64,
)


def _teacher_record() -> TeacherExportRecord:
    return TeacherExportRecord(
        dataset_name=DatasetName.AMAZON,
        teacher_model_name="LGHGCLNetV2",
        teacher_checkpoint="assets/teacher/amazon/best_model.pt",
        population_name=PopulationName.VALIDATION,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        node_id=42,
        ground_truth_label=1,
        teacher_prob=0.82,
        teacher_logit=1.5,
        hsd=0.33,
        hsd_quantile=0.7,
        asda_switch=True,
        mlp_logit=1.1,
        gnn_logit=1.9,
        branch_gap=-0.8,
        high_hsd_flag=True,
        relation_profile=RelationProfile(
            total_relations=3,
            active_relations=2,
            max_relation_neighbor_count=17,
            mean_relation_neighbor_count=10.0,
            max_relation_discrepancy=0.4,
            mean_relation_discrepancy=0.2,
        ),
        neighbor_summary=NeighborSummary(
            total_neighbors=20,
            labeled_neighbors=12,
            positive_neighbors=3,
            negative_neighbors=9,
            unlabeled_neighbors=8,
        ),
    )


def _data_manifest() -> DataManifest:
    return DataManifest(
        dataset_name="amazon",
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        feature_dim=25,
        relation_count=3,
        num_nodes=64,
        populations=(
            PopulationMetadata(
                population_name="validation",
                split_values=("validation",),
                node_ids_hash="0" * 64,
                contains_tuning_rows=True,
                contains_final_test_rows=False,
            ),
        ),
        artifacts=(
            DataArtifact(
                kind="source_mat",
                path="/tmp/amazon.mat",
                sha256="a" * 64,
            ),
        ),
    )


def _scorer_report(
    *,
    population: PopulationName = PopulationName.VALIDATION,
    probs: tuple[float, ...],
    labels: tuple[Literal[0, 1], ...],
    checkpoint: CheckpointProvenance = CHECKPOINT,
    distributed_gather: Literal["none", "accelerate_gather_for_metrics"] = "none",
) -> ScorerReport:
    n_total = len(labels)
    n_positive = sum(labels)
    n_negative = n_total - n_positive
    is_single = n_positive == 0 or n_negative == 0
    auroc = None if is_single else float(sklearn.metrics.roc_auc_score(labels, probs))
    auprc = None if is_single else float(sklearn.metrics.average_precision_score(labels, probs))
    return ScorerReport(
        dataset_name=DatasetName.AMAZON,
        population_name=population,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        run_id="run-123",
        report_split=population,
        eval_type="head_scoring",
        checkpoint_provenance=checkpoint,
        scorer_schema_version="head_scorer/v1",
        n_total=n_total,
        n_positive=n_positive,
        n_negative=n_negative,
        is_single_class_population=is_single,
        auroc=auroc,
        auprc=auprc,
        brier_score=float(sklearn.metrics.brier_score_loss(labels, probs)),
        prob_mean=sum(probs) / n_total,
        prob_std=float(np.std(probs)),
        prob_min=min(probs),
        prob_max=max(probs),
        prob_q25=float(np.quantile(probs, 0.25)),
        prob_q50=float(np.quantile(probs, 0.50)),
        prob_q75=float(np.quantile(probs, 0.75)),
        probs=probs,
        labels=labels,
        node_ids=tuple(range(100, 100 + n_total)),
        prompt_mode="eval_head",
        thinking_mode=ThinkingMode.NON_THINKING,
        pooling_path="pool_last_valid_token",
        uses_inference_mode=True,
        distributed_gather=distributed_gather,
        **_formal_report_provenance_kwargs(),
    )


# ---------------------------------------------------------------- ablate_teacher_prob


def test_ablate_teacher_prob_masks_only_prob_and_preserves_rest() -> None:
    manifest = _data_manifest()
    full_card = build_evidence_card(teacher_record=_teacher_record(), data_manifest=manifest)

    ablated = ablate_teacher_prob(full_card)

    assert ablated.teacher_summary.teacher_prob is None
    assert TEACHER_PROB_MASK in ablated.ablation_mask

    # Only the masked field differs; all other teacher_summary fields stay identical.
    full_fields = full_card.teacher_summary.model_dump()
    ablated_fields = ablated.teacher_summary.model_dump()
    full_fields.pop("teacher_prob")
    ablated_fields.pop("teacher_prob")
    assert full_fields == ablated_fields

    # Discrepancy summary, relation profile, neighbor summary, node_id are unaffected.
    assert ablated.discrepancy_summary == full_card.discrepancy_summary
    assert ablated.relation_profile == full_card.relation_profile
    assert ablated.neighbor_summary == full_card.neighbor_summary
    assert ablated.node_id == full_card.node_id


def test_ablate_teacher_prob_rejects_already_masked_card() -> None:
    manifest = _data_manifest()
    already_ablated = build_evidence_card(
        teacher_record=_teacher_record(),
        data_manifest=manifest,
        ablation_mask={EvidenceAblationMask.TEACHER_SUMMARY_TEACHER_PROB},
    )
    with pytest.raises(ValueError, match="already has TEACHER_SUMMARY_TEACHER_PROB"):
        ablate_teacher_prob(already_ablated)


# ---------------------------------------------------------------- audit happy-path


def test_audit_flags_dependency_high_when_auroc_drop_exceeds_threshold() -> None:
    full = _scorer_report(
        probs=(0.92, 0.10, 0.87, 0.08, 0.95, 0.15),
        labels=(1, 0, 1, 0, 1, 0),
    )
    # Ablated probs deliberately invert one positive-vs-negative pair so
    # AUROC drops from 1.0 to ~0.667 (3 of 9 pairs discordant).
    ablated = _scorer_report(
        probs=(0.35, 0.50, 0.55, 0.40, 0.62, 0.45),
        labels=(1, 0, 1, 0, 1, 0),
    )

    audit = run_teacher_prob_ablation_audit(
        full_report=full,
        ablated_report=ablated,
        dependency_threshold=0.10,
    )

    assert isinstance(audit, TeacherProbAblationAudit)
    assert audit.ablation_target == "teacher_summary.teacher_prob"
    assert audit.n_total == 6
    assert audit.n_positive == 3
    assert audit.n_negative == 3
    assert audit.auroc_full == pytest.approx(1.0)
    assert audit.auroc_ablated is not None
    assert audit.auroc_delta is not None and audit.auroc_delta > 0.0
    assert audit.teacher_prob_dependency_high is True
    # Path audit strings echo faithfully from the source report.
    assert audit.prompt_mode == "eval_head"
    assert audit.pooling_path == "pool_last_valid_token"
    assert audit.distributed_gather == "none"


def test_audit_does_not_flag_when_auroc_drop_below_threshold() -> None:
    full = _scorer_report(
        probs=(0.80, 0.20, 0.75, 0.25, 0.70, 0.30),
        labels=(1, 0, 1, 0, 1, 0),
    )
    ablated = _scorer_report(
        probs=(0.78, 0.22, 0.73, 0.27, 0.68, 0.32),
        labels=(1, 0, 1, 0, 1, 0),
    )
    audit = run_teacher_prob_ablation_audit(
        full_report=full,
        ablated_report=ablated,
        dependency_threshold=0.10,
    )
    assert audit.teacher_prob_dependency_high is False
    assert audit.auroc_delta is not None
    assert audit.auroc_delta < 0.10


# ---------------------------------------------------------------- audit fail-closed guards


def test_audit_rejects_checkpoint_provenance_mismatch() -> None:
    other_ckpt = CheckpointProvenance(
        path="outputs/gated/ckpt-step-9.safetensors",
        step=9,
        content_hash="b" * 64,
    )
    full = _scorer_report(probs=(0.9, 0.1), labels=(1, 0))
    ablated = _scorer_report(probs=(0.6, 0.4), labels=(1, 0), checkpoint=other_ckpt)
    with pytest.raises(ValueError, match="checkpoint_provenance must match"):
        run_teacher_prob_ablation_audit(
            full_report=full, ablated_report=ablated, dependency_threshold=0.05
        )


def test_audit_rejects_population_mismatch() -> None:
    full = _scorer_report(probs=(0.9, 0.1), labels=(1, 0))
    ablated = _scorer_report(
        population=PopulationName.FINAL_TEST,
        probs=(0.6, 0.4),
        labels=(1, 0),
    )
    with pytest.raises(ValueError, match="population_name mismatch"):
        run_teacher_prob_ablation_audit(
            full_report=full, ablated_report=ablated, dependency_threshold=0.05
        )


def test_audit_rejects_path_audit_drift() -> None:
    full = _scorer_report(probs=(0.9, 0.1), labels=(1, 0))
    ablated = _scorer_report(
        probs=(0.6, 0.4),
        labels=(1, 0),
        distributed_gather="accelerate_gather_for_metrics",
    )
    with pytest.raises(ValueError, match="distributed_gather"):
        run_teacher_prob_ablation_audit(
            full_report=full, ablated_report=ablated, dependency_threshold=0.05
        )


def test_audit_rejects_invalid_threshold() -> None:
    full = _scorer_report(probs=(0.9, 0.1), labels=(1, 0))
    ablated = _scorer_report(probs=(0.6, 0.4), labels=(1, 0))
    with pytest.raises(ValueError, match=r"dependency_threshold must be in \(0, 1\)"):
        run_teacher_prob_ablation_audit(
            full_report=full, ablated_report=ablated, dependency_threshold=0.0
        )
    with pytest.raises(ValueError, match=r"dependency_threshold must be in \(0, 1\)"):
        run_teacher_prob_ablation_audit(
            full_report=full, ablated_report=ablated, dependency_threshold=1.0
        )


def test_audit_handles_single_class_population_without_fabricating_metric() -> None:
    full = _scorer_report(probs=(0.9, 0.85, 0.7), labels=(1, 1, 1))
    ablated = _scorer_report(probs=(0.6, 0.55, 0.4), labels=(1, 1, 1))
    audit = run_teacher_prob_ablation_audit(
        full_report=full, ablated_report=ablated, dependency_threshold=0.05
    )
    assert audit.auroc_full is None
    assert audit.auroc_ablated is None
    assert audit.auroc_delta is None
    assert audit.teacher_prob_dependency_high is False


def test_audit_report_is_frozen_and_extra_forbid() -> None:
    full = _scorer_report(probs=(0.9, 0.1), labels=(1, 0))
    ablated = _scorer_report(probs=(0.6, 0.4), labels=(1, 0))
    audit = run_teacher_prob_ablation_audit(
        full_report=full, ablated_report=ablated, dependency_threshold=0.05
    )
    # Frozen.
    with pytest.raises(ValidationError):
        audit.__dict__["teacher_prob_dependency_high"] = False  # type: ignore[index]
        TeacherProbAblationAudit.model_validate(
            {**audit.model_dump(), "unrelated_field": 0}
        )
