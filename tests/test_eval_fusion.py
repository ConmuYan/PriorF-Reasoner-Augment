from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import sklearn.metrics
from pydantic import ValidationError

from eval.eval_fusion import FusionEvalConfig, FusionPopulationInputs, run_formal_fusion_eval
from evidence.prompt_builder import ThinkingMode
from evidence.leakage_policy import formal_leakage_provenance_fields
from eval.head_scoring import CheckpointProvenance, ScorerReport
from priorf_teacher.schema import DatasetName, GraphRegime, PopulationName



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


def _scorer_report(*, population: PopulationName, probs: tuple[float, ...], labels: tuple[Literal[0, 1], ...], checkpoint: CheckpointProvenance = CHECKPOINT) -> ScorerReport:
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
        distributed_gather="none",
        **_formal_report_provenance_kwargs(),
    )


def _inputs(*, population: PopulationName, head_probs: tuple[float, ...], teacher_probs: tuple[float, ...], labels: tuple[Literal[0, 1], ...], checkpoint: CheckpointProvenance = CHECKPOINT) -> FusionPopulationInputs:
    report = _scorer_report(population=population, probs=head_probs, labels=labels, checkpoint=checkpoint)
    return FusionPopulationInputs(
        head_report=report,
        teacher_probs=teacher_probs,
        teacher_node_ids=report.node_ids,
    )


def test_formal_fusion_selects_alpha_on_validation_and_freezes_for_reporting():
    validation_inputs = _inputs(
        population=PopulationName.VALIDATION,
        head_probs=(0.61, 0.32, 0.51, 0.40, 0.37, 0.58),
        teacher_probs=(0.84, 0.14, 0.17, 0.25, 0.92, 0.44),
        labels=(1, 0, 1, 0, 1, 0),
    )
    report_inputs = _inputs(
        population=PopulationName.FINAL_TEST,
        head_probs=(0.64, 0.29, 0.57, 0.33, 0.41, 0.55),
        teacher_probs=(0.79, 0.18, 0.22, 0.30, 0.88, 0.40),
        labels=(1, 0, 1, 0, 1, 0),
    )
    config = FusionEvalConfig(alpha_candidates=(0.0, 0.5, 1.0), min_student_alpha=0.25)

    report = run_formal_fusion_eval(
        validation_inputs=validation_inputs,
        report_inputs=report_inputs,
        config=config,
        run_id="run-123",
        **_score_head_audit_kwargs(),
    )

    assert report.selection.optimal_alpha == pytest.approx(0.5)
    assert report.selected_on_validation_only is True
    assert report.validation_metrics.population.population_name == PopulationName.VALIDATION
    assert report.report_metrics.population.population_name == PopulationName.FINAL_TEST
    assert report.selection.primary_metric_value > report.selection.teacher_primary_metric_value
    assert report.student_contribution_pass is True
    assert report.selection.teacher_degradation_tolerance_triggered is False
    assert report.dataset_name == DatasetName.AMAZON
    assert report.graph_regime == GraphRegime.TRANSDUCTIVE_STANDARD
    assert report.run_id == "run-123"
    assert report.eval_type == "fusion"
    assert report.validation_split == PopulationName.VALIDATION
    assert report.report_split == PopulationName.FINAL_TEST
    assert report.selected_alpha == pytest.approx(report.selection.optimal_alpha)

    old_payload = report.model_dump(mode="python")
    for field_name in (
        "leakage_policy_version",
        "neighbor_label_policy",
        "evidence_card_projection",
        "student_visible_forbidden_fields",
        "teacher_prob_masked",
        "teacher_logit_masked",
        "neighbor_label_counts_visible",
        "formal_safe_result",
        "prompt_audit_path",
        "prompt_audit_hash",
    ):
        old_payload.pop(field_name, None)
    with pytest.raises(ValidationError):
        type(report).model_validate(old_payload)


def test_student_contribution_pass_is_false_when_optimal_alpha_below_minimum():
    validation_inputs = _inputs(
        population=PopulationName.VALIDATION,
        head_probs=(0.21, 0.58, 0.82, 0.77, 0.77, 0.78),
        teacher_probs=(0.53, 0.61, 0.60, 0.46, 0.08, 0.26),
        labels=(1, 0, 1, 0, 1, 0),
    )
    report_inputs = _inputs(
        population=PopulationName.FINAL_TEST,
        head_probs=(0.24, 0.55, 0.80, 0.74, 0.74, 0.75),
        teacher_probs=(0.51, 0.60, 0.58, 0.44, 0.10, 0.28),
        labels=(1, 0, 1, 0, 1, 0),
    )
    config = FusionEvalConfig(alpha_candidates=(0.0, 0.1, 0.2), min_student_alpha=0.25)

    report = run_formal_fusion_eval(
        validation_inputs=validation_inputs,
        report_inputs=report_inputs,
        config=config,
        run_id="run-123",
        **_score_head_audit_kwargs(),
    )

    assert report.selection.optimal_alpha == pytest.approx(0.1)
    assert report.student_contribution_pass is False


def test_guardrail_fallback_sets_tolerance_triggered_when_no_candidate_passes():
    validation_inputs = _inputs(
        population=PopulationName.VALIDATION,
        head_probs=(0.25, 0.84, 0.15, 0.52, 0.82, 0.27),
        teacher_probs=(0.87, 0.22, 0.72, 0.10, 0.64, 0.30),
        labels=(1, 0, 1, 0, 1, 0),
    )
    report_inputs = _inputs(
        population=PopulationName.FINAL_TEST,
        head_probs=(0.28, 0.80, 0.19, 0.56, 0.79, 0.31),
        teacher_probs=(0.84, 0.24, 0.69, 0.12, 0.61, 0.33),
        labels=(1, 0, 1, 0, 1, 0),
    )
    config = FusionEvalConfig(alpha_candidates=(0.5, 1.0), teacher_degradation_tolerance=0.0)

    report = run_formal_fusion_eval(
        validation_inputs=validation_inputs,
        report_inputs=report_inputs,
        config=config,
        run_id="run-123",
        **_score_head_audit_kwargs(),
    )

    assert report.selection.teacher_degradation_tolerance_triggered is True
    assert report.selection.secondary_guardrail_pass is False
    assert report.student_contribution_pass is False


def test_formal_fusion_requires_checkpoint_provenance():
    valid_report = _scorer_report(
        population=PopulationName.VALIDATION,
        probs=(0.9, 0.1, 0.8, 0.2),
        labels=(1, 0, 1, 0),
    )
    payload = valid_report.model_dump(mode="python")
    payload.pop("checkpoint_provenance")

    with pytest.raises(ValidationError):
        FusionPopulationInputs(
            head_report=payload,
            teacher_probs=(0.8, 0.2, 0.7, 0.3),
            teacher_node_ids=(100, 101, 102, 103),
        )


def test_formal_fusion_rejects_validation_as_report_population():
    validation_inputs = _inputs(
        population=PopulationName.VALIDATION,
        head_probs=(0.9, 0.1, 0.8, 0.2),
        teacher_probs=(0.8, 0.2, 0.7, 0.3),
        labels=(1, 0, 1, 0),
    )
    report_inputs = _inputs(
        population=PopulationName.VALIDATION,
        head_probs=(0.7, 0.3, 0.75, 0.25),
        teacher_probs=(0.75, 0.25, 0.72, 0.28),
        labels=(1, 0, 1, 0),
    )

    with pytest.raises(ValueError, match="test-like"):
        run_formal_fusion_eval(
            validation_inputs=validation_inputs,
            report_inputs=report_inputs,
            run_id="run-123",
            **_score_head_audit_kwargs(),
        )
