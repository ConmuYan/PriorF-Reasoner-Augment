"""Task 10 — formal head-only evaluation contract tests."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pytest
import sklearn.metrics
from pydantic import ValidationError

import eval.eval_head_only as eval_head_only
from eval.calibration import compute_calibration_summary, compute_threshold_metrics, select_validation_threshold
from eval.head_scoring import CheckpointProvenance, HeadScoringInputs, HeadScoringSample, ScorerReport
from eval.temperature_scaling import apply_temperature_to_probs, fit_temperature_on_validation, logits_from_probs
from evidence.evidence_schema import DataManifestLike, build_evidence_card
from evidence.leakage_policy import formal_leakage_provenance_fields
from evidence.prompt_builder import ThinkingMode
from graph_data.manifests import DataArtifact, DataManifest, PopulationMetadata
from priorf_teacher.schema import (
    DatasetName,
    GraphRegime,
    NeighborSummary,
    PopulationName,
    RelationProfile,
    TeacherExportRecord,
)


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


def _teacher_record(population_name: PopulationName, node_id: int, ground_truth_label: Literal[0, 1]) -> TeacherExportRecord:
    return TeacherExportRecord(
        dataset_name=DatasetName.AMAZON,
        teacher_model_name="LGHGCLNetV2",
        teacher_checkpoint="outputs/gated/teacher/best_model.pt",
        population_name=population_name,
        node_id=node_id,
        ground_truth_label=ground_truth_label,
        teacher_prob=0.2 + 0.1 * ground_truth_label,
        teacher_logit=-0.5 + node_id,
        hsd=0.1 * (node_id + 1),
        hsd_quantile=0.25,
        asda_switch=(node_id % 2 == 0),
        mlp_logit=-0.25 + node_id,
        gnn_logit=-0.5 + node_id,
        branch_gap=0.25,
        relation_profile=_relation_profile(),
        neighbor_summary=_neighbor_summary(),
        high_hsd_flag=False,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
    )


def _data_manifest() -> DataManifest:
    return DataManifest(
        dataset_name=DatasetName.AMAZON.value,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD.value,
        feature_dim=25,
        relation_count=3,
        num_nodes=256,
        populations=(
            PopulationMetadata(
                population_name=PopulationName.VALIDATION.value,
                split_values=(PopulationName.VALIDATION.value,),
                node_ids_hash="a" * 64,
                contains_tuning_rows=True,
                contains_final_test_rows=False,
            ),
            PopulationMetadata(
                population_name=PopulationName.FINAL_TEST.value,
                split_values=(PopulationName.FINAL_TEST.value,),
                node_ids_hash="b" * 64,
                contains_tuning_rows=False,
                contains_final_test_rows=True,
            ),
        ),
        artifacts=(DataArtifact(kind="source_mat", path="assets/data/Amazon_canonical.mat", sha256="c" * 64),),
    )


def _score_head_audit_kwargs() -> dict[str, str]:
    return {
        "prompt_audit_path": "outputs/tests/prompt_audit.json",
        "prompt_audit_hash": "a" * 64,
    }


def _formal_report_provenance_kwargs() -> dict:
    return formal_leakage_provenance_fields(**_score_head_audit_kwargs())


def _checkpoint_bundle() -> eval_head_only.FormalHeadOnlyCheckpointBundle:
    base = dict(
        run_id="run-123",
        checkpoint_step=12,
        commit="deadbeef",
        config_fingerprint="cfg-123",
        data_manifest_hash="manifest-123",
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
    )
    return eval_head_only.FormalHeadOnlyCheckpointBundle(
        llm_backbone=eval_head_only.FormalHeadOnlyCheckpointComponent(
            path="outputs/gated/checkpoints/step_12/backbone.safetensors",
            content_hash="a" * 64,
            **base,
        ),
        peft_adapter=eval_head_only.FormalHeadOnlyCheckpointComponent(
            path="outputs/gated/checkpoints/step_12/adapter.safetensors",
            content_hash="b" * 64,
            **base,
        ),
        cls_head=eval_head_only.FormalHeadOnlyCheckpointComponent(
            path="outputs/gated/checkpoints/step_12/cls_head.safetensors",
            content_hash="d" * 64,
            **base,
        ),
    )


def _population_metadata(name: PopulationName) -> PopulationMetadata:
    if name == PopulationName.VALIDATION:
        return PopulationMetadata(
            population_name=name.value,
            split_values=(name.value,),
            node_ids_hash="e" * 64,
            contains_tuning_rows=True,
            contains_final_test_rows=False,
        )
    return PopulationMetadata(
        population_name=name.value,
        split_values=(name.value,),
        node_ids_hash="f" * 64,
        contains_tuning_rows=False,
        contains_final_test_rows=(name == PopulationName.FINAL_TEST),
    )


def _inputs(population_name: PopulationName, node_ids: tuple[int, ...], labels: tuple[Literal[0, 1], ...]) -> HeadScoringInputs:
    manifest = _data_manifest()
    bundle = _checkpoint_bundle()
    samples = []
    for node_id, label in zip(node_ids, labels, strict=True):
        card = build_evidence_card(
            teacher_record=_teacher_record(population_name, node_id, label),
            data_manifest=cast(DataManifestLike, manifest),
        )
        samples.append(HeadScoringSample(evidence_card=card, ground_truth_label=label, node_id=node_id))
    return HeadScoringInputs(
        samples=tuple(samples),
        dataset_name=DatasetName.AMAZON,
        population_name=population_name,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        checkpoint_provenance=bundle.cls_head.to_shared_checkpoint_provenance(),
    )


def _scorer_report(
    *,
    population_name: PopulationName,
    probs: tuple[float, ...],
    labels: tuple[Literal[0, 1], ...],
    node_ids: tuple[int, ...],
) -> ScorerReport:
    probs_np = np.asarray(probs, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int64)
    is_single_class = bool(np.all(labels_np == 0) or np.all(labels_np == 1))
    return ScorerReport(
        dataset_name=DatasetName.AMAZON,
        population_name=population_name,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        run_id="run-123",
        report_split=population_name,
        eval_type="head_scoring",
        checkpoint_provenance=_checkpoint_bundle().cls_head.to_shared_checkpoint_provenance(),
        scorer_schema_version="head_scorer/v1",
        n_total=len(probs),
        n_positive=int((labels_np == 1).sum()),
        n_negative=int((labels_np == 0).sum()),
        is_single_class_population=is_single_class,
        auroc=None if is_single_class else float(sklearn.metrics.roc_auc_score(labels_np, probs_np)),
        auprc=None if is_single_class else float(sklearn.metrics.average_precision_score(labels_np, probs_np)),
        brier_score=float(np.mean((probs_np - labels_np.astype(np.float64)) ** 2)),
        prob_mean=float(np.mean(probs_np)),
        prob_std=float(np.std(probs_np)),
        prob_min=float(np.min(probs_np)),
        prob_max=float(np.max(probs_np)),
        prob_q25=float(np.quantile(probs_np, 0.25, method="linear")),
        prob_q50=float(np.quantile(probs_np, 0.50, method="linear")),
        prob_q75=float(np.quantile(probs_np, 0.75, method="linear")),
        probs=probs,
        labels=labels,
        node_ids=node_ids,
        prompt_mode="eval_head",
        thinking_mode=ThinkingMode.NON_THINKING,
        pooling_path="pool_last_valid_token",
        uses_inference_mode=True,
        distributed_gather="none",
        **_formal_report_provenance_kwargs(),
    )


def test_formal_head_only_eval_uses_shared_scorer_and_frozen_validation_threshold(monkeypatch: pytest.MonkeyPatch):
    validation_inputs = _inputs(PopulationName.VALIDATION, (3, 4, 8, 9), (0, 0, 1, 1))
    report_inputs = _inputs(PopulationName.FINAL_TEST, (21, 22, 23, 24), (0, 1, 1, 0))
    validation_report = _scorer_report(
        population_name=PopulationName.VALIDATION,
        probs=(0.10, 0.40, 0.60, 0.90),
        labels=(0, 0, 1, 1),
        node_ids=(3, 4, 8, 9),
    )
    report_population = _scorer_report(
        population_name=PopulationName.FINAL_TEST,
        probs=(0.55, 0.58, 0.59, 0.95),
        labels=(0, 1, 1, 0),
        node_ids=(21, 22, 23, 24),
    )

    calls: list[HeadScoringInputs] = []

    def fake_score_head(**kwargs):
        calls.append(kwargs["inputs"])
        if kwargs["inputs"] == validation_inputs:
            return validation_report
        if kwargs["inputs"] == report_inputs:
            return report_population
        raise AssertionError("unexpected HeadScoringInputs")

    monkeypatch.setattr(eval_head_only, "score_head", fake_score_head)

    report = eval_head_only.run_formal_head_only_eval(
        validation_inputs=validation_inputs,
        report_inputs=report_inputs,
        validation_population_metadata=_population_metadata(PopulationName.VALIDATION),
        report_population_metadata=_population_metadata(PopulationName.FINAL_TEST),
        model=object(),
        cls_head=object(),
        tokenizer=object(),
        thinking_mode=ThinkingMode.NON_THINKING,
        checkpoint_source="best_checkpoint",
        checkpoint_bundle=_checkpoint_bundle(),
        run_id="run-123",
        **_score_head_audit_kwargs(),
        threshold_selection_metric="f1",
        include_oracle_diagnostics=True,
        calibration_bins=4,
    )

    assert calls == [validation_inputs, report_inputs]
    assert report.validation_threshold.source_population_name == PopulationName.VALIDATION
    assert report.validation_threshold.selected_threshold == pytest.approx(0.60)
    assert report.headline_metrics.auroc == pytest.approx(report_population.auroc)
    assert report.headline_metrics.auprc == pytest.approx(report_population.auprc)
    assert report.headline_metrics.prediction_std == pytest.approx(report_population.prob_std)
    assert report.headline_metrics.f1_at_val_threshold == pytest.approx(0.0)
    assert report.headline_metrics.precision_at_val_threshold == pytest.approx(0.0)
    assert report.headline_metrics.recall_at_val_threshold == pytest.approx(0.0)
    assert report.headline_metrics.specificity_at_val_threshold == pytest.approx(0.5)
    assert report.diagnostics is not None
    assert report.diagnostics.oracle_same_population_threshold is not None
    assert report.diagnostics.oracle_same_population_threshold.source_population_name == PopulationName.FINAL_TEST
    assert report.diagnostics.oracle_same_population_metrics is not None
    assert report.diagnostics.oracle_same_population_metrics.f1 == pytest.approx(0.8)
    assert report.diagnostics.oracle_same_population_metrics.f1 > report.headline_metrics.f1_at_val_threshold
    assert report.diagnostics.diagnostic_only is True
    assert report.diagnostics.formal_excluded is True
    assert report.threshold_source == "validation"
    assert report.dataset_name == DatasetName.AMAZON
    assert report.graph_regime == GraphRegime.TRANSDUCTIVE_STANDARD
    assert report.run_id == "run-123"
    assert report.validation_split == PopulationName.VALIDATION
    assert report.report_split == PopulationName.FINAL_TEST
    assert report.eval_type == "head_only"

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
        eval_head_only.FormalHeadOnlyReport.model_validate(old_payload)

    missing_identity_payload = report.model_dump(mode="python")
    missing_identity_payload.pop("dataset_name")
    with pytest.raises(ValidationError):
        eval_head_only.FormalHeadOnlyReport.model_validate(missing_identity_payload)

    missing_hash_payload = report.model_dump(mode="python")
    missing_hash_payload.pop("prompt_audit_hash")
    with pytest.raises(ValidationError):
        eval_head_only.FormalHeadOnlyReport.model_validate(missing_hash_payload)

    contaminated_payload = report.model_dump(mode="python")
    contaminated_payload["neighbor_label_counts_visible"] = True
    with pytest.raises(ValidationError):
        eval_head_only.FormalHeadOnlyReport.model_validate(contaminated_payload)

    wrong_projection_payload = report.model_dump(mode="python")
    wrong_projection_payload["evidence_card_projection"] = "internal_full"
    with pytest.raises(ValidationError):
        eval_head_only.FormalHeadOnlyReport.model_validate(wrong_projection_payload)


def test_formal_head_only_eval_can_apply_temperature_scaling(monkeypatch: pytest.MonkeyPatch):
    validation_inputs = _inputs(PopulationName.VALIDATION, (3, 4, 8, 9), (0, 0, 1, 1))
    report_inputs = _inputs(PopulationName.FINAL_TEST, (21, 22, 23, 24), (0, 1, 1, 0))
    validation_report = _scorer_report(
        population_name=PopulationName.VALIDATION,
        probs=(0.01, 0.15, 0.85, 0.99),
        labels=(0, 1, 1, 0),
        node_ids=(3, 4, 8, 9),
    )
    report_population = _scorer_report(
        population_name=PopulationName.FINAL_TEST,
        probs=(0.02, 0.20, 0.80, 0.98),
        labels=(0, 1, 1, 0),
        node_ids=(21, 22, 23, 24),
    )

    def fake_score_head(**kwargs):
        if kwargs["inputs"] == validation_inputs:
            return validation_report
        if kwargs["inputs"] == report_inputs:
            return report_population
        raise AssertionError("unexpected HeadScoringInputs")

    monkeypatch.setattr(eval_head_only, "score_head", fake_score_head)

    expected_temperature = fit_temperature_on_validation(
        logits=logits_from_probs(probs=validation_report.probs),
        labels=validation_report.labels,
    )
    expected_validation_probs = apply_temperature_to_probs(
        probs=validation_report.probs,
        temperature=expected_temperature,
    )
    expected_report_probs = apply_temperature_to_probs(
        probs=report_population.probs,
        temperature=expected_temperature,
    )
    expected_threshold = select_validation_threshold(
        probs=expected_validation_probs,
        labels=validation_report.labels,
        source_population_name=PopulationName.VALIDATION,
        selection_metric="f1",
    )
    expected_calibration = compute_calibration_summary(
        probs=expected_report_probs,
        labels=report_population.labels,
        population_name=PopulationName.FINAL_TEST,
        num_bins=4,
    )

    report = eval_head_only.run_formal_head_only_eval(
        validation_inputs=validation_inputs,
        report_inputs=report_inputs,
        validation_population_metadata=_population_metadata(PopulationName.VALIDATION),
        report_population_metadata=_population_metadata(PopulationName.FINAL_TEST),
        model=object(),
        cls_head=object(),
        tokenizer=object(),
        thinking_mode=ThinkingMode.NON_THINKING,
        checkpoint_source="best_checkpoint",
        checkpoint_bundle=_checkpoint_bundle(),
        run_id="run-123",
        **_score_head_audit_kwargs(),
        threshold_selection_metric="f1",
        include_oracle_diagnostics=False,
        calibration_bins=4,
        apply_temperature_scaling=True,
    )

    assert report.temperature == pytest.approx(expected_temperature)
    assert report.validation_threshold.selected_threshold == pytest.approx(
        expected_threshold.selected_threshold
    )
    assert report.calibration.brier_score == pytest.approx(expected_calibration.brier_score)
    assert report.calibration.expected_calibration_error == pytest.approx(
        expected_calibration.expected_calibration_error
    )
    assert report.calibration.max_calibration_gap == pytest.approx(
        expected_calibration.max_calibration_gap
    )
    assert report.headline_metrics.auroc == pytest.approx(report_population.auroc)
    assert report.headline_metrics.auprc == pytest.approx(report_population.auprc)
    assert report.headline_metrics.prediction_std == pytest.approx(np.std(expected_report_probs))


def test_formal_head_only_eval_rejects_validation_as_report_population():
    validation_inputs = _inputs(PopulationName.VALIDATION, (3, 4), (0, 1))
    report_inputs = _inputs(PopulationName.FINAL_TEST, (21, 22), (0, 1))

    with pytest.raises(ValueError):
        eval_head_only.run_formal_head_only_eval(
            validation_inputs=validation_inputs,
            report_inputs=report_inputs,
            validation_population_metadata=_population_metadata(PopulationName.VALIDATION),
            report_population_metadata=_population_metadata(PopulationName.VALIDATION),
            model=object(),
            cls_head=object(),
            tokenizer=object(),
            thinking_mode=ThinkingMode.NON_THINKING,
            checkpoint_source="best_checkpoint",
            checkpoint_bundle=_checkpoint_bundle(),
            run_id="run-123",
            **_score_head_audit_kwargs(),
            threshold_selection_metric="f1",
            include_oracle_diagnostics=False,
        )


def test_formal_head_only_eval_requires_cls_head_provenance_to_match_inputs(monkeypatch: pytest.MonkeyPatch):
    validation_inputs = _inputs(PopulationName.VALIDATION, (3, 4), (0, 1))
    report_inputs = _inputs(PopulationName.FINAL_TEST, (21, 22), (0, 1))
    mismatched_inputs = validation_inputs.model_copy(
        update={
            "checkpoint_provenance": CheckpointProvenance(
                path="outputs/gated/checkpoints/other/cls_head.safetensors",
                step=12,
                content_hash="d" * 64,
            )
        }
    )

    def fail_if_score_head_runs(**_kwargs):
        pytest.fail("score_head should not run")

    monkeypatch.setattr(eval_head_only, "score_head", fail_if_score_head_runs)

    with pytest.raises(ValueError, match="must match the fail-closed cls_head checkpoint provenance"):
        eval_head_only.run_formal_head_only_eval(
            validation_inputs=mismatched_inputs,
            report_inputs=report_inputs,
            validation_population_metadata=_population_metadata(PopulationName.VALIDATION),
            report_population_metadata=_population_metadata(PopulationName.FINAL_TEST),
            model=object(),
            cls_head=object(),
            tokenizer=object(),
            thinking_mode=ThinkingMode.NON_THINKING,
            checkpoint_source="best_checkpoint",
            checkpoint_bundle=_checkpoint_bundle(),
            run_id="run-123",
            **_score_head_audit_kwargs(),
            threshold_selection_metric="f1",
            include_oracle_diagnostics=False,
        )


def test_formal_head_only_eval_rejects_single_class_population(monkeypatch: pytest.MonkeyPatch):
    validation_inputs = _inputs(PopulationName.VALIDATION, (3, 4, 8, 9), (0, 0, 1, 1))
    report_inputs = _inputs(PopulationName.FINAL_TEST, (21, 22, 23), (0, 0, 0))
    validation_report = _scorer_report(
        population_name=PopulationName.VALIDATION,
        probs=(0.10, 0.40, 0.60, 0.90),
        labels=(0, 0, 1, 1),
        node_ids=(3, 4, 8, 9),
    )
    single_class_report = _scorer_report(
        population_name=PopulationName.FINAL_TEST,
        probs=(0.10, 0.20, 0.30),
        labels=(0, 0, 0),
        node_ids=(21, 22, 23),
    )

    def fake_score_head(**kwargs):
        if kwargs["inputs"] == validation_inputs:
            return validation_report
        return single_class_report

    monkeypatch.setattr(eval_head_only, "score_head", fake_score_head)

    with pytest.raises(ValueError, match="single-class population is not formal"):
        eval_head_only.run_formal_head_only_eval(
            validation_inputs=validation_inputs,
            report_inputs=report_inputs,
            validation_population_metadata=_population_metadata(PopulationName.VALIDATION),
            report_population_metadata=_population_metadata(PopulationName.FINAL_TEST),
            model=object(),
            cls_head=object(),
            tokenizer=object(),
            thinking_mode=ThinkingMode.NON_THINKING,
            checkpoint_source="best_checkpoint",
            checkpoint_bundle=_checkpoint_bundle(),
            run_id="run-123",
            **_score_head_audit_kwargs(),
            threshold_selection_metric="f1",
            include_oracle_diagnostics=False,
        )


def test_checkpoint_bundle_requires_shared_provenance_identity():
    base = dict(
        run_id="run-123",
        checkpoint_step=12,
        commit="deadbeef",
        config_fingerprint="cfg-123",
        data_manifest_hash="manifest-123",
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
    )

    with pytest.raises(ValueError, match="peft_adapter.commit must match llm_backbone.commit"):
        eval_head_only.FormalHeadOnlyCheckpointBundle(
            llm_backbone=eval_head_only.FormalHeadOnlyCheckpointComponent(
                path="backbone.safetensors",
                content_hash="a" * 64,
                **base,
            ),
            peft_adapter=eval_head_only.FormalHeadOnlyCheckpointComponent(
                path="adapter.safetensors",
                content_hash="b" * 64,
                commit="cafebabe",
                run_id=base["run_id"],
                checkpoint_step=base["checkpoint_step"],
                config_fingerprint=base["config_fingerprint"],
                data_manifest_hash=base["data_manifest_hash"],
                graph_regime=base["graph_regime"],
            ),
            cls_head=eval_head_only.FormalHeadOnlyCheckpointComponent(
                path="cls_head.safetensors",
                content_hash="c" * 64,
                **base,
            ),
        )


def test_select_validation_threshold_and_threshold_metrics_are_deterministic():
    selection = select_validation_threshold(
        probs=(0.10, 0.40, 0.60, 0.90),
        labels=(0, 0, 1, 1),
        source_population_name=PopulationName.VALIDATION,
        selection_metric="f1",
    )

    assert selection.selected_threshold == pytest.approx(0.60)
    assert selection.metrics_at_selected_threshold.precision == pytest.approx(1.0)
    assert selection.metrics_at_selected_threshold.recall == pytest.approx(1.0)
    assert selection.metrics_at_selected_threshold.specificity == pytest.approx(1.0)
    assert selection.metrics_at_selected_threshold.f1 == pytest.approx(1.0)

    metrics = compute_threshold_metrics(
        probs=(0.55, 0.58, 0.59, 0.95),
        labels=(0, 1, 1, 0),
        threshold=selection.selected_threshold,
    )
    assert metrics.f1 == pytest.approx(0.0)
    assert metrics.specificity == pytest.approx(0.5)


def test_compute_calibration_summary_reports_brier_ece_and_max_gap():
    summary = compute_calibration_summary(
        probs=(0.10, 0.40, 0.80, 0.90),
        labels=(0, 0, 1, 1),
        population_name=PopulationName.FINAL_TEST,
        num_bins=2,
    )

    expected_brier = np.mean((np.asarray((0.10, 0.40, 0.80, 0.90)) - np.asarray((0, 0, 1, 1))) ** 2)
    low_bin_gap = abs(np.mean((0.10, 0.40)) - np.mean((0, 0)))
    high_bin_gap = abs(np.mean((0.80, 0.90)) - np.mean((1, 1)))
    expected_ece = 0.5 * low_bin_gap + 0.5 * high_bin_gap

    assert summary.brier_score == pytest.approx(expected_brier)
    assert summary.expected_calibration_error == pytest.approx(expected_ece)
    assert summary.max_calibration_gap == pytest.approx(max(low_bin_gap, high_bin_gap))
    assert summary.population_name == PopulationName.FINAL_TEST
    assert summary.num_bins == 2
