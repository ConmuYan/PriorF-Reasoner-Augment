"""Formal validation-selected fusion evaluation.

This module implements the formal fusion contract:

* accept separate validation and test-like prediction inputs,
* tune ``alpha`` on validation only via a constrained selection rule,
* report test-like metrics using the frozen validation-selected ``alpha``,
* fail closed on provenance / schema / population misuse.
"""

from __future__ import annotations

from math import isfinite
from typing import Literal

import numpy as np
import sklearn.metrics
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eval.head_scoring import CheckpointProvenance, ScorerReport
from llm.fusion import fuse_probabilities
from priorf_teacher.schema import DatasetName, GraphRegime, PopulationName

__all__ = (
    "FusionEvalConfig",
    "FusionEvalReport",
    "FusionPopulationInputs",
    "PopulationFusionMetrics",
    "ProbabilityMetrics",
    "run_formal_fusion_eval",
)

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class FusionPopulationInputs(BaseModel):
    """One population worth of aligned teacher + head-only predictions."""

    model_config = _STRICT_MODEL_CONFIG

    head_report: ScorerReport
    teacher_probs: tuple[float, ...] = Field(min_length=1)
    teacher_node_ids: tuple[int, ...] = Field(min_length=1)

    @field_validator("teacher_probs")
    @classmethod
    def _teacher_probabilities_must_be_finite_and_bounded(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        for value in values:
            if not isfinite(float(value)):
                raise ValueError("teacher_probs must be finite")
            if value < 0.0 or value > 1.0:
                raise ValueError("teacher_probs must be within [0.0, 1.0]")
        return values

    @model_validator(mode="after")
    def _teacher_vectors_must_align_to_head_report(self) -> "FusionPopulationInputs":
        if self.head_report.checkpoint_provenance is None:
            raise ValueError("head_report.checkpoint_provenance is required; formal fusion eval fails closed when provenance is missing")
        if len(self.teacher_probs) != self.head_report.n_total:
            raise ValueError("teacher_probs length must equal head_report.n_total")
        if len(self.teacher_node_ids) != self.head_report.n_total:
            raise ValueError("teacher_node_ids length must equal head_report.n_total")
        if tuple(self.teacher_node_ids) != self.head_report.node_ids:
            raise ValueError("teacher_node_ids must exactly match head_report.node_ids")
        return self


class FusionEvalConfig(BaseModel):
    """Frozen selection/reporting policy for formal fusion evaluation."""

    model_config = _STRICT_MODEL_CONFIG

    alpha_candidates: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    primary_metric: Literal["auprc", "auroc", "f1_at_frozen_threshold"] = "auprc"
    secondary_guardrail_metric: Literal["auroc"] = "auroc"
    teacher_degradation_tolerance: float = Field(default=0.0, ge=0.0, le=1.0)
    min_student_alpha: float = Field(default=0.05, ge=0.0, le=1.0)
    frozen_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    tie_breaker: Literal["smaller_alpha", "larger_alpha"] = "smaller_alpha"

    @field_validator("alpha_candidates")
    @classmethod
    def _alpha_candidates_must_be_sorted_unique_and_bounded(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values:
            raise ValueError("alpha_candidates must not be empty")
        last = None
        seen: set[float] = set()
        for value in values:
            if not isfinite(float(value)):
                raise ValueError("alpha_candidates must be finite")
            if value < 0.0 or value > 1.0:
                raise ValueError("alpha_candidates must lie within [0.0, 1.0]")
            if value in seen:
                raise ValueError("alpha_candidates must be unique")
            if last is not None and value < last:
                raise ValueError("alpha_candidates must be sorted in ascending order")
            seen.add(value)
            last = value
        return values

    @model_validator(mode="after")
    def _f1_primary_requires_frozen_threshold(self) -> "FusionEvalConfig":
        if self.primary_metric == "f1_at_frozen_threshold" and self.frozen_threshold is None:
            raise ValueError("primary_metric=f1_at_frozen_threshold requires frozen_threshold")
        return self


class PopulationMetadataEcho(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    dataset_name: DatasetName
    population_name: PopulationName
    graph_regime: GraphRegime
    n_total: int = Field(ge=1)


class ProbabilityMetrics(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    n_total: int = Field(ge=1)
    n_positive: int = Field(ge=0)
    n_negative: int = Field(ge=0)
    is_single_class_population: bool
    auroc: float | None
    auprc: float | None
    brier_score: float
    frozen_threshold: float | None
    f1_at_frozen_threshold: float | None
    precision_at_frozen_threshold: float | None
    recall_at_frozen_threshold: float | None
    specificity_at_frozen_threshold: float | None
    prob_mean: float
    prob_std: float
    prob_min: float
    prob_max: float
    prob_q25: float
    prob_q50: float
    prob_q75: float

    @model_validator(mode="after")
    def _counts_and_threshold_metrics_must_align(self) -> "ProbabilityMetrics":
        if self.n_positive + self.n_negative != self.n_total:
            raise ValueError("n_positive + n_negative must equal n_total")
        threshold_metrics = (
            self.f1_at_frozen_threshold,
            self.precision_at_frozen_threshold,
            self.recall_at_frozen_threshold,
            self.specificity_at_frozen_threshold,
        )
        if self.frozen_threshold is None and any(metric is not None for metric in threshold_metrics):
            raise ValueError("threshold metrics require frozen_threshold")
        if self.frozen_threshold is not None and any(metric is None for metric in threshold_metrics):
            raise ValueError("all threshold metrics must be populated when frozen_threshold is set")
        return self


class PopulationFusionMetrics(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    population: PopulationMetadataEcho
    teacher_only: ProbabilityMetrics
    head_only: ProbabilityMetrics
    fusion: ProbabilityMetrics


class FusionSelectionSummary(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    selected_from_population: Literal[PopulationName.VALIDATION] = PopulationName.VALIDATION
    optimal_alpha: float = Field(ge=0.0, le=1.0)
    primary_metric_name: Literal["auprc", "auroc", "f1_at_frozen_threshold"]
    primary_metric_value: float
    teacher_primary_metric_value: float
    secondary_guardrail_metric_name: Literal["auroc"]
    secondary_guardrail_value: float
    teacher_guardrail_baseline: float
    secondary_guardrail_pass: bool
    teacher_degradation_tolerance_triggered: bool


class FusionEvalReport(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    checkpoint_provenance: CheckpointProvenance
    selected_on_validation_only: Literal[True] = True
    student_contribution_pass: bool
    validation_metrics: PopulationFusionMetrics
    report_metrics: PopulationFusionMetrics
    selection: FusionSelectionSummary


def run_formal_fusion_eval(
    *,
    validation_inputs: FusionPopulationInputs,
    report_inputs: FusionPopulationInputs,
    config: FusionEvalConfig = FusionEvalConfig(),
) -> FusionEvalReport:
    """Tune alpha on validation only, then report with the frozen alpha elsewhere."""

    _validate_formal_inputs(validation_inputs=validation_inputs, report_inputs=report_inputs)

    validation_labels = tuple(int(label) for label in validation_inputs.head_report.labels)
    validation_head_probs = tuple(float(prob) for prob in validation_inputs.head_report.probs)
    validation_teacher_probs = tuple(float(prob) for prob in validation_inputs.teacher_probs)

    validation_teacher_metrics = _compute_probability_metrics(
        labels=validation_labels,
        probs=validation_teacher_probs,
        frozen_threshold=config.frozen_threshold,
    )
    validation_head_metrics = _compute_probability_metrics(
        labels=validation_labels,
        probs=validation_head_probs,
        frozen_threshold=config.frozen_threshold,
    )

    teacher_primary_metric_value = _metric_value(
        metrics=validation_teacher_metrics,
        metric_name=config.primary_metric,
        context="validation teacher-only",
    )
    teacher_guardrail_baseline = _metric_value(
        metrics=validation_teacher_metrics,
        metric_name=config.secondary_guardrail_metric,
        context="validation teacher-only guardrail",
    )

    best_passing: tuple[float, float, float, ProbabilityMetrics] | None = None
    best_any: tuple[float, float, float, ProbabilityMetrics] | None = None

    for alpha in config.alpha_candidates:
        validation_fused_probs = fuse_probabilities(
            teacher_probs=validation_teacher_probs,
            student_probs=validation_head_probs,
            alpha=alpha,
        )
        fused_metrics = _compute_probability_metrics(
            labels=validation_labels,
            probs=validation_fused_probs,
            frozen_threshold=config.frozen_threshold,
        )
        primary_metric_value = _metric_value(
            metrics=fused_metrics,
            metric_name=config.primary_metric,
            context=f"validation fusion alpha={alpha}",
        )
        guardrail_value = _metric_value(
            metrics=fused_metrics,
            metric_name=config.secondary_guardrail_metric,
            context=f"validation fusion guardrail alpha={alpha}",
        )
        tie_key = -alpha if config.tie_breaker == "smaller_alpha" else alpha
        candidate = (primary_metric_value, tie_key, alpha, fused_metrics)
        if best_any is None or candidate[:2] > best_any[:2]:
            best_any = candidate
        if guardrail_value >= teacher_guardrail_baseline - config.teacher_degradation_tolerance:
            if best_passing is None or candidate[:2] > best_passing[:2]:
                best_passing = candidate

    if best_any is None:
        raise ValueError("alpha selection produced no candidates")

    selected_candidate = best_passing or best_any
    _, _, optimal_alpha, validation_fusion_metrics = selected_candidate
    selected_guardrail_value = _metric_value(
        metrics=validation_fusion_metrics,
        metric_name=config.secondary_guardrail_metric,
        context="selected validation fusion guardrail",
    )
    secondary_guardrail_pass = selected_guardrail_value >= teacher_guardrail_baseline - config.teacher_degradation_tolerance
    teacher_degradation_tolerance_triggered = best_passing is None

    report_labels = tuple(int(label) for label in report_inputs.head_report.labels)
    report_head_probs = tuple(float(prob) for prob in report_inputs.head_report.probs)
    report_teacher_probs = tuple(float(prob) for prob in report_inputs.teacher_probs)
    report_fusion_probs = fuse_probabilities(
        teacher_probs=report_teacher_probs,
        student_probs=report_head_probs,
        alpha=optimal_alpha,
    )

    report_teacher_metrics = _compute_probability_metrics(
        labels=report_labels,
        probs=report_teacher_probs,
        frozen_threshold=config.frozen_threshold,
    )
    report_head_metrics = _compute_probability_metrics(
        labels=report_labels,
        probs=report_head_probs,
        frozen_threshold=config.frozen_threshold,
    )
    report_fusion_metrics = _compute_probability_metrics(
        labels=report_labels,
        probs=report_fusion_probs,
        frozen_threshold=config.frozen_threshold,
    )

    selected_primary_metric_value = _metric_value(
        metrics=validation_fusion_metrics,
        metric_name=config.primary_metric,
        context="selected validation fusion primary metric",
    )
    student_contribution_pass = bool(
        optimal_alpha >= config.min_student_alpha
        and secondary_guardrail_pass
        and selected_primary_metric_value > teacher_primary_metric_value
    )

    return FusionEvalReport(
        checkpoint_provenance=validation_inputs.head_report.checkpoint_provenance,
        student_contribution_pass=student_contribution_pass,
        validation_metrics=PopulationFusionMetrics(
            population=_population_metadata(validation_inputs),
            teacher_only=validation_teacher_metrics,
            head_only=validation_head_metrics,
            fusion=validation_fusion_metrics,
        ),
        report_metrics=PopulationFusionMetrics(
            population=_population_metadata(report_inputs),
            teacher_only=report_teacher_metrics,
            head_only=report_head_metrics,
            fusion=report_fusion_metrics,
        ),
        selection=FusionSelectionSummary(
            optimal_alpha=optimal_alpha,
            primary_metric_name=config.primary_metric,
            primary_metric_value=selected_primary_metric_value,
            teacher_primary_metric_value=teacher_primary_metric_value,
            secondary_guardrail_metric_name=config.secondary_guardrail_metric,
            secondary_guardrail_value=selected_guardrail_value,
            teacher_guardrail_baseline=teacher_guardrail_baseline,
            secondary_guardrail_pass=secondary_guardrail_pass,
            teacher_degradation_tolerance_triggered=teacher_degradation_tolerance_triggered,
        ),
    )


def _validate_formal_inputs(*, validation_inputs: FusionPopulationInputs, report_inputs: FusionPopulationInputs) -> None:
    validation_report = validation_inputs.head_report
    report_report = report_inputs.head_report

    if validation_report.population_name != PopulationName.VALIDATION:
        raise ValueError("validation_inputs.head_report.population_name must be validation")
    if report_report.population_name == PopulationName.VALIDATION:
        raise ValueError("report_inputs.head_report.population_name must be test-like, not validation")
    if validation_report.dataset_name != report_report.dataset_name:
        raise ValueError("validation and report populations must share dataset_name")
    if validation_report.graph_regime != report_report.graph_regime:
        raise ValueError("validation and report populations must share graph_regime")
    if validation_report.checkpoint_provenance != report_report.checkpoint_provenance:
        raise ValueError("validation and report populations must share identical checkpoint_provenance")


def _population_metadata(inputs: FusionPopulationInputs) -> PopulationMetadataEcho:
    report = inputs.head_report
    return PopulationMetadataEcho(
        dataset_name=report.dataset_name,
        population_name=report.population_name,
        graph_regime=report.graph_regime,
        n_total=report.n_total,
    )


def _compute_probability_metrics(*, labels: tuple[int, ...], probs: tuple[float, ...], frozen_threshold: float | None) -> ProbabilityMetrics:
    if len(labels) != len(probs):
        raise ValueError("labels and probs must have identical lengths")
    if not labels:
        raise ValueError("labels and probs must be non-empty")
    for label in labels:
        if label not in (0, 1):
            raise ValueError("labels must be binary 0/1")
    for prob in probs:
        if not isfinite(float(prob)):
            raise ValueError("probs must be finite")
        if prob < 0.0 or prob > 1.0:
            raise ValueError("probs must be within [0.0, 1.0]")

    labels_np = np.asarray(labels, dtype=np.int64)
    probs_np = np.asarray(probs, dtype=np.float64)
    n_total = int(labels_np.size)
    n_positive = int(labels_np.sum())
    n_negative = int(n_total - n_positive)
    is_single_class_population = n_positive == 0 or n_negative == 0

    auroc: float | None = None
    auprc: float | None = None
    if not is_single_class_population:
        auroc = float(sklearn.metrics.roc_auc_score(labels_np, probs_np))
        auprc = float(sklearn.metrics.average_precision_score(labels_np, probs_np))

    f1_at_threshold: float | None = None
    precision_at_threshold: float | None = None
    recall_at_threshold: float | None = None
    specificity_at_threshold: float | None = None
    if frozen_threshold is not None:
        preds_np = (probs_np >= frozen_threshold).astype(np.int64)
        f1_at_threshold = float(sklearn.metrics.f1_score(labels_np, preds_np, zero_division=0))
        precision_at_threshold = float(sklearn.metrics.precision_score(labels_np, preds_np, zero_division=0))
        recall_at_threshold = float(sklearn.metrics.recall_score(labels_np, preds_np, zero_division=0))
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(labels_np, preds_np, labels=[0, 1]).ravel()
        specificity_at_threshold = float(tn / (tn + fp)) if (tn + fp) else 0.0

    return ProbabilityMetrics(
        n_total=n_total,
        n_positive=n_positive,
        n_negative=n_negative,
        is_single_class_population=is_single_class_population,
        auroc=auroc,
        auprc=auprc,
        brier_score=float(sklearn.metrics.brier_score_loss(labels_np, probs_np)),
        frozen_threshold=frozen_threshold,
        f1_at_frozen_threshold=f1_at_threshold,
        precision_at_frozen_threshold=precision_at_threshold,
        recall_at_frozen_threshold=recall_at_threshold,
        specificity_at_frozen_threshold=specificity_at_threshold,
        prob_mean=float(np.mean(probs_np)),
        prob_std=float(np.std(probs_np)),
        prob_min=float(np.min(probs_np)),
        prob_max=float(np.max(probs_np)),
        prob_q25=float(np.quantile(probs_np, 0.25)),
        prob_q50=float(np.quantile(probs_np, 0.50)),
        prob_q75=float(np.quantile(probs_np, 0.75)),
    )


def _metric_value(*, metrics: ProbabilityMetrics, metric_name: Literal["auprc", "auroc", "f1_at_frozen_threshold"], context: str) -> float:
    value = getattr(metrics, metric_name)
    if value is None:
        raise ValueError(f"{context} requires metric {metric_name}, but it is unavailable for this population")
    return float(value)
