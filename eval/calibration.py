"""Shared calibration and frozen-threshold contracts for formal evaluation.

This module keeps threshold selection and calibration reporting separate from the
canonical ``eval.head_scoring`` probability path.  Downstream formal evaluators
must consume a scorer report, select thresholds on validation only, and then
apply the frozen threshold to a distinct report population.
"""

from __future__ import annotations

from math import isfinite
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from priorf_teacher.schema import PopulationName

__all__ = (
    "CalibrationSummary",
    "FrozenThresholdReport",
    "ThresholdMetrics",
    "compute_calibration_summary",
    "compute_threshold_metrics",
    "select_validation_threshold",
)

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_CALIBRATION_SCHEMA_VERSION = "calibration/v1"
_THRESHOLD_TIE_BREAK_POLICY = "max_metric_then_max_specificity_then_min_threshold"
ThresholdSelectionMetric = Literal["f1"]


class ThresholdMetrics(BaseModel):
    """Binary metrics computed at one explicit frozen threshold."""

    model_config = _STRICT_MODEL_CONFIG

    schema_version: Literal["calibration/v1"] = _CALIBRATION_SCHEMA_VERSION
    threshold: StrictFloat = Field(ge=0.0, le=1.0)
    precision: StrictFloat = Field(ge=0.0, le=1.0)
    recall: StrictFloat = Field(ge=0.0, le=1.0)
    specificity: StrictFloat = Field(ge=0.0, le=1.0)
    f1: StrictFloat = Field(ge=0.0, le=1.0)
    tp: StrictInt = Field(ge=0)
    fp: StrictInt = Field(ge=0)
    tn: StrictInt = Field(ge=0)
    fn: StrictInt = Field(ge=0)
    n_pred_positive: StrictInt = Field(ge=0)
    n_pred_negative: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def _counts_consistent(self) -> "ThresholdMetrics":
        if self.tp + self.fp != self.n_pred_positive:
            raise ValueError("tp + fp must equal n_pred_positive")
        if self.tn + self.fn != self.n_pred_negative:
            raise ValueError("tn + fn must equal n_pred_negative")
        return self


class FrozenThresholdReport(BaseModel):
    """Validation-only threshold selection artifact."""

    model_config = _STRICT_MODEL_CONFIG

    schema_version: Literal["calibration/v1"] = _CALIBRATION_SCHEMA_VERSION
    source_population_name: PopulationName
    selection_metric: ThresholdSelectionMetric
    selected_threshold: StrictFloat = Field(ge=0.0, le=1.0)
    tie_break_policy: Literal["max_metric_then_max_specificity_then_min_threshold"] = (
        _THRESHOLD_TIE_BREAK_POLICY
    )
    metrics_at_selected_threshold: ThresholdMetrics

    @model_validator(mode="after")
    def _threshold_echo_consistent(self) -> "FrozenThresholdReport":
        if abs(self.selected_threshold - self.metrics_at_selected_threshold.threshold) > 1e-12:
            raise ValueError("selected_threshold must equal metrics_at_selected_threshold.threshold")
        return self


class CalibrationSummary(BaseModel):
    """Compact calibration audit for one evaluated population."""

    model_config = _STRICT_MODEL_CONFIG

    schema_version: Literal["calibration/v1"] = _CALIBRATION_SCHEMA_VERSION
    population_name: PopulationName
    num_bins: StrictInt = Field(ge=1)
    brier_score: StrictFloat = Field(ge=0.0)
    expected_calibration_error: StrictFloat = Field(ge=0.0)
    max_calibration_gap: StrictFloat = Field(ge=0.0)


def compute_threshold_metrics(*, probs, labels, threshold: float) -> ThresholdMetrics:
    """Compute frozen-threshold metrics for binary probabilities and labels."""

    probs_np, labels_np = _validate_probs_and_labels(probs=probs, labels=labels)
    threshold_value = float(threshold)
    if not isfinite(threshold_value) or threshold_value < 0.0 or threshold_value > 1.0:
        raise ValueError("threshold must be finite and within [0.0, 1.0]")
    if _is_single_class(labels_np):
        raise ValueError("threshold metrics require both positive and negative labels")

    predictions = probs_np >= threshold_value
    positives = labels_np == 1
    negatives = labels_np == 0

    tp = int(np.logical_and(predictions, positives).sum())
    fp = int(np.logical_and(predictions, negatives).sum())
    tn = int(np.logical_and(~predictions, negatives).sum())
    fn = int(np.logical_and(~predictions, positives).sum())

    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)

    return ThresholdMetrics(
        threshold=threshold_value,
        precision=float(precision),
        recall=float(recall),
        specificity=float(specificity),
        f1=float(f1),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        n_pred_positive=int(predictions.sum()),
        n_pred_negative=int((~predictions).sum()),
    )


def select_validation_threshold(
    *,
    probs,
    labels,
    source_population_name: PopulationName,
    selection_metric: ThresholdSelectionMetric,
) -> FrozenThresholdReport:
    """Select one deterministic frozen threshold on validation only."""

    if source_population_name != PopulationName.VALIDATION:
        raise ValueError("frozen threshold selection is validation-only")
    if selection_metric != "f1":
        raise ValueError(f"unsupported selection_metric: {selection_metric}")

    probs_np, labels_np = _validate_probs_and_labels(probs=probs, labels=labels)
    if _is_single_class(labels_np):
        raise ValueError("validation threshold selection requires both positive and negative labels")

    best_metrics: ThresholdMetrics | None = None
    best_key: tuple[float, float, float] | None = None
    candidate_thresholds = sorted({0.0, 1.0, *(float(value) for value in probs_np.tolist())})
    for threshold in candidate_thresholds:
        metrics = compute_threshold_metrics(probs=probs_np, labels=labels_np, threshold=threshold)
        key = (float(metrics.f1), float(metrics.specificity), -float(metrics.threshold))
        if best_key is None or key > best_key:
            best_key = key
            best_metrics = metrics

    assert best_metrics is not None
    return FrozenThresholdReport(
        source_population_name=source_population_name,
        selection_metric=selection_metric,
        selected_threshold=float(best_metrics.threshold),
        metrics_at_selected_threshold=best_metrics,
    )


def compute_calibration_summary(
    *,
    probs,
    labels,
    population_name: PopulationName,
    num_bins: int,
) -> CalibrationSummary:
    """Compute Brier + equal-width-bin ECE / max-gap calibration summary."""

    probs_np, labels_np = _validate_probs_and_labels(probs=probs, labels=labels)
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")

    brier_score = float(np.mean((probs_np - labels_np.astype(np.float64)) ** 2))
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64)
    expected_calibration_error = 0.0
    max_gap = 0.0

    for index in range(num_bins):
        left = bin_edges[index]
        right = bin_edges[index + 1]
        if index == num_bins - 1:
            in_bin = (probs_np >= left) & (probs_np <= right)
        else:
            in_bin = (probs_np >= left) & (probs_np < right)
        if not np.any(in_bin):
            continue
        bin_probs = probs_np[in_bin]
        bin_labels = labels_np[in_bin].astype(np.float64)
        gap = abs(float(bin_probs.mean()) - float(bin_labels.mean()))
        expected_calibration_error += (float(bin_probs.shape[0]) / float(probs_np.shape[0])) * gap
        max_gap = max(max_gap, gap)

    return CalibrationSummary(
        population_name=population_name,
        num_bins=int(num_bins),
        brier_score=brier_score,
        expected_calibration_error=float(expected_calibration_error),
        max_calibration_gap=float(max_gap),
    )


def _validate_probs_and_labels(*, probs, labels) -> tuple[np.ndarray, np.ndarray]:
    probs_np = np.asarray(tuple(float(value) for value in probs), dtype=np.float64)
    labels_np = np.asarray(tuple(int(value) for value in labels), dtype=np.int64)
    if probs_np.ndim != 1 or labels_np.ndim != 1:
        raise ValueError("probs and labels must be 1-D")
    if probs_np.shape[0] == 0 or labels_np.shape[0] == 0:
        raise ValueError("probs and labels must be non-empty")
    if probs_np.shape[0] != labels_np.shape[0]:
        raise ValueError("probs and labels must have the same length")
    if not np.all(np.isfinite(probs_np)):
        raise ValueError("probs must be finite")
    if np.any(probs_np < 0.0) or np.any(probs_np > 1.0):
        raise ValueError("probs must lie in [0.0, 1.0]")
    if np.any((labels_np != 0) & (labels_np != 1)):
        raise ValueError("labels must be strictly 0 or 1")
    return probs_np, labels_np


def _is_single_class(labels_np: np.ndarray) -> bool:
    return bool(np.all(labels_np == 0) or np.all(labels_np == 1))
