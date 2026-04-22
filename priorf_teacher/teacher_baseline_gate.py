"""Fail-closed teacher baseline gate for Task 2.

This module computes a baseline metric from caller-provided validation labels and
teacher probabilities only.  It does not run teacher inference and it does not
accept diagnostic/test-like populations for metric calculation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from graph_data.manifests import load_data_manifest
from graph_data.validators import compute_file_sha256
from priorf_teacher.schema import DatasetName, GraphRegime, MetricName, PopulationName, TeacherBaselineReport

_INTERFERING_TEACHER_PREFIX: Final[str] = "TEACHER_BASELINE_"
_INTERFERING_PRIORF_PREFIX: Final[str] = "PRIORF_"
_INTERFERING_PRIORF_TOKENS: Final[tuple[str, ...]] = ("THRESHOLD", "METRIC", "POPULATION", "GRAPH_REGIME")


class _TeacherBaselineGateInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data_manifest_path: str = Field(min_length=1)
    teacher_checkpoint_path: str = Field(min_length=1)
    teacher_model_name: str = Field(min_length=1)
    teacher_checkpoint_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    dataset_name: DatasetName
    graph_regime: GraphRegime
    metric_name: MetricName
    metric_threshold: float = Field(ge=0.0, le=1.0)
    population_name: PopulationName
    validation_ground_truth_label: tuple[int, ...]
    validation_teacher_prob: tuple[float, ...]
    code_git_sha: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    report_path: str = Field(min_length=1)
    f1_positive_threshold: float | None = None

    @field_validator("data_manifest_path", "teacher_checkpoint_path", "teacher_model_name", "teacher_checkpoint_sha256", "code_git_sha", "report_path")
    @classmethod
    def _strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("required string parameter must not be blank")
        return value

    @field_validator("f1_positive_threshold")
    @classmethod
    def _f1_threshold_must_be_valid_when_present(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if value < 0.0 or value > 1.0:
            raise ValueError("f1_positive_threshold must be in [0, 1]")
        return value


def run_teacher_baseline_gate(
    *,
    data_manifest_path: str | Path,
    teacher_checkpoint_path: str | Path,
    teacher_model_name: str,
    teacher_checkpoint_sha256: str,
    dataset_name: DatasetName,
    graph_regime: GraphRegime,
    metric_name: MetricName,
    metric_threshold: float,
    population_name: PopulationName,
    validation_ground_truth_label: Sequence[int],
    validation_teacher_prob: Sequence[float],
    code_git_sha: str,
    report_path: str | Path,
    f1_positive_threshold: float | None = None,
) -> TeacherBaselineReport:
    """Run the validation-only teacher baseline gate and write its report.

    All parameters are explicit.  The core contract uses enum-typed dataset,
    graph-regime, metric, and population fields; runtime callers that pass raw
    strings are immediately coerced or rejected by Pydantic validation.
    """

    inputs = _TeacherBaselineGateInputs.model_validate(
        {
            "data_manifest_path": str(data_manifest_path),
            "teacher_checkpoint_path": str(teacher_checkpoint_path),
            "teacher_model_name": teacher_model_name,
            "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
            "dataset_name": dataset_name,
            "graph_regime": graph_regime,
            "metric_name": metric_name,
            "metric_threshold": metric_threshold,
            "population_name": population_name,
            "validation_ground_truth_label": tuple(validation_ground_truth_label),
            "validation_teacher_prob": tuple(validation_teacher_prob),
            "code_git_sha": code_git_sha,
            "report_path": str(report_path),
            "f1_positive_threshold": f1_positive_threshold,
        }
    )
    _raise_on_env_interference(os.environ)
    if inputs.population_name != PopulationName.VALIDATION:
        raise ValueError("teacher baseline gate may only run on validation population")

    manifest = load_data_manifest(inputs.data_manifest_path)
    if manifest.dataset_name != inputs.dataset_name.value:
        raise ValueError("dataset_name does not match data manifest")
    if manifest.graph_regime != inputs.graph_regime.value:
        raise ValueError("graph_regime does not match data manifest")
    _require_manifest_population(manifest.populations, inputs.population_name)

    metric_value = _compute_metric(
        metric_name=inputs.metric_name,
        labels=inputs.validation_ground_truth_label,
        probabilities=inputs.validation_teacher_prob,
        f1_positive_threshold=inputs.f1_positive_threshold,
    )
    report = TeacherBaselineReport(
        dataset_name=inputs.dataset_name,
        teacher_model_name=inputs.teacher_model_name,
        teacher_checkpoint_sha256=inputs.teacher_checkpoint_sha256,
        graph_regime=inputs.graph_regime,
        population_name=inputs.population_name,
        metric_name=inputs.metric_name,
        metric_value=metric_value,
        threshold=inputs.metric_threshold,
        passed=metric_value >= inputs.metric_threshold,
        data_manifest_sha256=compute_file_sha256(inputs.data_manifest_path),
        code_git_sha=inputs.code_git_sha,
        export_timestamp_utc=datetime.now(timezone.utc),
    )
    _write_report(report, inputs.report_path)
    return report


def _require_manifest_population(populations: Sequence[object], population_name: PopulationName) -> None:
    for population in populations:
        if getattr(population, "population_name") == population_name.value:
            return
    raise ValueError(f"data manifest does not contain population {population_name.value!r}")


def _raise_on_env_interference(environ: Mapping[str, str]) -> None:
    interfering = []
    for name in environ:
        upper = name.upper()
        if upper.startswith(_INTERFERING_TEACHER_PREFIX):
            interfering.append(name)
        elif upper.startswith(_INTERFERING_PRIORF_PREFIX) and any(token in upper for token in _INTERFERING_PRIORF_TOKENS):
            interfering.append(name)
    if interfering:
        joined = ", ".join(sorted(interfering))
        raise RuntimeError(f"environment variables may not override teacher baseline gate inputs: {joined}")


def _compute_metric(
    *,
    metric_name: MetricName,
    labels: Sequence[int],
    probabilities: Sequence[float],
    f1_positive_threshold: float | None,
) -> float:
    y_true, y_score = _validated_metric_arrays(labels, probabilities)
    if metric_name == MetricName.AUROC:
        return _auroc(y_true, y_score)
    if metric_name == MetricName.AUPRC:
        return _average_precision(y_true, y_score)
    if metric_name == MetricName.F1_MACRO:
        if f1_positive_threshold is None:
            raise ValueError("f1_positive_threshold must be explicitly provided for f1_macro")
        return _f1_macro(y_true, y_score >= f1_positive_threshold)
    raise ValueError(f"unsupported metric: {metric_name}")


def _validated_metric_arrays(labels: Sequence[int], probabilities: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    if len(labels) != len(probabilities):
        raise ValueError("validation labels and probabilities must have the same length")
    if not labels:
        raise ValueError("validation metric inputs must not be empty")
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(y_score).all():
        raise ValueError("teacher probabilities must be finite")
    if not np.logical_or(y_true == 0, y_true == 1).all():
        raise ValueError("validation labels must be binary ground_truth_label values")
    if not np.logical_and(y_score >= 0.0, y_score <= 1.0).all():
        raise ValueError("teacher probabilities must be in [0, 1]")
    return y_true, y_score


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = int(y_true.sum())
    negatives = int(y_true.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("auroc requires both positive and negative validation labels")
    ranks = _average_ranks(y_score)
    positive_rank_sum = float(ranks[y_true == 1].sum())
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        raise ValueError("auprc requires at least one positive validation label")
    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    true_positives = np.cumsum(sorted_true)
    ranks = np.arange(1, sorted_true.size + 1, dtype=np.float64)
    precision_at_hits = true_positives[sorted_true == 1] / ranks[sorted_true == 1]
    return float(precision_at_hits.sum() / positives)


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for klass in (0, 1):
        true_positive = int(np.logical_and(y_true == klass, y_pred == klass).sum())
        false_positive = int(np.logical_and(y_true != klass, y_pred == klass).sum())
        false_negative = int(np.logical_and(y_true == klass, y_pred != klass).sum())
        denominator = (2 * true_positive) + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)
    return float(sum(scores) / len(scores))


def _write_report(report: TeacherBaselineReport, path: str | Path) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=report_path.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
        handle.write(report.model_dump_json(indent=2))
        handle.write("\n")
    tmp_path.replace(report_path)


def _parse_json_array(value: str) -> list[float | int]:
    candidate = Path(value)
    text = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("expected a JSON array")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PriorF teacher baseline gate on validation rows only.")
    parser.add_argument("--data-manifest-path", required=True)
    parser.add_argument("--teacher-checkpoint-path", required=True)
    parser.add_argument("--teacher-model-name", required=True)
    parser.add_argument("--teacher-checkpoint-sha256", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--graph-regime", required=True)
    parser.add_argument("--metric-name", required=True)
    parser.add_argument("--metric-threshold", required=True, type=float)
    parser.add_argument("--population-name", required=True)
    parser.add_argument("--ground-truth-labels", required=True, help="JSON array or path to JSON array of validation labels")
    parser.add_argument("--teacher-probs", required=True, help="JSON array or path to JSON array of validation teacher probabilities")
    parser.add_argument("--code-git-sha", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--f1-positive-threshold", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        report = run_teacher_baseline_gate(
            data_manifest_path=args.data_manifest_path,
            teacher_checkpoint_path=args.teacher_checkpoint_path,
            teacher_model_name=args.teacher_model_name,
            teacher_checkpoint_sha256=args.teacher_checkpoint_sha256,
            dataset_name=DatasetName(args.dataset_name),
            graph_regime=GraphRegime(args.graph_regime),
            metric_name=MetricName(args.metric_name),
            metric_threshold=args.metric_threshold,
            population_name=PopulationName(args.population_name),
            validation_ground_truth_label=[int(item) for item in _parse_json_array(args.ground_truth_labels)],
            validation_teacher_prob=[float(item) for item in _parse_json_array(args.teacher_probs)],
            code_git_sha=args.code_git_sha,
            report_path=args.report_path,
            f1_positive_threshold=args.f1_positive_threshold,
        )
    except (ValidationError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"teacher baseline gate failed: {exc}", file=sys.stderr)
        return 2
    if not report.passed:
        print("teacher baseline gate did not pass", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
