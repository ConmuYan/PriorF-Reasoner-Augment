"""Validation-only calibration helpers for generation-lane scores.

The generation lane emits a raw JSON ``score``.  These helpers fit a small
post-hoc calibration artifact on validation only, then apply that frozen map
without overwriting the raw score.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import sklearn.metrics
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, model_validator

from priorf_teacher.schema import DatasetName, GraphRegime, PopulationName

GEN_SCORE_CALIBRATION_SCHEMA_VERSION = "gen_score_calibration/v1"
GEN_SCORE_CALIBRATION_FEATURE_SET = ("raw_gen_score_bin",)
LOW_BIN = "low"
HIGH_BIN = "high"

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_HEX64_PATTERN = r"^[0-9a-fA-F]{64}$"


class GenScoreCalibrationBin(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    bin_name: Literal["low", "high"]
    raw_score_min_inclusive: StrictFloat = Field(ge=0.0, le=1.0)
    raw_score_max_exclusive: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    raw_score_max_inclusive: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    n_total: StrictInt = Field(ge=0)
    n_positive: StrictInt = Field(ge=0)
    empirical_positive_rate: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    calibrated_probability: StrictFloat = Field(ge=0.0, le=1.0)
    used_global_fallback: bool

    @model_validator(mode="after")
    def _counts_consistent(self) -> "GenScoreCalibrationBin":
        if self.n_positive > self.n_total:
            raise ValueError("n_positive cannot exceed n_total")
        if self.n_total == 0 and self.empirical_positive_rate is not None:
            raise ValueError("empty calibration bins must not carry empirical_positive_rate")
        if self.n_total > 0 and self.empirical_positive_rate is None:
            raise ValueError("non-empty calibration bins require empirical_positive_rate")
        if (self.raw_score_max_exclusive is None) == (self.raw_score_max_inclusive is None):
            raise ValueError("exactly one raw score upper bound must be present")
        return self


class GenScoreCalibrationArtifact(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    schema_version: Literal["gen_score_calibration/v1"] = GEN_SCORE_CALIBRATION_SCHEMA_VERSION
    dataset_name: DatasetName
    graph_regime: GraphRegime
    source_population_name: Literal["validation"]
    fit_split_hash: StrictStr = Field(pattern=_HEX64_PATTERN)
    feature_set: tuple[Literal["raw_gen_score_bin"], ...]
    raw_score_threshold: StrictFloat = Field(default=0.5, gt=0.0, lt=1.0)
    smoothing_alpha: StrictFloat = Field(default=1.0, ge=0.0)
    min_count: StrictInt = Field(default=1, ge=0)
    n_total: StrictInt = Field(ge=1)
    n_positive: StrictInt = Field(ge=0)
    n_negative: StrictInt = Field(ge=0)
    global_empirical_positive_rate: StrictFloat = Field(ge=0.0, le=1.0)
    global_smoothed_positive_rate: StrictFloat = Field(ge=0.0, le=1.0)
    bins: tuple[GenScoreCalibrationBin, GenScoreCalibrationBin]
    oof_metrics: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _artifact_consistent(self) -> "GenScoreCalibrationArtifact":
        if self.n_positive + self.n_negative != self.n_total:
            raise ValueError("n_positive + n_negative must equal n_total")
        if self.feature_set != GEN_SCORE_CALIBRATION_FEATURE_SET:
            raise ValueError("feature_set must be raw_gen_score_bin for v1")
        names = tuple(bin.bin_name for bin in self.bins)
        if names != (LOW_BIN, HIGH_BIN):
            raise ValueError("bins must be ordered as low, high")
        if sum(bin.n_total for bin in self.bins) != self.n_total:
            raise ValueError("bin totals must sum to n_total")
        if sum(bin.n_positive for bin in self.bins) != self.n_positive:
            raise ValueError("bin positives must sum to n_positive")
        return self


def split_hash(node_ids: Sequence[int]) -> str:
    payload = json.dumps([int(node_id) for node_id in node_ids], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_bin_calibration(
    *,
    dataset_name: DatasetName | str,
    graph_regime: GraphRegime | str,
    source_population_name: PopulationName | str,
    labels: Sequence[int],
    raw_gen_scores: Sequence[float],
    node_ids: Sequence[int],
    smoothing_alpha: float = 1.0,
    min_count: int = 1,
    raw_score_threshold: float = 0.5,
    oof_metrics: dict[str, Any] | None = None,
) -> GenScoreCalibrationArtifact:
    population = PopulationName(source_population_name)
    if population != PopulationName.VALIDATION:
        raise ValueError("generation score calibration may be fit on validation only")
    if len(labels) != len(raw_gen_scores) or len(labels) != len(node_ids):
        raise ValueError("labels, raw_gen_scores, and node_ids must have equal length")
    if not labels:
        raise ValueError("calibration fit requires at least one validation row")
    if smoothing_alpha < 0:
        raise ValueError("smoothing_alpha must be non-negative")
    if min_count < 0:
        raise ValueError("min_count must be non-negative")
    if not 0.0 < raw_score_threshold < 1.0:
        raise ValueError("raw_score_threshold must be in (0, 1)")

    y = [int(label) for label in labels]
    scores = [float(score) for score in raw_gen_scores]
    if any(label not in (0, 1) for label in y):
        raise ValueError("labels must be 0 or 1")
    if any((not math.isfinite(score)) or score < 0.0 or score > 1.0 for score in scores):
        raise ValueError("raw_gen_scores must be finite probabilities in [0, 1]")

    n_total = len(y)
    n_positive = sum(y)
    global_smoothed = _smoothed_probability(
        positive=n_positive,
        total=n_total,
        smoothing_alpha=float(smoothing_alpha),
    )

    bins: list[GenScoreCalibrationBin] = []
    for bin_name in (LOW_BIN, HIGH_BIN):
        indices = [
            idx
            for idx, score in enumerate(scores)
            if (score < raw_score_threshold if bin_name == LOW_BIN else score >= raw_score_threshold)
        ]
        total = len(indices)
        positive = sum(y[idx] for idx in indices)
        empirical = None if total == 0 else positive / total
        use_global = total < int(min_count)
        calibrated = global_smoothed if use_global else _smoothed_probability(
            positive=positive,
            total=total,
            smoothing_alpha=float(smoothing_alpha),
        )
        bins.append(
            GenScoreCalibrationBin(
                bin_name=bin_name,
                raw_score_min_inclusive=0.0 if bin_name == LOW_BIN else raw_score_threshold,
                raw_score_max_exclusive=raw_score_threshold if bin_name == LOW_BIN else None,
                raw_score_max_inclusive=1.0 if bin_name == HIGH_BIN else None,
                n_total=total,
                n_positive=positive,
                empirical_positive_rate=empirical,
                calibrated_probability=calibrated,
                used_global_fallback=use_global,
            )
        )

    return GenScoreCalibrationArtifact(
        dataset_name=DatasetName(dataset_name),
        graph_regime=GraphRegime(graph_regime),
        source_population_name="validation",
        fit_split_hash=split_hash(node_ids),
        feature_set=GEN_SCORE_CALIBRATION_FEATURE_SET,
        raw_score_threshold=float(raw_score_threshold),
        smoothing_alpha=float(smoothing_alpha),
        min_count=int(min_count),
        n_total=n_total,
        n_positive=n_positive,
        n_negative=n_total - n_positive,
        global_empirical_positive_rate=n_positive / n_total,
        global_smoothed_positive_rate=global_smoothed,
        bins=tuple(bins),  # type: ignore[arg-type]
        oof_metrics=oof_metrics,
    )


def apply_bin_calibration(
    *,
    artifact: GenScoreCalibrationArtifact,
    raw_gen_scores: Sequence[float],
) -> tuple[float, ...]:
    low = artifact.bins[0].calibrated_probability
    high = artifact.bins[1].calibrated_probability
    threshold = float(artifact.raw_score_threshold)
    calibrated: list[float] = []
    for raw_score in raw_gen_scores:
        score = float(raw_score)
        if not math.isfinite(score) or score < 0.0 or score > 1.0:
            raise ValueError("raw_gen_scores must be finite probabilities in [0, 1]")
        calibrated.append(float(high if score >= threshold else low))
    return tuple(calibrated)


def load_calibration_artifact(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> GenScoreCalibrationArtifact:
    artifact_path = Path(path)
    if expected_sha256 is not None:
        actual = file_sha256(artifact_path)
        if actual != expected_sha256:
            raise ValueError(f"gen score calibration artifact hash mismatch: expected {expected_sha256}, got {actual}")
    return GenScoreCalibrationArtifact.model_validate(json.loads(artifact_path.read_text(encoding="utf-8")))


def write_calibration_artifact(artifact: GenScoreCalibrationArtifact, path: str | Path) -> str:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return file_sha256(artifact_path)


def calibration_metric_bundle(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float | None]:
    y = np.asarray([int(label) for label in labels], dtype=np.int64)
    p = np.asarray([float(score) for score in scores], dtype=np.float64)
    if y.shape[0] != p.shape[0]:
        raise ValueError("labels and scores must have equal length")
    if y.shape[0] == 0:
        raise ValueError("metric bundle requires at least one row")
    if not np.all(np.isfinite(p)):
        raise ValueError("scores contain non-finite values")
    payload: dict[str, float | None] = {
        "brier_score": float(sklearn.metrics.brier_score_loss(y, p)),
        "log_loss": float(sklearn.metrics.log_loss(y, p, labels=[0, 1])),
        "ece_2bin": _expected_calibration_error(y, p, bins=(0.0, 0.5, 1.0)),
        "ece_10bin": _expected_calibration_error(y, p, bins=tuple(float(v) for v in np.linspace(0.0, 1.0, 11))),
    }
    if len(set(y.tolist())) == 2:
        payload["auroc"] = float(sklearn.metrics.roc_auc_score(y, p))
        payload["auprc"] = float(sklearn.metrics.average_precision_score(y, p))
    else:
        payload["auroc"] = None
        payload["auprc"] = None
    return payload


def fit_oof_bin_calibration_metrics(
    *,
    dataset_name: DatasetName | str,
    graph_regime: GraphRegime | str,
    labels: Sequence[int],
    raw_gen_scores: Sequence[float],
    node_ids: Sequence[int],
    smoothing_alpha: float = 1.0,
    min_count: int = 1,
    n_splits: int = 5,
) -> dict[str, Any]:
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray([int(label) for label in labels], dtype=np.int64)
    if len(set(y.tolist())) != 2:
        return {
            "available": False,
            "reason": "out-of-fold calibration metrics require both classes",
            "raw_metrics": calibration_metric_bundle(labels, raw_gen_scores),
        }
    split_count = min(int(n_splits), int(np.bincount(y).min()), len(y))
    if split_count < 2:
        return {
            "available": False,
            "reason": "not enough minority-class examples for at least two stratified folds",
            "raw_metrics": calibration_metric_bundle(labels, raw_gen_scores),
        }
    calibrated: list[float | None] = [None] * len(y)
    folds: list[dict[str, Any]] = []
    splitter = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=0)
    raw_scores = [float(score) for score in raw_gen_scores]
    node_id_values = [int(node_id) for node_id in node_ids]
    for fold, (train_index, test_index) in enumerate(splitter.split(np.zeros(len(y)), y)):
        artifact = fit_bin_calibration(
            dataset_name=dataset_name,
            graph_regime=graph_regime,
            source_population_name=PopulationName.VALIDATION,
            labels=[int(y[idx]) for idx in train_index],
            raw_gen_scores=[raw_scores[idx] for idx in train_index],
            node_ids=[node_id_values[idx] for idx in train_index],
            smoothing_alpha=smoothing_alpha,
            min_count=min_count,
        )
        fold_scores = apply_bin_calibration(
            artifact=artifact,
            raw_gen_scores=[raw_scores[idx] for idx in test_index],
        )
        for idx, score in zip(test_index, fold_scores, strict=True):
            calibrated[int(idx)] = float(score)
        folds.append(
            {
                "fold": int(fold),
                "train_n": int(len(train_index)),
                "test_n": int(len(test_index)),
                "fit_split_hash": artifact.fit_split_hash,
                "bins": [bin.model_dump(mode="json") for bin in artifact.bins],
            }
        )
    calibrated_scores = [float(score) for score in calibrated if score is not None]
    return {
        "available": True,
        "n_splits": int(split_count),
        "raw_metrics": calibration_metric_bundle(labels, raw_gen_scores),
        "calibrated_oof_metrics": calibration_metric_bundle(labels, calibrated_scores),
        "folds": folds,
    }


def _smoothed_probability(*, positive: int, total: int, smoothing_alpha: float) -> float:
    if total < 0 or positive < 0 or positive > total:
        raise ValueError("invalid bin counts")
    denominator = total + 2.0 * smoothing_alpha
    if denominator == 0:
        raise ValueError("smoothing_alpha=0 requires non-empty bins")
    return float((positive + smoothing_alpha) / denominator)


def _expected_calibration_error(y: np.ndarray, p: np.ndarray, *, bins: tuple[float, ...]) -> float:
    total = int(y.shape[0])
    value = 0.0
    for left, right in zip(bins[:-1], bins[1:], strict=True):
        if right == 1.0:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        count = int(mask.sum())
        if count:
            value += (count / total) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(value)
