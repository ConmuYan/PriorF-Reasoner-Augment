from __future__ import annotations

from math import isfinite, log

import numpy as np

__all__ = (
    "apply_temperature_to_probs",
    "fit_temperature_on_validation",
    "logits_from_probs",
)

_MIN_TEMPERATURE = 0.1
_MAX_TEMPERATURE = 10.0
_DEFAULT_EPS = 1e-6


def _validate_probs(*, probs, eps: float) -> np.ndarray:
    probs_np = np.asarray(tuple(probs), dtype=np.float64)
    if probs_np.ndim != 1 or probs_np.shape[0] < 1:
        raise ValueError("probs must be a non-empty 1-D array")
    if not np.all(np.isfinite(probs_np)):
        raise ValueError("probs must be finite")
    if np.any(probs_np < 0.0) or np.any(probs_np > 1.0):
        raise ValueError("probs must lie in [0.0, 1.0]")
    if not isfinite(eps) or eps <= 0.0 or eps >= 0.5:
        raise ValueError("eps must be finite and within (0.0, 0.5)")
    return probs_np


def _validate_logits_and_labels(*, logits, labels) -> tuple[np.ndarray, np.ndarray]:
    logits_np = np.asarray(tuple(logits), dtype=np.float64)
    labels_np = np.asarray(tuple(labels), dtype=np.int64)
    if logits_np.ndim != 1 or logits_np.shape[0] < 1:
        raise ValueError("logits must be a non-empty 1-D array")
    if labels_np.ndim != 1 or labels_np.shape[0] != logits_np.shape[0]:
        raise ValueError("labels must be a 1-D array with the same length as logits")
    if not np.all(np.isfinite(logits_np)):
        raise ValueError("logits must be finite")
    if np.any((labels_np != 0) & (labels_np != 1)):
        raise ValueError("labels must be binary 0/1")
    if np.unique(labels_np).shape[0] < 2:
        raise ValueError("temperature fitting requires both positive and negative labels")
    return logits_np, labels_np.astype(np.float64)


def logits_from_probs(*, probs, eps: float = _DEFAULT_EPS) -> tuple[float, ...]:
    probs_np = _validate_probs(probs=probs, eps=eps)
    clipped = np.clip(probs_np, eps, 1.0 - eps)
    logits_np = np.log(clipped) - np.log1p(-clipped)
    return tuple(float(value) for value in logits_np.tolist())


def apply_temperature_to_probs(
    *,
    probs,
    temperature: float,
    eps: float = _DEFAULT_EPS,
) -> tuple[float, ...]:
    if not isfinite(temperature) or temperature < _MIN_TEMPERATURE or temperature > _MAX_TEMPERATURE:
        raise ValueError("temperature must be finite and within [0.1, 10.0]")
    logits_np = np.asarray(logits_from_probs(probs=probs, eps=eps), dtype=np.float64)
    scaled_logits = np.clip(logits_np / float(temperature), -50.0, 50.0)
    scaled_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    return tuple(float(value) for value in scaled_probs.tolist())


def _binary_nll(*, logits_np: np.ndarray, labels_np: np.ndarray, temperature: float) -> float:
    scaled_logits = np.clip(logits_np / float(temperature), -50.0, 50.0)
    probs_np = 1.0 / (1.0 + np.exp(-scaled_logits))
    probs_np = np.clip(probs_np, _DEFAULT_EPS, 1.0 - _DEFAULT_EPS)
    loss = -np.mean(labels_np * np.log(probs_np) + (1.0 - labels_np) * np.log1p(-probs_np))
    return float(loss)


def fit_temperature_on_validation(
    *,
    logits,
    labels,
    min_temperature: float = _MIN_TEMPERATURE,
    max_temperature: float = _MAX_TEMPERATURE,
    num_candidates: int = 401,
) -> float:
    if not isfinite(min_temperature) or not isfinite(max_temperature):
        raise ValueError("temperature bounds must be finite")
    if min_temperature <= 0.0 or max_temperature < min_temperature:
        raise ValueError("temperature bounds must satisfy 0 < min <= max")
    if num_candidates < 3:
        raise ValueError("num_candidates must be >= 3")

    logits_np, labels_np = _validate_logits_and_labels(logits=logits, labels=labels)
    candidate_temperatures = np.geomspace(float(min_temperature), float(max_temperature), int(num_candidates))
    candidate_temperatures = np.unique(
        np.concatenate([candidate_temperatures, np.asarray([1.0], dtype=np.float64)])
    )

    best_temperature: float | None = None
    best_key: tuple[float, float, float] | None = None
    for value in candidate_temperatures.tolist():
        temperature = float(value)
        loss = _binary_nll(logits_np=logits_np, labels_np=labels_np, temperature=temperature)
        key = (loss, abs(log(temperature)), temperature)
        if best_key is None or key < best_key:
            best_key = key
            best_temperature = temperature

    assert best_temperature is not None
    return float(best_temperature)
