from __future__ import annotations

import numpy as np
import pytest

from eval.temperature_scaling import apply_temperature_to_probs, fit_temperature_on_validation, logits_from_probs


def _binary_nll(probs: tuple[float, ...], labels: tuple[int, ...]) -> float:
    probs_np = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    labels_np = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(labels_np * np.log(probs_np) + (1.0 - labels_np) * np.log1p(-probs_np)))


def test_logits_from_probs_roundtrip_through_temperature_one() -> None:
    probs = (0.1, 0.25, 0.75, 0.9)

    logits = logits_from_probs(probs=probs)
    recovered = apply_temperature_to_probs(probs=probs, temperature=1.0)

    assert len(logits) == len(probs)
    assert recovered == pytest.approx(probs)


def test_fit_temperature_on_validation_reduces_nll_for_overconfident_probs() -> None:
    probs = (0.001, 0.02, 0.98, 0.999)
    labels = (0, 1, 1, 0)

    baseline_nll = _binary_nll(probs, labels)
    temperature = fit_temperature_on_validation(
        logits=logits_from_probs(probs=probs),
        labels=labels,
    )
    calibrated_probs = apply_temperature_to_probs(probs=probs, temperature=temperature)
    calibrated_nll = _binary_nll(calibrated_probs, labels)

    assert 0.1 <= temperature <= 10.0
    assert calibrated_nll <= baseline_nll


def test_fit_temperature_requires_both_classes() -> None:
    probs = (0.2, 0.3, 0.4)
    labels = (0, 0, 0)

    with pytest.raises(ValueError, match="requires both positive and negative labels"):
        fit_temperature_on_validation(
            logits=logits_from_probs(probs=probs),
            labels=labels,
        )
