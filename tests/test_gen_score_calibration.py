from __future__ import annotations

import pytest

from eval.gen_score_calibration import (
    GEN_SCORE_CALIBRATION_SCHEMA_VERSION,
    apply_bin_calibration,
    calibration_metric_bundle,
    fit_bin_calibration,
    fit_oof_bin_calibration_metrics,
)
from priorf_teacher.schema import DatasetName, GraphRegime, PopulationName


def test_gen_score_calibration_fit_requires_validation_population() -> None:
    with pytest.raises(ValueError, match="validation only"):
        fit_bin_calibration(
            dataset_name=DatasetName.AMAZON,
            graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
            source_population_name=PopulationName.FINAL_TEST,
            labels=(0, 1),
            raw_gen_scores=(0.05, 0.95),
            node_ids=(1, 2),
        )


def test_bin_calibration_smooths_extreme_bins_and_applies_without_overwriting_raw() -> None:
    artifact = fit_bin_calibration(
        dataset_name=DatasetName.AMAZON,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        source_population_name=PopulationName.VALIDATION,
        labels=(0, 0, 1, 1),
        raw_gen_scores=(0.05, 0.05, 0.95, 0.95),
        node_ids=(1, 2, 3, 4),
        smoothing_alpha=1.0,
        min_count=1,
    )

    assert artifact.schema_version == GEN_SCORE_CALIBRATION_SCHEMA_VERSION
    assert artifact.bins[0].calibrated_probability == pytest.approx(0.25)
    assert artifact.bins[1].calibrated_probability == pytest.approx(0.75)
    raw_scores = (0.05, 0.95)

    calibrated = apply_bin_calibration(artifact=artifact, raw_gen_scores=raw_scores)

    assert raw_scores == (0.05, 0.95)
    assert calibrated == pytest.approx((0.25, 0.75))


def test_bin_calibration_min_count_uses_global_fallback() -> None:
    artifact = fit_bin_calibration(
        dataset_name="amazon",
        graph_regime="transductive_standard",
        source_population_name="validation",
        labels=(0, 0, 1),
        raw_gen_scores=(0.05, 0.05, 0.95),
        node_ids=(1, 2, 3),
        smoothing_alpha=1.0,
        min_count=2,
    )

    assert artifact.bins[1].used_global_fallback is True
    assert artifact.bins[1].calibrated_probability == pytest.approx(2 / 5)


def test_oof_calibration_metrics_are_computed() -> None:
    payload = fit_oof_bin_calibration_metrics(
        dataset_name="amazon",
        graph_regime="transductive_standard",
        labels=(0, 0, 0, 0, 1, 1, 1, 1),
        raw_gen_scores=(0.05, 0.05, 0.95, 0.05, 0.95, 0.95, 0.05, 0.95),
        node_ids=tuple(range(8)),
        n_splits=2,
    )

    assert payload["available"] is True
    assert payload["n_splits"] == 2
    assert "raw_metrics" in payload
    assert "calibrated_oof_metrics" in payload
    assert payload["calibrated_oof_metrics"]["brier_score"] is not None


def test_fit_artifact_accepts_structured_oof_metrics() -> None:
    oof_metrics = {
        "available": True,
        "raw_metrics": {"brier_score": 0.25, "log_loss": 0.7},
        "calibrated_oof_metrics": {"brier_score": 0.2, "log_loss": 0.6},
        "folds": [{"fold": 0, "train_n": 2, "test_n": 2}],
    }

    artifact = fit_bin_calibration(
        dataset_name="amazon",
        graph_regime="transductive_standard",
        source_population_name="validation",
        labels=(0, 0, 1, 1),
        raw_gen_scores=(0.05, 0.05, 0.95, 0.95),
        node_ids=(1, 2, 3, 4),
        oof_metrics=oof_metrics,
    )

    assert artifact.oof_metrics == oof_metrics


def test_calibration_metric_bundle_includes_log_loss_and_ece() -> None:
    metrics = calibration_metric_bundle((0, 1), (0.1, 0.9))

    assert metrics["brier_score"] == pytest.approx(0.01)
    assert metrics["log_loss"] is not None
    assert metrics["ece_2bin"] is not None
