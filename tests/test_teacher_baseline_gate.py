from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from graph_data.manifests import DataArtifact, DataManifest, PopulationMetadata, write_data_manifest
from graph_data.validators import compute_file_sha256
from priorf_teacher.schema import DatasetName, GraphRegime, MetricName, PopulationName, TeacherBaselineReport
from priorf_teacher.teacher_baseline_gate import main, run_teacher_baseline_gate

GIT_SHA = "a" * 40
CKPT_SHA = "b" * 64
NODE_HASH = "c" * 64


def _write_data_manifest(tmp_path: Path, *, graph_regime: str = GraphRegime.TRANSDUCTIVE_STANDARD.value) -> Path:
    source_path = tmp_path / "source.mat"
    source_path.write_text("source", encoding="utf-8")
    manifest = DataManifest(
        dataset_name=DatasetName.AMAZON.value,
        graph_regime=graph_regime,
        feature_dim=25,
        relation_count=3,
        num_nodes=2,
        populations=(
            PopulationMetadata(
                population_name=PopulationName.VALIDATION.value,
                split_values=(PopulationName.VALIDATION.value,),
                node_ids_hash=NODE_HASH,
                contains_tuning_rows=True,
                contains_final_test_rows=False,
            ),
        ),
        artifacts=(DataArtifact(kind="source_mat", path=str(source_path), sha256=compute_file_sha256(source_path)),),
    )
    manifest_path = tmp_path / "data_manifest.json"
    write_data_manifest(manifest, manifest_path)
    return manifest_path


def _run_gate(tmp_path: Path, **overrides):
    data_manifest_path = overrides.pop("data_manifest_path", _write_data_manifest(tmp_path))
    payload = {
        "data_manifest_path": data_manifest_path,
        "teacher_checkpoint_path": "checkpoints/priorf.pt",
        "teacher_model_name": "PriorF-GNN",
        "teacher_checkpoint_sha256": CKPT_SHA,
        "dataset_name": DatasetName.AMAZON,
        "graph_regime": GraphRegime.TRANSDUCTIVE_STANDARD,
        "metric_name": MetricName.AUROC,
        "metric_threshold": 0.9,
        "population_name": PopulationName.VALIDATION,
        "validation_ground_truth_label": [0, 1],
        "validation_teacher_prob": [0.1, 0.9],
        "code_git_sha": GIT_SHA,
        "report_path": tmp_path / "teacher_report.json",
    }
    payload.update(overrides)
    return run_teacher_baseline_gate(**payload)


def _cli_args(tmp_path: Path, *, metric_threshold: float = 0.9, probs: str = "[0.1, 0.9]") -> list[str]:
    data_manifest_path = _write_data_manifest(tmp_path)
    return [
        "--data-manifest-path",
        str(data_manifest_path),
        "--teacher-checkpoint-path",
        "checkpoints/priorf.pt",
        "--teacher-model-name",
        "PriorF-GNN",
        "--teacher-checkpoint-sha256",
        CKPT_SHA,
        "--dataset-name",
        DatasetName.AMAZON.value,
        "--graph-regime",
        GraphRegime.TRANSDUCTIVE_STANDARD.value,
        "--metric-name",
        MetricName.AUROC.value,
        "--metric-threshold",
        str(metric_threshold),
        "--population-name",
        PopulationName.VALIDATION.value,
        "--ground-truth-labels",
        "[0, 1]",
        "--teacher-probs",
        probs,
        "--code-git-sha",
        GIT_SHA,
        "--report-path",
        str(tmp_path / "cli_report.json"),
    ]


def test_metric_at_or_above_threshold_passes_and_report_is_complete(tmp_path):
    report = _run_gate(tmp_path)

    assert report.passed is True
    assert report.metric_value == pytest.approx(1.0)
    assert report.threshold == 0.9
    assert report.population_name == PopulationName.VALIDATION
    assert report.data_manifest_sha256 == compute_file_sha256(tmp_path / "data_manifest.json")
    loaded = TeacherBaselineReport.model_validate_json((tmp_path / "teacher_report.json").read_text(encoding="utf-8"))
    assert loaded == report


def test_metric_below_threshold_returns_failed_report_and_cli_nonzero(tmp_path):
    report = _run_gate(tmp_path, validation_teacher_prob=[0.9, 0.1], metric_threshold=0.5)
    assert report.passed is False
    assert TeacherBaselineReport.model_validate_json((tmp_path / "teacher_report.json").read_text(encoding="utf-8")).passed is False

    exit_code = main(_cli_args(tmp_path, metric_threshold=0.5, probs="[0.9, 0.1]"))
    assert exit_code != 0
    cli_report = TeacherBaselineReport.model_validate_json((tmp_path / "cli_report.json").read_text(encoding="utf-8"))
    assert cli_report.passed is False


def test_population_other_than_validation_raises(tmp_path):
    with pytest.raises(ValueError, match="validation"):
        _run_gate(tmp_path, population_name=PopulationName.UNUSED_HOLDOUT)


def test_invalid_graph_regime_raises_validation_error(tmp_path):
    with pytest.raises(ValidationError):
        _run_gate(tmp_path, graph_regime="invalid")


def test_invalid_metric_name_raises_validation_error(tmp_path):
    with pytest.raises(ValidationError):
        _run_gate(tmp_path, metric_name="accuracy")


def test_missing_required_argument_raises(tmp_path):
    data_manifest_path = _write_data_manifest(tmp_path)
    with pytest.raises(TypeError):
        run_teacher_baseline_gate(
            data_manifest_path=data_manifest_path,
            teacher_checkpoint_path="checkpoints/priorf.pt",
            teacher_model_name="PriorF-GNN",
            teacher_checkpoint_sha256=CKPT_SHA,
            dataset_name=DatasetName.AMAZON,
            graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
            metric_name=MetricName.AUROC,
            metric_threshold=0.9,
            population_name=PopulationName.VALIDATION,
            validation_ground_truth_label=[0, 1],
            validation_teacher_prob=[0.1, 0.9],
            report_path=tmp_path / "report.json",
        )


def test_env_var_interference_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("TEACHER_BASELINE_THRESHOLD", "0.1")
    with pytest.raises(RuntimeError, match="environment"):
        _run_gate(tmp_path)


def test_teacher_checkpoint_sha256_missing_or_bad_format_rejected(tmp_path):
    with pytest.raises(ValidationError):
        _run_gate(tmp_path, teacher_checkpoint_sha256="")
    with pytest.raises(ValidationError):
        _run_gate(tmp_path, teacher_checkpoint_sha256="not-a-sha")


def test_report_data_manifest_sha_equals_manifest_sha(tmp_path):
    data_manifest_path = _write_data_manifest(tmp_path)
    report = _run_gate(tmp_path, data_manifest_path=data_manifest_path)
    assert report.data_manifest_sha256 == compute_file_sha256(data_manifest_path)


@pytest.mark.parametrize("timestamp", [datetime(2026, 4, 22), datetime(2026, 4, 22, tzinfo=timezone(timedelta(hours=8)))])
def test_report_timestamp_must_be_utc_tz_aware(timestamp):
    with pytest.raises(ValidationError):
        TeacherBaselineReport(
            dataset_name=DatasetName.AMAZON,
            teacher_model_name="PriorF-GNN",
            teacher_checkpoint_sha256=CKPT_SHA,
            graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
            population_name=PopulationName.VALIDATION,
            metric_name=MetricName.AUROC,
            metric_value=0.9,
            threshold=0.8,
            passed=True,
            data_manifest_sha256="d" * 64,
            code_git_sha=GIT_SHA,
            export_timestamp_utc=timestamp,
        )


def test_cli_nonzero_on_invalid_arguments_and_env_interference(tmp_path, monkeypatch):
    bad_args = _cli_args(tmp_path)
    bad_args[bad_args.index("--graph-regime") + 1] = "invalid"
    assert main(bad_args) != 0

    monkeypatch.setenv("PRIORF_GATE_THRESHOLD", "0.1")
    assert main(_cli_args(tmp_path)) != 0
