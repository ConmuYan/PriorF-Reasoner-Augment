from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from graph_data.manifests import DataArtifact, DataManifest, PopulationMetadata, write_data_manifest
from graph_data.validators import compute_file_sha256, compute_node_ids_hash
from priorf_teacher.export_pipeline import read_teacher_export_artifact, write_teacher_export_artifact
from priorf_teacher.schema import (
    DatasetName,
    GraphRegime,
    NeighborSummary,
    PopulationName,
    RelationProfile,
    TeacherBaselineReport,
    TeacherExportManifest,
    TeacherExportRecord,
    TeacherProvenance,
)

GIT_SHA = "a" * 40
CKPT_SHA = "b" * 64
MANIFEST_SHA_PLACEHOLDER = "c" * 64
NODE_HASH = compute_node_ids_hash(np.array([0, 1], dtype=np.int64))
UTC_NOW = datetime(2026, 4, 22, 7, 0, 0, tzinfo=timezone.utc)


def _population(name: PopulationName = PopulationName.VALIDATION, node_hash: str = NODE_HASH) -> PopulationMetadata:
    return PopulationMetadata(
        population_name=name.value,
        split_values=(name.value,),
        node_ids_hash=node_hash,
        contains_tuning_rows=name in {PopulationName.TRAIN, PopulationName.VALIDATION},
        contains_final_test_rows=name == PopulationName.FINAL_TEST,
    )


def _write_data_manifest(tmp_path: Path, *, graph_regime: str = GraphRegime.TRANSDUCTIVE_STANDARD.value) -> Path:
    source_path = tmp_path / "source.mat"
    source_path.write_text("source", encoding="utf-8")
    manifest = DataManifest(
        dataset_name=DatasetName.AMAZON.value,
        graph_regime=graph_regime,
        feature_dim=25,
        relation_count=3,
        num_nodes=2,
        populations=(_population(),),
        artifacts=(DataArtifact(kind="source_mat", path=str(source_path), sha256=compute_file_sha256(source_path)),),
    )
    manifest_path = tmp_path / "data_manifest.json"
    write_data_manifest(manifest, manifest_path)
    return manifest_path


def _baseline_report(
    tmp_path: Path,
    data_manifest_path: Path,
    *,
    passed: bool = True,
    graph_regime: GraphRegime = GraphRegime.TRANSDUCTIVE_STANDARD,
    data_manifest_sha: str | None = None,
) -> Path:
    report = TeacherBaselineReport(
        dataset_name=DatasetName.AMAZON,
        teacher_model_name="PriorF-GNN",
        teacher_checkpoint_sha256=CKPT_SHA,
        graph_regime=graph_regime,
        population_name=PopulationName.VALIDATION,
        metric_name="auroc",
        metric_value=0.91 if passed else 0.41,
        threshold=0.90,
        passed=passed,
        data_manifest_sha256=data_manifest_sha or compute_file_sha256(data_manifest_path),
        code_git_sha=GIT_SHA,
        export_timestamp_utc=UTC_NOW,
    )
    path = tmp_path / "teacher_baseline_report.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def _provenance(data_manifest_path: Path, *, graph_regime: GraphRegime = GraphRegime.TRANSDUCTIVE_STANDARD) -> TeacherProvenance:
    return TeacherProvenance(
        code_git_sha=GIT_SHA,
        teacher_checkpoint_path="checkpoints/priorf.pt",
        teacher_checkpoint_sha256=CKPT_SHA,
        data_manifest_path=str(data_manifest_path),
        data_manifest_sha256=compute_file_sha256(data_manifest_path),
        export_timestamp_utc=UTC_NOW,
        random_seed=7,
        graph_regime=graph_regime,
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


def _record(**overrides) -> TeacherExportRecord:
    payload = {
        "dataset_name": DatasetName.AMAZON,
        "teacher_model_name": "PriorF-GNN",
        "teacher_checkpoint": "checkpoints/priorf.pt",
        "population_name": PopulationName.VALIDATION,
        "node_id": 0,
        "ground_truth_label": 1,
        "teacher_prob": 0.92,
        "teacher_logit": 2.1,
        "hsd": 0.3,
        "hsd_quantile": 0.8,
        "asda_switch": True,
        "mlp_logit": 1.7,
        "gnn_logit": 2.2,
        "branch_gap": 0.5,
        "relation_profile": _relation_profile(),
        "neighbor_summary": _neighbor_summary(),
        "high_hsd_flag": True,
        "graph_regime": GraphRegime.TRANSDUCTIVE_STANDARD,
    }
    payload.update(overrides)
    return TeacherExportRecord.model_validate(payload)


def _export_manifest(data_manifest_path: Path, **overrides) -> TeacherExportManifest:
    payload = {
        "dataset_name": DatasetName.AMAZON,
        "population_name": PopulationName.VALIDATION,
        "graph_regime": GraphRegime.TRANSDUCTIVE_STANDARD,
        "row_count": 1,
        "node_ids_hash": NODE_HASH,
        "split_values": (PopulationName.VALIDATION.value,),
        "contains_tuning_rows": True,
        "contains_final_test_rows": False,
        "provenance": _provenance(data_manifest_path),
        "schema_version": "teacher_export/v1",
    }
    payload.update(overrides)
    return TeacherExportManifest.model_validate(payload)


def _output_dir(tmp_path: Path) -> Path:
    return tmp_path / "outputs" / "gated" / "teacher_exports" / "amazon" / "validation"


@pytest.mark.parametrize("bad_regime", ["test", "standard", ""])
def test_unknown_graph_regime_rejected(bad_regime):
    payload = _record().model_dump(mode="json")
    payload["graph_regime"] = bad_regime
    with pytest.raises(ValidationError):
        TeacherExportRecord.model_validate(payload)


def test_illegal_population_name_rejected():
    payload = _record().model_dump(mode="json")
    payload["population_name"] = "test"
    with pytest.raises(ValidationError):
        TeacherExportRecord.model_validate(payload)


def test_extra_field_rejected():
    payload = _record().model_dump(mode="json")
    payload["extra"] = "forbidden"
    with pytest.raises(ValidationError):
        TeacherExportRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [("teacher_prob", 1.1), ("teacher_prob", -0.1), ("hsd_quantile", 1.2), ("node_id", -1)],
)
def test_teacher_export_record_value_ranges_rejected(field, value):
    payload = _record().model_dump(mode="json")
    payload[field] = value
    with pytest.raises(ValidationError):
        TeacherExportRecord.model_validate(payload)


def test_teacher_export_manifest_requires_full_provenance(tmp_path):
    data_manifest_path = _write_data_manifest(tmp_path)
    payload = _export_manifest(data_manifest_path).model_dump(mode="json")
    del payload["provenance"]["code_git_sha"]
    with pytest.raises(ValidationError):
        TeacherExportManifest.model_validate(payload)


@pytest.mark.parametrize("timestamp", [datetime(2026, 4, 22), datetime(2026, 4, 22, tzinfo=timezone(timedelta(hours=8)))])
def test_export_timestamp_must_be_utc_tz_aware(timestamp):
    payload = {
        "code_git_sha": GIT_SHA,
        "teacher_checkpoint_path": "checkpoints/priorf.pt",
        "teacher_checkpoint_sha256": CKPT_SHA,
        "data_manifest_path": "manifest.json",
        "data_manifest_sha256": MANIFEST_SHA_PLACEHOLDER,
        "export_timestamp_utc": timestamp,
        "random_seed": 7,
        "graph_regime": GraphRegime.TRANSDUCTIVE_STANDARD,
    }
    with pytest.raises(ValidationError):
        TeacherProvenance.model_validate(payload)


def test_label_field_rejected_by_schema_and_artifact_reader(tmp_path):
    payload = _record().model_dump(mode="json")
    payload["label"] = 1
    with pytest.raises(ValidationError):
        TeacherExportRecord.model_validate(payload)

    artifact_path = tmp_path / "bad_label.parquet"
    pq.write_table(pa.Table.from_pylist([{**_record().model_dump(mode="json"), "label": 1}]), artifact_path)
    with pytest.raises(ValidationError):
        read_teacher_export_artifact(artifact_path)


def test_export_pipeline_rejects_two_populations_in_one_artifact(tmp_path):
    data_manifest_path = _write_data_manifest(tmp_path)
    report_path = _baseline_report(tmp_path, data_manifest_path)
    manifest = _export_manifest(data_manifest_path)
    records = [_record(), _record(population_name=PopulationName.TRAIN, node_id=1)]

    with pytest.raises(ValueError, match="one population"):
        write_teacher_export_artifact(
            data_manifest_path=data_manifest_path,
            teacher_baseline_report_path=report_path,
            output_dir=_output_dir(tmp_path),
            export_manifest=manifest,
            records=records,
        )


def test_export_pipeline_rejects_failed_baseline_report(tmp_path):
    data_manifest_path = _write_data_manifest(tmp_path)
    report_path = _baseline_report(tmp_path, data_manifest_path, passed=False)

    with pytest.raises(ValueError, match="passed"):
        write_teacher_export_artifact(
            data_manifest_path=data_manifest_path,
            teacher_baseline_report_path=report_path,
            output_dir=_output_dir(tmp_path),
            export_manifest=_export_manifest(data_manifest_path),
            records=[_record()],
        )


def test_export_pipeline_rejects_manifest_sha_mismatch(tmp_path):
    data_manifest_path = _write_data_manifest(tmp_path)
    report_path = _baseline_report(tmp_path, data_manifest_path, data_manifest_sha="d" * 64)

    with pytest.raises(ValueError, match="data_manifest_sha256"):
        write_teacher_export_artifact(
            data_manifest_path=data_manifest_path,
            teacher_baseline_report_path=report_path,
            output_dir=_output_dir(tmp_path),
            export_manifest=_export_manifest(data_manifest_path),
            records=[_record()],
        )


def test_export_pipeline_rejects_graph_regime_mismatch(tmp_path):
    data_manifest_path = _write_data_manifest(tmp_path)
    report_path = _baseline_report(tmp_path, data_manifest_path, graph_regime=GraphRegime.INDUCTIVE_MASKED)

    with pytest.raises(ValueError, match="graph_regime"):
        write_teacher_export_artifact(
            data_manifest_path=data_manifest_path,
            teacher_baseline_report_path=report_path,
            output_dir=_output_dir(tmp_path),
            export_manifest=_export_manifest(data_manifest_path),
            records=[_record()],
        )


def test_export_pipeline_rejects_record_graph_regime_mismatch(tmp_path):
    data_manifest_path = _write_data_manifest(tmp_path)
    report_path = _baseline_report(tmp_path, data_manifest_path)

    with pytest.raises(ValueError, match="graph_regime"):
        write_teacher_export_artifact(
            data_manifest_path=data_manifest_path,
            teacher_baseline_report_path=report_path,
            output_dir=_output_dir(tmp_path),
            export_manifest=_export_manifest(data_manifest_path),
            records=[_record(graph_regime=GraphRegime.INDUCTIVE_MASKED)],
        )


def test_export_pipeline_rejects_output_dir_outside_gated_teacher_exports(tmp_path):
    data_manifest_path = _write_data_manifest(tmp_path)
    report_path = _baseline_report(tmp_path, data_manifest_path)

    with pytest.raises(ValueError, match="output_dir"):
        write_teacher_export_artifact(
            data_manifest_path=data_manifest_path,
            teacher_baseline_report_path=report_path,
            output_dir=tmp_path / "outputs" / "diagnostic" / "teacher_exports" / "amazon" / "validation",
            export_manifest=_export_manifest(data_manifest_path),
            records=[_record()],
        )


def test_export_pipeline_reread_row_count_mismatch_raises(tmp_path):
    data_manifest_path = _write_data_manifest(tmp_path)
    report_path = _baseline_report(tmp_path, data_manifest_path)
    manifest = _export_manifest(data_manifest_path, row_count=2)

    with pytest.raises(RuntimeError, match="row_count"):
        write_teacher_export_artifact(
            data_manifest_path=data_manifest_path,
            teacher_baseline_report_path=report_path,
            output_dir=_output_dir(tmp_path),
            export_manifest=manifest,
            records=[_record()],
        )
