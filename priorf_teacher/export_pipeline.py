"""Validated teacher export read/write channel and fail-closed preflight.

Task 2 deliberately does not run teacher forward inference and does not accept
legacy assets/teacher_exports files as validated inputs.  Callers must provide
already-materialized TeacherExportRecord objects and a matching manifest.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from graph_data.manifests import DataManifest, PopulationMetadata, load_data_manifest
from graph_data.validators import compute_file_sha256
from priorf_teacher.schema import (
    DatasetName,
    PopulationName,
    TeacherBaselineReport,
    TeacherExportManifest,
    TeacherExportRecord,
)

_ARTIFACT_NAME: Final[str] = "teacher_export.parquet"
_MANIFEST_NAME: Final[str] = "teacher_export_manifest.json"
_OUTPUT_PREFIX_PARTS: Final[tuple[str, str, str]] = ("outputs", "gated", "teacher_exports")
_FORBIDDEN_OUTPUT_PARTS: Final[tuple[str, ...]] = ("assets", "formal", "diagnostic")


def run_teacher_export_preflight(
    *,
    data_manifest_path: str | Path,
    teacher_baseline_report_path: str | Path,
    output_dir: str | Path,
    export_manifest: TeacherExportManifest,
) -> None:
    """Validate launch-time teacher export preconditions without writing records."""

    data_manifest, baseline_report = _load_and_validate_inputs(data_manifest_path, teacher_baseline_report_path)
    _validate_manifest_against_inputs(export_manifest, data_manifest, baseline_report, data_manifest_path)
    _ensure_output_dir_allowed(output_dir, export_manifest.dataset_name, export_manifest.population_name)


def write_teacher_export_artifact(
    *,
    data_manifest_path: str | Path,
    teacher_baseline_report_path: str | Path,
    output_dir: str | Path,
    export_manifest: TeacherExportManifest,
    records: tuple[TeacherExportRecord, ...] | list[TeacherExportRecord],
) -> tuple[Path, Path]:
    """Atomically write one validated teacher export artifact and manifest."""

    data_manifest, baseline_report = _load_and_validate_inputs(data_manifest_path, teacher_baseline_report_path)
    _validate_manifest_against_inputs(export_manifest, data_manifest, baseline_report, data_manifest_path)
    output_path = _ensure_output_dir_allowed(output_dir, export_manifest.dataset_name, export_manifest.population_name)
    validated_records = _validate_records_against_manifest(tuple(records), export_manifest)

    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / _ARTIFACT_NAME
    manifest_path = output_path / _MANIFEST_NAME
    tmp_artifact = output_path / f".{_ARTIFACT_NAME}.tmp"
    tmp_manifest = output_path / f".{_MANIFEST_NAME}.tmp"
    try:
        _write_records_parquet(validated_records, tmp_artifact)
        tmp_manifest.write_text(export_manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_artifact, artifact_path)
        os.replace(tmp_manifest, manifest_path)
    except Exception:
        _remove_if_exists(tmp_artifact)
        _remove_if_exists(tmp_manifest)
        _remove_if_exists(artifact_path)
        _remove_if_exists(manifest_path)
        raise

    try:
        reread_records = read_teacher_export_artifact(artifact_path)
        reread_manifest = read_teacher_export_manifest(manifest_path)
        _validate_reread_artifact(reread_records, reread_manifest)
    except Exception as exc:
        _remove_if_exists(artifact_path)
        _remove_if_exists(manifest_path)
        raise RuntimeError(f"teacher export artifact failed post-write validation: {exc}") from exc
    return artifact_path, manifest_path


def read_teacher_export_manifest(path: str | Path) -> TeacherExportManifest:
    manifest_path = Path(path)
    if _is_legacy_assets_path(manifest_path):
        raise ValueError("legacy assets/teacher_exports manifests are not validated Task 2 inputs")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    _reject_label_key(payload)
    return TeacherExportManifest.model_validate(payload)


def read_teacher_export_artifact(path: str | Path) -> tuple[TeacherExportRecord, ...]:
    artifact_path = Path(path)
    if _is_legacy_assets_path(artifact_path):
        raise ValueError("legacy assets/teacher_exports artifacts are not validated Task 2 inputs")
    table = pq.read_table(artifact_path)
    if "label" in table.schema.names:
        raise ValidationError.from_exception_data(
            "TeacherExportRecord",
            [{"type": "extra_forbidden", "loc": ("label",), "input": "label"}],
        )
    records = []
    for row in table.to_pylist():
        _reject_label_key(row)
        records.append(TeacherExportRecord.model_validate(row))
    return tuple(records)


def _load_and_validate_inputs(
    data_manifest_path: str | Path,
    teacher_baseline_report_path: str | Path,
) -> tuple[DataManifest, TeacherBaselineReport]:
    manifest = load_data_manifest(data_manifest_path)
    report_payload = json.loads(Path(teacher_baseline_report_path).read_text(encoding="utf-8"))
    _reject_label_key(report_payload)
    report = TeacherBaselineReport.model_validate(report_payload)
    if report.passed is not True:
        raise ValueError("teacher export pipeline requires a passed TeacherBaselineReport")
    manifest_sha = compute_file_sha256(data_manifest_path)
    if report.data_manifest_sha256 != manifest_sha:
        raise ValueError("TeacherBaselineReport.data_manifest_sha256 does not match current data manifest")
    if report.graph_regime.value != manifest.graph_regime:
        raise ValueError("TeacherBaselineReport.graph_regime does not match DataManifest.graph_regime")
    return manifest, report


def _validate_manifest_against_inputs(
    export_manifest: TeacherExportManifest,
    data_manifest: DataManifest,
    baseline_report: TeacherBaselineReport,
    data_manifest_path: str | Path,
) -> None:
    if export_manifest.dataset_name.value != data_manifest.dataset_name:
        raise ValueError("TeacherExportManifest.dataset_name does not match DataManifest.dataset_name")
    if export_manifest.dataset_name != baseline_report.dataset_name:
        raise ValueError("TeacherExportManifest.dataset_name does not match TeacherBaselineReport.dataset_name")
    if export_manifest.graph_regime.value != data_manifest.graph_regime:
        raise ValueError("TeacherExportManifest.graph_regime does not match DataManifest.graph_regime")
    if export_manifest.graph_regime != baseline_report.graph_regime:
        raise ValueError("TeacherExportManifest.graph_regime does not match TeacherBaselineReport.graph_regime")
    if export_manifest.provenance.data_manifest_path != str(data_manifest_path):
        raise ValueError("TeacherProvenance.data_manifest_path does not match current data manifest path")
    manifest_sha = compute_file_sha256(data_manifest_path)
    if export_manifest.provenance.data_manifest_sha256 != manifest_sha:
        raise ValueError("TeacherProvenance.data_manifest_sha256 does not match current data manifest")
    if export_manifest.provenance.teacher_checkpoint_sha256 != baseline_report.teacher_checkpoint_sha256:
        raise ValueError("TeacherProvenance.teacher_checkpoint_sha256 does not match baseline report")
    population = _find_population(data_manifest, export_manifest.population_name)
    if export_manifest.node_ids_hash != population.node_ids_hash:
        raise ValueError("TeacherExportManifest.node_ids_hash does not match DataManifest population node_ids_hash")
    if export_manifest.split_values != population.split_values:
        raise ValueError("TeacherExportManifest.split_values does not match DataManifest population split_values")
    if export_manifest.contains_tuning_rows != population.contains_tuning_rows:
        raise ValueError("TeacherExportManifest.contains_tuning_rows does not match DataManifest population metadata")
    if export_manifest.contains_final_test_rows != population.contains_final_test_rows:
        raise ValueError("TeacherExportManifest.contains_final_test_rows does not match DataManifest population metadata")


def _find_population(data_manifest: DataManifest, population_name: PopulationName) -> PopulationMetadata:
    for population in data_manifest.populations:
        if population.population_name == population_name.value:
            return population
    raise ValueError(f"DataManifest does not contain population {population_name.value!r}")


def _ensure_output_dir_allowed(output_dir: str | Path, dataset_name: DatasetName, population_name: PopulationName) -> Path:
    output_path = Path(output_dir)
    parts = output_path.parts
    if any(part in _FORBIDDEN_OUTPUT_PARTS for part in parts):
        raise ValueError("teacher export output_dir must not be under forbidden assets/formal/diagnostic paths")
    required = (*_OUTPUT_PREFIX_PARTS, dataset_name.value, population_name.value)
    if not _contains_contiguous_parts(parts, required):
        expected = "/".join(required)
        raise ValueError(f"teacher export output_dir must be under {expected}")
    return output_path


def _contains_contiguous_parts(parts: tuple[str, ...], required: tuple[str, ...]) -> bool:
    if len(parts) < len(required):
        return False
    for index in range(0, len(parts) - len(required) + 1):
        if parts[index : index + len(required)] == required:
            return True
    return False


def _validate_records_against_manifest(
    records: tuple[TeacherExportRecord, ...],
    export_manifest: TeacherExportManifest,
) -> tuple[TeacherExportRecord, ...]:
    if not records:
        raise ValueError("teacher export artifact requires at least one record")
    populations = {record.population_name for record in records}
    if len(populations) != 1:
        raise ValueError("teacher export artifact must contain exactly one population")
    for record in records:
        if record.graph_regime != export_manifest.graph_regime:
            raise ValueError("TeacherExportRecord.graph_regime must match TeacherExportManifest.graph_regime")
        if record.population_name != export_manifest.population_name:
            raise ValueError("TeacherExportRecord.population_name must match TeacherExportManifest.population_name")
        if record.dataset_name != export_manifest.dataset_name:
            raise ValueError("TeacherExportRecord.dataset_name must match TeacherExportManifest.dataset_name")
        if record.teacher_checkpoint != export_manifest.provenance.teacher_checkpoint_path:
            raise ValueError("TeacherExportRecord.teacher_checkpoint must match provenance.teacher_checkpoint_path")
    return records


def _write_records_parquet(records: tuple[TeacherExportRecord, ...], path: Path) -> None:
    rows = [record.model_dump(mode="json") for record in records]
    for row in rows:
        _reject_label_key(row)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _validate_reread_artifact(
    records: tuple[TeacherExportRecord, ...],
    export_manifest: TeacherExportManifest,
) -> None:
    if len(records) != export_manifest.row_count:
        raise RuntimeError("reread teacher export row_count does not match manifest")
    for record in records:
        if record.graph_regime != export_manifest.graph_regime:
            raise RuntimeError("reread teacher export graph_regime does not match manifest")
        if record.population_name != export_manifest.population_name:
            raise RuntimeError("reread teacher export population_name does not match manifest")
        if record.dataset_name != export_manifest.dataset_name:
            raise RuntimeError("reread teacher export dataset_name does not match manifest")


def _reject_label_key(value: object) -> None:
    if isinstance(value, dict):
        if "label" in value:
            raise ValidationError.from_exception_data(
                "TeacherExportRecord",
                [{"type": "extra_forbidden", "loc": ("label",), "input": value["label"]}],
            )
        for child in value.values():
            _reject_label_key(child)
    elif isinstance(value, list):
        for child in value:
            _reject_label_key(child)


def _is_legacy_assets_path(path: Path) -> bool:
    parts = path.parts
    return _contains_contiguous_parts(parts, ("assets", "teacher_exports"))


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
