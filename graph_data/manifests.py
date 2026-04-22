"""Pydantic data manifest contracts for PriorF-Reasoner.

All downstream train/eval entrypoints must depend on these manifest contracts
instead of accepting naked parquet/csv paths directly.  This module defines the
contract only; it does not implement downstream training or evaluation logic.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from graph_data.mat_loader import StandardizedMatData
from graph_data.validators import DataValidationError, validate_benchmark_name, validate_no_population_overlap

PopulationName = Literal["train", "validation", "unused_holdout", "diagnostic_holdout", "final_test"]
GraphRegime = Literal["transductive_standard", "inductive_masked"]
ArtifactKind = Literal["source_mat", "standardized_data", "manifest"]


class PopulationMetadata(BaseModel):
    """Required metadata for any artifact population containing examples or predictions."""

    model_config = ConfigDict(extra="forbid")

    population_name: PopulationName
    split_values: tuple[str, ...]
    node_ids_hash: str = Field(min_length=64, max_length=64)
    contains_tuning_rows: bool
    contains_final_test_rows: bool

    @field_validator("split_values")
    @classmethod
    def _split_values_must_be_explicit(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("split_values must not be empty")
        if any(str(item).strip() == "" for item in value):
            raise ValueError("split_values must not contain blank values")
        return tuple(str(item) for item in value)

    @model_validator(mode="after")
    def _population_semantics_are_not_ambiguous(self) -> "PopulationMetadata":
        if self.population_name == "unused_holdout" and self.contains_final_test_rows:
            raise ValueError("unused_holdout must not be marked as containing final_test rows")
        if self.population_name == "final_test" and self.contains_tuning_rows:
            raise ValueError("final_test must not contain tuning rows")
        return self


class DataArtifact(BaseModel):
    """Manifest entry for a concrete data artifact path and checksum."""

    model_config = ConfigDict(extra="forbid")

    kind: ArtifactKind
    path: str
    sha256: str = Field(min_length=64, max_length=64)

    @field_validator("path")
    @classmethod
    def _path_must_be_explicit(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("artifact path must not be blank")
        return value


class DataManifest(BaseModel):
    """Fail-closed data manifest required before train/eval entrypoints."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["priorf-reasoner.data-manifest.v1"] = "priorf-reasoner.data-manifest.v1"
    dataset_name: Literal["amazon", "yelpchi"]
    graph_regime: GraphRegime
    feature_dim: int
    relation_count: Literal[3]
    num_nodes: int = Field(gt=0)
    populations: tuple[PopulationMetadata, ...]
    artifacts: tuple[DataArtifact, ...]
    created_at_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _manifest_must_fail_closed(self) -> "DataManifest":
        if self.dataset_name == "amazon" and self.feature_dim != 25:
            raise ValueError("Amazon manifest feature_dim must be 25")
        if self.dataset_name == "yelpchi" and self.feature_dim != 32:
            raise ValueError("YelpChi manifest feature_dim must be 32")
        if not self.populations:
            raise ValueError("DataManifest requires at least one population")
        names = [population.population_name for population in self.populations]
        if len(set(names)) != len(names):
            raise ValueError("population_name values must be unique")
        if "test" in names:
            raise ValueError("ambiguous population name 'test' is forbidden")
        validate_no_population_overlap(
            (population.population_name, population.node_ids_hash) for population in self.populations
        )
        if not any(artifact.kind == "source_mat" for artifact in self.artifacts):
            raise ValueError("DataManifest requires a source_mat artifact")
        return self


def build_data_manifest(
    *,
    data: StandardizedMatData,
    graph_regime: GraphRegime,
    populations: tuple[PopulationMetadata, ...],
    artifacts: tuple[DataArtifact, ...] | None,
) -> DataManifest:
    """Build a validated DataManifest from standardized `.mat` data."""

    if artifacts is None or not artifacts:
        raise ValueError("artifacts must be explicitly provided with sha256 provenance")
    return DataManifest(
        dataset_name=validate_benchmark_name(data.metadata.dataset_name),
        graph_regime=graph_regime,
        feature_dim=data.metadata.feature_dim,
        relation_count=3,
        num_nodes=data.metadata.num_nodes,
        populations=populations,
        artifacts=artifacts,
    )


def load_data_manifest(path: str | Path) -> DataManifest:
    """Load and validate a DataManifest JSON file, failing closed on errors."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise DataValidationError(f"data manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"data manifest is not valid JSON: {manifest_path}") from exc
    try:
        return DataManifest.model_validate(payload)
    except ValueError as exc:
        raise DataValidationError(f"invalid data manifest: {manifest_path}: {exc}") from exc


def write_data_manifest(manifest: DataManifest, path: str | Path) -> None:
    """Write a validated DataManifest as deterministic JSON."""

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
