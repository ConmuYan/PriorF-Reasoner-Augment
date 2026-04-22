from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.io import savemat

from graph_data.manifests import (
    DataArtifact,
    DataManifest,
    PopulationMetadata,
    build_data_manifest,
    load_data_manifest,
    write_data_manifest,
)
from graph_data.mat_loader import load_standard_mat
from graph_data.validators import DataValidationError, compute_file_sha256, compute_node_ids_hash


def _population(name: str, node_hash: str | None = None, **overrides):
    payload = {
        "population_name": name,
        "split_values": (name,),
        "node_ids_hash": node_hash or "a" * 64,
        "contains_tuning_rows": name in {"train", "validation"},
        "contains_final_test_rows": name == "final_test",
    }
    payload.update(overrides)
    return PopulationMetadata.model_validate(payload)


def _standardized_data(tmp_path):
    mat_path = tmp_path / "amazon.mat"
    payload = {
        "x": np.zeros((4, 25), dtype=np.float64),
        "ground_truth_label": np.array([0, 1, 0, 1], dtype=np.int64),
        "node_ids": np.arange(4, dtype=np.int64),
        "split_vector": np.array(["train", "train", "validation", "unused_holdout"], dtype=object),
        "relation_0": np.eye(4, dtype=np.float64),
        "relation_1": np.eye(4, dtype=np.float64),
        "relation_2": np.eye(4, dtype=np.float64),
    }
    savemat(mat_path, payload)
    return load_standard_mat(mat_path, dataset_name="amazon")


def test_data_manifest_validates_and_round_trips(tmp_path):
    data = _standardized_data(tmp_path)
    train_hash = compute_node_ids_hash(np.array([0, 1], dtype=np.int64))
    validation_hash = compute_node_ids_hash(np.array([2], dtype=np.int64))
    holdout_hash = compute_node_ids_hash(np.array([3], dtype=np.int64))
    artifact = DataArtifact(
        kind="source_mat",
        path=data.metadata.source_path,
        sha256=compute_file_sha256(data.metadata.source_path),
    )
    manifest = build_data_manifest(
        data=data,
        graph_regime="transductive_standard",
        populations=(
            _population("train", train_hash, split_values=("train",)),
            _population("validation", validation_hash, split_values=("validation",)),
            _population(
                "unused_holdout",
                holdout_hash,
                split_values=("unused_holdout",),
                contains_tuning_rows=False,
                contains_final_test_rows=False,
            ),
        ),
        artifacts=(artifact,),
    )

    assert isinstance(manifest, DataManifest)
    assert manifest.dataset_name == "amazon"
    assert manifest.feature_dim == 25
    assert manifest.relation_count == 3
    assert manifest.artifacts[0].kind == "source_mat"

    manifest_path = tmp_path / "manifest.json"
    write_data_manifest(manifest, manifest_path)
    loaded = load_data_manifest(manifest_path)
    assert loaded == manifest


def test_population_name_test_is_forbidden():
    with pytest.raises(ValidationError):
        PopulationMetadata.model_validate(
            {
                "population_name": "test",
                "split_values": ("test",),
                "node_ids_hash": "b" * 64,
                "contains_tuning_rows": False,
                "contains_final_test_rows": True,
            }
        )


def test_unused_holdout_must_not_be_final_test():
    with pytest.raises(ValidationError, match="unused_holdout"):
        _population(
            "unused_holdout",
            "c" * 64,
            contains_tuning_rows=False,
            contains_final_test_rows=True,
        )


def test_manifest_requires_population_metadata():
    with pytest.raises(ValidationError, match="at least one population"):
        DataManifest(
            dataset_name="amazon",
            graph_regime="transductive_standard",
            feature_dim=25,
            relation_count=3,
            num_nodes=4,
            populations=(),
            artifacts=(DataArtifact(kind="source_mat", path="amazon.mat", sha256="0" * 64),),
        )


def test_manifest_rejects_missing_source_artifact():
    with pytest.raises(ValidationError, match="source_mat"):
        DataManifest(
            dataset_name="amazon",
            graph_regime="transductive_standard",
            feature_dim=25,
            relation_count=3,
            num_nodes=4,
            populations=(_population("train", "d" * 64),),
            artifacts=(DataArtifact(kind="manifest", path="manifest.json", sha256="0" * 64),),
        )


def test_manifest_rejects_population_overlap():
    duplicate_hash = "e" * 64
    with pytest.raises(ValidationError, match="overlaps"):
        DataManifest(
            dataset_name="amazon",
            graph_regime="transductive_standard",
            feature_dim=25,
            relation_count=3,
            num_nodes=4,
            populations=(
                _population("train", duplicate_hash),
                _population("validation", duplicate_hash),
            ),
            artifacts=(DataArtifact(kind="source_mat", path="amazon.mat", sha256="0" * 64),),
        )


def test_manifest_forbids_unknown_fields_and_invalid_json_fails_closed(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(DataValidationError, match="not valid JSON"):
        load_data_manifest(manifest_path)

    with pytest.raises(ValidationError):
        DataManifest.model_validate(
            {
                "dataset_name": "amazon",
                "graph_regime": "transductive_standard",
                "feature_dim": 25,
                "relation_count": 3,
                "num_nodes": 4,
                "populations": [_population("train", "f" * 64).model_dump()],
                "artifacts": [DataArtifact(kind="source_mat", path="amazon.mat", sha256="0" * 64).model_dump()],
                "raw_path": "forbidden.csv",
            }
        )


def test_nested_pydantic_contracts_forbid_unknown_fields(tmp_path):
    data = _standardized_data(tmp_path)

    with pytest.raises(ValidationError):
        type(data).model_validate({**data.model_dump(), "label": [0, 1, 0, 1]})

    with pytest.raises(ValidationError):
        PopulationMetadata.model_validate(
            {
                "population_name": "train",
                "split_values": ("train",),
                "node_ids_hash": "1" * 64,
                "contains_tuning_rows": True,
                "contains_final_test_rows": False,
                "unexpected": "forbidden",
            }
        )

    with pytest.raises(ValidationError):
        DataArtifact.model_validate(
            {
                "kind": "source_mat",
                "path": "amazon.mat",
                "sha256": "0" * 64,
                "unexpected": "forbidden",
            }
        )


def test_data_artifact_requires_sha256():
    with pytest.raises(ValidationError):
        DataArtifact.model_validate({"kind": "source_mat", "path": "amazon.mat"})


def test_build_data_manifest_requires_explicit_artifacts(tmp_path):
    data = _standardized_data(tmp_path)
    populations = (_population("train", compute_node_ids_hash(np.array([0, 1], dtype=np.int64))),)

    with pytest.raises(ValueError, match="artifacts must be explicitly provided"):
        build_data_manifest(
            data=data,
            graph_regime="transductive_standard",
            populations=populations,
            artifacts=None,
        )

    with pytest.raises(ValueError, match="artifacts must be explicitly provided"):
        build_data_manifest(
            data=data,
            graph_regime="transductive_standard",
            populations=populations,
            artifacts=(),
        )
