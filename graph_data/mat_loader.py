"""Standard `.mat` benchmark loader for PriorF-Reasoner data foundations.

This module normalizes benchmark data into the canonical field names required by
Task 1: `x`, `ground_truth_label`, `relations`, `node_ids`, `split_vector`, and
`metadata`.  It fails closed on missing fields or invalid benchmark contracts.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator
from scipy import sparse
from scipy.io import loadmat

from graph_data.validators import (
    DataValidationError,
    as_dense_float_array,
    as_one_dimensional_array,
    compute_node_ids_hash,
    expected_feature_dim,
    split_values,
    validate_benchmark_name,
    validate_feature_dimension,
    validate_ground_truth_label,
    validate_node_ids_stable,
    validate_relations,
    validate_split_reproducible,
)

_DATA_KEYS = {"x", "ground_truth_label", "node_ids", "split_vector"}
_RELATION_PREFIXES = ("relation_", "net_", "adj_", "graph_")


class MatDatasetMetadata(BaseModel):
    """Metadata produced while normalizing one benchmark `.mat` file."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    source_path: str
    feature_dim: int
    relation_count: int
    num_nodes: int
    node_ids_hash: str
    split_hash: str
    split_values: tuple[str, ...]


class StandardizedMatData(BaseModel):
    """Canonical in-memory representation emitted by the `.mat` loader."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    x: np.ndarray
    ground_truth_label: np.ndarray
    relations: tuple[Any, Any, Any]
    node_ids: np.ndarray
    split_vector: np.ndarray
    metadata: MatDatasetMetadata

    @model_validator(mode="after")
    def _validate_contract(self) -> "StandardizedMatData":
        expected_count = int(self.x.shape[0])
        validate_feature_dimension(self.x, dataset_name=self.metadata.dataset_name)
        if self.ground_truth_label.shape[0] != expected_count:
            raise DataValidationError("ground_truth_label length does not match x row count")
        validate_ground_truth_label(self.ground_truth_label)
        validate_node_ids_stable(self.node_ids, expected_count=expected_count)
        validate_split_reproducible(self.split_vector, expected_count=expected_count)
        validate_relations(self.relations, num_nodes=expected_count)
        return self


def load_standard_mat(path: str | Path, *, dataset_name: str) -> StandardizedMatData:
    """Load and validate a standard Amazon or YelpChi benchmark `.mat` file.

    Parameters
    ----------
    path:
        Path to the benchmark `.mat` file.
    dataset_name:
        Canonical benchmark name, `amazon` or `yelpchi`.

    Returns
    -------
    StandardizedMatData
        Pydantic contract with canonical data fields.

    Raises
    ------
    DataValidationError
        If any required field is missing or violates the fail-closed contract.
    """

    source_path = Path(path)
    if not source_path.is_file():
        raise DataValidationError(f".mat file does not exist: {source_path}")

    canonical_dataset = validate_benchmark_name(dataset_name)
    raw = loadmat(source_path, squeeze_me=True)

    x = as_dense_float_array(_require_key(raw, "x"), field_name="x")
    validate_feature_dimension(x, dataset_name=canonical_dataset)
    num_nodes = int(x.shape[0])

    ground_truth_label = as_one_dimensional_array(
        _require_key(raw, "ground_truth_label"),
        field_name="ground_truth_label",
    )
    if ground_truth_label.shape[0] != num_nodes:
        raise DataValidationError("ground_truth_label length does not match x row count")
    validate_ground_truth_label(ground_truth_label)

    if "node_ids" not in raw:
        raise DataValidationError("node_ids missing from source .mat; fail-closed per contract")
    raw_node_ids = as_one_dimensional_array(raw["node_ids"], field_name="node_ids")
    validate_node_ids_stable(raw_node_ids, expected_count=num_nodes)
    node_ids = raw_node_ids.astype(np.int64, copy=False)

    split_vector = as_one_dimensional_array(
        _require_key(raw, "split_vector"),
        field_name="split_vector",
    )
    split_hash = validate_split_reproducible(split_vector, expected_count=num_nodes)

    relations = tuple(_extract_relations(raw, num_nodes=num_nodes))
    validate_relations(relations, num_nodes=num_nodes)

    metadata = MatDatasetMetadata(
        dataset_name=canonical_dataset,
        source_path=str(source_path),
        feature_dim=expected_feature_dim(canonical_dataset),
        relation_count=len(relations),
        num_nodes=num_nodes,
        node_ids_hash=compute_node_ids_hash(node_ids),
        split_hash=split_hash,
        split_values=split_values(split_vector),
    )

    return StandardizedMatData(
        x=x,
        ground_truth_label=ground_truth_label,
        relations=relations,  # type: ignore[arg-type]
        node_ids=node_ids,
        split_vector=split_vector,
        metadata=metadata,
    )


def _require_key(raw: Mapping[str, Any], key: str) -> Any:
    """Return one required raw `.mat` key or fail closed without alias fallback."""

    if key in raw:
        return raw[key]
    raise DataValidationError(f"missing required {key} from source .mat; fail-closed per contract")


def _extract_relations(raw: Mapping[str, Any], *, num_nodes: int) -> list[Any]:
    bundled = raw.get("relations")
    if bundled is not None:
        relations = _unpack_relation_bundle(bundled)
    else:
        relations = _discover_relation_matrices(raw, num_nodes=num_nodes)
    if len(relations) != 3:
        raise DataValidationError(f"expected exactly 3 relation matrices; got {len(relations)}")
    return relations


def _unpack_relation_bundle(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [item for item in value.ravel().tolist()]
    if isinstance(value, (list, tuple)):
        return list(value)
    if sparse.issparse(value) or np.asarray(value).ndim == 2:
        raise DataValidationError("relations bundle must contain exactly three relation matrices, not one matrix")
    raise DataValidationError("relations must be a sequence or object array of relation matrices")


def _discover_relation_matrices(raw: Mapping[str, Any], *, num_nodes: int) -> list[Any]:
    relation_items: list[tuple[str, Any]] = []
    for key, value in raw.items():
        if key.startswith("__") or key in _DATA_KEYS:
            continue
        if not key.startswith(_RELATION_PREFIXES):
            continue
        shape = getattr(value, "shape", None)
        if shape == (num_nodes, num_nodes):
            relation_items.append((key, value))
    relation_items.sort(key=lambda item: item[0])
    return [value for _, value in relation_items]
