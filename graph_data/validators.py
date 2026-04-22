"""Fail-closed validation utilities for benchmark graph data.

The functions in this module validate data-foundation contracts only.  They do
not implement training, evaluation, thresholding, alpha selection, or diagnostic
logic.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import sparse

BenchmarkName = Literal["amazon", "yelpchi"]

EXPECTED_FEATURE_DIMS: dict[BenchmarkName, int] = {
    "amazon": 25,
    "yelpchi": 32,
}
EXPECTED_RELATION_COUNT = 3


class DataValidationError(ValueError):
    """Raised when graph data violates a fail-closed data contract."""


def validate_benchmark_name(dataset_name: str) -> BenchmarkName:
    """Return the benchmark name only when it is exactly canonical."""

    if dataset_name == "amazon":
        return "amazon"
    if dataset_name == "yelpchi":
        return "yelpchi"
    raise DataValidationError("dataset_name must be exactly 'amazon' or 'yelpchi'")


def expected_feature_dim(dataset_name: str) -> int:
    """Return the required feature dimension for an exact canonical benchmark."""

    return EXPECTED_FEATURE_DIMS[validate_benchmark_name(dataset_name)]


def as_dense_float_array(value: Any, *, field_name: str) -> np.ndarray:
    """Convert a matrix-like input to a two-dimensional finite float array."""

    if sparse.issparse(value):
        array = value.toarray()
    else:
        array = np.asarray(value)
    if array.ndim != 2:
        raise DataValidationError(f"{field_name} must be a 2D matrix; got shape {array.shape}")
    try:
        dense = array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{field_name} must be numeric") from exc
    validate_finite_array(dense, field_name=field_name)
    return dense


def as_one_dimensional_array(value: Any, *, field_name: str) -> np.ndarray:
    """Convert a vector-like input to a one-dimensional numpy array."""

    array = np.asarray(value)
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim > 1:
        array = array.squeeze()
    if array.ndim != 1:
        raise DataValidationError(f"{field_name} must be a 1D vector; got shape {array.shape}")
    if array.size == 0:
        raise DataValidationError(f"{field_name} must not be empty")
    return array


def validate_finite_array(array: np.ndarray, *, field_name: str) -> None:
    """Fail closed if a numeric array contains NaN or infinite values."""

    try:
        finite = np.isfinite(array)
    except TypeError as exc:
        raise DataValidationError(f"{field_name} must be numeric for finite-value validation") from exc
    if not bool(finite.all()):
        raise DataValidationError(f"{field_name} contains NaN or Inf")


def validate_feature_dimension(x: np.ndarray, *, dataset_name: str) -> None:
    """Validate the benchmark-specific feature dimension."""

    if x.ndim != 2:
        raise DataValidationError(f"x must be a 2D feature matrix; got shape {x.shape}")
    required = expected_feature_dim(dataset_name)
    observed = int(x.shape[1])
    if observed != required:
        canonical_dataset = validate_benchmark_name(dataset_name)
        raise DataValidationError(f"{canonical_dataset} requires feature dimension {required}; got {observed}")




def validate_ground_truth_label(array: np.ndarray) -> None:
    """Validate that ground-truth labels are finite binary values only."""

    labels = as_one_dimensional_array(array, field_name="ground_truth_label")
    if labels.dtype.kind not in {"b", "i", "u", "f"}:
        raise DataValidationError("ground_truth_label must contain numeric binary values")
    numeric_labels = labels.astype(np.float64, copy=False)
    validate_finite_array(numeric_labels, field_name="ground_truth_label")
    if not bool(np.all(numeric_labels == np.floor(numeric_labels))):
        raise DataValidationError("ground_truth_label must contain integer binary values")
    values = set(numeric_labels.astype(np.int64, copy=False).tolist())
    if not values.issubset({0, 1}):
        raise DataValidationError("ground_truth_label values must be a subset of {0, 1}")


def compute_file_sha256(path: str | Path) -> str:
    """Compute the SHA-256 digest of an artifact file's raw bytes."""

    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise DataValidationError(f"artifact file does not exist: {artifact_path}")
    digest = sha256()
    with artifact_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def validate_relation_count(relations: Sequence[Any]) -> None:
    """Validate that exactly three graph relation matrices are present."""

    if len(relations) != EXPECTED_RELATION_COUNT:
        raise DataValidationError(
            f"expected {EXPECTED_RELATION_COUNT} relation matrices; got {len(relations)}"
        )


def validate_relations(relations: Sequence[Any], *, num_nodes: int) -> None:
    """Validate relation count, shape, and finite relation weights."""

    validate_relation_count(relations)
    for index, relation in enumerate(relations):
        shape = getattr(relation, "shape", None)
        if shape != (num_nodes, num_nodes):
            raise DataValidationError(
                f"relations[{index}] must have shape {(num_nodes, num_nodes)}; got {shape}"
            )
        if sparse.issparse(relation):
            data = relation.data
            if data.size:
                validate_finite_array(np.asarray(data), field_name=f"relations[{index}].data")
        else:
            validate_finite_array(np.asarray(relation), field_name=f"relations[{index}]")


def validate_node_ids_stable(node_ids: np.ndarray, *, expected_count: int) -> None:
    """Validate that node ids define a stable one-to-one node order."""

    ids = as_one_dimensional_array(node_ids, field_name="node_ids")
    if ids.shape[0] != expected_count:
        raise DataValidationError(f"node_ids length {ids.shape[0]} does not match node count {expected_count}")
    if ids.dtype.kind in {"f", "c"}:
        validate_finite_array(ids.astype(np.float64, copy=False), field_name="node_ids")
        if not np.all(ids == np.floor(ids)):
            raise DataValidationError("node_ids must be integral values")
    try:
        sortable = ids.astype(np.int64, copy=False)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("node_ids must be integral and sortable") from exc
    if len(np.unique(sortable)) != expected_count:
        raise DataValidationError("node_ids must be unique")
    if not bool(np.all(sortable[:-1] < sortable[1:])):
        raise DataValidationError("node_ids must be strictly increasing to make node order stable")


def _canonical_bytes(array: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(array)
    return b"|".join(
        [
            str(contiguous.dtype).encode("utf-8"),
            repr(tuple(contiguous.shape)).encode("utf-8"),
            contiguous.tobytes(),
        ]
    )


def compute_array_hash(array: np.ndarray) -> str:
    """Compute a deterministic SHA-256 hash for an array's dtype, shape, and bytes."""

    return sha256(_canonical_bytes(np.asarray(array))).hexdigest()


def compute_node_ids_hash(node_ids: np.ndarray) -> str:
    """Compute the canonical node-id hash used by data manifests."""

    ids = as_one_dimensional_array(node_ids, field_name="node_ids")
    return compute_array_hash(ids.astype(np.int64, copy=False))


def compute_split_hash(split_vector: np.ndarray) -> str:
    """Compute the canonical split-vector hash used to prove split reproducibility."""

    split = as_one_dimensional_array(split_vector, field_name="split_vector")
    if split.dtype.kind in {"f", "c"}:
        validate_finite_array(split.astype(np.float64, copy=False), field_name="split_vector")
    return compute_array_hash(split)


def validate_split_reproducible(split_vector: np.ndarray, *, expected_count: int, expected_hash: str | None = None) -> str:
    """Validate split vector shape/content and optionally match a frozen split hash."""

    split = as_one_dimensional_array(split_vector, field_name="split_vector")
    if split.shape[0] != expected_count:
        raise DataValidationError(
            f"split_vector length {split.shape[0]} does not match node count {expected_count}"
        )
    if split.dtype.kind in {"f", "c"}:
        validate_finite_array(split.astype(np.float64, copy=False), field_name="split_vector")
    digest = compute_split_hash(split)
    if expected_hash is not None and digest != expected_hash:
        raise DataValidationError("split_vector hash does not match the expected reproducible split")
    return digest


def split_values(split_vector: np.ndarray) -> tuple[str, ...]:
    """Return deterministic split values for manifest population metadata."""

    split = as_one_dimensional_array(split_vector, field_name="split_vector")
    values = sorted({str(item) for item in split.tolist()})
    if not values:
        raise DataValidationError("split_vector must contain at least one split value")
    return tuple(values)


def validate_no_population_overlap(population_hashes: Iterable[tuple[str, str]]) -> None:
    """Fail closed if two named populations have the same node-id hash."""

    seen: dict[str, str] = {}
    for population_name, node_hash in population_hashes:
        previous = seen.get(node_hash)
        if previous is not None and previous != population_name:
            raise DataValidationError(
                f"population {population_name!r} overlaps with {previous!r} by node_ids_hash"
            )
        seen[node_hash] = population_name
