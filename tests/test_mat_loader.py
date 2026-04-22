from __future__ import annotations

import numpy as np
import pytest
from scipy.io import savemat

from graph_data.mat_loader import StandardizedMatData, load_standard_mat
from graph_data.validators import (
    DataValidationError,
    compute_split_hash,
    validate_split_reproducible,
)


def _write_mat(path, *, dataset: str = "amazon", x_override=None, labels=None, relations=3, node_ids=None, split=None):
    feature_dim = 25 if dataset == "amazon" else 32
    x = np.arange(4 * feature_dim, dtype=np.float64).reshape(4, feature_dim)
    if x_override is not None:
        x = x_override
    payload = {
        "x": x,
        "ground_truth_label": np.array([0, 1, 0, 1], dtype=np.int64) if labels is None else labels,
        "node_ids": np.arange(4, dtype=np.int64) if node_ids is None else node_ids,
        "split_vector": np.array(["train", "train", "validation", "unused_holdout"], dtype=object)
        if split is None
        else split,
    }
    for index in range(relations):
        payload[f"relation_{index}"] = np.eye(4, dtype=np.float64) * (index + 1)
    savemat(path, payload)


def test_load_standard_mat_returns_canonical_fields(tmp_path):
    mat_path = tmp_path / "amazon.mat"
    _write_mat(mat_path)

    data = load_standard_mat(mat_path, dataset_name="amazon")

    assert isinstance(data, StandardizedMatData)
    assert set(type(data).model_fields) == {"x", "ground_truth_label", "relations", "node_ids", "split_vector", "metadata"}
    assert not hasattr(data, "label")
    assert data.x.shape == (4, 25)
    assert data.ground_truth_label.tolist() == [0, 1, 0, 1]
    assert len(data.relations) == 3
    assert data.node_ids.tolist() == [0, 1, 2, 3]
    assert data.metadata.dataset_name == "amazon"
    assert data.metadata.feature_dim == 25
    assert data.metadata.relation_count == 3
    assert data.metadata.split_values == ("train", "unused_holdout", "validation")


def test_load_standard_mat_accepts_yelpchi_feature_dimension(tmp_path):
    mat_path = tmp_path / "yelpchi.mat"
    _write_mat(mat_path, dataset="yelpchi")

    data = load_standard_mat(mat_path, dataset_name="yelpchi")

    assert data.x.shape == (4, 32)
    assert data.metadata.feature_dim == 32


def test_wrong_feature_dimension_fails_closed(tmp_path):
    mat_path = tmp_path / "amazon_bad_dim.mat"
    _write_mat(mat_path, x_override=np.zeros((4, 24), dtype=np.float64))

    with pytest.raises(DataValidationError, match="feature dimension 25"):
        load_standard_mat(mat_path, dataset_name="amazon")


def test_relation_count_must_be_three(tmp_path):
    mat_path = tmp_path / "bad_relations.mat"
    _write_mat(mat_path, relations=2)

    with pytest.raises(DataValidationError, match="exactly 3 relation"):
        load_standard_mat(mat_path, dataset_name="amazon")


def test_nan_or_inf_fails_closed(tmp_path):
    mat_path = tmp_path / "nan.mat"
    x = np.zeros((4, 25), dtype=np.float64)
    x[1, 1] = np.nan
    _write_mat(mat_path, x_override=x)

    with pytest.raises(DataValidationError, match="NaN or Inf"):
        load_standard_mat(mat_path, dataset_name="amazon")


def test_node_order_must_be_stable(tmp_path):
    mat_path = tmp_path / "unstable_nodes.mat"
    _write_mat(mat_path, node_ids=np.array([0, 2, 1, 3], dtype=np.int64))

    with pytest.raises(DataValidationError, match="strictly increasing"):
        load_standard_mat(mat_path, dataset_name="amazon")


def test_split_reproducibility_uses_frozen_hash(tmp_path):
    mat_path = tmp_path / "split.mat"
    split = np.array([0, 0, 1, 2], dtype=np.int64)
    _write_mat(mat_path, split=split)
    data = load_standard_mat(mat_path, dataset_name="amazon")

    expected_hash = compute_split_hash(split)
    assert validate_split_reproducible(data.split_vector, expected_count=4, expected_hash=expected_hash) == expected_hash

    with pytest.raises(DataValidationError, match="hash does not match"):
        validate_split_reproducible(data.split_vector, expected_count=4, expected_hash="0" * 64)


def test_raw_label_field_is_rejected_not_normalized(tmp_path):
    mat_path = tmp_path / "raw_label.mat"
    payload = {
        "x": np.zeros((4, 25), dtype=np.float64),
        "label": np.array([0, 1, 0, 1], dtype=np.int64),
        "node_ids": np.arange(4, dtype=np.int64),
        "split_vector": np.array([0, 0, 1, 2], dtype=np.int64),
        "relation_0": np.eye(4, dtype=np.float64),
        "relation_1": np.eye(4, dtype=np.float64),
        "relation_2": np.eye(4, dtype=np.float64),
    }
    savemat(mat_path, payload)

    with pytest.raises(DataValidationError, match="ground_truth_label"):
        load_standard_mat(mat_path, dataset_name="amazon")


@pytest.mark.parametrize("bad_node_ids", [np.array([np.nan, 1, 2, 3]), np.array([0, np.inf, 2, 3])])
def test_non_finite_node_ids_fail_before_integer_cast(tmp_path, bad_node_ids):
    mat_path = tmp_path / "bad_node_ids.mat"
    _write_mat(mat_path, node_ids=bad_node_ids)

    with pytest.raises(DataValidationError, match="node_ids contains NaN or Inf"):
        load_standard_mat(mat_path, dataset_name="amazon")


def test_missing_node_ids_fails_closed(tmp_path):
    mat_path = tmp_path / "missing_node_ids.mat"
    payload = {
        "x": np.zeros((4, 25), dtype=np.float64),
        "ground_truth_label": np.array([0, 1, 0, 1], dtype=np.int64),
        "split_vector": np.array([0, 0, 1, 2], dtype=np.int64),
        "relation_0": np.eye(4, dtype=np.float64),
        "relation_1": np.eye(4, dtype=np.float64),
        "relation_2": np.eye(4, dtype=np.float64),
    }
    savemat(mat_path, payload)

    with pytest.raises(DataValidationError, match="node_ids missing from source \\.mat; fail-closed per contract"):
        load_standard_mat(mat_path, dataset_name="amazon")


@pytest.mark.parametrize(
    ("payload_override", "expected_message"),
    [
        ({"features": np.zeros((4, 25), dtype=np.float64)}, "missing required x"),
        ({"split": np.array([0, 0, 1, 2], dtype=np.int64)}, "missing required split_vector"),
    ],
)
def test_input_key_aliases_are_rejected(tmp_path, payload_override, expected_message):
    mat_path = tmp_path / "alias_rejected.mat"
    payload = {
        "x": np.zeros((4, 25), dtype=np.float64),
        "ground_truth_label": np.array([0, 1, 0, 1], dtype=np.int64),
        "node_ids": np.arange(4, dtype=np.int64),
        "split_vector": np.array([0, 0, 1, 2], dtype=np.int64),
        "relation_0": np.eye(4, dtype=np.float64),
        "relation_1": np.eye(4, dtype=np.float64),
        "relation_2": np.eye(4, dtype=np.float64),
    }
    if "features" in payload_override:
        del payload["x"]
    if "split" in payload_override:
        del payload["split_vector"]
    payload.update(payload_override)
    savemat(mat_path, payload)

    with pytest.raises(DataValidationError, match=expected_message):
        load_standard_mat(mat_path, dataset_name="amazon")


@pytest.mark.parametrize(
    "bad_labels",
    [
        np.array([0, 1, -1, 1], dtype=np.int64),
        np.array([0, 1, 2, 1], dtype=np.int64),
        np.array([0, 1, np.nan, 1], dtype=np.float64),
        np.array([0, 1, np.inf, 1], dtype=np.float64),
    ],
)
def test_ground_truth_label_rejects_non_binary_values(tmp_path, bad_labels):
    mat_path = tmp_path / "bad_ground_truth_label.mat"
    _write_mat(mat_path, labels=bad_labels)

    with pytest.raises(DataValidationError, match="ground_truth_label"):
        load_standard_mat(mat_path, dataset_name="amazon")


@pytest.mark.parametrize("dataset_name", ["Amazon", "amazon ", "yelp_chi", "YelpChi"])
def test_dataset_name_must_be_exact_canonical(tmp_path, dataset_name):
    mat_path = tmp_path / "amazon.mat"
    _write_mat(mat_path)

    with pytest.raises(DataValidationError, match="dataset_name must be exactly"):
        load_standard_mat(mat_path, dataset_name=dataset_name)
