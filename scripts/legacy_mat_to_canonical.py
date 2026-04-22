"""Convert legacy PriorF-GNN .mat files to canonical graph_data format.

Legacy keys (Dou et al. CIKM 2020):
  - features  → x
  - label     → ground_truth_label
  - net_upu / net_usu / net_uvu  → relation_0 / relation_1 / relation_2

Canonical keys (graph_data.mat_loader contract):
  - x, ground_truth_label, node_ids, split_vector, relation_0, relation_1, relation_2

Split is reproduced with sklearn StratifiedShuffleSplit(seed=717, 7:1:2).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from sklearn.model_selection import StratifiedShuffleSplit

DATASETS = {
    "amazon": {
        "source": "assets/data/Amazon.mat",
        "output": "assets/data/Amazon_canonical.mat",
        "legacy_relations": ("net_upu", "net_usu", "net_uvu"),
    },
    "yelpchi": {
        "source": "assets/data/YelpChi.mat",
        "output": "assets/data/YelpChi_canonical.mat",
        "legacy_relations": ("net_rur", "net_rtr", "net_rsr"),
    },
}

SEED = 717
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2


def _reproduce_split(
    labels: np.ndarray,
    *,
    seed: int = SEED,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> np.ndarray:
    """Return a string split_vector reproducing the 7:1:2 stratified split."""

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    n = labels.shape[0]
    indices = np.arange(n)

    # First split: train vs remainder (val + test)
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=(1 - train_ratio), random_state=seed
    )
    train_idx, remainder_idx = next(sss1.split(indices, labels))

    # Second split: val vs test from remainder
    test_relative = test_ratio / (val_ratio + test_ratio)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_relative, random_state=seed
    )
    val_sub_idx, test_sub_idx = next(sss2.split(remainder_idx, labels[remainder_idx]))
    val_idx = remainder_idx[val_sub_idx]
    test_idx = remainder_idx[test_sub_idx]

    split_vector = np.empty(n, dtype=object)
    split_vector[train_idx] = "train"
    split_vector[val_idx] = "validation"
    split_vector[test_idx] = "final_test"
    return split_vector


def convert_legacy_mat(
    source_path: str | Path,
    output_path: str | Path,
    relation_keys: tuple[str, ...],
) -> None:
    """Read a legacy .mat and emit a canonical one."""

    source_path = Path(source_path)
    output_path = Path(output_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Legacy .mat not found: {source_path}")

    raw = loadmat(str(source_path), squeeze_me=True)

    # --- canonical fields ---
    features = raw["features"]
    if hasattr(features, "toarray"):
        features = features.toarray()
    x = features.astype(np.float64)

    labels = raw["label"].flatten()
    ground_truth_label = labels.astype(np.int64)

    n = x.shape[0]
    node_ids = np.arange(n, dtype=np.int64)
    split_vector = _reproduce_split(ground_truth_label)

    # --- relations (keep sparse as sparse; densify only if already dense) ---
    from scipy import sparse as sp
    relations: dict[str, np.ndarray] = {}
    for i, legacy_key in enumerate(relation_keys):
        mat = raw[legacy_key]
        if sp.issparse(mat):
            # Ensure float64 dtype but keep sparse format for savemat
            if mat.dtype != np.float64:
                mat = mat.astype(np.float64)
        elif hasattr(mat, "toarray"):
            mat = mat.toarray().astype(np.float64)
        else:
            mat = mat.astype(np.float64)
        relations[f"relation_{i}"] = mat

    # --- save canonical .mat ---
    payload = {
        "x": x,
        "ground_truth_label": ground_truth_label,
        "node_ids": node_ids,
        "split_vector": split_vector,
        **relations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(output_path), payload)
    print(f"Canonical .mat written: {output_path}  (nodes={n})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert legacy .mat to canonical format")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="amazon")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    convert_legacy_mat(cfg["source"], cfg["output"], cfg["legacy_relations"])


if __name__ == "__main__":
    main()
