"""Dataset splitting utilities with stratified sampling and graph structure preservation."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.data import HeteroData

from lghgcl.logging_utils import get_logger

logger = get_logger(__name__)


def split_data(
    data: HeteroData,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 717,
) -> HeteroData:
    """Split dataset into train/val/test with stratified sampling.

    Strictly follows 7:1:2 ratio and fixes seed for reproducibility.
    The split is performed on the 'review' nodes based on their 'y' labels.
    Masks are added to the data object: train_mask, val_mask, test_mask.

    Args:
        data: HeteroData object containing 'review' nodes and 'y' labels.
        train_ratio: Proportion of training data (default 0.7).
        val_ratio: Proportion of validation data (default 0.1).
        test_ratio: Proportion of testing data (default 0.2).
        seed: Random seed (default 717).

    Returns:
        HeteroData object with added masks.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"

    y = data["review"].y.numpy()
    num_nodes = len(y)
    indices = np.arange(num_nodes)

    # First split: Train vs (Val + Test)
    # Train = 0.7, Remainder = 0.3
    try:
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_ratio), random_state=seed)
        train_idx, remainder_idx = next(sss1.split(indices, y))
    except ValueError:
        logger.warning("Stratified split failed (dataset too small?). Falling back to random split.")
        # Fallback to random shuffle
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(indices)
        train_size = int(len(indices) * train_ratio)
        train_idx = shuffled[:train_size]
        remainder_idx = shuffled[train_size:]

    # Second split: Val vs Test from Remainder
    # Remainder is 0.3 of total.
    # Val is 0.1 of total -> 0.1 / 0.3 = 1/3 of remainder
    # Test is 0.2 of total -> 0.2 / 0.3 = 2/3 of remainder

    y_remainder = y[remainder_idx]
    # Adjust test_size for the second split relative to the remainder
    # val_relative = val_ratio / (val_ratio + test_ratio)
    test_relative = test_ratio / (val_ratio + test_ratio)

    try:
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_relative, random_state=seed)
        val_sub_idx, test_sub_idx = next(sss2.split(remainder_idx, y_remainder))
    except ValueError:
        logger.warning("Stratified split failed for validation set. Falling back to random split.")
        rng = np.random.default_rng(seed)
        shuffled_rem = rng.permutation(len(remainder_idx))
        # Validation size relative to remainder: 1 - test_relative
        val_size = int(len(remainder_idx) * (1 - test_relative))
        # Ensure at least 1 val sample if remainder is not empty
        if val_size == 0 and len(remainder_idx) > 0:
            # If test_relative is large, val might be 0.
            # 1/3 is approx 0.33. If remainder is 2, 0.33 * 2 = 0.66 -> 0.
            # But we want 1 sample.
            val_size = 1

        # If val_size takes everything, ensure test has something?
        if val_size == len(remainder_idx) and len(remainder_idx) > 1:
            val_size = len(remainder_idx) - 1

        val_sub_idx = shuffled_rem[:val_size]
        test_sub_idx = shuffled_rem[val_size:]

    val_idx = remainder_idx[val_sub_idx]
    test_idx = remainder_idx[test_sub_idx]

    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # Assign to data
    data["review"].train_mask = train_mask
    data["review"].val_mask = val_mask
    data["review"].test_mask = test_mask

    logger.info(
        "Data Split (Seed=%d): Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
        seed,
        len(train_idx),
        100 * len(train_idx) / num_nodes,
        len(val_idx),
        100 * len(val_idx) / num_nodes,
        len(test_idx),
        100 * len(test_idx) / num_nodes,
    )

    return data
