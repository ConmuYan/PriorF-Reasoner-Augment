"""Stratified subsampling for label scarcity experiments."""

from __future__ import annotations

import torch


def stratified_subsample_mask(
    train_mask: torch.Tensor,
    y: torch.Tensor,
    keep_ratio: float,
    seed: int,
) -> torch.Tensor:
    """Subsample training nodes while preserving class distribution.

    Args:
        train_mask: Boolean mask of current training nodes.
        y: Labels for all nodes.
        keep_ratio: Fraction of training nodes to keep (0..1).
        seed: Random seed for reproducibility.

    Returns:
        New boolean mask with subsampled training nodes.
    """
    train_idx = torch.where(train_mask)[0]
    train_y = y[train_idx]
    new_train_mask = torch.zeros_like(train_mask)

    torch.manual_seed(seed)
    for cls in torch.unique(train_y):
        cls_idx = train_idx[train_y == cls]
        num_keep = max(1, int(len(cls_idx) * keep_ratio))
        perm = torch.randperm(len(cls_idx))
        keep_idx = cls_idx[perm[:num_keep]]
        new_train_mask[keep_idx] = True

    return new_train_mask
