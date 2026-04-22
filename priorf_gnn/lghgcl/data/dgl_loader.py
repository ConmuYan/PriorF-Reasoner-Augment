"""DGL数据加载器 - 支持tfinance和tsocial单关系型数据集.

This module provides data loading capabilities for single-relational graph
datasets in DGL binary format, specifically designed for tfinance and tsocial.

tfinance: 39K节点, 42M边, 10维特征, 4.6%欺诈率
tsocial: 5.8M节点, 146M边, 10维特征, 3.0%欺诈率

Preprocessing: log(x+1) transform is applied to raw features for tfinance/tsocial
to match BWGNN's preprocessing convention (T-Finance sourced from BWGNN repo).
F8/F9 (already normalized to [0,1]) are left unchanged.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import HeteroData
from torch_scatter import scatter_mean

from lghgcl.logging_utils import get_logger

logger = get_logger(__name__)


def _ensure_dgl_importable():
    """Ensure ``import dgl`` works even when graphbolt C++ lib is missing.

    DGL 2.1.x bundles graphbolt that expects ``libgraphbolt_pytorch_<ver>.so``
    matching the installed PyTorch version.  When there is a mismatch the
    entire ``import dgl`` fails.  We work around this by pre-populating
    ``sys.modules`` with stub sub-modules so that DGL's own ``__init__``
    never tries to load the missing .so file.
    """
    import sys
    import types

    stubs = [
        "dgl.graphbolt",
        "dgl.graphbolt.base",
        "dgl.graphbolt.dataloader",
        "dgl.graphbolt.impl",
        "dgl.graphbolt.impl.legacy_dataset",
        "dgl.graphbolt.impl.ondisk_dataset",
        "dgl.graphbolt.impl.ondisk_metadata",
    ]
    for name in stubs:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod


def _load_dgl_graphs(path: str):
    """Load DGL binary graph file.

    Automatically patches the graphbolt import when the C++ library is
    missing due to a PyTorch / DGL version mismatch.
    """
    _ensure_dgl_importable()
    from dgl.data.utils import load_graphs

    return load_graphs(path)


def compute_hsd(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Compute Heterophily-Induced Structural Discrepancy (HSD).

    HSD measures the average L2 distance between a node's features and
    its neighbors' features. Higher HSD indicates the node is more different
    from its local neighborhood, which is a strong signal for anomaly detection.

    Args:
        x: Node features (N, D)
        edge_index: Edge indices (2, E)

    Returns:
        HSD scores (N,)
    """
    if edge_index.size(1) == 0:
        return torch.zeros(x.size(0))

    row, col = edge_index
    dist = torch.norm(x[row] - x[col], p=2, dim=1)
    hsd = scatter_mean(dist, row, dim=0, dim_size=x.size(0))
    hsd = torch.nan_to_num(hsd, nan=0.0)
    return hsd


def load_tfinance_dataset(
    path: str | Path,
    hsd_invert: bool = False,
) -> HeteroData:
    """Load tfinance dataset from DGL binary format.

    Args:
        path: Path to tfinance DGL binary file
        hsd_invert: If True, invert HSD values (max - HSD) for homophilic fraud

    Returns:
        HeteroData with node features, labels, HSD, and edge index
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DGL dataset not found: {path}")

    logger.info("Loading tfinance dataset from: %s", path)

    # Load DGL graph
    graphs, label_dict = _load_dgl_graphs(str(path))
    if len(graphs) == 0:
        raise ValueError("No graphs found in DGL dataset")

    g = graphs[0]
    logger.info("  Graph: %d nodes, %d edges", g.num_nodes(), g.num_edges())

    # Extract features
    if "feature" not in g.ndata:
        raise KeyError("Expected 'feature' in node data")
    features = g.ndata["feature"]
    if features.dtype == torch.int64:
        features = features.float()
    logger.info("  Features: shape=%s, dtype=%s", features.shape, features.dtype)

    # Apply log(x+1) transform to raw count/amount features (F0-F7).
    # F8/F9 are already normalized to [0,1] and left unchanged.
    # This matches BWGNN's preprocessing for T-Finance data.
    import numpy as np

    features = features.float()  # ensure float32 for model compatibility

    n_raw_feat = features.shape[1]
    n_log_cols = n_raw_feat  # default: log-transform all
    # Detect already-normalized columns (values in [0,1] only)
    for col_idx in range(n_raw_feat):
        col = features[:, col_idx]
        if col.min() >= 0.0 and col.max() <= 1.0:
            n_log_cols = col_idx  # columns before this are raw
            break
    if n_log_cols > 0:
        features_log = torch.log1p(features[:, :n_log_cols])
        features = torch.cat([features_log, features[:, n_log_cols:]], dim=1)
        logger.info(
            "  log(x+1) applied to columns [0:%d], columns [%d:%d] unchanged", n_log_cols, n_log_cols, n_raw_feat
        )
    else:
        logger.info("  No log transform needed (all features in [0,1])")
    logger.info("  Features after transform: range=[%.4f, %.4f]", features.min().item(), features.max().item())

    # Extract labels - tfinance uses one-hot encoding [normal, fraud]
    if "label" not in g.ndata:
        raise KeyError("Expected 'label' in node data")
    label_raw = g.ndata["label"]

    # Handle one-hot encoding: [1,0]=normal(0), [0,1]=fraud(1)
    if label_raw.ndim == 2 and label_raw.shape[1] == 2:
        labels = label_raw[:, 1].long()  # Take second column as fraud label
        logger.info("  Labels: converted from one-hot, fraud=%d", labels.sum().item())
    else:
        labels = label_raw.long().squeeze()
        logger.info("  Labels: shape=%s", labels.shape)

    n_nodes = features.shape[0]
    n_fraud = int(labels.sum().item())
    fraud_ratio = 100.0 * n_fraud / n_nodes
    logger.info("  Dataset: nodes=%d, fraud=%d (%.2f%%)", n_nodes, n_fraud, fraud_ratio)

    # Convert DGL edge list to edge_index tensor
    src, dst = g.edges()
    edge_index = torch.stack([src, dst], dim=0)

    # Compute HSD
    logger.info("  Computing HSD...")
    hsd = compute_hsd(features, edge_index)

    # Optional HSD inversion for homophilic fraud patterns
    if hsd_invert and hsd.max() > 0:
        hsd = hsd.max() - hsd
        logger.info("  HSD inverted (fraud-is-homophilic mode)")

    # Append HSD as additional feature
    x_full = torch.cat([features, hsd.unsqueeze(1)], dim=1)
    logger.info("  Final features: %d dims (original + HSD)", x_full.shape[1])

    # Log HSD statistics
    hsd_np = hsd.numpy()
    logger.info(
        "  HSD stats: min=%.4f, max=%.4f, mean=%.4f, std=%.4f", hsd_np.min(), hsd_np.max(), hsd_np.mean(), hsd_np.std()
    )

    # Per-class HSD analysis
    fraud_mask = labels == 1
    if fraud_mask.any():
        hsd_fraud = hsd_np[fraud_mask].mean()
        hsd_normal = hsd_np[~fraud_mask].mean()
        logger.info("  HSD(fraud)=%.4f, HSD(normal)=%.4f, delta=%.4f", hsd_fraud, hsd_normal, hsd_fraud - hsd_normal)

    # Build HeteroData (single relation: 'edge')
    data = HeteroData()
    data["review"].x = x_full
    data["review"].y = labels
    data["review"].hsd = hsd
    data["review", "edge", "review"].edge_index = edge_index

    logger.info("tfinance dataset loaded successfully")
    return data
