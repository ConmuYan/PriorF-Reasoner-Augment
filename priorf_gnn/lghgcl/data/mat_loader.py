"""Loader for CARE-GNN benchmark .mat datasets (YelpChi / Amazon).

Loads pre-built graph structure, features, and labels directly from .mat files
published by Dou et al. (CIKM 2020).  Returns a HeteroData object compatible
with the existing LG-HGCL pipeline.

YelpChi.mat relations:  net_rur (same user), net_rsr (same star rating), net_rtr (same month)
Amazon.mat  relations:  net_upu (shared product), net_usu (same star user), net_uvu (top-5% text sim)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from torch_geometric.data import HeteroData
from torch_scatter import scatter_mean

from lghgcl.logging_utils import get_logger

logger = get_logger(__name__)

# Canonical 3-relation mapping: always slot 0, 1, 2
YELP_RELATIONS = ("net_rur", "net_rtr", "net_rsr")
AMAZON_RELATIONS = ("net_upu", "net_usu", "net_uvu")

# Edge-type short names used in HeteroData  (align with existing REL2ID)
REL_NAMES = ("rur", "rtr", "rsr")


def _detect_dataset(mat: dict) -> str:
    """Detect whether the .mat is YelpChi or Amazon based on keys."""
    if "net_rur" in mat:
        return "yelp"
    if "net_upu" in mat:
        return "amazon"
    raise ValueError(f"Cannot detect dataset type from .mat keys: {[k for k in mat if not k.startswith('__')]}")


def _sparse_to_edge_index(adj: sp.spmatrix) -> torch.Tensor:
    """Convert a scipy sparse matrix to a PyG edge_index (2, E) tensor."""
    coo = adj.tocoo()
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)
    # Remove self-loops
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]


def compute_hsd(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Compute Heterophily-Induced Structural Discrepancy (HSD)."""
    if edge_index.size(1) == 0:
        return torch.zeros(x.size(0))
    row, col = edge_index
    dist = torch.norm(x[row] - x[col], p=2, dim=1)
    hsd = scatter_mean(dist, row, dim=0, dim_size=x.size(0))
    hsd = torch.nan_to_num(hsd, nan=0.0)
    return hsd


def load_mat_dataset(
    mat_path: str | Path,
    hsd_invert: bool = False,
) -> HeteroData:
    """Load a .mat benchmark dataset and return HeteroData.

    Args:
        mat_path: Path to YelpChi.mat or Amazon.mat.
        hsd_invert: If True, invert HSD values.

    Returns:
        HeteroData with node features, labels, HSD, and three edge types.
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f".mat file not found: {mat_path}")

    logger.info("Loading benchmark .mat dataset: %s", mat_path.name)
    mat = sio.loadmat(str(mat_path))

    ds_type = _detect_dataset(mat)
    rel_keys = YELP_RELATIONS if ds_type == "yelp" else AMAZON_RELATIONS

    # --- Features ---
    features = mat["features"]
    if sp.issparse(features):
        features = features.toarray()
    features = features.astype(np.float32)

    # --- Labels ---
    labels = mat["label"].flatten().astype(np.int64)

    # --- Edge indices ---
    edge_indices = {}
    for mat_key, rel_name in zip(rel_keys, REL_NAMES, strict=True):
        adj = mat[mat_key]
        ei = _sparse_to_edge_index(adj)
        edge_indices[rel_name] = ei
        logger.info("  %s (%s): %d edges", rel_name, mat_key, ei.size(1))

    # --- Build HeteroData ---
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels)

    # Compute HSD on union of all edges
    union_edge = torch.cat(list(edge_indices.values()), dim=1)
    hsd = compute_hsd(x, union_edge)
    if hsd_invert and hsd.max() > 0:
        hsd = hsd.max() - hsd
        logger.info("  HSD inverted (fraud-is-homophilic mode)")

    # Append HSD as extra feature
    x_full = torch.cat([x, hsd.unsqueeze(1)], dim=1)

    data = HeteroData()
    data["review"].x = x_full
    data["review"].y = y
    data["review"].hsd = hsd
    for rel_name, ei in edge_indices.items():
        data["review", rel_name, "review"].edge_index = ei

    n = x_full.size(0)
    n_fraud = int(y.sum().item())
    logger.info(
        "Benchmark dataset loaded: %s, nodes=%d, features=%d(+1 HSD), fraud=%d (%.2f%%)",
        ds_type,
        n,
        features.shape[1],
        n_fraud,
        100.0 * n_fraud / n,
    )
    for rel_name in REL_NAMES:
        key = ("review", rel_name, "review")
        if key in data.edge_types:
            ne = data[key].edge_index.size(1)
            logger.info("  %s: %d edges, avg_deg=%.2f", rel_name, ne, ne / n)

    # HSD stats
    hsd_np = hsd.numpy()
    logger.info(
        "  HSD: min=%.4f max=%.4f mean=%.4f",
        hsd_np.min(),
        hsd_np.max(),
        hsd_np.mean(),
    )
    fraud_mask = labels == 1
    if fraud_mask.any():
        logger.info(
            "  HSD (fraud): mean=%.4f  HSD (normal): mean=%.4f",
            hsd_np[fraud_mask].mean(),
            hsd_np[~fraud_mask].mean(),
        )

    return data


def save_hetero(data: HeteroData, out_path: str | Path) -> Path:
    """Save a HeteroData object to disk."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    return out_path
