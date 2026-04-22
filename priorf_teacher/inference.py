"""Teacher inference bridge: run PriorF-GNN forward and emit TeacherExportRecords.

Produces one record per node, tagged by population from the canonical split_vector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from scipy.io import loadmat

from lghgcl.models.lg_hgcl_v2 import LGHGCLNetV2

from priorf_teacher.schema import (
    DatasetName,
    GraphRegime,
    NeighborSummary,
    PopulationName,
    RelationProfile,
    TeacherExportRecord,
)


def _load_model(
    checkpoint_path: str | Path,
    *,
    device: str = "cuda",
    **model_kwargs,
) -> LGHGCLNetV2:
    """Load a LGHGCLNetV2 from a .pt checkpoint using arbitrary hyperparams."""
    model = LGHGCLNetV2(**model_kwargs)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _load_canonical_flat(canonical_mat_path: str | Path) -> dict[str, torch.Tensor | np.ndarray]:
    """Load canonical graph_data `.mat` into flat tensors for LGHGCLNetV2.

    The Reasoner asset contract stores canonical keys (`x`,
    `ground_truth_label`, `split_vector`, `relation_0..2`) rather than the
    legacy CARE-GNN keys.  This loader avoids importing the teacher repository's
    legacy loader, whose training dependencies can be unavailable in export-only
    environments.
    """

    raw = loadmat(str(canonical_mat_path), squeeze_me=True)
    features = raw["x"]
    if sp.issparse(features):
        features = features.toarray()
    x_base = torch.from_numpy(np.asarray(features, dtype=np.float32))
    y = torch.from_numpy(np.asarray(raw["ground_truth_label"]).reshape(-1).astype(np.int64))
    split_vector = np.asarray(raw["split_vector"]).astype(str)

    edge_indexes: list[torch.Tensor] = []
    edge_types: list[torch.Tensor] = []
    for rel_id in range(3):
        edge_index = _relation_to_edge_index(raw[f"relation_{rel_id}"])
        edge_indexes.append(edge_index)
        edge_types.append(torch.full((edge_index.size(1),), rel_id, dtype=torch.long))

    edge_index = torch.cat(edge_indexes, dim=1) if edge_indexes else torch.empty((2, 0), dtype=torch.long)
    edge_type = torch.cat(edge_types, dim=0) if edge_types else torch.empty((0,), dtype=torch.long)
    hsd = _compute_hsd_without_torch_scatter(x_base, edge_index)
    x = torch.cat([x_base, hsd.unsqueeze(1)], dim=1)
    return {
        "x": x,
        "y": y,
        "hsd": hsd,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "split_vector": split_vector,
    }


def _relation_to_edge_index(relation: object) -> torch.Tensor:
    """Convert a canonical sparse/dense relation matrix into edge_index."""

    if sp.issparse(relation):
        coo = relation.tocoo()
        row = torch.from_numpy(coo.row.astype(np.int64))
        col = torch.from_numpy(coo.col.astype(np.int64))
    else:
        matrix = np.asarray(relation)
        row_np, col_np = np.nonzero(matrix)
        row = torch.from_numpy(row_np.astype(np.int64))
        col = torch.from_numpy(col_np.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)
    keep = edge_index[0] != edge_index[1]
    return edge_index[:, keep]


def _compute_hsd_without_torch_scatter(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Compute HSD mean edge-distance per source node with built-in torch ops."""

    if edge_index.size(1) == 0:
        return torch.zeros(x.size(0), dtype=x.dtype)
    row, col = edge_index
    dist = torch.norm(x[row] - x[col], p=2, dim=1)
    sums = torch.zeros(x.size(0), dtype=dist.dtype)
    counts = torch.zeros(x.size(0), dtype=dist.dtype)
    sums.scatter_add_(0, row, dist)
    counts.scatter_add_(0, row, torch.ones_like(dist))
    hsd = sums / counts.clamp(min=1)
    return torch.nan_to_num(hsd, nan=0.0)


def _precompute_adjacency_buckets(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_nodes: int,
) -> dict[str, np.ndarray]:
    """Precompute per-node outgoing edges grouped by relation."""
    row = edge_index[0].cpu().numpy().astype(np.int64)
    col = edge_index[1].cpu().numpy().astype(np.int64)
    etype = edge_type.cpu().numpy().astype(np.int64)
    # Sort by source node for contiguous slicing.
    order = np.argsort(row, kind="stable")
    row_s = row[order]
    col_s = col[order]
    etype_s = etype[order]
    # Row pointer: start index per node.
    row_ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    counts = np.bincount(row_s, minlength=num_nodes)
    row_ptr[1:] = np.cumsum(counts)
    return {"row_ptr": row_ptr, "col": col_s, "etype": etype_s}


def _compute_node_relation_profile_fast(
    node_id: int,
    buckets: dict[str, np.ndarray],
    hsd: np.ndarray,
) -> RelationProfile:
    start = int(buckets["row_ptr"][node_id])
    end = int(buckets["row_ptr"][node_id + 1])
    if end == start:
        return RelationProfile(
            total_relations=3,
            active_relations=0,
            max_relation_neighbor_count=0,
            mean_relation_neighbor_count=0.0,
            max_relation_discrepancy=0.0,
            mean_relation_discrepancy=0.0,
        )
    neighbors = buckets["col"][start:end]
    etypes = buckets["etype"][start:end]
    active = np.unique(etypes)
    counts = [int((etypes == t).sum()) for t in active]
    max_count = int(max(counts))
    mean_count = float(np.mean(counts))
    discrep = np.abs(hsd[neighbors] - float(hsd[node_id]))
    return RelationProfile(
        total_relations=3,
        active_relations=int(len(active)),
        max_relation_neighbor_count=max_count,
        mean_relation_neighbor_count=mean_count,
        max_relation_discrepancy=float(discrep.max()),
        mean_relation_discrepancy=float(discrep.mean()),
    )


def _compute_node_neighbor_summary_fast(
    node_id: int,
    buckets: dict[str, np.ndarray],
    labels: np.ndarray,
) -> NeighborSummary:
    start = int(buckets["row_ptr"][node_id])
    end = int(buckets["row_ptr"][node_id + 1])
    if end == start:
        return NeighborSummary(
            total_neighbors=0,
            labeled_neighbors=0,
            positive_neighbors=0,
            negative_neighbors=0,
            unlabeled_neighbors=0,
        )
    neighbors = buckets["col"][start:end]
    nl = labels[neighbors]
    positive = int((nl == 1).sum())
    negative = int((nl == 0).sum())
    labeled = positive + negative
    total = int(end - start)
    unlabeled = total - labeled
    return NeighborSummary(
        total_neighbors=total,
        labeled_neighbors=labeled,
        positive_neighbors=positive,
        negative_neighbors=negative,
        unlabeled_neighbors=unlabeled,
    )


def _build_records(
    *,
    dataset_name: DatasetName,
    population_name: PopulationName,
    node_ids: np.ndarray,
    labels: np.ndarray,
    logits: np.ndarray,
    hsd: np.ndarray,
    asda_switch: np.ndarray,
    mlp_logits: np.ndarray,
    gnn_logits: np.ndarray,
    buckets: dict[str, np.ndarray],
    full_hsd: np.ndarray,
    full_labels: np.ndarray,
    teacher_model_name: str,
    teacher_checkpoint: str,
    graph_regime: GraphRegime,
) -> list[TeacherExportRecord]:
    """Build TeacherExportRecords for a single population."""
    probs = 1.0 / (1.0 + np.exp(-logits))

    # hsd quantile computed on full graph, sliced for this population
    ranks_full = np.argsort(np.argsort(full_hsd))
    full_quantiles = (ranks_full + 1) / len(full_hsd)
    # high_hsd_flag threshold is also full-graph
    hsd_mean = float(full_hsd.mean())
    hsd_std = float(full_hsd.std())

    records: list[TeacherExportRecord] = []
    for idx, nid in enumerate(node_ids):
        relation_profile = _compute_node_relation_profile_fast(
            int(nid), buckets, full_hsd
        )
        neighbor_summary = _compute_node_neighbor_summary_fast(
            int(nid), buckets, full_labels
        )
        records.append(
            TeacherExportRecord(
                dataset_name=dataset_name,
                teacher_model_name=teacher_model_name,
                teacher_checkpoint=teacher_checkpoint,
                population_name=population_name,
                node_id=int(nid),
                ground_truth_label=int(labels[idx]),  # type: ignore[arg-type]
                teacher_prob=float(probs[idx]),
                teacher_logit=float(logits[idx]),
                hsd=float(hsd[idx]),
                hsd_quantile=float(full_quantiles[int(nid)]),
                asda_switch=bool(asda_switch[idx]),
                mlp_logit=float(mlp_logits[idx]),
                gnn_logit=float(gnn_logits[idx]),
                branch_gap=float(mlp_logits[idx] - gnn_logits[idx]),
                relation_profile=relation_profile,
                neighbor_summary=neighbor_summary,
                high_hsd_flag=bool(full_hsd[int(nid)] > hsd_mean + hsd_std),
                graph_regime=graph_regime,
            )
        )
    return records


def run_teacher_inference(
    *,
    dataset_name: DatasetName,
    legacy_mat_path: str | Path,
    canonical_mat_path: str | Path,
    checkpoint_path: str | Path,
    model_hyperparams: dict[str, Any],
    teacher_model_name: str,
    teacher_checkpoint: str,
    graph_regime: GraphRegime = GraphRegime.TRANSDUCTIVE_STANDARD,
    device: str = "cuda",
) -> dict[str, list[TeacherExportRecord]]:
    """Run full-graph inference and return records keyed by population."""

    # 1. Load canonical graph_data asset directly.  `legacy_mat_path` remains in
    # the signature for backward CLI compatibility but is not trusted as the
    # source of split/population semantics.
    del legacy_mat_path
    flat = _load_canonical_flat(canonical_mat_path)
    split_vector = flat["split_vector"]
    all_node_ids = np.arange(len(split_vector), dtype=np.int64)

    x = flat["x"]
    y = flat["y"].numpy()
    hsd = flat["hsd"]
    edge_index = flat["edge_index"]
    edge_type = flat["edge_type"]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, np.ndarray)
    assert isinstance(hsd, torch.Tensor)
    assert isinstance(edge_index, torch.Tensor)
    assert isinstance(edge_type, torch.Tensor)

    # 3. Load model
    model = _load_model(str(checkpoint_path), **model_hyperparams, device=device)

    # 4. Forward pass (full graph)
    with torch.no_grad():
        x_dev = x.to(device)
        ei_dev = edge_index.to(device)
        et_dev = edge_type.to(device)
        hsd_dev = hsd.to(device)

        logits = model(x_dev, ei_dev, et_dev, hsd_dev).cpu().numpy()

        # Extract branch embeddings from last forward
        z_mlp, z_gnn = model.emb
        assert z_mlp is not None and z_gnn is not None
        z_mlp = z_mlp.cpu()
        z_gnn = z_gnn.cpu()

        # MLP-only logit
        mlp_input = torch.cat([z_mlp, torch.zeros_like(z_mlp)], dim=-1).to(device)
        mlp_logits = model.out(mlp_input).squeeze(-1).cpu().numpy()

        # GNN-only logit
        gnn_input = torch.cat([torch.zeros_like(z_gnn), z_gnn], dim=-1).to(device)
        gnn_logits = model.out(gnn_input).squeeze(-1).cpu().numpy()

        # ASDA switch (node-level frequency switch α_bar)
        if model.use_asda and model.asda is not None:
            hsd_norm = (hsd_dev - hsd_dev.mean()) / (hsd_dev.std() + 1e-8)
            alpha_bar = model.asda.node_switch(hsd_norm.unsqueeze(1)).squeeze(-1)
            asda_switch = (alpha_bar > 0.5).cpu().numpy()
        else:
            asda_switch = np.zeros(len(hsd), dtype=bool)

    # 5. Build records per population
    population_map: dict[str, PopulationName] = {
        "train": PopulationName.TRAIN,
        "validation": PopulationName.VALIDATION,
        "final_test": PopulationName.FINAL_TEST,
    }
    result: dict[str, list[TeacherExportRecord]] = {}
    hsd_np = hsd.numpy()
    num_nodes = x.shape[0]
    buckets = _precompute_adjacency_buckets(edge_index, edge_type, num_nodes)
    for pop_str, pop_enum in population_map.items():
        mask = split_vector == pop_str
        if not mask.any():
            continue
        indices = np.where(mask)[0]
        result[pop_str] = _build_records(
            dataset_name=dataset_name,
            population_name=pop_enum,
            node_ids=all_node_ids[indices],
            labels=y[indices],
            logits=logits[indices],
            hsd=hsd_np[indices],
            asda_switch=asda_switch[indices],
            mlp_logits=mlp_logits[indices],
            gnn_logits=gnn_logits[indices],
            buckets=buckets,
            full_hsd=hsd_np,
            full_labels=y,
            teacher_model_name=teacher_model_name,
            teacher_checkpoint=teacher_checkpoint,
            graph_regime=graph_regime,
        )

    return result
