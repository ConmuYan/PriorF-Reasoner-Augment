"""End-to-end pipeline for LG-HGCL.

Stages:
  1. Load benchmark .mat dataset, split, cache
  2. Train with custom Trainer (TensorBoard, checkpoint, early stop)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
from torch_geometric.data import Data, HeteroData

from lghgcl.config import (
    AppConfig,
    DataConfig,
)
from lghgcl.data.mat_loader import load_mat_dataset, save_hetero
from lghgcl.data.scarcity import stratified_subsample_mask
from lghgcl.data.split import split_data
from lghgcl.exceptions import TrainingError
from lghgcl.logging_utils import get_logger
from lghgcl.models.lg_hgcl import LGHGCLNet
from lghgcl.models.lg_hgcl_v2 import LGHGCLNetV2
from lghgcl.monitoring.summary import summarize_model
from lghgcl.monitoring.weights import collect_weight_stats, save_weight_stats
from lghgcl.training.trainer import Trainer, TrainerConfig
from lghgcl.utils.seed import seed_everything
from lghgcl.visualization import plot_tsne

logger = get_logger(__name__)


def run_pipeline(cfg: AppConfig) -> Path:
    """Run the full LG-HGCL pipeline and return the output directory."""

    output_dir = Path(cfg.project.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.project.seed)

    if cfg.data.unified_data_dir:
        hetero = _stage_unified(cfg.data, cfg.project.seed)
    else:
        processed_dir = Path(cfg.data.processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        hetero = _stage_mat(cfg.data, processed_dir)

    _stage_train(hetero, output_dir, cfg)

    return output_dir


# ------------------------------------------------------------------
# Stages
# ------------------------------------------------------------------


def _stage_mat(cfg: DataConfig, out_dir: Path) -> HeteroData:
    """Load pre-built graph from .mat or .dgl dataset, split, and cache."""
    cache = out_dir / "data.pt"
    if cache.exists():
        try:
            d = torch.load(cache, map_location="cpu", weights_only=False)
            if hasattr(d["review"], "hsd") and hasattr(d["review"], "train_mask"):
                logger.info("Loaded cached benchmark graph from %s", cache)
                return d
        except Exception:
            pass
        logger.info("Cached benchmark graph is outdated. Rebuilding...")

    if cfg.data_source == "dgl":
        from lghgcl.data.dgl_loader import load_tfinance_dataset

        hetero = load_tfinance_dataset(
            path=cfg.dgl_path,
            hsd_invert=cfg.hsd_invert,
        )
    else:
        hetero = load_mat_dataset(
            mat_path=cfg.mat_path,
            hsd_invert=cfg.hsd_invert,
        )

    hetero = split_data(
        hetero,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.split_seed,
    )
    save_hetero(hetero, cache)
    return hetero


_UNIFIED_REL_MAP: dict[str, tuple[str, ...]] = {
    "amazon": ("upu", "usu", "uvu"),
    "yelpchi": ("rur", "rtr", "rsr"),
}


def _stage_unified(cfg: DataConfig, seed: int) -> HeteroData:
    """Load pre-computed unified data from processed_data/{dataset}/seed_{seed}/data.pt."""
    dataset = cfg.dataset
    if not dataset:
        raise ValueError("data.dataset must be set when using unified_data_dir")

    data_path = Path(cfg.unified_data_dir) / dataset / f"seed_{seed}" / "data.pt"
    if not data_path.exists():
        raise FileNotFoundError(f"Unified data file not found: {data_path}. Run comp/generate_unified_data.py first.")

    raw = torch.load(data_path, map_location="cpu", weights_only=False)
    logger.info("Loaded unified data from %s", data_path)

    hsd = raw.hsd
    if cfg.hsd_invert and hsd.max() > 0:
        hsd = hsd.max() - hsd
        logger.info("HSD inverted (fraud-is-homophilic mode)")

    x_full = torch.cat([raw.x, hsd.unsqueeze(1)], dim=1)

    hetero = HeteroData()
    hetero["review"].x = x_full
    hetero["review"].y = raw.y
    hetero["review"].hsd = hsd
    hetero["review"].train_mask = raw.train_mask
    hetero["review"].val_mask = raw.val_mask
    hetero["review"].test_mask = raw.test_mask

    rel_names = _UNIFIED_REL_MAP.get(dataset, ("rur", "rtr", "rsr"))
    for rel_name in rel_names:
        if rel_name in raw.edge_index_dict:
            hetero["review", rel_name, "review"].edge_index = raw.edge_index_dict[rel_name]

    n = x_full.size(0)
    n_train = int(raw.train_mask.sum().item())
    n_val = int(raw.val_mask.sum().item())
    n_test = int(raw.test_mask.sum().item())
    logger.info(
        "Unified dataset: dataset=%s, seed=%d, nodes=%d, features=%d(+1 HSD), train=%d, val=%d, test=%d",
        dataset,
        seed,
        n,
        raw.x.size(1),
        n_train,
        n_val,
        n_test,
    )

    return hetero


def _stage_train(hetero: HeteroData, out_dir: Path, cfg: AppConfig) -> None:
    """Convert to homo, build model, train with Trainer."""
    try:
        data = hetero_to_homo(hetero, cfg.ablation)

        if cfg.data.scarcity_ratio > 0:
            keep = cfg.data.scarcity_ratio / cfg.data.train_ratio
            if keep <= 0 or keep > 1:
                raise TrainingError(
                    f"Invalid scarcity keep_ratio={keep:.4f} "
                    f"(scarcity_ratio={cfg.data.scarcity_ratio}, train_ratio={cfg.data.train_ratio})"
                )
            data.train_mask = stratified_subsample_mask(
                data.train_mask,
                data.y,
                keep_ratio=keep,
                seed=cfg.project.seed,
            )
            logger.info(
                "Scarcity: train nodes %d -> %d (keep=%.2f%%)",
                int(data.train_mask.sum().item() / keep),
                int(data.train_mask.sum().item()),
                keep * 100,
            )

        x_raw = data.x
        y_raw = data.y
        edge_index_raw = data.edge_index
        edge_type_raw = data.edge_type
        hsd_raw = data.hsd
        if x_raw is None or y_raw is None or edge_index_raw is None or edge_type_raw is None or hsd_raw is None:
            raise TrainingError("Homogeneous training data is missing required tensors")
        x = cast(torch.Tensor, x_raw)
        y = cast(torch.Tensor, y_raw)
        edge_index = cast(torch.Tensor, edge_index_raw)
        edge_type = cast(torch.Tensor, edge_type_raw)
        hsd = cast(torch.Tensor, hsd_raw)

        # Determine effective model dimensions based on ablation
        abl = cfg.ablation
        in_dim = int(x.size(1))

        # Select model: ASDA v2 vs SCRE v1
        use_asda = abl.use_asda
        if use_asda:
            model = LGHGCLNetV2(
                in_dim=in_dim,
                mlp_hidden=cfg.model.mlp_hidden,
                gnn_hidden=cfg.model.gnn_hidden,
                out_hidden=cfg.model.out_hidden,
                num_relations=cfg.model.num_relations,
                dropout=cfg.model.dropout,
                use_asda=True,
                use_mlp_branch=abl.use_mlp_branch,
                use_gnn_branch=abl.use_gnn_branch,
                asda_tau=cfg.model.asda_tau,
            )
        else:
            model = LGHGCLNet(
                in_dim=in_dim,
                mlp_hidden=cfg.model.mlp_hidden,
                gnn_hidden=cfg.model.gnn_hidden,
                out_hidden=cfg.model.out_hidden,
                num_relations=cfg.model.num_relations,
                dropout=cfg.model.dropout,
                use_scre=abl.use_scre,
                use_mlp_branch=abl.use_mlp_branch,
                use_gnn_branch=abl.use_gnn_branch,
            )

        trainer_cfg = _build_trainer_config(cfg, out_dir, use_asda)

        trainer = Trainer(model=model, data=data, cfg=trainer_cfg)
        best_metrics = trainer.fit()

        # Post-training artifacts: rank-0 only for DDP
        is_main = int(os.environ.get("RANK", 0)) == 0
        if is_main:
            # Save final metrics
            (out_dir / "best_test_metrics.json").write_text(
                json.dumps(best_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # Model summary + full reproducible config
            summary = asdict(summarize_model(model))
            summary["config"] = {
                "project": asdict(cfg.project),
                "data": asdict(cfg.data),
                "model": asdict(cfg.model),
                "train": asdict(cfg.train),
                "ablation": asdict(cfg.ablation),
            }
            summary["trainer_config"] = asdict(trainer_cfg)
            (out_dir / "model_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # Weight stats
            stats = collect_weight_stats(model)
            save_weight_stats(stats, out_dir / "weight_stats.csv")

            # t-SNE visualisation
            model.eval()
            with torch.no_grad():
                device = next(model.parameters()).device
                if use_asda:
                    _ = model(x.to(device), edge_index.to(device), edge_type.to(device), hsd.to(device))
                else:
                    _ = model(x.to(device), edge_index.to(device), edge_type.to(device))
                if model.emb is not None:
                    parts = [e for e in model.emb if e is not None]
                    if parts:
                        z_final = torch.cat(parts, dim=-1).cpu().numpy()
                        plot_tsne(z_final, y.cpu().numpy(), out_dir / "tsne_embeddings.png")

            logger.info("Pipeline complete. Artifacts saved to %s", out_dir)

    except Exception as e:
        raise TrainingError("Training pipeline failed") from e


def _build_trainer_config(cfg: AppConfig, out_dir: Path, use_asda: bool) -> TrainerConfig:
    lambda_sdcl = cfg.model.lambda_sdcl if cfg.ablation.use_sdcl else 0.0

    return TrainerConfig(
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        grad_clip=cfg.train.grad_clip,
        focal_alpha=cfg.model.focal_alpha,
        focal_gamma=cfg.model.focal_gamma,
        lambda_sdcl=lambda_sdcl,
        sdcl_tau=cfg.model.sdcl_tau,
        sdcl_anchors=cfg.model.sdcl_anchors,
        sdcl_negatives=cfg.model.sdcl_negatives,
        sdcl_quantile=cfg.model.sdcl_quantile,
        sdcl_cosine_decay=cfg.model.sdcl_cosine_decay,
        sdcl_schedule=cfg.model.sdcl_schedule,
        sdcl_rounds=cfg.model.sdcl_rounds,
        sdcl_anneal_warmup=cfg.model.sdcl_anneal_warmup,
        sdcl_anneal_total=cfg.model.sdcl_anneal_total,
        use_asda=use_asda,
        scheduler_T0=cfg.train.scheduler_T0,
        scheduler_Tmult=cfg.train.scheduler_Tmult,
        patience=cfg.train.patience,
        min_delta=cfg.train.min_delta,
        save_dir=str(out_dir),
        resume_path=cfg.train.resume_path,
        save_every=cfg.train.save_every,
        use_ddp=cfg.train.use_ddp,
        local_rank=cfg.train.local_rank,
        use_amp=cfg.train.use_amp,
        eval_every=cfg.train.eval_every,
    )


def _maybe_drop_hsd_feature(x: torch.Tensor, hsd: torch.Tensor, use_hsd_feature: bool) -> torch.Tensor:
    if use_hsd_feature or x.size(1) == 0:
        return x

    if x.size(0) == hsd.size(0) and torch.allclose(x[:, -1], hsd.to(x.dtype)):
        return x[:, :-1]

    logger.warning(
        "use_hsd_feature=false requested, but the last feature column does not match data.hsd; keeping input features"
    )
    return x


def hetero_to_homo(hetero: HeteroData, ablation: Data | object | None = None) -> Data:
    """Convert review-only HeteroData to homogeneous Data for training.

    Supports both:
    - Multi-relational YelpChi: rur, rtr, rsr
    - Multi-relational Amazon: upu, usu, uvu
    - Single-relational (tfinance/tsocial): edge
    """
    use_hsd_feature = getattr(ablation, "use_hsd_feature", True)
    use_relations = {
        "rur": getattr(ablation, "use_rur", True),
        "rtr": getattr(ablation, "use_rtr", True),
        "rsr": getattr(ablation, "use_rsr", True),
        "upu": getattr(ablation, "use_upu", getattr(ablation, "use_rur", True)),
        "usu": getattr(ablation, "use_usu", getattr(ablation, "use_rtr", True)),
        "uvu": getattr(ablation, "use_uvu", getattr(ablation, "use_rsr", True)),
    }

    hsd = hetero["review"].hsd
    x = _maybe_drop_hsd_feature(hetero["review"].x, hsd, use_hsd_feature)
    y = hetero["review"].y

    edge_indexes = []
    edge_types = []

    rel2id = {"rur": 0, "rtr": 1, "rsr": 2, "upu": 0, "usu": 1, "uvu": 2}
    for rel, rel_id in rel2id.items():
        if not use_relations[rel]:
            continue
        key = ("review", rel, "review")
        if key in hetero.edge_types:
            e = hetero[key].edge_index
            edge_indexes.append(e)
            edge_types.append(torch.full((e.size(1),), rel_id, dtype=torch.long))

    single_key = ("review", "edge", "review")
    if not edge_indexes and single_key in hetero.edge_types:
        e = hetero[single_key].edge_index
        edge_indexes.append(e)
        edge_types.append(torch.zeros(e.size(1), dtype=torch.long))

    edge_index = torch.cat(edge_indexes, dim=1) if edge_indexes else torch.empty((2, 0), dtype=torch.long)
    edge_type = torch.cat(edge_types, dim=0) if edge_types else torch.empty((0,), dtype=torch.long)

    train_mask = getattr(hetero["review"], "train_mask", None)
    val_mask = getattr(hetero["review"], "val_mask", None)
    test_mask = getattr(hetero["review"], "test_mask", None)

    return Data(
        x=x,
        y=y,
        hsd=hsd,
        edge_index=edge_index,
        edge_type=edge_type,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
