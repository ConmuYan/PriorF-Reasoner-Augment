"""Configuration system for LG-HGCL.

YAML-driven frozen dataclass configuration with support for benchmark .mat
datasets, training optimisation, ablation switches, and DDP multi-GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Project-level settings."""

    seed: int = 42
    output_dir: str = "runs/default"


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    file: str | None = "train.log"
    console: bool = True


@dataclass(frozen=True)
class DebugConfig:
    """Debug and instrumentation configuration."""

    enabled: bool = False
    log_tensor_shapes: bool = False
    log_every_n_steps: int = 50


@dataclass(frozen=True)
class DataConfig:
    """Dataset and preprocessing configuration."""

    # --- data source ---
    data_source: str = "mat"  # "mat" (.mat benchmark) or "dgl" (DGL binary)
    dataset: str = ""  # dataset name: "amazon" or "yelpchi" (used for unified data routing)
    mat_path: str = ""  # path to .mat file
    dgl_path: str = ""  # path to DGL binary file (tfinance/tsocial)
    processed_dir: str = "processed/default"
    unified_data_dir: str = ""  # non-empty -> load from {unified_data_dir}/{dataset}/seed_{seed}/data.pt

    # --- graph construction ---
    hsd_invert: bool = False  # True -> invert HSD (max-HSD) for homophilic fraud

    # --- data split ratios ---
    train_ratio: float = 0.7  # training set ratio
    val_ratio: float = 0.1  # validation set ratio
    test_ratio: float = 0.2  # test set ratio
    split_seed: int = 717  # random seed for stratified split
    scarcity_ratio: float = 0.0  # 0 = disabled; >0 = keep this fraction of train pool


@dataclass(frozen=True)
class ModelConfig:
    """LG-HGCL model hyperparameters."""

    mlp_hidden: int = 256
    gnn_hidden: int = 256
    out_hidden: int = 256
    num_relations: int = 3
    dropout: float = 0.1
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    # --- ASDA (Adaptive Structural Discrepancy Attention) ---
    use_asda: bool = True  # True -> ASDA, False -> SCRE
    asda_tau: float = 0.1  # Temperature for ASDA edge attention softmax
    # --- SDCL (Structure-Discrepancy Contrastive Loss) ---
    lambda_sdcl: float = 0.0  # SDCL weight (0 = disabled)
    sdcl_tau: float = 0.1  # InfoNCE temperature
    sdcl_anchors: int = 256  # anchors per group per step
    sdcl_negatives: int = 64  # negative samples per anchor group
    sdcl_quantile: float = 0.3  # HSD tail fraction
    sdcl_cosine_decay: bool = False  # cosine-anneal lambda_sdcl to 0
    sdcl_rounds: int = 1  # multi-round anchor sampling (variance reduction)
    # --- SDCL Curriculum Annealing ---
    sdcl_schedule: str = "cosine"  # cosine, constant, increasing, delayed
    sdcl_anneal_warmup: int = 0  # Warmup epochs for SDCL lambda (also used as delayed start point)
    sdcl_anneal_total: int = 0  # Total epochs for SDCL annealing


@dataclass(frozen=True)
class TrainConfig:
    """Training loop hyperparameters."""

    epochs: int = 100
    lr: float = 4e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    # Scheduler
    scheduler_T0: int = 10
    scheduler_Tmult: int = 2
    # Early stopping
    patience: int = 20
    min_delta: float = 1e-4
    # Checkpoint
    resume_path: str | None = None
    save_every: int = 10
    # DDP
    use_ddp: bool = False
    local_rank: int = 0
    # AMP
    use_amp: bool = False
    # Evaluation
    eval_every: int = 1
    save_emb: bool = True


@dataclass(frozen=True)
class AblationConfig:
    """Ablation experiment switches.

    Each flag controls whether the corresponding component is active.
    Priority (for GNN pre-processing): ASDA > SCRE > raw X
      - use_asda=True  -> ASDA pre-processing (primary mode)
      - use_asda=False and use_scre=True -> SCRE (ablation: ASDA vs SCRE)
      - use_asda=False and use_scre=False -> raw X fed to RGCN
    """

    use_asda: bool = True  # False -> fall back to SCRE or raw X
    use_scre: bool = True  # False (when use_asda=False) -> raw X into RGCN
    use_hsd_feature: bool = True  # False -> exclude HSD from node features
    use_sdcl: bool = False  # True -> enable SDCL contrastive loss
    use_mlp_branch: bool = True  # False -> GNN-only
    use_gnn_branch: bool = True  # False -> MLP-only
    use_rur: bool = True  # False -> drop R-U-R edges
    use_rtr: bool = True  # False -> drop R-T-R edges
    use_rsr: bool = True  # False -> drop R-S-R edges


@dataclass(frozen=True)
class AppConfig:
    """Root application configuration."""

    project: ProjectConfig = ProjectConfig()
    logging: LoggingConfig = LoggingConfig()
    debug: DebugConfig = DebugConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    ablation: AblationConfig = AblationConfig()


def load_config(path: str | Path) -> AppConfig:
    """Load an :class:`AppConfig` from a YAML file."""

    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppConfig(
        project=_decode_dataclass(ProjectConfig, raw.get("project", {})),
        logging=_decode_dataclass(LoggingConfig, raw.get("logging", {})),
        debug=_decode_dataclass(DebugConfig, raw.get("debug", {})),
        data=_decode_dataclass(DataConfig, raw.get("data", {})),
        model=_decode_dataclass(ModelConfig, raw.get("model", {})),
        train=_decode_dataclass(TrainConfig, raw.get("train", {})),
        ablation=_decode_dataclass(AblationConfig, raw.get("ablation", {})),
    )


def _decode_dataclass(cls: type[Any], values: dict[str, Any]) -> Any:
    allowed = {k: v for k, v in values.items() if k in cls.__dataclass_fields__}
    return cls(**allowed)


def override_config(cfg: AppConfig, overrides: list[str]) -> AppConfig:
    """Apply CLI overrides like ``train.use_ddp=true`` to *cfg*.

    Supported value types: int, float, bool, str, None.
    """
    from dataclasses import fields, replace

    section_objs = {f.name: getattr(cfg, f.name) for f in fields(AppConfig)}

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override format (expected key=value): {item}")
        key, raw_value = item.split("=", 1)
        parts = key.split(".")
        if len(parts) != 2:
            raise ValueError(f"Override key must be section.field: {key}")
        section, field_name = parts
        if section not in section_objs:
            raise ValueError(f"Unknown config section: {section}")

        sub_cfg = section_objs[section]
        if not hasattr(sub_cfg, field_name):
            raise ValueError(f"Unknown field '{field_name}' in {section}")

        # Infer type from the dataclass field
        target_type = type(getattr(sub_cfg, field_name))
        value = _cast_value(raw_value, target_type)
        section_objs[section] = replace(sub_cfg, **{field_name: value})

    return replace(cfg, **section_objs)


def _cast_value(raw: str, target_type: type):
    """Convert a CLI string to the target type."""
    if target_type is bool or raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    if raw.lower() == "none" or raw.lower() == "null":
        return None
    if target_type is int:
        return int(raw)
    if target_type is float:
        return float(raw)
    return raw
