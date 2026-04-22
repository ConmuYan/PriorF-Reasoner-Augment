"""Custom training loop for LG-HGCL with full experiment support.

Features:
- TensorBoard logging (scalar metrics per epoch)
- Checkpoint save / resume
- Early stopping on validation AUPRC
- Learning rate scheduling (CosineAnnealingWarmRestarts)
- Gradient clipping
- Multi-GPU via PyTorch DDP (optional)
- Full-batch training
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.data import Data

from lghgcl.logging_utils import get_logger
from lghgcl.metrics import evaluate_performance
from lghgcl.models.losses import focal_loss_with_logits, sdcl_loss_v4

logger = get_logger(__name__)


@dataclass
class TrainerConfig:
    """Training hyperparameters."""

    epochs: int = 100
    lr: float = 4e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    # SDCL
    lambda_sdcl: float = 0.0
    sdcl_tau: float = 0.1
    sdcl_anchors: int = 256
    sdcl_negatives: int = 64
    sdcl_quantile: float = 0.3
    sdcl_schedule: str = "cosine"  # cosine, constant, increasing, delayed
    sdcl_cosine_decay: bool = False  # legacy: True overrides schedule to "cosine"
    sdcl_rounds: int = 1
    sdcl_anneal_warmup: int = 0
    sdcl_anneal_total: int = 0
    # Model
    use_asda: bool = True  # True -> v2 (ASDA), False -> v1 (SCRE)
    # Scheduler
    scheduler_T0: int = 10
    scheduler_Tmult: int = 2
    # Early stopping
    patience: int = 20
    min_delta: float = 1e-4
    # Checkpoint
    save_dir: str = "runs/default"
    resume_path: str | None = None
    save_every: int = 10
    # DDP
    use_ddp: bool = False
    local_rank: int = 0
    # Mixed precision
    use_amp: bool = False
    # Evaluation
    eval_every: int = 1


class Trainer:
    """Self-contained LG-HGCL training loop."""

    def __init__(
        self,
        model: nn.Module,
        data: Data,
        cfg: TrainerConfig,
    ) -> None:
        self.cfg = cfg
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Rank info (for DDP rank-0-only I/O)
        self._rank = int(os.environ.get("RANK", 0))
        self._is_main = self._rank == 0

        # Device
        if cfg.use_ddp:
            self.device = torch.device(f"cuda:{cfg.local_rank}")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # Data → device
        self.data = data.to(self.device)

        # Model
        self.model = model.to(self.device)
        if cfg.use_ddp:
            self.model = DDP(self.model, device_ids=[cfg.local_rank])

        # Optimiser & scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=cfg.scheduler_T0,
            T_mult=cfg.scheduler_Tmult,
        )
        self.scaler = GradScaler("cuda", enabled=cfg.use_amp)

        # TensorBoard (rank-0 only)
        if self._is_main:
            self.writer = SummaryWriter(log_dir=str(self.save_dir / "tb"))
        else:
            self.writer = None

        # Early stopping state
        self._best_metric = -float("inf")
        self._patience_counter = 0
        self._best_epoch = 0
        self.start_epoch = 0

        # History
        self.history: list[dict] = []

        # Resume
        if cfg.resume_path and Path(cfg.resume_path).exists():
            self._load_checkpoint(cfg.resume_path)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def fit(self) -> dict:
        """Run full training loop. Returns best test metrics."""
        if self._is_main:
            logger.info(
                "Training: epochs=%d lr=%.1e device=%s ddp=%s",
                self.cfg.epochs,
                self.cfg.lr,
                self.device,
                self.cfg.use_ddp,
            )
        best_test_metrics: dict = {}

        for epoch in range(self.start_epoch, self.cfg.epochs):
            t0 = time.time()
            train_loss = self._train_one_epoch(epoch)
            elapsed = time.time() - t0

            # Clear CUDA cache each epoch to prevent memory fragmentation/leakage
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Log train loss (rank-0 only)
            if self.writer is not None:
                self.writer.add_scalar("loss/train_total", train_loss["total"], epoch)
                self.writer.add_scalar("loss/focal", train_loss["focal"], epoch)
                self.writer.add_scalar("loss/sdcl", train_loss["sdcl"], epoch)
                self.writer.add_scalar("loss/lambda_sdcl", train_loss["lambda_sdcl"], epoch)
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

            # Validation
            val_metrics: dict = {}
            test_metrics: dict = {}
            if epoch % self.cfg.eval_every == 0 or epoch == self.cfg.epochs - 1:
                val_metrics = self._evaluate("val")
                test_metrics = self._evaluate("test")
                if self.writer is not None:
                    for k, v in val_metrics.items():
                        if isinstance(v, (int, float)):
                            self.writer.add_scalar(f"val/{k}", v, epoch)
                    for k, v in test_metrics.items():
                        if isinstance(v, (int, float)):
                            self.writer.add_scalar(f"test/{k}", v, epoch)

            self.scheduler.step()

            record = {
                "epoch": epoch,
                "elapsed": elapsed,
                **{f"train_{k}": v for k, v in train_loss.items()},
                **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))},
                **{f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))},
            }
            self.history.append(record)

            if self._is_main:
                logger.info(
                    "Epoch %d/%d  loss=%.4f  val_AUPRC=%.4f  test_AUPRC=%.4f  %.1fs",
                    epoch + 1,
                    self.cfg.epochs,
                    train_loss["total"],
                    val_metrics.get("AUPRC", 0.0),
                    test_metrics.get("AUPRC", 0.0),
                    elapsed,
                )

            # Early stopping (on val AUPRC)
            current_metric = val_metrics.get("AUPRC", 0.0)
            if current_metric > self._best_metric + self.cfg.min_delta:
                self._best_metric = current_metric
                self._patience_counter = 0
                self._best_epoch = epoch
                best_test_metrics = test_metrics
                if self._is_main:
                    self._save_checkpoint(epoch, is_best=True)
            else:
                self._patience_counter += 1

            # Periodic save (rank-0 only)
            if (epoch + 1) % self.cfg.save_every == 0 and self._is_main:
                self._save_checkpoint(epoch, is_best=False)

            if self._patience_counter >= self.cfg.patience:
                if self._is_main:
                    logger.info(
                        "Early stopping at epoch %d (best=%d, val_AUPRC=%.4f)",
                        epoch,
                        self._best_epoch,
                        self._best_metric,
                    )
                break

        # DDP barrier: wait for all ranks before loading best model
        if self.cfg.use_ddp and dist.is_initialized():
            dist.barrier()

        if self.writer is not None:
            self.writer.close()
        if self._is_main:
            self._save_history()

        # Load best model
        best_path = self.save_dir / "best_model.pt"
        if best_path.exists():
            self._load_model_weights(best_path)
            if self._is_main:
                logger.info("Loaded best model from epoch %d", self._best_epoch)

        return best_test_metrics

    # ------------------------------------------------------------------
    #  Train one epoch (full-batch)
    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        d = self.data
        train_mask_raw = d.train_mask
        y_raw = d.y
        hsd_raw = d.hsd
        if train_mask_raw is None or y_raw is None or hsd_raw is None:
            raise ValueError("Training data is missing train_mask, y, or hsd")
        train_mask = cast(torch.Tensor, train_mask_raw)
        y = cast(torch.Tensor, y_raw)
        hsd_all = cast(torch.Tensor, hsd_raw)

        self.optimizer.zero_grad()

        with autocast("cuda", enabled=self.cfg.use_amp):
            logits = self._forward()
            target_logits = logits[train_mask]
            target_y = y[train_mask].float()

            loss_focal = focal_loss_with_logits(
                target_logits,
                target_y,
                alpha=self.cfg.focal_alpha,
                gamma=self.cfg.focal_gamma,
            )

            # SDCL: structure-discrepancy contrastive on embeddings
            loss_sdcl_val = torch.zeros(1, device=target_logits.device)
            if self.cfg.lambda_sdcl > 0:
                hsd = hsd_all[train_mask].float()
                m = self.model.module if isinstance(self.model, DDP) else self.model
                z_proj = m.z_proj[train_mask]
                loss_sdcl_val = sdcl_loss_v4(
                    z_proj,
                    target_y,
                    hsd,
                    temperature=self.cfg.sdcl_tau,
                    num_anchors=self.cfg.sdcl_anchors,
                    quantile=self.cfg.sdcl_quantile,
                    num_rounds=self.cfg.sdcl_rounds,
                )

            # SDCL lambda scheduling
            if self.cfg.lambda_sdcl > 0:
                # Resolve effective schedule (legacy sdcl_cosine_decay compat)
                schedule = self.cfg.sdcl_schedule
                if self.cfg.sdcl_cosine_decay and schedule not in ("increasing", "delayed"):
                    schedule = "cosine"
                # Annealed curriculum takes priority if configured
                if self.cfg.sdcl_anneal_total > 0:
                    warmup = self.cfg.sdcl_anneal_warmup
                    total = self.cfg.sdcl_anneal_total
                    if epoch < warmup:
                        lambda_sdcl = self.cfg.lambda_sdcl * (epoch / max(1, warmup))
                    elif epoch < total:
                        progress = (epoch - warmup) / max(1, total - warmup)
                        lambda_sdcl = self.cfg.lambda_sdcl * 0.5 * (1 + math.cos(math.pi * progress))
                    else:
                        lambda_sdcl = 0.0
                elif schedule == "cosine":
                    progress = epoch / max(self.cfg.epochs - 1, 1)
                    lambda_sdcl = self.cfg.lambda_sdcl * 0.5 * (1 + math.cos(math.pi * progress))
                elif schedule == "increasing":
                    progress = epoch / max(self.cfg.epochs - 1, 1)
                    lambda_sdcl = self.cfg.lambda_sdcl * progress
                elif schedule == "delayed":
                    start = self.cfg.sdcl_anneal_warmup  # reuse as delay point
                    if epoch < start:
                        lambda_sdcl = 0.0
                    else:
                        remaining = self.cfg.epochs - start
                        progress = (epoch - start) / max(remaining - 1, 1)
                        lambda_sdcl = self.cfg.lambda_sdcl * (1 + math.cos(math.pi * (1 - progress))) / 2
                else:  # "constant" or any unknown value
                    lambda_sdcl = self.cfg.lambda_sdcl
            else:
                lambda_sdcl = 0.0

            loss = loss_focal + lambda_sdcl * loss_sdcl_val

        self.scaler.scale(loss).backward()
        if self.cfg.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "total": float(loss),
            "focal": float(loss_focal),
            "sdcl": float(loss_sdcl_val),
            "lambda_sdcl": float(lambda_sdcl),
        }

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, split: str) -> dict:
        self.model.eval()
        d = self.data
        mask_raw = getattr(d, f"{split}_mask", None)
        y_raw = d.y
        if mask_raw is None or y_raw is None:
            return {}
        mask = cast(torch.Tensor, mask_raw)
        y = cast(torch.Tensor, y_raw)
        if mask.sum() == 0:
            return {}

        logits = self._forward()
        probs = torch.sigmoid(logits[mask]).cpu().numpy()
        y_true = y[mask].cpu().numpy()
        return evaluate_performance(y_true, probs)

    # ------------------------------------------------------------------
    #  Forward helper (unwraps DDP)
    # ------------------------------------------------------------------
    def _forward(self) -> torch.Tensor:
        d = self.data
        m = self.model.module if isinstance(self.model, DDP) else self.model
        x_raw = d.x
        edge_index_raw = d.edge_index
        edge_type_raw = d.edge_type
        if x_raw is None or edge_index_raw is None or edge_type_raw is None:
            raise ValueError("Training data is missing x, edge_index, or edge_type")
        x = cast(torch.Tensor, x_raw)
        edge_index = cast(torch.Tensor, edge_index_raw)
        edge_type = cast(torch.Tensor, edge_type_raw)

        # ASDA v2 model takes hsd as 4th argument; SCRE v1 takes 3
        if getattr(self.cfg, "use_asda", False):
            hsd_raw = d.hsd
            if hsd_raw is None:
                raise ValueError("ASDA training requires data.hsd")
            hsd = cast(torch.Tensor, hsd_raw)
            return m(x, edge_index, edge_type, hsd)
        return m(x, edge_index, edge_type)

    # ------------------------------------------------------------------
    #  Checkpoint
    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        m = self.model.module if isinstance(self.model, DDP) else self.model
        state = {
            "epoch": epoch,
            "model_state_dict": m.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self._best_metric,
            "patience_counter": self._patience_counter,
            "best_epoch": self._best_epoch,
        }
        if is_best:
            torch.save(state, self.save_dir / "best_model.pt")
        else:
            torch.save(state, self.save_dir / f"checkpoint_epoch{epoch}.pt")

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        m = self.model.module if isinstance(self.model, DDP) else self.model
        m.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self._best_metric = ckpt.get("best_metric", -float("inf"))
        self._patience_counter = ckpt.get("patience_counter", 0)
        self._best_epoch = ckpt.get("best_epoch", 0)
        if self._is_main:
            logger.info("Resumed from epoch %d", self.start_epoch)

    def _load_model_weights(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        m = self.model.module if isinstance(self.model, DDP) else self.model
        m.load_state_dict(ckpt["model_state_dict"])

    def _save_history(self) -> None:
        (self.save_dir / "training_history.json").write_text(
            json.dumps(self.history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
