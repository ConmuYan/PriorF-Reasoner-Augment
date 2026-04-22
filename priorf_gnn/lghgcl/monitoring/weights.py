"""Model weight statistics monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lghgcl.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class WeightStats:
    """Summary statistics of a parameter tensor."""

    name: str
    mean: float
    std: float
    min: float
    max: float
    numel: int


def collect_weight_stats(model: torch.nn.Module) -> list[WeightStats]:
    """Collect weight statistics for all trainable parameters."""

    stats: list[WeightStats] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        t = p.detach().float().cpu().numpy()
        stats.append(
            WeightStats(
                name=name,
                mean=float(np.mean(t)),
                std=float(np.std(t)),
                min=float(np.min(t)),
                max=float(np.max(t)),
                numel=int(p.numel()),
            )
        )
    return stats


def save_weight_stats(stats: list[WeightStats], out_path: str | Path) -> Path:
    """Save weight stats to a CSV file."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["name,mean,std,min,max,numel"]
    for s in stats:
        lines.append(f"{s.name},{s.mean},{s.std},{s.min},{s.max},{s.numel}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved weight stats to %s", out_path)
    return out_path
