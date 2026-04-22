"""Model summary and visualization helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lghgcl.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelSummary:
    """Basic model summary information."""

    num_parameters: int
    trainable_parameters: int
    estimated_size_mb: float


def summarize_model(model: torch.nn.Module) -> ModelSummary:
    """Compute basic model size statistics."""

    total = 0
    trainable = 0
    bytes_total = 0
    for p in model.parameters():
        total += int(p.numel())
        bytes_total += int(p.numel() * p.element_size())
        if p.requires_grad:
            trainable += int(p.numel())
    return ModelSummary(
        num_parameters=total,
        trainable_parameters=trainable,
        estimated_size_mb=float(bytes_total / (1024 * 1024)),
    )
