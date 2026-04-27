"""Lightweight TensorBoard logging adapter shared by Stage-1 and Stage-2 drivers.

Wraps ``torch.utils.tensorboard.SummaryWriter`` behind a small interface so
the training scripts can log scalars without caring whether TensorBoard
is enabled.  When disabled (``enabled=False``) every method is a no-op,
which keeps the training loops branch-free.

The writer is created lazily so importing this module never touches the
filesystem.  Logs are written under ``log_dir`` as TensorBoard event
files; pointing ``tensorboard --logdir <log_dir>`` at that directory
shows the run in real time.

This module intentionally does NOT depend on ``wandb``.  The repo's
runs happen on offline GPU hosts with no outbound network, so wandb
would require operator setup (API key + ``wandb login``) that is out
of scope for this driver-level integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


class TensorBoardLogger:
    """Thin SummaryWriter wrapper with a no-op disabled mode."""

    def __init__(self, log_dir: Path | str | None, *, enabled: bool = True) -> None:
        self._enabled = bool(enabled and log_dir is not None)
        self._writer: Any | None = None
        self._log_dir: Path | None = Path(log_dir) if log_dir is not None else None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def log_dir(self) -> Path | None:
        return self._log_dir

    def _ensure_writer(self) -> Any | None:
        if not self._enabled:
            return None
        if self._writer is not None:
            return self._writer
        from torch.utils.tensorboard import SummaryWriter

        assert self._log_dir is not None
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self._log_dir))
        return self._writer

    def log_scalar(self, tag: str, value: float | int | None, step: int) -> None:
        if not self._enabled or value is None:
            return
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return
        writer = self._ensure_writer()
        if writer is None:
            return
        writer.add_scalar(tag, scalar, int(step))

    def log_scalars(self, prefix: str, values: Mapping[str, float | int | None], step: int) -> None:
        if not self._enabled:
            return
        for name, value in values.items():
            tag = f"{prefix}/{name}" if prefix else name
            self.log_scalar(tag, value, step)

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.flush()
            finally:
                self._writer.close()
                self._writer = None
