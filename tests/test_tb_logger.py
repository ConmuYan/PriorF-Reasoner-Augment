"""Smoke tests for the shared TensorBoard logger used by Stage 1/2 drivers."""

from __future__ import annotations

from pathlib import Path

import pytest

from train._tb_logger import TensorBoardLogger


def _read_event_scalars(log_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Parse all tfevent files under ``log_dir`` into a {tag: [(step, value), ...]} map."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    acc.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in acc.Tags().get("scalars", []):
        out[tag] = [(event.step, float(event.value)) for event in acc.Scalars(tag)]
    return out


def test_tb_logger_writes_event_file_when_enabled(tmp_path: Path) -> None:
    log_dir = tmp_path / "tb"
    logger = TensorBoardLogger(log_dir, enabled=True)
    assert logger.enabled is True
    assert logger.log_dir == log_dir

    logger.log_scalar("train/L_gen", 0.42, 1)
    logger.log_scalars(
        "validation",
        {"auprc": 0.13, "auroc": 0.71, "missing": None},
        step=10,
    )
    logger.close()

    assert log_dir.is_dir()
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    assert event_files, f"no tfevent file written under {log_dir}"

    scalars = _read_event_scalars(log_dir)

    def _only_value(tag: str, expected_step: int) -> float:
        events = scalars[tag]
        assert len(events) == 1
        step, value = events[0]
        assert step == expected_step
        return value

    # tfevents store float32, so compare with tolerance.
    assert _only_value("train/L_gen", 1) == pytest.approx(0.42, rel=1e-5)
    assert _only_value("validation/auprc", 10) == pytest.approx(0.13, rel=1e-5)
    assert _only_value("validation/auroc", 10) == pytest.approx(0.71, rel=1e-5)
    # ``None`` values must not produce a scalar event.
    assert "validation/missing" not in scalars


def test_tb_logger_disabled_is_noop(tmp_path: Path) -> None:
    log_dir = tmp_path / "tb_disabled"
    logger = TensorBoardLogger(log_dir, enabled=False)

    assert logger.enabled is False
    logger.log_scalar("train/L_gen", 0.5, 1)
    logger.log_scalars("validation", {"auprc": 0.3}, step=5)
    logger.flush()
    logger.close()

    # Disabled logger must never touch the filesystem.
    assert not log_dir.exists()


def test_tb_logger_handles_none_log_dir() -> None:
    logger = TensorBoardLogger(None, enabled=True)
    assert logger.enabled is False
    # Calling logging methods is still safe.
    logger.log_scalar("train/x", 1.0, 1)
    logger.close()


def test_tb_logger_skips_non_numeric_values(tmp_path: Path) -> None:
    log_dir = tmp_path / "tb_nan"
    logger = TensorBoardLogger(log_dir, enabled=True)
    logger.log_scalar("train/x", "not-a-number", 1)  # type: ignore[arg-type]
    logger.log_scalar("train/y", 2.5, 2)
    logger.close()

    scalars = _read_event_scalars(log_dir)
    assert "train/x" not in scalars
    assert scalars["train/y"] == [(2, pytest.approx(2.5, rel=1e-5))]
