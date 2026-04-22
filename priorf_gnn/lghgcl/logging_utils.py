"""Logging utilities for LG-HGCL."""

from __future__ import annotations

import logging
from logging import Logger
from logging.config import dictConfig
from pathlib import Path

from lghgcl.config import LoggingConfig


def configure_logging(cfg: LoggingConfig, output_dir: str | Path) -> None:
    """Configure Python logging using a single project-wide schema."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    handlers: dict[str, dict] = {}
    root_handlers: list[str] = []

    if cfg.console:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "level": cfg.level,
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        }
        root_handlers.append("console")

    if cfg.file:
        log_path = out_dir / cfg.file
        handlers["file"] = {
            "class": "logging.FileHandler",
            "level": cfg.level,
            "formatter": "standard",
            "filename": str(log_path),
            "encoding": "utf-8",
        }
        root_handlers.append("file")

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                }
            },
            "handlers": handlers,
            "root": {"level": cfg.level, "handlers": root_handlers},
        }
    )


def get_logger(name: str) -> Logger:
    """Return a module-level logger."""

    return logging.getLogger(name)
