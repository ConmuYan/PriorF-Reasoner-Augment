"""Train LG-HGCL using a YAML configuration file."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

import torch.distributed as dist

from lghgcl.config import load_config, override_config
from lghgcl.exceptions import LGHGCLError
from lghgcl.logging_utils import configure_logging, get_logger
from lghgcl.pipeline import run_pipeline

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument(
        "--override", nargs="*", default=[],
        help="Override config values, e.g. train.use_ddp=true model.gnn_hidden=128",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI main entry."""

    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.override:
        cfg = override_config(cfg, args.override)

    # DDP: init process group & set local_rank from environment
    if cfg.train.use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # Patch local_rank into config (frozen dataclass → reconstruct)
        from dataclasses import replace
        new_train = replace(cfg.train, local_rank=local_rank)
        cfg = replace(cfg, train=new_train)

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        import torch
        torch.cuda.set_device(local_rank)

    rank = int(os.environ.get("RANK", 0))
    configure_logging(cfg.logging, cfg.project.output_dir)

    if rank == 0:
        logger.info("Loaded config from %s", Path(args.config).resolve())
        if args.override:
            logger.info("CLI overrides: %s", args.override)

    try:
        out_dir = run_pipeline(cfg)
        if rank == 0:
            logger.info("Pipeline finished. Output: %s", out_dir)
        return 0
    except LGHGCLError:
        logger.error("Pipeline failed:\n%s", traceback.format_exc())
        return 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
