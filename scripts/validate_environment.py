#!/usr/bin/env python3
"""Validate that the target conda environment is functional for PriorF-Reasoner.

Usage:
    # Basic check (auto-detects packages)
    conda run -p /data1/mq/conda_envs/priorfgnn python scripts/validate_environment.py

    # Against a baseline JSON exported from source server
    python scripts/validate_environment.py --baseline source_env.json
"""

import argparse
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EnvSnapshot:
    python_version: str
    packages: Dict[str, str]
    cuda_available: bool
    cuda_version: Optional[str]
    device_count: int
    torch_version: Optional[str]


def get_package_version(name: str) -> Optional[str]:
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def capture_snapshot() -> EnvSnapshot:
    packages = {}
    for pkg in [
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "datasets",
        "numpy",
        "scipy",
        "scikit_learn",
        "pandas",
        "matplotlib",
        "tqdm",
        "tensorboard",
    ]:
        v = get_package_version(pkg)
        if v:
            packages[pkg] = v

    cuda_available = False
    cuda_version = None
    device_count = 0
    torch_version = packages.get("torch")

    if "torch" in packages:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_version = str(torch.version.cuda) if cuda_available else None
        device_count = torch.cuda.device_count() if cuda_available else 0

    return EnvSnapshot(
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        packages=packages,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        device_count=device_count,
        torch_version=torch_version,
    )


def export_baseline(output_path: Path) -> None:
    snap = capture_snapshot()
    data = asdict(snap)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Baseline exported to {output_path}")


def validate_against_baseline(snap: EnvSnapshot, baseline_path: Path) -> bool:
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline: Dict[str, Any] = json.load(f)

    errors: List[str] = []
    warnings: List[str] = []

    # Python version (minor must match, patch can differ)
    src_py = baseline.get("python_version", "")
    tgt_py = snap.python_version
    if src_py and tgt_py:
        src_parts = src_py.split(".")
        tgt_parts = tgt_py.split(".")
        if src_parts[:2] != tgt_parts[:2]:
            errors.append(
                f"Python minor version mismatch: source={src_py} target={tgt_py}"
            )

    # Core packages exact version check
    core = ["torch", "transformers", "accelerate", "peft"]
    for pkg in core:
        src_v = baseline.get("packages", {}).get(pkg)
        tgt_v = snap.packages.get(pkg)
        if src_v and tgt_v and src_v != tgt_v:
            warnings.append(
                f"Version drift {pkg}: source={src_v} target={tgt_v}"
            )
        if src_v and not tgt_v:
            errors.append(f"Missing package: {pkg} (source had {src_v})")

    # CUDA availability
    if baseline.get("cuda_available") and not snap.cuda_available:
        errors.append("CUDA was available on source but NOT on target")

    # Device count (target can have fewer, warn if zero)
    src_dev = baseline.get("device_count", 0)
    tgt_dev = snap.device_count
    if src_dev > 0 and tgt_dev == 0:
        warnings.append(f"Source had {src_dev} GPU(s); target has 0")

    # Print results
    print(f"{'='*50}")
    print("Environment Validation Report")
    print(f"{'='*50}")
    print(f"Python       : {snap.python_version}")
    print(f"CUDA avail   : {snap.cuda_available}")
    print(f"CUDA version : {snap.cuda_version}")
    print(f"GPU count    : {snap.device_count}")
    for pkg, v in sorted(snap.packages.items()):
        print(f"  {pkg:<20} {v}")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  [WARN] {w}")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  [FAIL] {e}")

    print(f"\n{'='*50}")
    if not errors and not warnings:
        print("Validation PASSED.")
        return True
    elif not errors:
        print("Validation PASSED with warnings.")
        return True
    else:
        print("Validation FAILED.")
        return False


def standalone_check() -> bool:
    snap = capture_snapshot()
    print(f"{'='*50}")
    print("Environment Standalone Check")
    print(f"{'='*50}")
    print(f"Python       : {snap.python_version}")
    print(f"CUDA avail   : {snap.cuda_available}")
    print(f"CUDA version : {snap.cuda_version}")
    print(f"GPU count    : {snap.device_count}")
    for pkg, v in sorted(snap.packages.items()):
        print(f"  {pkg:<20} {v}")

    critical = ["torch", "transformers", "accelerate", "peft"]
    missing = [p for p in critical if p not in snap.packages]
    if missing:
        print(f"\n[FAIL] Missing critical packages: {missing}")
        return False
    if not snap.cuda_available:
        print("\n[WARN] CUDA not available; GPU workloads will fail")
    print("\nStandalone check PASSED.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate target environment")
    parser.add_argument(
        "--export-baseline",
        metavar="PATH",
        help="Export current environment as baseline JSON",
    )
    parser.add_argument(
        "--baseline",
        metavar="PATH",
        help="Validate against a previously exported baseline JSON",
    )
    args = parser.parse_args()

    if args.export_baseline:
        export_baseline(Path(args.export_baseline))
        return 0

    if args.baseline:
        snap = capture_snapshot()
        ok = validate_against_baseline(snap, Path(args.baseline))
        return 0 if ok else 1

    ok = standalone_check()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
