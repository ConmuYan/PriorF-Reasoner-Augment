#!/usr/bin/env python3
"""Validate project files on the target server against a source manifest.

Checks:
  1. Every manifest file exists on target and SHA256 matches.
  2. (Optional) No extra unexpected files exist in monitored directories.

Usage:
    python scripts/validate_manifest.py migration_manifest.json --root .
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Set, Tuple


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_actual_files(root: Path, monitored_prefixes: List[str]) -> Set[str]:
    actual: Set[str] = set()
    for prefix in monitored_prefixes:
        base = root / prefix
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file():
                actual.add(p.relative_to(root).as_posix())
    return actual


def validate(manifest: dict, root: Path, strict: bool, monitored_prefixes: List[str]) -> bool:
    files = manifest.get("files", [])
    ok = 0
    missing = 0
    mismatch = 0
    unexpected: List[str] = []

    expected_paths = {entry["path"] for entry in files}

    if strict and monitored_prefixes:
        actual_paths = collect_actual_files(root, monitored_prefixes)
        unexpected = sorted(actual_paths - expected_paths)

    for entry in files:
        rel = entry["path"]
        target = root / rel
        if not target.exists():
            print(f"[MISS] {rel}")
            missing += 1
            continue
        actual_hash = sha256_file(target)
        expected_hash = entry["sha256"]
        if actual_hash != expected_hash:
            print(f"[HASH] {rel}  expected={expected_hash[:16]}... actual={actual_hash[:16]}...")
            mismatch += 1
            continue
        ok += 1

    print(f"\n{'='*50}")
    print(f"Manifest: {manifest.get('git_commit', 'N/A')}")
    print(f"Total expected : {len(files)}")
    print(f"OK             : {ok}")
    print(f"Missing        : {missing}")
    print(f"Hash mismatch  : {mismatch}")
    if strict:
        print(f"Unexpected     : {len(unexpected)}")
        for u in unexpected[:10]:
            print(f"  + {u}")
        if len(unexpected) > 10:
            print(f"  ... and {len(unexpected)-10} more")
    print(f"{'='*50}")

    return missing == 0 and mismatch == 0 and (not strict or len(unexpected) == 0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate migration manifest")
    parser.add_argument("manifest", help="Path to manifest JSON")
    parser.add_argument("--root", default=".", help="Project root on target machine")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if extra unexpected files exist under monitored prefixes",
    )
    parser.add_argument(
        "--monitored-prefixes",
        nargs="+",
        default=["assets", "outputs"],
        help="Directories to scan for unexpected files in strict mode",
    )
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    root = Path(args.root).resolve()

    passed = validate(manifest, root, args.strict, args.monitored_prefixes)
    if passed:
        print("\nValidation PASSED.")
        return 0
    else:
        print("\nValidation FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
