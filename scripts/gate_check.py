"""Fail-closed gate-manifest validation for formal launcher entrypoints."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from schemas.gate_manifest import GateManifest, load_gate_manifest

_REQUIRED_TRUE_FIELDS = (
    "data_validation_pass",
    "teacher_baseline_pass",
    "subset_head_gate_pass",
    "validation_eval_parity_pass",
    "student_contribution_pass",
    "strict_schema_parse_pass",
    "smoke_pipeline_pass",
    "teacher_prob_ablation_pass",
    "population_contract_pass",
)


def gate_check(*, manifest_path: str | Path) -> GateManifest:
    manifest = load_gate_manifest(manifest_path)
    failed = [field_name for field_name in _REQUIRED_TRUE_FIELDS if getattr(manifest, field_name) is not True]
    if failed:
        raise ValueError("gate manifest failed required gates: " + ", ".join(failed))
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a formal gate manifest and fail closed on any gate failure.")
    parser.add_argument("--manifest-path", required=True, help="Path to gate_manifest.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        manifest = gate_check(manifest_path=args.manifest_path)
    except Exception as exc:  # fail closed for schema, IO, and gate violations
        print(f"gate_check: FAIL: {exc}", file=sys.stderr)
        return 1

    print(
        "gate_check: PASS: "
        f"schema_version={manifest.schema_version} graph_regime={manifest.graph_regime.value} commit={manifest.commit}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
