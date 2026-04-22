from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _write_manifest(tmp_path: Path, *, passed: bool) -> Path:
    path = tmp_path / "gate_manifest.json"
    payload = {
        "schema_version": "gate_manifest/v1",
        "graph_regime": "transductive_standard",
        "commit": "a" * 40,
        "generated_at": datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc).isoformat(),
        "config_fingerprint": "cfg-123",
        "data_manifest_hash": "b" * 64,
        "data_validation_pass": passed,
        "teacher_baseline_pass": passed,
        "subset_head_gate_pass": passed,
        "validation_eval_parity_pass": passed,
        "student_contribution_pass": passed,
        "strict_schema_parse_pass": passed,
        "smoke_pipeline_pass": passed,
        "teacher_prob_ablation_pass": passed,
        "population_contract_pass": passed,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_formal_launcher_requires_manifest(tmp_path: Path):
    completed = subprocess.run(
        ["bash", "scripts/run_full_pipeline.sh", "--mode", "formal", "--output-root", str(tmp_path / "outputs")],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "requires --gate-manifest" in completed.stderr


def test_formal_launcher_blocks_failed_gate(tmp_path: Path):
    manifest = _write_manifest(tmp_path, passed=False)
    marker = tmp_path / "should_not_exist.txt"
    completed = subprocess.run(
        [
            "bash",
            "scripts/run_full_pipeline.sh",
            "--mode",
            "formal",
            "--gate-manifest",
            str(manifest),
            "--output-root",
            str(tmp_path / "outputs"),
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path(r'{marker}').write_text('ran', encoding='utf-8')",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert not marker.exists()
    assert not (tmp_path / "outputs" / "formal").exists()


def test_formal_launcher_runs_only_after_gate_pass_and_routes_to_formal_namespace(tmp_path: Path):
    manifest = _write_manifest(tmp_path, passed=True)
    completed = subprocess.run(
        [
            "bash",
            "scripts/run_full_pipeline.sh",
            "--mode",
            "formal",
            "--gate-manifest",
            str(manifest),
            "--output-root",
            str(tmp_path / "outputs"),
            "--",
            sys.executable,
            "-c",
            "import os; print(os.environ['PRIORF_OUTPUT_NAMESPACE']); print(os.environ['PRIORF_OUTPUT_DIR'])",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    stdout_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    assert "formal" in stdout_lines
    assert str(tmp_path / "outputs" / "formal") in stdout_lines
    assert (tmp_path / "outputs" / "formal").is_dir()


def test_non_formal_modes_do_not_enter_formal_namespace(tmp_path: Path):
    for mode in ("gated", "diagnostic"):
        completed = subprocess.run(
            ["bash", "scripts/run_full_pipeline.sh", "--mode", mode, "--output-root", str(tmp_path / "outputs")],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert str(tmp_path / "outputs" / mode) in completed.stdout

    assert (tmp_path / "outputs" / "gated").is_dir()
    assert (tmp_path / "outputs" / "diagnostic").is_dir()
    assert not (tmp_path / "outputs" / "formal").exists()
