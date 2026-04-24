"""Fail-closed smoke pipeline tests (Task 15).

These tests cover two concerns:

1. Canonical plumbing smoke sweep invoked by ``scripts/run_smoke.sh``.
   A tiny end-to-end build of Evidence Card + prompt must succeed; a
   corrupted teacher record must fail closed with a schema error, not
   silently collapse into a generic OK.

2. Stage launcher fail-closed semantics for
   ``scripts/run_smoke.sh`` / ``scripts/run_stage1.sh`` /
   ``scripts/run_stage2.sh`` / ``scripts/run_eval.sh``.  Formal mode
   requires a passing gate manifest; smoke never enters the formal
   namespace; eval without --gate-manifest exits non-zero.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from evidence.evidence_schema import build_evidence_card
from evidence.prompt_builder import PromptMode, ThinkingMode, build_prompt
from graph_data.manifests import DataArtifact, DataManifest, PopulationMetadata
from priorf_teacher.schema import (
    DatasetName,
    GraphRegime,
    NeighborSummary,
    PopulationName,
    RelationProfile,
    TeacherExportRecord,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _teacher_record(*, teacher_prob: float = 0.8) -> TeacherExportRecord:
    return TeacherExportRecord(
        dataset_name=DatasetName.AMAZON,
        teacher_model_name="LGHGCLNetV2",
        teacher_checkpoint="assets/teacher/amazon/best_model.pt",
        population_name=PopulationName.VALIDATION,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        node_id=7,
        ground_truth_label=1,
        teacher_prob=teacher_prob,
        teacher_logit=1.2,
        hsd=0.4,
        hsd_quantile=0.6,
        asda_switch=True,
        mlp_logit=0.9,
        gnn_logit=1.6,
        branch_gap=-0.7,
        high_hsd_flag=True,
        relation_profile=RelationProfile(
            total_relations=3,
            active_relations=2,
            max_relation_neighbor_count=12,
            mean_relation_neighbor_count=8.0,
            max_relation_discrepancy=0.35,
            mean_relation_discrepancy=0.17,
        ),
        neighbor_summary=NeighborSummary(
            total_neighbors=15,
            labeled_neighbors=9,
            positive_neighbors=2,
            negative_neighbors=7,
            unlabeled_neighbors=6,
        ),
    )


def _data_manifest() -> DataManifest:
    return DataManifest(
        dataset_name="amazon",
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        feature_dim=25,
        relation_count=3,
        num_nodes=32,
        populations=(
            PopulationMetadata(
                population_name="validation",
                split_values=("validation",),
                node_ids_hash="0" * 64,
                contains_tuning_rows=True,
                contains_final_test_rows=False,
            ),
        ),
        artifacts=(
            DataArtifact(kind="source_mat", path="/tmp/amazon.mat", sha256="a" * 64),
        ),
    )


# ---------------------------------------------------------------- canonical smoke


def test_smoke_evidence_card_and_prompt_roundtrip_on_minimal_subset() -> None:
    manifest = _data_manifest()
    card = build_evidence_card(teacher_record=_teacher_record(), data_manifest=manifest)

    assert card.dataset_name == DatasetName.AMAZON
    assert card.teacher_summary.teacher_prob == pytest.approx(0.8)

    bundle = build_prompt(
        evidence_card=card,
        mode=PromptMode.EVAL_HEAD,
        thinking_mode=ThinkingMode.NON_THINKING,
    )
    assert bundle.messages  # non-empty
    user_content = "\n".join(
        message.content for message in bundle.messages if message.role == "user"
    )
    assert "teacher_prob" in user_content


def test_smoke_fails_closed_on_invalid_teacher_record() -> None:
    with pytest.raises(ValidationError):
        TeacherExportRecord(
            dataset_name=DatasetName.AMAZON,
            teacher_model_name="",  # blank -> schema violation
            teacher_checkpoint="ckpt",
            population_name=PopulationName.VALIDATION,
            graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
            node_id=7,
            ground_truth_label=1,
            teacher_prob=0.8,
            teacher_logit=1.0,
            hsd=0.3,
            hsd_quantile=0.5,
            asda_switch=True,
            mlp_logit=1.0,
            gnn_logit=1.0,
            branch_gap=0.0,
            high_hsd_flag=False,
            relation_profile=RelationProfile(
                total_relations=1,
                active_relations=1,
                max_relation_neighbor_count=1,
                mean_relation_neighbor_count=1.0,
                max_relation_discrepancy=0.0,
                mean_relation_discrepancy=0.0,
            ),
            neighbor_summary=NeighborSummary(
                total_neighbors=1,
                labeled_neighbors=1,
                positive_neighbors=1,
                negative_neighbors=0,
                unlabeled_neighbors=0,
            ),
        )


# ---------------------------------------------------------------- launcher fail-closed


def _write_passing_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "gate_manifest.json"
    payload = {
        "schema_version": "gate_manifest/v1",
        "graph_regime": "transductive_standard",
        "commit": "a" * 40,
        "generated_at": datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc).isoformat(),
        "config_fingerprint": "cfg-smoke",
        "data_manifest_hash": "b" * 64,
        "data_validation_pass": True,
        "teacher_baseline_pass": True,
        "subset_head_gate_pass": True,
        "validation_eval_parity_pass": True,
        "student_contribution_pass": True,
        "strict_schema_parse_pass": True,
        "smoke_pipeline_pass": True,
        "teacher_prob_ablation_pass": True,
        "population_contract_pass": True,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_run_smoke_sh_help_succeeds() -> None:
    completed = subprocess.run(
        ["bash", "scripts/run_smoke.sh", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "diagnostic smoke" in completed.stdout


def test_run_eval_sh_requires_gate_manifest(tmp_path: Path) -> None:
    completed = subprocess.run(
        ["bash", "scripts/run_eval.sh", "--output-root", str(tmp_path / "outputs")],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "--gate-manifest" in completed.stderr
    assert not (tmp_path / "outputs" / "formal").exists()


def test_run_eval_sh_runs_command_under_formal_namespace_when_gate_passes(
    tmp_path: Path,
) -> None:
    manifest = _write_passing_manifest(tmp_path)
    completed = subprocess.run(
        [
            "bash",
            "scripts/run_eval.sh",
            "--gate-manifest",
            str(manifest),
            "--output-root",
            str(tmp_path / "outputs"),
            "--",
            sys.executable,
            "-c",
            "import os; print(os.environ['PRIORF_OUTPUT_NAMESPACE'])",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "formal" in completed.stdout
    assert (tmp_path / "outputs" / "formal").is_dir()


def test_run_stage1_sh_defaults_to_gated_namespace(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            "bash",
            "scripts/run_stage1.sh",
            "--output-root",
            str(tmp_path / "outputs"),
            "--",
            sys.executable,
            "-c",
            "import os; print(os.environ['PRIORF_OUTPUT_NAMESPACE'])",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "gated" in completed.stdout
    assert (tmp_path / "outputs" / "gated").is_dir()
    assert not (tmp_path / "outputs" / "formal").exists()


def test_run_stage2_sh_defaults_to_gated_namespace(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            "bash",
            "scripts/run_stage2.sh",
            "--output-root",
            str(tmp_path / "outputs"),
            "--",
            sys.executable,
            "-c",
            "import os; print(os.environ['PRIORF_OUTPUT_NAMESPACE'])",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "gated" in completed.stdout
    assert (tmp_path / "outputs" / "gated").is_dir()
    assert not (tmp_path / "outputs" / "formal").exists()


def test_run_stage2_sh_promotes_to_formal_when_gate_manifest_passes(
    tmp_path: Path,
) -> None:
    manifest = _write_passing_manifest(tmp_path)
    completed = subprocess.run(
        [
            "bash",
            "scripts/run_stage2.sh",
            "--output-root",
            str(tmp_path / "outputs"),
            "--gate-manifest",
            str(manifest),
            "--",
            sys.executable,
            "-c",
            "import os; print(os.environ['PRIORF_OUTPUT_NAMESPACE'])",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "formal" in completed.stdout
    assert (tmp_path / "outputs" / "formal").is_dir()


def test_run_stage1_sh_rejects_failing_gate_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "bad_manifest.json"
    payload = {
        "schema_version": "gate_manifest/v1",
        "graph_regime": "transductive_standard",
        "commit": "a" * 40,
        "generated_at": datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc).isoformat(),
        "config_fingerprint": "cfg",
        "data_manifest_hash": "b" * 64,
        "data_validation_pass": False,
        "teacher_baseline_pass": True,
        "subset_head_gate_pass": True,
        "validation_eval_parity_pass": True,
        "student_contribution_pass": True,
        "strict_schema_parse_pass": True,
        "smoke_pipeline_pass": True,
        "teacher_prob_ablation_pass": True,
        "population_contract_pass": True,
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    marker = tmp_path / "should_not_run.txt"
    completed = subprocess.run(
        [
            "bash",
            "scripts/run_stage1.sh",
            "--gate-manifest",
            str(manifest_path),
            "--output-root",
            str(tmp_path / "outputs"),
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path(r'{marker}').write_text('ran', encoding='utf-8')",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert not marker.exists()
    assert not (tmp_path / "outputs" / "formal").exists()
