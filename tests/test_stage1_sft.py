from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest

from evidence.output_schema import PredLabel
from evidence.prompt_builder import PromptMode, ThinkingMode, build_prompt
from graph_data.manifests import DataArtifact, DataManifest, PopulationMetadata
from priorf_teacher.schema import DatasetName, GraphRegime, NeighborSummary, PopulationName, RelationProfile, TeacherExportRecord
from train import stage1_sft

REPO_ROOT = Path(__file__).resolve().parents[1]
_SUBPROCESS_ENV = {
    **os.environ,
    "MKL_SERVICE_FORCE_INTEL": "1",
}


def _relation_profile() -> RelationProfile:
    return RelationProfile(
        total_relations=3,
        active_relations=2,
        max_relation_neighbor_count=5,
        mean_relation_neighbor_count=1.5,
        max_relation_discrepancy=0.4,
        mean_relation_discrepancy=0.2,
    )


def _neighbor_summary() -> NeighborSummary:
    return NeighborSummary(
        total_neighbors=4,
        labeled_neighbors=3,
        positive_neighbors=1,
        negative_neighbors=2,
        unlabeled_neighbors=1,
    )


def _manifest(*, regime: GraphRegime = GraphRegime.TRANSDUCTIVE_STANDARD) -> DataManifest:
    return DataManifest(
        dataset_name=DatasetName.AMAZON.value,
        graph_regime=regime.value,
        feature_dim=25,
        relation_count=3,
        num_nodes=64,
        populations=(
            PopulationMetadata(
                population_name=PopulationName.TRAIN.value,
                split_values=(PopulationName.TRAIN.value,),
                node_ids_hash="a" * 64,
                contains_tuning_rows=True,
                contains_final_test_rows=False,
            ),
        ),
        artifacts=(DataArtifact(kind="source_mat", path="assets/data/Amazon_canonical.mat", sha256="c" * 64),),
    )


def _teacher_record(
    *,
    label: Literal[0, 1],
    teacher_prob: float,
    population: PopulationName = PopulationName.TRAIN,
) -> TeacherExportRecord:
    return TeacherExportRecord(
        dataset_name=DatasetName.AMAZON,
        teacher_model_name="LGHGCLNetV2",
        teacher_checkpoint="outputs/gated/teacher/best_model.pt",
        population_name=population,
        node_id=7,
        ground_truth_label=label,
        teacher_prob=teacher_prob,
        teacher_logit=1.0,
        hsd=0.2,
        hsd_quantile=0.6,
        asda_switch=True,
        mlp_logit=0.8,
        gnn_logit=1.2,
        branch_gap=-0.4,
        relation_profile=_relation_profile(),
        neighbor_summary=_neighbor_summary(),
        high_hsd_flag=False,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
    )


class _FakeAdapterModel:
    def save_pretrained(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (target / "adapter_model.safetensors").write_bytes(b"fake-stage1-adapter")


def test_build_stage1_training_sample_uses_student_visible_card_and_fixed_targets() -> None:
    fraud_sample = stage1_sft._build_stage1_training_sample(
        _teacher_record(label=1, teacher_prob=0.91),
        _manifest(),
    )
    benign_sample = stage1_sft._build_stage1_training_sample(
        _teacher_record(label=0, teacher_prob=0.12),
        _manifest(),
    )

    assert fraud_sample.sft_target_label == PredLabel.FRAUD
    assert fraud_sample.sft_target_score == pytest.approx(0.95)
    assert benign_sample.sft_target_label == PredLabel.BENIGN
    assert benign_sample.sft_target_score == pytest.approx(0.05)

    assert fraud_sample.evidence_card.teacher_summary.teacher_prob is None
    assert fraud_sample.evidence_card.teacher_summary.teacher_logit is None

    bundle = build_prompt(
        evidence_card=fraud_sample.evidence_card,
        mode=PromptMode.TRAIN,
        thinking_mode=ThinkingMode.NON_THINKING,
        ground_truth_label_for_sft=fraud_sample.sft_target_label,
        score_target_for_sft=fraud_sample.sft_target_score,
    )
    assert bundle.messages[-1].content.startswith("{")
    assert "teacher_prob" not in bundle.messages[-1].content
    assert "discrepancy_severity" in bundle.messages[-1].content


def test_stage1_adapter_artifact_helpers_roundtrip(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "final_checkpoint"
    alias_dir = tmp_path / "legacy_alias"

    artifacts = stage1_sft._persist_adapter_artifacts(
        checkpoint_dir=checkpoint_dir,
        model=_FakeAdapterModel(),
    )

    assert artifacts["adapter_dir"].endswith("peft_adapter")
    assert artifacts["adapter_weights_sha256"] == stage1_sft.file_sha256(
        checkpoint_dir / "peft_adapter" / "adapter_model.safetensors"
    )
    assert artifacts["adapter_config_sha256"] == stage1_sft.file_sha256(
        checkpoint_dir / "peft_adapter" / "adapter_config.json"
    )

    stage1_sft._copy_adapter_artifacts(
        source_checkpoint_dir=checkpoint_dir,
        destination_dir=alias_dir,
    )

    assert (alias_dir / "peft_adapter" / "adapter_model.safetensors").read_bytes() == (
        checkpoint_dir / "peft_adapter" / "adapter_model.safetensors"
    ).read_bytes()


def test_resolve_total_steps_uses_max_steps_or_epochs() -> None:
    assert stage1_sft._resolve_total_steps(
        SimpleNamespace(max_steps=11, num_epochs=None, batch_size=4),
        n_train=5,
    ) == 11
    assert stage1_sft._resolve_total_steps(
        SimpleNamespace(max_steps=None, num_epochs=2, batch_size=3),
        n_train=5,
    ) == 4
    with pytest.raises(ValueError, match="either --max-steps or --num-epochs"):
        stage1_sft._resolve_total_steps(
            SimpleNamespace(max_steps=None, num_epochs=None, batch_size=1),
            n_train=5,
        )


def test_stage1_sft_module_help_succeeds() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "train.stage1_sft", "--help"],
        cwd=REPO_ROOT,
        env=_SUBPROCESS_ENV,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--teacher-export-train" in completed.stdout
    assert "--output-dir" in completed.stdout
