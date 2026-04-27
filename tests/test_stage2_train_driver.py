from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
import sklearn.metrics

from eval.head_scoring import CheckpointProvenance, ScorerReport
from evidence.prompt_builder import ThinkingMode
from evidence.leakage_policy import formal_leakage_provenance_fields
from graph_data.manifests import DataArtifact, DataManifest, PopulationMetadata
from priorf_teacher.schema import DatasetName, GraphRegime, NeighborSummary, PopulationName, RelationProfile, TeacherExportRecord
from scripts import run_formal_head_only_eval as formal_head_only_driver
from scripts import run_stage2_train as stage2_train_driver




def _score_head_audit_kwargs() -> dict[str, str]:
    return {
        "prompt_audit_path": "outputs/tests/prompt_audit.json",
        "prompt_audit_hash": "a" * 64,
    }


def _formal_report_provenance_kwargs() -> dict:
    return formal_leakage_provenance_fields(**_score_head_audit_kwargs())

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
    regime: GraphRegime = GraphRegime.TRANSDUCTIVE_STANDARD,
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
        graph_regime=regime,
    )


class _FakeModel:
    def save_pretrained(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (target / "adapter_model.safetensors").write_bytes(b"fake-adapter-weights")


def _scorer_report(
    *,
    population_name: PopulationName,
    probs: tuple[float, ...],
    labels: tuple[Literal[0, 1], ...],
) -> ScorerReport:
    probs_np = np.asarray(probs, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int64)
    is_single_class = bool(np.all(labels_np == 0) or np.all(labels_np == 1))
    return ScorerReport(
        dataset_name=DatasetName.AMAZON,
        population_name=population_name,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        run_id="run-123",
        report_split=population_name,
        eval_type="head_scoring",
        checkpoint_provenance=CheckpointProvenance(
            path="/tmp/original/cls_head.pt",
            step=5,
            content_hash="a" * 64,
        ),
        scorer_schema_version="head_scorer/v1",
        n_total=len(probs),
        n_positive=int((labels_np == 1).sum()),
        n_negative=int((labels_np == 0).sum()),
        is_single_class_population=is_single_class,
        auroc=None if is_single_class else float(sklearn.metrics.roc_auc_score(labels_np, probs_np)),
        auprc=None if is_single_class else float(sklearn.metrics.average_precision_score(labels_np, probs_np)),
        brier_score=float(np.mean((probs_np - labels_np.astype(np.float64)) ** 2)),
        prob_mean=float(np.mean(probs_np)),
        prob_std=float(np.std(probs_np)),
        prob_min=float(np.min(probs_np)),
        prob_max=float(np.max(probs_np)),
        prob_q25=float(np.quantile(probs_np, 0.25, method="linear")),
        prob_q50=float(np.quantile(probs_np, 0.50, method="linear")),
        prob_q75=float(np.quantile(probs_np, 0.75, method="linear")),
        probs=probs,
        labels=labels,
        node_ids=tuple(range(100, 100 + len(probs))),
        prompt_mode="eval_head",
        thinking_mode=ThinkingMode.NON_THINKING,
        pooling_path="pool_last_valid_token",
        uses_inference_mode=True,
        distributed_gather="none",
        **_formal_report_provenance_kwargs(),
    )


def test_checkpoint_artifact_helpers_roundtrip(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "best_checkpoint"
    alias_dir = tmp_path / "legacy_alias"
    model = _FakeModel()
    cls_head = stage2_train_driver._LinearClsHead(hidden_size=4, seed=7)

    provenance = stage2_train_driver._persist_checkpoint_artifacts(
        checkpoint_dir=checkpoint_dir,
        model=model,
        cls_head=cls_head,
        checkpoint_step=17,
    )

    assert provenance.step == 17
    assert provenance.path.endswith("cls_head.pt")
    assert provenance.content_hash == stage2_train_driver._file_sha256(checkpoint_dir / "cls_head.pt")
    assert (checkpoint_dir / "peft_adapter" / "adapter_config.json").exists()
    adapter_hash = stage2_train_driver._directory_sha256(checkpoint_dir / "peft_adapter")
    assert len(adapter_hash) == 64
    assert all(char in "0123456789abcdef" for char in adapter_hash)

    stage2_train_driver._copy_checkpoint_artifacts(
        source_checkpoint_dir=checkpoint_dir,
        destination_dir=alias_dir,
    )

    assert (alias_dir / "peft_adapter" / "adapter_model.safetensors").read_bytes() == (
        checkpoint_dir / "peft_adapter" / "adapter_model.safetensors"
    ).read_bytes()
    assert (alias_dir / "cls_head.pt").read_bytes() == (checkpoint_dir / "cls_head.pt").read_bytes()


def test_prompt_audit_artifact_hash_is_recorded(tmp_path: Path) -> None:
    sample = stage2_train_driver._build_training_sample(
        _teacher_record(label=1, teacher_prob=0.37),
        _manifest(),
    )

    audit_path, audit_hash, payload = stage2_train_driver._write_prompt_audit_artifact(
        output_dir=tmp_path,
        dataset=DatasetName.AMAZON,
        train_samples=(sample,),
        validation_records=(),
        data_manifest=_manifest(),
        git_commit="1" * 40,
    )

    assert audit_path == tmp_path / "prompt_audit.json"
    assert audit_hash == stage2_train_driver._file_sha256(audit_path)
    assert payload["leakage_audit_pass"] is True
    assert payload["violation_count"] == 0
    assert payload["sample_counts"]["train"]["samples"] == 1
    assert "labeled_neighbors" in payload["forbidden_fields"]


def test_training_sample_sft_score_does_not_copy_teacher_probability() -> None:
    fraud_sample = stage2_train_driver._build_training_sample(
        _teacher_record(label=1, teacher_prob=0.37),
        _manifest(),
    )
    benign_sample = stage2_train_driver._build_training_sample(
        _teacher_record(label=0, teacher_prob=0.63),
        _manifest(),
    )

    assert fraud_sample.sft_target_score == pytest.approx(0.95)
    assert benign_sample.sft_target_score == pytest.approx(0.05)
    assert fraud_sample.teacher_prob == pytest.approx(0.37)
    assert benign_sample.teacher_prob == pytest.approx(0.63)
    assert fraud_sample.evidence_card.teacher_summary.teacher_prob is None
    assert fraud_sample.evidence_card.teacher_summary.teacher_logit is None


def test_replace_checkpoint_provenance_preserves_report_fields() -> None:
    report = _scorer_report(
        population_name=PopulationName.VALIDATION,
        probs=(0.1, 0.4, 0.7, 0.9),
        labels=(0, 0, 1, 1),
    )
    new_provenance = CheckpointProvenance(
        path="/tmp/best_checkpoint/cls_head.pt",
        step=50,
        content_hash="b" * 64,
    )

    replaced = stage2_train_driver._replace_checkpoint_provenance(report, new_provenance)

    assert replaced.checkpoint_provenance == new_provenance
    assert replaced.probs == report.probs
    assert replaced.labels == report.labels
    assert replaced.auroc == pytest.approx(report.auroc)
    assert replaced.auprc == pytest.approx(report.auprc)


def test_best_checkpoint_metric_value_uses_requested_metric() -> None:
    report = _scorer_report(
        population_name=PopulationName.VALIDATION,
        probs=(0.1, 0.3, 0.8, 0.9),
        labels=(0, 0, 1, 1),
    )

    assert stage2_train_driver._best_checkpoint_metric_value(report, "validation_auroc") == pytest.approx(report.auroc)
    assert stage2_train_driver._best_checkpoint_metric_value(report, "validation_auprc") == pytest.approx(report.auprc)


def test_stage2_driver_default_best_checkpoint_metric_is_auprc() -> None:
    args = stage2_train_driver._parse_args([
        "--dataset", "amazon",
        "--qwen-path", "/tmp/qwen",
        "--teacher-export-train", "/tmp/train.parquet",
        "--data-manifest", "/tmp/manifest.json",
        "--output-dir", "/tmp/out",
        "--max-steps", "1",
    ])

    assert args.best_checkpoint_metric == "validation_auprc"


def test_stratified_record_subset_keeps_class_balance_proportional() -> None:
    records = (
        *(_teacher_record(label=1, teacher_prob=0.9) for _ in range(10)),
        *(_teacher_record(label=0, teacher_prob=0.1) for _ in range(90)),
    )

    subset = stage2_train_driver._stratified_record_subset(
        records,
        subset_size=20,
        rng=random.Random(0),
    )

    assert len(subset) == 20
    pos = sum(1 for r in subset if int(r.ground_truth_label) == 1)
    neg = sum(1 for r in subset if int(r.ground_truth_label) == 0)
    # 10/100 positives at subset_size=20 -> stratified rounding gives 2 positives.
    assert pos == 2
    assert neg == 18


def test_next_stratified_batch_indices_always_includes_both_classes() -> None:
    rng = random.Random(0)
    positive_indices = [0, 1, 2]
    negative_indices = [10, 11, 12, 13, 14]
    positive_pool: list[int] = []
    negative_pool: list[int] = []

    for _ in range(8):
        batch = stage2_train_driver._next_stratified_batch_indices(
            positive_indices=positive_indices,
            negative_indices=negative_indices,
            positive_pool=positive_pool,
            negative_pool=negative_pool,
            batch_size=2,
            rng=rng,
        )
        assert len(batch) == 2
        assert any(idx in positive_indices for idx in batch)
        assert any(idx in negative_indices for idx in batch)


def test_next_stratified_batch_indices_rejects_single_class_pool() -> None:
    rng = random.Random(0)
    with pytest.raises(ValueError, match="positive and negative"):
        stage2_train_driver._next_stratified_batch_indices(
            positive_indices=[],
            negative_indices=[0, 1],
            positive_pool=[],
            negative_pool=[],
            batch_size=2,
            rng=rng,
        )


def test_next_stratified_batch_indices_rejects_batch_size_one() -> None:
    rng = random.Random(0)
    with pytest.raises(ValueError, match="--batch-size >= 2"):
        stage2_train_driver._next_stratified_batch_indices(
            positive_indices=[0],
            negative_indices=[1],
            positive_pool=[],
            negative_pool=[],
            batch_size=1,
            rng=rng,
        )


def test_rolling_mean_returns_simple_average_until_window_fills() -> None:
    rm = stage2_train_driver._RollingMean(window=3)

    import math
    assert math.isnan(rm.value)
    assert rm.update(2.0) == pytest.approx(2.0)
    assert rm.update(4.0) == pytest.approx(3.0)  # (2 + 4) / 2
    assert rm.update(6.0) == pytest.approx(4.0)  # (2 + 4 + 6) / 3


def test_rolling_mean_drops_oldest_when_window_full() -> None:
    rm = stage2_train_driver._RollingMean(window=3)
    rm.update(1.0)
    rm.update(2.0)
    rm.update(3.0)

    assert rm.update(4.0) == pytest.approx((2.0 + 3.0 + 4.0) / 3.0)
    assert rm.update(5.0) == pytest.approx((3.0 + 4.0 + 5.0) / 3.0)
    assert len(rm) == rm.window == 3


def test_rolling_mean_skips_non_numeric_input() -> None:
    rm = stage2_train_driver._RollingMean(window=2)
    rm.update(1.0)
    rm.update("not-a-number")  # type: ignore[arg-type]
    assert rm.value == pytest.approx(1.0)
    assert len(rm) == 1


def test_rolling_mean_window_must_be_positive() -> None:
    with pytest.raises(ValueError, match="window must be >= 1"):
        stage2_train_driver._RollingMean(window=0)


def test_stage2_driver_rejects_unsupported_gradient_accumulation() -> None:
    args = SimpleNamespace(gradient_accumulation_steps=2)

    with pytest.raises(ValueError, match="gradient-accumulation-steps 1"):
        stage2_train_driver._validate_unsupported_driver_args(args)


def test_build_trainable_peft_model_uses_requested_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, object, object]] = []

    def fake_get_peft_model(base_model, lora_cfg):
        calls.append(("new", base_model, lora_cfg))
        return "fresh-model"

    class _FakePeftModel:
        @staticmethod
        def from_pretrained(base_model, model_id, is_trainable=False):
            calls.append(("warm", base_model, (model_id, is_trainable)))
            return "warm-model"

    import types
    import sys

    monkeypatch.setitem(
        sys.modules,
        "peft",
        types.SimpleNamespace(get_peft_model=fake_get_peft_model, PeftModel=_FakePeftModel),
    )

    base_model = object()
    lora_cfg = object()
    assert stage2_train_driver._build_trainable_peft_model(
        base_model=base_model,
        lora_cfg=lora_cfg,
        warm_start_peft_adapter=None,
    ) == "fresh-model"
    warm_dir = tmp_path / "stage1_adapter"
    assert stage2_train_driver._build_trainable_peft_model(
        base_model=base_model,
        lora_cfg=lora_cfg,
        warm_start_peft_adapter=warm_dir,
    ) == "warm-model"

    assert calls == [
        ("new", base_model, lora_cfg),
        ("warm", base_model, (str(warm_dir), True)),
    ]


def test_stage2_driver_resolves_shared_graph_regime_and_rejects_mismatch() -> None:
    inductive_manifest = _manifest(regime=GraphRegime.INDUCTIVE_MASKED)
    inductive_records = (
        _teacher_record(label=1, teacher_prob=0.8, regime=GraphRegime.INDUCTIVE_MASKED),
    )

    assert stage2_train_driver._resolve_shared_graph_regime(inductive_manifest, inductive_records) == GraphRegime.INDUCTIVE_MASKED

    with pytest.raises(ValueError, match="graph_regime mismatch"):
        stage2_train_driver._resolve_shared_graph_regime(
            _manifest(regime=GraphRegime.TRANSDUCTIVE_STANDARD),
            inductive_records,
        )


def test_formal_head_only_driver_resolves_shared_graph_regime() -> None:
    regime = GraphRegime.INDUCTIVE_MASKED

    resolved = formal_head_only_driver._resolve_shared_graph_regime(
        _manifest(regime=regime),
        (_teacher_record(label=1, teacher_prob=0.8, population=PopulationName.VALIDATION, regime=regime),),
        (_teacher_record(label=0, teacher_prob=0.2, population=PopulationName.FINAL_TEST, regime=regime),),
    )

    assert resolved == regime


def test_stage2_driver_help_mentions_warm_start_adapter() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_stage2_train.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        env=_SUBPROCESS_ENV,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--warm-start-peft-adapter" in completed.stdout


def test_run_formal_head_only_eval_help_mentions_checkpoint_source() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_formal_head_only_eval.py", "--help"],
        check=False,
        capture_output=True,
        env=_SUBPROCESS_ENV,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--checkpoint-source" in completed.stdout


def test_run_full_stage2_pipeline_help_mentions_best_checkpoint_controls() -> None:
    completed = subprocess.run(
        ["bash", "scripts/run_full_stage2_pipeline.sh", "--help"],
        check=False,
        capture_output=True,
        env=_SUBPROCESS_ENV,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--checkpoint-source" in completed.stdout
    assert "--validation-every-n-steps" in completed.stdout
