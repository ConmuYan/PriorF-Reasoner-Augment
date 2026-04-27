"""Task 14 — formal faithfulness evaluation contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

import eval.faithfulness as faithfulness
from eval.faithfulness import FaithfulnessInputs, FrozenDecisionPolicy, evaluate_faithfulness
from eval.head_scoring import CheckpointProvenance, HeadScoringInputs, HeadScoringSample
from evidence.evidence_schema import EvidenceAblationMask, build_evidence_card, build_student_evidence_card
from evidence.prompt_builder import ThinkingMode
from graph_data.manifests import DataArtifact, DataManifest, PopulationMetadata
from priorf_teacher.schema import (
    DatasetName,
    GraphRegime,
    NeighborSummary,
    PopulationName,
    RelationProfile,
    TeacherExportRecord,
)


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


def _manifest() -> DataManifest:
    return DataManifest(
        dataset_name=DatasetName.AMAZON.value,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD.value,
        feature_dim=25,
        relation_count=3,
        num_nodes=128,
        populations=(
            PopulationMetadata(
                population_name=PopulationName.VALIDATION.value,
                split_values=(PopulationName.VALIDATION.value,),
                node_ids_hash="a" * 64,
                contains_tuning_rows=True,
                contains_final_test_rows=False,
            ),
        ),
        artifacts=(DataArtifact(kind="source_mat", path="amazon.mat", sha256="b" * 64),),
    )


def _record(node_id: int, ground_truth_label: int) -> TeacherExportRecord:
    return TeacherExportRecord.model_validate(
        {
            "dataset_name": DatasetName.AMAZON,
            "teacher_model_name": "PriorF-GNN",
            "teacher_checkpoint": "outputs/gated/teacher/best.pt",
            "population_name": PopulationName.VALIDATION,
            "node_id": node_id,
            "ground_truth_label": ground_truth_label,
            "teacher_prob": 0.8,
            "teacher_logit": 1.4,
            "hsd": 0.3,
            "hsd_quantile": 0.7,
            "asda_switch": True,
            "mlp_logit": 1.1,
            "gnn_logit": 1.9,
            "branch_gap": 0.5,
            "relation_profile": _relation_profile(),
            "neighbor_summary": _neighbor_summary(),
            "high_hsd_flag": True,
            "graph_regime": GraphRegime.TRANSDUCTIVE_STANDARD,
        }
    )


def _sample(node_id: int, label: int) -> HeadScoringSample:
    return HeadScoringSample(
        evidence_card=build_student_evidence_card(
            teacher_record=_record(node_id=node_id, ground_truth_label=label),
            data_manifest=_manifest(),
        ),
        ground_truth_label=label,
        node_id=node_id,
    )


def _full_inputs(node_ids: tuple[int, ...] = (10, 11), labels: tuple[int, ...] = (1, 0)) -> HeadScoringInputs:
    samples = tuple(_sample(node_id=nid, label=label) for nid, label in zip(node_ids, labels, strict=True))
    return HeadScoringInputs(
        samples=samples,
        dataset_name=DatasetName.AMAZON,
        population_name=PopulationName.VALIDATION,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        checkpoint_provenance=CheckpointProvenance(
            path="outputs/gated/checkpoints/step_12.safetensors",
            step=12,
            content_hash="deadbeef" * 8,
        ),
    )


def _faithfulness_inputs(full_inputs: HeadScoringInputs | None = None, **overrides) -> FaithfulnessInputs:
    payload = {
        "full_inputs": full_inputs or _full_inputs(),
        "run_id": "run-123",
        "thinking_mode": ThinkingMode.NON_THINKING,
        "frozen_decision_policy": FrozenDecisionPolicy(
            alpha=0.35,
            threshold=0.5,
            alpha_source="artifacts/validation_alpha.json",
            threshold_source="artifacts/validation_threshold.json",
        ),
        "selected_evidence_fields": (
            EvidenceAblationMask.TEACHER_SUMMARY_TEACHER_PROB,
            EvidenceAblationMask.DISCREPANCY_SUMMARY_BRANCH_GAP_ABS,
        ),
        "teacher_prob_ablation_fields": (
            EvidenceAblationMask.TEACHER_SUMMARY_TEACHER_PROB,
        ),
        "minimum_formal_sample_size": 2,
        "prompt_audit_path": "outputs/tests/prompt_audit.json",
        "prompt_audit_hash": "a" * 64,
    }
    payload.update(overrides)
    return FaithfulnessInputs(**payload)


class ScoreAwareTokenizer:
    """Tokenizer double whose final token reflects schema-preserving ablations."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, **kwargs):
        text = "\n".join(message["content"] for message in messages)
        node_match = re.search(r"node_id:\s*(\d+)", text)
        node_id = int(node_match.group(1)) if node_match else 0
        bonus = 0
        if "teacher_prob: Masked / Not Available" in text:
            bonus += 40
        if "branch_gap_abs: Masked / Not Available" in text:
            bonus += 20
        if "hsd: Masked / Not Available" in text:
            bonus += 10
        token_value = 100 + node_id - bonus
        self.calls.append({"messages": messages, "kwargs": dict(kwargs), "text": text, "token_value": token_value})
        input_ids = torch.tensor([[11, 22, token_value]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TraceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, *, input_ids, attention_mask, output_hidden_states, use_cache):
        assert output_hidden_states is True
        assert use_cache is False
        hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4, dtype=torch.float32)
        hidden[:, -1, 0] = input_ids[:, -1].to(torch.float32) - 100.0
        return type("Outputs", (), {"hidden_states": (hidden,)})()


class TraceClsHead:
    def __init__(self) -> None:
        self.eval_count = 0

    def eval(self) -> "TraceClsHead":
        self.eval_count += 1
        return self

    def __call__(self, hidden_prompt_only: torch.Tensor) -> torch.Tensor:
        return (hidden_prompt_only[:, 0] / 20.0) - 0.2


def _sigmoid(value: float) -> float:
    return float(torch.sigmoid(torch.tensor(value, dtype=torch.float32)).item())


def test_evaluate_faithfulness_reuses_frozen_head_path_and_reports_expected_metrics():
    tokenizer = ScoreAwareTokenizer()
    model = TraceModel()
    cls_head = TraceClsHead()

    report = evaluate_faithfulness(
        inputs=_faithfulness_inputs(),
        model=model,
        cls_head=cls_head,
        tokenizer=tokenizer,
    )

    assert report.n_total == 2
    assert report.minimum_formal_sample_size == 2
    assert report.run_id == "run-123"
    assert report.report_split == PopulationName.VALIDATION
    assert report.eval_type == "faithfulness"
    assert report.prompt_mode == "eval_head"
    assert report.thinking_mode is ThinkingMode.NON_THINKING
    assert report.pooling_path == "pool_last_valid_token"
    assert report.uses_inference_mode is True
    assert report.distributed_gather == "none"
    assert report.frozen_decision_policy.alpha_source == "artifacts/validation_alpha.json"
    assert report.frozen_decision_policy.threshold_source == "artifacts/validation_threshold.json"
    missing_identity_payload = report.model_dump(mode="python")
    missing_identity_payload.pop("dataset_name")
    with pytest.raises(ValidationError):
        type(report).model_validate(missing_identity_payload)

    assert len(tokenizer.calls) == 8  # 4 faithfulness passes × 2 samples

    full_probs = (_sigmoid(((-30) / 20.0) - 0.2), _sigmoid(((-29) / 20.0) - 0.2))
    suff_probs = (_sigmoid(((-40) / 20.0) - 0.2), _sigmoid(((-39) / 20.0) - 0.2))
    comp_probs = (_sigmoid(((-50) / 20.0) - 0.2), _sigmoid(((-49) / 20.0) - 0.2))
    teacher_probs = full_probs

    assert report.full_report.probs == pytest.approx(full_probs)
    assert report.sufficiency_report.probs == pytest.approx(suff_probs)
    assert report.comprehensiveness_report.probs == pytest.approx(comp_probs)
    assert report.teacher_prob_ablation_report.probs == pytest.approx(teacher_probs)

    expected_sufficiency = tuple(suff - full for full, suff in zip(full_probs, suff_probs, strict=True))
    expected_comprehensiveness = tuple(comp - full for full, comp in zip(full_probs, comp_probs, strict=True))
    expected_impact = tuple(teacher - full for full, teacher in zip(full_probs, teacher_probs, strict=True))

    assert tuple(item.sufficiency for item in report.sample_results) == pytest.approx(expected_sufficiency)
    assert tuple(item.comprehensiveness for item in report.sample_results) == pytest.approx(expected_comprehensiveness)
    assert tuple(item.evidence_ablation_impact for item in report.sample_results) == pytest.approx(expected_impact)
    assert report.mean_sufficiency == pytest.approx(sum(expected_sufficiency) / 2.0)
    assert report.mean_comprehensiveness == pytest.approx(sum(expected_comprehensiveness) / 2.0)
    assert report.mean_evidence_ablation_impact == pytest.approx(sum(expected_impact) / 2.0)
    assert report.decision_flip_rate_sufficiency == pytest.approx(0.0)
    assert report.decision_flip_rate_comprehensiveness == pytest.approx(0.0)
    assert report.decision_flip_rate_teacher_prob_ablation == pytest.approx(0.0)

    old_payload = report.model_dump(mode="python")
    for field_name in (
        "leakage_policy_version",
        "neighbor_label_policy",
        "evidence_card_projection",
        "student_visible_forbidden_fields",
        "teacher_prob_masked",
        "teacher_logit_masked",
        "neighbor_label_counts_visible",
        "formal_safe_result",
        "prompt_audit_path",
        "prompt_audit_hash",
    ):
        old_payload.pop(field_name, None)
    with pytest.raises(ValidationError):
        type(report).model_validate(old_payload)


def test_schema_preserving_ablation_keeps_message_shape_but_masks_values_only():
    tokenizer = ScoreAwareTokenizer()
    report = evaluate_faithfulness(
        inputs=_faithfulness_inputs(full_inputs=_full_inputs(node_ids=(10,), labels=(1,)), minimum_formal_sample_size=1),
        model=TraceModel(),
        cls_head=TraceClsHead(),
        tokenizer=tokenizer,
    )

    assert report.n_total == 1
    full_text = tokenizer.calls[0]["text"]
    sufficiency_text = tokenizer.calls[1]["text"]
    comprehensiveness_text = tokenizer.calls[2]["text"]
    teacher_prob_text = tokenizer.calls[3]["text"]

    assert "teacher_prob: Masked / Not Available" in full_text
    assert "teacher_prob: Masked / Not Available" in sufficiency_text
    assert "branch_gap_abs: 0.5" in sufficiency_text
    assert "hsd: Masked / Not Available" in sufficiency_text

    assert "teacher_prob: Masked / Not Available" in comprehensiveness_text
    assert "branch_gap_abs: Masked / Not Available" in comprehensiveness_text
    assert "hsd: 0.3" in comprehensiveness_text
    assert "teacher_prob: Masked / Not Available" in teacher_prob_text
    assert "branch_gap_abs: 0.5" in teacher_prob_text

    for idx in range(4):
        messages = tokenizer.calls[idx]["messages"]
        assert [message["role"] for message in messages] == ["system", "user", "assistant"]


def test_formal_faithfulness_fails_closed_when_sample_count_is_too_small():
    with pytest.raises(ValueError, match="minimum_formal_sample_size"):
        _faithfulness_inputs(full_inputs=_full_inputs(node_ids=(10,), labels=(1,)), minimum_formal_sample_size=2)


def test_teacher_prob_ablation_contract_requires_teacher_prob_field():
    with pytest.raises(ValueError, match="teacher_summary.teacher_prob"):
        _faithfulness_inputs(
            teacher_prob_ablation_fields=(
                EvidenceAblationMask.TEACHER_SUMMARY_TEACHER_LOGIT,
            )
        )


def test_formal_full_inputs_must_start_unablated():
    full_inputs = _full_inputs(node_ids=(10,), labels=(1,))
    masked_card = full_inputs.samples[0].evidence_card.model_copy(
        update={"ablation_mask": frozenset({EvidenceAblationMask.TEACHER_SUMMARY_TEACHER_PROB})}
    )
    masked_sample = full_inputs.samples[0].model_copy(update={"evidence_card": masked_card})
    masked_inputs = full_inputs.model_copy(update={"samples": (masked_sample,)})

    with pytest.raises(ValueError, match="mask direct teacher score fields"):
        _faithfulness_inputs(full_inputs=masked_inputs, minimum_formal_sample_size=1)


def test_module_reuses_score_head_without_reimplementing_prompt_or_tokenizer_logic():
    source = Path(faithfulness.__file__).read_text(encoding="utf-8")
    assert "score_head(" in source
    assert "build_prompt(" not in source
    assert ".apply_chat_template(" not in source
    assert "pool_last_valid_token(" not in source


def test_faithfulness_rejects_internal_full_cards_by_default():
    sample = HeadScoringSample(
        evidence_card=build_evidence_card(teacher_record=_record(node_id=10, ground_truth_label=1), data_manifest=_manifest()),
        ground_truth_label=1,
        node_id=10,
    )
    inputs = HeadScoringInputs(
        samples=(sample,),
        dataset_name=DatasetName.AMAZON,
        population_name=PopulationName.VALIDATION,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        checkpoint_provenance=CheckpointProvenance(
            path="outputs/gated/checkpoints/step_12.safetensors",
            step=12,
            content_hash="deadbeef" * 8,
        ),
    )
    with pytest.raises(ValueError, match="student-safe"):
        _faithfulness_inputs(full_inputs=inputs, minimum_formal_sample_size=1)
