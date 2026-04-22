"""Task 7 — canonical joint trainer fail-closed tests."""

from __future__ import annotations

import re
import types
from typing import Literal, cast

import pytest
import torch
from pydantic import ValidationError

import train.train_stage2_canonical as canonical
from eval.head_scoring import CheckpointProvenance, HeadScoringInputs, HeadScoringSample, ScorerReport
from evidence.evidence_schema import DataManifestLike, build_evidence_card
from evidence.output_schema import PredLabel
from evidence.prompt_builder import ThinkingMode
from graph_data.manifests import DataArtifact, DataManifest, PopulationMetadata
from priorf_teacher.schema import DatasetName, GraphRegime, NeighborSummary, PopulationName, RelationProfile, TeacherExportRecord
from train.train_stage2_canonical import (
    CanonicalTrainerConfig,
    CanonicalTrainerRunRecord,
    CanonicalTrainingBatch,
    CanonicalTrainingSample,
    run_canonical_train_step,
    run_validation_with_unified_scorer,
)


def _config(**overrides) -> CanonicalTrainerConfig:
    payload = {
        "dataset_name": DatasetName.AMAZON,
        "graph_regime": GraphRegime.TRANSDUCTIVE_STANDARD,
        "model_name_or_path": "/data1/mq/models/Qwen3-4B-Instruct-2507",
        "output_dir": "outputs/gated/canonical_train/amazon",
        "learning_rate": 1e-3,
        "train_batch_size": 2,
        "max_steps": 1,
        "gradient_accumulation_steps": 1,
        "lambda_cls": 2.0,
        "lambda_distill": 3.0,
        "teacher_prob_clip_min": 0.1,
        "teacher_prob_clip_max": 0.9,
        "thinking_mode": ThinkingMode.NON_THINKING,
    }
    payload.update(overrides)
    return CanonicalTrainerConfig.model_validate(payload)


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


def _manifest(population: PopulationName = PopulationName.TRAIN) -> DataManifest:
    return DataManifest(
        dataset_name=DatasetName.AMAZON.value,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD.value,
        feature_dim=25,
        relation_count=3,
        num_nodes=64,
        populations=(
            PopulationMetadata(
                population_name=population.value,
                split_values=(population.value,),
                node_ids_hash=("a" if population == PopulationName.TRAIN else "b") * 64,
                contains_tuning_rows=population in {PopulationName.TRAIN, PopulationName.VALIDATION},
                contains_final_test_rows=False,
            ),
        ),
        artifacts=(DataArtifact(kind="source_mat", path="assets/data/Amazon_canonical.mat", sha256="c" * 64),),
    )


def _teacher_record(
    *,
    node_id: int,
    ground_truth_label: Literal[0, 1],
    teacher_prob: float,
    population: PopulationName = PopulationName.TRAIN,
) -> TeacherExportRecord:
    return TeacherExportRecord(
        dataset_name=DatasetName.AMAZON,
        teacher_model_name="LGHGCLNetV2",
        teacher_checkpoint="outputs/gated/teacher/best_model.pt",
        population_name=population,
        node_id=node_id,
        ground_truth_label=ground_truth_label,
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


def _sample(node_id: int, label: Literal[0, 1], teacher_prob: float) -> CanonicalTrainingSample:
    card = build_evidence_card(
        teacher_record=_teacher_record(node_id=node_id, ground_truth_label=label, teacher_prob=teacher_prob),
        data_manifest=cast(DataManifestLike, _manifest(PopulationName.TRAIN)),
    )
    return CanonicalTrainingSample(
        evidence_card=card,
        ground_truth_label=label,
        sft_target_label=PredLabel.FRAUD if label else PredLabel.BENIGN,
        sft_target_score=0.95 if label else 0.05,
        teacher_prob=teacher_prob,
        node_id=node_id,
    )


class TraceTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": dict(kwargs)})
        text = "\n".join(message["content"] for message in messages)
        match = re.search(r"node_id:\s*(\d+)", text)
        node_id = int(match.group(1)) if match else 0
        is_train = bool(messages[-1]["content"])
        suffix = 700 if is_train else 1000
        ids = torch.tensor([[11, 22, 33, suffix + node_id]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class TinyJointModel(torch.nn.Module):
    def __init__(self, *, nonfinite_loss: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.nonfinite_loss = nonfinite_loss
        self.forward_calls: list[dict] = []

    def forward(self, *, input_ids, attention_mask, output_hidden_states, use_cache, labels=None):
        self.forward_calls.append(
            {
                "input_ids": input_ids.detach().clone(),
                "attention_mask": attention_mask.detach().clone(),
                "labels_present": labels is not None,
                "output_hidden_states": output_hidden_states,
                "use_cache": use_cache,
            }
        )
        hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4, dtype=torch.float32, device=input_ids.device)
        hidden[:, :, 0] = input_ids.to(torch.float32) * self.weight
        hidden[:, :, 1] = self.weight
        hidden[:, :, 2] = attention_mask.to(torch.float32)
        hidden[:, :, 3] = 1.0
        loss = None
        if labels is not None:
            if self.nonfinite_loss:
                loss = self.weight / torch.tensor(0.0, device=input_ids.device)
            else:
                loss = ((input_ids.to(torch.float32).mean() * 0.001) + self.weight).pow(2)
        return types.SimpleNamespace(loss=loss, hidden_states=(hidden,))


class TinyClsHead(torch.nn.Module):
    def __init__(self, *, nonfinite: bool = False) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(0.01)
        self.nonfinite = nonfinite
        self.inputs: list[torch.Tensor] = []

    def forward(self, hidden_prompt_only: torch.Tensor) -> torch.Tensor:
        self.inputs.append(hidden_prompt_only.detach().clone())
        out = self.linear(hidden_prompt_only).squeeze(-1)
        if self.nonfinite:
            return out / torch.tensor(0.0, device=out.device)
        return out


def test_config_rejects_any_objective_mutation_or_diagnostic_probe():
    bad_overrides = [
        {"require_generation_loss": False},
        {"require_classification_loss": False},
        {"require_distillation_loss": False},
        {"lambda_cls": 0.0},
        {"lambda_distill": 0.0},
        {"diagnostic_mode": True},
        {"frozen_backbone_probe": True},
        {"class_imbalance_recipe": True},
        {"teacher_prob_clip_min": 0.9, "teacher_prob_clip_max": 0.1},
    ]
    for override in bad_overrides:
        with pytest.raises(ValidationError):
            _config(**override)


def test_canonical_train_step_uses_three_losses_and_prompt_only_cls_path():
    cfg = _config()
    batch = CanonicalTrainingBatch(samples=(_sample(3, 1, 1.0), _sample(5, 0, 0.0)))
    tokenizer = TraceTokenizer()
    model = TinyJointModel()
    cls_head = TinyClsHead()
    optimizer = torch.optim.SGD(list(model.parameters()) + list(cls_head.parameters()), lr=0.01)

    before = model.weight.detach().clone()
    report = run_canonical_train_step(
        config=cfg,
        batch=batch,
        model=model,
        cls_head=cls_head,
        tokenizer=tokenizer,
        optimizer=optimizer,
    )

    assert report.n_samples == 2
    assert report.generation_loss > 0.0
    assert report.classification_loss > 0.0
    assert report.distillation_loss > 0.0
    assert report.distillation_target == "clipped_teacher_prob_bce"
    assert report.generation_prompt_mode == "train"
    assert report.classification_prompt_mode == "eval_head"
    assert report.used_accelerate_backward is True
    assert not torch.equal(before, model.weight.detach())

    # One TRAIN generation forward + one EVAL_HEAD classification forward per sample.
    assert [call["labels_present"] for call in model.forward_calls] == [True, False, True, False]
    assert all(call["output_hidden_states"] is True for call in model.forward_calls)
    assert all(call["use_cache"] is False for call in model.forward_calls)

    # TRAIN prompts carry assistant JSON; EVAL_HEAD prompts use the assistant role with empty content.
    assert [bool(call["messages"][-1]["content"]) for call in tokenizer.calls] == [True, False, True, False]
    assert all(
        call["kwargs"]
        == {"tokenize": True, "return_dict": True, "return_tensors": "pt", "add_generation_prompt": False}
        for call in tokenizer.calls
    )

    # Classification hidden states come from the prompt-only token stream (1000+node), not TRAIN JSON stream (700+node).
    cls_last_token_values = [float(tensor[0, 0].item() / before.item()) for tensor in cls_head.inputs]
    assert cls_last_token_values == pytest.approx([1003.0, 1005.0], abs=1e-4)


def test_train_step_fail_closed_on_training_nan_and_frozen_probe():
    cfg = _config()
    batch = CanonicalTrainingBatch(samples=(_sample(3, 1, 0.5),))
    tokenizer = TraceTokenizer()

    model = TinyJointModel(nonfinite_loss=True)
    cls_head = TinyClsHead()
    optimizer = torch.optim.SGD(list(model.parameters()) + list(cls_head.parameters()), lr=0.01)
    with pytest.raises(ValueError, match="generation_loss"):
        run_canonical_train_step(config=cfg, batch=batch, model=model, cls_head=cls_head, tokenizer=tokenizer, optimizer=optimizer)

    frozen_model = TinyJointModel()
    for param in frozen_model.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.SGD(cls_head.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="frozen model probe"):
        run_canonical_train_step(config=cfg, batch=batch, model=frozen_model, cls_head=cls_head, tokenizer=tokenizer, optimizer=optimizer)


def test_validation_checkpoint_path_must_call_unified_scorer(monkeypatch: pytest.MonkeyPatch):
    sample = _sample(7, 1, 0.8)
    val_sample = HeadScoringSample(evidence_card=sample.evidence_card.model_copy(update={"population_name": PopulationName.VALIDATION}), ground_truth_label=1, node_id=7)
    validation_inputs = HeadScoringInputs(
        samples=(val_sample,),
        dataset_name=DatasetName.AMAZON,
        population_name=PopulationName.VALIDATION,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        checkpoint_provenance=CheckpointProvenance(path="outputs/gated/ckpt.safetensors", step=3, content_hash="d" * 64),
    )
    expected = ScorerReport(
        dataset_name=DatasetName.AMAZON,
        population_name=PopulationName.VALIDATION,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        checkpoint_provenance=validation_inputs.checkpoint_provenance,
        scorer_schema_version="head_scorer/v1",
        n_total=1,
        n_positive=1,
        n_negative=0,
        is_single_class_population=True,
        auroc=None,
        auprc=None,
        brier_score=0.01,
        prob_mean=0.9,
        prob_std=0.0,
        prob_min=0.9,
        prob_max=0.9,
        prob_q25=0.9,
        prob_q50=0.9,
        prob_q75=0.9,
        probs=(0.9,),
        labels=(1,),
        node_ids=(7,),
        prompt_mode="eval_head",
        thinking_mode=ThinkingMode.NON_THINKING,
        pooling_path="pool_last_valid_token",
        uses_inference_mode=True,
        distributed_gather="none",
    )
    calls = []

    def score_head_spy(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(canonical, "score_head", score_head_spy)
    got = run_validation_with_unified_scorer(
        validation_inputs=validation_inputs,
        model=TinyJointModel(),
        cls_head=TinyClsHead(),
        tokenizer=TraceTokenizer(),
        thinking_mode=ThinkingMode.NON_THINKING,
    )

    assert got is expected
    assert len(calls) == 1
    assert calls[0]["inputs"] is validation_inputs
    assert calls[0]["thinking_mode"] is ThinkingMode.NON_THINKING


def test_run_record_requires_population_metadata_and_checkpoint_provenance():
    cfg = _config()
    report = canonical.CanonicalStepReport(
        n_samples=1,
        generation_loss=1.0,
        classification_loss=2.0,
        distillation_loss=3.0,
        total_loss=1.0 + cfg.lambda_cls * 2.0 + cfg.lambda_distill * 3.0,
        lambda_cls=cfg.lambda_cls,
        lambda_distill=cfg.lambda_distill,
        generation_prompt_mode="train",
        classification_prompt_mode="eval_head",
        distillation_target="clipped_teacher_prob_bce",
        thinking_mode=cfg.thinking_mode,
        graph_regime=cfg.graph_regime,
        used_accelerate_backward=True,
    )
    train_pop = _manifest(PopulationName.TRAIN).populations[0]
    val_pop = _manifest(PopulationName.VALIDATION).populations[0]
    record = CanonicalTrainerRunRecord(
        config=cfg,
        checkpoint_provenance=CheckpointProvenance(path="outputs/gated/ckpt.safetensors", step=1, content_hash="e" * 64),
        train_population=train_pop,
        validation_population=val_pop,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        last_step=report,
    )

    assert record.config == cfg
    assert record.checkpoint_provenance is not None
    assert record.train_population.population_name == "train"
    assert record.validation_population is not None
    assert record.validation_population.population_name == "validation"
    assert record.graph_regime == cfg.graph_regime
