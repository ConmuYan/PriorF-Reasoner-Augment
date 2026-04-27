"""Canonical joint trainer contract for PriorF-Reasoner Task 7.

This module defines the narrow canonical training surface.  It intentionally does
not implement diagnostic probes, fusion, faithfulness, formal reports, or gate
manifests.  The only accepted objective is::

    L = L_gen + lambda_cls * L_cls + lambda_distill * L_distill

where ``L_distill`` is BCEWithLogitsLoss against clipped ``teacher_prob`` and
``L_cls`` is BCEWithLogitsLoss against ``ground_truth_label``.  The
classification forward path uses ``PromptMode.EVAL_HEAD`` so it consumes
prompt-only hidden states and remains independent from generation target order.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any, Literal, Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from eval.head_scoring import CheckpointProvenance, HeadScoringInputs, ScorerReport, score_head
from evidence.evidence_schema import EvidenceAblationMask, EvidenceCard
from evidence.leakage_policy import (
    EVIDENCE_CARD_PROJECTION,
    LEAKAGE_POLICY_VERSION,
    NEIGHBOR_LABEL_COUNTS_VISIBLE,
    NEIGHBOR_LABEL_POLICY,
    STUDENT_VISIBLE_FORBIDDEN_FIELD_PATHS,
    TEACHER_LOGIT_MASKED,
    TEACHER_PROB_MASKED,
)
from evidence.output_schema import PredLabel
from evidence.prompt_builder import PromptMode, ThinkingMode, build_prompt
from graph_data.manifests import PopulationMetadata
from llm.hidden_state_pooling import pool_last_valid_token
from priorf_teacher.schema import DatasetName, GraphRegime, PopulationName

__all__ = (
    "CanonicalTrainerConfig",
    "CanonicalTrainingSample",
    "CanonicalTrainingBatch",
    "CanonicalStepReport",
    "CanonicalTrainerRunRecord",
    "TrainableClsHead",
    "run_canonical_train_step",
    "run_validation_with_unified_scorer",
    "prepare_canonical_components",
)

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_GIT_HASH_PATTERN = r"^([0-9a-f]{40}|[0-9a-f]{64})$"
_HEX64_PATTERN = r"^[0-9a-f]{64}$"
@runtime_checkable
class TrainableClsHead(Protocol):
    """Training-time classification-head surface used by the canonical trainer."""

    def __call__(self, hidden_prompt_only: torch.Tensor) -> torch.Tensor: ...

    def train(self, mode: bool = True) -> "TrainableClsHead": ...

    def eval(self) -> "TrainableClsHead": ...


class CanonicalTrainerConfig(BaseModel):
    """Fail-closed canonical joint trainer configuration.

    The ``require_*`` fields are Literal[True] so any attempted config that
    disables one objective term fails during schema validation instead of
    silently changing the canonical objective into a probe/ablation.
    """

    model_config = _STRICT_MODEL_CONFIG

    schema_version: Literal["canonical_trainer/v1"] = "canonical_trainer/v1"
    dataset_name: DatasetName
    graph_regime: GraphRegime
    train_population_name: Literal[PopulationName.TRAIN] = PopulationName.TRAIN
    validation_population_name: Literal[PopulationName.VALIDATION] = PopulationName.VALIDATION
    model_name_or_path: StrictStr = Field(min_length=1)
    output_dir: StrictStr = Field(min_length=1)
    learning_rate: StrictFloat = Field(gt=0.0)
    train_batch_size: StrictInt = Field(ge=1)
    max_steps: StrictInt = Field(ge=1)
    gradient_accumulation_steps: StrictInt = Field(ge=1)
    lambda_cls: StrictFloat = Field(gt=0.0)
    lambda_distill: StrictFloat = Field(gt=0.0)
    teacher_prob_clip_min: StrictFloat = Field(default=1e-4, ge=0.0, le=1.0)
    teacher_prob_clip_max: StrictFloat = Field(default=1.0 - 1e-4, ge=0.0, le=1.0)
    max_grad_norm: StrictFloat | None = Field(default=None, gt=0.0)
    thinking_mode: ThinkingMode

    # Explicit blockers for silent objective mutation / diagnostic routing.
    require_generation_loss: Literal[True] = True
    require_classification_loss: Literal[True] = True
    require_distillation_loss: Literal[True] = True
    use_accelerate: Literal[True] = True
    diagnostic_mode: Literal[False] = False
    frozen_backbone_probe: Literal[False] = False
    class_imbalance_recipe: Literal[False] = False

    @model_validator(mode="after")
    def _clip_bounds_ordered(self) -> "CanonicalTrainerConfig":
        if self.teacher_prob_clip_min >= self.teacher_prob_clip_max:
            raise ValueError("teacher_prob_clip_min must be < teacher_prob_clip_max")
        return self


class CanonicalTrainingSample(BaseModel):
    """One canonical joint-training sample.

    ``teacher_prob`` is explicit so distillation cannot silently switch to
    teacher logits or another target.  The student-visible Evidence Card may
    mask direct teacher-score shortcuts; if it does not, the card value is
    cross-checked against this explicit distillation target.
    """

    model_config = _STRICT_MODEL_CONFIG

    evidence_card: EvidenceCard
    ground_truth_label: Literal[0, 1]
    sft_target_label: PredLabel
    sft_target_score: StrictFloat = Field(ge=0.0, le=1.0)
    teacher_prob: StrictFloat = Field(ge=0.0, le=1.0)
    node_id: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def _sample_matches_card(self) -> "CanonicalTrainingSample":
        if self.evidence_card.population_name != PopulationName.TRAIN:
            raise ValueError("canonical training samples must come from train population")
        if self.evidence_card.node_id != self.node_id:
            raise ValueError("sample.node_id must match evidence_card.node_id")
        card_teacher_prob = self.evidence_card.teacher_summary.teacher_prob
        if card_teacher_prob is None:
            if EvidenceAblationMask.TEACHER_SUMMARY_TEACHER_PROB not in self.evidence_card.ablation_mask:
                raise ValueError("masked teacher_summary.teacher_prob must be declared in ablation_mask")
            return self
        if abs(float(card_teacher_prob) - float(self.teacher_prob)) > 1e-7:
            raise ValueError("sample.teacher_prob must match evidence_card.teacher_summary.teacher_prob")
        return self


class CanonicalTrainingBatch(BaseModel):
    """A small explicit batch wrapper; dataloaders may materialize this object."""

    model_config = _STRICT_MODEL_CONFIG

    samples: tuple[CanonicalTrainingSample, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _batch_header_consistent(self) -> "CanonicalTrainingBatch":
        first = self.samples[0].evidence_card
        for idx, sample in enumerate(self.samples):
            card = sample.evidence_card
            if card.dataset_name != first.dataset_name:
                raise ValueError(f"samples[{idx}] dataset_name mismatch")
            if card.graph_regime != first.graph_regime:
                raise ValueError(f"samples[{idx}] graph_regime mismatch")
        return self


class CanonicalStepReport(BaseModel):
    """Immutable scalar audit report for one optimizer step."""

    model_config = _STRICT_MODEL_CONFIG

    schema_version: Literal["canonical_step/v1"] = "canonical_step/v1"
    n_samples: int = Field(ge=1)
    generation_loss: float = Field(ge=0.0)
    classification_loss: float = Field(ge=0.0)
    distillation_loss: float = Field(ge=0.0)
    total_loss: float = Field(ge=0.0)
    lambda_cls: float = Field(gt=0.0)
    lambda_distill: float = Field(gt=0.0)
    generation_prompt_mode: Literal["train"]
    classification_prompt_mode: Literal["eval_head"]
    distillation_target: Literal["clipped_teacher_prob_bce"]
    thinking_mode: ThinkingMode
    graph_regime: GraphRegime
    used_accelerate_backward: bool

    @model_validator(mode="after")
    def _losses_finite(self) -> "CanonicalStepReport":
        for field_name in ("generation_loss", "classification_loss", "distillation_loss", "total_loss"):
            value = float(getattr(self, field_name))
            if not isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        expected_total = self.generation_loss + self.lambda_cls * self.classification_loss + self.lambda_distill * self.distillation_loss
        if abs(self.total_loss - expected_total) > 1e-5:
            raise ValueError("total_loss must equal L_gen + lambda_cls*L_cls + lambda_distill*L_distill")
        return self


class CanonicalTrainerRunRecord(BaseModel):
    """Run-level provenance record; not a formal report or gate manifest."""

    model_config = _STRICT_MODEL_CONFIG

    schema_version: Literal["canonical_trainer_run/v1"] = "canonical_trainer_run/v1"
    config: CanonicalTrainerConfig
    checkpoint_provenance: CheckpointProvenance | None
    train_population: PopulationMetadata
    validation_population: PopulationMetadata | None
    graph_regime: GraphRegime
    last_step: CanonicalStepReport | None = None
    validation_report: ScorerReport | None = None
    git_commit: StrictStr = Field(pattern=_GIT_HASH_PATTERN)
    git_dirty: StrictBool
    git_diff_hash: StrictStr | None = Field(default=None, pattern=_HEX64_PATTERN)
    teacher_export_train_sha256: StrictStr = Field(pattern=_HEX64_PATTERN)
    teacher_export_validation_sha256: StrictStr | None = Field(pattern=_HEX64_PATTERN)
    data_manifest_sha256: StrictStr = Field(pattern=_HEX64_PATTERN)
    teacher_checkpoint_sha256: StrictStr | None = Field(default=None, pattern=_HEX64_PATTERN)
    adapter_dir_sha256: StrictStr = Field(pattern=_HEX64_PATTERN)
    cls_head_sha256: StrictStr = Field(pattern=_HEX64_PATTERN)
    leakage_policy_version: Literal["evidence_leakage_policy/v1"]
    neighbor_label_policy: Literal["removed_from_student_visible"]
    evidence_card_projection: Literal["student_safe_v1"]
    student_visible_forbidden_fields: tuple[str, ...]
    teacher_prob_masked: Literal[True]
    teacher_logit_masked: Literal[True]
    neighbor_label_counts_visible: Literal[False]
    formal_safe_result: StrictBool
    diagnostic_only: StrictBool
    code_state_clean_for_formal: StrictBool
    prompt_audit_path: StrictStr = Field(min_length=1)
    prompt_audit_hash: StrictStr = Field(pattern=_HEX64_PATTERN)

    @model_validator(mode="after")
    def _record_consistent(self) -> "CanonicalTrainerRunRecord":
        if self.graph_regime != self.config.graph_regime:
            raise ValueError("run record graph_regime must match config.graph_regime")
        if self.train_population.population_name != PopulationName.TRAIN.value:
            raise ValueError("train_population must be named train")
        if self.validation_population is not None and self.validation_population.population_name != PopulationName.VALIDATION.value:
            raise ValueError("validation_population must be named validation")
        if self.student_visible_forbidden_fields != STUDENT_VISIBLE_FORBIDDEN_FIELD_PATHS:
            raise ValueError("run record student_visible_forbidden_fields must match leakage policy")
        if self.code_state_clean_for_formal != (not self.git_dirty):
            raise ValueError("code_state_clean_for_formal must equal not git_dirty")
        if self.git_dirty and self.git_diff_hash is None:
            raise ValueError("dirty run records must include git_diff_hash")
        if self.formal_safe_result:
            if self.git_dirty:
                raise ValueError("formal_safe_result cannot be true when git_dirty is true")
            if self.diagnostic_only:
                raise ValueError("formal_safe_result cannot be true for diagnostic_only runs")
            if not self.code_state_clean_for_formal:
                raise ValueError("formal_safe_result requires code_state_clean_for_formal")
            if self.teacher_export_validation_sha256 is None:
                raise ValueError("formal_safe_result requires teacher_export_validation_sha256")
        return self


def leakage_policy_record_fields(
    *,
    prompt_audit_path: str,
    prompt_audit_hash: str,
    formal_safe_result: bool,
    diagnostic_only: bool,
    code_state_clean_for_formal: bool,
) -> dict[str, object]:
    return {
        "leakage_policy_version": LEAKAGE_POLICY_VERSION,
        "neighbor_label_policy": NEIGHBOR_LABEL_POLICY,
        "evidence_card_projection": EVIDENCE_CARD_PROJECTION,
        "student_visible_forbidden_fields": STUDENT_VISIBLE_FORBIDDEN_FIELD_PATHS,
        "teacher_prob_masked": TEACHER_PROB_MASKED,
        "teacher_logit_masked": TEACHER_LOGIT_MASKED,
        "neighbor_label_counts_visible": NEIGHBOR_LABEL_COUNTS_VISIBLE,
        "formal_safe_result": bool(formal_safe_result),
        "diagnostic_only": bool(diagnostic_only),
        "code_state_clean_for_formal": bool(code_state_clean_for_formal),
        "prompt_audit_path": prompt_audit_path,
        "prompt_audit_hash": prompt_audit_hash,
    }


def prepare_canonical_components(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    cls_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """Prepare canonical train components through Accelerate."""

    return accelerator.prepare(model, cls_head, optimizer)


def run_canonical_train_step(
    *,
    config: CanonicalTrainerConfig,
    batch: CanonicalTrainingBatch,
    model: torch.nn.Module,
    cls_head: TrainableClsHead,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator | None = None,
) -> CanonicalStepReport:
    """Run exactly one canonical joint-training optimizer step.

    This function is intentionally small and auditable.  Full dataloader loops,
    checkpoint writing, and validation scheduling can wrap it, but cannot bypass
    its fail-closed objective requirements.
    """

    accelerator = accelerator or Accelerator()
    _require_trainable_parameters(model, "model")
    _require_trainable_parameters(cls_head, "cls_head")

    model.train()
    cls_head.train(True)
    optimizer.zero_grad(set_to_none=True)

    device = _model_device(model)
    gen_losses: list[torch.Tensor] = []
    cls_logits: list[torch.Tensor] = []
    cls_targets: list[torch.Tensor] = []
    teacher_targets: list[torch.Tensor] = []

    for sample in batch.samples:
        if sample.evidence_card.graph_regime != config.graph_regime:
            raise ValueError("sample graph_regime must match canonical trainer config")
        gen_loss = _compute_generation_loss(
            sample=sample,
            model=model,
            tokenizer=tokenizer,
            thinking_mode=config.thinking_mode,
            device=device,
        )
        gen_losses.append(gen_loss)

        cls_logit = _compute_classification_logit(
            sample=sample,
            model=model,
            cls_head=cls_head,
            tokenizer=tokenizer,
            thinking_mode=config.thinking_mode,
            device=device,
        )
        cls_logits.append(cls_logit.reshape(()))
        cls_targets.append(torch.tensor(float(sample.ground_truth_label), dtype=torch.float32, device=device))
        clipped_teacher_prob = min(max(float(sample.teacher_prob), config.teacher_prob_clip_min), config.teacher_prob_clip_max)
        teacher_targets.append(torch.tensor(clipped_teacher_prob, dtype=torch.float32, device=device))

    generation_loss = torch.stack(gen_losses).mean()
    logits = torch.stack(cls_logits).to(torch.float32)
    cls_target = torch.stack(cls_targets).to(torch.float32)
    teacher_target = torch.stack(teacher_targets).to(torch.float32)
    classification_loss = F.binary_cross_entropy_with_logits(logits, cls_target)
    distillation_loss = F.binary_cross_entropy_with_logits(logits, teacher_target)
    total_loss = generation_loss + config.lambda_cls * classification_loss + config.lambda_distill * distillation_loss

    _require_finite_tensor(generation_loss, "generation_loss")
    _require_finite_tensor(classification_loss, "classification_loss")
    _require_finite_tensor(distillation_loss, "distillation_loss")
    _require_finite_tensor(total_loss, "total_loss")

    accelerator.backward(total_loss)
    if config.max_grad_norm is not None:
        accelerator.clip_grad_norm_(list(model.parameters()) + list(_module_parameters(cls_head)), config.max_grad_norm)
    optimizer.step()

    return CanonicalStepReport(
        n_samples=len(batch.samples),
        generation_loss=float(generation_loss.detach().cpu().item()),
        classification_loss=float(classification_loss.detach().cpu().item()),
        distillation_loss=float(distillation_loss.detach().cpu().item()),
        total_loss=float(total_loss.detach().cpu().item()),
        lambda_cls=float(config.lambda_cls),
        lambda_distill=float(config.lambda_distill),
        generation_prompt_mode=PromptMode.TRAIN.value,
        classification_prompt_mode=PromptMode.EVAL_HEAD.value,
        distillation_target="clipped_teacher_prob_bce",
        thinking_mode=config.thinking_mode,
        graph_regime=config.graph_regime,
        used_accelerate_backward=True,
    )


def run_validation_with_unified_scorer(
    *,
    validation_inputs: HeadScoringInputs,
    model: torch.nn.Module,
    cls_head: Any,
    tokenizer: Any,
    thinking_mode: ThinkingMode,
    prompt_audit_path: str,
    prompt_audit_hash: str,
    accelerator: object | None = None,
) -> ScorerReport:
    """Validate through the unified scorer used by offline head evaluation."""

    return score_head(
        inputs=validation_inputs,
        model=model,
        cls_head=cls_head,
        tokenizer=tokenizer,
        thinking_mode=thinking_mode,
        prompt_audit_path=prompt_audit_path,
        prompt_audit_hash=prompt_audit_hash,
        accelerator=accelerator,
    )


def _compute_generation_loss(
    *,
    sample: CanonicalTrainingSample,
    model: torch.nn.Module,
    tokenizer: Any,
    thinking_mode: ThinkingMode,
    device: torch.device,
) -> torch.Tensor:
    bundle = build_prompt(
        evidence_card=sample.evidence_card,
        mode=PromptMode.TRAIN,
        thinking_mode=thinking_mode,
        ground_truth_label_for_sft=sample.sft_target_label,
        score_target_for_sft=float(sample.sft_target_score),
    )
    encoded = _encode_generation_training_messages(tokenizer, bundle.messages, device=device)
    outputs = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        labels=encoded["labels"],
        output_hidden_states=True,
        use_cache=False,
    )
    loss = getattr(outputs, "loss", None)
    if loss is None:
        raise ValueError("canonical trainer requires model outputs.loss for L_gen")
    if loss.dim() != 0:
        raise ValueError("model outputs.loss for L_gen must be scalar")
    return loss


def _compute_classification_logit(
    *,
    sample: CanonicalTrainingSample,
    model: torch.nn.Module,
    cls_head: TrainableClsHead,
    tokenizer: Any,
    thinking_mode: ThinkingMode,
    device: torch.device,
) -> torch.Tensor:
    bundle = build_prompt(
        evidence_card=sample.evidence_card,
        mode=PromptMode.EVAL_HEAD,
        thinking_mode=thinking_mode,
    )
    encoded = _encode_messages(tokenizer, bundle.messages, device=device)
    outputs = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = getattr(outputs, "hidden_states", None)
    if not hidden_states:
        raise ValueError("canonical trainer requires hidden_states for L_cls")
    pooled = pool_last_valid_token(hidden_states[-1], encoded["attention_mask"])
    logits = cls_head(pooled).to(torch.float32)
    if logits.dim() != 1 or int(logits.shape[0]) != 1:
        raise ValueError(f"cls_head must return shape [1] for B=1 sample; got {tuple(logits.shape)}")
    _require_finite_tensor(logits, "classification logits")
    return logits[0]


def _encode_generation_training_messages(
    tokenizer: Any,
    messages: Sequence[Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not messages:
        raise ValueError("generation training requires non-empty messages")
    final_message = messages[-1]
    if final_message.role != "assistant" or not final_message.content:
        raise ValueError("generation training requires a non-empty final assistant target")

    encoded = _encode_messages(tokenizer, messages, device=device, add_generation_prompt=False)
    prefix_encoded = _encode_messages(
        tokenizer,
        messages[:-1],
        device=device,
        add_generation_prompt=True,
    )
    prefix_len = _require_prefix_match(
        full_input_ids=encoded["input_ids"],
        prefix_input_ids=prefix_encoded["input_ids"],
    )
    labels = encoded["input_ids"].clone()
    labels = labels.masked_fill(encoded["attention_mask"] != 1, -100)
    labels[:, :prefix_len] = -100
    if not torch.any(labels != -100).item():
        raise ValueError("generation training labels must include assistant target tokens")
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


def _encode_messages(
    tokenizer: Any,
    messages: Sequence[Any],
    *,
    device: torch.device,
    add_generation_prompt: bool = False,
) -> dict[str, torch.Tensor]:
    payload = [{"role": message.role, "content": message.content} for message in messages]
    encoded = tokenizer.apply_chat_template(
        payload,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=add_generation_prompt,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    if input_ids.dim() != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("tokenizer must return 2-D input_ids and matching attention_mask")
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _require_prefix_match(*, full_input_ids: torch.Tensor, prefix_input_ids: torch.Tensor) -> int:
    if full_input_ids.dim() != 2 or prefix_input_ids.dim() != 2:
        raise ValueError("full and prefix input_ids must be 2-D")
    if int(full_input_ids.shape[0]) != 1 or int(prefix_input_ids.shape[0]) != 1:
        raise ValueError("generation training currently expects one sample per encoded row")
    prefix_len = int(prefix_input_ids.shape[1])
    full_len = int(full_input_ids.shape[1])
    if prefix_len >= full_len:
        raise ValueError("assistant target must add tokens beyond the prompt prefix")
    if not torch.equal(full_input_ids[:, :prefix_len], prefix_input_ids):
        raise ValueError("assistant-only loss mask requires generation prefix to match full training input")
    return prefix_len


def _require_trainable_parameters(module: object, name: str) -> None:
    params = list(_module_parameters(module))
    if not params:
        raise ValueError(f"canonical trainer requires {name} to expose trainable parameters")
    if not any(param.requires_grad for param in params):
        raise ValueError(f"canonical trainer forbids frozen {name} probe mode")


def _module_parameters(module: object):
    parameters = getattr(module, "parameters", None)
    if parameters is None:
        return ()
    return tuple(parameters())


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _require_finite_tensor(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite (training NaN/Inf blocked)")
