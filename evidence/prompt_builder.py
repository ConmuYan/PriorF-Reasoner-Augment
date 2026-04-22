"""Canonical prompt builder for Evidence Cards.

Task 3 scope only: builds typed chat-message bundles.  This module intentionally
avoids tokenizer/chat-template calls, model inference, trainer logic, filesystem
IO, and dependency imports from transformers/torch/accelerate/peft/trl/datasets.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from evidence.evidence_schema import EvidenceAblationMask, EvidenceCard
from evidence.output_schema import PredLabel, StrictOutput, canonical_serialize
from priorf_teacher.schema import PopulationName

_STRICT_MODEL_CONFIG: Final[ConfigDict] = ConfigDict(
    extra="forbid",
    frozen=True,
    populate_by_name=False,
    str_to_lower=False,
    str_to_upper=False,
    str_strip_whitespace=False,
)
_ENV_PREFIXES: Final[tuple[str, str, str]] = ("PRIORF_PROMPT_", "EVIDENCE_", "PROMPT_BUILDER_")
_MASKED_SENTINEL: Final[str] = "Masked / Not Available"


class PromptMode(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    EVAL_HEAD = "eval_head"
    EVAL_GEN = "eval_gen"
    GENERATION = "generation"


class ThinkingMode(str, Enum):
    THINKING = "thinking"
    NON_THINKING = "non_thinking"


class ChatMessage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    role: Literal["system", "user", "assistant"]
    content: StrictStr

    @model_validator(mode="after")
    def _non_assistant_content_must_be_non_empty(self) -> "ChatMessage":
        if self.role != "assistant" and not self.content:
            raise ValueError("system and user message content must be non-empty")
        return self


class PromptBundle(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    sft_target_label: StrictOutput | None
    mode: PromptMode
    thinking_mode: ThinkingMode

    @model_validator(mode="after")
    def _sft_target_matches_mode(self) -> "PromptBundle":
        if self.mode == PromptMode.TRAIN and self.sft_target_label is None:
            raise ValueError("train mode requires sft_target_label")
        if self.mode != PromptMode.TRAIN and self.sft_target_label is not None:
            raise ValueError("non-train modes must not carry sft_target_label")
        if not any(message.role == "assistant" for message in self.messages):
            raise ValueError("PromptBundle must contain an assistant role")
        return self


class FewShotExample(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    evidence_card: EvidenceCard
    sft_target_label: StrictOutput
    source_population: PopulationName

    @model_validator(mode="after")
    def _few_shot_must_be_train_population(self) -> "FewShotExample":
        if self.source_population != PopulationName.TRAIN:
            raise ValueError("few-shot examples must come from train population")
        return self


def build_prompt(
    *,
    evidence_card: EvidenceCard,
    mode: PromptMode,
    thinking_mode: ThinkingMode,
    ground_truth_label_for_sft: PredLabel | str | None = None,
    score_target_for_sft: float | None = None,
    few_shot_examples: tuple[FewShotExample, ...] = (),
) -> PromptBundle:
    """Build the one canonical message structure for train/validation/eval modes."""

    _reject_prompt_environment_overrides()
    validated_card = EvidenceCard.model_validate(evidence_card)
    validated_mode = PromptMode(mode)
    validated_thinking_mode = ThinkingMode(thinking_mode)
    validated_examples = tuple(FewShotExample.model_validate(example) for example in few_shot_examples)

    if validated_mode == PromptMode.TRAIN:
        if ground_truth_label_for_sft is None or score_target_for_sft is None:
            raise ValueError("train mode requires ground_truth_label_for_sft and score_target_for_sft")
        target = _build_sft_target(
            label=PredLabel(ground_truth_label_for_sft),
            score=score_target_for_sft,
        )
        assistant_content = canonical_serialize(target)
    else:
        if ground_truth_label_for_sft is not None or score_target_for_sft is not None:
            raise ValueError("non-train modes must not receive SFT labels or score targets")
        target = None
        assistant_content = ""

    if validated_mode in {PromptMode.EVAL_HEAD, PromptMode.EVAL_GEN, PromptMode.GENERATION} and validated_examples:
        raise ValueError("eval_head, eval_gen, and generation modes must not include few-shot examples")

    messages = _build_messages(
        evidence_card=validated_card,
        thinking_mode=validated_thinking_mode,
        few_shot_examples=validated_examples,
        assistant_content=assistant_content,
    )
    return PromptBundle(
        messages=messages,
        sft_target_label=target,
        mode=validated_mode,
        thinking_mode=validated_thinking_mode,
    )


def _build_sft_target(*, label: PredLabel, score: float) -> StrictOutput:
    return StrictOutput(
        rationale="The structured graph evidence should be considered before assigning the training target.",
        evidence=("Evidence Card fields are the only permitted basis for this structured target.",),
        pattern_hint="Use the declared graph-regime structural pattern; no raw text or raw graph matrix is available.",
        label=label,
        score=score,
    )


def _build_messages(
    *,
    evidence_card: EvidenceCard,
    thinking_mode: ThinkingMode,
    few_shot_examples: tuple[FewShotExample, ...],
    assistant_content: str,
) -> tuple[ChatMessage, ...]:
    messages: list[ChatMessage] = [
        ChatMessage(
            role="system",
            content=(
                "You are PriorF-Reasoner. Produce exactly one strict JSON object with keys "
                "rationale, evidence, pattern_hint, label, score in that order. "
                f"Thinking mode is explicitly fixed to {thinking_mode.value}."
            ),
        )
    ]
    for example in few_shot_examples:
        messages.append(ChatMessage(role="user", content=_render_evidence_card(example.evidence_card)))
        messages.append(ChatMessage(role="assistant", content=canonical_serialize(example.sft_target_label)))
    messages.append(ChatMessage(role="user", content=_render_evidence_card(evidence_card)))
    messages.append(ChatMessage(role="assistant", content=assistant_content))
    return tuple(messages)


def _render_evidence_card(card: EvidenceCard) -> str:
    mask = card.ablation_mask
    lines = [
        "Evidence Card:",
        f"schema_version: {card.schema_version}",
        f"dataset_name: {card.dataset_name.value}",
        f"population_name: {card.population_name.value}",
        f"graph_regime: {card.graph_regime.value}",
        f"node_id: {card.node_id}",
        "task_instruction:",
        f"  text: {card.task_instruction.text}",
        f"  graph_regime: {card.graph_regime.value}",
        f"  schema_hint_order: {', '.join(card.task_instruction.schema_hint_order)}",
        "teacher_summary:",
        f"  teacher_prob: {_render_value('teacher_summary.teacher_prob', card.teacher_summary.teacher_prob, mask)}",
        f"  teacher_logit: {_render_value('teacher_summary.teacher_logit', card.teacher_summary.teacher_logit, mask)}",
        f"  hsd: {_render_value('teacher_summary.hsd', card.teacher_summary.hsd, mask)}",
        f"  hsd_quantile: {_render_value('teacher_summary.hsd_quantile', card.teacher_summary.hsd_quantile, mask)}",
        f"  asda_switch: {_render_value('teacher_summary.asda_switch', card.teacher_summary.asda_switch, mask)}",
        f"  mlp_logit: {_render_value('teacher_summary.mlp_logit', card.teacher_summary.mlp_logit, mask)}",
        f"  gnn_logit: {_render_value('teacher_summary.gnn_logit', card.teacher_summary.gnn_logit, mask)}",
        f"  branch_gap: {_render_value('teacher_summary.branch_gap', card.teacher_summary.branch_gap, mask)}",
        f"  high_hsd_flag: {_render_value('teacher_summary.high_hsd_flag', card.teacher_summary.high_hsd_flag, mask)}",
        "discrepancy_summary:",
        f"  branch_gap_abs: {_render_value('discrepancy_summary.branch_gap_abs', card.discrepancy_summary.branch_gap_abs, mask)}",
        f"  teacher_mlp_agreement: {_render_value('discrepancy_summary.teacher_mlp_agreement', card.discrepancy_summary.teacher_mlp_agreement, mask)}",
        f"  teacher_gnn_agreement: {_render_value('discrepancy_summary.teacher_gnn_agreement', card.discrepancy_summary.teacher_gnn_agreement, mask)}",
        f"  discrepancy_severity: {_render_value('discrepancy_summary.discrepancy_severity', card.discrepancy_summary.discrepancy_severity, mask)}",
        f"  route_hint: {_render_value('discrepancy_summary.route_hint', card.discrepancy_summary.route_hint, mask)}",
        "relation_profile:",
        f"  total_relations: {_render_value('relation_profile.total_relations', card.relation_profile.total_relations, mask)}",
        f"  active_relations: {_render_value('relation_profile.active_relations', card.relation_profile.active_relations, mask)}",
        f"  max_relation_neighbor_count: {_render_value('relation_profile.max_relation_neighbor_count', card.relation_profile.max_relation_neighbor_count, mask)}",
        f"  mean_relation_neighbor_count: {_render_value('relation_profile.mean_relation_neighbor_count', card.relation_profile.mean_relation_neighbor_count, mask)}",
        f"  max_relation_discrepancy: {_render_value('relation_profile.max_relation_discrepancy', card.relation_profile.max_relation_discrepancy, mask)}",
        f"  mean_relation_discrepancy: {_render_value('relation_profile.mean_relation_discrepancy', card.relation_profile.mean_relation_discrepancy, mask)}",
        "neighbor_summary:",
        f"  total_neighbors: {_render_value('neighbor_summary.total_neighbors', card.neighbor_summary.total_neighbors, mask)}",
        f"  labeled_neighbors: {_render_value('neighbor_summary.labeled_neighbors', card.neighbor_summary.labeled_neighbors, mask)}",
        f"  positive_neighbors: {_render_value('neighbor_summary.positive_neighbors', card.neighbor_summary.positive_neighbors, mask)}",
        f"  negative_neighbors: {_render_value('neighbor_summary.negative_neighbors', card.neighbor_summary.negative_neighbors, mask)}",
        f"  unlabeled_neighbors: {_render_value('neighbor_summary.unlabeled_neighbors', card.neighbor_summary.unlabeled_neighbors, mask)}",
        f"ablation_mask: {', '.join(sorted(item.value for item in mask)) if mask else 'None'}",
    ]
    return "\n".join(lines)


def _render_value(mask_value: str, value: object, mask: frozenset[EvidenceAblationMask]) -> str:
    if EvidenceAblationMask(mask_value) in mask:
        return _MASKED_SENTINEL
    if value is None:
        return _MASKED_SENTINEL
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _reject_prompt_environment_overrides() -> None:
    for key in os.environ:
        if key.startswith(_ENV_PREFIXES):
            raise RuntimeError(f"prompt builder environment override is forbidden: {key}")
