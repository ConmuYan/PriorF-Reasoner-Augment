"""Canonical prompt builder for Evidence Cards.

Task 3 scope only: builds typed chat-message bundles.  This module intentionally
avoids tokenizer/chat-template calls, model inference, trainer logic, filesystem
IO, and dependency imports from transformers/torch/accelerate/peft/trl/datasets.
"""

from __future__ import annotations

import math
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
            evidence_card=validated_card,
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
    bundle = PromptBundle(
        messages=messages,
        sft_target_label=target,
        mode=validated_mode,
        thinking_mode=validated_thinking_mode,
    )
    from evidence.prompt_audit import assert_prompt_audit_passes, audit_prompt_bundle

    assert_prompt_audit_passes(audit_prompt_bundle(bundle))
    return bundle


def _build_sft_target(*, evidence_card: EvidenceCard, label: PredLabel, score: float) -> StrictOutput:
    discrepancy = evidence_card.discrepancy_summary
    relation = evidence_card.relation_profile
    teacher = evidence_card.teacher_summary
    rationale = (
        "Reasoning uses non-score structural signals from the Evidence Card: "
        f"hsd_quantile={_compact_numeric(teacher.hsd_quantile)}, "
        f"branch_gap={_compact_numeric(teacher.branch_gap)}, "
        f"discrepancy_severity={discrepancy.discrepancy_severity}, "
        f"route_hint={discrepancy.route_hint}, active_relations={relation.active_relations}, "
        f"total_neighbors={evidence_card.neighbor_summary.total_neighbors}."
    )
    return StrictOutput(
        rationale=rationale,
        evidence=(
            f"discrepancy_severity={discrepancy.discrepancy_severity}; route_hint={discrepancy.route_hint}",
            f"relation_discrepancy_mean={_compact_numeric(relation.mean_relation_discrepancy)}; "
            f"active_relations={relation.active_relations}",
            f"neighborhood_size={evidence_card.neighbor_summary.total_neighbors}",
        ),
        pattern_hint=(
            f"{evidence_card.graph_regime.value}; hsd_quantile={_compact_numeric(teacher.hsd_quantile)}; "
            f"branch_gap_abs={_compact_numeric(discrepancy.branch_gap_abs)}"
        ),
        label=label,
        score=_rounded_score(score),
    )


def _compact_numeric(value: float | int | None) -> str:
    if value is None:
        raise ValueError("SFT target fields must not be None")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("SFT target numeric fields must be finite")
    if numeric.is_integer():
        return str(int(numeric))
    compact = f"{numeric:.4f}".rstrip("0").rstrip(".")
    if compact == "-0":
        return "0"
    return compact


def _rounded_score(score: float) -> float:
    numeric = float(score)
    if not math.isfinite(numeric):
        raise ValueError("score_target_for_sft must be finite")
    rounded = round(numeric, 4)
    return min(1.0, max(0.0, rounded))


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
                "The first character must be '{' and the final character must be '}'. "
                "Do not output markdown, code fences, preamble text, or trailing text. "
                "Use double-quoted JSON strings, keep rationale / evidence / pattern_hint concise, "
                "use label exactly 'fraud' or 'benign', and emit score as a JSON number in [0,1]. "
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
        f"evidence_card_projection: {card.evidence_card_projection}",
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
