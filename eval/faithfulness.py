"""Formal faithfulness evaluation built on the frozen head-scoring path.

This module intentionally does not build prompts, call the tokenizer, or pool
hidden states directly.  Instead, every faithfulness variant reuses
``eval.head_scoring.score_head`` so the formal faithfulness path stays locked to
exactly the same prompt builder, chat-template serialization, hidden-state
contract, and classification-head inference path as formal head evaluation.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, model_validator

from eval.head_scoring import CheckpointProvenance, HeadScoringInputs, ScorerReport, score_head
from evidence.evidence_schema import EvidenceAblationMask, EvidenceCard, STUDENT_PROMPT_ABLATION_MASK
from evidence.leakage_policy import (
    EVIDENCE_CARD_PROJECTION,
    FORMAL_SAFE_RESULT,
    HEX64_PATTERN,
    LEAKAGE_POLICY_VERSION,
    NEIGHBOR_LABEL_COUNTS_VISIBLE,
    NEIGHBOR_LABEL_POLICY,
    STUDENT_VISIBLE_FORBIDDEN_FIELD_PATHS,
    TEACHER_LOGIT_MASKED,
    TEACHER_PROB_MASKED,
    formal_leakage_provenance_fields,
    validate_formal_leakage_payload,
)
from evidence.prompt_builder import ThinkingMode
from priorf_teacher.schema import DatasetName, GraphRegime, PopulationName

__all__ = (
    "FrozenDecisionPolicy",
    "FaithfulnessInputs",
    "FaithfulnessReport",
    "FaithfulnessSampleResult",
    "evaluate_faithfulness",
)

_STRICT_MODEL_CONFIG: Final[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
_FAITHFULNESS_SCHEMA_VERSION: Final[str] = "faithfulness/v1"
_ALL_ABLATION_FIELDS: Final[frozenset[EvidenceAblationMask]] = frozenset(EvidenceAblationMask)


class FrozenDecisionPolicy(BaseModel):
    """Frozen alpha / threshold provenance for one formal faithfulness run."""

    model_config = _STRICT_MODEL_CONFIG

    alpha: StrictFloat = Field(ge=0.0, le=1.0)
    threshold: StrictFloat = Field(ge=0.0, le=1.0)
    alpha_source: StrictStr = Field(min_length=1)
    threshold_source: StrictStr = Field(min_length=1)


class FaithfulnessInputs(BaseModel):
    """Validated inputs for formal faithfulness evaluation."""

    model_config = _STRICT_MODEL_CONFIG

    full_inputs: HeadScoringInputs
    run_id: StrictStr = Field(min_length=1)
    thinking_mode: ThinkingMode
    frozen_decision_policy: FrozenDecisionPolicy
    selected_evidence_fields: tuple[EvidenceAblationMask, ...] = Field(min_length=1)
    teacher_prob_ablation_fields: tuple[EvidenceAblationMask, ...] = Field(min_length=1)
    minimum_formal_sample_size: StrictInt = Field(ge=1)
    faithfulness_schema_version: Literal["faithfulness/v1"] = _FAITHFULNESS_SCHEMA_VERSION
    prompt_audit_path: StrictStr = Field(min_length=1)
    prompt_audit_hash: StrictStr = Field(pattern=HEX64_PATTERN)

    @model_validator(mode="after")
    def _validate_formal_contract(self) -> "FaithfulnessInputs":
        if len(self.full_inputs.samples) < self.minimum_formal_sample_size:
            raise ValueError(
                "formal faithfulness requires at least minimum_formal_sample_size samples; "
                "use a smoke/diagnostic lane instead"
            )
        for sample in self.full_inputs.samples:
            card = sample.evidence_card
            if card.evidence_card_projection != EVIDENCE_CARD_PROJECTION:
                raise ValueError("formal faithfulness inputs must use student-safe Evidence Cards by default")
            if not STUDENT_PROMPT_ABLATION_MASK.issubset(card.ablation_mask):
                raise ValueError("formal faithfulness student-safe inputs must mask direct teacher score fields")
            neighbor_payload = card.neighbor_summary.model_dump(mode="json")
            forbidden = {"labeled_neighbors", "positive_neighbors", "negative_neighbors", "unlabeled_neighbors"}
            leaked = sorted(forbidden.intersection(neighbor_payload))
            if leaked:
                raise ValueError("formal faithfulness inputs expose forbidden neighbor fields: " + ", ".join(leaked))

        selected = frozenset(self.selected_evidence_fields)
        if len(selected) != len(self.selected_evidence_fields):
            raise ValueError("selected_evidence_fields must be unique")
        if selected == _ALL_ABLATION_FIELDS:
            raise ValueError("selected_evidence_fields must not cover the entire Evidence Card")

        teacher_prob_mask = frozenset(self.teacher_prob_ablation_fields)
        if len(teacher_prob_mask) != len(self.teacher_prob_ablation_fields):
            raise ValueError("teacher_prob_ablation_fields must be unique")
        if EvidenceAblationMask.TEACHER_SUMMARY_TEACHER_PROB not in teacher_prob_mask:
            raise ValueError("teacher_prob_ablation_fields must include teacher_summary.teacher_prob")
        return self


class FaithfulnessSampleResult(BaseModel):
    """Per-sample formal faithfulness deltas under frozen threshold alignment."""

    model_config = _STRICT_MODEL_CONFIG

    node_id: StrictInt = Field(ge=0)
    ground_truth_label: Literal[0, 1]
    full_prob: float = Field(ge=0.0, le=1.0)
    sufficiency_prob: float = Field(ge=0.0, le=1.0)
    comprehensiveness_prob: float = Field(ge=0.0, le=1.0)
    teacher_prob_ablation_prob: float = Field(ge=0.0, le=1.0)
    full_decision: Literal[0, 1]
    sufficiency_decision: Literal[0, 1]
    comprehensiveness_decision: Literal[0, 1]
    teacher_prob_ablation_decision: Literal[0, 1]
    sufficiency: float
    comprehensiveness: float
    evidence_ablation_impact: float


class FaithfulnessReport(BaseModel):
    """Strict formal faithfulness report."""

    model_config = _STRICT_MODEL_CONFIG

    dataset_name: DatasetName
    population_name: PopulationName
    graph_regime: GraphRegime
    run_id: StrictStr = Field(min_length=1)
    report_split: PopulationName
    eval_type: Literal["faithfulness"]
    checkpoint_provenance: CheckpointProvenance
    scorer_schema_version: Literal["head_scorer/v1"]
    faithfulness_schema_version: Literal["faithfulness/v1"]

    n_total: StrictInt = Field(ge=1)
    minimum_formal_sample_size: StrictInt = Field(ge=1)
    frozen_decision_policy: FrozenDecisionPolicy
    selected_evidence_fields: tuple[EvidenceAblationMask, ...] = Field(min_length=1)
    teacher_prob_ablation_fields: tuple[EvidenceAblationMask, ...] = Field(min_length=1)

    full_report: ScorerReport
    sufficiency_report: ScorerReport
    comprehensiveness_report: ScorerReport
    teacher_prob_ablation_report: ScorerReport

    sample_results: tuple[FaithfulnessSampleResult, ...] = Field(min_length=1)
    mean_sufficiency: float
    mean_comprehensiveness: float
    mean_evidence_ablation_impact: float
    decision_flip_rate_sufficiency: float = Field(ge=0.0, le=1.0)
    decision_flip_rate_comprehensiveness: float = Field(ge=0.0, le=1.0)
    decision_flip_rate_teacher_prob_ablation: float = Field(ge=0.0, le=1.0)

    prompt_mode: Literal["eval_head"]
    thinking_mode: ThinkingMode
    pooling_path: Literal["pool_last_valid_token"]
    uses_inference_mode: bool
    distributed_gather: Literal["none", "accelerate_gather_for_metrics"]
    leakage_policy_version: Literal["evidence_leakage_policy/v1"]
    neighbor_label_policy: Literal["removed_from_student_visible"]
    evidence_card_projection: Literal["student_safe_v1"]
    student_visible_forbidden_fields: tuple[str, ...]
    teacher_prob_masked: Literal[True]
    teacher_logit_masked: Literal[True]
    neighbor_label_counts_visible: Literal[False]
    formal_safe_result: Literal[True]
    prompt_audit_path: StrictStr = Field(min_length=1)
    prompt_audit_hash: StrictStr = Field(pattern=HEX64_PATTERN)

    @model_validator(mode="after")
    def _formal_leakage_fields_consistent(self) -> "FaithfulnessReport":
        validate_formal_leakage_payload(self.model_dump(mode="python"), context="FaithfulnessReport")
        return self

    @model_validator(mode="after")
    def _nested_reports_must_match(self) -> "FaithfulnessReport":
        nested = (
            self.full_report,
            self.sufficiency_report,
            self.comprehensiveness_report,
            self.teacher_prob_ablation_report,
        )
        if len(self.sample_results) != self.n_total:
            raise ValueError("len(sample_results) must equal n_total")
        if self.report_split != self.population_name:
            raise ValueError("report_split must match population_name")
        for report in nested:
            if report.dataset_name != self.dataset_name:
                raise ValueError("nested report dataset_name mismatch")
            if report.population_name != self.population_name:
                raise ValueError("nested report population_name mismatch")
            if report.graph_regime != self.graph_regime:
                raise ValueError("nested report graph_regime mismatch")
            if report.checkpoint_provenance != self.checkpoint_provenance:
                raise ValueError("nested report checkpoint_provenance mismatch")
            if report.scorer_schema_version != self.scorer_schema_version:
                raise ValueError("nested report scorer_schema_version mismatch")
            if report.n_total != self.n_total:
                raise ValueError("nested report n_total mismatch")
            if report.prompt_mode != self.prompt_mode:
                raise ValueError("nested report prompt_mode mismatch")
            if report.thinking_mode != self.thinking_mode:
                raise ValueError("nested report thinking_mode mismatch")
            if report.pooling_path != self.pooling_path:
                raise ValueError("nested report pooling_path mismatch")
            if report.uses_inference_mode != self.uses_inference_mode:
                raise ValueError("nested report uses_inference_mode mismatch")
            if report.distributed_gather != self.distributed_gather:
                raise ValueError("nested report distributed_gather mismatch")
        return self


def evaluate_faithfulness(
    *,
    inputs: FaithfulnessInputs,
    model,
    cls_head,
    tokenizer,
    accelerator: object | None = None,
) -> FaithfulnessReport:
    """Run formal faithfulness variants through the frozen head-scoring path."""

    validated_inputs = FaithfulnessInputs.model_validate(inputs)

    full_report = score_head(
        inputs=validated_inputs.full_inputs,
        model=model,
        cls_head=cls_head,
        tokenizer=tokenizer,
        thinking_mode=validated_inputs.thinking_mode,
        prompt_audit_path=validated_inputs.prompt_audit_path,
        prompt_audit_hash=validated_inputs.prompt_audit_hash,
        accelerator=accelerator,
    )
    sufficiency_report = score_head(
        inputs=_with_ablation(
            validated_inputs.full_inputs,
            _ALL_ABLATION_FIELDS.difference(validated_inputs.selected_evidence_fields),
        ),
        model=model,
        cls_head=cls_head,
        tokenizer=tokenizer,
        thinking_mode=validated_inputs.thinking_mode,
        prompt_audit_path=validated_inputs.prompt_audit_path,
        prompt_audit_hash=validated_inputs.prompt_audit_hash,
        accelerator=accelerator,
    )
    comprehensiveness_report = score_head(
        inputs=_with_ablation(validated_inputs.full_inputs, validated_inputs.selected_evidence_fields),
        model=model,
        cls_head=cls_head,
        tokenizer=tokenizer,
        thinking_mode=validated_inputs.thinking_mode,
        prompt_audit_path=validated_inputs.prompt_audit_path,
        prompt_audit_hash=validated_inputs.prompt_audit_hash,
        accelerator=accelerator,
    )
    teacher_prob_ablation_report = score_head(
        inputs=_with_ablation(validated_inputs.full_inputs, validated_inputs.teacher_prob_ablation_fields),
        model=model,
        cls_head=cls_head,
        tokenizer=tokenizer,
        thinking_mode=validated_inputs.thinking_mode,
        prompt_audit_path=validated_inputs.prompt_audit_path,
        prompt_audit_hash=validated_inputs.prompt_audit_hash,
        accelerator=accelerator,
    )

    _assert_path_equivalence(
        baseline=full_report,
        variants=(sufficiency_report, comprehensiveness_report, teacher_prob_ablation_report),
    )

    threshold = float(validated_inputs.frozen_decision_policy.threshold)
    sample_results: list[FaithfulnessSampleResult] = []
    sufficiency_values: list[float] = []
    comprehensiveness_values: list[float] = []
    impact_values: list[float] = []
    sufficiency_flips = 0
    comprehensiveness_flips = 0
    teacher_prob_flips = 0

    for sample, full_prob, suff_prob, comp_prob, teacher_prob in zip(
        validated_inputs.full_inputs.samples,
        full_report.probs,
        sufficiency_report.probs,
        comprehensiveness_report.probs,
        teacher_prob_ablation_report.probs,
        strict=True,
    ):
        full_decision = _decision(full_prob, threshold)
        sufficiency_decision = _decision(suff_prob, threshold)
        comprehensiveness_decision = _decision(comp_prob, threshold)
        teacher_prob_decision = _decision(teacher_prob, threshold)

        full_target_prob = _decision_aligned_probability(full_prob, full_decision)
        suff_target_prob = _decision_aligned_probability(suff_prob, full_decision)
        comp_target_prob = _decision_aligned_probability(comp_prob, full_decision)
        teacher_target_prob = _decision_aligned_probability(teacher_prob, full_decision)

        sufficiency = full_target_prob - suff_target_prob
        comprehensiveness = full_target_prob - comp_target_prob
        evidence_ablation_impact = full_target_prob - teacher_target_prob

        sufficiency_values.append(sufficiency)
        comprehensiveness_values.append(comprehensiveness)
        impact_values.append(evidence_ablation_impact)
        sufficiency_flips += int(sufficiency_decision != full_decision)
        comprehensiveness_flips += int(comprehensiveness_decision != full_decision)
        teacher_prob_flips += int(teacher_prob_decision != full_decision)

        sample_results.append(
            FaithfulnessSampleResult(
                node_id=sample.node_id,
                ground_truth_label=sample.ground_truth_label,
                full_prob=full_prob,
                sufficiency_prob=suff_prob,
                comprehensiveness_prob=comp_prob,
                teacher_prob_ablation_prob=teacher_prob,
                full_decision=full_decision,
                sufficiency_decision=sufficiency_decision,
                comprehensiveness_decision=comprehensiveness_decision,
                teacher_prob_ablation_decision=teacher_prob_decision,
                sufficiency=sufficiency,
                comprehensiveness=comprehensiveness,
                evidence_ablation_impact=evidence_ablation_impact,
            )
        )

    n_total = len(sample_results)
    return FaithfulnessReport(
        dataset_name=validated_inputs.full_inputs.dataset_name,
        population_name=validated_inputs.full_inputs.population_name,
        graph_regime=validated_inputs.full_inputs.graph_regime,
        run_id=validated_inputs.run_id,
        report_split=validated_inputs.full_inputs.population_name,
        eval_type="faithfulness",
        checkpoint_provenance=validated_inputs.full_inputs.checkpoint_provenance,
        scorer_schema_version=validated_inputs.full_inputs.scorer_schema_version,
        faithfulness_schema_version=validated_inputs.faithfulness_schema_version,
        n_total=n_total,
        minimum_formal_sample_size=validated_inputs.minimum_formal_sample_size,
        frozen_decision_policy=validated_inputs.frozen_decision_policy,
        selected_evidence_fields=validated_inputs.selected_evidence_fields,
        teacher_prob_ablation_fields=validated_inputs.teacher_prob_ablation_fields,
        full_report=full_report,
        sufficiency_report=sufficiency_report,
        comprehensiveness_report=comprehensiveness_report,
        teacher_prob_ablation_report=teacher_prob_ablation_report,
        sample_results=tuple(sample_results),
        mean_sufficiency=float(np.mean(np.asarray(sufficiency_values, dtype=np.float64))),
        mean_comprehensiveness=float(np.mean(np.asarray(comprehensiveness_values, dtype=np.float64))),
        mean_evidence_ablation_impact=float(np.mean(np.asarray(impact_values, dtype=np.float64))),
        decision_flip_rate_sufficiency=float(sufficiency_flips / n_total),
        decision_flip_rate_comprehensiveness=float(comprehensiveness_flips / n_total),
        decision_flip_rate_teacher_prob_ablation=float(teacher_prob_flips / n_total),
        prompt_mode=full_report.prompt_mode,
        thinking_mode=full_report.thinking_mode,
        pooling_path=full_report.pooling_path,
        uses_inference_mode=full_report.uses_inference_mode,
        distributed_gather=full_report.distributed_gather,
        **formal_leakage_provenance_fields(
            prompt_audit_path=validated_inputs.prompt_audit_path,
            prompt_audit_hash=validated_inputs.prompt_audit_hash,
        ),
    )


def _with_ablation(inputs: HeadScoringInputs, mask: frozenset[EvidenceAblationMask] | tuple[EvidenceAblationMask, ...]) -> HeadScoringInputs:
    applied_mask = frozenset(mask)
    ablated_samples = tuple(
        sample.model_copy(update={"evidence_card": _apply_ablation_mask(sample.evidence_card, applied_mask)})
        for sample in inputs.samples
    )
    return inputs.model_copy(update={"samples": ablated_samples})


def _apply_ablation_mask(card: EvidenceCard, mask: frozenset[EvidenceAblationMask]) -> EvidenceCard:
    payload = card.model_dump(mode="python")
    payload["ablation_mask"] = frozenset(card.ablation_mask) | mask
    for item in mask:
        section, field_name = item.value.split(".", 1)
        if section in {"teacher_summary", "discrepancy_summary"}:
            payload[section][field_name] = None
    return EvidenceCard.model_validate(payload)


def _assert_path_equivalence(*, baseline: ScorerReport, variants: tuple[ScorerReport, ...]) -> None:
    for variant in variants:
        if variant.prompt_mode != baseline.prompt_mode:
            raise ValueError("faithfulness variants must reuse the same prompt_mode as the baseline scorer")
        if variant.thinking_mode != baseline.thinking_mode:
            raise ValueError("faithfulness variants must reuse the same thinking_mode as the baseline scorer")
        if variant.pooling_path != baseline.pooling_path:
            raise ValueError("faithfulness variants must reuse the same pooling_path as the baseline scorer")
        if variant.uses_inference_mode != baseline.uses_inference_mode:
            raise ValueError("faithfulness variants must reuse inference_mode settings")
        if variant.distributed_gather != baseline.distributed_gather:
            raise ValueError("faithfulness variants must reuse the same distributed gather path")


def _decision(prob: float, threshold: float) -> Literal[0, 1]:
    return 1 if float(prob) >= threshold else 0


def _decision_aligned_probability(prob: float, reference_decision: Literal[0, 1]) -> float:
    return float(prob) if reference_decision == 1 else float(1.0 - prob)
