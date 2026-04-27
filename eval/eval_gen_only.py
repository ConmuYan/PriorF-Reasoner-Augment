"""Formal generation-only evaluation with strict vs normalized parsing.

The formal/headline parse-success metric is always ``strict_schema_parse_rate``.
The normalized parser is recorded only for diagnostics.  When strict parsing
fails, this evaluator uses explicit ``penalty_mode`` so parse failures remain
in the denominator and are mapped to a predetermined worst-case prediction.
"""

from __future__ import annotations

import json
import math
from typing import Literal

import numpy as np
import sklearn.metrics
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationError, model_validator

from evidence.evidence_schema import EvidenceCard
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
from evidence.output_schema import StrictOutput, parse_strict
from llm.parsing import NormalizedParseError, parse_normalized_output
from priorf_teacher.schema import DatasetName, GraphRegime, PopulationName

__all__ = (
    "GenOnlyEvalInputs",
    "GenOnlyEvalReport",
    "GenOnlyEvalSample",
    "evaluate_gen_only",
)

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_GEN_EVAL_SCHEMA_VERSION = "gen_eval/v1"


class GenOnlyEvalSample(BaseModel):
    """One generation-eval row: evidence card, ground truth, and raw output."""

    model_config = _STRICT_MODEL_CONFIG

    evidence_card: EvidenceCard
    generated_text: StrictStr = Field(min_length=1)
    ground_truth_label: Literal[0, 1]
    node_id: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def _node_id_matches_card(self) -> "GenOnlyEvalSample":
        if self.evidence_card.node_id != self.node_id:
            raise ValueError("sample.node_id must equal sample.evidence_card.node_id")
        return self


class GenOnlyEvalInputs(BaseModel):
    """Validated inputs to the formal generation-only evaluator."""

    model_config = _STRICT_MODEL_CONFIG

    samples: tuple[GenOnlyEvalSample, ...] = Field(min_length=1)
    dataset_name: DatasetName
    population_name: PopulationName
    graph_regime: GraphRegime
    run_id: StrictStr = Field(min_length=1)
    teacher_conditioned: Literal[True] = True
    strict_parse_failure_policy: Literal["penalty_mode"] = "penalty_mode"
    gen_eval_schema_version: Literal["gen_eval/v1"] = _GEN_EVAL_SCHEMA_VERSION
    prompt_audit_path: StrictStr = Field(min_length=1)
    prompt_audit_hash: StrictStr = Field(pattern=HEX64_PATTERN)

    @model_validator(mode="after")
    def _samples_consistent_with_header(self) -> "GenOnlyEvalInputs":
        for idx, sample in enumerate(self.samples):
            card = sample.evidence_card
            if card.dataset_name != self.dataset_name:
                raise ValueError(
                    f"samples[{idx}].evidence_card.dataset_name must equal top-level dataset_name"
                )
            if card.population_name != self.population_name:
                raise ValueError(
                    f"samples[{idx}].evidence_card.population_name must equal top-level population_name"
                )
            if card.graph_regime != self.graph_regime:
                raise ValueError(
                    f"samples[{idx}].evidence_card.graph_regime must equal top-level graph_regime"
                )
        return self


class GenOnlyEvalReport(BaseModel):
    """Strict formal generation-eval report."""

    model_config = _STRICT_MODEL_CONFIG

    dataset_name: DatasetName
    population_name: PopulationName
    graph_regime: GraphRegime
    run_id: StrictStr = Field(min_length=1)
    report_split: PopulationName
    eval_type: Literal["gen_only"]
    gen_eval_schema_version: Literal["gen_eval/v1"]
    teacher_conditioned: Literal[True]
    prompt_mode: Literal["eval_gen"]

    headline_metric_name: Literal["strict_schema_parse_rate"]
    headline_metric_value: float = Field(ge=0.0, le=1.0)
    strict_schema_parse_rate: float = Field(ge=0.0, le=1.0)
    normalized_parse_rate: float = Field(ge=0.0, le=1.0)

    strict_parse_failure_policy: Literal["penalty_mode"]
    strict_parse_penalty_strategy: Literal["ground_truth_opposite_extreme_probability"]
    strict_parse_failure_count: int = Field(ge=0)
    strict_parse_failure_rate: float = Field(ge=0.0, le=1.0)
    strict_parse_failure_node_ids: tuple[int, ...]
    normalized_parse_success_count: int = Field(ge=0)

    n_total: int = Field(ge=1)
    n_positive: int = Field(ge=0)
    n_negative: int = Field(ge=0)
    is_single_class_population: bool

    auroc: float | None
    auprc: float | None
    brier_score: float

    probs: tuple[float, ...]
    labels: tuple[Literal[0, 1], ...]
    node_ids: tuple[int, ...]
    strict_parse_succeeded: tuple[bool, ...]
    normalized_parse_succeeded: tuple[bool, ...]
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
    def _formal_leakage_fields_consistent(self) -> "GenOnlyEvalReport":
        validate_formal_leakage_payload(self.model_dump(mode="python"), context="GenOnlyEvalReport")
        return self

    @model_validator(mode="after")
    def _counts_and_arrays_consistent(self) -> "GenOnlyEvalReport":
        if self.n_positive + self.n_negative != self.n_total:
            raise ValueError("n_positive + n_negative must equal n_total")
        if self.report_split != self.population_name:
            raise ValueError("report_split must match population_name")
        for field_name in (
            "probs",
            "labels",
            "node_ids",
            "strict_parse_succeeded",
            "normalized_parse_succeeded",
        ):
            if len(getattr(self, field_name)) != self.n_total:
                raise ValueError(f"len({field_name}) must equal n_total")
        if self.strict_parse_failure_count != len(self.strict_parse_failure_node_ids):
            raise ValueError("strict_parse_failure_count must equal len(strict_parse_failure_node_ids)")
        for prob in self.probs:
            if not math.isfinite(prob) or prob < 0.0 or prob > 1.0:
                raise ValueError("probs must be finite and within [0.0, 1.0]")
        return self


def evaluate_gen_only(*, inputs: GenOnlyEvalInputs) -> GenOnlyEvalReport:
    """Run formal generation-only evaluation using strict headline parsing."""

    strict_success_mask: list[bool] = []
    normalized_success_mask: list[bool] = []
    strict_failure_node_ids: list[int] = []
    probs: list[float] = []
    labels: list[int] = []
    node_ids: list[int] = []

    for sample in inputs.samples:
        strict_output = _try_parse_strict(sample.generated_text)
        normalized_output = _try_parse_normalized(sample.generated_text)

        strict_succeeded = strict_output is not None
        normalized_succeeded = normalized_output is not None
        strict_success_mask.append(strict_succeeded)
        normalized_success_mask.append(normalized_succeeded)
        labels.append(int(sample.ground_truth_label))
        node_ids.append(int(sample.node_id))

        if strict_succeeded:
            probs.append(_predicted_positive_probability(strict_output))
        else:
            strict_failure_node_ids.append(int(sample.node_id))
            probs.append(_worst_case_positive_probability(sample.ground_truth_label))

    probs_np = np.asarray(probs, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int64)
    node_ids_np = np.asarray(node_ids, dtype=np.int64)

    n_total = int(probs_np.shape[0])
    if n_total == 0:
        raise ValueError("n_total must be > 0; population is empty")

    if not np.all(np.isfinite(probs_np)):
        raise ValueError("probs contain non-finite values")

    n_positive = int((labels_np == 1).sum())
    n_negative = int((labels_np == 0).sum())
    if n_positive + n_negative != n_total:
        raise ValueError("labels must be strictly 0 or 1")

    strict_parse_success_count = int(sum(strict_success_mask))
    normalized_parse_success_count = int(sum(normalized_success_mask))
    strict_schema_parse_rate = strict_parse_success_count / n_total
    normalized_parse_rate = normalized_parse_success_count / n_total
    strict_parse_failure_count = n_total - strict_parse_success_count
    strict_parse_failure_rate = strict_parse_failure_count / n_total

    is_single_class = (n_positive == 0) or (n_negative == 0)
    brier_score = float(np.mean((probs_np - labels_np.astype(np.float64)) ** 2))
    if is_single_class:
        auroc: float | None = None
        auprc: float | None = None
    else:
        auroc = float(sklearn.metrics.roc_auc_score(labels_np, probs_np))
        auprc = float(sklearn.metrics.average_precision_score(labels_np, probs_np))

    return GenOnlyEvalReport(
        dataset_name=inputs.dataset_name,
        population_name=inputs.population_name,
        graph_regime=inputs.graph_regime,
        run_id=inputs.run_id,
        report_split=inputs.population_name,
        eval_type="gen_only",
        gen_eval_schema_version=inputs.gen_eval_schema_version,
        teacher_conditioned=True,
        prompt_mode="eval_gen",
        headline_metric_name="strict_schema_parse_rate",
        headline_metric_value=strict_schema_parse_rate,
        strict_schema_parse_rate=strict_schema_parse_rate,
        normalized_parse_rate=normalized_parse_rate,
        strict_parse_failure_policy="penalty_mode",
        strict_parse_penalty_strategy="ground_truth_opposite_extreme_probability",
        strict_parse_failure_count=strict_parse_failure_count,
        strict_parse_failure_rate=strict_parse_failure_rate,
        strict_parse_failure_node_ids=tuple(strict_failure_node_ids),
        normalized_parse_success_count=normalized_parse_success_count,
        n_total=n_total,
        n_positive=n_positive,
        n_negative=n_negative,
        is_single_class_population=is_single_class,
        auroc=auroc,
        auprc=auprc,
        brier_score=brier_score,
        probs=tuple(float(value) for value in probs_np.tolist()),
        labels=tuple(int(value) for value in labels_np.tolist()),
        node_ids=tuple(int(value) for value in node_ids_np.tolist()),
        strict_parse_succeeded=tuple(strict_success_mask),
        normalized_parse_succeeded=tuple(normalized_success_mask),
        **formal_leakage_provenance_fields(
            prompt_audit_path=inputs.prompt_audit_path,
            prompt_audit_hash=inputs.prompt_audit_hash,
        ),
    )


def _try_parse_strict(text: str) -> StrictOutput | None:
    try:
        return parse_strict(text)
    except (json.JSONDecodeError, TypeError, ValidationError, ValueError):
        return None


def _try_parse_normalized(text: str) -> StrictOutput | None:
    try:
        return parse_normalized_output(text)
    except (json.JSONDecodeError, TypeError, ValidationError, ValueError, NormalizedParseError):
        return None


def _predicted_positive_probability(output: StrictOutput) -> float:
    return float(output.score)


def _worst_case_positive_probability(ground_truth_label: Literal[0, 1]) -> float:
    return 0.0 if int(ground_truth_label) == 1 else 1.0
