"""Formal head-only evaluation with validation-only threshold freezing.

This module is the formal runner layered on top of the shared
``eval.head_scoring.score_head`` probability path.  It does not reimplement
prompting, tokenization, pooling, or predict-proba logic; instead it:

1. verifies checkpoint provenance across backbone / adapter / cls-head,
2. runs the unified scorer on validation and report populations,
3. selects a threshold on validation only,
4. applies that frozen threshold to a distinct report population,
5. keeps oracle same-population threshold metrics diagnostic-only.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from eval.calibration import (
    CalibrationSummary,
    FrozenThresholdReport,
    ThresholdMetrics,
    compute_calibration_summary,
    compute_threshold_metrics,
    select_validation_threshold,
)
from eval.head_scoring import CheckpointProvenance, HeadScoringInputs, ScorerReport, score_head
from eval.temperature_scaling import apply_temperature_to_probs, fit_temperature_on_validation, logits_from_probs
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
from graph_data.manifests import PopulationMetadata
from priorf_teacher.schema import DatasetName, GraphRegime, PopulationName

__all__ = (
    "FormalHeadOnlyCheckpointBundle",
    "FormalHeadOnlyCheckpointComponent",
    "FormalHeadOnlyDiagnostics",
    "FormalHeadOnlyHeadlineMetrics",
    "FormalHeadOnlyReport",
    "run_formal_head_only_eval",
)

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_FORMAL_HEAD_ONLY_SCHEMA_VERSION = "formal_head_only_eval/v1"


class FormalHeadOnlyCheckpointComponent(BaseModel):
    """One checkpoint-bearing component required by formal head-only eval."""

    model_config = _STRICT_MODEL_CONFIG

    path: StrictStr = Field(min_length=1)
    content_hash: StrictStr = Field(min_length=1)
    run_id: StrictStr = Field(min_length=1)
    checkpoint_step: StrictInt = Field(ge=0)
    commit: StrictStr = Field(min_length=1)
    config_fingerprint: StrictStr = Field(min_length=1)
    data_manifest_hash: StrictStr = Field(min_length=1)
    graph_regime: GraphRegime

    def to_shared_checkpoint_provenance(self) -> CheckpointProvenance:
        return CheckpointProvenance(
            path=self.path,
            step=int(self.checkpoint_step),
            content_hash=self.content_hash,
        )


class FormalHeadOnlyCheckpointBundle(BaseModel):
    """Backbone / adapter / cls-head provenance that must match exactly."""

    model_config = _STRICT_MODEL_CONFIG

    llm_backbone: FormalHeadOnlyCheckpointComponent
    peft_adapter: FormalHeadOnlyCheckpointComponent
    cls_head: FormalHeadOnlyCheckpointComponent

    @model_validator(mode="after")
    def _components_must_share_identity(self) -> FormalHeadOnlyCheckpointBundle:
        reference = self.llm_backbone
        for field_name in ("peft_adapter", "cls_head"):
            component = getattr(self, field_name)
            for shared_field in (
                "run_id",
                "checkpoint_step",
                "commit",
                "config_fingerprint",
                "data_manifest_hash",
                "graph_regime",
            ):
                if getattr(component, shared_field) != getattr(reference, shared_field):
                    raise ValueError(
                        f"{field_name}.{shared_field} must match llm_backbone.{shared_field}"
                    )
        return self


class FormalHeadOnlyHeadlineMetrics(BaseModel):
    """Headline metrics allowed in the formal report."""

    model_config = _STRICT_MODEL_CONFIG

    auroc: float
    auprc: float
    f1_at_val_threshold: float = Field(ge=0.0, le=1.0)
    precision_at_val_threshold: float = Field(ge=0.0, le=1.0)
    recall_at_val_threshold: float = Field(ge=0.0, le=1.0)
    specificity_at_val_threshold: float = Field(ge=0.0, le=1.0)
    prediction_std: float = Field(ge=0.0)


class FormalHeadOnlyDiagnostics(BaseModel):
    """Diagnostics kept out of headline metrics for formal reporting."""

    model_config = _STRICT_MODEL_CONFIG

    diagnostic_only: Literal[True] = True
    formal_excluded: Literal[True] = True
    oracle_same_population_threshold: FrozenThresholdReport | None = None
    oracle_same_population_metrics: ThresholdMetrics | None = None

    @model_validator(mode="after")
    def _oracle_fields_move_together(self) -> FormalHeadOnlyDiagnostics:
        if (self.oracle_same_population_threshold is None) != (
            self.oracle_same_population_metrics is None
        ):
            raise ValueError(
                "oracle_same_population_threshold and oracle_same_population_metrics must both be set or both be None"
            )
        return self


class FormalHeadOnlyReport(BaseModel):
    """Strict report for formal head-only evaluation."""

    model_config = _STRICT_MODEL_CONFIG

    schema_version: Literal["formal_head_only_eval/v1"] = _FORMAL_HEAD_ONLY_SCHEMA_VERSION
    dataset_name: DatasetName
    graph_regime: GraphRegime
    run_id: StrictStr = Field(min_length=1)
    validation_split: Literal[PopulationName.VALIDATION]
    report_split: PopulationName
    eval_type: Literal["head_only"]
    checkpoint_source: Literal["best_checkpoint", "final_checkpoint"]
    thinking_mode: ThinkingMode
    population_metadata: PopulationMetadata
    validation_population_metadata: PopulationMetadata
    checkpoint_bundle: FormalHeadOnlyCheckpointBundle
    scorer_checkpoint_provenance: CheckpointProvenance
    validation_threshold: FrozenThresholdReport
    temperature: float | None = Field(default=None, ge=0.1, le=10.0)
    calibration: CalibrationSummary
    headline_metrics: FormalHeadOnlyHeadlineMetrics
    diagnostics: FormalHeadOnlyDiagnostics | None = None
    threshold_source: Literal["validation"]
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
    def _formal_leakage_fields_consistent(self) -> FormalHeadOnlyReport:
        validate_formal_leakage_payload(self.model_dump(mode="python"), context="FormalHeadOnlyReport")
        return self

    @model_validator(mode="after")
    def _report_consistent(self) -> FormalHeadOnlyReport:
        if self.report_split != PopulationName(self.population_metadata.population_name):
            raise ValueError("report_split must match population_metadata.population_name")
        if self.validation_split != PopulationName.VALIDATION:
            raise ValueError("validation_split must be validation")
        if self.population_metadata.population_name == PopulationName.VALIDATION.value:
            raise ValueError("formal report population must be test-like, not validation")
        if self.population_metadata.contains_tuning_rows:
            raise ValueError("formal report population must not contain tuning rows")
        if self.validation_population_metadata.population_name != PopulationName.VALIDATION.value:
            raise ValueError("validation_population_metadata must be named validation")
        if not self.validation_population_metadata.contains_tuning_rows:
            raise ValueError("validation_population_metadata must contain tuning rows")
        if self.graph_regime != self.checkpoint_bundle.cls_head.graph_regime:
            raise ValueError("report graph_regime must match checkpoint provenance graph_regime")
        if self.population_metadata.population_name != self.calibration.population_name.value:
            raise ValueError("calibration population_name must match population_metadata.population_name")
        if self.validation_threshold.source_population_name != PopulationName.VALIDATION:
            raise ValueError("validation_threshold must be sourced from validation")
        if self.scorer_checkpoint_provenance != self.checkpoint_bundle.cls_head.to_shared_checkpoint_provenance():
            raise ValueError("scorer_checkpoint_provenance must match cls_head checkpoint provenance")
        return self


def run_formal_head_only_eval(
    *,
    validation_inputs: HeadScoringInputs,
    report_inputs: HeadScoringInputs,
    validation_population_metadata: PopulationMetadata,
    report_population_metadata: PopulationMetadata,
    model: Any,
    cls_head: Any,
    tokenizer: Any,
    thinking_mode: ThinkingMode,
    checkpoint_source: Literal["best_checkpoint", "final_checkpoint"],
    checkpoint_bundle: FormalHeadOnlyCheckpointBundle,
    run_id: str,
    prompt_audit_path: str,
    prompt_audit_hash: str,
    threshold_selection_metric: Literal["f1"],
    include_oracle_diagnostics: bool,
    calibration_bins: int = 10,
    apply_temperature_scaling: bool = False,
    accelerator: object | None = None,
) -> FormalHeadOnlyReport:
    """Run formal head-only evaluation through the shared scorer contract."""

    _validate_population_contracts(
        validation_inputs=validation_inputs,
        report_inputs=report_inputs,
        validation_population_metadata=validation_population_metadata,
        report_population_metadata=report_population_metadata,
        checkpoint_bundle=checkpoint_bundle,
    )
    _validate_shared_scorer_provenance(validation_inputs, checkpoint_bundle.cls_head)
    _validate_shared_scorer_provenance(report_inputs, checkpoint_bundle.cls_head)

    validation_report = score_head(
        inputs=validation_inputs,
        model=model,
        cls_head=cls_head,
        tokenizer=tokenizer,
        thinking_mode=thinking_mode,
        prompt_audit_path=prompt_audit_path,
        prompt_audit_hash=prompt_audit_hash,
        accelerator=accelerator,
    )
    report_population = score_head(
        inputs=report_inputs,
        model=model,
        cls_head=cls_head,
        tokenizer=tokenizer,
        thinking_mode=thinking_mode,
        prompt_audit_path=prompt_audit_path,
        prompt_audit_hash=prompt_audit_hash,
        accelerator=accelerator,
    )

    _require_computable_population(
        validation_report,
        population_role="validation threshold selection",
    )
    _require_computable_population(
        report_population,
        population_role="formal head-only reporting",
    )

    validation_probs = tuple(float(value) for value in validation_report.probs)
    report_probs = tuple(float(value) for value in report_population.probs)
    temperature: float | None = None
    if apply_temperature_scaling:
        temperature = fit_temperature_on_validation(
            logits=logits_from_probs(probs=validation_report.probs),
            labels=validation_report.labels,
        )
        validation_probs = apply_temperature_to_probs(
            probs=validation_report.probs,
            temperature=temperature,
        )
        report_probs = apply_temperature_to_probs(
            probs=report_population.probs,
            temperature=temperature,
        )

    validation_threshold = select_validation_threshold(
        probs=validation_probs,
        labels=validation_report.labels,
        source_population_name=PopulationName(validation_report.population_name),
        selection_metric=threshold_selection_metric,
    )
    frozen_threshold_metrics = compute_threshold_metrics(
        probs=report_probs,
        labels=report_population.labels,
        threshold=validation_threshold.selected_threshold,
    )
    calibration = compute_calibration_summary(
        probs=report_probs,
        labels=report_population.labels,
        population_name=PopulationName(report_population.population_name),
        num_bins=calibration_bins,
    )

    diagnostics: FormalHeadOnlyDiagnostics | None = None
    if include_oracle_diagnostics:
        oracle_threshold = select_validation_threshold(
            probs=report_probs,
            labels=report_population.labels,
            source_population_name=PopulationName.VALIDATION,
            selection_metric=threshold_selection_metric,
        )
        diagnostics = FormalHeadOnlyDiagnostics(
            oracle_same_population_threshold=FrozenThresholdReport(
                source_population_name=PopulationName(report_population.population_name),
                selection_metric=oracle_threshold.selection_metric,
                selected_threshold=oracle_threshold.selected_threshold,
                metrics_at_selected_threshold=oracle_threshold.metrics_at_selected_threshold,
            ),
            oracle_same_population_metrics=oracle_threshold.metrics_at_selected_threshold,
        )

    assert report_population.auroc is not None
    assert report_population.auprc is not None
    return FormalHeadOnlyReport(
        dataset_name=report_inputs.dataset_name,
        graph_regime=report_inputs.graph_regime,
        run_id=run_id,
        validation_split=PopulationName.VALIDATION,
        report_split=report_inputs.population_name,
        eval_type="head_only",
        checkpoint_source=checkpoint_source,
        thinking_mode=thinking_mode,
        population_metadata=report_population_metadata,
        validation_population_metadata=validation_population_metadata,
        checkpoint_bundle=checkpoint_bundle,
        scorer_checkpoint_provenance=report_population.checkpoint_provenance,
        validation_threshold=validation_threshold,
        temperature=temperature,
        threshold_source="validation",
        calibration=calibration,
        headline_metrics=FormalHeadOnlyHeadlineMetrics(
            auroc=float(report_population.auroc),
            auprc=float(report_population.auprc),
            f1_at_val_threshold=float(frozen_threshold_metrics.f1),
            precision_at_val_threshold=float(frozen_threshold_metrics.precision),
            recall_at_val_threshold=float(frozen_threshold_metrics.recall),
            specificity_at_val_threshold=float(frozen_threshold_metrics.specificity),
            prediction_std=float(np.std(np.asarray(report_probs, dtype=np.float64))),
        ),
        diagnostics=diagnostics,
        **formal_leakage_provenance_fields(
            prompt_audit_path=prompt_audit_path,
            prompt_audit_hash=prompt_audit_hash,
        ),
    )


def _validate_population_contracts(
    *,
    validation_inputs: HeadScoringInputs,
    report_inputs: HeadScoringInputs,
    validation_population_metadata: PopulationMetadata,
    report_population_metadata: PopulationMetadata,
    checkpoint_bundle: FormalHeadOnlyCheckpointBundle,
) -> None:
    if validation_inputs.population_name != PopulationName.VALIDATION:
        raise ValueError("validation_inputs.population_name must be validation")
    if validation_population_metadata.population_name != PopulationName.VALIDATION.value:
        raise ValueError("validation_population_metadata.population_name must be validation")
    if not validation_population_metadata.contains_tuning_rows:
        raise ValueError("validation population must contain tuning rows for threshold selection")
    if report_population_metadata.contains_tuning_rows:
        raise ValueError("formal report population must not contain tuning rows")
    if report_population_metadata.population_name == PopulationName.VALIDATION.value:
        raise ValueError("formal report population must be distinct from validation")
    if report_inputs.population_name.value != report_population_metadata.population_name:
        raise ValueError("report_inputs.population_name must match report_population_metadata.population_name")
    if validation_inputs.population_name.value != validation_population_metadata.population_name:
        raise ValueError(
            "validation_inputs.population_name must match validation_population_metadata.population_name"
        )
    if validation_inputs.dataset_name != report_inputs.dataset_name:
        raise ValueError("validation_inputs and report_inputs must share dataset_name")
    if validation_inputs.graph_regime != report_inputs.graph_regime:
        raise ValueError("validation_inputs and report_inputs must share graph_regime")
    if validation_inputs.graph_regime != checkpoint_bundle.cls_head.graph_regime:
        raise ValueError("checkpoint graph_regime must match eval inputs graph_regime")


def _validate_shared_scorer_provenance(
    inputs: HeadScoringInputs,
    cls_head_checkpoint: FormalHeadOnlyCheckpointComponent,
) -> None:
    expected = cls_head_checkpoint.to_shared_checkpoint_provenance()
    if inputs.checkpoint_provenance != expected:
        raise ValueError(
            "HeadScoringInputs.checkpoint_provenance must match the fail-closed cls_head checkpoint provenance"
        )


def _require_computable_population(report: ScorerReport, *, population_role: str) -> None:
    if report.is_single_class_population:
        raise ValueError(f"{population_role} requires both classes; single-class population is not formal")
    if report.auroc is None or report.auprc is None:
        raise ValueError(f"{population_role} requires computable AUROC and AUPRC")
