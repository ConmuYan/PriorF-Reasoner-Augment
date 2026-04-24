"""Formal teacher-probability ablation audit.

This module implements the Task 11 contract: given a validated full Evidence
Card and a head-scoring report on both the full and the teacher-prob-ablated
population, produce a strict, immutable audit record that marks the system's
dependency on the teacher probability field.

Scope boundary:

* Schema-preserving ablation only. The Evidence Card schema already
  supports explicit masked fields via ``EvidenceAblationMask`` and the
  prompt builder renders ``"Masked / Not Available"`` sentinels. This
  module never rewrites the card into ``{}``, never deletes the card, and
  never changes the prompt distribution beyond the single masked field.

* Ablation target is restricted to
  ``EvidenceAblationMask.TEACHER_SUMMARY_TEACHER_PROB``. Other ablations
  go through ``evidence.evidence_schema.build_evidence_card`` with
  explicit masks; they are out of scope here.

* Audit never selects threshold / alpha, never writes artifacts, and
  never touches the canonical scorer. It consumes two pre-computed
  ``ScorerReport`` instances and produces a frozen audit.

* Headline metric is AUROC. AUPRC / Brier deltas are reported but are
  diagnostic signals, not the dependency trigger.

The "headline dependency flag" is ``teacher_prob_dependency_high``. A
population is marked ``True`` when removing the teacher probability field
degrades head AUROC by more than a caller-specified threshold. Any fail
condition (single-class population, provenance mismatch, schema drift,
non-schema-preserving ablation) raises; no silent fallback.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from eval.head_scoring import CheckpointProvenance, ScorerReport
from evidence.evidence_schema import EvidenceAblationMask, EvidenceCard
from priorf_teacher.schema import DatasetName, GraphRegime, PopulationName

__all__ = (
    "TeacherProbAblationAudit",
    "ablate_teacher_prob",
    "run_teacher_prob_ablation_audit",
)

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)

# The only ablation target this module produces or audits.
TEACHER_PROB_MASK = EvidenceAblationMask.TEACHER_SUMMARY_TEACHER_PROB


def ablate_teacher_prob(card: EvidenceCard) -> EvidenceCard:
    """Return a schema-preserving copy of ``card`` with teacher_prob masked.

    Fail-closed:

    * rejects a card that already carries the teacher_prob mask,
    * rejects a card whose teacher_summary.teacher_prob is already None,
    * otherwise produces a new ``EvidenceCard`` with
      ``teacher_summary.teacher_prob = None`` and
      ``ablation_mask = card.ablation_mask | {TEACHER_SUMMARY_TEACHER_PROB}``.

    The prompt builder will render the masked field as the canonical
    ``"Masked / Not Available"`` sentinel; all other fields remain byte
    identical. Only the ``teacher_summary.teacher_prob`` line changes,
    preserving prompt distribution under the single-field perturbation.
    """
    if TEACHER_PROB_MASK in card.ablation_mask:
        raise ValueError(
            "ablate_teacher_prob: card already has TEACHER_SUMMARY_TEACHER_PROB masked"
        )
    if card.teacher_summary.teacher_prob is None:
        raise ValueError(
            "ablate_teacher_prob: teacher_summary.teacher_prob is already None; "
            "refusing to produce a card whose pre-ablation value is unknown"
        )

    new_teacher_summary = card.teacher_summary.model_copy(
        update={"teacher_prob": None}
    )
    return card.model_copy(
        update={
            "teacher_summary": new_teacher_summary,
            "ablation_mask": frozenset(card.ablation_mask) | {TEACHER_PROB_MASK},
        }
    )


class TeacherProbAblationAudit(BaseModel):
    """Strict, immutable teacher-prob ablation audit record.

    Field set is pinned: provenance + population + counts + AUROC/AUPRC/Brier
    before and after the single-field ablation + deltas + headline flag +
    path audit strings echoed from the source ``ScorerReport``s. No
    thresholds. No alphas. No probabilities replayed. Downstream consumers
    must not add fields via duck-typing; ``extra="forbid"``.
    """

    model_config = _STRICT_MODEL_CONFIG

    dataset_name: DatasetName
    population_name: PopulationName
    graph_regime: GraphRegime
    provenance: CheckpointProvenance

    ablation_target: Literal["teacher_summary.teacher_prob"] = (
        "teacher_summary.teacher_prob"
    )

    n_total: int = Field(ge=1)
    n_positive: int = Field(ge=0)
    n_negative: int = Field(ge=0)

    auroc_full: float | None
    auroc_ablated: float | None
    auprc_full: float | None
    auprc_ablated: float | None
    brier_full: float | None
    brier_ablated: float | None

    auroc_delta: float | None
    auprc_delta: float | None
    brier_delta: float | None

    dependency_threshold: float = Field(gt=0.0, lt=1.0)
    teacher_prob_dependency_high: StrictBool

    prompt_mode: str
    thinking_mode: str
    pooling_path: str
    uses_inference_mode: bool
    distributed_gather: str

    @model_validator(mode="after")
    def _counts_must_be_consistent(self) -> "TeacherProbAblationAudit":
        if self.n_positive + self.n_negative != self.n_total:
            raise ValueError("n_positive + n_negative must equal n_total")
        return self

    @model_validator(mode="after")
    def _delta_flag_must_match_auroc_delta(self) -> "TeacherProbAblationAudit":
        if self.auroc_delta is None:
            if self.teacher_prob_dependency_high is not False:
                raise ValueError(
                    "teacher_prob_dependency_high must be False when auroc_delta is None"
                )
            return self
        expected = self.auroc_delta >= self.dependency_threshold
        if bool(self.teacher_prob_dependency_high) != expected:
            raise ValueError(
                "teacher_prob_dependency_high must equal (auroc_delta >= dependency_threshold)"
            )
        return self


def run_teacher_prob_ablation_audit(
    *,
    full_report: ScorerReport,
    ablated_report: ScorerReport,
    dependency_threshold: float,
) -> TeacherProbAblationAudit:
    """Compare two pre-computed head reports on full vs teacher-prob-ablated.

    Fail-closed verifications (all raise ``ValueError`` on violation):

    * both reports reference the same ``CheckpointProvenance``
      (run_id / checkpoint_step / commit / config_fingerprint /
      data_manifest_hash / graph_regime / dataset_name),
    * both reports describe the same population (``population_name``,
      ``n_total``, ``n_positive``, ``n_negative``),
    * both reports used the same canonical path (``prompt_mode``,
      ``thinking_mode``, ``pooling_path``, ``uses_inference_mode``,
      ``distributed_gather``) -- drift here would invalidate the audit,
    * ``dependency_threshold`` is in the open interval (0, 1),
    * if the population is single-class, both AUROC values are required
      to be ``None``, the audit is emitted with all three delta fields
      ``None`` and the dependency flag forced to ``False``; no attempt
      is made to synthesize a metric.

    The audit itself is a pure computation: it never triggers inference,
    never re-selects a threshold, and never mutates either report.
    """
    if not isinstance(full_report, ScorerReport):
        raise TypeError("full_report must be a ScorerReport")
    if not isinstance(ablated_report, ScorerReport):
        raise TypeError("ablated_report must be a ScorerReport")
    if not (0.0 < dependency_threshold < 1.0):
        raise ValueError(
            f"dependency_threshold must be in (0, 1); got {dependency_threshold!r}"
        )

    if full_report.checkpoint_provenance != ablated_report.checkpoint_provenance:
        raise ValueError(
            "checkpoint_provenance must match between full and ablated reports; "
            "teacher-prob ablation audit requires a single checkpoint context"
        )
    if full_report.dataset_name != ablated_report.dataset_name:
        raise ValueError("dataset_name mismatch between full and ablated reports")
    if full_report.graph_regime != ablated_report.graph_regime:
        raise ValueError("graph_regime mismatch between full and ablated reports")
    if full_report.population_name != ablated_report.population_name:
        raise ValueError("population_name mismatch between full and ablated reports")
    if full_report.n_total != ablated_report.n_total:
        raise ValueError("n_total mismatch between full and ablated reports")
    if full_report.n_positive != ablated_report.n_positive:
        raise ValueError("n_positive mismatch between full and ablated reports")
    if full_report.n_negative != ablated_report.n_negative:
        raise ValueError("n_negative mismatch between full and ablated reports")
    if full_report.is_single_class_population != ablated_report.is_single_class_population:
        raise ValueError(
            "is_single_class_population mismatch between full and ablated reports"
        )

    for attr in (
        "prompt_mode",
        "thinking_mode",
        "pooling_path",
        "uses_inference_mode",
        "distributed_gather",
    ):
        full_value = getattr(full_report, attr)
        ablated_value = getattr(ablated_report, attr)
        if full_value != ablated_value:
            raise ValueError(
                f"path audit field {attr} drifted between reports: "
                f"full={full_value!r} ablated={ablated_value!r}; "
                "teacher-prob ablation must reuse the canonical path"
            )

    single_class = full_report.is_single_class_population

    def _delta(full_value: float | None, ablated_value: float | None) -> float | None:
        if full_value is None or ablated_value is None:
            return None
        return float(full_value) - float(ablated_value)

    auroc_delta = _delta(full_report.auroc, ablated_report.auroc)
    auprc_delta = _delta(full_report.auprc, ablated_report.auprc)
    brier_delta = _delta(full_report.brier_score, ablated_report.brier_score)

    if single_class:
        teacher_prob_dependency_high = False
    else:
        if auroc_delta is None:
            raise ValueError(
                "non-single-class population with None AUROC; cannot audit teacher-prob dependency"
            )
        teacher_prob_dependency_high = bool(auroc_delta >= dependency_threshold)

    return TeacherProbAblationAudit(
        dataset_name=full_report.dataset_name,
        population_name=full_report.population_name,
        graph_regime=full_report.graph_regime,
        provenance=full_report.checkpoint_provenance,
        n_total=full_report.n_total,
        n_positive=full_report.n_positive,
        n_negative=full_report.n_negative,
        auroc_full=full_report.auroc,
        auroc_ablated=ablated_report.auroc,
        auprc_full=full_report.auprc,
        auprc_ablated=ablated_report.auprc,
        brier_full=full_report.brier_score,
        brier_ablated=ablated_report.brier_score,
        auroc_delta=auroc_delta,
        auprc_delta=auprc_delta,
        brier_delta=brier_delta,
        dependency_threshold=float(dependency_threshold),
        teacher_prob_dependency_high=teacher_prob_dependency_high,
        prompt_mode=full_report.prompt_mode,
        thinking_mode=full_report.thinking_mode,
        pooling_path=full_report.pooling_path,
        uses_inference_mode=full_report.uses_inference_mode,
        distributed_gather=full_report.distributed_gather,
    )
