"""Canonical unified head-scoring entrypoint for PriorF-Reasoner.

This module implements Task 5: the single ``predict_proba`` + metrics pipeline
shared by epoch-end validation (Task 9+) and offline head-only evaluation
(Task 13+).  The scorer has a deliberately narrow responsibility surface:

* it accepts a validated ``HeadScoringInputs`` bundle,
* builds ``PromptMode.EVAL_HEAD`` prompts through the one canonical prompt
  builder (Task 3),
* serializes them through ``tokenizer.apply_chat_template`` with fixed kwargs,
* runs a ``torch.inference_mode`` forward (no generation, no KV cache),
* pools the last valid prompt token through the single canonical pooling
  entrypoint (Task 4),
* applies a classification head (supplied by the caller) whose logits are
  hard-cast to ``torch.float32`` before ``torch.sigmoid``,
* returns a strictly-typed ``ScorerReport`` containing threshold-free metrics
  and full provenance.

The scorer does **not** pick thresholds, alpha, or checkpoints; does **not**
train, persist, or architect the classification head; does **not** perform
fusion, faithfulness, or formal reporting; does **not** read environment
variables, write files, or touch the filesystem or network.  All of those
belong to downstream tasks.

Distributed contract (when ``accelerator`` is not ``None``):

1. The caller is responsible for sharding ``inputs.samples`` per
   ``accelerator.process_index`` / ``accelerator.num_processes`` **before**
   calling ``score_head``; the scorer does **not** re-shard.
2. ``report.n_total / n_positive / n_negative / probs / labels / node_ids``
   are **world-level** aggregates, produced via
   ``accelerator.gather_for_metrics``; on every rank the returned
   ``ScorerReport`` has identical field values.
3. If the caller forgets to pre-shard and passes the full population to every
   rank, ``gather_for_metrics`` will duplicate it ``num_processes`` times and
   inflate ``n_total``; Task 9 ``gate_check`` must verify
   ``n_total == expected population size``; this scorer does not re-validate
   that invariant.
"""

from __future__ import annotations

import math
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import sklearn.metrics
import torch
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

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
from evidence.prompt_builder import PromptMode, ThinkingMode, build_prompt
from llm.hidden_state_pooling import pool_last_valid_token
from priorf_teacher.schema import DatasetName, GraphRegime, PopulationName

__all__ = (
    "score_head",
    "ScorerReport",
    "ClsHead",
    "HeadScoringInputs",
    "HeadScoringSample",
    "CheckpointProvenance",
)

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)

_SCORER_SCHEMA_VERSION = "head_scorer/v1"


@runtime_checkable
class ClsHead(Protocol):
    """Protocol pinning the classification-head surface used by ``score_head``.

    The Protocol is deliberately narrow: ``__call__`` performs the head
    forward on a ``[B, H]`` float tensor and returns a ``[B]`` 1-D logit
    tensor; ``eval`` is declared on the Protocol so ``isinstance`` can
    fail-closed at the interface boundary (bare callables without an
    ``eval`` method do not satisfy the Protocol).
    """

    def __call__(self, hidden_prompt_only: "torch.Tensor") -> "torch.Tensor": ...

    def eval(self) -> "ClsHead": ...


class CheckpointProvenance(BaseModel):
    """Strict, immutable checkpoint provenance carried into every report."""

    model_config = _STRICT_MODEL_CONFIG

    path: StrictStr = Field(min_length=1)
    step: StrictInt = Field(ge=0)
    content_hash: StrictStr = Field(min_length=1)


class HeadScoringSample(BaseModel):
    """One validated sample: an Evidence Card plus its ground-truth label."""

    model_config = _STRICT_MODEL_CONFIG

    evidence_card: EvidenceCard
    ground_truth_label: Literal[0, 1]
    node_id: StrictInt = Field(ge=0)


class HeadScoringInputs(BaseModel):
    """Inputs to :func:`score_head`; frozen and cross-validated."""

    model_config = _STRICT_MODEL_CONFIG

    samples: tuple[HeadScoringSample, ...] = Field(min_length=1)
    dataset_name: DatasetName
    population_name: PopulationName
    graph_regime: GraphRegime
    checkpoint_provenance: CheckpointProvenance
    run_id: StrictStr = Field(default="head_scoring_ad_hoc", min_length=1)
    scorer_schema_version: Literal["head_scorer/v1"] = _SCORER_SCHEMA_VERSION

    @model_validator(mode="after")
    def _samples_consistent_with_header(self) -> "HeadScoringInputs":
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


class ScorerReport(BaseModel):
    """Strict, immutable head-scoring report.

    The field set is exactly this union: provenance + counts + single-class
    marker + threshold-free metrics + probability distribution summary +
    per-sample arrays + path audit + distributed audit.  Any other field
    name (``accuracy``, ``f1``, ``threshold``, ``alpha``, ``optimal_*``,
    ``fusion_*``, ``faithfulness_*``, ``oracle_*``, ``selected_checkpoint``,
    ``checkpoint_policy``) is forbidden by construction and rejected by
    pydantic's ``extra='forbid'``.
    """

    model_config = _STRICT_MODEL_CONFIG

    # Provenance
    dataset_name: DatasetName
    population_name: PopulationName
    graph_regime: GraphRegime
    checkpoint_provenance: CheckpointProvenance
    run_id: StrictStr = Field(min_length=1)
    report_split: PopulationName
    eval_type: Literal["head_scoring"]
    scorer_schema_version: Literal["head_scorer/v1"]

    # Counts
    n_total: int = Field(ge=0)
    n_positive: int = Field(ge=0)
    n_negative: int = Field(ge=0)

    # Single-class marker
    is_single_class_population: bool

    # Threshold-free metrics (None iff single-class)
    auroc: float | None
    auprc: float | None
    brier_score: float | None

    # Probability distribution summary
    prob_mean: float
    prob_std: float
    prob_min: float
    prob_max: float
    prob_q25: float
    prob_q50: float
    prob_q75: float

    # Per-sample arrays, same order
    probs: tuple[float, ...]
    labels: tuple[Literal[0, 1], ...]
    node_ids: tuple[int, ...]

    # Path audit
    prompt_mode: Literal["eval_head"]
    thinking_mode: ThinkingMode
    pooling_path: Literal["pool_last_valid_token"]
    uses_inference_mode: bool

    # Distributed audit
    distributed_gather: Literal["none", "accelerate_gather_for_metrics"]

    # Leakage/projection provenance
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
    def _formal_leakage_fields_consistent(self) -> "ScorerReport":
        validate_formal_leakage_payload(self.model_dump(mode="python"), context="ScorerReport")
        return self

    @model_validator(mode="after")
    def _counts_and_arrays_consistent(self) -> "ScorerReport":
        if self.report_split != self.population_name:
            raise ValueError("report_split must match population_name")
        if self.n_positive + self.n_negative != self.n_total:
            raise ValueError("n_positive + n_negative must equal n_total")
        if len(self.probs) != self.n_total:
            raise ValueError("len(probs) must equal n_total")
        if len(self.labels) != self.n_total:
            raise ValueError("len(labels) must equal n_total")
        if len(self.node_ids) != self.n_total:
            raise ValueError("len(node_ids) must equal n_total")
        for p in self.probs:
            if not math.isfinite(p) or p < 0.0 or p > 1.0:
                raise ValueError("probs elements must be finite and within [0.0, 1.0]")
        return self


def score_head(
    *,
    inputs: HeadScoringInputs,
    model: "torch.nn.Module",
    cls_head: ClsHead,
    tokenizer,
    thinking_mode: ThinkingMode,
    prompt_audit_path: str,
    prompt_audit_hash: str,
    accelerator: "object | None" = None,
) -> ScorerReport:
    """Run the canonical head ``predict_proba`` path and return a ``ScorerReport``.

    This is the one and only function shared by validation (epoch-end) and
    offline head-only evaluation.  All strictly-pinned behaviors:

    * ``PromptMode.EVAL_HEAD`` + the explicit ``thinking_mode`` are used for
      every sample; no other prompt builder is called.
    * ``tokenizer.apply_chat_template`` is invoked with
      ``tokenize=True, return_dict=True, return_tensors="pt",
      add_generation_prompt=False``; ``padding`` is not passed, and
      ``tokenizer.padding_side`` is not touched.
    * Strict ``B=1`` per-sample forward; ``attention_mask`` is all-ones, so
      :func:`llm.hidden_state_pooling.pool_last_valid_token` degenerates to
      the canonical "last prompt token" hidden-state.
    * The forward is run inside ``torch.inference_mode``; no ``model.generate``
      is called; ``use_cache=False``; ``output_hidden_states=True``.
    * ``cls_head`` logits are hard-cast to ``torch.float32`` before
      ``torch.sigmoid`` to prevent silent precision drift when downstream
      metrics are computed via ``scikit-learn``.
    * Threshold-free metrics only: ``auroc`` and ``auprc`` are ``None`` on
      single-class populations; ``brier_score`` is still computed.

    When ``accelerator`` is provided, probabilities / labels / node-ids are
    gathered via ``accelerator.gather_for_metrics`` exactly once each; the
    scorer never invokes ``torch.distributed`` directly.  See the module
    docstring for the caller-side pre-sharding contract.
    """

    model.eval()
    cls_head.eval()

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    local_probs: list[float] = []
    local_labels: list[int] = []
    local_node_ids: list[int] = []

    with torch.inference_mode():
        for sample in inputs.samples:
            bundle = build_prompt(
                evidence_card=sample.evidence_card,
                mode=PromptMode.EVAL_HEAD,
                thinking_mode=thinking_mode,
            )
            messages = [
                {"role": message.role, "content": message.content}
                for message in bundle.messages
            ]
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=False,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            last_hidden = outputs.hidden_states[-1]
            pooled = pool_last_valid_token(last_hidden, attention_mask)

            logit = cls_head(pooled).to(torch.float32)
            if logit.dim() != 1:
                raise ValueError(
                    f"cls_head must return a 1-D logit tensor; got dim={logit.dim()}"
                )
            if int(logit.shape[0]) != 1:
                raise ValueError(
                    f"cls_head must return a length-{input_ids.shape[0]} logit; got {tuple(logit.shape)}"
                )
            if not torch.isfinite(logit).all().item():
                raise ValueError("cls_head produced a non-finite logit (eval NaN)")

            prob = torch.sigmoid(logit)
            prob_value = float(prob.detach().to("cpu", dtype=torch.float32).item())
            local_probs.append(prob_value)
            local_labels.append(int(sample.ground_truth_label))
            local_node_ids.append(int(sample.node_id))

    local_probs_t = torch.tensor(local_probs, dtype=torch.float32)
    local_labels_t = torch.tensor(local_labels, dtype=torch.long)
    local_node_ids_t = torch.tensor(local_node_ids, dtype=torch.long)

    if accelerator is None:
        probs_t = local_probs_t
        labels_t = local_labels_t
        node_ids_t = local_node_ids_t
        distributed_gather: Literal["none", "accelerate_gather_for_metrics"] = "none"
    else:
        target_device = getattr(accelerator, "device", torch.device("cpu"))
        probs_t = accelerator.gather_for_metrics(local_probs_t.to(target_device))
        labels_t = accelerator.gather_for_metrics(local_labels_t.to(target_device))
        node_ids_t = accelerator.gather_for_metrics(local_node_ids_t.to(target_device))
        distributed_gather = "accelerate_gather_for_metrics"

    probs_np = probs_t.detach().to(device="cpu", dtype=torch.float32).numpy()
    labels_np = labels_t.detach().to(device="cpu", dtype=torch.long).numpy().astype(np.int64)
    node_ids_np = node_ids_t.detach().to(device="cpu", dtype=torch.long).numpy().astype(np.int64)

    n_total = int(probs_np.shape[0])
    if n_total == 0:
        raise ValueError("n_total must be > 0; population is empty after aggregation")

    if not np.all(np.isfinite(probs_np)):
        raise ValueError("probs contain non-finite values (eval NaN)")
    if np.any(probs_np < 0.0) or np.any(probs_np > 1.0):
        raise ValueError("probs must lie in [0.0, 1.0]")

    n_positive = int((labels_np == 1).sum())
    n_negative = int((labels_np == 0).sum())
    if n_positive + n_negative != n_total:
        raise ValueError("labels must be strictly 0 or 1")

    is_single_class = (n_positive == 0) or (n_negative == 0)

    brier_score = float(np.mean((probs_np - labels_np.astype(np.float64)) ** 2))
    if is_single_class:
        auroc: float | None = None
        auprc: float | None = None
    else:
        auroc = float(sklearn.metrics.roc_auc_score(labels_np, probs_np))
        auprc = float(sklearn.metrics.average_precision_score(labels_np, probs_np))

    prob_mean = float(np.mean(probs_np))
    prob_std = float(np.std(probs_np))
    prob_min = float(np.min(probs_np))
    prob_max = float(np.max(probs_np))
    prob_q25 = float(np.quantile(probs_np, 0.25, method="linear"))
    prob_q50 = float(np.quantile(probs_np, 0.50, method="linear"))
    prob_q75 = float(np.quantile(probs_np, 0.75, method="linear"))

    return ScorerReport(
        dataset_name=inputs.dataset_name,
        population_name=inputs.population_name,
        graph_regime=inputs.graph_regime,
        checkpoint_provenance=inputs.checkpoint_provenance,
        run_id=inputs.run_id,
        report_split=inputs.population_name,
        eval_type="head_scoring",
        scorer_schema_version=inputs.scorer_schema_version,
        n_total=n_total,
        n_positive=n_positive,
        n_negative=n_negative,
        is_single_class_population=is_single_class,
        auroc=auroc,
        auprc=auprc,
        brier_score=brier_score,
        prob_mean=prob_mean,
        prob_std=prob_std,
        prob_min=prob_min,
        prob_max=prob_max,
        prob_q25=prob_q25,
        prob_q50=prob_q50,
        prob_q75=prob_q75,
        probs=tuple(float(value) for value in probs_np.tolist()),
        labels=tuple(int(value) for value in labels_np.tolist()),
        node_ids=tuple(int(value) for value in node_ids_np.tolist()),
        prompt_mode="eval_head",
        thinking_mode=thinking_mode,
        pooling_path="pool_last_valid_token",
        uses_inference_mode=True,
        distributed_gather=distributed_gather,
        **formal_leakage_provenance_fields(
            prompt_audit_path=prompt_audit_path,
            prompt_audit_hash=prompt_audit_hash,
        ),
    )
