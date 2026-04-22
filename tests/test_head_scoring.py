"""Task 5 — `eval.head_scoring` canonical scorer tests.

All 21 contract tests for the unified head-scoring entrypoint.  Tests use
dummy model / tokenizer / cls-head doubles; no real Qwen model, no disk or
network I/O, no environment reads.
"""

from __future__ import annotations

import builtins
import importlib
import inspect
import pathlib
import re
import sys
import types

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from evidence.evidence_schema import build_evidence_card
from evidence.prompt_builder import PromptMode, ThinkingMode
from graph_data.manifests import DataArtifact, DataManifest, PopulationMetadata
from priorf_teacher.schema import (
    DatasetName,
    GraphRegime,
    NeighborSummary,
    PopulationName,
    RelationProfile,
    TeacherExportRecord,
)

import eval.head_scoring as head_scoring
from eval.head_scoring import (
    CheckpointProvenance,
    ClsHead,
    HeadScoringInputs,
    HeadScoringSample,
    ScorerReport,
    score_head,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


_HIDDEN_DIM = 8
_MANIFEST_NODES = 32


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


def _record(
    *,
    node_id: int,
    ground_truth_label: int = 1,
    population: PopulationName = PopulationName.VALIDATION,
    dataset: DatasetName = DatasetName.AMAZON,
    regime: GraphRegime = GraphRegime.TRANSDUCTIVE_STANDARD,
) -> TeacherExportRecord:
    return TeacherExportRecord.model_validate(
        {
            "dataset_name": dataset,
            "teacher_model_name": "PriorF-GNN",
            "teacher_checkpoint": "checkpoints/priorf.pt",
            "population_name": population,
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
            "graph_regime": regime,
        }
    )


def _manifest(
    *,
    dataset: DatasetName = DatasetName.AMAZON,
    regime: GraphRegime = GraphRegime.TRANSDUCTIVE_STANDARD,
    population: PopulationName = PopulationName.VALIDATION,
) -> DataManifest:
    return DataManifest(
        dataset_name=dataset.value,
        graph_regime=regime.value,
        feature_dim=25,
        relation_count=3,
        num_nodes=_MANIFEST_NODES,
        populations=(
            PopulationMetadata(
                population_name=population.value,
                split_values=(population.value,),
                node_ids_hash="a" * 64,
                contains_tuning_rows=(population == PopulationName.VALIDATION),
                contains_final_test_rows=(population == PopulationName.FINAL_TEST),
            ),
        ),
        artifacts=(DataArtifact(kind="source_mat", path="amazon.mat", sha256="0" * 64),),
    )


def _card(
    *,
    node_id: int,
    ground_truth_label: int = 1,
    population: PopulationName = PopulationName.VALIDATION,
    dataset: DatasetName = DatasetName.AMAZON,
    regime: GraphRegime = GraphRegime.TRANSDUCTIVE_STANDARD,
):
    return build_evidence_card(
        teacher_record=_record(
            node_id=node_id,
            ground_truth_label=ground_truth_label,
            population=population,
            dataset=dataset,
            regime=regime,
        ),
        data_manifest=_manifest(dataset=dataset, regime=regime, population=population),
    )


def _sample(node_id: int, label: int, **card_kwargs) -> HeadScoringSample:
    return HeadScoringSample(
        evidence_card=_card(node_id=node_id, ground_truth_label=label, **card_kwargs),
        ground_truth_label=label,
        node_id=node_id,
    )


def _checkpoint_provenance() -> CheckpointProvenance:
    return CheckpointProvenance(
        path="/tmp/ckpt/step_42.safetensors",
        step=42,
        content_hash="deadbeef" * 8,
    )


def _inputs(
    labels: list[int],
    *,
    node_ids: list[int] | None = None,
    population: PopulationName = PopulationName.VALIDATION,
    dataset: DatasetName = DatasetName.AMAZON,
    regime: GraphRegime = GraphRegime.TRANSDUCTIVE_STANDARD,
) -> HeadScoringInputs:
    if node_ids is None:
        node_ids = list(range(len(labels)))
    samples = tuple(
        _sample(node_id=nid, label=lbl, population=population, dataset=dataset, regime=regime)
        for nid, lbl in zip(node_ids, labels)
    )
    return HeadScoringInputs(
        samples=samples,
        dataset_name=dataset,
        population_name=population,
        graph_regime=regime,
        checkpoint_provenance=_checkpoint_provenance(),
    )


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class DummyTokenizer:
    """Deterministic stand-in for a chat-template tokenizer."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": dict(kwargs)})
        text = " ".join(m["content"] for m in messages)
        match = re.search(r"node_id:\s*(\d+)", text)
        nid = int(match.group(1)) if match else 0
        ids = torch.tensor([[11, 22, 33, 100 + nid]], dtype=torch.long)
        mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask}


class DummyModel(torch.nn.Module):
    """Deterministic stand-in: hidden[:, -1, 0] == input_ids[:, -1] - 100 (== node_id)."""

    def __init__(self, hidden_dim: int = _HIDDEN_DIM) -> None:
        super().__init__()
        self._anchor = torch.nn.Parameter(torch.zeros(1))
        self.hidden_dim = hidden_dim
        self.forward_calls: list[dict] = []
        self.inference_mode_observed: list[bool] = []
        self.eval_count = 0

    def eval(self):  # type: ignore[override]
        self.eval_count += 1
        return super().eval()

    def forward(self, *, input_ids, attention_mask, output_hidden_states, use_cache):
        self.forward_calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
                "output_hidden_states": output_hidden_states,
                "use_cache": use_cache,
            }
        )
        self.inference_mode_observed.append(torch.is_inference_mode_enabled())
        batch_size, seq_len = input_ids.shape
        hidden = torch.zeros(batch_size, seq_len, self.hidden_dim, dtype=torch.float32)
        nid_value = input_ids[:, -1].to(torch.float32) - 100.0
        hidden[:, -1, 0] = nid_value
        return types.SimpleNamespace(hidden_states=(hidden,))


class DummyClsHead:
    """Deterministic stand-in with a controllable logit dtype and optional override."""

    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        scale: float = 0.5,
        override=None,
    ) -> None:
        self.dtype = dtype
        self.scale = scale
        self.override = override
        self.call_count = 0
        self.eval_count = 0
        self.last_input_dtype: torch.dtype | None = None

    def __call__(self, hidden_prompt_only: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        self.last_input_dtype = hidden_prompt_only.dtype
        if self.override is not None:
            override = self.override
            if callable(override):
                value = override(hidden_prompt_only)
            else:
                value = override
            if not isinstance(value, torch.Tensor):
                value = torch.tensor([float(value)], dtype=self.dtype)
            return value
        logit = hidden_prompt_only[:, 0].to(torch.float32) * self.scale
        return logit.to(self.dtype)

    def eval(self):
        self.eval_count += 1
        return self


class MockAccelerator:
    """Mock Accelerate interface returning preset world-level tensors."""

    def __init__(
        self,
        *,
        world_probs: torch.Tensor,
        world_labels: torch.Tensor,
        world_node_ids: torch.Tensor,
        process_index: int = 0,
        num_processes: int = 1,
    ) -> None:
        self.device = torch.device("cpu")
        self.process_index = process_index
        self.num_processes = num_processes
        self._world_probs = world_probs
        self._world_labels = world_labels
        self._world_node_ids = world_node_ids
        self.gather_calls: list[torch.Tensor] = []

    def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        self.gather_calls.append(tensor.detach().clone())
        call_idx = len(self.gather_calls)
        if call_idx == 1:
            return self._world_probs.clone()
        if call_idx == 2:
            return self._world_labels.clone()
        if call_idx == 3:
            return self._world_node_ids.clone()
        raise AssertionError("unexpected extra gather_for_metrics call")


# ---------------------------------------------------------------------------
# Common valid ScorerReport payload for schema tests
# ---------------------------------------------------------------------------


def _valid_report_kwargs() -> dict:
    return dict(
        dataset_name=DatasetName.AMAZON,
        population_name=PopulationName.VALIDATION,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        checkpoint_provenance=_checkpoint_provenance(),
        scorer_schema_version="head_scorer/v1",
        n_total=3,
        n_positive=1,
        n_negative=2,
        is_single_class_population=False,
        auroc=0.75,
        auprc=0.6,
        brier_score=0.2,
        prob_mean=0.5,
        prob_std=0.1,
        prob_min=0.4,
        prob_max=0.6,
        prob_q25=0.45,
        prob_q50=0.5,
        prob_q75=0.55,
        probs=(0.4, 0.5, 0.6),
        labels=(0, 0, 1),
        node_ids=(10, 11, 12),
        prompt_mode="eval_head",
        thinking_mode=ThinkingMode.NON_THINKING,
        pooling_path="pool_last_valid_token",
        uses_inference_mode=True,
        distributed_gather="none",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


# Test 1 — Schema field presence & types + extra-key rejection
def test_scorer_report_schema_field_presence_and_extra_forbidden():
    report = ScorerReport(**_valid_report_kwargs())

    expected_fields = {
        "dataset_name",
        "population_name",
        "graph_regime",
        "checkpoint_provenance",
        "scorer_schema_version",
        "n_total",
        "n_positive",
        "n_negative",
        "is_single_class_population",
        "auroc",
        "auprc",
        "brier_score",
        "prob_mean",
        "prob_std",
        "prob_min",
        "prob_max",
        "prob_q25",
        "prob_q50",
        "prob_q75",
        "probs",
        "labels",
        "node_ids",
        "prompt_mode",
        "thinking_mode",
        "pooling_path",
        "uses_inference_mode",
        "distributed_gather",
    }
    assert set(ScorerReport.model_fields.keys()) == expected_fields

    payload = report.model_dump(mode="python")
    payload["extra_key"] = 1
    with pytest.raises(ValidationError):
        ScorerReport.model_validate(payload)


# Test 2 — Forbidden metric names absent and rejected
def test_scorer_report_forbidden_names_absent():
    forbidden = (
        "accuracy",
        "f1",
        "precision",
        "recall",
        "threshold",
        "optimal_threshold",
        "fixed_threshold_ts",
        "fixed_threshold_val",
        "alpha",
        "selected_checkpoint",
        "checkpoint_policy",
        "oracle_threshold",
        "faithfulness_score",
        "fusion_alpha",
    )
    for name in forbidden:
        assert name not in ScorerReport.model_fields

    report = ScorerReport(**_valid_report_kwargs())
    base = report.model_dump(mode="python")
    for name in forbidden:
        payload = {**base, name: 0.5}
        with pytest.raises(ValidationError):
            ScorerReport.model_validate(payload)


# Test 3 — Prompt-builder spy: every call uses EVAL_HEAD + explicit thinking_mode
def test_prompt_builder_called_with_eval_head_and_thinking_mode(monkeypatch):
    calls: list[dict] = []
    original_build_prompt = head_scoring.build_prompt

    def spy(**kwargs):
        calls.append(kwargs)
        return original_build_prompt(**kwargs)

    monkeypatch.setattr(head_scoring, "build_prompt", spy)

    inputs = _inputs([0, 1, 0, 1])
    score_head(
        inputs=inputs,
        model=DummyModel(),
        cls_head=DummyClsHead(),
        tokenizer=DummyTokenizer(),
        thinking_mode=ThinkingMode.NON_THINKING,
    )

    assert len(calls) == len(inputs.samples)
    for call in calls:
        assert call["mode"] is PromptMode.EVAL_HEAD
        assert call["thinking_mode"] is ThinkingMode.NON_THINKING


# Test 4 — Pooling spy: exactly the canonical entrypoint is used
def test_pool_last_valid_token_is_the_only_pooling(monkeypatch):
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    original_pool = head_scoring.pool_last_valid_token

    def spy(hidden_states, attention_mask):
        calls.append((tuple(hidden_states.shape), tuple(attention_mask.shape)))
        return original_pool(hidden_states, attention_mask)

    monkeypatch.setattr(head_scoring, "pool_last_valid_token", spy)

    inputs = _inputs([0, 1, 0])
    score_head(
        inputs=inputs,
        model=DummyModel(),
        cls_head=DummyClsHead(),
        tokenizer=DummyTokenizer(),
        thinking_mode=ThinkingMode.NON_THINKING,
    )

    assert len(calls) == len(inputs.samples) >= 1

    source = inspect.getsource(head_scoring)
    assert "hidden_states[:, -1, :]" not in source
    assert "hidden_states[:,-1,:]" not in source
    assert source.count("def pool") == 0, "scorer must not define any local pooling function"


# Test 5 — model.generate is forbidden
def test_model_generate_is_never_called():
    class NoGenModel(DummyModel):
        def generate(self, *args, **kwargs):  # noqa: D401 - trap
            raise AssertionError("model.generate must not be called by score_head")

    score_head(
        inputs=_inputs([0, 1]),
        model=NoGenModel(),
        cls_head=DummyClsHead(),
        tokenizer=DummyTokenizer(),
        thinking_mode=ThinkingMode.NON_THINKING,
    )


# Test 6 — .eval() is called + ClsHead Protocol isinstance semantics
def test_eval_called_and_clshead_isinstance_behaviour():
    model = DummyModel()
    cls_head = DummyClsHead()

    score_head(
        inputs=_inputs([0, 1]),
        model=model,
        cls_head=cls_head,
        tokenizer=DummyTokenizer(),
        thinking_mode=ThinkingMode.NON_THINKING,
    )

    assert model.eval_count >= 1
    assert cls_head.eval_count >= 1

    assert isinstance(cls_head, ClsHead)
    assert isinstance(torch.nn.Linear(4, 1), ClsHead)

    bare_lambda = lambda tensor: tensor  # noqa: E731 - pinpoints the Protocol gap
    assert not isinstance(bare_lambda, ClsHead)


# Test 7 — No autograd / inference_mode is in force
def test_inference_mode_is_in_force_and_no_autograd(monkeypatch):
    captured = {"dtypes": [], "requires_grad": []}
    original_sigmoid = torch.sigmoid

    def spy_sigmoid(tensor):
        captured["dtypes"].append(tensor.dtype)
        captured["requires_grad"].append(bool(tensor.requires_grad))
        return original_sigmoid(tensor)

    monkeypatch.setattr(torch, "sigmoid", spy_sigmoid)

    model = DummyModel()
    score_head(
        inputs=_inputs([0, 1, 0]),
        model=model,
        cls_head=DummyClsHead(),
        tokenizer=DummyTokenizer(),
        thinking_mode=ThinkingMode.NON_THINKING,
    )

    assert captured["requires_grad"], "sigmoid must have been called at least once"
    assert all(rg is False for rg in captured["requires_grad"])
    assert model.inference_mode_observed, "model forward must run at least once"
    assert all(flag is True for flag in model.inference_mode_observed)

    source = inspect.getsource(head_scoring)
    assert "torch.inference_mode(" in source
    assert "torch.enable_grad" not in source


# Test 8 — Single-class population
@pytest.mark.parametrize("label", [0, 1])
def test_single_class_population_marks_auroc_auprc_none(label):
    report = score_head(
        inputs=_inputs([label, label, label, label]),
        model=DummyModel(),
        cls_head=DummyClsHead(),
        tokenizer=DummyTokenizer(),
        thinking_mode=ThinkingMode.NON_THINKING,
    )
    assert report.is_single_class_population is True
    assert report.auroc is None
    assert report.auprc is None
    assert report.brier_score is not None
    assert np.isfinite(report.brier_score)
    if label == 0:
        assert report.n_positive == 0
        assert report.n_negative == report.n_total
    else:
        assert report.n_negative == 0
        assert report.n_positive == report.n_total


# Test 9 — Empty population
def test_empty_population_is_blocked():
    with pytest.raises(ValidationError):
        HeadScoringInputs(
            samples=(),
            dataset_name=DatasetName.AMAZON,
            population_name=PopulationName.VALIDATION,
            graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
            checkpoint_provenance=_checkpoint_provenance(),
        )

    empty_probs = torch.tensor([], dtype=torch.float32)
    empty_labels = torch.tensor([], dtype=torch.long)
    empty_node_ids = torch.tensor([], dtype=torch.long)
    accelerator = MockAccelerator(
        world_probs=empty_probs,
        world_labels=empty_labels,
        world_node_ids=empty_node_ids,
    )
    with pytest.raises(ValueError, match="n_total"):
        score_head(
            inputs=_inputs([0, 1]),
            model=DummyModel(),
            cls_head=DummyClsHead(),
            tokenizer=DummyTokenizer(),
            thinking_mode=ThinkingMode.NON_THINKING,
            accelerator=accelerator,
        )


# Test 10 — Permutation equivariance
def test_permutation_equivariance():
    labels = [0, 1, 0, 1]
    node_ids = [3, 5, 7, 9]
    inputs_a = _inputs(labels=labels, node_ids=node_ids)
    perm = [2, 0, 3, 1]
    inputs_b = _inputs(
        labels=[labels[i] for i in perm],
        node_ids=[node_ids[i] for i in perm],
    )

    report_a = score_head(
        inputs=inputs_a,
        model=DummyModel(),
        cls_head=DummyClsHead(),
        tokenizer=DummyTokenizer(),
        thinking_mode=ThinkingMode.NON_THINKING,
    )
    report_b = score_head(
        inputs=inputs_b,
        model=DummyModel(),
        cls_head=DummyClsHead(),
        tokenizer=DummyTokenizer(),
        thinking_mode=ThinkingMode.NON_THINKING,
    )

    triples_a = list(zip(report_a.node_ids, report_a.probs, report_a.labels))
    triples_b = list(zip(report_b.node_ids, report_b.probs, report_b.labels))

    # b's element-wise order must match the applied permutation of a's order.
    expected_b = [triples_a[i] for i in perm]
    assert triples_b == expected_b

    # Set of triples is preserved.
    assert sorted(triples_a) == sorted(triples_b)

    # Threshold-free metrics are invariant under permutation.
    assert report_a.auroc == report_b.auroc
    assert report_a.auprc == report_b.auprc
    assert report_a.brier_score == report_b.brier_score


# Test 11 — Determinism
def test_determinism_between_two_identical_runs():
    inputs = _inputs([0, 1, 0, 1])
    report_1 = score_head(
        inputs=inputs,
        model=DummyModel(),
        cls_head=DummyClsHead(),
        tokenizer=DummyTokenizer(),
        thinking_mode=ThinkingMode.NON_THINKING,
    )
    report_2 = score_head(
        inputs=inputs,
        model=DummyModel(),
        cls_head=DummyClsHead(),
        tokenizer=DummyTokenizer(),
        thinking_mode=ThinkingMode.NON_THINKING,
    )
    assert report_1.probs == report_2.probs
    assert report_1.labels == report_2.labels
    assert report_1.node_ids == report_2.node_ids
    assert report_1.auroc == report_2.auroc
    assert report_1.auprc == report_2.auprc
    assert report_1.brier_score == report_2.brier_score
    assert report_1.prob_mean == report_2.prob_mean
    assert report_1.prob_std == report_2.prob_std


# Test 12 — NaN / Inf logit
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_logit_raises(bad):
    def override(_pooled):
        return torch.tensor([bad], dtype=torch.float32)

    cls_head = DummyClsHead(override=override)
    with pytest.raises(ValueError):
        score_head(
            inputs=_inputs([0, 1]),
            model=DummyModel(),
            cls_head=cls_head,
            tokenizer=DummyTokenizer(),
            thinking_mode=ThinkingMode.NON_THINKING,
        )


# Test 13 — accelerator is None
def test_accelerator_none_reports_none_and_avoids_torch_distributed():
    modules_before = set(sys.modules)
    report = score_head(
        inputs=_inputs([0, 1, 0, 1]),
        model=DummyModel(),
        cls_head=DummyClsHead(),
        tokenizer=DummyTokenizer(),
        thinking_mode=ThinkingMode.NON_THINKING,
    )
    assert report.distributed_gather == "none"

    newly_imported = set(sys.modules) - modules_before
    assert not any(name.startswith("torch.distributed") for name in newly_imported)

    source = inspect.getsource(head_scoring)
    # Forbid real references (attribute access / imports); allow prose in the
    # module docstring (e.g. "does not invoke torch.distributed directly").
    assert "import torch.distributed" not in source
    assert "from torch.distributed" not in source
    assert "torch.distributed." not in source


# Test 14 — accelerator is not None + world-level consistency
def test_accelerator_branch_and_world_level_consistency():
    world_probs = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32)
    world_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    world_node_ids = torch.tensor([10, 11, 12, 13], dtype=torch.long)

    # Rank 0 sees the first half; rank 1 sees the second half.
    inputs_r0 = _inputs(labels=[0, 0], node_ids=[10, 11])
    inputs_r1 = _inputs(labels=[1, 1], node_ids=[12, 13])

    acc_r0 = MockAccelerator(
        world_probs=world_probs,
        world_labels=world_labels,
        world_node_ids=world_node_ids,
        process_index=0,
        num_processes=2,
    )
    acc_r1 = MockAccelerator(
        world_probs=world_probs,
        world_labels=world_labels,
        world_node_ids=world_node_ids,
        process_index=1,
        num_processes=2,
    )

    report_r0 = score_head(
        inputs=inputs_r0,
        model=DummyModel(),
        cls_head=DummyClsHead(),
        tokenizer=DummyTokenizer(),
        thinking_mode=ThinkingMode.NON_THINKING,
        accelerator=acc_r0,
    )
    report_r1 = score_head(
        inputs=inputs_r1,
        model=DummyModel(),
        cls_head=DummyClsHead(),
        tokenizer=DummyTokenizer(),
        thinking_mode=ThinkingMode.NON_THINKING,
        accelerator=acc_r1,
    )

    assert len(acc_r0.gather_calls) == 3
    assert len(acc_r1.gather_calls) == 3

    assert report_r0.distributed_gather == "accelerate_gather_for_metrics"
    assert report_r1.distributed_gather == "accelerate_gather_for_metrics"

    # World-level consistency: both ranks see identical aggregated state.
    assert report_r0.model_dump(mode="python") == report_r1.model_dump(mode="python")

    # Sanity: aggregated arrays reflect the preset world tensors.
    assert report_r0.probs == tuple(float(v) for v in world_probs.tolist())
    assert report_r0.labels == tuple(int(v) for v in world_labels.tolist())
    assert report_r0.node_ids == tuple(int(v) for v in world_node_ids.tolist())
    assert report_r0.n_total == 4
    assert report_r0.n_positive == 2
    assert report_r0.n_negative == 2


# Test 15 — Provenance echo
def test_provenance_echoed_into_report():
    inputs = _inputs([0, 1])
    report = score_head(
        inputs=inputs,
        model=DummyModel(),
        cls_head=DummyClsHead(),
        tokenizer=DummyTokenizer(),
        thinking_mode=ThinkingMode.NON_THINKING,
    )
    assert report.dataset_name == inputs.dataset_name
    assert report.population_name == inputs.population_name
    assert report.graph_regime == inputs.graph_regime
    assert report.checkpoint_provenance == inputs.checkpoint_provenance
    assert report.scorer_schema_version == inputs.scorer_schema_version
    assert report.prompt_mode == "eval_head"
    assert report.pooling_path == "pool_last_valid_token"
    assert report.uses_inference_mode is True
    assert report.thinking_mode == ThinkingMode.NON_THINKING


# Test 16 — Inputs model_validator mismatch
def test_inputs_sample_mismatch_with_header_raises():
    mismatched_card = _card(
        node_id=0,
        ground_truth_label=1,
        population=PopulationName.TRAIN,
    )
    mismatched_sample = HeadScoringSample(
        evidence_card=mismatched_card,
        ground_truth_label=1,
        node_id=0,
    )
    good_sample = _sample(node_id=1, label=0)

    with pytest.raises(ValidationError):
        HeadScoringInputs(
            samples=(good_sample, mismatched_sample),
            dataset_name=DatasetName.AMAZON,
            population_name=PopulationName.VALIDATION,
            graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
            checkpoint_provenance=_checkpoint_provenance(),
        )


# Test 17 — No file writes
def test_scorer_never_writes_files(monkeypatch, tmp_path):
    writes: list[tuple[str, object, object]] = []

    original_open = builtins.open

    def spy_open(file, mode="r", *args, **kwargs):
        if any(ch in mode for ch in "wax+"):
            writes.append(("open", file, mode))
        return original_open(file, mode, *args, **kwargs)

    def spy_write_text(self, *args, **kwargs):
        writes.append(("write_text", self, None))

    def spy_write_bytes(self, *args, **kwargs):
        writes.append(("write_bytes", self, None))

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(pathlib.Path, "write_text", spy_write_text, raising=False)
    monkeypatch.setattr(pathlib.Path, "write_bytes", spy_write_bytes, raising=False)

    outputs_dir = pathlib.Path.cwd() / "outputs"
    preexisting = outputs_dir.exists()

    score_head(
        inputs=_inputs([0, 1]),
        model=DummyModel(),
        cls_head=DummyClsHead(),
        tokenizer=DummyTokenizer(),
        thinking_mode=ThinkingMode.NON_THINKING,
    )

    assert not writes, f"scorer unexpectedly issued file writes: {writes}"
    if not preexisting:
        assert not outputs_dir.exists(), "scorer unexpectedly created outputs/"


# Test 18 — No env dependency
def test_scorer_is_independent_of_env_vars(monkeypatch):
    monkeypatch.setenv("PRIORF_SCORER_X", "1")
    monkeypatch.setenv("HEAD_SCORING_Y", "foo")
    report_with_env = score_head(
        inputs=_inputs([0, 1, 0]),
        model=DummyModel(),
        cls_head=DummyClsHead(),
        tokenizer=DummyTokenizer(),
        thinking_mode=ThinkingMode.NON_THINKING,
    )

    monkeypatch.delenv("PRIORF_SCORER_X", raising=False)
    monkeypatch.delenv("HEAD_SCORING_Y", raising=False)
    report_without_env = score_head(
        inputs=_inputs([0, 1, 0]),
        model=DummyModel(),
        cls_head=DummyClsHead(),
        tokenizer=DummyTokenizer(),
        thinking_mode=ThinkingMode.NON_THINKING,
    )

    assert report_with_env.model_dump(mode="python") == report_without_env.model_dump(mode="python")


# Test 19 — Strict __all__
def test_module_all_is_exactly_pinned():
    assert head_scoring.__all__ == (
        "score_head",
        "ScorerReport",
        "ClsHead",
        "HeadScoringInputs",
        "HeadScoringSample",
        "CheckpointProvenance",
    )


# Test 20 — Logit dtype hard cast to fp32 before sigmoid
@pytest.mark.parametrize("logit_dtype", [torch.bfloat16, torch.float16])
def test_logit_is_hard_cast_to_float32_before_sigmoid(monkeypatch, logit_dtype):
    captured_dtypes: list[torch.dtype] = []
    original_sigmoid = torch.sigmoid

    def spy_sigmoid(tensor):
        captured_dtypes.append(tensor.dtype)
        return original_sigmoid(tensor)

    monkeypatch.setattr(torch, "sigmoid", spy_sigmoid)

    cls_head = DummyClsHead(dtype=logit_dtype)
    report = score_head(
        inputs=_inputs([0, 1, 0, 1]),
        model=DummyModel(),
        cls_head=cls_head,
        tokenizer=DummyTokenizer(),
        thinking_mode=ThinkingMode.NON_THINKING,
    )

    assert captured_dtypes, "sigmoid must have been invoked"
    assert all(dtype == torch.float32 for dtype in captured_dtypes), captured_dtypes
    for value in report.probs:
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0


# Test 21 — B=1 / no-padding hard assertions
def test_forward_is_batch_one_with_all_ones_mask_and_no_padding_kwarg():
    model = DummyModel()
    tokenizer = DummyTokenizer()

    score_head(
        inputs=_inputs([0, 1, 0]),
        model=model,
        cls_head=DummyClsHead(),
        tokenizer=tokenizer,
        thinking_mode=ThinkingMode.NON_THINKING,
    )

    assert model.forward_calls, "model.forward must be invoked"
    for call in model.forward_calls:
        input_ids = call["input_ids"]
        attention_mask = call["attention_mask"]
        assert input_ids.shape[0] == 1
        assert attention_mask.shape == input_ids.shape
        assert attention_mask.dim() == 2
        assert attention_mask.shape[0] == 1
        assert bool(attention_mask.all().item()) is True
        assert call["output_hidden_states"] is True
        assert call["use_cache"] is False

    required_kwargs = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": False,
    }
    for record in tokenizer.calls:
        kwargs = record["kwargs"]
        for key, value in required_kwargs.items():
            assert key in kwargs and kwargs[key] == value, (key, kwargs)
        assert kwargs.get("padding", False) is False
        assert "padding_side" not in kwargs
