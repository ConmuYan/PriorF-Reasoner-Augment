"""Task 6 — validation/offline head-eval parity contract.

This test module intentionally does not train or load the 4B model weights.  It
locks the executable boundary that Task 7+ must reuse:

* epoch-end validation and offline head-only eval both call ``score_head``;
* ``score_head`` always uses ``PromptMode.EVAL_HEAD`` through the canonical
  prompt builder;
* chat-template kwargs, token ids, attention masks, absolute hidden-state
  indices, classifier inputs, logits, probabilities, and fixed-threshold parity
  predictions are sample-wise identical.

The final test loads the real local Qwen3 tokenizer from ``/data1/mq/models``
with ``local_files_only=True`` to prove that the exact four-kwarg
``apply_chat_template`` path accepted by ``eval.head_scoring`` works outside the
dummy tokenizer used by Task 5.
"""

from __future__ import annotations

import re
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import pytest
import torch

import eval.head_scoring as head_scoring
from eval.head_scoring import CheckpointProvenance, HeadScoringInputs, HeadScoringSample, score_head
from evidence.evidence_schema import DataManifestLike, build_evidence_card
from evidence.prompt_builder import PromptMode, ThinkingMode, build_prompt
from graph_data.manifests import DataArtifact, DataManifest, PopulationMetadata
from llm.hidden_state_pooling import pool_last_valid_token
from priorf_teacher.schema import (
    DatasetName,
    GraphRegime,
    NeighborSummary,
    PopulationName,
    RelationProfile,
    TeacherExportRecord,
)


QWEN3_LOCAL_PATH = Path("/data1/mq/models/Qwen3-4B-Instruct-2507")
HIDDEN_DIM = 5


def _score_head_audit_kwargs() -> dict[str, str]:
    return {
        "prompt_audit_path": "outputs/tests/prompt_audit.json",
        "prompt_audit_hash": "a" * 64,
    }


@dataclass
class ParityTrace:
    prompt_modes: list[PromptMode] = field(default_factory=list)
    thinking_modes: list[ThinkingMode] = field(default_factory=list)
    prompt_messages: list[tuple[tuple[str, str], ...]] = field(default_factory=list)
    tokenizer_messages: list[tuple[tuple[str, str], ...]] = field(default_factory=list)
    tokenizer_kwargs: list[dict[str, object]] = field(default_factory=list)
    input_ids: list[torch.Tensor] = field(default_factory=list)
    attention_masks: list[torch.Tensor] = field(default_factory=list)
    hidden_last_indices: list[torch.Tensor] = field(default_factory=list)
    classifier_inputs: list[torch.Tensor] = field(default_factory=list)
    logits: list[torch.Tensor] = field(default_factory=list)


class TraceTokenizer:
    """Deterministic tokenizer double that records the chat-template contract."""

    def __init__(self, trace: ParityTrace) -> None:
        self.trace = trace

    def apply_chat_template(self, messages, **kwargs):
        self.trace.tokenizer_messages.append(tuple((msg["role"], msg["content"]) for msg in messages))
        self.trace.tokenizer_kwargs.append(dict(kwargs))

        text = "\n".join(msg["content"] for msg in messages)
        match = re.search(r"node_id:\s*(\d+)", text)
        node_id = int(match.group(1)) if match else 0

        # Variable-length prompt-only sequence, no padding.  The final token
        # encodes node_id, allowing the test to prove last-valid hidden pooling.
        prefix_len = 3 + (node_id % 3)
        ids = torch.tensor([[11 + node_id, *range(20, 20 + prefix_len), 1000 + node_id]], dtype=torch.long)
        mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask}


class TraceModel(torch.nn.Module):
    """Model double whose hidden state makes token position/index auditable."""

    def __init__(self, trace: ParityTrace) -> None:
        super().__init__()
        self.trace = trace
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, *, input_ids, attention_mask, output_hidden_states, use_cache):
        assert output_hidden_states is True
        assert use_cache is False
        self.trace.input_ids.append(input_ids.detach().cpu().clone())
        self.trace.attention_masks.append(attention_mask.detach().cpu().clone())

        batch_size, seq_len = input_ids.shape
        hidden = torch.zeros(batch_size, seq_len, HIDDEN_DIM, dtype=torch.float32, device=input_ids.device)
        positions = torch.arange(seq_len, dtype=torch.float32, device=input_ids.device).view(1, seq_len)
        hidden[:, :, 0] = input_ids.to(torch.float32)
        hidden[:, :, 1] = positions
        hidden[:, :, 2] = input_ids.to(torch.float32) * 0.01 + positions
        hidden[:, :, 3] = 1.0
        hidden[:, :, 4] = self.anchor
        return types.SimpleNamespace(hidden_states=(hidden,))


class TraceClsHead:
    """Classification-head double that records exact pooled inputs and logits."""

    def __init__(self, trace: ParityTrace) -> None:
        self.trace = trace
        self.eval_count = 0

    def eval(self) -> "TraceClsHead":
        self.eval_count += 1
        return self

    def __call__(self, hidden_prompt_only: torch.Tensor) -> torch.Tensor:
        self.trace.classifier_inputs.append(hidden_prompt_only.detach().cpu().clone())
        logits = ((hidden_prompt_only[:, 0] - 1000.0) * 0.5) + (hidden_prompt_only[:, 1] * 0.125) - 0.75
        self.trace.logits.append(logits.detach().cpu().clone())
        return logits


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


def _teacher_record(node_id: int, ground_truth_label: Literal[0, 1]) -> TeacherExportRecord:
    return TeacherExportRecord(
        dataset_name=DatasetName.AMAZON,
        teacher_model_name="LGHGCLNetV2",
        teacher_checkpoint="outputs/gated/teacher/best_model.pt",
        population_name=PopulationName.VALIDATION,
        node_id=node_id,
        ground_truth_label=ground_truth_label,  # explicit ground-truth field; never a generic "label"
        teacher_prob=0.2 + 0.1 * ground_truth_label,
        teacher_logit=-0.5 + node_id,
        hsd=0.1 * (node_id + 1),
        hsd_quantile=0.25,
        asda_switch=(node_id % 2 == 0),
        mlp_logit=-0.25 + node_id,
        gnn_logit=-0.5 + node_id,
        branch_gap=0.25,
        relation_profile=_relation_profile(),
        neighbor_summary=_neighbor_summary(),
        high_hsd_flag=False,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
    )


def _data_manifest() -> DataManifest:
    return DataManifest(
        dataset_name=DatasetName.AMAZON.value,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD.value,
        feature_dim=25,
        relation_count=3,
        num_nodes=128,
        populations=(
            PopulationMetadata(
                population_name=PopulationName.VALIDATION.value,
                split_values=(PopulationName.VALIDATION.value,),
                node_ids_hash="a" * 64,
                contains_tuning_rows=True,
                contains_final_test_rows=False,
            ),
        ),
        artifacts=(DataArtifact(kind="source_mat", path="assets/data/Amazon_canonical.mat", sha256="b" * 64),),
    )


def _inputs() -> HeadScoringInputs:
    manifest = _data_manifest()
    samples = []
    sample_specs: tuple[tuple[int, Literal[0, 1]], ...] = ((3, 0), (4, 1), (8, 0), (9, 1))
    for node_id, ground_truth_label in sample_specs:
        card = build_evidence_card(
            teacher_record=_teacher_record(node_id=node_id, ground_truth_label=ground_truth_label),
            data_manifest=cast(DataManifestLike, manifest),
        )
        samples.append(
            HeadScoringSample(
                evidence_card=card,
                ground_truth_label=ground_truth_label,
                node_id=node_id,
            )
        )
    return HeadScoringInputs(
        samples=tuple(samples),
        dataset_name=DatasetName.AMAZON,
        population_name=PopulationName.VALIDATION,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        checkpoint_provenance=CheckpointProvenance(
            path="outputs/gated/checkpoints/step_12.safetensors",
            step=12,
            content_hash="deadbeef" * 8,
        ),
    )


def _run_score_with_trace(monkeypatch: pytest.MonkeyPatch, inputs: HeadScoringInputs):
    trace = ParityTrace()
    original_build_prompt = head_scoring.build_prompt
    original_pool = head_scoring.pool_last_valid_token

    def build_prompt_spy(**kwargs):
        trace.prompt_modes.append(kwargs["mode"])
        trace.thinking_modes.append(kwargs["thinking_mode"])
        bundle = original_build_prompt(**kwargs)
        trace.prompt_messages.append(tuple((message.role, message.content) for message in bundle.messages))
        return bundle

    def pool_spy(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0).expand_as(attention_mask)
        last_idx = positions.masked_fill(attention_mask != 1, -1).argmax(dim=1)
        trace.hidden_last_indices.append(last_idx.detach().cpu().clone())
        return original_pool(hidden_states, attention_mask)

    monkeypatch.setattr(head_scoring, "build_prompt", build_prompt_spy)
    monkeypatch.setattr(head_scoring, "pool_last_valid_token", pool_spy)

    report = score_head(
        inputs=inputs,
        model=TraceModel(trace),
        cls_head=TraceClsHead(trace),
        tokenizer=TraceTokenizer(trace),
        thinking_mode=ThinkingMode.NON_THINKING,
        **_score_head_audit_kwargs(),
    )
    return report, trace


def test_validation_and_offline_head_eval_are_samplewise_identical(monkeypatch: pytest.MonkeyPatch):
    inputs = _inputs()

    validation_report, validation_trace = _run_score_with_trace(monkeypatch, inputs)
    monkeypatch.undo()
    offline_report, offline_trace = _run_score_with_trace(monkeypatch, inputs)

    # Same top-level report fields and sample-wise outputs.
    assert validation_report.model_dump(mode="json") == offline_report.model_dump(mode="json")
    assert validation_report.prompt_mode == "eval_head"
    assert validation_report.pooling_path == "pool_last_valid_token"
    assert validation_report.uses_inference_mode is True
    assert validation_report.distributed_gather == "none"

    # Same prompt builder path, same chat messages, same tokenizer contract.
    assert validation_trace.prompt_modes == [PromptMode.EVAL_HEAD] * len(inputs.samples)
    assert offline_trace.prompt_modes == [PromptMode.EVAL_HEAD] * len(inputs.samples)
    assert validation_trace.thinking_modes == [ThinkingMode.NON_THINKING] * len(inputs.samples)
    assert validation_trace.prompt_messages == offline_trace.prompt_messages
    assert validation_trace.tokenizer_messages == validation_trace.prompt_messages
    assert offline_trace.tokenizer_messages == offline_trace.prompt_messages
    expected_kwargs = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": False,
    }
    assert validation_trace.tokenizer_kwargs == [expected_kwargs] * len(inputs.samples)
    assert offline_trace.tokenizer_kwargs == [expected_kwargs] * len(inputs.samples)

    # Same token ids, same masks, same absolute last-valid-token indices.
    _assert_tensor_lists_equal(validation_trace.input_ids, offline_trace.input_ids)
    _assert_tensor_lists_equal(validation_trace.attention_masks, offline_trace.attention_masks)
    _assert_tensor_lists_equal(validation_trace.hidden_last_indices, offline_trace.hidden_last_indices)
    for input_ids, attention_mask, last_idx in zip(
        validation_trace.input_ids,
        validation_trace.attention_masks,
        validation_trace.hidden_last_indices,
    ):
        assert torch.equal(attention_mask, torch.ones_like(attention_mask))
        assert last_idx.item() == input_ids.shape[1] - 1

    # Same classifier inputs/logits, hence same probabilities and fixed-threshold
    # parity decisions.  The threshold is only used for equality checking here;
    # it is not selected from data and is not added to ScorerReport.
    _assert_tensor_lists_equal(validation_trace.classifier_inputs, offline_trace.classifier_inputs)
    _assert_tensor_lists_equal(validation_trace.logits, offline_trace.logits)
    expected_probs = tuple(float(torch.sigmoid(logit).item()) for logit in validation_trace.logits)
    assert validation_report.probs == pytest.approx(expected_probs, abs=1e-7)
    assert offline_report.probs == pytest.approx(expected_probs, abs=1e-7)
    assert tuple(prob >= 0.5 for prob in validation_report.probs) == tuple(
        prob >= 0.5 for prob in offline_report.probs
    )


def test_real_local_qwen3_tokenizer_accepts_canonical_eval_head_chat_template():
    if not QWEN3_LOCAL_PATH.is_dir():
        pytest.fail(f"required local Qwen3 tokenizer path is missing: {QWEN3_LOCAL_PATH}")

    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(QWEN3_LOCAL_PATH),
        local_files_only=True,
        trust_remote_code=True,
    )

    sample = _inputs().samples[0]
    bundle = build_prompt(
        evidence_card=sample.evidence_card,
        mode=PromptMode.EVAL_HEAD,
        thinking_mode=ThinkingMode.NON_THINKING,
    )
    messages = [{"role": message.role, "content": message.content} for message in bundle.messages]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=False,
    )

    assert set(encoded) >= {"input_ids", "attention_mask"}
    assert encoded["input_ids"].dim() == 2
    assert encoded["attention_mask"].shape == encoded["input_ids"].shape
    assert encoded["input_ids"].shape[0] == 1
    assert encoded["input_ids"].shape[1] > 0

    seq_len = encoded["input_ids"].shape[1]
    hidden = torch.arange(seq_len * HIDDEN_DIM, dtype=torch.float32).view(1, seq_len, HIDDEN_DIM)
    pooled = pool_last_valid_token(hidden, encoded["attention_mask"])
    positions = torch.arange(seq_len).unsqueeze(0).expand_as(encoded["attention_mask"])
    last_idx = positions.masked_fill(encoded["attention_mask"] != 1, -1).argmax(dim=1)
    assert torch.equal(pooled, hidden.gather(1, last_idx.view(1, 1, 1).expand(-1, 1, HIDDEN_DIM)).squeeze(1))


def _assert_tensor_lists_equal(left: list[torch.Tensor], right: list[torch.Tensor]) -> None:
    assert len(left) == len(right)
    for left_tensor, right_tensor in zip(left, right):
        assert torch.equal(left_tensor, right_tensor)
