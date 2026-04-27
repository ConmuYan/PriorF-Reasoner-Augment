from __future__ import annotations

import os
import subprocess
import sys

import torch

from graph_data.manifests import DataArtifact, DataManifest, PopulationMetadata
from eval.head_scoring import CheckpointProvenance
from priorf_teacher.schema import DatasetName, GraphRegime, NeighborSummary, PopulationName, RelationProfile, TeacherExportRecord
from scripts.run_formal_gen_only_eval import _parse_args as _parse_formal_gen_only_args
from scripts._formal_eval_helpers import _trim_trailing_text_after_strict_json, build_head_scoring_inputs, generate_structured_outputs


_SUBPROCESS_ENV = {
    **os.environ,
    "MKL_SERVICE_FORCE_INTEL": "1",
}


def _record() -> TeacherExportRecord:
    return TeacherExportRecord(
        dataset_name=DatasetName.AMAZON,
        teacher_model_name="LGHGCLNetV2",
        teacher_checkpoint="outputs/gated/teacher/best_model.pt",
        population_name=PopulationName.FINAL_TEST,
        node_id=3,
        ground_truth_label=1,
        teacher_prob=0.82,
        teacher_logit=1.0,
        hsd=0.2,
        hsd_quantile=0.6,
        asda_switch=True,
        mlp_logit=0.8,
        gnn_logit=1.2,
        branch_gap=-0.4,
        relation_profile=RelationProfile(
            total_relations=3,
            active_relations=2,
            max_relation_neighbor_count=5,
            mean_relation_neighbor_count=1.5,
            max_relation_discrepancy=0.4,
            mean_relation_discrepancy=0.2,
        ),
        neighbor_summary=NeighborSummary(
            total_neighbors=4,
            labeled_neighbors=3,
            positive_neighbors=1,
            negative_neighbors=2,
            unlabeled_neighbors=1,
        ),
        high_hsd_flag=False,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
    )


def _manifest() -> DataManifest:
    return DataManifest(
        dataset_name=DatasetName.AMAZON.value,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD.value,
        feature_dim=25,
        relation_count=3,
        num_nodes=64,
        populations=(
            PopulationMetadata(
                population_name=PopulationName.FINAL_TEST.value,
                split_values=(PopulationName.FINAL_TEST.value,),
                node_ids_hash="a" * 64,
                contains_tuning_rows=False,
                contains_final_test_rows=True,
            ),
        ),
        artifacts=(DataArtifact(kind="source_mat", path="assets/data/Amazon_canonical.mat", sha256="c" * 64),),
    )


class _GenerateTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": dict(kwargs)})
        return {
            "input_ids": torch.tensor([[11, 22, 33]], dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }

    def decode(self, token_ids, *, skip_special_tokens: bool):
        assert skip_special_tokens is True
        assert token_ids.tolist() == [99]
        return '{"rationale":"r","evidence":["e"],"pattern_hint":"p","label":"fraud","score":0.9}'


class _GenerateModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def generate(self, *, input_ids, attention_mask, max_new_tokens, do_sample, temperature, top_p, top_k, pad_token_id, eos_token_id):
        assert do_sample is False
        assert temperature is None
        assert top_p is None
        assert top_k is None
        assert max_new_tokens == 16
        return torch.cat([input_ids, torch.tensor([[99]], device=input_ids.device)], dim=1)



def _run_help(script_name: str):
    return subprocess.run(
        [sys.executable, f"scripts/{script_name}", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=_SUBPROCESS_ENV,
    )



def test_run_formal_gen_only_eval_help_imports() -> None:
    completed = _run_help("run_formal_gen_only_eval.py")

    assert completed.returncode == 0, completed.stderr
    assert "--max-new-tokens" in completed.stdout
    assert "--teacher-export" in completed.stdout


def test_generate_structured_outputs_uses_assistant_generation_prompt() -> None:
    tokenizer = _GenerateTokenizer()

    generated = generate_structured_outputs(
        (_record(),),
        _manifest(),
        model=_GenerateModel(),
        tokenizer=tokenizer,
        thinking_mode="non_thinking",
        max_new_tokens=16,
    )

    assert generated == ('{"rationale":"r","evidence":["e"],"pattern_hint":"p","label":"fraud","score":0.9}',)
    assert tokenizer.calls[0]["kwargs"]["add_generation_prompt"] is True
    assert tokenizer.calls[0]["messages"][-1]["role"] == "user"
    prompt_text = "\n".join(message["content"] for message in tokenizer.calls[0]["messages"])
    assert "teacher_prob: Masked / Not Available" in prompt_text
    assert "teacher_logit: Masked / Not Available" in prompt_text
    assert "teacher_prob: 0.82" not in prompt_text


def test_generate_structured_outputs_progress_logging(capsys) -> None:
    tokenizer = _GenerateTokenizer()
    records = (
        _record(),
        _record().model_copy(update={"node_id": 4}),
    )

    generated = generate_structured_outputs(
        records,
        _manifest(),
        model=_GenerateModel(),
        tokenizer=tokenizer,
        thinking_mode="non_thinking",
        max_new_tokens=16,
        progress_label="audit-gen",
        progress_every=1,
    )

    captured = capsys.readouterr()
    assert generated == (
        '{"rationale":"r","evidence":["e"],"pattern_hint":"p","label":"fraud","score":0.9}',
        '{"rationale":"r","evidence":["e"],"pattern_hint":"p","label":"fraud","score":0.9}',
    )
    assert "[audit-gen] 1/2" in captured.out
    assert "[audit-gen] 2/2" in captured.out
    assert "eta=" in captured.out



def test_build_head_scoring_inputs_can_switch_between_student_visible_and_full_cards() -> None:
    checkpoint = CheckpointProvenance(path="/tmp/cls_head.pt", step=1, content_hash="a" * 64)

    student_inputs = build_head_scoring_inputs(
        (_record(),),
        _manifest(),
        population=PopulationName.FINAL_TEST,
        checkpoint_provenance=checkpoint,
    )
    full_inputs = build_head_scoring_inputs(
        (_record(),),
        _manifest(),
        population=PopulationName.FINAL_TEST,
        checkpoint_provenance=checkpoint,
        student_visible=False,
    )

    student_card = student_inputs.samples[0].evidence_card
    full_card = full_inputs.samples[0].evidence_card

    assert student_card.teacher_summary.teacher_prob is None
    assert student_card.teacher_summary.teacher_logit is None
    assert full_card.teacher_summary.teacher_prob == 0.82
    assert full_card.teacher_summary.teacher_logit == 1.0



def test_trim_trailing_text_after_strict_json_keeps_only_leading_strict_object() -> None:
    text = (
        '{"rationale":"r","evidence":["e"],"pattern_hint":"p","label":"fraud","score":0.9}'
        "\n\nExtra trailing text"
    )

    assert _trim_trailing_text_after_strict_json(text) == (
        '{"rationale":"r","evidence":["e"],"pattern_hint":"p","label":"fraud","score":0.9}'
    )



def test_trim_trailing_text_after_strict_json_does_not_accept_wrapped_json() -> None:
    text = (
        "Here is the JSON: "
        '{"rationale":"r","evidence":["e"],"pattern_hint":"p","label":"fraud","score":0.9}'
    )

    assert _trim_trailing_text_after_strict_json(text) == text



def test_run_formal_gen_only_eval_defaults_to_768_tokens() -> None:
    args = _parse_formal_gen_only_args(
        [
            "--dataset",
            "amazon",
            "--qwen-path",
            "/tmp/qwen",
            "--peft-adapter",
            "/tmp/adapter",
            "--cls-head",
            "/tmp/cls_head.pt",
            "--teacher-export",
            "/tmp/teacher_export.parquet",
            "--data-manifest",
            "/tmp/data_manifest.json",
            "--population",
            "validation",
            "--output-dir",
            "/tmp/out",
            "--run-id",
            "run",
            "--commit",
            "deadbeef",
            "--config-fingerprint",
            "cfg",
        ]
    )

    assert args.max_new_tokens == 768



def test_run_formal_fusion_eval_help_imports() -> None:
    completed = _run_help("run_formal_fusion_eval.py")

    assert completed.returncode == 0, completed.stderr
    assert "--alpha-candidates" in completed.stdout
    assert "--teacher-export-validation" in completed.stdout



def test_run_formal_faithfulness_help_imports() -> None:
    completed = _run_help("run_formal_faithfulness.py")

    assert completed.returncode == 0, completed.stderr
    assert "--formal-head-only-report" in completed.stdout
    assert "--selected-evidence-fields" in completed.stdout
