from __future__ import annotations

import pytest
from pydantic import ValidationError

from evidence.evidence_schema import EvidenceAblationMask, build_evidence_card
from evidence.output_schema import PredLabel, StrictOutput, canonical_serialize
from evidence.prompt_builder import FewShotExample, PromptMode, ThinkingMode, build_prompt
from graph_data.manifests import DataArtifact, DataManifest, PopulationMetadata
from priorf_teacher.schema import DatasetName, GraphRegime, NeighborSummary, PopulationName, RelationProfile, TeacherExportRecord


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


def _record(**overrides) -> TeacherExportRecord:
    payload = {
        "dataset_name": DatasetName.AMAZON,
        "teacher_model_name": "PriorF-GNN",
        "teacher_checkpoint": "checkpoints/priorf.pt",
        "population_name": PopulationName.VALIDATION,
        "node_id": 1,
        "ground_truth_label": 1,
        "teacher_prob": 0.92,
        "teacher_logit": 2.1,
        "hsd": 0.3,
        "hsd_quantile": 0.8,
        "asda_switch": True,
        "mlp_logit": 1.7,
        "gnn_logit": 2.2,
        "branch_gap": 0.5,
        "relation_profile": _relation_profile(),
        "neighbor_summary": _neighbor_summary(),
        "high_hsd_flag": True,
        "graph_regime": GraphRegime.TRANSDUCTIVE_STANDARD,
    }
    payload.update(overrides)
    return TeacherExportRecord.model_validate(payload)


def _manifest() -> DataManifest:
    return DataManifest(
        dataset_name="amazon",
        graph_regime="transductive_standard",
        feature_dim=25,
        relation_count=3,
        num_nodes=3,
        populations=(
            PopulationMetadata(
                population_name="validation",
                split_values=("validation",),
                node_ids_hash="a" * 64,
                contains_tuning_rows=True,
                contains_final_test_rows=False,
            ),
        ),
        artifacts=(DataArtifact(kind="source_mat", path="amazon.mat", sha256="0" * 64),),
    )


def _card(**kwargs):
    return build_evidence_card(teacher_record=_record(), data_manifest=_manifest(), **kwargs)


def _target(label: PredLabel = PredLabel.FRAUD) -> StrictOutput:
    return StrictOutput(
        rationale="example rationale",
        evidence=("example evidence",),
        pattern_hint="example hint",
        label=label,
        score=0.75,
    )


def _roles(bundle):
    return tuple(message.role for message in bundle.messages)


def test_prompt_mode_and_thinking_mode_unknown_raise():
    with pytest.raises(ValueError):
        PromptMode("unknown")
    with pytest.raises(TypeError):
        build_prompt(evidence_card=_card(), mode=PromptMode.TRAIN)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        build_prompt(evidence_card=_card(), mode=PromptMode.TRAIN, thinking_mode="unknown")  # type: ignore[arg-type]


def test_train_mode_assistant_content_is_canonical_serialized_target():
    bundle = build_prompt(
        evidence_card=_card(),
        mode=PromptMode.TRAIN,
        thinking_mode=ThinkingMode.NON_THINKING,
        ground_truth_label_for_sft=PredLabel.FRAUD,
        score_target_for_sft=0.87,
    )
    assert bundle.sft_target_label is not None
    assert bundle.messages[-1].role == "assistant"
    assert bundle.messages[-1].content == canonical_serialize(bundle.sft_target_label)


def test_non_train_modes_have_empty_assistant_and_no_sft_target():
    for mode in (PromptMode.VALIDATION, PromptMode.EVAL_HEAD, PromptMode.EVAL_GEN, PromptMode.GENERATION):
        bundle = build_prompt(evidence_card=_card(), mode=mode, thinking_mode=ThinkingMode.NON_THINKING)
        assert bundle.sft_target_label is None
        assert bundle.messages[-1].role == "assistant"
        assert bundle.messages[-1].content == ""


def test_train_requires_sft_inputs_and_other_modes_forbid_them():
    with pytest.raises(ValueError, match="train mode requires"):
        build_prompt(evidence_card=_card(), mode=PromptMode.TRAIN, thinking_mode=ThinkingMode.NON_THINKING)
    with pytest.raises(ValueError, match="non-train"):
        build_prompt(
            evidence_card=_card(),
            mode=PromptMode.VALIDATION,
            thinking_mode=ThinkingMode.NON_THINKING,
            ground_truth_label_for_sft=PredLabel.BENIGN,
            score_target_for_sft=0.1,
        )


def test_cross_mode_message_structure_system_user_fewshot_text_and_order_are_invariant():
    example_card = build_evidence_card(
        teacher_record=_record(population_name=PopulationName.TRAIN, node_id=0),
        data_manifest=_manifest(),
    )
    example = FewShotExample(evidence_card=example_card, sft_target_label=_target(PredLabel.BENIGN), source_population=PopulationName.TRAIN)
    train_bundle = build_prompt(
        evidence_card=_card(),
        mode=PromptMode.TRAIN,
        thinking_mode=ThinkingMode.NON_THINKING,
        ground_truth_label_for_sft=PredLabel.FRAUD,
        score_target_for_sft=0.9,
        few_shot_examples=(example,),
    )
    validation_bundle = build_prompt(
        evidence_card=_card(),
        mode=PromptMode.VALIDATION,
        thinking_mode=ThinkingMode.NON_THINKING,
        few_shot_examples=(example,),
    )
    assert _roles(train_bundle) == _roles(validation_bundle) == ("system", "user", "assistant", "user", "assistant")
    for train_message, validation_message in zip(train_bundle.messages[:-1], validation_bundle.messages[:-1]):
        assert train_message == validation_message
    assert train_bundle.messages[-1].content != validation_bundle.messages[-1].content


def test_few_shot_population_must_be_train_and_forbidden_in_eval_modes():
    with pytest.raises(ValidationError, match="train population"):
        FewShotExample(evidence_card=_card(), sft_target_label=_target(), source_population=PopulationName.VALIDATION)

    example_card = build_evidence_card(
        teacher_record=_record(population_name=PopulationName.TRAIN, node_id=0),
        data_manifest=_manifest(),
    )
    example = FewShotExample(evidence_card=example_card, sft_target_label=_target(), source_population=PopulationName.TRAIN)
    for mode in (PromptMode.EVAL_HEAD, PromptMode.EVAL_GEN, PromptMode.GENERATION):
        with pytest.raises(ValueError, match="few-shot"):
            build_prompt(evidence_card=_card(), mode=mode, thinking_mode=ThinkingMode.NON_THINKING, few_shot_examples=(example,))


def test_ablation_mask_renders_sentinel_without_changing_roles_or_order():
    unmasked = build_prompt(evidence_card=_card(), mode=PromptMode.VALIDATION, thinking_mode=ThinkingMode.NON_THINKING)
    masked_card = _card(
        ablation_mask=frozenset(
            {
                EvidenceAblationMask.TEACHER_SUMMARY_TEACHER_PROB,
                EvidenceAblationMask.RELATION_PROFILE_TOTAL_RELATIONS,
            }
        )
    )
    masked = build_prompt(evidence_card=masked_card, mode=PromptMode.VALIDATION, thinking_mode=ThinkingMode.NON_THINKING)
    assert _roles(masked) == _roles(unmasked)
    assert masked.messages[0] == unmasked.messages[0]
    assert "Masked / Not Available" in masked.messages[-2].content
    assert "teacher_prob: Masked / Not Available" in masked.messages[-2].content
    assert "total_relations: Masked / Not Available" in masked.messages[-2].content
    assert "teacher_prob: 0.0" not in masked.messages[-2].content
    assert "teacher_prob: -1" not in masked.messages[-2].content
    assert "teacher_prob: 999" not in masked.messages[-2].content


def test_graph_regime_is_rendered_in_task_instruction_text():
    bundle = build_prompt(evidence_card=_card(), mode=PromptMode.VALIDATION, thinking_mode=ThinkingMode.NON_THINKING)
    assert "task_instruction:" in bundle.messages[-2].content
    assert "graph_regime: transductive_standard" in bundle.messages[-2].content


@pytest.mark.parametrize("env_key", ["PRIORF_PROMPT_STYLE", "EVIDENCE_MODE", "PROMPT_BUILDER_FLAG"])
def test_prompt_related_environment_variables_fail_closed(monkeypatch, env_key):
    monkeypatch.setenv(env_key, "1")
    with pytest.raises(RuntimeError, match="environment override"):
        build_prompt(evidence_card=_card(), mode=PromptMode.VALIDATION, thinking_mode=ThinkingMode.NON_THINKING)
