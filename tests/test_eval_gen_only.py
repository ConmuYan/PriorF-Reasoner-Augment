from __future__ import annotations

import math

import pytest
from pydantic import ValidationError
from sklearn import metrics

from eval.eval_gen_only import GenOnlyEvalInputs, GenOnlyEvalSample, evaluate_gen_only
from evidence.evidence_schema import build_evidence_card
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


def _record(node_id: int, ground_truth_label: int) -> TeacherExportRecord:
    return TeacherExportRecord(
        dataset_name=DatasetName.AMAZON,
        teacher_model_name="PriorF-GNN",
        teacher_checkpoint="checkpoints/priorf.pt",
        population_name=PopulationName.VALIDATION,
        node_id=node_id,
        ground_truth_label=ground_truth_label,  # type: ignore[arg-type]
        teacher_prob=0.92 if ground_truth_label == 1 else 0.08,
        teacher_logit=2.1 if ground_truth_label == 1 else -2.1,
        hsd=0.3,
        hsd_quantile=0.8,
        asda_switch=True,
        mlp_logit=1.7,
        gnn_logit=2.2,
        branch_gap=0.5,
        relation_profile=_relation_profile(),
        neighbor_summary=_neighbor_summary(),
        high_hsd_flag=True,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
    )


def _manifest() -> DataManifest:
    return DataManifest(
        dataset_name="amazon",
        graph_regime="transductive_standard",
        feature_dim=25,
        relation_count=3,
        num_nodes=16,
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




def _eval_audit_kwargs() -> dict[str, str]:
    return {
        "run_id": "run-123",
        "prompt_audit_path": "outputs/tests/prompt_audit.json",
        "prompt_audit_hash": "a" * 64,
    }

def _sample(*, node_id: int, ground_truth_label: int, generated_text: str) -> GenOnlyEvalSample:
    card = build_evidence_card(
        teacher_record=_record(node_id=node_id, ground_truth_label=ground_truth_label),
        data_manifest=_manifest(),
    )
    return GenOnlyEvalSample(
        evidence_card=card,
        generated_text=generated_text,
        ground_truth_label=ground_truth_label,  # type: ignore[arg-type]
        node_id=node_id,
    )


def test_eval_gen_only_uses_strict_parse_as_headline_and_keeps_failures_in_denominator():
    inputs = GenOnlyEvalInputs(
        samples=(
            _sample(
                node_id=1,
                ground_truth_label=1,
                generated_text=(
                    '{"rationale":"strict ok","evidence":["teacher evidence"],'
                    '"pattern_hint":"pattern","label":"fraud","score":0.8}'
                ),
            ),
            _sample(
                node_id=2,
                ground_truth_label=0,
                generated_text=(
                    '```json\n'
                    '{"Rationale":"diagnostic only","Evidence":"alias evidence",'
                    '"patternHint":"aliased hint","pred_label":"BENIGN","confidence":0.75}'
                    '\n```'
                ),
            ),
            _sample(node_id=3, ground_truth_label=1, generated_text="not json at all"),
        ),
        dataset_name=DatasetName.AMAZON,
        population_name=PopulationName.VALIDATION,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        **_eval_audit_kwargs(),
    )

    report = evaluate_gen_only(inputs=inputs)

    assert report.headline_metric_name == "strict_schema_parse_rate"
    assert report.headline_metric_value == pytest.approx(1 / 3)
    assert report.strict_schema_parse_rate == pytest.approx(1 / 3)
    assert report.normalized_parse_rate == pytest.approx(2 / 3)
    assert report.teacher_conditioned is True
    assert report.prompt_mode == "eval_gen"
    assert report.run_id == "run-123"
    assert report.report_split == PopulationName.VALIDATION
    assert report.eval_type == "gen_only"
    assert report.strict_parse_failure_policy == "penalty_mode"
    assert report.strict_parse_penalty_strategy == "ground_truth_opposite_extreme_probability"
    assert report.strict_parse_failure_count == 2
    assert report.strict_parse_failure_rate == pytest.approx(2 / 3)
    assert report.strict_parse_failure_node_ids == (2, 3)
    assert report.normalized_parse_success_count == 2
    assert report.strict_parse_succeeded == (True, False, False)
    assert report.normalized_parse_succeeded == (True, True, False)
    assert report.probs == pytest.approx((0.8, 1.0, 0.0))
    assert report.labels == (1, 0, 1)
    assert report.node_ids == (1, 2, 3)
    assert report.brier_score == pytest.approx((0.04 + 1.0 + 1.0) / 3)
    assert report.auroc == pytest.approx(metrics.roc_auc_score([1, 0, 1], [0.8, 1.0, 0.0]))
    assert report.auprc == pytest.approx(metrics.average_precision_score([1, 0, 1], [0.8, 1.0, 0.0]))

    missing_identity_payload = report.model_dump(mode="python")
    missing_identity_payload.pop("run_id")
    with pytest.raises(ValidationError):
        type(report).model_validate(missing_identity_payload)


def test_eval_gen_only_treats_score_as_fraud_probability_even_for_benign_label():
    inputs = GenOnlyEvalInputs(
        samples=(
            _sample(
                node_id=6,
                ground_truth_label=0,
                generated_text=(
                    '{"rationale":"strict ok","evidence":["teacher evidence"],'
                    '"pattern_hint":"pattern","label":"benign","score":0.05}'
                ),
            ),
            _sample(
                node_id=7,
                ground_truth_label=1,
                generated_text=(
                    '{"rationale":"strict ok","evidence":["teacher evidence"],'
                    '"pattern_hint":"pattern","label":"fraud","score":0.95}'
                ),
            ),
        ),
        dataset_name=DatasetName.AMAZON,
        population_name=PopulationName.VALIDATION,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        **_eval_audit_kwargs(),
    )

    report = evaluate_gen_only(inputs=inputs)

    assert report.probs == pytest.approx((0.05, 0.95))
    assert report.auroc == pytest.approx(1.0)
    assert report.auprc == pytest.approx(1.0)


def test_eval_gen_only_single_class_population_leaves_auroc_and_auprc_none():
    inputs = GenOnlyEvalInputs(
        samples=(
            _sample(
                node_id=4,
                ground_truth_label=1,
                generated_text=(
                    '{"rationale":"strict ok","evidence":["teacher evidence"],'
                    '"pattern_hint":"pattern","label":"fraud","score":0.9}'
                ),
            ),
            _sample(node_id=5, ground_truth_label=1, generated_text="malformed"),
        ),
        dataset_name=DatasetName.AMAZON,
        population_name=PopulationName.VALIDATION,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        **_eval_audit_kwargs(),
    )

    report = evaluate_gen_only(inputs=inputs)

    assert report.is_single_class_population is True
    assert report.auroc is None
    assert report.auprc is None
    assert math.isfinite(report.brier_score)


def test_old_prefixed_gen_only_report_without_leakage_fields_fails_validation():
    inputs = GenOnlyEvalInputs(
        samples=(
            _sample(
                node_id=8,
                ground_truth_label=1,
                generated_text=(
                    '{"rationale":"strict ok","evidence":["teacher evidence"],'
                    '"pattern_hint":"pattern","label":"fraud","score":0.9}'
                ),
            ),
        ),
        dataset_name=DatasetName.AMAZON,
        population_name=PopulationName.VALIDATION,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        **_eval_audit_kwargs(),
    )
    payload = evaluate_gen_only(inputs=inputs).model_dump(mode="python")
    for field_name in (
        "leakage_policy_version",
        "neighbor_label_policy",
        "evidence_card_projection",
        "student_visible_forbidden_fields",
        "teacher_prob_masked",
        "teacher_logit_masked",
        "neighbor_label_counts_visible",
        "formal_safe_result",
        "prompt_audit_path",
        "prompt_audit_hash",
    ):
        payload.pop(field_name, None)
    with pytest.raises(ValidationError):
        type(evaluate_gen_only(inputs=inputs)).model_validate(payload)

    payload = evaluate_gen_only(inputs=inputs).model_dump(mode="python")
    payload["teacher_prob_masked"] = False
    with pytest.raises(ValidationError):
        type(evaluate_gen_only(inputs=inputs)).model_validate(payload)
