from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError
import scripts.generate_gate_manifest as gate_generator
from scripts.generate_gate_manifest import _check_population_contract, _derive_head_pass, _strip_runtime_metadata, _validate_raw_formal_report


def test_derive_head_pass_reads_headline_metrics_auroc() -> None:
    head_report = {
        "headline_metrics": {
            "auroc": 0.83,
        }
    }

    assert _derive_head_pass(head_report, 0.7) is True
    assert _derive_head_pass(head_report, 0.9) is False


def test_check_population_contract_supports_real_nested_report_shapes() -> None:
    teacher_baseline = {
        "dataset_name": "amazon",
        "graph_regime": "transductive_standard",
        "eval_type": "head_only",
    }
    head_report = {
        "dataset_name": "amazon",
        "graph_regime": "transductive_standard",
        "headline_metrics": {
            "auroc": 0.98,
        },
    }
    fusion_report = {
        "dataset_name": "amazon",
        "graph_regime": "transductive_standard",
        "validation_metrics": {
            "population": {
                "dataset_name": "amazon",
                "graph_regime": "transductive_standard",
            }
        },
        "report_metrics": {
            "population": {
                "dataset_name": "amazon",
                "graph_regime": "transductive_standard",
            }
        },
    }
    gen_report = {
        "dataset_name": "amazon",
        "graph_regime": "transductive_standard",
    }
    faith_report = {
        "dataset_name": "amazon",
        "graph_regime": "transductive_standard",
    }
    data_manifest = SimpleNamespace(
        dataset_name="amazon",
        graph_regime="transductive_standard",
    )

    assert _check_population_contract(
        teacher_baseline,
        head_report,
        fusion_report,
        gen_report,
        faith_report,
        data_manifest,
    ) is True


def test_strip_runtime_metadata_allows_schema_validation_of_real_report_shape() -> None:
    class _ExtraForbidModel(BaseModel):
        model_config = ConfigDict(extra="forbid")

        schema_version: str
        graph_regime: str

    payload = {
        "schema_version": "formal_head_only/v1",
        "graph_regime": "transductive_standard",
        "_runtime_provenance": {
            "dataset": "amazon",
            "commit": "1" * 40,
        },
    }

    with pytest.raises(ValidationError):
        _ExtraForbidModel.model_validate(payload)

    stripped = _strip_runtime_metadata(payload)

    assert "_runtime_provenance" not in stripped
    _ExtraForbidModel.model_validate(stripped)


def _formal_leakage_payload(prompt_audit_path: str, prompt_audit_hash: str) -> dict:
    return {
        "dataset_name": "amazon",
        "graph_regime": "transductive_standard",
        "eval_type": "head_only",
        "leakage_policy_version": "evidence_leakage_policy/v1",
        "neighbor_label_policy": "removed_from_student_visible",
        "evidence_card_projection": "student_safe_v1",
        "student_visible_forbidden_fields": [
            "neighbor_summary.labeled_neighbors",
            "neighbor_summary.positive_neighbors",
            "neighbor_summary.negative_neighbors",
            "neighbor_summary.unlabeled_neighbors",
        ],
        "teacher_prob_masked": True,
        "teacher_logit_masked": True,
        "neighbor_label_counts_visible": False,
        "formal_safe_result": True,
        "prompt_audit_path": prompt_audit_path,
        "prompt_audit_hash": prompt_audit_hash,
    }


def test_gate_generator_raw_report_validation_fails_missing_leakage_fields(tmp_path: Path) -> None:
    audit = tmp_path / "prompt_audit.json"
    audit.write_text("{}", encoding="utf-8")
    payload = _formal_leakage_payload(str(audit), "0" * 64)
    payload.pop("prompt_audit_hash")
    with pytest.raises(ValueError, match="prompt_audit_hash"):
        _validate_raw_formal_report(
            payload=payload,
            path=tmp_path / "report.json",
            report_name="head_only",
            expected_dataset="amazon",
            expected_graph_regime="transductive_standard",
            require_top_level_identity=True,
        )


def test_gate_generator_raw_report_validation_checks_prompt_audit_hash(tmp_path: Path) -> None:
    audit = tmp_path / "prompt_audit.json"
    audit.write_text("{}", encoding="utf-8")
    payload = _formal_leakage_payload(str(audit), "0" * 64)
    with pytest.raises(ValueError, match="prompt_audit_hash mismatch"):
        _validate_raw_formal_report(
            payload=payload,
            path=tmp_path / "report.json",
            report_name="fusion",
            expected_dataset="amazon",
            expected_graph_regime="transductive_standard",
            require_top_level_identity=True,
        )



def test_gate_generator_raw_report_validation_rejects_missing_top_level_dataset(tmp_path: Path) -> None:
    audit = tmp_path / "prompt_audit.json"
    audit.write_text("{}", encoding="utf-8")
    from scripts._formal_eval_helpers import file_sha256

    payload = _formal_leakage_payload(str(audit), file_sha256(audit))
    payload.pop("dataset_name")
    with pytest.raises(ValueError, match="top-level dataset_name"):
        _validate_raw_formal_report(
            payload=payload,
            path=tmp_path / "report.json",
            report_name="head_only",
            expected_dataset="amazon",
            expected_graph_regime="transductive_standard",
        )


def test_gate_generator_raw_report_validation_accepts_top_level_identity(tmp_path: Path) -> None:
    audit = tmp_path / "prompt_audit.json"
    audit.write_text("{}", encoding="utf-8")
    from scripts._formal_eval_helpers import file_sha256

    payload = _formal_leakage_payload(str(audit), file_sha256(audit))
    _validate_raw_formal_report(
        payload=payload,
        path=tmp_path / "report.json",
        report_name="head_only",
        expected_dataset="amazon",
        expected_graph_regime="transductive_standard",
    )


def test_synthetic_compliant_bundle_passes_generate_gate_manifest_and_gate_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the positive path through generate_gate_manifest.py and gate_check.py."""

    from graph_data.manifests import DataArtifact, DataManifest, PopulationMetadata
    from priorf_teacher.schema import DatasetName, GraphRegime, PopulationName, TeacherBaselineReport
    from scripts._formal_eval_helpers import file_sha256
    from scripts.gate_check import gate_check

    import eval.eval_head_only as eval_head_only
    from eval.eval_gen_only import GenOnlyEvalInputs, GenOnlyEvalSample, evaluate_gen_only
    from eval.eval_fusion import FusionEvalConfig, FusionPopulationInputs, run_formal_fusion_eval
    from eval.faithfulness import evaluate_faithfulness
    from tests import test_eval_head_only as head_helpers
    from tests import test_eval_gen_only as gen_helpers
    from tests import test_eval_fusion as fusion_helpers
    from tests import test_faithfulness as faith_helpers

    audit = tmp_path / "prompt_audit.json"
    audit.write_text('{"leakage_audit_pass": true, "violations": []}\n', encoding="utf-8")
    audit_hash = file_sha256(audit)
    audit_kwargs = {"prompt_audit_path": str(audit), "prompt_audit_hash": audit_hash}

    runs_root = tmp_path / "formal"
    (runs_root / "head_only").mkdir(parents=True)
    (runs_root / "gen_only").mkdir(parents=True)
    (runs_root / "fusion").mkdir(parents=True)
    (runs_root / "faithfulness").mkdir(parents=True)

    validation_inputs = head_helpers._inputs(PopulationName.VALIDATION, (3, 4, 8, 9), (0, 0, 1, 1))
    report_inputs = head_helpers._inputs(PopulationName.FINAL_TEST, (21, 22, 23, 24), (0, 1, 1, 0))
    validation_report = head_helpers._scorer_report(
        population_name=PopulationName.VALIDATION,
        probs=(0.10, 0.20, 0.80, 0.90),
        labels=(0, 0, 1, 1),
        node_ids=(3, 4, 8, 9),
    ).model_copy(update=audit_kwargs)
    report_population = head_helpers._scorer_report(
        population_name=PopulationName.FINAL_TEST,
        probs=(0.10, 0.90, 0.80, 0.20),
        labels=(0, 1, 1, 0),
        node_ids=(21, 22, 23, 24),
    ).model_copy(update=audit_kwargs)

    def fake_score_head(**kwargs):
        if kwargs["inputs"] == validation_inputs:
            return validation_report
        if kwargs["inputs"] == report_inputs:
            return report_population
        raise AssertionError("unexpected head scoring input")

    monkeypatch.setattr(eval_head_only, "score_head", fake_score_head)
    head_report = eval_head_only.run_formal_head_only_eval(
        validation_inputs=validation_inputs,
        report_inputs=report_inputs,
        validation_population_metadata=head_helpers._population_metadata(PopulationName.VALIDATION),
        report_population_metadata=head_helpers._population_metadata(PopulationName.FINAL_TEST),
        model=object(),
        cls_head=object(),
        tokenizer=object(),
        thinking_mode=head_helpers.ThinkingMode.NON_THINKING,
        checkpoint_source="best_checkpoint",
        checkpoint_bundle=head_helpers._checkpoint_bundle(),
        run_id="run-123",
        threshold_selection_metric="f1",
        include_oracle_diagnostics=False,
        calibration_bins=4,
        **audit_kwargs,
    )
    (runs_root / "head_only" / "formal_head_only_report_amazon.json").write_text(
        head_report.model_dump_json(indent=2), encoding="utf-8"
    )

    gen_samples = tuple(
        GenOnlyEvalSample(
            evidence_card=gen_helpers._sample(
                node_id=node_id,
                ground_truth_label=label,
                generated_text=generated_text,
            ).evidence_card,
            generated_text=generated_text,
            ground_truth_label=label,
            node_id=node_id,
        )
        for node_id, label, generated_text in (
            (1, 1, '{"rationale":"ok","evidence":["e"],"pattern_hint":"p","label":"fraud","score":0.9}'),
            (2, 0, '{"rationale":"ok","evidence":["e"],"pattern_hint":"p","label":"benign","score":0.1}'),
        )
    )
    gen_report = evaluate_gen_only(
        inputs=GenOnlyEvalInputs(
            samples=gen_samples,
            dataset_name=DatasetName.AMAZON,
            population_name=PopulationName.VALIDATION,
            graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
            run_id="run-123",
            **audit_kwargs,
        )
    )
    (runs_root / "gen_only" / "formal_gen_only_report_amazon.json").write_text(
        gen_report.model_dump_json(indent=2), encoding="utf-8"
    )

    fusion_report = run_formal_fusion_eval(
        validation_inputs=fusion_helpers._inputs(
            population=PopulationName.VALIDATION,
            head_probs=(0.61, 0.32, 0.51, 0.40, 0.37, 0.58),
            teacher_probs=(0.84, 0.14, 0.17, 0.25, 0.92, 0.44),
            labels=(1, 0, 1, 0, 1, 0),
        ),
        report_inputs=fusion_helpers._inputs(
            population=PopulationName.FINAL_TEST,
            head_probs=(0.64, 0.29, 0.57, 0.33, 0.41, 0.55),
            teacher_probs=(0.79, 0.18, 0.22, 0.30, 0.88, 0.40),
            labels=(1, 0, 1, 0, 1, 0),
        ),
        config=FusionEvalConfig(alpha_candidates=(0.0, 0.5, 1.0), min_student_alpha=0.25),
        run_id="run-123",
        **audit_kwargs,
    )
    (runs_root / "fusion" / "formal_fusion_report_amazon.json").write_text(
        fusion_report.model_dump_json(indent=2), encoding="utf-8"
    )

    faith_report = evaluate_faithfulness(
        inputs=faith_helpers._faithfulness_inputs(run_id="run-123", **audit_kwargs),
        model=faith_helpers.TraceModel(),
        cls_head=faith_helpers.TraceClsHead(),
        tokenizer=faith_helpers.ScoreAwareTokenizer(),
    )
    (runs_root / "faithfulness" / "faithfulness_report_amazon.json").write_text(
        faith_report.model_dump_json(indent=2), encoding="utf-8"
    )

    data_manifest = DataManifest(
        dataset_name="amazon",
        graph_regime="transductive_standard",
        feature_dim=25,
        relation_count=3,
        num_nodes=256,
        populations=(
            PopulationMetadata(
                population_name="validation",
                split_values=("validation",),
                node_ids_hash="1" * 64,
                contains_tuning_rows=True,
                contains_final_test_rows=False,
            ),
            PopulationMetadata(
                population_name="final_test",
                split_values=("final_test",),
                node_ids_hash="2" * 64,
                contains_tuning_rows=False,
                contains_final_test_rows=True,
            ),
        ),
        artifacts=(DataArtifact(kind="source_mat", path="amazon.mat", sha256="3" * 64),),
    )
    data_manifest_path = tmp_path / "data_manifest.json"
    data_manifest_path.write_text(data_manifest.model_dump_json(indent=2), encoding="utf-8")
    data_manifest_hash = file_sha256(data_manifest_path)

    baseline = TeacherBaselineReport(
        dataset_name=DatasetName.AMAZON,
        teacher_model_name="PriorF-GNN",
        teacher_checkpoint_sha256="4" * 64,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        population_name=PopulationName.VALIDATION,
        metric_name="auroc",
        metric_value=0.90,
        threshold=0.70,
        passed=True,
        data_manifest_sha256=data_manifest_hash,
        code_git_sha="a" * 40,
        export_timestamp_utc=datetime.now(timezone.utc),
    )
    baseline_path = tmp_path / "teacher_baseline.json"
    baseline_path.write_text(baseline.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.setattr(gate_generator, "capture_git_state", lambda _root: {"git_commit": "a" * 40, "git_dirty": False})
    output_path = tmp_path / "gate_manifest.json"
    exit_code = gate_generator.main([
        "--dataset", "amazon",
        "--runs-root", str(runs_root),
        "--run-id", "run-123",
        "--output-path", str(output_path),
        "--data-manifest", str(data_manifest_path),
        "--teacher-baseline-report", str(baseline_path),
        "--commit", "a" * 40,
        "--config-fingerprint", "cfg-123",
        "--data-validation-pass", "true",
        "--smoke-pipeline-pass", "true",
        "--prompt-audit-path", str(audit),
        "--leakage-audit-pass", "true",
        "--min-head-auroc", "0.7",
        "--min-strict-parse-rate", "0.95",
        "--max-teacher-prob-flip-rate", "0.5",
    ])

    assert exit_code == 0
    manifest = gate_check(manifest_path=output_path)
    assert manifest.dataset_name == DatasetName.AMAZON
    assert manifest.population_contract_pass is True
    assert manifest.provenance.generator_git_dirty is False
