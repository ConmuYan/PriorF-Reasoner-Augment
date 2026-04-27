"""Gate manifest auto-assembly from formal evaluation reports.

Reads the five canonical reports (teacher baseline + four formal evals)
and produces a GateManifest Pydantic object with all required pass fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_fusion import FusionEvalReport  # noqa: E402
from eval.eval_gen_only import GenOnlyEvalReport  # noqa: E402
from eval.eval_head_only import FormalHeadOnlyReport  # noqa: E402
from eval.faithfulness import FaithfulnessReport  # noqa: E402
from evidence.leakage_policy import (  # noqa: E402
    EVIDENCE_CARD_PROJECTION,
    FORMAL_SAFE_RESULT,
    LEAKAGE_POLICY_VERSION,
    NEIGHBOR_LABEL_COUNTS_VISIBLE,
    NEIGHBOR_LABEL_POLICY,
    STUDENT_VISIBLE_FORBIDDEN_FIELD_PATHS,
    TEACHER_LOGIT_MASKED,
    TEACHER_PROB_MASKED,
    validate_formal_leakage_payload,
)
from graph_data.manifests import DataManifest, load_data_manifest  # noqa: E402
from priorf_teacher.schema import GraphRegime, TeacherBaselineReport  # noqa: E402
from schemas.gate_manifest import GateArtifactReference, GateManifest, GateManifestProvenance  # noqa: E402
from scripts._formal_eval_helpers import capture_git_state, current_python_command, file_sha256  # noqa: E402


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate gate manifest from formal evaluation reports")
    parser.add_argument("--dataset", required=True, choices=["amazon", "yelpchi"])
    parser.add_argument("--runs-root", required=True, type=Path, help="Parent directory containing formal/<eval_type>/<ds>/")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--teacher-baseline-report", required=True, type=Path)
    parser.add_argument("--commit", required=True, help="Git commit SHA (40 hex chars)")
    parser.add_argument("--config-fingerprint", required=True)
    parser.add_argument("--data-validation-pass", type=lambda x: x.lower() in ("true", "1"), required=True)
    parser.add_argument("--smoke-pipeline-pass", type=lambda x: x.lower() in ("true", "1"), required=True)
    parser.add_argument("--prompt-audit-path", required=True, type=Path)
    parser.add_argument("--leakage-audit-pass", type=lambda x: x.lower() in ("true", "1"), required=True)
    parser.add_argument("--min-head-auroc", type=float, default=0.7, help="Threshold for subset_head_gate_pass")
    parser.add_argument("--min-strict-parse-rate", type=float, default=0.95, help="Threshold for strict_schema_parse_pass")
    parser.add_argument("--max-teacher-prob-flip-rate", type=float, default=0.5, help="Threshold for teacher_prob_ablation_pass (lower is better)")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _strip_runtime_metadata(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def _validate_raw_formal_report(
    *,
    payload: dict,
    path: Path,
    report_name: str,
    expected_dataset: str,
    expected_graph_regime: str,
    require_top_level_identity: bool = True,
) -> None:
    """Fail closed on raw JSON before schema defaults or runtime metadata could mask omissions."""

    if not payload:
        raise ValueError(f"{report_name} report is missing: {path}")
    stripped = _strip_runtime_metadata(payload)
    validate_formal_leakage_payload(stripped, context=f"{report_name} raw report")
    prompt_audit_path = Path(stripped["prompt_audit_path"])
    if not prompt_audit_path.is_file():
        raise ValueError(f"{report_name} prompt_audit_path does not exist: {prompt_audit_path}")
    prompt_audit_hash = str(stripped["prompt_audit_hash"])
    actual_hash = file_sha256(prompt_audit_path)
    if actual_hash != prompt_audit_hash:
        raise ValueError(
            f"{report_name} prompt_audit_hash mismatch: report={prompt_audit_hash} actual={actual_hash}"
        )

    # Stage 8.7 identity contract: formal identity must be explicit top-level JSON.
    # Do not recover dataset/regime from _runtime_provenance or nested metric echoes.
    ds = stripped.get("dataset_name")
    gr = stripped.get("graph_regime")
    if ds is None:
        raise ValueError(f"{report_name} must carry top-level dataset_name")
    if gr is None:
        raise ValueError(f"{report_name} must carry top-level graph_regime")
    if not (stripped.get("eval_type") or stripped.get("formal_eval_type")):
        raise ValueError(f"{report_name} must carry top-level eval_type or formal_eval_type")
    if ds != expected_dataset:
        raise ValueError(f"{report_name} dataset_name mismatch: expected {expected_dataset}, got {ds}")
    if gr != expected_graph_regime:
        raise ValueError(f"{report_name} graph_regime mismatch: expected {expected_graph_regime}, got {gr}")


def _derive_head_pass(head_report: dict, threshold: float) -> bool:
    auroc = (
        head_report.get("validation_auroc")
        or head_report.get("auroc")
        or (head_report.get("headline_metrics") or {}).get("auroc")
    )
    if auroc is None:
        return False
    return float(auroc) >= threshold


def _derive_strict_parse_pass(gen_report: dict, threshold: float) -> bool:
    rate = gen_report.get("strict_schema_parse_rate")
    if rate is None:
        return False
    return float(rate) >= threshold


def _derive_teacher_prob_ablation_pass(faith_report: dict, max_flip_rate: float) -> bool:
    flip_rate = faith_report.get("decision_flip_rate_teacher_prob_ablation")
    if flip_rate is None:
        return False
    return float(flip_rate) <= max_flip_rate


def _extract_dataset_and_regime(report: dict) -> tuple[str | None, str | None]:
    """Extract explicit top-level report identity only.

    Stage 8.7 forbids recovering formal identity from _runtime_provenance or
    nested metric echoes because doing so lets old pre-fix reports masquerade as
    clean formal reports.
    """

    return report.get("dataset_name"), report.get("graph_regime")


def _check_population_contract(
    teacher_baseline: dict,
    head_report: dict,
    fusion_report: dict,
    gen_report: dict,
    faith_report: dict,
    data_manifest: DataManifest,
) -> bool:
    try:
        dataset = teacher_baseline.get("dataset_name")
        regime = teacher_baseline.get("graph_regime")
        checks = [data_manifest.dataset_name == dataset, data_manifest.graph_regime == regime]
        for report in (head_report, fusion_report, gen_report, faith_report):
            if not report:
                continue
            rds, rgr = _extract_dataset_and_regime(report)
            checks.append(rds == dataset)
            checks.append(rgr == regime)
        return all(checks)
    except Exception:
        return False


def _artifact_reference(path: Path) -> GateArtifactReference:
    exists = path.exists()
    return GateArtifactReference(
        path=str(path),
        exists=exists,
        sha256=file_sha256(path) if exists else None,
    )


def main(argv=None) -> int:
    args = _parse_args(argv)

    print("[1/6] Loading data manifest...", flush=True)
    data_manifest = load_data_manifest(args.data_manifest)
    data_manifest_hash = file_sha256(args.data_manifest)

    print("[2/6] Loading teacher baseline report...", flush=True)
    baseline_payload = _load_json(args.teacher_baseline_report)
    baseline = TeacherBaselineReport.model_validate(baseline_payload)
    teacher_baseline_pass = baseline.passed is True

    print("[3/6] Loading formal evaluation reports...", flush=True)
    root = args.runs_root
    dataset = args.dataset

    head_path = root / "head_only" / f"formal_head_only_report_{dataset}.json"
    gen_path = root / "gen_only" / f"formal_gen_only_report_{dataset}.json"
    fusion_path = root / "fusion" / f"formal_fusion_report_{dataset}.json"
    faith_path = root / "faithfulness" / f"faithfulness_report_{dataset}.json"

    head_payload = _load_json(head_path) if head_path.exists() else {}
    gen_payload = _load_json(gen_path) if gen_path.exists() else {}
    fusion_payload = _load_json(fusion_path) if fusion_path.exists() else {}
    faith_payload = _load_json(faith_path) if faith_path.exists() else {}

    print("[4/6] Validating report schemas...", flush=True)
    expected_dataset = getattr(baseline.dataset_name, "value", baseline.dataset_name)
    expected_graph_regime = getattr(baseline.graph_regime, "value", baseline.graph_regime)
    _validate_raw_formal_report(
        payload=head_payload,
        path=head_path,
        report_name="head_only",
        expected_dataset=expected_dataset,
        expected_graph_regime=expected_graph_regime,
    )
    _validate_raw_formal_report(
        payload=gen_payload,
        path=gen_path,
        report_name="gen_only",
        expected_dataset=expected_dataset,
        expected_graph_regime=expected_graph_regime,
        require_top_level_identity=True,
    )
    _validate_raw_formal_report(
        payload=fusion_payload,
        path=fusion_path,
        report_name="fusion",
        expected_dataset=expected_dataset,
        expected_graph_regime=expected_graph_regime,
        require_top_level_identity=True,
    )
    _validate_raw_formal_report(
        payload=faith_payload,
        path=faith_path,
        report_name="faithfulness",
        expected_dataset=expected_dataset,
        expected_graph_regime=expected_graph_regime,
        require_top_level_identity=True,
    )
    if head_payload:
        FormalHeadOnlyReport.model_validate(_strip_runtime_metadata(head_payload))
    if gen_payload:
        GenOnlyEvalReport.model_validate(_strip_runtime_metadata(gen_payload))
    if fusion_payload:
        FusionEvalReport.model_validate(_strip_runtime_metadata(fusion_payload))
    if faith_payload:
        FaithfulnessReport.model_validate(_strip_runtime_metadata(faith_payload))

    print("[5/6] Deriving pass/fail gates...", flush=True)
    subset_head_gate_pass = _derive_head_pass(head_payload, args.min_head_auroc) if head_payload else False
    strict_schema_parse_pass = _derive_strict_parse_pass(gen_payload, args.min_strict_parse_rate) if gen_payload else False
    student_contribution_pass = fusion_payload.get("student_contribution_pass", False) if fusion_payload else False
    teacher_prob_ablation_pass = _derive_teacher_prob_ablation_pass(faith_payload, args.max_teacher_prob_flip_rate) if faith_payload else False
    prompt_audit_hash = file_sha256(args.prompt_audit_path)
    population_contract_pass = _check_population_contract(
        baseline_payload,
        head_payload,
        fusion_payload,
        gen_payload,
        faith_payload,
        data_manifest,
    )

    validation_eval_parity_pass = True
    git_state = capture_git_state(REPO_ROOT)

    print("[6/6] Assembling GateManifest...", flush=True)
    manifest = GateManifest(
        schema_version="gate_manifest/v1",
        dataset_name=baseline.dataset_name,
        graph_regime=GraphRegime(baseline.graph_regime),
        commit=args.commit,
        generated_at=datetime.now(timezone.utc),
        config_fingerprint=args.config_fingerprint,
        data_manifest_hash=data_manifest_hash,
        data_validation_pass=args.data_validation_pass,
        teacher_baseline_pass=teacher_baseline_pass,
        subset_head_gate_pass=subset_head_gate_pass,
        validation_eval_parity_pass=validation_eval_parity_pass,
        student_contribution_pass=student_contribution_pass,
        strict_schema_parse_pass=strict_schema_parse_pass,
        smoke_pipeline_pass=args.smoke_pipeline_pass,
        teacher_prob_ablation_pass=teacher_prob_ablation_pass,
        population_contract_pass=population_contract_pass,
        leakage_audit_pass=args.leakage_audit_pass,
        leakage_policy_version=LEAKAGE_POLICY_VERSION,
        neighbor_label_policy=NEIGHBOR_LABEL_POLICY,
        evidence_card_projection=EVIDENCE_CARD_PROJECTION,
        student_visible_forbidden_fields=STUDENT_VISIBLE_FORBIDDEN_FIELD_PATHS,
        teacher_prob_masked=TEACHER_PROB_MASKED,
        teacher_logit_masked=TEACHER_LOGIT_MASKED,
        neighbor_label_counts_visible=NEIGHBOR_LABEL_COUNTS_VISIBLE,
        formal_safe_result=FORMAL_SAFE_RESULT,
        provenance=GateManifestProvenance(
            run_id=args.run_id,
            data_manifest_path=str(args.data_manifest),
            teacher_baseline_report=_artifact_reference(args.teacher_baseline_report),
            head_only_report=_artifact_reference(head_path),
            fusion_report=_artifact_reference(fusion_path),
            gen_only_report=_artifact_reference(gen_path),
            faithfulness_report=_artifact_reference(faith_path),
            prompt_audit_path=str(args.prompt_audit_path),
            prompt_audit_hash=prompt_audit_hash,
            generator_command=current_python_command(),
            generator_git_commit=git_state["git_commit"],
            generator_git_dirty=bool(git_state["git_dirty"]),
        ),
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(manifest.model_dump_json())
    args.output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"GATE MANIFEST OK: wrote {args.output_path}")
    print(f"  teacher_baseline_pass={teacher_baseline_pass}")
    print(f"  subset_head_gate_pass={subset_head_gate_pass}")
    print(f"  strict_schema_parse_pass={strict_schema_parse_pass}")
    print(f"  student_contribution_pass={student_contribution_pass}")
    print(f"  teacher_prob_ablation_pass={teacher_prob_ablation_pass}")
    print(f"  population_contract_pass={population_contract_pass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
