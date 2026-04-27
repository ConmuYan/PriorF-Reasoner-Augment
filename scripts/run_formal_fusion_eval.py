from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_fusion import FusionEvalConfig, FusionEvalReport, FusionPopulationInputs, run_formal_fusion_eval  # noqa: E402
from eval.head_scoring import score_head  # noqa: E402
from evidence.prompt_builder import PromptMode, ThinkingMode  # noqa: E402
from graph_data.manifests import load_data_manifest  # noqa: E402
from priorf_teacher.export_pipeline import read_teacher_export_artifact  # noqa: E402
from priorf_teacher.schema import PopulationName  # noqa: E402
from scripts._formal_eval_helpers import (  # noqa: E402
    build_checkpoint_provenance,
    build_eval_prompt_audit_entries,
    build_head_scoring_inputs,
    build_subset_runtime_provenance,
    capture_git_state,
    current_python_command,
    file_sha256,
    load_model_bundle,
    stratified_records,
    write_prompt_audit_artifact,
)



def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Formal fusion evaluation driver")
    parser.add_argument("--dataset", required=True, choices=["amazon", "yelpchi"])
    parser.add_argument("--qwen-path", required=True, type=Path)
    parser.add_argument("--peft-adapter", required=True, type=Path)
    parser.add_argument("--cls-head", required=True, type=Path)
    parser.add_argument("--checkpoint-source", choices=["best_checkpoint", "final_checkpoint"], default="final_checkpoint")
    parser.add_argument("--teacher-export-validation", required=True, type=Path)
    parser.add_argument("--teacher-export-final-test", required=True, type=Path)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-subset", type=int, default=256)
    parser.add_argument("--final-test-subset", type=int, default=256)
    parser.add_argument("--alpha-candidates", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--primary-metric", choices=["auprc", "auroc", "f1_at_frozen_threshold"], default="auprc")
    parser.add_argument("--teacher-degradation-tolerance", type=float, default=0.0)
    parser.add_argument("--min-student-alpha", type=float, default=0.05)
    parser.add_argument("--frozen-threshold", type=float, default=None)
    parser.add_argument("--tie-breaker", choices=["smaller_alpha", "larger_alpha"], default="smaller_alpha")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--config-fingerprint", required=True)
    parser.add_argument("--step", type=int, default=0)
    return parser.parse_args(argv)



def _teacher_payload(records) -> tuple[tuple[float, ...], tuple[int, ...]]:
    return (
        tuple(float(record.teacher_prob) for record in records),
        tuple(int(record.node_id) for record in records),
    )



def main(argv=None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading canonical exports + data manifest...", flush=True)
    val_records_full = read_teacher_export_artifact(args.teacher_export_validation)
    test_records_full = read_teacher_export_artifact(args.teacher_export_final_test)
    val_records = stratified_records(val_records_full, args.validation_subset, args.seed)
    test_records = stratified_records(test_records_full, args.final_test_subset, args.seed)
    data_manifest = load_data_manifest(args.data_manifest)
    data_manifest_hash = file_sha256(args.data_manifest)
    print(
        f"       validation: {len(val_records)} / {len(val_records_full)}  "
        f"final_test: {len(test_records)} / {len(test_records_full)}",
        flush=True,
    )
    audit_entries = (
        *build_eval_prompt_audit_entries(
            val_records,
            data_manifest,
            population=PopulationName.VALIDATION,
            mode=PromptMode.EVAL_HEAD,
            thinking_mode=ThinkingMode.NON_THINKING,
            location_prefix="validation",
        ),
        *build_eval_prompt_audit_entries(
            test_records,
            data_manifest,
            population=PopulationName.FINAL_TEST,
            mode=PromptMode.EVAL_HEAD,
            thinking_mode=ThinkingMode.NON_THINKING,
            location_prefix="final_test",
        ),
    )
    prompt_audit_path, prompt_audit_hash, _prompt_audit_payload = write_prompt_audit_artifact(
        output_path=args.output_dir / f"prompt_audit_fusion_{args.dataset}.json",
        dataset=args.dataset,
        eval_type="fusion",
        entries=audit_entries,
        extra={"validation_samples": len(val_records), "final_test_samples": len(test_records)},
    )


    print("[2/5] Loading Qwen3 backbone + PEFT adapter + cls_head...", flush=True)
    bundle = load_model_bundle(
        qwen_path=args.qwen_path,
        peft_adapter=args.peft_adapter,
        cls_head_path=args.cls_head,
        gpu_index=args.gpu_index,
    )
    checkpoint_provenance = build_checkpoint_provenance(
        cls_head_path=args.cls_head,
        step=int(args.step),
    )

    print("[3/5] Running score_head on validation and final_test populations...", flush=True)
    validation_head_inputs = build_head_scoring_inputs(
        val_records,
        data_manifest,
        population=PopulationName.VALIDATION,
        checkpoint_provenance=checkpoint_provenance,
        run_id=args.run_id,
    )
    report_head_inputs = build_head_scoring_inputs(
        test_records,
        data_manifest,
        population=PopulationName.FINAL_TEST,
        checkpoint_provenance=checkpoint_provenance,
        run_id=args.run_id,
    )
    validation_head_report = score_head(
        inputs=validation_head_inputs,
        model=bundle["model"],
        cls_head=bundle["cls_head"],
        tokenizer=bundle["tokenizer"],
        thinking_mode=ThinkingMode.NON_THINKING,
        prompt_audit_path=str(prompt_audit_path.resolve()),
        prompt_audit_hash=prompt_audit_hash,
        accelerator=None,
    )
    report_head_report = score_head(
        inputs=report_head_inputs,
        model=bundle["model"],
        cls_head=bundle["cls_head"],
        tokenizer=bundle["tokenizer"],
        thinking_mode=ThinkingMode.NON_THINKING,
        prompt_audit_path=str(prompt_audit_path.resolve()),
        prompt_audit_hash=prompt_audit_hash,
        accelerator=None,
    )

    print("[4/5] Running run_formal_fusion_eval...", flush=True)
    val_teacher_probs, val_teacher_node_ids = _teacher_payload(val_records)
    test_teacher_probs, test_teacher_node_ids = _teacher_payload(test_records)
    report: FusionEvalReport = run_formal_fusion_eval(
        validation_inputs=FusionPopulationInputs(
            head_report=validation_head_report,
            teacher_probs=val_teacher_probs,
            teacher_node_ids=val_teacher_node_ids,
        ),
        report_inputs=FusionPopulationInputs(
            head_report=report_head_report,
            teacher_probs=test_teacher_probs,
            teacher_node_ids=test_teacher_node_ids,
        ),
        config=FusionEvalConfig(
            alpha_candidates=tuple(float(value) for value in args.alpha_candidates),
            primary_metric=args.primary_metric,
            teacher_degradation_tolerance=float(args.teacher_degradation_tolerance),
            min_student_alpha=float(args.min_student_alpha),
            frozen_threshold=args.frozen_threshold,
            tie_breaker=args.tie_breaker,
        ),
        run_id=args.run_id,
        prompt_audit_path=str(prompt_audit_path.resolve()),
        prompt_audit_hash=prompt_audit_hash,
    )

    print("[5/5] Writing FusionEvalReport JSON...", flush=True)
    payload = json.loads(report.model_dump_json())
    git_state = capture_git_state(REPO_ROOT)
    payload["_runtime_provenance"] = {
        "dataset": args.dataset,
        "checkpoint_source": args.checkpoint_source,
        "checkpoint_type": args.checkpoint_source,
        "run_id": args.run_id,
        "commit": args.commit,
        "training_commit": args.commit,
        "config_fingerprint": args.config_fingerprint,
        "step": int(args.step),
        "seed": int(args.seed),
        "qwen_path": str(args.qwen_path),
        "peft_adapter": str(args.peft_adapter),
        "cls_head": str(args.cls_head),
        "cls_head_sha256": bundle["cls_head_sha256"],
        "peft_adapter_sha256": bundle["peft_adapter_sha256"],
        "data_manifest": str(args.data_manifest),
        "data_manifest_sha256": data_manifest_hash,
        "validation_subset": len(val_records),
        "final_test_subset": len(test_records),
        "alpha_candidates": [float(value) for value in args.alpha_candidates],
        "primary_metric": args.primary_metric,
        "teacher_degradation_tolerance": float(args.teacher_degradation_tolerance),
        "min_student_alpha": float(args.min_student_alpha),
        "frozen_threshold": args.frozen_threshold,
        "prompt_audit_path": str(prompt_audit_path),
        "prompt_audit_hash": prompt_audit_hash,
        "evaluation_command": current_python_command(),
        **git_state,
        **build_subset_runtime_provenance(
            prefix="validation",
            population_name=PopulationName.VALIDATION.value,
            records_full=val_records_full,
            selected_records=val_records,
            subset_requested=args.validation_subset,
            teacher_export_path=args.teacher_export_validation,
        ),
        **build_subset_runtime_provenance(
            prefix="final_test",
            population_name=PopulationName.FINAL_TEST.value,
            records_full=test_records_full,
            selected_records=test_records,
            subset_requested=args.final_test_subset,
            teacher_export_path=args.teacher_export_final_test,
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = args.output_dir / f"formal_fusion_report_{args.dataset}.json"
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"FORMAL FUSION EVAL OK: wrote {report_path}")
    print(
        f"  alpha={report.selection.optimal_alpha:.4f}  "
        f"student_contribution_pass={report.student_contribution_pass}  "
        f"report_fusion_auprc={report.report_metrics.fusion.auprc}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
