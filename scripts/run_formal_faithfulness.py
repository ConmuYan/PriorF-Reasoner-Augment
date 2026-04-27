from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.faithfulness import FaithfulnessInputs, FaithfulnessReport, FrozenDecisionPolicy, evaluate_faithfulness  # noqa: E402
from evidence.evidence_schema import EvidenceAblationMask  # noqa: E402
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


_MASK_CHOICES = [mask.value for mask in EvidenceAblationMask]


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Formal faithfulness evaluation driver")
    parser.add_argument("--dataset", required=True, choices=["amazon", "yelpchi"])
    parser.add_argument("--qwen-path", required=True, type=Path)
    parser.add_argument("--peft-adapter", required=True, type=Path)
    parser.add_argument("--cls-head", required=True, type=Path)
    parser.add_argument("--checkpoint-source", choices=["best_checkpoint", "final_checkpoint"], default="final_checkpoint")
    parser.add_argument("--teacher-export", required=True, type=Path)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--population", required=True, choices=["validation", "final_test", "unused_holdout", "diagnostic_holdout"])
    parser.add_argument("--formal-head-only-report", required=True, type=Path)
    parser.add_argument("--formal-fusion-report", required=True, type=Path)
    parser.add_argument("--selected-evidence-fields", nargs="+", required=True, choices=_MASK_CHOICES)
    parser.add_argument("--teacher-prob-ablation-fields", nargs="+", required=True, choices=_MASK_CHOICES)
    parser.add_argument("--minimum-formal-sample-size", type=int, required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--config-fingerprint", required=True)
    parser.add_argument("--step", type=int, default=0)
    return parser.parse_args(argv)


def _load_frozen_decision_policy(head_only_report_path: Path, fusion_report_path: Path) -> FrozenDecisionPolicy:
    head_payload = json.loads(head_only_report_path.read_text(encoding="utf-8"))
    fusion_payload = json.loads(fusion_report_path.read_text(encoding="utf-8"))
    return FrozenDecisionPolicy(
        alpha=float(fusion_payload["selection"]["optimal_alpha"]),
        threshold=float(head_payload["validation_threshold"]["selected_threshold"]),
        alpha_source=str(fusion_report_path.resolve()),
        threshold_source=str(head_only_report_path.resolve()),
    )


def main(argv=None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading canonical exports + data manifest...", flush=True)
    records_full = read_teacher_export_artifact(args.teacher_export)
    records = stratified_records(records_full, args.subset_size, args.seed)
    data_manifest = load_data_manifest(args.data_manifest)
    data_manifest_hash = file_sha256(args.data_manifest)
    print(f"       population: {len(records)} / {len(records_full)}", flush=True)
    prompt_audit_path, prompt_audit_hash, _prompt_audit_payload = write_prompt_audit_artifact(
        output_path=args.output_dir / f"prompt_audit_faithfulness_{args.dataset}_{args.population}.json",
        dataset=args.dataset,
        eval_type="faithfulness",
        entries=build_eval_prompt_audit_entries(
            records,
            data_manifest,
            population=PopulationName(args.population),
            mode=PromptMode.EVAL_HEAD,
            thinking_mode=ThinkingMode.NON_THINKING,
            location_prefix=args.population,
        ),
        extra={
            "population": args.population,
            "samples": len(records),
            "selected_evidence_fields": list(args.selected_evidence_fields),
            "teacher_prob_ablation_fields": list(args.teacher_prob_ablation_fields),
        },
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

    print("[3/5] Building HeadScoringInputs + frozen decision policy...", flush=True)
    full_inputs = build_head_scoring_inputs(
        records,
        data_manifest,
        population=PopulationName(args.population),
        checkpoint_provenance=checkpoint_provenance,
        run_id=args.run_id,
        student_visible=True,
    )
    frozen_policy = _load_frozen_decision_policy(
        args.formal_head_only_report,
        args.formal_fusion_report,
    )

    print("[4/5] Running evaluate_faithfulness...", flush=True)
    report: FaithfulnessReport = evaluate_faithfulness(
        inputs=FaithfulnessInputs(
            full_inputs=full_inputs,
            run_id=args.run_id,
            thinking_mode=ThinkingMode.NON_THINKING,
            frozen_decision_policy=frozen_policy,
            selected_evidence_fields=tuple(EvidenceAblationMask(value) for value in args.selected_evidence_fields),
            teacher_prob_ablation_fields=tuple(EvidenceAblationMask(value) for value in args.teacher_prob_ablation_fields),
            minimum_formal_sample_size=int(args.minimum_formal_sample_size),
            prompt_audit_path=str(prompt_audit_path.resolve()),
            prompt_audit_hash=prompt_audit_hash,
        ),
        model=bundle["model"],
        cls_head=bundle["cls_head"],
        tokenizer=bundle["tokenizer"],
        accelerator=None,
    )

    print("[5/5] Writing FaithfulnessReport JSON...", flush=True)
    payload = json.loads(report.model_dump_json())
    git_state = capture_git_state(REPO_ROOT)
    payload["_runtime_provenance"] = {
        "dataset": args.dataset,
        "population": args.population,
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
        "formal_head_only_report": str(args.formal_head_only_report),
        "formal_fusion_report": str(args.formal_fusion_report),
        "subset_size": len(records),
        "selected_evidence_fields": list(args.selected_evidence_fields),
        "teacher_prob_ablation_fields": list(args.teacher_prob_ablation_fields),
        "minimum_formal_sample_size": int(args.minimum_formal_sample_size),
        "prompt_audit_path": str(prompt_audit_path),
        "prompt_audit_hash": prompt_audit_hash,
        "evaluation_command": current_python_command(),
        **git_state,
        **build_subset_runtime_provenance(
            prefix="",
            population_name=args.population,
            records_full=records_full,
            selected_records=records,
            subset_requested=args.subset_size,
            teacher_export_path=args.teacher_export,
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = args.output_dir / f"faithfulness_report_{args.dataset}.json"
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"FORMAL FAITHFULNESS EVAL OK: wrote {report_path}")
    print(
        f"  mean_sufficiency={report.mean_sufficiency:.4f}  "
        f"mean_comprehensiveness={report.mean_comprehensiveness:.4f}  "
        f"teacher_prob_flip_rate={report.decision_flip_rate_teacher_prob_ablation:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
