from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_gen_only import GenOnlyEvalInputs, GenOnlyEvalReport, GenOnlyEvalSample, evaluate_gen_only  # noqa: E402
from evidence.evidence_schema import build_student_evidence_card  # noqa: E402
from evidence.prompt_builder import PromptMode, ThinkingMode  # noqa: E402
from graph_data.manifests import load_data_manifest  # noqa: E402
from priorf_teacher.export_pipeline import read_teacher_export_artifact  # noqa: E402
from priorf_teacher.schema import PopulationName  # noqa: E402
from scripts._formal_eval_helpers import (  # noqa: E402
    build_eval_prompt_audit_entries,
    build_subset_runtime_provenance,
    capture_git_state,
    current_python_command,
    file_sha256,
    generate_structured_outputs,
    load_model_bundle,
    stratified_records,
    write_prompt_audit_artifact,
)


def _label_counts(records) -> tuple[int, int]:
    positive_count = sum(1 for record in records if int(record.ground_truth_label) == 1)
    negative_count = len(records) - positive_count
    return positive_count, negative_count


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Formal generation-only evaluation driver")
    parser.add_argument("--dataset", required=True, choices=["amazon", "yelpchi"])
    parser.add_argument("--qwen-path", required=True, type=Path)
    parser.add_argument("--peft-adapter", required=True, type=Path)
    parser.add_argument("--cls-head", required=True, type=Path)
    parser.add_argument("--checkpoint-source", choices=["best_checkpoint", "final_checkpoint"], default="final_checkpoint")
    parser.add_argument("--teacher-export", required=True, type=Path)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--population", required=True, choices=["validation", "final_test", "unused_holdout", "diagnostic_holdout"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--config-fingerprint", required=True)
    parser.add_argument("--step", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading canonical exports + data manifest...", flush=True)
    records_full = read_teacher_export_artifact(args.teacher_export)
    records = stratified_records(records_full, args.subset_size, args.seed)
    data_manifest = load_data_manifest(args.data_manifest)
    data_manifest_hash = file_sha256(args.data_manifest)
    positive_count, negative_count = _label_counts(records)
    print(f"       population: {len(records)} / {len(records_full)}", flush=True)
    print(
        f"       positives={positive_count} negatives={negative_count} max_new_tokens={int(args.max_new_tokens)}",
        flush=True,
    )
    prompt_audit_path, prompt_audit_hash, _prompt_audit_payload = write_prompt_audit_artifact(
        output_path=args.output_dir / f"prompt_audit_gen_only_{args.dataset}_{args.population}.json",
        dataset=args.dataset,
        eval_type="gen_only",
        entries=build_eval_prompt_audit_entries(
            records,
            data_manifest,
            population=PopulationName(args.population),
            mode=PromptMode.EVAL_GEN,
            thinking_mode=ThinkingMode.NON_THINKING,
            include_assistant_target=False,
            location_prefix=args.population,
        ),
        extra={"population": args.population, "samples": len(records)},
    )


    print("[2/5] Loading Qwen3 backbone + PEFT adapter...", flush=True)
    bundle = load_model_bundle(
        qwen_path=args.qwen_path,
        peft_adapter=args.peft_adapter,
        cls_head_path=args.cls_head,
        gpu_index=args.gpu_index,
    )

    print("[3/5] Generating structured outputs on canonical eval_gen prompts...", flush=True)
    generated_texts = generate_structured_outputs(
        records,
        data_manifest,
        model=bundle["model"],
        tokenizer=bundle["tokenizer"],
        thinking_mode=ThinkingMode.NON_THINKING,
        max_new_tokens=int(args.max_new_tokens),
        progress_label="formal-gen",
    )

    print("[4/5] Running evaluate_gen_only...", flush=True)
    samples = []
    for record, generated_text in zip(records, generated_texts, strict=True):
        samples.append(
            GenOnlyEvalSample(
                evidence_card=build_student_evidence_card(teacher_record=record, data_manifest=data_manifest),
                generated_text=generated_text,
                ground_truth_label=int(record.ground_truth_label),
                node_id=int(record.node_id),
            )
        )
    report: GenOnlyEvalReport = evaluate_gen_only(
        inputs=GenOnlyEvalInputs(
            samples=tuple(samples),
            dataset_name=records[0].dataset_name,
            population_name=PopulationName(args.population),
            graph_regime=records[0].graph_regime,
            run_id=args.run_id,
            prompt_audit_path=str(prompt_audit_path.resolve()),
            prompt_audit_hash=prompt_audit_hash,
        )
    )

    print("[5/5] Writing GenOnlyEvalReport JSON...", flush=True)
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
        "subset_size": len(records),
        "max_new_tokens": int(args.max_new_tokens),
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
    report_path = args.output_dir / f"formal_gen_only_report_{args.dataset}.json"
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"FORMAL GEN-ONLY EVAL OK: wrote {report_path}")
    print(
        f"  strict_parse={report.strict_schema_parse_rate:.4f}  "
        f"normalized_parse={report.normalized_parse_rate:.4f}  "
        f"auroc={report.auroc}  auprc={report.auprc}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
