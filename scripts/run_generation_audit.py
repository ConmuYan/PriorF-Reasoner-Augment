from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_gen_only import GenOnlyEvalInputs, GenOnlyEvalSample, _try_parse_normalized, _try_parse_strict, evaluate_gen_only  # noqa: E402
from evidence.evidence_schema import build_student_evidence_card  # noqa: E402
from graph_data.manifests import load_data_manifest  # noqa: E402
from priorf_teacher.export_pipeline import read_teacher_export_artifact  # noqa: E402
from priorf_teacher.schema import DatasetName, GraphRegime, PopulationName  # noqa: E402
from scripts._formal_eval_helpers import (  # noqa: E402
    build_subset_runtime_provenance,
    capture_git_state,
    current_python_command,
    file_sha256,
    generate_structured_outputs,
    hash_directory_files,
    stratified_records,
)

_AUDIT_SCHEMA_VERSION: Final[str] = "generation_audit/v1"
_STRUCTURAL_TOKENS: Final[tuple[str, ...]] = (
    "hsd",
    "branch_gap",
    "discrepancy",
    "route_hint",
    "relation",
    "neighbor",
)


def _label_counts(records: tuple[Any, ...]) -> tuple[int, int]:
    positive_count = sum(1 for record in records if int(record.ground_truth_label) == 1)
    negative_count = len(records) - positive_count
    return positive_count, negative_count


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generation audit driver with raw-generation export")
    parser.add_argument("--dataset", required=True, choices=["amazon", "yelpchi"])
    parser.add_argument("--qwen-path", required=True, type=Path)
    parser.add_argument("--peft-adapter", required=True, type=Path)
    parser.add_argument("--teacher-export", required=True, type=Path)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--population", required=True, choices=["validation", "final_test", "unused_holdout", "diagnostic_holdout"])
    parser.add_argument("--checkpoint-source", choices=["best_checkpoint", "final_checkpoint"], default="final_checkpoint")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subset-size", type=int, default=128)
    parser.add_argument("--review-sample-size", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def _contains_structural_signal(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in _STRUCTURAL_TOKENS)


def _load_generation_model(*, qwen_path: Path, peft_adapter: Path, gpu_index: int) -> dict[str, Any]:
    device_str = "cpu" if gpu_index < 0 else f"cuda:{gpu_index}"
    torch_dtype = torch.bfloat16 if device_str != "cpu" else torch.float32

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(qwen_path))
    base_model = AutoModelForCausalLM.from_pretrained(
        str(qwen_path),
        dtype=torch_dtype,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, str(peft_adapter))
    model.to(device_str)
    model.eval()
    return {
        "device": device_str,
        "model": model,
        "tokenizer": tokenizer,
        "peft_adapter_sha256": hash_directory_files(peft_adapter),
    }


def _semantic_usefulness_summary(review_samples: list[dict[str, Any]], *, graph_regime: str) -> dict[str, Any]:
    parsed_samples = [sample for sample in review_samples if sample["parsed_output"] is not None]
    parsed_count = len(parsed_samples)
    if parsed_count == 0:
        return {
            "review_sample_count": len(review_samples),
            "parsed_review_sample_count": 0,
            "rationale_structural_signal_rate": 0.0,
            "evidence_structural_signal_rate": 0.0,
            "pattern_hint_mentions_graph_regime_rate": 0.0,
        }
    return {
        "review_sample_count": len(review_samples),
        "parsed_review_sample_count": parsed_count,
        "rationale_structural_signal_rate": sum(
            1 for sample in parsed_samples if sample["semantic_checks"]["rationale_has_structural_signal"]
        ) / parsed_count,
        "evidence_structural_signal_rate": sum(
            1 for sample in parsed_samples if sample["semantic_checks"]["evidence_has_structural_signal"]
        ) / parsed_count,
        "pattern_hint_mentions_graph_regime_rate": sum(
            1 for sample in parsed_samples if sample["semantic_checks"]["pattern_hint_mentions_graph_regime"]
        ) / parsed_count,
        "graph_regime": graph_regime,
    }


def _build_review_sample(
    *,
    record: Any,
    generated_text: str,
    graph_regime: GraphRegime,
    data_manifest: Any,
) -> dict[str, Any]:
    strict_output = _try_parse_strict(generated_text)
    normalized_output = _try_parse_normalized(generated_text)
    parsed_output = strict_output or normalized_output
    evidence_card = build_student_evidence_card(teacher_record=record, data_manifest=data_manifest)
    if strict_output is not None:
        evidence_items = strict_output.evidence
        rationale = strict_output.rationale
        pattern_hint = strict_output.pattern_hint
    elif normalized_output is not None:
        evidence_items = normalized_output.evidence
        rationale = normalized_output.rationale
        pattern_hint = normalized_output.pattern_hint
    else:
        evidence_items = ()
        rationale = ""
        pattern_hint = ""
    return {
        "node_id": int(record.node_id),
        "ground_truth_label": int(record.ground_truth_label),
        "strict_parse_succeeded": strict_output is not None,
        "normalized_parse_succeeded": normalized_output is not None,
        "generated_text": generated_text,
        "parsed_output": None if parsed_output is None else parsed_output.model_dump(mode="json"),
        "semantic_checks": {
            "rationale_has_structural_signal": _contains_structural_signal(rationale),
            "evidence_has_structural_signal": any(_contains_structural_signal(item) for item in evidence_items),
            "pattern_hint_mentions_graph_regime": graph_regime.value in pattern_hint,
        },
        "evidence_card_excerpt": {
            "discrepancy_severity": evidence_card.discrepancy_summary.discrepancy_severity,
            "route_hint": evidence_card.discrepancy_summary.route_hint,
            "active_relations": evidence_card.relation_profile.active_relations,
            "total_neighbors": evidence_card.neighbor_summary.total_neighbors,
            "evidence_card_projection": evidence_card.evidence_card_projection,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading canonical export + data manifest...", flush=True)
    records_full = read_teacher_export_artifact(args.teacher_export)
    records = stratified_records(records_full, args.subset_size, args.seed)
    data_manifest = load_data_manifest(args.data_manifest)
    dataset_enum = DatasetName(args.dataset)
    population_enum = PopulationName(args.population)
    graph_regime = GraphRegime(records[0].graph_regime)
    if any(record.dataset_name != dataset_enum for record in records):
        raise ValueError("teacher export dataset_name must match --dataset")
    if any(record.population_name != population_enum for record in records):
        raise ValueError("teacher export population_name must match --population")
    positive_count, negative_count = _label_counts(records)
    print(f"       population: {len(records)} / {len(records_full)}", flush=True)
    print(
        f"       graph_regime={graph_regime.value} positives={positive_count} negatives={negative_count} "
        f"max_new_tokens={int(args.max_new_tokens)}",
        flush=True,
    )

    print("[2/6] Loading backbone + PEFT adapter...", flush=True)
    bundle = _load_generation_model(
        qwen_path=args.qwen_path,
        peft_adapter=args.peft_adapter,
        gpu_index=args.gpu_index,
    )

    print("[3/6] Generating raw structured outputs...", flush=True)
    generated_texts = generate_structured_outputs(
        records,
        data_manifest,
        model=bundle["model"],
        tokenizer=bundle["tokenizer"],
        thinking_mode="non_thinking",
        max_new_tokens=int(args.max_new_tokens),
        progress_label="audit-gen",
    )

    print("[4/6] Computing gen-only syntax and discriminative summaries...", flush=True)
    samples = tuple(
        GenOnlyEvalSample(
            evidence_card=build_student_evidence_card(teacher_record=record, data_manifest=data_manifest),
            generated_text=generated_text,
            ground_truth_label=int(record.ground_truth_label),
            node_id=int(record.node_id),
        )
        for record, generated_text in zip(records, generated_texts, strict=True)
    )
    report = evaluate_gen_only(
        inputs=GenOnlyEvalInputs(
            samples=samples,
            dataset_name=dataset_enum,
            population_name=population_enum,
            graph_regime=graph_regime,
            run_id=args.run_id,
        )
    )

    print("[5/6] Writing raw generations + audit report...", flush=True)
    raw_path = args.output_dir / f"raw_generations_{args.dataset}_{args.population}.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for record, generated_text in zip(records, generated_texts, strict=True):
            handle.write(json.dumps({
                "node_id": int(record.node_id),
                "ground_truth_label": int(record.ground_truth_label),
                "generated_text": generated_text,
            }, ensure_ascii=False) + "\n")

    review_limit = max(1, min(int(args.review_sample_size), len(records)))
    review_samples = [
        _build_review_sample(
            record=record,
            generated_text=generated_text,
            graph_regime=graph_regime,
            data_manifest=data_manifest,
        )
        for record, generated_text in zip(records[:review_limit], generated_texts[:review_limit], strict=True)
    ]
    payload = {
        "schema_version": _AUDIT_SCHEMA_VERSION,
        "dataset_name": dataset_enum.value,
        "population_name": population_enum.value,
        "graph_regime": graph_regime.value,
        "n_total": int(report.n_total),
        "syntax": {
            "headline_metric_name": report.headline_metric_name,
            "strict_schema_parse_rate": report.strict_schema_parse_rate,
            "normalized_parse_rate": report.normalized_parse_rate,
            "strict_parse_failure_count": report.strict_parse_failure_count,
            "strict_parse_failure_node_ids": list(report.strict_parse_failure_node_ids),
        },
        "semantic_usefulness": _semantic_usefulness_summary(review_samples, graph_regime=graph_regime.value),
        "discriminative_power": {
            "auroc": report.auroc,
            "auprc": report.auprc,
            "brier_score": report.brier_score,
            "note": "gen-only discriminative power comes from strict parsing with penalty_mode on parse failure",
        },
        "review_samples": review_samples,
        "raw_generations_path": str(raw_path),
        "raw_generations_sha256": file_sha256(raw_path),
        "_runtime_provenance": {
            "dataset": args.dataset,
            "population": args.population,
            "checkpoint_source": args.checkpoint_source,
            "run_id": args.run_id,
            "seed": int(args.seed),
            "qwen_path": str(args.qwen_path),
            "peft_adapter": str(args.peft_adapter),
            "peft_adapter_sha256": bundle["peft_adapter_sha256"],
            "data_manifest": str(args.data_manifest),
            "data_manifest_sha256": file_sha256(args.data_manifest),
            "audit_command": current_python_command(),
            **capture_git_state(REPO_ROOT),
            **build_subset_runtime_provenance(
                prefix="",
                population_name=args.population,
                records_full=records_full,
                selected_records=records,
                subset_requested=args.subset_size,
                teacher_export_path=args.teacher_export,
            ),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    audit_path = args.output_dir / f"generation_audit_{args.dataset}_{args.population}.json"
    audit_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("[6/6] Done.", flush=True)
    print(f"GENERATION AUDIT OK: wrote {audit_path}")
    print(f"  strict_parse={report.strict_schema_parse_rate:.4f}  normalized_parse={report.normalized_parse_rate:.4f}")
    print(f"  auroc={report.auroc}  auprc={report.auprc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
