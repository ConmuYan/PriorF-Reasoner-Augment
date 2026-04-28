from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    hash_directory_files,
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--stop-after-strict-json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-check-every", type=int, default=8)
    parser.add_argument("--parallel-shards", type=int, default=1)
    parser.add_argument("--gpu-indices", nargs="+", type=int, default=None)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=64)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--config-fingerprint", required=True)
    parser.add_argument("--step", type=int, default=0)
    return parser.parse_args(argv)


def _split_indexed_records(records: tuple[Any, ...], shard_count: int) -> list[tuple[tuple[int, Any], ...]]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    indexed = tuple(enumerate(records))
    return [tuple(indexed[shard_index::shard_count]) for shard_index in range(shard_count)]


def _generate_shard_worker(
    *,
    shard_index: int,
    indexed_records: tuple[tuple[int, Any], ...],
    data_manifest: Any,
    qwen_path: Path,
    peft_adapter: Path,
    cls_head: Path,
    gpu_index: int,
    max_new_tokens: int,
    batch_size: int,
    stop_after_strict_json: bool,
    stop_check_every: int,
    progress_every: int | None,
    output_path: Path,
    status_path: Path,
) -> None:
    try:
        bundle = load_model_bundle(
            qwen_path=qwen_path,
            peft_adapter=peft_adapter,
            cls_head_path=cls_head,
            gpu_index=int(gpu_index),
        )
        records = tuple(record for _, record in indexed_records)
        generated_texts = generate_structured_outputs(
            records,
            data_manifest,
            model=bundle["model"],
            tokenizer=bundle["tokenizer"],
            thinking_mode=ThinkingMode.NON_THINKING,
            max_new_tokens=int(max_new_tokens),
            batch_size=int(batch_size),
            stop_after_strict_json=bool(stop_after_strict_json),
            stop_check_every=int(stop_check_every),
            progress_label=f"formal-gen-shard{shard_index}-gpu{gpu_index}",
            progress_every=progress_every,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for (record_index, record), generated_text in zip(indexed_records, generated_texts, strict=True):
                handle.write(json.dumps({
                    "record_index": int(record_index),
                    "node_id": int(record.node_id),
                    "generated_text": generated_text,
                }, ensure_ascii=False) + "\n")
        status_path.write_text(json.dumps({
            "ok": True,
            "shard_index": int(shard_index),
            "gpu_index": int(gpu_index),
            "records": len(indexed_records),
            "output_path": str(output_path),
        }, indent=2) + "\n", encoding="utf-8")
    except BaseException as exc:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps({
            "ok": False,
            "shard_index": int(shard_index),
            "gpu_index": int(gpu_index),
            "error": repr(exc),
        }, indent=2) + "\n", encoding="utf-8")
        raise


def _generate_parallel_structured_outputs(
    *,
    records: tuple[Any, ...],
    data_manifest: Any,
    qwen_path: Path,
    peft_adapter: Path,
    cls_head: Path,
    gpu_indices: list[int],
    parallel_shards: int,
    max_new_tokens: int,
    batch_size: int,
    stop_after_strict_json: bool,
    stop_check_every: int,
    progress_every: int | None,
    output_dir: Path,
) -> tuple[str, ...]:
    if parallel_shards < 1:
        raise ValueError("--parallel-shards must be >= 1")
    if len(gpu_indices) < parallel_shards:
        raise ValueError("--gpu-indices must provide at least --parallel-shards entries")
    shard_records = _split_indexed_records(records, parallel_shards)
    shard_dir = output_dir / "generation_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    processes = []
    for shard_index, indexed_records in enumerate(shard_records):
        shard_output_path = shard_dir / f"shard_{shard_index:02d}.jsonl"
        shard_status_path = shard_dir / f"shard_{shard_index:02d}.status.json"
        process = ctx.Process(
            target=_generate_shard_worker,
            kwargs={
                "shard_index": shard_index,
                "indexed_records": indexed_records,
                "data_manifest": data_manifest,
                "qwen_path": qwen_path,
                "peft_adapter": peft_adapter,
                "cls_head": cls_head,
                "gpu_index": int(gpu_indices[shard_index]),
                "max_new_tokens": int(max_new_tokens),
                "batch_size": int(batch_size),
                "stop_after_strict_json": bool(stop_after_strict_json),
                "stop_check_every": int(stop_check_every),
                "progress_every": progress_every,
                "output_path": shard_output_path,
                "status_path": shard_status_path,
            },
        )
        process.start()
        processes.append((process, shard_output_path, shard_status_path))

    failures = []
    for process, _output_path, status_path in processes:
        process.join()
        if process.exitcode != 0:
            status_payload = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
            failures.append({"exitcode": process.exitcode, "status": status_payload})
    if failures:
        raise RuntimeError(f"parallel generation shard failure(s): {failures}")

    generated_by_index: dict[int, str] = {}
    for _process, shard_output_path, _status_path in processes:
        with shard_output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                generated_by_index[int(row["record_index"])] = str(row["generated_text"])
    missing = [index for index in range(len(records)) if index not in generated_by_index]
    if missing:
        raise RuntimeError(f"parallel generation missing record indices: {missing[:10]}")
    return tuple(generated_by_index[index] for index in range(len(records)))


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

    gpu_indices = [int(value) for value in (args.gpu_indices or [args.gpu_index])]
    if int(args.parallel_shards) > 1:
        print(
            f"[2/5] Launching {int(args.parallel_shards)} generation shards on GPUs {gpu_indices}...",
            flush=True,
        )
        print("[3/5] Generating structured outputs on canonical eval_gen prompts...", flush=True)
        generated_texts = _generate_parallel_structured_outputs(
            records=records,
            data_manifest=data_manifest,
            qwen_path=args.qwen_path,
            peft_adapter=args.peft_adapter,
            cls_head=args.cls_head,
            gpu_indices=gpu_indices,
            parallel_shards=int(args.parallel_shards),
            max_new_tokens=int(args.max_new_tokens),
            batch_size=int(args.batch_size),
            stop_after_strict_json=bool(args.stop_after_strict_json),
            stop_check_every=int(args.stop_check_every),
            progress_every=args.progress_every,
            output_dir=args.output_dir,
        )
        bundle_hashes = {
            "cls_head_sha256": file_sha256(args.cls_head),
            "peft_adapter_sha256": hash_directory_files(args.peft_adapter),
        }
    else:
        print("[2/5] Loading Qwen3 backbone + PEFT adapter...", flush=True)
        bundle = load_model_bundle(
            qwen_path=args.qwen_path,
            peft_adapter=args.peft_adapter,
            cls_head_path=args.cls_head,
            gpu_index=gpu_indices[0],
        )
        print("[3/5] Generating structured outputs on canonical eval_gen prompts...", flush=True)
        generated_texts = generate_structured_outputs(
            records,
            data_manifest,
            model=bundle["model"],
            tokenizer=bundle["tokenizer"],
            thinking_mode=ThinkingMode.NON_THINKING,
            max_new_tokens=int(args.max_new_tokens),
            batch_size=int(args.batch_size),
            stop_after_strict_json=bool(args.stop_after_strict_json),
            stop_check_every=int(args.stop_check_every),
            progress_label="formal-gen",
            progress_every=args.progress_every,
        )
        bundle_hashes = {
            "cls_head_sha256": bundle["cls_head_sha256"],
            "peft_adapter_sha256": bundle["peft_adapter_sha256"],
        }

    raw_generations_path = args.output_dir / f"raw_generations_{args.dataset}_{args.population}.jsonl"
    with raw_generations_path.open("w", encoding="utf-8") as handle:
        for record, generated_text in zip(records, generated_texts, strict=True):
            handle.write(json.dumps({
                "node_id": int(record.node_id),
                "ground_truth_label": int(record.ground_truth_label),
                "generated_text": generated_text,
            }, ensure_ascii=False) + "\n")
    raw_generations_hash = file_sha256(raw_generations_path)

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
        "cls_head_sha256": bundle_hashes["cls_head_sha256"],
        "peft_adapter_sha256": bundle_hashes["peft_adapter_sha256"],
        "data_manifest": str(args.data_manifest),
        "data_manifest_sha256": data_manifest_hash,
        "subset_size": len(records),
        "max_new_tokens": int(args.max_new_tokens),
        "batch_size": int(args.batch_size),
        "stop_after_strict_json": bool(args.stop_after_strict_json),
        "stop_check_every": int(args.stop_check_every),
        "parallel_shards": int(args.parallel_shards),
        "gpu_indices": gpu_indices,
        "raw_generations_path": str(raw_generations_path),
        "raw_generations_sha256": raw_generations_hash,
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
