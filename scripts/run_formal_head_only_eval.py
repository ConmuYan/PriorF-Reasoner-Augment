"""Formal head-only evaluation driver.

Wraps ``eval.eval_head_only.run_formal_head_only_eval`` for production
runs over validation + final_test populations produced by the canonical
pipeline (``scripts/generate_teacher_exports.py``).  The CLI forbids
re-scoring or re-training: it consumes the already-produced PEFT adapter
and ``cls_head.pt`` written by ``scripts/run_stage2_train.py`` and runs
``score_head`` twice (once per population).

Outputs:

* ``<output-dir>/formal_head_only_report_<dataset>.json`` - full
  ``FormalHeadOnlyReport`` including validation-frozen threshold,
  calibration summary, and headline metrics on the report population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_head_only import (  # noqa: E402
    FormalHeadOnlyCheckpointBundle,
    FormalHeadOnlyCheckpointComponent,
    FormalHeadOnlyReport,
    run_formal_head_only_eval,
)
from eval.head_scoring import CheckpointProvenance, HeadScoringInputs, HeadScoringSample  # noqa: E402
from evidence.evidence_schema import build_student_evidence_card  # noqa: E402
from evidence.prompt_builder import PromptMode, ThinkingMode  # noqa: E402
from graph_data.manifests import PopulationMetadata, load_data_manifest  # noqa: E402
from priorf_teacher.export_pipeline import read_teacher_export_artifact  # noqa: E402
from priorf_teacher.schema import GraphRegime, PopulationName  # noqa: E402
from scripts._formal_eval_helpers import (  # noqa: E402
    build_eval_prompt_audit_entries,
    build_subset_runtime_provenance,
    capture_git_state,
    current_python_command,
    file_sha256,
    stratified_records,
    write_prompt_audit_artifact,
)


class _LinearClsHead(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, 1)

    def forward(self, hidden_prompt_only: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_prompt_only.to(self.linear.weight.dtype)).squeeze(-1)


def _build_inputs(records, data_manifest, *, population, checkpoint_provenance, run_id: str):
    samples = []
    for record in records:
        card = build_student_evidence_card(teacher_record=record, data_manifest=data_manifest)
        samples.append(
            HeadScoringSample(
                evidence_card=card,
                ground_truth_label=int(record.ground_truth_label),
                node_id=int(record.node_id),
            )
        )
    return HeadScoringInputs(
        samples=tuple(samples),
        dataset_name=records[0].dataset_name,
        population_name=population,
        graph_regime=records[0].graph_regime,
        checkpoint_provenance=checkpoint_provenance,
        run_id=run_id,
    )


def _find_population(data_manifest, population_name: PopulationName) -> PopulationMetadata:
    for population in data_manifest.populations:
        if population.population_name == population_name.value:
            return population
    raise ValueError(f"population {population_name.value!r} missing from data manifest")


def _resolve_shared_graph_regime(data_manifest, *record_groups) -> GraphRegime:
    observed_regimes: set[GraphRegime] = {GraphRegime(data_manifest.graph_regime)}
    for records in record_groups:
        observed_regimes.update(GraphRegime(record.graph_regime) for record in records)
    if len(observed_regimes) != 1:
        regimes = ", ".join(sorted(regime.value for regime in observed_regimes))
        raise ValueError(f"graph_regime mismatch across data manifest / teacher exports: {regimes}")
    return next(iter(observed_regimes))


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, choices=["amazon", "yelpchi"])
    p.add_argument("--qwen-path", required=True, type=Path)
    p.add_argument("--peft-adapter", required=True, type=Path)
    p.add_argument("--cls-head", required=True, type=Path)
    p.add_argument("--checkpoint-source",
                   choices=["best_checkpoint", "final_checkpoint"],
                   default="final_checkpoint")
    p.add_argument("--teacher-export-validation", required=True, type=Path)
    p.add_argument("--teacher-export-final-test", required=True, type=Path)
    p.add_argument("--data-manifest", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--validation-subset", type=int, default=256)
    p.add_argument("--final-test-subset", type=int, default=256)
    p.add_argument("--calibration-bins", type=int, default=10)
    p.add_argument("--apply-temperature-scaling", action="store_true")
    p.add_argument("--include-oracle-diagnostics", action="store_true")
    p.add_argument("--run-id", required=True)
    p.add_argument("--commit", required=True, help="Git commit sha (40 hex) of the training run.")
    p.add_argument("--config-fingerprint", required=True, help="Same fingerprint used at training time.")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:  # noqa: C901
    args = _parse_args(argv)
    device_str = "cpu" if args.gpu_index < 0 else f"cuda:{args.gpu_index}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading canonical exports + data manifest...", flush=True)
    val_records_full = read_teacher_export_artifact(args.teacher_export_validation)
    test_records_full = read_teacher_export_artifact(args.teacher_export_final_test)
    val_records = stratified_records(val_records_full, args.validation_subset, args.seed)
    test_records = stratified_records(test_records_full, args.final_test_subset, args.seed)
    data_manifest = load_data_manifest(args.data_manifest)
    data_manifest_hash = file_sha256(args.data_manifest)
    shared_graph_regime = _resolve_shared_graph_regime(data_manifest, val_records_full, test_records_full)
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
        output_path=args.output_dir / f"prompt_audit_head_only_{args.dataset}.json",
        dataset=args.dataset,
        eval_type="head_only",
        entries=audit_entries,
        extra={"validation_samples": len(val_records), "final_test_samples": len(test_records)},
    )


    print(f"[2/6] Loading Qwen3 backbone on {device_str}: {args.qwen_path}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(str(args.qwen_path))
    base_model = AutoModelForCausalLM.from_pretrained(
        str(args.qwen_path),
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    print(f"[3/6] Loading PEFT adapter + cls_head: {args.peft_adapter}", flush=True)
    from peft import PeftModel  # noqa: E402

    model = PeftModel.from_pretrained(base_model, str(args.peft_adapter))
    model.to(device_str)
    model.eval()
    hidden_size = int(base_model.config.hidden_size)
    cls_head = _LinearClsHead(hidden_size=hidden_size)
    cls_head.load_state_dict(torch.load(args.cls_head, map_location="cpu", weights_only=True))
    cls_head.to(device_str).to(torch.bfloat16)
    cls_head.eval()
    cls_head_sha256 = file_sha256(args.cls_head)

    # Hash the adapter directory deterministically so checkpoint-bundle
    # content hashes are reproducible.
    adapter_files = sorted(args.peft_adapter.rglob("*"))
    adapter_hasher = hashlib.sha256()
    for f in adapter_files:
        if f.is_file():
            adapter_hasher.update(str(f.relative_to(args.peft_adapter)).encode())
            adapter_hasher.update(f.read_bytes())
    adapter_sha256 = adapter_hasher.hexdigest()

    print("[4/6] Building formal checkpoint bundle + HeadScoringInputs...", flush=True)
    common_checkpoint_kwargs: dict[str, Any] = dict(
        run_id=args.run_id,
        checkpoint_step=int(args.step),
        commit=args.commit,
        config_fingerprint=args.config_fingerprint,
        data_manifest_hash=data_manifest_hash,
        graph_regime=shared_graph_regime,
    )
    backbone_hash = hashlib.sha256(f"{args.qwen_path}".encode()).hexdigest()
    checkpoint_bundle = FormalHeadOnlyCheckpointBundle(
        llm_backbone=FormalHeadOnlyCheckpointComponent(
            path=str(args.qwen_path.resolve()),
            content_hash=backbone_hash,
            **common_checkpoint_kwargs,
        ),
        peft_adapter=FormalHeadOnlyCheckpointComponent(
            path=str(args.peft_adapter.resolve()),
            content_hash=adapter_sha256,
            **common_checkpoint_kwargs,
        ),
        cls_head=FormalHeadOnlyCheckpointComponent(
            path=str(args.cls_head.resolve()),
            content_hash=cls_head_sha256,
            **common_checkpoint_kwargs,
        ),
    )
    shared_checkpoint_provenance = CheckpointProvenance(
        path=str(args.cls_head.resolve()),
        step=int(args.step),
        content_hash=cls_head_sha256,
    )

    validation_inputs = _build_inputs(
        val_records, data_manifest,
        population=PopulationName.VALIDATION,
        checkpoint_provenance=shared_checkpoint_provenance,
        run_id=args.run_id,
    )
    report_inputs = _build_inputs(
        test_records, data_manifest,
        population=PopulationName.FINAL_TEST,
        checkpoint_provenance=shared_checkpoint_provenance,
        run_id=args.run_id,
    )
    val_population_metadata = _find_population(data_manifest, PopulationName.VALIDATION)
    test_population_metadata = _find_population(data_manifest, PopulationName.FINAL_TEST)

    print("[5/6] Running formal head-only eval (validation -> threshold freeze -> final_test)...", flush=True)
    report: FormalHeadOnlyReport = run_formal_head_only_eval(
        validation_inputs=validation_inputs,
        report_inputs=report_inputs,
        validation_population_metadata=val_population_metadata,
        report_population_metadata=test_population_metadata,
        model=model,
        cls_head=cls_head,
        tokenizer=tokenizer,
        thinking_mode=ThinkingMode.NON_THINKING,
        checkpoint_source=args.checkpoint_source,
        checkpoint_bundle=checkpoint_bundle,
        run_id=args.run_id,
        prompt_audit_path=str(prompt_audit_path.resolve()),
        prompt_audit_hash=prompt_audit_hash,
        threshold_selection_metric="f1",
        include_oracle_diagnostics=bool(args.include_oracle_diagnostics),
        calibration_bins=int(args.calibration_bins),
        apply_temperature_scaling=bool(args.apply_temperature_scaling),
        accelerator=None,
    )

    print("[6/6] Writing FormalHeadOnlyReport JSON...", flush=True)
    report_payload = json.loads(report.model_dump_json())
    git_state = capture_git_state(REPO_ROOT)
    report_payload["_runtime_provenance"] = {
        "dataset": args.dataset,
        "validation_population": PopulationName.VALIDATION.value,
        "report_population": PopulationName.FINAL_TEST.value,
        "checkpoint_source": args.checkpoint_source,
        "checkpoint_type": args.checkpoint_source,
        "run_id": args.run_id,
        "training_commit": args.commit,
        "seed": int(args.seed),
        "qwen_path": str(args.qwen_path),
        "peft_adapter": str(args.peft_adapter),
        "cls_head": str(args.cls_head),
        "cls_head_sha256": cls_head_sha256,
        "peft_adapter_sha256": adapter_sha256,
        "data_manifest": str(args.data_manifest),
        "data_manifest_sha256": data_manifest_hash,
        "apply_temperature_scaling": bool(args.apply_temperature_scaling),
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
    report_path = args.output_dir / f"formal_head_only_report_{args.dataset}.json"
    report_path.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"FORMAL HEAD-ONLY EVAL OK: wrote {report_path}")
    headline = report.headline_metrics
    if report.temperature is not None:
        print(f"  validation-fit temperature = {report.temperature:.4f}")
    print(
        f"  validation-frozen threshold = {report.validation_threshold.selected_threshold:.4f}"
    )
    print(
        f"  final_test: auroc={headline.auroc:.4f}  auprc={headline.auprc:.4f}  "
        f"f1@val_thr={headline.f1_at_val_threshold:.4f}  "
        f"prec@val_thr={headline.precision_at_val_threshold:.4f}  "
        f"rec@val_thr={headline.recall_at_val_threshold:.4f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
