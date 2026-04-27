"""Stage-2 head-only inference + ScorerReport on canonical teacher exports.

Loads a trained ``scripts/run_stage2_train.py`` artifact (PEFT adapter +
``cls_head.pt``), together with a canonical ``TeacherExportRecord``
parquet for one population, and runs the unified ``score_head`` contract
to produce a ``ScorerReport``.

Outputs:

* ``<output-dir>/scorer_report_<dataset>_<population>.json`` containing
  the ``ScorerReport`` model + a ``_runtime_provenance`` block pinning
  the backbone, adapter, cls_head, and teacher export paths + sha256s.

The script is namespace-agnostic - callers pick
``outputs/gated/eval/...`` / ``outputs/formal/eval/...`` etc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.head_scoring import (  # noqa: E402
    CheckpointProvenance,
    HeadScoringInputs,
    HeadScoringSample,
    ScorerReport,
    score_head,
)
from evidence.evidence_schema import build_student_evidence_card  # noqa: E402
from evidence.prompt_builder import PromptMode, ThinkingMode  # noqa: E402
from graph_data.manifests import load_data_manifest  # noqa: E402
from priorf_teacher.export_pipeline import read_teacher_export_artifact  # noqa: E402
from priorf_teacher.schema import PopulationName  # noqa: E402
from scripts._formal_eval_helpers import (  # noqa: E402
    build_eval_prompt_audit_entries,
    write_prompt_audit_artifact,
)


class _LinearClsHead(torch.nn.Module):
    """Mirror of training cls_head layout; state_dict keys match."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, 1)

    def forward(self, hidden_prompt_only: torch.Tensor) -> torch.Tensor:
        hidden_for_head = hidden_prompt_only.to(self.linear.weight.dtype)
        return self.linear(hidden_for_head).squeeze(-1)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stratified_subset(records, *, subset_size: int, seed: int):
    import random
    if subset_size >= len(records):
        return tuple(records)
    rng = random.Random(seed)
    pos = [r for r in records if int(r.ground_truth_label) == 1]
    neg = [r for r in records if int(r.ground_truth_label) == 0]
    want_pos = max(1, int(round(subset_size * len(pos) / max(1, len(records)))))
    want_pos = min(want_pos, len(pos))
    want_neg = min(subset_size - want_pos, len(neg))
    return tuple(rng.sample(pos, want_pos) + rng.sample(neg, want_neg))


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, choices=["amazon", "yelpchi"])
    p.add_argument("--qwen-path", required=True, type=Path)
    p.add_argument("--peft-adapter", required=True, type=Path)
    p.add_argument("--cls-head", required=True, type=Path)
    p.add_argument("--teacher-export", required=True, type=Path)
    p.add_argument("--data-manifest", required=True, type=Path)
    p.add_argument("--population", required=True,
                   choices=["validation", "final_test"],
                   help="Population to score (names must match TeacherExportRecord "
                        "and DataManifest).")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--subset-size", type=int, default=None,
                   help="Optional stratified subset for quick inference runs.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--step", type=int, default=0,
                   help="Step count recorded in CheckpointProvenance (metadata only).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    device_str = "cpu" if args.gpu_index < 0 else f"cuda:{args.gpu_index}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading canonical TeacherExportRecord records: {args.teacher_export}", flush=True)
    records = read_teacher_export_artifact(args.teacher_export)
    print(f"       loaded {len(records)} records", flush=True)
    if args.subset_size is not None:
        records = _stratified_subset(records, subset_size=args.subset_size, seed=args.seed)
        print(f"       stratified subset -> {len(records)} records", flush=True)
    data_manifest = load_data_manifest(args.data_manifest)

    print("[2/6] Building canonical EvidenceCards + scoring samples...", flush=True)
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

    print(f"[3/6] Loading Qwen3 backbone on {device_str}: {args.qwen_path}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(str(args.qwen_path))
    base_model = AutoModelForCausalLM.from_pretrained(
        str(args.qwen_path),
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    print(f"[4/6] Loading PEFT adapter: {args.peft_adapter}", flush=True)
    from peft import PeftModel  # noqa: E402

    model = PeftModel.from_pretrained(base_model, str(args.peft_adapter))
    model.to(device_str)
    model.eval()

    print(f"[5/6] Loading cls_head state_dict: {args.cls_head}", flush=True)
    hidden_size = int(base_model.config.hidden_size)
    cls_head = _LinearClsHead(hidden_size=hidden_size)
    state_dict = torch.load(args.cls_head, map_location="cpu", weights_only=True)
    cls_head.load_state_dict(state_dict)
    cls_head.to(device_str).to(torch.bfloat16)
    cls_head.eval()
    cls_head_sha256 = _file_sha256(args.cls_head)

    checkpoint_provenance = CheckpointProvenance(
        path=str(args.cls_head.resolve()),
        step=int(args.step),
        content_hash=cls_head_sha256,
    )
    inputs = HeadScoringInputs(
        samples=tuple(samples),
        dataset_name=records[0].dataset_name,
        population_name=PopulationName(args.population),
        graph_regime=records[0].graph_regime,
        checkpoint_provenance=checkpoint_provenance,
    )
    prompt_audit_path, prompt_audit_hash, _ = write_prompt_audit_artifact(
        output_path=args.output_dir / f"prompt_audit_head_scoring_{args.dataset}_{args.population}.json",
        dataset=args.dataset,
        eval_type="head_scoring",
        entries=build_eval_prompt_audit_entries(
            tuple(records),
            data_manifest,
            population=PopulationName(args.population),
            mode=PromptMode.EVAL_HEAD,
            thinking_mode=ThinkingMode.NON_THINKING,
            include_assistant_target=False,
            location_prefix=args.population,
        ),
        extra={"population": args.population, "subset_size": args.subset_size},
    )

    print(f"[6/6] Running score_head on {len(samples)} samples...", flush=True)
    report: ScorerReport = score_head(
        inputs=inputs,
        model=model,
        cls_head=cls_head,
        tokenizer=tokenizer,
        thinking_mode=ThinkingMode.NON_THINKING,
        prompt_audit_path=str(prompt_audit_path.resolve()),
        prompt_audit_hash=prompt_audit_hash,
        accelerator=None,
    )

    report_payload = json.loads(report.model_dump_json())
    report_payload["_runtime_provenance"] = {
        "dataset": args.dataset,
        "population": args.population,
        "qwen_path": str(args.qwen_path),
        "peft_adapter": str(args.peft_adapter),
        "cls_head": str(args.cls_head),
        "cls_head_sha256": cls_head_sha256,
        "teacher_export": str(args.teacher_export),
        "data_manifest": str(args.data_manifest),
        "prompt_audit_path": str(prompt_audit_path.resolve()),
        "prompt_audit_hash": prompt_audit_hash,
        "subset_size": args.subset_size,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = args.output_dir / f"scorer_report_{args.dataset}_{args.population}.json"
    report_path.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"INFERENCE OK: wrote {report_path}")
    print(
        f"  n_total={report.n_total}  n_positive={report.n_positive}  "
        f"n_negative={report.n_negative}  is_single_class={report.is_single_class_population}"
    )
    print(f"  auroc={report.auroc}  auprc={report.auprc}  brier={report.brier_score:.4f}")
    print(
        f"  prob_mean={report.prob_mean:.4f}  prob_std={report.prob_std:.4f}  "
        f"min={report.prob_min:.4f}  max={report.prob_max:.4f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
