"""Stage-2 canonical trainer driver for real Qwen3-4B + real teacher exports.

Wraps ``train.train_stage2_canonical.run_canonical_train_step`` into a
multi-step loop over TRAIN-population ``TeacherExportRecord`` rows.  Each
batch is materialised via ``build_evidence_card`` so Stage 2 sees exactly
the canonical EvidenceCard objects that Stage 3 evaluation consumes.

Outputs (all under ``<output-dir>/``, caller picks diagnostic / gated /
formal namespace; script itself is namespace-agnostic):

* ``peft_adapter/`` - PEFT LoRA adapter saved via ``save_pretrained``
* ``cls_head.pt``   - ``state_dict`` of the trainable classification head
* ``run_record.json`` - ``CanonicalTrainerRunRecord`` + extra provenance
  (HEAD commit, config fingerprint sha256, last-step losses, optional
  validation ``ScorerReport``).
* ``train_log.jsonl`` - per-step JSON lines with losses for debugging.

This is not a diagnostic smoke: it consumes canonical Task-2 artifacts
(``read_teacher_export_artifact`` + ``load_data_manifest``) and is expected
to be run under the ``gated`` namespace. Nothing is written under
``outputs/formal/`` by this driver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.head_scoring import (  # noqa: E402
    CheckpointProvenance,
    HeadScoringInputs,
    HeadScoringSample,
    score_head,
)
from evidence.evidence_schema import build_evidence_card  # noqa: E402
from evidence.output_schema import PredLabel  # noqa: E402
from evidence.prompt_builder import ThinkingMode  # noqa: E402
from graph_data.manifests import PopulationMetadata, load_data_manifest  # noqa: E402
from priorf_teacher.export_pipeline import read_teacher_export_artifact  # noqa: E402
from priorf_teacher.schema import (  # noqa: E402
    DatasetName,
    GraphRegime,
    PopulationName,
    TeacherExportRecord,
)
from train.train_stage2_canonical import (  # noqa: E402
    CanonicalStepReport,
    CanonicalTrainerConfig,
    CanonicalTrainerRunRecord,
    CanonicalTrainingBatch,
    CanonicalTrainingSample,
    run_canonical_train_step,
    run_validation_with_unified_scorer,
)

_GRAPH_REGIME: Final[GraphRegime] = GraphRegime.TRANSDUCTIVE_STANDARD
_ADAPTER_SUBDIR: Final[str] = "peft_adapter"
_CLS_HEAD_NAME: Final[str] = "cls_head.pt"
_RUN_RECORD_NAME: Final[str] = "run_record.json"
_TRAIN_LOG_NAME: Final[str] = "train_log.jsonl"


class _LinearClsHead(torch.nn.Module):
    """Trainable linear classification head returning a 1-D logit tensor."""

    def __init__(self, hidden_size: int, *, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.linear = torch.nn.Linear(hidden_size, 1)
        with torch.no_grad():
            self.linear.weight.normal_(generator=generator)
            self.linear.weight.mul_(0.01)
            self.linear.bias.zero_()

    def forward(self, hidden_prompt_only: torch.Tensor) -> torch.Tensor:
        hidden_for_head = hidden_prompt_only.to(self.linear.weight.dtype)
        return self.linear(hidden_for_head).squeeze(-1)


def _build_training_sample(
    teacher_record: TeacherExportRecord, data_manifest: Any
) -> CanonicalTrainingSample:
    card = build_evidence_card(teacher_record=teacher_record, data_manifest=data_manifest)
    label = int(teacher_record.ground_truth_label)
    return CanonicalTrainingSample(
        evidence_card=card,
        ground_truth_label=label,  # type: ignore[arg-type]
        sft_target_label=PredLabel.FRAUD if label == 1 else PredLabel.BENIGN,
        sft_target_score=float(teacher_record.teacher_prob),
        teacher_prob=float(teacher_record.teacher_prob),
        node_id=int(teacher_record.node_id),
    )


def _build_head_scoring_inputs(
    records: tuple[TeacherExportRecord, ...],
    data_manifest: Any,
    *,
    population: PopulationName,
    checkpoint_provenance: CheckpointProvenance,
) -> HeadScoringInputs:
    samples = []
    for record in records:
        card = build_evidence_card(teacher_record=record, data_manifest=data_manifest)
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
    )


def _find_population(data_manifest: Any, population_name: PopulationName) -> PopulationMetadata:
    for population in data_manifest.populations:
        if population.population_name == population_name.value:
            return population
    raise ValueError(f"population {population_name.value!r} missing from data manifest")


def _config_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=["amazon", "yelpchi"])
    parser.add_argument("--qwen-path", required=True, type=Path)
    parser.add_argument("--teacher-export-train", required=True, type=Path,
                        help="Path to canonical TRAIN teacher_export.parquet")
    parser.add_argument("--teacher-export-validation", type=Path, default=None,
                        help="Optional path to canonical VALIDATION teacher_export.parquet "
                             "for post-training validation via score_head.")
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Output directory (e.g. outputs/gated/stage2/<dataset>/<run_id>).")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--lambda-distill", type=float, default=0.5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Optional cap on number of TRAIN records (for faster smoke).")
    parser.add_argument("--validation-subset", type=int, default=128,
                        help="If --teacher-export-validation is set, run score_head on this many "
                             "validation rows at the end (stratified).")
    parser.add_argument("--thinking-mode", choices=["non_thinking"], default="non_thinking")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    args = _parse_args(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device_str = "cpu" if args.gpu_index < 0 else f"cuda:{args.gpu_index}"
    dataset_enum = DatasetName(args.dataset)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Loading canonical TeacherExportRecord TRAIN records: {args.teacher_export_train}", flush=True)
    train_records = read_teacher_export_artifact(args.teacher_export_train)
    print(f"       loaded {len(train_records)} TRAIN records", flush=True)
    print(f"       loading data manifest: {args.data_manifest}", flush=True)
    data_manifest = load_data_manifest(args.data_manifest)

    rng = random.Random(args.seed)
    if args.max_train_samples is not None and args.max_train_samples < len(train_records):
        train_records = tuple(rng.sample(list(train_records), args.max_train_samples))
        print(f"       restricted to {len(train_records)} TRAIN records for smoke", flush=True)

    print("[2/7] Building canonical EvidenceCards + training samples from TRAIN records...", flush=True)
    train_samples = tuple(
        _build_training_sample(r, data_manifest) for r in train_records
    )

    print(f"[3/7] Loading Qwen3 checkpoint on {device_str}: {args.qwen_path}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(str(args.qwen_path))
    base_model = AutoModelForCausalLM.from_pretrained(
        str(args.qwen_path),
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    print(f"[4/7] Wrapping with PEFT LoRA (r={args.lora_r}, alpha={args.lora_alpha})", flush=True)
    from peft import LoraConfig, get_peft_model  # noqa: E402

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.to(device_str)
    # Activation checkpointing drops activation memory for the backbone at
    # the cost of a recompute on backward; enables training at bf16 on a
    # single 48 GiB GPU with B=2 and long Evidence-Card prompts without OOM.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"       trainable / total: {trainable_params:,} / {total_params:,}", flush=True)

    hidden_size = int(base_model.config.hidden_size)
    cls_head = _LinearClsHead(hidden_size=hidden_size, seed=args.seed).to(device_str).to(torch.bfloat16)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(cls_head.parameters()),
        lr=args.learning_rate,
    )

    config = CanonicalTrainerConfig(
        dataset_name=dataset_enum,
        graph_regime=_GRAPH_REGIME,
        model_name_or_path=str(args.qwen_path),
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        train_batch_size=args.batch_size,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lambda_cls=args.lambda_cls,
        lambda_distill=args.lambda_distill,
        max_grad_norm=args.max_grad_norm,
        thinking_mode=ThinkingMode.NON_THINKING,
    )

    print(f"[5/7] Running {args.max_steps} canonical training steps...", flush=True)
    from accelerate import Accelerator  # noqa: E402

    accelerator = Accelerator()
    train_log_path = output_dir / _TRAIN_LOG_NAME
    train_log_f = train_log_path.open("w", encoding="utf-8")
    last_step_report: CanonicalStepReport | None = None
    for step in range(args.max_steps):
        batch_indices = [rng.randrange(len(train_samples)) for _ in range(args.batch_size)]
        batch_samples = tuple(train_samples[i] for i in batch_indices)
        batch = CanonicalTrainingBatch(samples=batch_samples)
        step_report = run_canonical_train_step(
            config=config,
            batch=batch,
            model=model,
            cls_head=cls_head,
            tokenizer=tokenizer,
            optimizer=optimizer,
            accelerator=accelerator,
        )
        last_step_report = step_report
        log_row = {
            "step": step,
            "L_gen": step_report.generation_loss,
            "L_cls": step_report.classification_loss,
            "L_distill": step_report.distillation_loss,
            "total": step_report.total_loss,
        }
        train_log_f.write(json.dumps(log_row) + "\n")
        train_log_f.flush()
        if step == 0 or (step + 1) % max(1, args.max_steps // 10) == 0 or step == args.max_steps - 1:
            print(
                f"       step {step + 1:>4}/{args.max_steps}: "
                f"L_gen={step_report.generation_loss:.4f} "
                f"L_cls={step_report.classification_loss:.4f} "
                f"L_distill={step_report.distillation_loss:.4f} "
                f"total={step_report.total_loss:.4f}",
                flush=True,
            )
    train_log_f.close()
    assert last_step_report is not None

    print("[6/7] Saving PEFT adapter + cls_head checkpoint...", flush=True)
    model.save_pretrained(str(output_dir / _ADAPTER_SUBDIR))
    cls_head_state = {k: v.detach().cpu() for k, v in cls_head.state_dict().items()}
    torch.save(cls_head_state, output_dir / _CLS_HEAD_NAME)
    cls_head_bytes = (output_dir / _CLS_HEAD_NAME).read_bytes()
    cls_head_sha256 = hashlib.sha256(cls_head_bytes).hexdigest()

    validation_report = None
    if args.teacher_export_validation is not None:
        print(f"[7/7] Running post-training validation via score_head: {args.teacher_export_validation}", flush=True)
        val_records = read_teacher_export_artifact(args.teacher_export_validation)
        # stratified subset for speed
        pos = [r for r in val_records if int(r.ground_truth_label) == 1]
        neg = [r for r in val_records if int(r.ground_truth_label) == 0]
        want_pos = max(1, int(round(args.validation_subset * len(pos) / max(1, len(val_records)))))
        want_pos = min(want_pos, len(pos))
        want_neg = min(args.validation_subset - want_pos, len(neg))
        subset = tuple(rng.sample(pos, want_pos) + rng.sample(neg, want_neg))
        checkpoint_provenance = CheckpointProvenance(
            path=str((output_dir / _CLS_HEAD_NAME).resolve()),
            step=args.max_steps,
            content_hash=cls_head_sha256,
        )
        validation_inputs = _build_head_scoring_inputs(
            subset,
            data_manifest,
            population=PopulationName.VALIDATION,
            checkpoint_provenance=checkpoint_provenance,
        )
        cls_head.eval()
        model.eval()
        validation_report = run_validation_with_unified_scorer(
            validation_inputs=validation_inputs,
            model=model,
            cls_head=cls_head,
            tokenizer=tokenizer,
            thinking_mode=ThinkingMode.NON_THINKING,
            accelerator=None,
        )
        print(
            f"       validation n={validation_report.n_total} "
            f"auroc={validation_report.auroc} auprc={validation_report.auprc} "
            f"brier={validation_report.brier_score:.4f}",
            flush=True,
        )
    else:
        print("[7/7] Skipping post-training validation (no --teacher-export-validation)", flush=True)

    train_population = _find_population(data_manifest, PopulationName.TRAIN)
    val_population = (
        _find_population(data_manifest, PopulationName.VALIDATION)
        if args.teacher_export_validation is not None
        else None
    )
    run_record = CanonicalTrainerRunRecord(
        config=config,
        checkpoint_provenance=CheckpointProvenance(
            path=str((output_dir / _CLS_HEAD_NAME).resolve()),
            step=args.max_steps,
            content_hash=cls_head_sha256,
        ),
        train_population=train_population,
        validation_population=val_population,
        graph_regime=_GRAPH_REGIME,
        last_step=last_step_report,
        validation_report=validation_report,
    )
    run_record_payload = json.loads(run_record.model_dump_json())
    run_record_payload["_runtime_provenance"] = {
        "dataset": args.dataset,
        "qwen_path": str(args.qwen_path),
        "teacher_export_train": str(args.teacher_export_train),
        "teacher_export_validation": (
            str(args.teacher_export_validation) if args.teacher_export_validation else None
        ),
        "data_manifest": str(args.data_manifest),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "config_fingerprint_sha256": _config_fingerprint(
            {k: v for k, v in vars(args).items() if not isinstance(v, Path)}
        ),
        "cls_head_sha256": cls_head_sha256,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / _RUN_RECORD_NAME).write_text(
        json.dumps(run_record_payload, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print(f"STAGE2 TRAIN OK: wrote {output_dir}")
    print(f"  peft_adapter : {output_dir / _ADAPTER_SUBDIR}")
    print(f"  cls_head     : {output_dir / _CLS_HEAD_NAME}")
    print(f"  run_record   : {output_dir / _RUN_RECORD_NAME}")
    print(f"  train_log    : {output_dir / _TRAIN_LOG_NAME}")
    print(
        f"  last step    : L_gen={last_step_report.generation_loss:.4f} "
        f"L_cls={last_step_report.classification_loss:.4f} "
        f"L_distill={last_step_report.distillation_loss:.4f} "
        f"total={last_step_report.total_loss:.4f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
