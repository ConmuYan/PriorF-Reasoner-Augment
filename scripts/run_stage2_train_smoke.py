"""Diagnostic-only Stage-2 canonical trainer 1-step smoke on real Qwen3-4B.

Runs exactly one ``run_canonical_train_step`` call with:

* Qwen3-4B backbone wrapped in a PEFT LoRA adapter (so only a tiny set of
  parameters is trainable on a single 4090);
* a trainable linear ``cls_head`` on top of the pooled hidden state;
* a micro batch of 2 TRAIN-population samples adapted from the legacy
  teacher-export parquet (same adapter as ``run_head_only_smoke.py``).

What this verifies end-to-end on real weights:

* ``PromptMode.TRAIN`` -> tokenizer.apply_chat_template -> labels mask ->
  model forward with ``output_hidden_states=True`` -> ``outputs.loss`` is
  a finite scalar (L_gen);
* ``PromptMode.EVAL_HEAD`` -> forward -> ``pool_last_valid_token`` ->
  ``cls_head`` -> finite logit (L_cls path);
* BCEWithLogitsLoss against ``ground_truth_label`` (L_cls);
* BCEWithLogitsLoss against clipped ``teacher_prob`` (L_distill);
* ``total_loss = L_gen + lambda_cls * L_cls + lambda_distill * L_distill``
  is finite, reaches ``accelerator.backward``, and the optimizer step
  commits without NaN;
* the resulting ``CanonicalStepReport`` round-trips Pydantic validation,
  proving ``extra="forbid"`` + ``_losses_finite`` guards did not trip.

Non-goals:

* this is not a full training run and not a checkpoint producer; no
  ``outputs/formal/`` artifact is written.  Stage-2 production training
  needs a dataloader + checkpoint schedule + validation loop that wraps
  ``run_canonical_train_step`` without bypassing its fail-closed objective.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evidence.output_schema import PredLabel  # noqa: E402
from evidence.prompt_builder import ThinkingMode  # noqa: E402
from priorf_teacher.schema import DatasetName, GraphRegime, PopulationName  # noqa: E402
from train.train_stage2_canonical import (  # noqa: E402
    CanonicalStepReport,
    CanonicalTrainerConfig,
    CanonicalTrainingBatch,
    CanonicalTrainingSample,
    run_canonical_train_step,
)

# We re-use the legacy-row -> EvidenceCard adapter from the head-only smoke.
import importlib.util

_HEAD_ONLY_SMOKE_PATH = REPO_ROOT / "scripts" / "run_head_only_smoke.py"
_spec = importlib.util.spec_from_file_location("_head_only_smoke_adapter", _HEAD_ONLY_SMOKE_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot locate {_HEAD_ONLY_SMOKE_PATH}")
_head_only_smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_head_only_smoke)
_row_to_evidence_card = _head_only_smoke._row_to_evidence_card
_stratified_subset = _head_only_smoke._stratified_subset


_NAMESPACE_ROOT: Final[str] = "outputs/diagnostic/stage2_train_smoke"


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
        # run_canonical_train_step expects a scalar per sample; it calls
        # cls_logit.reshape(()).  The loop is strict B=1 so this returns
        # shape [1, 1] -> squeeze() to shape [] or [1] both work with
        # reshape(()).
        hidden_for_head = hidden_prompt_only.to(self.linear.weight.dtype)
        return self.linear(hidden_for_head).squeeze(-1)


def _build_training_sample(row: pd.Series, *, dataset: DatasetName) -> CanonicalTrainingSample:
    card = _row_to_evidence_card(row, dataset=dataset)
    # Swap the diagnostic population to TRAIN for the canonical trainer's
    # fail-closed check (sample must live on TRAIN).
    train_card = card.model_copy(update={"population_name": PopulationName.TRAIN})
    label = int(row["label"])
    if label not in (0, 1):
        raise ValueError(f"legacy row label must be 0/1; got {label!r}")
    return CanonicalTrainingSample(
        evidence_card=train_card,
        ground_truth_label=label,  # type: ignore[arg-type]
        sft_target_label=PredLabel.FRAUD if label == 1 else PredLabel.BENIGN,
        sft_target_score=0.95 if label == 1 else 0.05,
        teacher_prob=float(row["teacher_prob"]),
        node_id=int(row["node_id"]),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--legacy-parquet", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=["amazon", "yelpchi"])
    parser.add_argument("--qwen-path", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    torch.manual_seed(args.seed)

    dataset_enum = DatasetName(args.dataset)
    device_str = "cpu" if args.gpu_index < 0 else f"cuda:{args.gpu_index}"

    print(f"[1/7] Loading legacy parquet: {args.legacy_parquet}", flush=True)
    df = pd.read_parquet(args.legacy_parquet)
    subset = _stratified_subset(df, subset_size=args.batch_size, seed=args.seed)
    print(
        f"       kept {len(subset)} rows "
        f"(positives={(subset['label'] == 1).sum()}, "
        f"negatives={(subset['label'] == 0).sum()})",
        flush=True,
    )

    print("[2/7] Building TRAIN-population samples...", flush=True)
    samples = tuple(_build_training_sample(row, dataset=dataset_enum) for _, row in subset.iterrows())
    batch = CanonicalTrainingBatch(samples=samples)

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
    # PEFT freezes the backbone; only LoRA adapters remain trainable.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"       trainable / total parameters: {trainable_params:,} / {total_params:,}")

    hidden_size = int(base_model.config.hidden_size)
    cls_head = _LinearClsHead(hidden_size=hidden_size, seed=args.seed).to(device_str).to(torch.bfloat16)
    cls_head.train(True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(cls_head.parameters()),
        lr=1e-4,
    )

    print("[5/7] Building canonical trainer config...", flush=True)
    config = CanonicalTrainerConfig(
        dataset_name=dataset_enum,
        graph_regime=GraphRegime.TRANSDUCTIVE_STANDARD,
        model_name_or_path=str(args.qwen_path),
        output_dir=str(args.output_root / "diagnostic" / "stage2_train_smoke"),
        learning_rate=1e-4,
        train_batch_size=args.batch_size,
        max_steps=1,
        gradient_accumulation_steps=1,
        lambda_cls=1.0,
        lambda_distill=0.5,
        thinking_mode=ThinkingMode.NON_THINKING,
    )

    print("[6/7] Running exactly one canonical train step...", flush=True)
    from accelerate import Accelerator  # noqa: E402

    accelerator = Accelerator()
    report: CanonicalStepReport = run_canonical_train_step(
        config=config,
        batch=batch,
        model=model,
        cls_head=cls_head,
        tokenizer=tokenizer,
        optimizer=optimizer,
        accelerator=accelerator,
    )

    print("[7/7] Writing CanonicalStepReport JSON to diagnostic namespace...", flush=True)
    output_dir = args.output_root / "diagnostic" / "stage2_train_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"canonical_step_report_{dataset_enum.value}.json"
    report_payload: dict[str, Any] = json.loads(report.model_dump_json())
    report_payload["_diagnostic_provenance"] = {
        "legacy_parquet": str(args.legacy_parquet),
        "qwen_path": str(args.qwen_path),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"STAGE2 SMOKE OK: wrote {report_path}")
    print(
        f"  L_gen={report.generation_loss:.4f}  L_cls={report.classification_loss:.4f}  "
        f"L_distill={report.distillation_loss:.4f}  total={report.total_loss:.4f}"
    )
    print(f"  lambda_cls={report.lambda_cls}  lambda_distill={report.lambda_distill}")
    print(f"  backward_via_accelerate={report.used_accelerate_backward}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
