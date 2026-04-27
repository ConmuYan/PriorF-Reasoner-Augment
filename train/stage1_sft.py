from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import torch

from evidence.evidence_schema import build_student_evidence_card
from evidence.leakage_policy import (
    EVIDENCE_CARD_PROJECTION,
    FORMAL_SAFE_RESULT,
    LEAKAGE_POLICY_VERSION,
    NEIGHBOR_LABEL_COUNTS_VISIBLE,
    NEIGHBOR_LABEL_POLICY,
    STUDENT_VISIBLE_FORBIDDEN_FIELD_PATHS,
    TEACHER_LOGIT_MASKED,
    TEACHER_PROB_MASKED,
)
from evidence.output_schema import PredLabel
from evidence.prompt_builder import PromptMode, ThinkingMode, build_prompt
from graph_data.manifests import PopulationMetadata, load_data_manifest
from priorf_teacher.export_pipeline import read_teacher_export_artifact
from priorf_teacher.schema import DatasetName, GraphRegime, PopulationName, TeacherExportRecord
from scripts._formal_eval_helpers import (
    build_subset_runtime_provenance,
    capture_git_state,
    current_python_command,
    file_sha256,
    stratified_records,
)
from train._tb_logger import TensorBoardLogger
from train.train_stage2_canonical import _encode_generation_training_messages

REPO_ROOT = Path(__file__).resolve().parents[1]
_ADAPTER_SUBDIR: Final[str] = "peft_adapter"
_RUN_RECORD_NAME: Final[str] = "run_record.json"
_TRAIN_LOG_NAME: Final[str] = "train_log.jsonl"
_CONFIG_NAME: Final[str] = "stage1_config.json"
_FINAL_CHECKPOINT_SUBDIR: Final[str] = "final_checkpoint"
_DEFAULT_LORA_TARGETS: Final[tuple[str, ...]] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class Stage1TrainingSample:
    evidence_card: Any
    ground_truth_label: int
    sft_target_label: PredLabel
    sft_target_score: float
    node_id: int


def _config_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_args_for_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple):
            payload[key] = list(value)
        else:
            payload[key] = value
    return payload


def _find_population(data_manifest: Any, population_name: PopulationName) -> PopulationMetadata:
    for population in data_manifest.populations:
        if population.population_name == population_name.value:
            return population
    raise ValueError(f"population {population_name.value!r} missing from data manifest")


def _resolve_shared_graph_regime(
    data_manifest: Any,
    records: tuple[TeacherExportRecord, ...],
) -> GraphRegime:
    observed_regimes: set[GraphRegime] = {GraphRegime(data_manifest.graph_regime)}
    observed_regimes.update(GraphRegime(record.graph_regime) for record in records)
    if len(observed_regimes) != 1:
        regimes = ", ".join(sorted(regime.value for regime in observed_regimes))
        raise ValueError(f"graph_regime mismatch across data manifest / teacher exports: {regimes}")
    return next(iter(observed_regimes))


def _resolve_total_steps(args: argparse.Namespace, *, n_train: int) -> int:
    if args.max_steps is not None:
        return int(args.max_steps)
    if args.num_epochs is None:
        raise ValueError("stage1_sft requires either --max-steps or --num-epochs")
    if n_train < 1:
        raise ValueError("stage1_sft requires at least one training sample")
    return max(1, int(math.ceil((int(args.num_epochs) * n_train) / int(args.batch_size))))


def _format_duration_seconds(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0.0:
        return "unknown"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _label_counts(records: tuple[TeacherExportRecord, ...]) -> tuple[int, int]:
    positive_count = sum(1 for record in records if int(record.ground_truth_label) == 1)
    negative_count = len(records) - positive_count
    return positive_count, negative_count


def _build_stage1_training_sample(
    teacher_record: TeacherExportRecord,
    data_manifest: Any,
) -> Stage1TrainingSample:
    evidence_card = build_student_evidence_card(teacher_record=teacher_record, data_manifest=data_manifest)
    label = int(teacher_record.ground_truth_label)
    return Stage1TrainingSample(
        evidence_card=evidence_card,
        ground_truth_label=label,
        sft_target_label=PredLabel.FRAUD if label == 1 else PredLabel.BENIGN,
        sft_target_score=0.95 if label == 1 else 0.05,
        node_id=int(teacher_record.node_id),
    )


def _compute_generation_loss(
    *,
    sample: Stage1TrainingSample,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
) -> torch.Tensor:
    bundle = build_prompt(
        evidence_card=sample.evidence_card,
        mode=PromptMode.TRAIN,
        thinking_mode=ThinkingMode.NON_THINKING,
        ground_truth_label_for_sft=sample.sft_target_label,
        score_target_for_sft=float(sample.sft_target_score),
    )
    encoded = _encode_generation_training_messages(tokenizer, bundle.messages, device=device)
    outputs = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        labels=encoded["labels"],
        use_cache=False,
    )
    loss = getattr(outputs, "loss", None)
    if loss is None:
        raise ValueError("stage1_sft requires model outputs.loss for generation fine-tuning")
    if loss.dim() != 0:
        raise ValueError("stage1_sft requires scalar generation loss")
    if not torch.isfinite(loss).item():
        raise ValueError("stage1_sft generation loss must be finite")
    return loss


def _adapter_weight_path(adapter_dir: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = adapter_dir / name
        if path.exists():
            return path
    raise ValueError(f"adapter weights missing under {adapter_dir}")


def _persist_adapter_artifacts(*, checkpoint_dir: Path, model: Any) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = checkpoint_dir / _ADAPTER_SUBDIR
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    model.save_pretrained(str(adapter_dir))
    adapter_weights_path = _adapter_weight_path(adapter_dir)
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise ValueError(f"adapter config missing under {adapter_dir}")
    return {
        "checkpoint_source": "final_checkpoint",
        "adapter_dir": str(adapter_dir.resolve()),
        "adapter_weights_path": str(adapter_weights_path.resolve()),
        "adapter_weights_sha256": file_sha256(adapter_weights_path),
        "adapter_config_path": str(adapter_config_path.resolve()),
        "adapter_config_sha256": file_sha256(adapter_config_path),
    }


def _copy_adapter_artifacts(*, source_checkpoint_dir: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    source_adapter_dir = source_checkpoint_dir / _ADAPTER_SUBDIR
    destination_adapter_dir = destination_dir / _ADAPTER_SUBDIR
    if destination_adapter_dir.exists():
        shutil.rmtree(destination_adapter_dir)
    shutil.copytree(source_adapter_dir, destination_adapter_dir)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Stage-1 structured-generation SFT driver")
    parser.add_argument("--dataset", required=True, choices=["amazon", "yelpchi"])
    parser.add_argument("--qwen-path", required=True, type=Path)
    parser.add_argument("--teacher-export-train", required=True, type=Path)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", nargs="+", default=list(_DEFAULT_LORA_TARGETS))
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tensorboard-dir", type=Path, default=None,
                        help="Directory for TensorBoard event files. "
                             "Defaults to <output-dir>/tb when --no-tensorboard is not set.")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable TensorBoard logging (default: enabled).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    device_str = f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print("[1/6] Loading canonical TRAIN teacher export + data manifest...", flush=True)
    train_records_full = read_teacher_export_artifact(args.teacher_export_train)
    if not train_records_full:
        raise ValueError("stage1_sft requires a non-empty TRAIN teacher export")
    if any(record.population_name != PopulationName.TRAIN for record in train_records_full):
        raise ValueError("stage1_sft requires teacher-export-train to contain only TRAIN records")
    dataset_enum = DatasetName(args.dataset)
    if any(record.dataset_name != dataset_enum for record in train_records_full):
        raise ValueError("teacher-export-train dataset_name must match --dataset")
    train_records = stratified_records(train_records_full, args.max_train_samples, args.seed)
    data_manifest = load_data_manifest(args.data_manifest)
    shared_graph_regime = _resolve_shared_graph_regime(data_manifest, train_records_full)
    total_steps = _resolve_total_steps(args, n_train=len(train_records))
    train_positive_count, train_negative_count = _label_counts(train_records)
    iters_per_epoch = max(1, int(math.ceil(len(train_records) / int(args.batch_size))))
    approx_total_epochs = float(total_steps) / float(iters_per_epoch)
    print(f"       train: {len(train_records)} / {len(train_records_full)} steps={total_steps}", flush=True)
    print(
        f"       graph_regime={shared_graph_regime.value} positives={train_positive_count} "
        f"negatives={train_negative_count} batch_size={int(args.batch_size)} "
        f"approx_epochs={approx_total_epochs:.2f}",
        flush=True,
    )

    print(f"[2/6] Loading Qwen checkpoint on {device}: {args.qwen_path}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_constant_schedule_with_warmup

    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(args.qwen_path))
    base_model = AutoModelForCausalLM.from_pretrained(
        str(args.qwen_path),
        dtype=torch_dtype,
        attn_implementation="eager",
    )

    print("[3/6] Wrapping backbone with PEFT LoRA...", flush=True)
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(args.lora_targets),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"       trainable / total: {trainable_params:,} / {total_params:,}", flush=True)

    print("[4/6] Materializing Stage-1 training samples...", flush=True)
    train_samples = tuple(_build_stage1_training_sample(record, data_manifest) for record in train_records)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    warmup_steps = 0 if total_steps <= 1 else int(round(total_steps * float(args.warmup_ratio)))
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    print(f"       warmup_steps={warmup_steps}", flush=True)

    print("[5/6] Running Stage-1 generation-only fine-tuning...", flush=True)
    epoch_indices: list[int] = []
    train_log_path = output_dir / _TRAIN_LOG_NAME
    train_log_f = train_log_path.open("w", encoding="utf-8")
    tb_log_dir: Path | None = None
    if not args.no_tensorboard:
        tb_log_dir = args.tensorboard_dir if args.tensorboard_dir is not None else output_dir / "tb"
    tb_logger = TensorBoardLogger(tb_log_dir, enabled=not args.no_tensorboard)
    if tb_logger.enabled:
        print(f"       tensorboard logdir = {tb_logger.log_dir}", flush=True)
    last_generation_loss: float | None = None
    train_started_at = time.perf_counter()
    samples_seen = 0
    for step in range(total_steps):
        if not epoch_indices:
            epoch_indices = list(range(len(train_samples)))
            rng.shuffle(epoch_indices)
        batch_indices = [epoch_indices.pop() for _ in range(min(args.batch_size, len(epoch_indices)))]
        while len(batch_indices) < args.batch_size:
            if not epoch_indices:
                epoch_indices = list(range(len(train_samples)))
                rng.shuffle(epoch_indices)
            batch_indices.append(epoch_indices.pop())
        batch_samples = tuple(train_samples[i] for i in batch_indices)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = [
            _compute_generation_loss(sample=sample, model=model, tokenizer=tokenizer, device=device)
            for sample in batch_samples
        ]
        loss = torch.stack(losses).mean()
        loss.backward()
        if args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(args.max_grad_norm))
        optimizer.step()
        scheduler.step()

        last_generation_loss = float(loss.detach().cpu())
        current_lr = float(scheduler.get_last_lr()[0])
        samples_seen += len(batch_samples)
        elapsed_seconds = time.perf_counter() - train_started_at
        steps_completed = step + 1
        steps_per_second = float(steps_completed) / elapsed_seconds if elapsed_seconds > 0.0 else float("inf")
        eta_seconds = (
            float(total_steps - steps_completed) / steps_per_second
            if math.isfinite(steps_per_second) and steps_per_second > 0.0
            else float("nan")
        )
        epoch_progress = float(steps_completed) / float(iters_per_epoch)
        train_log_f.write(json.dumps({
            "step": step + 1,
            "lr": current_lr,
            "generation_loss": last_generation_loss,
            "batch_node_ids": [sample.node_id for sample in batch_samples],
            "samples_seen": samples_seen,
            "elapsed_seconds": elapsed_seconds,
            "eta_seconds": eta_seconds,
        }) + "\n")
        train_log_f.flush()
        if step == 0 or (step + 1) % max(1, total_steps // 20) == 0 or step == total_steps - 1:
            print(
                f"       step {steps_completed:>4}/{total_steps}: "
                f"epoch~{epoch_progress:.2f}/{approx_total_epochs:.2f} "
                f"elapsed={_format_duration_seconds(elapsed_seconds)} "
                f"eta={_format_duration_seconds(eta_seconds)} "
                f"rate={steps_per_second:.2f} step/s "
                f"lr={current_lr:.2e} L_gen={last_generation_loss:.4f}",
                flush=True,
            )
    train_log_f.close()
    tb_logger.close()
    if last_generation_loss is None:
        raise ValueError("stage1_sft produced no training steps")

    print("[6/6] Saving adapter + config + run record...", flush=True)
    final_checkpoint_dir = output_dir / _FINAL_CHECKPOINT_SUBDIR
    checkpoint_artifacts = _persist_adapter_artifacts(checkpoint_dir=final_checkpoint_dir, model=model)
    _copy_adapter_artifacts(source_checkpoint_dir=final_checkpoint_dir, destination_dir=output_dir)

    config_payload = {
        "schema_version": "stage1_sft_config/v1",
        "dataset_name": dataset_enum.value,
        "graph_regime": shared_graph_regime.value,
        "train_population_name": PopulationName.TRAIN.value,
        "model_name_or_path": str(args.qwen_path),
        "output_dir": str(output_dir),
        "train_batch_size": int(args.batch_size),
        "max_steps": int(total_steps),
        "learning_rate": float(args.learning_rate),
        "warmup_ratio": float(args.warmup_ratio),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_targets": list(args.lora_targets),
        "thinking_mode": ThinkingMode.NON_THINKING.value,
    }
    runtime_provenance = {
        "dataset": args.dataset,
        "run_id": args.run_id,
        "qwen_path": str(args.qwen_path),
        "teacher_export_train": str(args.teacher_export_train),
        "data_manifest": str(args.data_manifest),
        "data_manifest_sha256": file_sha256(args.data_manifest),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "seed": int(args.seed),
        "training_command": current_python_command(),
        "config_fingerprint_sha256": _config_fingerprint(_normalize_args_for_fingerprint(args)),
        **capture_git_state(REPO_ROOT),
        **build_subset_runtime_provenance(
            prefix="train",
            population_name=PopulationName.TRAIN.value,
            records_full=train_records_full,
            selected_records=train_records,
            subset_requested=args.max_train_samples,
            teacher_export_path=args.teacher_export_train,
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    leakage_policy_payload = {
        "leakage_policy_version": LEAKAGE_POLICY_VERSION,
        "neighbor_label_policy": NEIGHBOR_LABEL_POLICY,
        "evidence_card_projection": EVIDENCE_CARD_PROJECTION,
        "student_visible_forbidden_fields": list(STUDENT_VISIBLE_FORBIDDEN_FIELD_PATHS),
        "teacher_prob_masked": TEACHER_PROB_MASKED,
        "teacher_logit_masked": TEACHER_LOGIT_MASKED,
        "neighbor_label_counts_visible": NEIGHBOR_LABEL_COUNTS_VISIBLE,
        "formal_safe_result": FORMAL_SAFE_RESULT,
    }
    run_record_payload = {
        "schema_version": "stage1_sft_run_record/v1",
        "config": config_payload,
        "train_population": _find_population(data_manifest, PopulationName.TRAIN).model_dump(mode="json"),
        "n_train_samples": len(train_samples),
        "final_step": {
            "step": total_steps,
            "generation_loss": last_generation_loss,
        },
        "checkpoint_artifacts": checkpoint_artifacts,
        **leakage_policy_payload,
        "_runtime_provenance": runtime_provenance,
    }
    _write_json(output_dir / _CONFIG_NAME, config_payload)
    _write_json(output_dir / _RUN_RECORD_NAME, run_record_payload)
    _write_json(final_checkpoint_dir / _CONFIG_NAME, config_payload)
    _write_json(final_checkpoint_dir / _RUN_RECORD_NAME, run_record_payload)

    print()
    print(f"STAGE1 SFT OK: wrote {output_dir}")
    print(f"  peft_adapter : {output_dir / _ADAPTER_SUBDIR}")
    print(f"  run_record   : {output_dir / _RUN_RECORD_NAME}")
    print(f"  train_log    : {output_dir / _TRAIN_LOG_NAME}")
    print(f"  final ckpt   : {final_checkpoint_dir}")
    print(f"  last L_gen   : {last_generation_loss:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
