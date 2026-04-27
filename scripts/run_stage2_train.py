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
* ``prompt_audit.json`` - fail-closed audit of student-visible prompt surfaces.

This is not a diagnostic smoke: it consumes canonical Task-2 artifacts
(``read_teacher_export_artifact`` + ``load_data_manifest``) and is expected
to be run under the ``gated`` namespace. Nothing is written under
``outputs/formal/`` by this driver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import subprocess
import time
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
    ScorerReport,
)
from evidence.evidence_schema import build_student_evidence_card  # noqa: E402
from evidence.leakage_policy import (  # noqa: E402
    FORMAL_SAFE_RESULT,
    STUDENT_VISIBLE_FORBIDDEN_FIELDS,
    STUDENT_VISIBLE_FORBIDDEN_PHRASES,
)
from evidence.output_schema import PredLabel  # noqa: E402
from evidence.prompt_audit import audit_prompt_bundle  # noqa: E402
from evidence.prompt_builder import PromptMode, ThinkingMode, build_prompt  # noqa: E402
from graph_data.manifests import PopulationMetadata, load_data_manifest  # noqa: E402
from priorf_teacher.export_pipeline import read_teacher_export_artifact  # noqa: E402
from priorf_teacher.schema import (  # noqa: E402
    DatasetName,
    GraphRegime,
    PopulationName,
    TeacherExportRecord,
)
from train._tb_logger import TensorBoardLogger  # noqa: E402
from train.train_stage2_canonical import (  # noqa: E402
    CanonicalStepReport,
    CanonicalTrainerConfig,
    CanonicalTrainerRunRecord,
    CanonicalTrainingBatch,
    CanonicalTrainingSample,
    leakage_policy_record_fields,
    run_canonical_train_step,
    run_validation_with_unified_scorer,
)

_ADAPTER_SUBDIR: Final[str] = "peft_adapter"
_CLS_HEAD_NAME: Final[str] = "cls_head.pt"
_RUN_RECORD_NAME: Final[str] = "run_record.json"
_TRAIN_LOG_NAME: Final[str] = "train_log.jsonl"
_PROMPT_AUDIT_NAME: Final[str] = "prompt_audit.json"
_FINAL_CHECKPOINT_SUBDIR: Final[str] = "final_checkpoint"
_BEST_CHECKPOINT_SUBDIR: Final[str] = "best_checkpoint"


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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_sha256(root: Path) -> str:
    hasher = hashlib.sha256()
    for item in sorted(root.rglob("*")):
        if item.is_file():
            hasher.update(str(item.relative_to(root)).encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(item.read_bytes())
            hasher.update(b"\0")
    return hasher.hexdigest()


def _git_output(args: list[str], *, repo_root: Path) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _capture_stage2_git_state(repo_root: Path) -> dict[str, object]:
    commit = _git_output(["rev-parse", "HEAD"], repo_root=repo_root).strip()
    status = _git_output(["status", "--porcelain=v1"], repo_root=repo_root)
    dirty = bool(status.strip())
    diff_hash = None
    if dirty:
        hasher = hashlib.sha256()
        for label, payload in (
            ("status", status),
            ("diff", _git_output(["diff", "--binary"], repo_root=repo_root)),
            ("cached", _git_output(["diff", "--cached", "--binary"], repo_root=repo_root)),
        ):
            hasher.update(label.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(payload.encode("utf-8"))
            hasher.update(b"\0")
        diff_hash = hasher.hexdigest()
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "git_diff_hash": diff_hash,
        "code_state_clean_for_formal": not dirty,
    }


def _teacher_checkpoint_sha256(records: tuple[TeacherExportRecord, ...]) -> str | None:
    checkpoints = sorted({str(record.teacher_checkpoint) for record in records if record.teacher_checkpoint})
    if len(checkpoints) != 1:
        return None
    path = Path(checkpoints[0])
    if not path.exists():
        return None
    if path.is_file():
        return _file_sha256(path)
    if path.is_dir():
        return _directory_sha256(path)
    return None


def _canonical_json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_prompt_audit_artifact(
    *,
    output_dir: Path,
    dataset: DatasetName,
    train_samples: tuple[CanonicalTrainingSample, ...],
    validation_records: tuple[TeacherExportRecord, ...],
    data_manifest: Any,
    git_commit: str,
) -> tuple[Path, str, dict[str, object]]:
    """Persist a fail-closed audit of student-visible Stage-2 prompt surfaces."""

    violations: list[dict[str, object]] = []
    split_counts: dict[str, dict[str, object]] = {}
    checked_messages = 0

    def audit_bundle(*, split: str, mode: PromptMode, node_id: int, sample_index: int, bundle: Any) -> None:
        nonlocal checked_messages
        result = audit_prompt_bundle(bundle, include_assistant_target=True)
        checked_messages += result.checked
        for violation in result.violations:
            violations.append(
                {
                    "split": split,
                    "mode": mode.value,
                    "node_id": int(node_id),
                    "sample_index": int(sample_index),
                    "location": violation.location,
                    "needle": violation.needle,
                    "kind": violation.kind,
                }
            )

    train_modes = (PromptMode.TRAIN, PromptMode.EVAL_HEAD, PromptMode.EVAL_GEN)
    split_counts["train"] = {
        "samples": len(train_samples),
        "modes": [mode.value for mode in train_modes],
    }
    for index, sample in enumerate(train_samples):
        for mode in train_modes:
            kwargs: dict[str, object] = {}
            if mode == PromptMode.TRAIN:
                kwargs = {
                    "ground_truth_label_for_sft": sample.sft_target_label,
                    "score_target_for_sft": float(sample.sft_target_score),
                }
            bundle = build_prompt(
                evidence_card=sample.evidence_card,
                mode=mode,
                thinking_mode=ThinkingMode.NON_THINKING,
                **kwargs,
            )
            audit_bundle(split="train", mode=mode, node_id=sample.node_id, sample_index=index, bundle=bundle)

    validation_modes = (PromptMode.EVAL_HEAD, PromptMode.EVAL_GEN)
    if validation_records:
        split_counts["validation"] = {
            "samples": len(validation_records),
            "modes": [mode.value for mode in validation_modes],
        }
        for index, record in enumerate(validation_records):
            card = build_student_evidence_card(teacher_record=record, data_manifest=data_manifest)
            for mode in validation_modes:
                bundle = build_prompt(
                    evidence_card=card,
                    mode=mode,
                    thinking_mode=ThinkingMode.NON_THINKING,
                )
                audit_bundle(split="validation", mode=mode, node_id=int(record.node_id), sample_index=index, bundle=bundle)

    payload: dict[str, object] = {
        "schema_version": "stage2_prompt_audit/v1",
        "dataset": dataset.value,
        "splits_checked": sorted(split_counts),
        "sample_counts": split_counts,
        "checked_messages": checked_messages,
        "forbidden_fields": list(STUDENT_VISIBLE_FORBIDDEN_FIELDS),
        "forbidden_phrases": list(STUDENT_VISIBLE_FORBIDDEN_PHRASES),
        "violations": violations,
        "violation_count": len(violations),
        "leakage_audit_pass": len(violations) == 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "audit_payload_sha256": None,
        "audit_file_sha256_source": "run_record.prompt_audit_hash is sha256(prompt_audit.json bytes)",
    }
    payload_without_hash = {k: v for k, v in payload.items() if k != "audit_payload_sha256"}
    payload["audit_payload_sha256"] = hashlib.sha256(_canonical_json_bytes(payload_without_hash)).hexdigest()

    audit_path = output_dir / _PROMPT_AUDIT_NAME
    audit_path.write_bytes(_canonical_json_bytes(payload))
    audit_hash = _file_sha256(audit_path)
    if violations:
        raise ValueError(f"prompt leakage audit failed; see {audit_path}")
    return audit_path, audit_hash, payload


def _persist_checkpoint_artifacts(
    *,
    checkpoint_dir: Path,
    model: Any,
    cls_head: torch.nn.Module,
    checkpoint_step: int,
) -> CheckpointProvenance:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = checkpoint_dir / _ADAPTER_SUBDIR
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    cls_head_path = checkpoint_dir / _CLS_HEAD_NAME
    if cls_head_path.exists():
        cls_head_path.unlink()
    model.save_pretrained(str(adapter_dir))
    cls_head_state = {k: v.detach().cpu() for k, v in cls_head.state_dict().items()}
    torch.save(cls_head_state, cls_head_path)
    return CheckpointProvenance(
        path=str(cls_head_path.resolve()),
        step=int(checkpoint_step),
        content_hash=_file_sha256(cls_head_path),
    )


def _copy_checkpoint_artifacts(*, source_checkpoint_dir: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    source_adapter_dir = source_checkpoint_dir / _ADAPTER_SUBDIR
    destination_adapter_dir = destination_dir / _ADAPTER_SUBDIR
    if destination_adapter_dir.exists():
        shutil.rmtree(destination_adapter_dir)
    shutil.copytree(source_adapter_dir, destination_adapter_dir)
    shutil.copy2(source_checkpoint_dir / _CLS_HEAD_NAME, destination_dir / _CLS_HEAD_NAME)


def _replace_checkpoint_provenance(
    report: ScorerReport,
    checkpoint_provenance: CheckpointProvenance,
) -> ScorerReport:
    report_payload = report.model_dump()
    report_payload["checkpoint_provenance"] = checkpoint_provenance.model_dump()
    return ScorerReport.model_validate(report_payload)


def _write_run_record(
    *,
    output_dir: Path,
    run_record: CanonicalTrainerRunRecord,
    runtime_provenance: dict[str, Any],
) -> None:
    run_record_payload = json.loads(run_record.model_dump_json())
    run_record_payload["_runtime_provenance"] = runtime_provenance
    (output_dir / _RUN_RECORD_NAME).write_text(
        json.dumps(run_record_payload, indent=2) + "\n", encoding="utf-8"
    )


def _best_checkpoint_metric_value(report: ScorerReport, metric_name: str) -> float:
    metric_value = report.auroc if metric_name == "validation_auroc" else report.auprc
    if metric_value is None:
        raise ValueError(f"{metric_name} unavailable on validation report")
    return float(metric_value)


def _build_training_sample(
    teacher_record: TeacherExportRecord, data_manifest: Any
) -> CanonicalTrainingSample:
    card = build_student_evidence_card(teacher_record=teacher_record, data_manifest=data_manifest)
    label = int(teacher_record.ground_truth_label)
    return CanonicalTrainingSample(
        evidence_card=card,
        ground_truth_label=label,  # type: ignore[arg-type]
        sft_target_label=PredLabel.FRAUD if label == 1 else PredLabel.BENIGN,
        sft_target_score=0.95 if label == 1 else 0.05,
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
    )


def _find_population(data_manifest: Any, population_name: PopulationName) -> PopulationMetadata:
    for population in data_manifest.populations:
        if population.population_name == population_name.value:
            return population
    raise ValueError(f"population {population_name.value!r} missing from data manifest")


def _config_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_unsupported_driver_args(args: argparse.Namespace) -> None:
    if int(args.gradient_accumulation_steps) != 1:
        raise ValueError(
            "run_stage2_train.py currently supports only --gradient-accumulation-steps 1; "
            "refusing to silently ignore gradient accumulation"
        )


def _resolve_shared_graph_regime(
    data_manifest: Any,
    *record_groups: tuple[TeacherExportRecord, ...],
) -> GraphRegime:
    observed_regimes: set[GraphRegime] = {GraphRegime(data_manifest.graph_regime)}
    for records in record_groups:
        observed_regimes.update(GraphRegime(record.graph_regime) for record in records)
    if len(observed_regimes) != 1:
        regimes = ", ".join(sorted(regime.value for regime in observed_regimes))
        raise ValueError(f"graph_regime mismatch across data manifest / teacher exports: {regimes}")
    return next(iter(observed_regimes))


def _format_duration_seconds(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0.0:
        return "unknown"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class _RollingMean:
    """Windowed simple moving average for noisy per-step scalars.

    With ``batch_size=2`` and stratified ``1+ / 1-`` batches, the per-step
    classification loss is bimodal (≈0.03 when both samples are correctly
    classified, ≈1.5 when the positive is misclassified).  TensorBoard
    benefits from showing both the raw scalar and a smoothed version so
    the underlying trend is visually separable from the per-batch noise.
    """

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError("rolling-mean window must be >= 1")
        self._window = int(window)
        self._buf: list[float] = []
        self._sum = 0.0

    @property
    def window(self) -> int:
        return self._window

    def __len__(self) -> int:
        return len(self._buf)

    def update(self, value: float) -> float:
        """Append ``value`` and return the current windowed mean."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return self.value
        self._buf.append(v)
        self._sum += v
        if len(self._buf) > self._window:
            self._sum -= self._buf.pop(0)
        return self.value

    @property
    def value(self) -> float:
        if not self._buf:
            return float("nan")
        return self._sum / float(len(self._buf))


def _label_counts(records: tuple[TeacherExportRecord, ...]) -> tuple[int, int]:
    positive_count = sum(1 for record in records if int(record.ground_truth_label) == 1)
    negative_count = len(records) - positive_count
    return positive_count, negative_count


def _stratified_record_subset(
    records: tuple[TeacherExportRecord, ...],
    *,
    subset_size: int,
    rng: random.Random,
) -> tuple[TeacherExportRecord, ...]:
    if subset_size >= len(records):
        return tuple(records)
    pos = [r for r in records if int(r.ground_truth_label) == 1]
    neg = [r for r in records if int(r.ground_truth_label) == 0]
    want_pos = max(1, int(round(subset_size * len(pos) / max(1, len(records)))))
    want_pos = min(want_pos, len(pos))
    want_neg = min(subset_size - want_pos, len(neg))
    return tuple(rng.sample(pos, want_pos) + rng.sample(neg, want_neg))


def _next_stratified_batch_indices(
    *,
    positive_indices: list[int],
    negative_indices: list[int],
    positive_pool: list[int],
    negative_pool: list[int],
    batch_size: int,
    rng: random.Random,
) -> list[int]:
    if batch_size < 2:
        raise ValueError("stratified Stage-2 training requires --batch-size >= 2")
    if not positive_indices or not negative_indices:
        raise ValueError("stratified Stage-2 training requires both positive and negative TRAIN samples")
    positive_quota = max(1, batch_size // 2)
    negative_quota = batch_size - positive_quota
    if negative_quota < 1:
        negative_quota = 1
        positive_quota = batch_size - negative_quota

    batch_indices: list[int] = []
    for _ in range(positive_quota):
        if not positive_pool:
            positive_pool.extend(positive_indices)
            rng.shuffle(positive_pool)
        batch_indices.append(positive_pool.pop())
    for _ in range(negative_quota):
        if not negative_pool:
            negative_pool.extend(negative_indices)
            rng.shuffle(negative_pool)
        batch_indices.append(negative_pool.pop())
    rng.shuffle(batch_indices)
    return batch_indices


def _build_trainable_peft_model(
    *,
    base_model: Any,
    lora_cfg: Any,
    warm_start_peft_adapter: Path | None,
) -> Any:
    if warm_start_peft_adapter is None:
        from peft import get_peft_model  # noqa: E402

        return get_peft_model(base_model, lora_cfg)

    from peft import PeftModel  # noqa: E402

    return PeftModel.from_pretrained(
        base_model,
        str(warm_start_peft_adapter),
        is_trainable=True,
    )


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
    parser.add_argument("--num-epochs", type=int, default=None,
                        help="Epochs over TRAIN. If set, --max-steps is computed as "
                             "ceil(num_epochs * n_train / batch_size). One of "
                             "--num-epochs / --max-steps must be given.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Explicit optimizer steps. Overrides --num-epochs when both set.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--lambda-distill", type=float, default=0.5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                        help="PEFT LoRA target_modules (default covers attention + MLP).")
    parser.add_argument("--warm-start-peft-adapter", type=Path, default=None,
                        help="Optional Stage-1 PEFT adapter directory to continue training from. "
                             "When set, the adapter is loaded as trainable instead of initializing a new LoRA adapter.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Fraction of total steps used for linear warmup.")
    parser.add_argument("--lr-scheduler", choices=["cosine", "linear", "constant"],
                        default="cosine")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Optional cap on number of TRAIN records (for faster smoke).")
    parser.add_argument("--validation-subset", type=int, default=128,
                        help="If --teacher-export-validation is set, run score_head on this many "
                             "validation rows periodically and at the end (stratified).")
    parser.add_argument("--validation-every-n-steps", type=int, default=0,
                        help="Run validation every N steps during training. 0 disables.")
    parser.add_argument("--best-checkpoint-metric",
                        choices=["validation_auroc", "validation_auprc"],
                        default="validation_auprc",
                        help="Metric used to snapshot output-dir/best_checkpoint on periodic "
                             "validation improvements.")
    parser.add_argument("--thinking-mode", choices=["non_thinking"], default="non_thinking")
    parser.add_argument("--tensorboard-dir", type=Path, default=None,
                        help="Directory for TensorBoard event files. "
                             "Defaults to <output-dir>/tb when --no-tensorboard is not set.")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable TensorBoard logging (default: enabled).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    args = _parse_args(argv)
    _validate_unsupported_driver_args(args)
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

    shared_graph_regime = _resolve_shared_graph_regime(data_manifest, train_records)
    train_positive_count, train_negative_count = _label_counts(train_records)

    print("[2/7] Building canonical EvidenceCards + training samples from TRAIN records...", flush=True)
    train_samples = tuple(
        _build_training_sample(r, data_manifest) for r in train_records
    )

    if args.max_steps is None and args.num_epochs is None:
        raise ValueError("one of --max-steps / --num-epochs must be provided")
    iters_per_epoch = max(1, (len(train_samples) + args.batch_size - 1) // args.batch_size)
    if args.max_steps is not None:
        total_steps = int(args.max_steps)
    else:
        total_steps = int(args.num_epochs) * iters_per_epoch
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1; got {total_steps}")
    warmup_steps = max(1, int(round(total_steps * max(0.0, args.warmup_ratio))))
    print(
        f"       iters_per_epoch={iters_per_epoch}  total_steps={total_steps}  "
        f"warmup_steps={warmup_steps}  lr_scheduler={args.lr_scheduler}",
        flush=True,
    )
    print(
        f"       graph_regime={shared_graph_regime.value} positives={train_positive_count} "
        f"negatives={train_negative_count} batch_size={int(args.batch_size)}",
        flush=True,
    )

    # Pre-select a stratified validation subset used for periodic monitoring.
    val_records_full: tuple[TeacherExportRecord, ...] = ()
    if args.teacher_export_validation is not None:
        val_records_full = read_teacher_export_artifact(args.teacher_export_validation)
        _resolve_shared_graph_regime(data_manifest, train_records, val_records_full)
    val_records_for_monitor: tuple[TeacherExportRecord, ...] = ()
    if val_records_full and args.validation_every_n_steps > 0:
        val_records_for_monitor = _stratified_record_subset(
            val_records_full,
            subset_size=int(args.validation_subset),
            rng=rng,
        )
        monitor_pos, monitor_neg = _label_counts(val_records_for_monitor)
        print(
            f"       preloaded {len(val_records_for_monitor)} stratified validation records "
            f"(positives={monitor_pos} negatives={monitor_neg}) "
            f"for periodic monitoring (every {args.validation_every_n_steps} steps)",
            flush=True,
        )

    git_state = _capture_stage2_git_state(REPO_ROOT)
    prompt_audit_path, prompt_audit_hash, prompt_audit_payload = _write_prompt_audit_artifact(
        output_dir=output_dir,
        dataset=dataset_enum,
        train_samples=train_samples,
        validation_records=val_records_full,
        data_manifest=data_manifest,
        git_commit=str(git_state["git_commit"]),
    )
    print(
        f"       prompt audit: {prompt_audit_path} "
        f"hash={prompt_audit_hash} pass={prompt_audit_payload['leakage_audit_pass']}",
        flush=True,
    )

    print(f"[3/7] Loading Qwen3 checkpoint on {device_str}: {args.qwen_path}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(str(args.qwen_path))
    base_model = AutoModelForCausalLM.from_pretrained(
        str(args.qwen_path),
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    print(
        f"[4/7] Wrapping with PEFT LoRA (r={args.lora_r}, alpha={args.lora_alpha}, "
        f"dropout={args.lora_dropout}, targets={args.lora_targets})",
        flush=True,
    )
    from peft import LoraConfig  # noqa: E402

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(args.lora_targets),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = _build_trainable_peft_model(
        base_model=base_model,
        lora_cfg=lora_cfg,
        warm_start_peft_adapter=args.warm_start_peft_adapter,
    )
    model.to(device_str)
    if args.warm_start_peft_adapter is not None:
        print(f"       warm-start adapter: {args.warm_start_peft_adapter}", flush=True)
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
        weight_decay=args.weight_decay,
    )

    from transformers import (  # noqa: E402
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )
    if args.lr_scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )
    elif args.lr_scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )
    else:
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
        )

    config = CanonicalTrainerConfig(
        dataset_name=dataset_enum,
        graph_regime=shared_graph_regime,
        model_name_or_path=str(args.qwen_path),
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        train_batch_size=args.batch_size,
        max_steps=total_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lambda_cls=args.lambda_cls,
        lambda_distill=args.lambda_distill,
        max_grad_norm=args.max_grad_norm,
        thinking_mode=ThinkingMode.NON_THINKING,
    )

    print(
        f"[5/7] Running {total_steps} canonical training steps "
        f"(epochs ~= {total_steps // iters_per_epoch if iters_per_epoch else 0}, "
        f"warmup={warmup_steps})...",
        flush=True,
    )
    from accelerate import Accelerator  # noqa: E402

    accelerator = Accelerator()
    train_log_path = output_dir / _TRAIN_LOG_NAME
    train_log_f = train_log_path.open("w", encoding="utf-8")
    tb_log_dir: Path | None = None
    if not args.no_tensorboard:
        tb_log_dir = args.tensorboard_dir if args.tensorboard_dir is not None else output_dir / "tb"
    tb_logger = TensorBoardLogger(tb_log_dir, enabled=not args.no_tensorboard)
    if tb_logger.enabled:
        print(f"       tensorboard logdir = {tb_logger.log_dir}", flush=True)
    smoothing_window = 50
    sma_L_gen = _RollingMean(smoothing_window)
    sma_L_cls = _RollingMean(smoothing_window)
    sma_L_distill = _RollingMean(smoothing_window)
    sma_total = _RollingMean(smoothing_window)
    last_step_report: CanonicalStepReport | None = None
    best_checkpoint_metric_value: float | None = None
    best_checkpoint_step: int | None = None
    best_checkpoint_provenance: CheckpointProvenance | None = None
    best_checkpoint_step_report: CanonicalStepReport | None = None
    best_validation_report: ScorerReport | None = None
    train_started_at = time.perf_counter()
    samples_seen = 0

    # Stratified per-batch sampling: each batch contains at least one
    # positive and one negative sample so the highly imbalanced TRAIN
    # population (Amazon ~13.5:1, YelpChi ~6:1) does not drown out the
    # minority-class gradient signal.  Positive and negative pools are
    # drained independently via separate shuffled queues; when a pool
    # empties it is reseeded with a fresh permutation, which is the
    # repeated-stratified-shuffle equivalent of multi-epoch training.
    positive_indices: list[int] = [
        idx for idx, sample in enumerate(train_samples) if int(sample.ground_truth_label) == 1
    ]
    negative_indices: list[int] = [
        idx for idx, sample in enumerate(train_samples) if int(sample.ground_truth_label) == 0
    ]
    positive_pool: list[int] = []
    negative_pool: list[int] = []
    for step in range(total_steps):
        batch_indices = _next_stratified_batch_indices(
            positive_indices=positive_indices,
            negative_indices=negative_indices,
            positive_pool=positive_pool,
            negative_pool=negative_pool,
            batch_size=int(args.batch_size),
            rng=rng,
        )
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
        # Advance LR scheduler AFTER the optimizer step so the newly set
        # learning rate applies to the NEXT step.  This preserves standard
        # PyTorch warmup behaviour (step 0 sees near-zero LR during warmup).
        scheduler.step()
        last_step_report = step_report
        current_lr = float(scheduler.get_last_lr()[0])
        samples_seen += len(batch_samples)
        steps_completed = step + 1
        elapsed_seconds = time.perf_counter() - train_started_at
        steps_per_second = float(steps_completed) / elapsed_seconds if elapsed_seconds > 0.0 else float("inf")
        eta_seconds = (
            float(total_steps - steps_completed) / steps_per_second
            if math.isfinite(steps_per_second) and steps_per_second > 0.0
            else float("nan")
        )
        epoch_progress = float(steps_completed) / float(iters_per_epoch)
        log_row = {
            "step": step,
            "lr": current_lr,
            "L_gen": step_report.generation_loss,
            "L_cls": step_report.classification_loss,
            "L_distill": step_report.distillation_loss,
            "total": step_report.total_loss,
            "samples_seen": samples_seen,
            "elapsed_seconds": elapsed_seconds,
            "eta_seconds": eta_seconds,
        }
        train_log_f.write(json.dumps(log_row) + "\n")
        train_log_f.flush()
        sma_L_gen.update(step_report.generation_loss)
        sma_L_cls.update(step_report.classification_loss)
        sma_L_distill.update(step_report.distillation_loss)
        sma_total.update(step_report.total_loss)
        tb_logger.log_scalars(
            "train",
            {
                "lr": current_lr,
                "L_gen": step_report.generation_loss,
                "L_cls": step_report.classification_loss,
                "L_distill": step_report.distillation_loss,
                "total": step_report.total_loss,
                "steps_per_sec": steps_per_second,
                "samples_seen": samples_seen,
                f"L_gen_sma{smoothing_window}": sma_L_gen.value,
                f"L_cls_sma{smoothing_window}": sma_L_cls.value,
                f"L_distill_sma{smoothing_window}": sma_L_distill.value,
                f"total_sma{smoothing_window}": sma_total.value,
            },
            step + 1,
        )
        if step == 0 or (step + 1) % max(1, total_steps // 20) == 0 or step == total_steps - 1:
            print(
                f"       step {steps_completed:>4}/{total_steps}: "
                f"epoch~{epoch_progress:.2f} "
                f"elapsed={_format_duration_seconds(elapsed_seconds)} "
                f"eta={_format_duration_seconds(eta_seconds)} "
                f"rate={steps_per_second:.2f} step/s "
                f"lr={current_lr:.2e} "
                f"L_gen={step_report.generation_loss:.4f} "
                f"L_cls={step_report.classification_loss:.4f} "
                f"L_distill={step_report.distillation_loss:.4f} "
                f"total={step_report.total_loss:.4f}",
                flush=True,
            )
        # Periodic in-loop validation using the pre-loaded monitor subset.
        if (
            val_records_for_monitor
            and args.validation_every_n_steps > 0
            and (step + 1) % args.validation_every_n_steps == 0
        ):
            model.eval()
            cls_head.eval()
            validation_started_at = time.perf_counter()
            monitor_provenance = CheckpointProvenance(
                path=str((output_dir / _CLS_HEAD_NAME).resolve()),
                step=step + 1,
                content_hash="m" * 64,  # interim monitor (not promoted)
            )
            monitor_inputs = _build_head_scoring_inputs(
                val_records_for_monitor,
                data_manifest,
                population=PopulationName.VALIDATION,
                checkpoint_provenance=monitor_provenance,
            )
            monitor_report = run_validation_with_unified_scorer(
                validation_inputs=monitor_inputs,
                model=model,
                cls_head=cls_head,
                tokenizer=tokenizer,
                thinking_mode=ThinkingMode.NON_THINKING,
                prompt_audit_path=str(prompt_audit_path.resolve()),
                prompt_audit_hash=prompt_audit_hash,
                accelerator=None,
            )
            validation_elapsed_seconds = time.perf_counter() - validation_started_at
            print(
                f"       [val@{step + 1}]  auroc={monitor_report.auroc}  "
                f"auprc={monitor_report.auprc}  brier={monitor_report.brier_score:.4f}  "
                f"prob_std={monitor_report.prob_std:.4f}  "
                f"elapsed={_format_duration_seconds(validation_elapsed_seconds)}",
                flush=True,
            )
            train_log_f.write(json.dumps({
                "step": step,
                "validation_auroc": monitor_report.auroc,
                "validation_auprc": monitor_report.auprc,
                "validation_brier": monitor_report.brier_score,
                "validation_prob_std": monitor_report.prob_std,
                "validation_elapsed_seconds": validation_elapsed_seconds,
            }) + "\n")
            train_log_f.flush()
            metric_value = _best_checkpoint_metric_value(monitor_report, args.best_checkpoint_metric)
            if best_checkpoint_metric_value is None or metric_value > best_checkpoint_metric_value:
                best_checkpoint_provenance = _persist_checkpoint_artifacts(
                    checkpoint_dir=output_dir / _BEST_CHECKPOINT_SUBDIR,
                    model=model,
                    cls_head=cls_head,
                    checkpoint_step=step + 1,
                )
                best_validation_report = _replace_checkpoint_provenance(
                    monitor_report,
                    best_checkpoint_provenance,
                )
                best_checkpoint_metric_value = metric_value
                best_checkpoint_step = step + 1
                best_checkpoint_step_report = step_report
                print(
                    f"       [best@{step + 1}]  {args.best_checkpoint_metric}={metric_value:.6f}",
                    flush=True,
                )
            model.train()
            cls_head.train(True)
    train_log_f.close()
    tb_logger.flush()
    assert last_step_report is not None

    print("[6/7] Saving PEFT adapter + cls_head checkpoint...", flush=True)
    final_checkpoint_dir = output_dir / _FINAL_CHECKPOINT_SUBDIR
    final_checkpoint_provenance = _persist_checkpoint_artifacts(
        checkpoint_dir=final_checkpoint_dir,
        model=model,
        cls_head=cls_head,
        checkpoint_step=total_steps,
    )
    _copy_checkpoint_artifacts(
        source_checkpoint_dir=final_checkpoint_dir,
        destination_dir=output_dir,
    )
    cls_head_sha256 = final_checkpoint_provenance.content_hash
    adapter_dir_sha256 = _directory_sha256(final_checkpoint_dir / _ADAPTER_SUBDIR)

    teacher_export_validation_sha256 = (
        _file_sha256(args.teacher_export_validation)
        if args.teacher_export_validation is not None
        else None
    )
    code_state_clean_for_formal = bool(git_state["code_state_clean_for_formal"])
    formal_safe_result = bool(
        FORMAL_SAFE_RESULT
        and code_state_clean_for_formal
        and prompt_audit_payload["leakage_audit_pass"]
        and teacher_export_validation_sha256 is not None
    )
    diagnostic_only = not formal_safe_result
    provenance_fields = {
        "git_commit": git_state["git_commit"],
        "git_dirty": git_state["git_dirty"],
        "git_diff_hash": git_state["git_diff_hash"],
        "teacher_export_train_sha256": _file_sha256(args.teacher_export_train),
        "teacher_export_validation_sha256": teacher_export_validation_sha256,
        "data_manifest_sha256": _file_sha256(args.data_manifest),
        "teacher_checkpoint_sha256": _teacher_checkpoint_sha256(tuple(train_records) + tuple(val_records_full)),
        "adapter_dir_sha256": adapter_dir_sha256,
        "cls_head_sha256": cls_head_sha256,
    }
    leakage_fields = leakage_policy_record_fields(
        prompt_audit_path=str(prompt_audit_path.resolve()),
        prompt_audit_hash=prompt_audit_hash,
        formal_safe_result=formal_safe_result,
        diagnostic_only=diagnostic_only,
        code_state_clean_for_formal=code_state_clean_for_formal,
    )

    validation_report = None
    if val_records_full:
        print(f"[7/7] Running post-training validation via score_head: {args.teacher_export_validation}", flush=True)
        subset = _stratified_record_subset(
            val_records_full,
            subset_size=int(args.validation_subset),
            rng=rng,
        )
        post_pos, post_neg = _label_counts(subset)
        print(
            f"       post-training stratified subset n={len(subset)} positives={post_pos} negatives={post_neg}",
            flush=True,
        )
        validation_inputs = _build_head_scoring_inputs(
            subset,
            data_manifest,
            population=PopulationName.VALIDATION,
            checkpoint_provenance=final_checkpoint_provenance,
        )
        cls_head.eval()
        model.eval()
        validation_report = run_validation_with_unified_scorer(
            validation_inputs=validation_inputs,
            model=model,
            cls_head=cls_head,
            tokenizer=tokenizer,
            thinking_mode=ThinkingMode.NON_THINKING,
            prompt_audit_path=str(prompt_audit_path.resolve()),
            prompt_audit_hash=prompt_audit_hash,
            accelerator=None,
        )
        print(
            f"       validation n={validation_report.n_total} "
            f"auroc={validation_report.auroc} auprc={validation_report.auprc} "
            f"brier={validation_report.brier_score:.4f}",
            flush=True,
        )
        tb_logger.log_scalars(
            "post_train_validation",
            {
                "auroc": validation_report.auroc,
                "auprc": validation_report.auprc,
                "brier": validation_report.brier_score,
                "prob_std": validation_report.prob_std,
                "n_total": validation_report.n_total,
            },
            total_steps,
        )
    else:
        print("[7/7] Skipping post-training validation (no --teacher-export-validation)", flush=True)
    tb_logger.close()

    train_population = _find_population(data_manifest, PopulationName.TRAIN)
    val_population = (
        _find_population(data_manifest, PopulationName.VALIDATION)
        if args.teacher_export_validation is not None
        else None
    )
    run_record = CanonicalTrainerRunRecord(
        config=config,
        checkpoint_provenance=final_checkpoint_provenance,
        train_population=train_population,
        validation_population=val_population,
        graph_regime=shared_graph_regime,
        last_step=last_step_report,
        validation_report=validation_report,
        **provenance_fields,
        **leakage_fields,
    )
    runtime_provenance = {
        "dataset": args.dataset,
        "qwen_path": str(args.qwen_path),
        "teacher_export_train": str(args.teacher_export_train),
        "teacher_export_validation": (
            str(args.teacher_export_validation) if args.teacher_export_validation else None
        ),
        "warm_start_peft_adapter": (
            str(args.warm_start_peft_adapter) if args.warm_start_peft_adapter else None
        ),
        "data_manifest": str(args.data_manifest),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "total_steps": total_steps,
        "checkpoint_source": "final_checkpoint",
        "checkpoint_dir": str(final_checkpoint_dir.resolve()),
        "best_checkpoint_dir": (
            str((output_dir / _BEST_CHECKPOINT_SUBDIR).resolve())
            if best_checkpoint_provenance is not None
            else None
        ),
        "best_checkpoint_step": best_checkpoint_step,
        "best_checkpoint_metric": args.best_checkpoint_metric,
        "best_checkpoint_metric_value": best_checkpoint_metric_value,
        "config_fingerprint_sha256": _config_fingerprint(
            {k: v for k, v in vars(args).items() if not isinstance(v, Path)}
        ),
        **provenance_fields,
        **leakage_fields,
        "prompt_audit_payload_sha256": prompt_audit_payload["audit_payload_sha256"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_run_record(
        output_dir=output_dir,
        run_record=run_record,
        runtime_provenance=runtime_provenance,
    )
    _write_run_record(
        output_dir=final_checkpoint_dir,
        run_record=run_record,
        runtime_provenance=runtime_provenance,
    )

    if best_checkpoint_provenance is not None and best_validation_report is not None:
        best_run_record = CanonicalTrainerRunRecord(
            config=config,
            checkpoint_provenance=best_checkpoint_provenance,
            train_population=train_population,
            validation_population=val_population,
            graph_regime=shared_graph_regime,
            last_step=best_checkpoint_step_report,
            validation_report=best_validation_report,
            **{
                **provenance_fields,
                "adapter_dir_sha256": _directory_sha256(output_dir / _BEST_CHECKPOINT_SUBDIR / _ADAPTER_SUBDIR),
                "cls_head_sha256": best_checkpoint_provenance.content_hash,
            },
            **leakage_fields,
        )
        _write_run_record(
            output_dir=output_dir / _BEST_CHECKPOINT_SUBDIR,
            run_record=best_run_record,
            runtime_provenance={
                **runtime_provenance,
                "checkpoint_source": "best_checkpoint",
                "checkpoint_dir": str((output_dir / _BEST_CHECKPOINT_SUBDIR).resolve()),
                "selected_checkpoint_step": best_checkpoint_step,
                "selected_checkpoint_metric": args.best_checkpoint_metric,
                "selected_checkpoint_metric_value": best_checkpoint_metric_value,
                "selected_checkpoint_validation_n": best_validation_report.n_total,
                "adapter_dir_sha256": _directory_sha256(output_dir / _BEST_CHECKPOINT_SUBDIR / _ADAPTER_SUBDIR),
                "cls_head_sha256": best_checkpoint_provenance.content_hash,
            },
        )

    print()
    print(f"STAGE2 TRAIN OK: wrote {output_dir}")
    print(f"  peft_adapter : {output_dir / _ADAPTER_SUBDIR}")
    print(f"  cls_head     : {output_dir / _CLS_HEAD_NAME}")
    print(f"  run_record   : {output_dir / _RUN_RECORD_NAME}")
    print(f"  train_log    : {output_dir / _TRAIN_LOG_NAME}")
    print(f"  prompt_audit : {prompt_audit_path}")
    print(f"  final ckpt   : {final_checkpoint_dir}")
    if best_checkpoint_provenance is not None and best_checkpoint_step is not None:
        print(
            f"  best ckpt    : {output_dir / _BEST_CHECKPOINT_SUBDIR} "
            f"({args.best_checkpoint_metric}={best_checkpoint_metric_value:.6f} @ step {best_checkpoint_step})"
        )
    print(
        f"  last step    : L_gen={last_step_report.generation_loss:.4f} "
        f"L_cls={last_step_report.classification_loss:.4f} "
        f"L_distill={last_step_report.distillation_loss:.4f} "
        f"total={last_step_report.total_loss:.4f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
