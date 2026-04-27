from __future__ import annotations

import hashlib
import json
import math
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from evidence.output_schema import parse_strict
from evidence.leakage_policy import STUDENT_VISIBLE_FORBIDDEN_FIELDS, STUDENT_VISIBLE_FORBIDDEN_PHRASES
from evidence.prompt_audit import audit_prompt_bundle
from eval.head_scoring import CheckpointProvenance, HeadScoringInputs, HeadScoringSample
from evidence.evidence_schema import build_evidence_card, build_student_evidence_card
from evidence.prompt_builder import PromptMode, ThinkingMode, build_prompt
from graph_data.manifests import PopulationMetadata
from priorf_teacher.schema import PopulationName, TeacherExportRecord

REPO_ROOT = Path(__file__).resolve().parents[1]


class LinearClsHead(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, 1)

    def forward(self, hidden_prompt_only: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_prompt_only.to(self.linear.weight.dtype)).squeeze(-1)



def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



def hash_directory_files(root: Path) -> str:
    hasher = hashlib.sha256()
    for item in sorted(root.rglob("*")):
        if item.is_file():
            hasher.update(str(item.relative_to(root)).encode("utf-8"))
            hasher.update(item.read_bytes())
    return hasher.hexdigest()



def canonical_json_sha256(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()



def selected_node_ids_sha256(records: tuple[TeacherExportRecord, ...]) -> str:
    return canonical_json_sha256([int(record.node_id) for record in records])



def selected_records_sha256(records: tuple[TeacherExportRecord, ...]) -> str:
    return canonical_json_sha256([record.model_dump(mode="json") for record in records])



def current_python_command() -> str:
    return shlex.join([sys.executable, *sys.argv])



def format_duration_seconds(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0.0:
        return "unknown"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"



def capture_git_state(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_output = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "git_commit": commit,
        "git_dirty": bool(dirty_output.strip()),
    }



def build_subset_runtime_provenance(
    *,
    prefix: str,
    population_name: str,
    records_full: tuple[TeacherExportRecord, ...],
    selected_records: tuple[TeacherExportRecord, ...],
    subset_requested: int | None,
    teacher_export_path: Path,
) -> dict[str, Any]:
    key_prefix = f"{prefix}_" if prefix else ""
    return {
        f"{key_prefix}population_name": population_name,
        f"{key_prefix}population_size_total": len(records_full),
        f"{key_prefix}subset_requested": None if subset_requested is None else int(subset_requested),
        f"{key_prefix}subset_size": len(selected_records),
        f"{key_prefix}full_population_evaluated": len(selected_records) == len(records_full),
        f"{key_prefix}teacher_export": str(teacher_export_path),
        f"{key_prefix}teacher_export_sha256": file_sha256(teacher_export_path),
        f"{key_prefix}selected_node_ids_sha256": selected_node_ids_sha256(selected_records),
        f"{key_prefix}selected_records_sha256": selected_records_sha256(selected_records),
    }



def stratified_records(records, subset_size: int | None, seed: int):
    if subset_size is None or subset_size >= len(records):
        return tuple(records)
    rng = random.Random(seed)
    pos = [record for record in records if int(record.ground_truth_label) == 1]
    neg = [record for record in records if int(record.ground_truth_label) == 0]
    want_pos = max(1, int(round(subset_size * len(pos) / max(1, len(records)))))
    want_pos = min(want_pos, len(pos))
    want_neg = min(subset_size - want_pos, len(neg))
    return tuple(rng.sample(pos, want_pos) + rng.sample(neg, want_neg))



def find_population(data_manifest: Any, population_name: PopulationName) -> PopulationMetadata:
    for population in data_manifest.populations:
        if population.population_name == population_name.value:
            return population
    raise ValueError(f"population {population_name.value!r} missing from data manifest")



def build_checkpoint_provenance(*, cls_head_path: Path, step: int) -> CheckpointProvenance:
    return CheckpointProvenance(
        path=str(cls_head_path.resolve()),
        step=int(step),
        content_hash=file_sha256(cls_head_path),
    )



def build_head_scoring_inputs(
    records: tuple[TeacherExportRecord, ...],
    data_manifest: Any,
    *,
    population: PopulationName,
    checkpoint_provenance: CheckpointProvenance,
    run_id: str = "head_scoring_ad_hoc",
    student_visible: bool = True,
) -> HeadScoringInputs:
    samples = []
    for record in records:
        if student_visible:
            card = build_student_evidence_card(teacher_record=record, data_manifest=data_manifest)
        else:
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
        run_id=run_id,
    )



def load_model_bundle(
    *,
    qwen_path: Path,
    peft_adapter: Path,
    cls_head_path: Path,
    gpu_index: int,
) -> dict[str, Any]:
    device_str = "cpu" if gpu_index < 0 else f"cuda:{gpu_index}"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(qwen_path))
    base_model = AutoModelForCausalLM.from_pretrained(
        str(qwen_path),
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, str(peft_adapter))
    model.to(device_str)
    model.eval()

    hidden_size = int(base_model.config.hidden_size)
    cls_head = LinearClsHead(hidden_size=hidden_size)
    cls_head.load_state_dict(torch.load(cls_head_path, map_location="cpu", weights_only=True))
    cls_head.to(device_str).to(torch.bfloat16)
    cls_head.eval()

    return {
        "device": device_str,
        "tokenizer": tokenizer,
        "model": model,
        "cls_head": cls_head,
        "cls_head_sha256": file_sha256(cls_head_path),
        "peft_adapter_sha256": hash_directory_files(peft_adapter),
    }



def generate_structured_outputs(
    records: tuple[TeacherExportRecord, ...],
    data_manifest: Any,
    *,
    model: Any,
    tokenizer: Any,
    thinking_mode: ThinkingMode,
    max_new_tokens: int,
    progress_label: str | None = None,
    progress_every: int | None = None,
) -> tuple[str, ...]:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id
    if pad_token_id is None or eos_token_id is None:
        raise ValueError("tokenizer must expose pad_token_id or eos_token_id for generation")

    generated_texts: list[str] = []
    total_records = len(records)
    progress_interval = max(1, progress_every or max(1, total_records // 10))
    generation_started_at = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for record_index, record in enumerate(records, start=1):
            card = build_student_evidence_card(teacher_record=record, data_manifest=data_manifest)
            bundle = build_prompt(
                evidence_card=card,
                mode=PromptMode.EVAL_GEN,
                thinking_mode=thinking_mode,
            )
            if bundle.messages[-1].role != "assistant" or bundle.messages[-1].content:
                raise ValueError("eval_gen prompt must end with an empty assistant message")
            messages = [
                {"role": message.role, "content": message.content}
                for message in bundle.messages[:-1]
            ]
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            if generated_ids.dim() != 2 or int(generated_ids.shape[0]) != 1:
                raise ValueError("model.generate must return a [1, T] token tensor")
            continuation_ids = generated_ids[0, input_ids.shape[1]:]
            if int(continuation_ids.shape[0]) < 1:
                raise ValueError("model.generate returned no continuation tokens")
            generated_text = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
            generated_text = _trim_trailing_text_after_strict_json(generated_text)
            if not generated_text:
                raise ValueError("model.generate decoded to empty text")
            generated_texts.append(generated_text)
            if progress_label is not None and (
                record_index == 1
                or record_index == total_records
                or record_index % progress_interval == 0
            ):
                elapsed_seconds = time.perf_counter() - generation_started_at
                records_per_second = (
                    float(record_index) / elapsed_seconds if elapsed_seconds > 0.0 else float("inf")
                )
                eta_seconds = (
                    float(total_records - record_index) / records_per_second
                    if math.isfinite(records_per_second) and records_per_second > 0.0
                    else float("nan")
                )
                print(
                    f"       [{progress_label}] {record_index}/{total_records} "
                    f"elapsed={format_duration_seconds(elapsed_seconds)} "
                    f"eta={format_duration_seconds(eta_seconds)} "
                    f"rate={records_per_second:.2f} rec/s",
                    flush=True,
                )
    return tuple(generated_texts)


def _trim_trailing_text_after_strict_json(text: str) -> str:
    stripped = text.strip()
    if not stripped or not stripped.startswith("{"):
        return stripped
    for end_index in range(len(stripped), 0, -1):
        candidate = stripped[:end_index].strip()
        if not candidate.endswith("}"):
            continue
        try:
            parse_strict(candidate)
        except Exception:
            continue
        return candidate
    return stripped


def write_prompt_audit_artifact(
    *,
    output_path: Path,
    dataset: str,
    eval_type: str,
    entries: tuple[tuple[str, Any, bool], ...],
    extra: dict[str, Any] | None = None,
) -> tuple[Path, str, dict[str, Any]]:
    """Persist a fail-closed audit for exact prompt bundles used by an eval driver.

    ``entries`` contains ``(location, PromptBundle, include_assistant_target)``.
    The artifact does not store full prompt text; it records counts, forbidden
    surfaces, and exact violations. The returned hash is sha256(file bytes).
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checked = 0
    violations: list[dict[str, str]] = []
    for location, bundle, include_assistant_target in entries:
        result = audit_prompt_bundle(bundle, include_assistant_target=include_assistant_target)
        checked += result.checked
        for violation in result.violations:
            violations.append({
                "location": f"{location}:{violation.location}",
                "needle": violation.needle,
                "kind": violation.kind,
            })
    payload: dict[str, Any] = {
        "schema_version": "formal_eval_prompt_audit/v1",
        "dataset": dataset,
        "eval_type": eval_type,
        "sample_count": len(entries),
        "checked_message_count": checked,
        "forbidden_fields": list(STUDENT_VISIBLE_FORBIDDEN_FIELDS),
        "forbidden_phrases": list(STUDENT_VISIBLE_FORBIDDEN_PHRASES),
        "violations": violations,
        "leakage_audit_pass": len(violations) == 0,
        "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "code_git_commit": capture_git_state(REPO_ROOT)["git_commit"],
    }
    if extra:
        payload.update(extra)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_hash = file_sha256(output_path)
    if violations:
        raise ValueError(f"formal eval prompt audit failed; see {output_path}")
    return output_path, audit_hash, payload


def build_eval_prompt_audit_entries(
    records: tuple[TeacherExportRecord, ...],
    data_manifest: Any,
    *,
    population: PopulationName,
    mode: PromptMode,
    thinking_mode: ThinkingMode,
    include_assistant_target: bool = True,
    location_prefix: str | None = None,
) -> tuple[tuple[str, Any, bool], ...]:
    entries: list[tuple[str, Any, bool]] = []
    prefix = location_prefix or population.value
    for index, record in enumerate(records):
        card = build_student_evidence_card(teacher_record=record, data_manifest=data_manifest)
        bundle = build_prompt(evidence_card=card, mode=mode, thinking_mode=thinking_mode)
        entries.append((f"{prefix}[{index}].node_id={int(record.node_id)}", bundle, include_assistant_target))
    return tuple(entries)
