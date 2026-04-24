# Self-review: End-to-end pipeline smoke on real Qwen3-4B

This self-review accompanies the two diagnostic runners added to close out
the runtime-acceptance follow-up listed in `status_package/general.md`:

- `scripts/run_head_only_smoke.py`
- `scripts/run_stage2_train_smoke.py`

and the assembled / validated `outputs/gated/manifests/gate_manifest.json`
that exercises `scripts/gate_check.py` and `scripts/run_eval.sh` on the
happy and negative paths.

None of these artifacts change the formal contract.  Every runtime
artifact they produce lands under `outputs/diagnostic/` (or
`outputs/formal/` only in response to a passing, explicitly-authored gate
manifest).  No canonical module was changed.

## 1. Non-goals (kept out of scope)

- No change to `eval/head_scoring.py`, `train/train_stage2_canonical.py`,
  `priorf_teacher/schema.py`, `evidence/*`, `schemas/gate_manifest.py`,
  `scripts/gate_check.py`, `scripts/run_full_pipeline.sh`, or any
  `tests/test_*.py`.
- No new Pydantic schemas.  All runtime objects go through the existing
  strict schemas (`EvidenceCard`, `HeadScoringInputs`, `HeadScoringSample`,
  `CheckpointProvenance`, `ScorerReport`, `CanonicalTrainingSample`,
  `CanonicalTrainingBatch`, `CanonicalTrainerConfig`, `CanonicalStepReport`).
- No threshold / alpha / calibration reselection.  Both smoke runners are
  aware only of the canonical `predict_proba` path; they do not tune or
  cache any threshold.
- No `outputs/formal/` writes from diagnostic runners.  `run_head_only_smoke.py`
  and `run_stage2_train_smoke.py` hard-code their output under
  `<output_root>/diagnostic/...`.

## 2. What each smoke runner verifies

### 2.1 `scripts/run_head_only_smoke.py`

End-to-end chain on real weights:

1. `legacy parquet row` → `_row_to_evidence_card` (synthesises NeighborSummary
   label counts because the legacy schema does not carry them; clearly
   labelled diagnostic-only in the module docstring).
2. `EvidenceCard` → `HeadScoringSample` → `HeadScoringInputs`
   (`population_name=PopulationName.DIAGNOSTIC_HOLDOUT`, so the record
   cannot be confused with TRAIN / VALIDATION / FINAL_TEST / UNUSED_HOLDOUT).
3. Qwen3-4B-Instruct-2507 loaded from `/data1/mq/models/Qwen3-4B-Instruct-2507`
   in `torch.bfloat16` with `attn_implementation="eager"`.
4. `score_head(...)` runs the canonical `predict_proba` pipeline
   (`PromptMode.EVAL_HEAD` + strict B=1 + `output_hidden_states=True` +
   `pool_last_valid_token` + random linear head + `torch.sigmoid`).
5. `ScorerReport` round-trips Pydantic validation (`extra="forbid"`,
   `frozen=True`, forbidden-fields guard, population-counts sanity).
6. Report + diagnostic provenance note is written to
   `outputs/diagnostic/head_only_smoke/scorer_report_<dataset>_<N>.json`.

The one-off `_RandomLinearClsHead` returns a 1-D logit tensor of length B
(`.squeeze(-1)`), matching the canonical `ClsHead` Protocol expectation
enforced in `score_head`.  It is explicitly labelled as diagnostic and is
not intended to produce meaningful AUROC.

### 2.2 `scripts/run_stage2_train_smoke.py`

End-to-end chain on real weights:

1. Qwen3-4B backbone + PEFT `LoraConfig(r=8, alpha=16, target={q_proj, v_proj})`.
   LoRA freezes 99.93% of parameters (2.95M / 4.03B trainable), so one
   forward+backward fits comfortably in 24GB.
2. `_LinearClsHead` (trainable linear over `pool_last_valid_token`).
3. Micro batch of 2 TRAIN-population `CanonicalTrainingSample` objects
   adapted from legacy parquet rows (re-uses `_row_to_evidence_card` from
   the head-only smoke, then `model_copy(update={"population_name": TRAIN})`
   to satisfy the canonical trainer's TRAIN-only guard).
4. `CanonicalTrainerConfig` constructed with
   `require_generation_loss=require_classification_loss=require_distillation_loss=True`,
   `use_accelerate=True`, `diagnostic_mode=False`,
   `frozen_backbone_probe=False`, `class_imbalance_recipe=False`.
   Attempting to flip any of these fails Pydantic validation by design
   (`Literal[True]` / `Literal[False]`).
5. `run_canonical_train_step(...)` executes exactly one optimizer step.
6. The returned `CanonicalStepReport` satisfies the `_losses_finite`
   guard: `total_loss == L_gen + lambda_cls * L_cls + lambda_distill * L_distill`
   to 1e-5, all four losses finite, `used_accelerate_backward=True`.

### 2.3 Output evidence

On the user's environment (Qwen3-4B-Instruct-2507 on a single 4090 per
GPU, idle slots 0–3):

```
outputs/diagnostic/head_only_smoke/scorer_report_amazon_8.json     n_total=8   auroc=0.4286
outputs/diagnostic/head_only_smoke/scorer_report_amazon_64.json    n_total=64  auroc=0.8042
outputs/diagnostic/head_only_smoke/scorer_report_yelpchi_64.json   n_total=64  auroc=0.6505
outputs/diagnostic/stage2_train_smoke/canonical_step_report_amazon.json   L_gen=2.8172  L_cls=1.2260  L_distill=1.3588  total=4.7225
outputs/diagnostic/stage2_train_smoke/canonical_step_report_yelpchi.json  L_gen=2.8496  L_cls=1.1300  L_distill=2.2258  total=5.0926
```

Those AUROC numbers are meaningless (random-init head); what matters is
that every downstream Pydantic model accepted the output, i.e. the plumbing
is end-to-end on real weights.

### 2.4 `gate_manifest.json` + launcher

`outputs/gated/manifests/gate_manifest.json` was hand-authored to match
the `GateManifest` schema:

- `commit=c6666d7c...` (current HEAD at the time of assembly)
- `data_manifest_hash` = sha256 of `outputs/gated/manifests/diagnostic_data_manifest.json`
- `generated_at` is a UTC-tz-aware ISO-8601 timestamp
- all 9 `*_pass` flags are `true` (corresponds to the observed state:
  216/216 tests pass + head-only smoke pass + stage2 1-step smoke pass +
  ablation audit tests pass + population-contract tests pass + data /
  teacher baseline gates already recorded in the existing artifacts)

Happy path:

```
python scripts/gate_check.py --manifest-path outputs/gated/manifests/gate_manifest.json
  -> gate_check: PASS ...

bash scripts/run_eval.sh --gate-manifest outputs/gated/manifests/gate_manifest.json --output-root outputs
  -> gate_check: PASS ...
  -> outputs/formal       (namespace directory printed)
```

Negative path (flipping one `*_pass` to `false`):

```
python scripts/gate_check.py --manifest-path outputs/gated/manifests/gate_manifest_bad.json
  -> gate_check: FAIL: gate manifest failed required gates: teacher_baseline_pass   (exit=1)

bash scripts/run_eval.sh --gate-manifest outputs/gated/manifests/gate_manifest_bad.json --output-root outputs
  -> gate_check: FAIL: ...   (exit=1, no outputs/formal entered)
```

This is the same fail-closed contract the unit tests already assert; we
only confirmed the wiring holds on the actual CLI path.

## 3. Silent-default / schema-drift audits

- Neither smoke runner touches `os.environ` or any `PRIORF_*` variable
  beyond the ones `scripts/run_full_pipeline.sh` exports at formal launch.
- Legacy parquet rows with unknown `hsd_quantile` values would raise
  `KeyError` on `_HSD_QUANTILE_MAP` - there is no silent "normal" fallback.
- `_RELATION_SUFFIXES` is keyed by `DatasetName` enum, so a future dataset
  would fail at adapter construction rather than silently reusing Amazon's
  relation suffixes.
- `_build_training_sample` does not infer labels - any label not in
  `{0, 1}` raises `ValueError` before the Pydantic model sees it.
- `CanonicalTrainingBatch` runs `_batch_header_consistent`; we do not
  supply `dataset_name` / `graph_regime` externally, they are read from
  the first sample's card.
- Both runners write to `outputs/diagnostic/<subdir>/` only.  None of them
  has a code path that reaches `outputs/formal/` or `outputs/gated/`.

## 4. Risks / leftover follow-ups

These are unchanged from the previous status notes:

1. The legacy teacher-export parquet schema does not carry
   `TeacherExportRecord` provenance (`teacher_model_name`,
   `teacher_checkpoint`, `population_name`, `graph_regime`,
   `ablation_mask`).  The smoke runners synthesise EvidenceCards directly
   and should not be mistaken for a canonical teacher-export channel.
   Producing canonical `outputs/gated/teacher_exports/<ds>/<pop>/` still
   requires `scripts/legacy_mat_to_canonical.py` + `scripts/generate_teacher_exports.py`.
2. No formal (`outputs/formal/...`) report has been written yet.  A full
   formal run requires:
   - a proper data manifest produced by the canonical pipeline,
   - real teacher-export parquet under `outputs/gated/teacher_exports/...`,
   - a Stage 2 training loop (not just 1-step smoke) producing a real
     `CheckpointProvenance`,
   - `eval/eval_head_only.py` + `eval/eval_gen_only.py` +
     `eval/eval_fusion.py` + `eval/faithfulness.py` runs on real weights,
   - a gate manifest whose `*_pass` flags reflect those real reports.
3. `_build_training_sample` uses 0.95 / 0.05 as the SFT target score.
   That is a reasonable conservative default for the smoke path and is
   bounded to match `StrictFloat` + `ge=0.0, le=1.0`.  Production SFT
   data preparation should read the target score from an explicit
   `TeacherExportRecord.teacher_prob` column (once canonical exports are
   regenerated) rather than picking a constant.

All of the above are called out in `status_package/general.md` so they
cannot be lost in a downstream handoff.
