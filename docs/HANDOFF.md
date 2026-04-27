# PriorF-Reasoner Handoff (2026-04-24)

> **Superseded / diagnostic-only notice (Stage 9.0c, 2026-04-27):** Pre-fix run_v2 / strat_v1 artifacts are diagnostic-only and superseded due to Evidence Card neighbor-label leakage discovered in Stage 4.5. Do not cite pre-fix run_v2 / strat_v1 metrics, gate manifests, or acceptance artifacts as clean formal results; regenerate under the current leakage-policy and prompt-audit gate before making formal claims.

All acceptance/pass language below is retained only as historical diagnostic context. It is not a current clean accepted result and must not be used for Stage 9.1 or later formal claims until regenerated under the current post-fix provenance and prompt-audit policy.


Handoff written after the post-QA tuned retraining (`run_v2`).  Paired
with `self-review/task_qa_and_tuned_training.md` and
`self-review/task_e2e_pipeline.md`; both are the authoritative primary
sources.  This file is the one-stop index for the next operator.

- **Repo root**: `/data1/mq/codes/graph-detection-main/PriorF-Reasoner`
- **Current HEAD**: `4674157` on `main`
- **Python env**: `/data1/anaconda3` (ruff 0.12.x, pyflakes 3.x, pytest 8.x,
  torch 2.x bf16, transformers, peft, accelerate)
- **Tests**: `PYTHONPATH=. python -m pytest tests/ -q` -> **216 passed**
- **Backbone used**: `/data1/mq/models/Qwen3-4B-Instruct-2507`
- **Datasets used**: Amazon (CIKM 2020) + YelpChi (CIKM 2020), canonical
  `.mat` regenerated under `assets/data/` and teacher exports under
  `outputs/gated/teacher_exports/{amazon,yelpchi}/`.

---

## 1. What has been done

### 1.1 Framework (Tasks 1-15) — already green before this handoff session

All 15 canonical tasks (data foundations, teacher contract, Evidence
Cards, hidden-state pooling, unified head scoring, parity test, gate
manifest, formal launcher, canonical joint trainer, formal head-only /
gen-only / fusion / faithfulness eval, teacher-prob ablation audit,
readme + runbook) were already implemented and tested.  They are
documented in `docs/010_project_contract.md`, `docs/020_top_level_design.md`,
`docs/030_harness_agent_execution_guide.md`, `docs/040_operator_runbook.md`,
`docs/050_fail_closed_guardrails.md`, `README.md`, and the
`self-review/` directory.  Zero files under `eval/`, `train/`,
`evidence/`, `schemas/`, `priorf_teacher/`, `graph_data/`, `llm/`, or
`tests/` were modified in this handoff session.

### 1.2 End-to-end runnable pipeline on real Qwen3-4B + Amazon/YelpChi

Added purely as driver scripts that compose the public canonical
contracts.  Commit sequence `eaefd59 -> b962032 -> 797a641 -> 4674157`.

**New driver scripts** (all under `scripts/`):

- `scripts/run_head_only_smoke.py` — diagnostic plumbing smoke for
  `score_head` on legacy parquet (`outputs/diagnostic/` only).
- `scripts/run_stage2_train_smoke.py` — 1-step Stage 2 smoke for
  `run_canonical_train_step`.
- `scripts/legacy_mat_to_canonical.py` — legacy CIKM `.mat` ->
  canonical `x / ground_truth_label / split_vector / relation_*` `.mat`.
- `scripts/generate_teacher_exports.py` — canonical `.mat` ->
  validated `TeacherExportRecord` parquet (train / validation /
  final_test) + `DataManifest` + `TeacherBaselineReport` with
  fail-closed git-sha pinning.
- `scripts/run_stage2_train.py` — multi-step canonical joint trainer
  driver wrapping `run_canonical_train_step` in a loop with LR
  scheduling, broader LoRA, epoch shuffling, and periodic validation.
- `scripts/run_stage2_inference.py` — head-only `score_head` inference
  driver over any TeacherExportRecord parquet -> `ScorerReport`.
- `scripts/run_formal_head_only_eval.py` — wraps
  `run_formal_head_only_eval` with a correct
  `FormalHeadOnlyCheckpointBundle` -> `FormalHeadOnlyReport`.
- `scripts/run_full_stage2_pipeline.sh` — one-command driver that
  chains legacy -> canonical -> teacher_exports -> train -> inference
  -> formal eval for a single dataset.

**One-command example**:

```bash
bash scripts/run_full_stage2_pipeline.sh \
  --dataset amazon \
  --qwen-path /data1/mq/models/Qwen3-4B-Instruct-2507 \
  --max-steps 500 --batch-size 2 \
  --max-train-samples 4096 \
  --validation-subset 128 --final-test-subset 512 \
  --skip-teacher-exports
```

### 1.3 Canonical teacher exports on real data (2026-04-24, `eaefd59`)

Produced under `outputs/gated/teacher_exports/<ds>/<pop>/teacher_export.parquet`
with `_git_head_sha_or_fail`-pinned `code_git_sha = eaefd594...`:

| dataset | train | validation | final_test | validation AUROC |
|---|---:|---:|---:|---:|
| Amazon  | 8 360  | 1 194  | 2 390  | **0.9744** (passed) |
| YelpChi | 32 167 | 4 595  | 9 192  | **0.9867** (passed) |

Plus per-dataset `manifests/<ds>/data_manifest.json`.

### 1.4 `run_v2` QA + tuned retraining (commits `797a641`, `4674157`)

> Historical context: this section's AUROC-leading numbers predate the
> 2026-04-27 metric / sampling convention.  See **§7.9** for the
> current AUPRC + stratified-batch contract that all new runs must
> follow.

**Quality pass on new scripts** (post-pipeline):

- `pyflakes` + `ruff --select=F,B,UP,SIM,ARG` over the five new drivers:
  one unused import + one duplicate import block fixed; no other
  functional issues.
- Contract cross-check against `score_head`,
  `run_canonical_train_step`, `run_formal_head_only_eval`,
  `build_evidence_card`: kwargs all match, no positional reordering,
  `CheckpointProvenance.content_hash` is always the real sha256 of the
  saved `cls_head.pt` (periodic in-loop monitor uses a clearly marked
  sentinel that cannot be promoted).
- 216/216 tests still pass.

**Diagnosed `run_v1` convergence problem**:

1. no LR schedule (flat lr=1e-4, distill loss oscillated 0.36 <-> 1.14
   in adjacent steps);
2. tiny trainable rank (LoRA on `{q_proj, v_proj}` only = 2.9 M params);
3. random-with-replacement sampler over a 1024-sample cap, yielding
   uneven positive / negative exposure;
4. resulting `prob_std = 0.001` on post-training validation — the head
   was ranking correctly but predictions collapsed into a tiny band.

**Trainer upgrades** (all in `scripts/run_stage2_train.py`; canonical
modules untouched):

- `--warmup-ratio` (default 0.1) + `--lr-scheduler {cosine,linear,constant}`
  via transformers schedulers.
- `--lora-targets` default expanded to
  `{q, k, v, o, gate, up, down}_proj` (33 M trainable params).
  `--lora-dropout`, `--weight-decay` added.
- Epoch-based shuffling: per-epoch permutation, every sample seen before
  any repeat.
- `--num-epochs` computes
  `total_steps = ceil(epochs * n_train / batch_size)`;
  `--max-steps` wins if both set; raises if neither set.
- `--validation-every-n-steps N` runs
  `run_validation_with_unified_scorer` on a pre-loaded stratified
  monitor subset every N steps; writes interim auroc/auprc/brier/
  prob_std rows to `train_log.jsonl`.
- Per-step LR logged.

**Tmux-backed `run_v2` training** (sessions `stage2-amazon-v2`,
`stage2-yelpchi-v2`, both completed):

```
--max-steps 500 --batch-size 2 --max-train-samples 4096
--learning-rate 1e-4 --warmup-ratio 0.1 --lr-scheduler cosine
--lora-r 16 --lora-alpha 32 --lora-dropout 0.05
--lambda-cls 1.0 --lambda-distill 0.3
--validation-subset 128 --validation-every-n-steps 50
```

### 1.5 Final `run_v2` metrics (all from canonical `FormalHeadOnlyReport`)

| | Amazon run_v1 | Amazon **run_v2** | YelpChi run_v1 | YelpChi **run_v2** |
|---|---:|---:|---:|---:|
| val-frozen threshold | 0.0647 | **0.2082** | 0.1394 | **0.1112** |
| final_test AUROC | 0.5093 | **0.9857** | 0.6287 | **0.9429** |
| final_test AUPRC | 0.0882 | **0.9051** | 0.1889 | **0.7233** |
| F1 @ val_thr | 0.1053 | **0.8696** | 0.1818 | **0.7368** |
| precision @ val_thr | 0.0690 | **0.8824** | 0.1750 | **0.7179** |
| recall @ val_thr | 0.2222 | **0.8571** | 0.1892 | **0.7568** |
| specificity @ val_thr | -- | **0.9916** | -- | **0.9498** |
| Brier | 0.0654 | **0.0331** | 0.1229 | **0.1097** |
| ECE | -- | 0.0783 | -- | 0.1035 |
| prob_std | 0.0040 | **0.0895** | 0.0130 | **0.0545** |

Amazon AUROC +48 absolute points, YelpChi AUROC +31 absolute points
versus `run_v1`.

---

## 2. What has not been done (explicit follow-ups)

Ordered by the expected next-step payoff.  Every item is scoped so the
next operator can start without re-discovering context.

### 2.1 Stage-2 training best-checkpoint support (high priority) ✅ COMPLETED

**Goal:** Allow training to track the best validation-metric checkpoint and emit a
`best_checkpoint/` directory alongside `final_checkpoint/`, without breaking
canonical contracts.
Target:

- Use the full TRAIN parquet (Amazon 8 360, YelpChi 32 167 rows).
- Run 2 - 4 epochs with bs=2 (Amazon ~8 k-16 k steps, YelpChi
  ~32 k-64 k steps).
- Add a best-checkpoint selector: the trainer already emits periodic
  validation AUROC to `train_log.jsonl` — extend it to
  (a) snapshot `peft_adapter/` + `cls_head.pt` on a new best AUROC,
  (b) at the end emit both `final/` and `best/` directories with
  matching `run_record.json` entries,
  (c) make `run_formal_head_only_eval` consume `best/`.
- Empirically sweep `{lr in [5e-5, 1e-4, 2e-4], lora_r in [16, 32],
  lambda_distill in [0.1, 0.3, 0.5]}`; the run_v2 config is a
  reasonable midpoint, not an optimum.

Commands to resume (reuse existing teacher_exports):

```bash
tmux new-session -d -s stage2-amazon-v3 '
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=/data1/mq/codes/graph-detection-main/PriorF-Reasoner \
  python scripts/run_stage2_train.py \
    --dataset amazon \
    --qwen-path /data1/mq/models/Qwen3-4B-Instruct-2507 \
    --teacher-export-train outputs/gated/teacher_exports/amazon/train/teacher_export.parquet \
    --teacher-export-validation outputs/gated/teacher_exports/amazon/validation/teacher_export.parquet \
    --data-manifest manifests/amazon/data_manifest.json \
    --output-dir outputs/gated/stage2/amazon/run_v3 \
    --num-epochs 2 --batch-size 2 \
    --learning-rate 1e-4 --warmup-ratio 0.05 --lr-scheduler cosine \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --lambda-cls 1.0 --lambda-distill 0.3 \
    --validation-subset 256 --validation-every-n-steps 500 \
    --gpu-index 0 2>&1 | tee outputs/gated/stage2/logs/amazon_run_v3.log'
```

### 2.2 Temperature calibration (high priority for fusion) ✅ COMPLETED

`run_v2` ECE is 0.08 (Amazon) / 0.10 (YelpChi) with
`max_calibration_gap` of 0.56 / 0.79.  The head is well-ordered but
under-calibrated.  `eval.calibration.compute_calibration_summary`
already computes ECE and is the contract; adding a validation-only
temperature fit is a self-contained module:

- Add `eval/temperature_scaling.py` with
  `fit_temperature_on_validation(logits: np.ndarray, labels: np.ndarray) -> float`
  (simple `minimize(nll, T=1.0)` with `0.1 <= T <= 10`).
- Extend `FormalHeadOnlyReport` with `temperature: float | None`; make
  `run_formal_head_only_eval` accept `apply_temperature_scaling: bool`
  and rescale the report population probabilities accordingly *after*
  threshold selection, preserving fail-closed semantics.
- Re-run `scripts/run_formal_head_only_eval.py` with the new flag on
  both datasets.

This is the smallest high-value gap.  Fusion and gating both rely on
well-calibrated probabilities.

### 2.3 CLI drivers for the remaining three formal runners ✅ COMPLETED

The library functions are implemented + tested + green:

- `eval/eval_gen_only.py` (strict-schema parse rate + normalized parse
  rate; see `self-review/task_08_formal_launcher_gate_integration.md`
  if the file exists in this repo copy),
- `eval/eval_fusion.py` (validation-only alpha search + student
  contribution gate; `llm/fusion.py` provides the math),
- `eval/faithfulness.py` (faithfulness formal runner, not yet CLI-driven)
- `eval/eval_faithfulness.py` (faithfulness formal runner, not yet CLI-driven).

What is missing is a CLI driver analogous to
`scripts/run_formal_head_only_eval.py`.  Template to follow:

- load PEFT adapter + cls_head + tokenizer via the same helpers as
  `run_stage2_inference.py`,
- generate Evidence-Card rationales with the backbone
  (`model.generate(..., max_new_tokens=...)` — follow the canonical
  prompt built by `evidence.prompt_builder.build_prompt`, enforce
  `ThinkingMode.NON_THINKING`),
- hand the generated strings + probabilities to the respective eval
  module,
- wrap the returned report in a dump + runtime provenance block.

Target output:

- `scripts/run_formal_gen_only_eval.py` ->
  `outputs/formal/gen_only/<ds>/formal_gen_only_report_<ds>.json`.
- `scripts/run_formal_fusion_eval.py` ->
  `outputs/formal/fusion/<ds>/formal_fusion_report_<ds>.json`.
- `scripts/run_formal_faithfulness.py` ->
  `outputs/formal/faithfulness/<ds>/faithfulness_report_<ds>.json`.

Do **not** write new eval schemas; use the existing Pydantic models.

### 2.4 Gate manifest auto-assembly (medium priority) ✅ COMPLETED

`scripts/gate_check.py` already validates a `GateManifest`; what is
missing is the script that reads the four formal reports and produces
one.

Concrete spec:

- `scripts/generate_gate_manifest.py`:
  - args: `--dataset {amazon,yelpchi} --runs-root outputs/formal
    --run-id <X> --output-path outputs/gated/manifests/gate_manifest_<ds>.json`
  - reads `formal_head_only_report_<ds>.json`,
    `formal_gen_only_report_<ds>.json`,
    `formal_fusion_report_<ds>.json`,
    `faithfulness_report_<ds>.json`,
    and `teacher_baseline_report.json`,
  - extracts the relevant pass fields
    (`teacher_baseline_pass`, `head_formal_pass`, `gen_strict_parse_pass`,
    `fusion_student_contribution_pass`, `faithfulness_sample_size_pass`,
    `teacher_prob_ablation_pass`, `graph_regime_match_pass`,
    `seed_stability_pass`, `evidence_hash_stable_pass`),
  - writes a `GateManifest` Pydantic object.

Then `scripts/run_eval.sh --gate-manifest ...` closes the last
silent-fallback hole: gate manifest is produced from real reports
instead of hand-written.

### 2.5 Teacher-prob ablation audit on `run_v2` (medium priority)

`evidence/ablations.py` is implemented; it was run on smoke-scale
Evidence Cards during Task 11.  Run it on the `run_v2` checkpoints
now that the student is actually discriminative:

- Produce an ablation probability trace: full Evidence Card vs
  card-with-teacher-prob-field-masked vs card-with-all-teacher-masked,
  on validation only.
- Read the `teacher_prob_dependency_high` flag from the produced
  `TeacherProbAblationAuditReport`.  `run_v2` may still flag this high
  because the student can lean on the teacher-probability field;
  that’s what the gate is for.

### 2.6 Held-out external test population (high-value, requires data)

All three `DataManifest.populations` in the current canonical teacher
exports have `contains_tuning_rows = True` except `final_test`, but
the `final_test` distribution is inherited from the teacher that also
produced the Evidence Card fields, so "generalization beyond the
teacher distribution" cannot be claimed yet.  For a real external
claim:

- acquire or construct a second graph source whose labels were *not*
  seen by the LGHGCLNet teacher,
- rerun `scripts/generate_teacher_exports.py --population external_test`
  (requires a schema extension because `PopulationName` today is
  `{train, validation, final_test}`),
- feed into `run_formal_head_only_eval.py` as the report population.

This is the only follow-up that requires a data-side change, not just
code.

### 2.7 Minor cosmetic / maintenance items

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is deprecated in
  the torch version used; switch to `PYTORCH_ALLOC_CONF` in the
  launchers.
- `transformers` warns `torch_dtype` is deprecated; switch to `dtype`
  in `scripts/run_stage2_train.py`, `run_stage2_inference.py`,
  `run_formal_head_only_eval.py`.
- `pyproject.toml` (if present) could add a `[tool.ruff]` config with
  the F + B rules enabled so CI catches future drift.

---

## 3. File map to get oriented fast

### 3.1 Newly-added driver scripts (entry points)

- `scripts/run_stage2_train.py`        (upgraded trainer — LR sched, LoRA, epochs, monitor)
- `scripts/run_stage2_inference.py`    (head-only ScorerReport)
- `scripts/run_formal_head_only_eval.py` (FormalHeadOnlyReport)
- `scripts/run_full_stage2_pipeline.sh` (chains 5 steps)
- `scripts/generate_teacher_exports.py` (canonical parquet + manifest)
- `scripts/legacy_mat_to_canonical.py`  (legacy .mat -> canonical .mat)
- `scripts/run_head_only_smoke.py`      (diagnostic plumbing)
- `scripts/run_stage2_train_smoke.py`   (1-step trainer smoke)

### 3.2 Untouched but key existing contracts

- `eval/head_scoring.py`             (`score_head`, `HeadScoringInputs`)
- `eval/eval_head_only.py`           (`run_formal_head_only_eval`, `FormalHeadOnlyReport`)
- `eval/calibration.py`              (`compute_calibration_summary`, threshold selection)
- `eval/eval_gen_only.py`            (generation formal runner, not yet CLI-driven)
- `eval/eval_fusion.py`              (fusion formal runner, not yet CLI-driven)
- `eval/faithfulness.py`             (faithfulness formal runner, not yet CLI-driven)
- `train/train_stage2_canonical.py`  (`run_canonical_train_step`, `run_validation_with_unified_scorer`)
- `evidence/evidence_schema.py`      (`build_evidence_card`, `EvidenceCard`)
- `evidence/output_schema.py`        (`PredLabel`, strict output schema)
- `evidence/ablations.py`            (teacher-prob ablation audit)
- `graph_data/manifests.py`          (`DataManifest`, population metadata)
- `priorf_teacher/export_pipeline.py`(`read_teacher_export_artifact`)
- `priorf_teacher/schema.py`         (`TeacherExportRecord`, enums)

### 3.3 Runtime artifacts (gitignored under `outputs/`)

- `outputs/gated/teacher_exports/{amazon,yelpchi}/{train,validation,final_test}/teacher_export.parquet`
- `outputs/gated/stage2/{amazon,yelpchi}/run_v2/{peft_adapter/,cls_head.pt,run_record.json,train_log.jsonl}`
- `outputs/gated/stage2/logs/{amazon,yelpchi}_run_v2.log`
- `outputs/gated/eval/{amazon,yelpchi}/scorer_report_*_final_test.json`
- `outputs/formal/head_only/{amazon,yelpchi}/formal_head_only_report_*.json`
- `manifests/{amazon,yelpchi}/data_manifest.json`

### 3.4 Docs that describe the contract (do not drift from these)

- `README.md`
- `docs/010_project_contract.md`
- `docs/020_top_level_design.md`
- `docs/030_harness_agent_execution_guide.md`
- `docs/040_operator_runbook.md`
- `docs/050_fail_closed_guardrails.md`
- `status_package/general.md`  (running status; updated through `run_v2`)
- `self-review/task_e2e_pipeline.md`
- `self-review/task_qa_and_tuned_training.md`

---

## 4. How to resume in one session

1. `git log -5` — expect `4674157` at HEAD.
2. `PYTHONPATH=. python -m pytest tests/ -q` — expect 216 passed.
3. Pick a follow-up from §2.  §2.1 (full-TRAIN + best-checkpoint) and
   §2.2 (temperature calibration) are the two highest-payoff items and
   are independent.
4. If §2.3 (extra formal CLI drivers) is picked, use
   `scripts/run_formal_head_only_eval.py` as the template.
5. Commit with the existing convention: short title + mechanical bullet
   list of what changed + what was verified; do not weaken or delete
   tests; do not edit modules under `eval/`, `train/`, `evidence/`,
   `schemas/`, `priorf_teacher/`, `graph_data/`, `llm/`, or `tests/`
   unless the task explicitly requires it.

## 5. Verified open invariants

These all hold at HEAD `4674157`.  If any of them breaks, it is a
regression that must be reverted before moving on.

- **216 tests pass.**  No test was weakened or deleted.
- **Canonical modules unchanged** since the pre-`eaefd59` baseline.
- **Fail-closed guardrails:** every `TeacherProvenance.code_git_sha` is
  a real 40-hex HEAD from `_git_head_sha_or_fail`; every
  `FormalHeadOnlyReport` passes every Pydantic model_validator; every
  `CanonicalStepReport` satisfies
  `|total - (L_gen + lambda_cls*L_cls + lambda_distill*L_distill)| < 1e-5`.
- **`outputs/formal/` is only reachable via the gated formal eval
  path.**  No driver writes under `outputs/formal/` without a
  `FormalHeadOnlyReport` that passes every validator.
- **216 tests pass.**  Worth stating twice; this is the tripwire.

---

## 6. Shortcut-learning fix and current verification state (2026-04-24)

This section supersedes the older "canonical modules unchanged" and "216 tests
pass" statements above for the current working tree.  Current checked HEAD
during this update was `661b89f`, with uncommitted working-tree changes.

### 6.1 Root cause found

Generation failed because the student had multiple shortcut paths and one
training-objective bug:

- `scripts/run_stage2_train.py` copied `teacher_record.teacher_prob` into
  `sft_target_score`, so the assistant target directly taught teacher-prob
  copying.
- `train/train_stage2_canonical.py` computed generation loss over the full
  chat transcript, including system/user prompt tokens, instead of masking to
  assistant-only JSON target tokens.
- `scripts/_formal_eval_helpers.py` generated from an eval bundle that still
  included an empty assistant message and did not use the Qwen chat-template
  generation prompt.
- Even after removing `teacher_prob` from the SFT target, the Evidence Card
  rendered `teacher_summary.teacher_prob` and `teacher_summary.teacher_logit`
  directly to the student.  A 1-sample GPU probe showed the model immediately
  began generated JSON by restating `teacher_prob`, confirming this was still a
  live shortcut.

### 6.2 Code changes made in this working tree

- `evidence/evidence_schema.py`
  - Added `STUDENT_PROMPT_ABLATION_MASK`.
  - Added `build_student_evidence_card(...)`, which masks
    `teacher_summary.teacher_prob` and `teacher_summary.teacher_logit` for
    student-visible prompts while preserving schema-valid ablation metadata.
- `evidence/prompt_builder.py`
  - SFT targets now use Evidence Card structural fields in rationale/evidence
    text and no longer serialize `teacher_prob` into the assistant target.
- `train/train_stage2_canonical.py`
  - Generation loss now builds an assistant-only label mask using a
    chat-template prefix encode with `add_generation_prompt=True`.
  - `CanonicalTrainingSample.teacher_prob` remains the explicit distillation
    target.  If the Evidence Card masks `teacher_prob`, the mask must be
    declared in `ablation_mask`; otherwise unmasked card values are still
    cross-checked against the explicit target.
- Student driver paths now use `build_student_evidence_card(...)`:
  - `scripts/run_stage2_train.py`
  - `scripts/run_stage2_inference.py`
  - `scripts/run_formal_head_only_eval.py`
  - `scripts/run_formal_gen_only_eval.py`
  - `scripts/_formal_eval_helpers.py`
- `scripts/run_full_pipeline.sh`
  - Fixed launcher code-127 risk from bare `python` by resolving the Python
    executable from `$PYTHON`, the command after `--`, or PATH before running
    `scripts/gate_check.py`.
- Local Codex/OMX config outside the repo was also adjusted:
  - OMX notify hook was disabled for the Codex plugin flow because `node` was
    not on PATH and caused `hook exited with code 127`.
  - Spark/explore config was switched from the old spark model to
    `gpt-5.4-mini`.

### 6.3 Tests and static checks

Use `/data1/anaconda3` for full pytest/ruff and
`/data1/mq/conda_envs/priorfgnn` for GPU runs.

Verified:

```bash
/data1/anaconda3/bin/python -m pytest tests/ -q
# 231 passed in 19.98s

/data1/anaconda3/bin/python -m ruff check \
  evidence/evidence_schema.py evidence/prompt_builder.py \
  train/train_stage2_canonical.py \
  scripts/_formal_eval_helpers.py scripts/run_stage2_train.py \
  scripts/run_stage2_inference.py scripts/run_formal_head_only_eval.py \
  scripts/run_formal_gen_only_eval.py \
  tests/test_evidence_schema.py tests/test_prompt_builder.py \
  tests/test_canonical_trainer.py tests/test_formal_eval_driver_smoke.py \
  tests/test_stage2_train_driver.py
# All checks passed

bash -n scripts/run_full_pipeline.sh
git diff --check
```

`/data1/anaconda3` and sandboxed `/data1/mq/conda_envs/priorfgnn` could not
initialize CUDA inside the sandbox (`torch.cuda.is_available() == False`,
`Can't initialize NVML`).  Escalated/non-sandbox
`/data1/mq/conda_envs/priorfgnn` saw all 8 GPUs and was used for GPU
verification.

### 6.4 GPU smoke runs

The user requested using the least-occupied GPUs, including GPU5/6/7.  GPUs
5/6/7 had the most free memory among usable cards, so three lanes were run in
parallel.

Unmasked intermediate smoke, before masking student-visible teacher scores:

- `outputs/gated/stage2/amazon/shortcut_fix_smoke_gpu5_base_20260424_escalated`
- `outputs/gated/stage2/amazon/shortcut_fix_smoke_gpu6_lowdistill_20260424_escalated`
- `outputs/gated/stage2/amazon/shortcut_fix_smoke_gpu7_lowlr_20260424_escalated`
- Gen-only reports under `outputs/verification/gen_only/shortcut_fix_smoke_*`
  all had `strict_schema_parse_rate=0.0` and
  `normalized_parse_rate=0.0`.

Masked student-card smoke, after masking `teacher_prob` and `teacher_logit`:

| lane | best step | best validation AUROC | best validation AUPRC | gen strict parse | gen normalized parse |
|---|---:|---:|---:|---:|---:|
| `masked_shortcut_smoke_gpu5_base_20260424` | 4 | 0.5333 | 0.1250 | 0.0000 | 0.2500 |
| `masked_shortcut_smoke_gpu6_lowdistill_20260424` | 4 | 0.2000 | 0.0769 | 0.0000 | 0.3750 |
| `masked_shortcut_smoke_gpu7_lowlr_20260424` | 2 | 0.0667 | 0.0667 | 0.0000 | 0.2500 |

Gen-only reports:

- `outputs/verification/gen_only/masked_shortcut_smoke_gpu5_base_20260424/formal_gen_only_report_amazon.json`
- `outputs/verification/gen_only/masked_shortcut_smoke_gpu6_lowdistill_20260424/formal_gen_only_report_amazon.json`
- `outputs/verification/gen_only/masked_shortcut_smoke_gpu7_lowlr_20260424/formal_gen_only_report_amazon.json`

Interpretation: the masked-card path is healthier than the unmasked path
(`normalized_parse_rate` is no longer always zero), but **strict schema parse
is still zero on these 8-sample, 4-step smokes**.  These are not formal-passing
artifacts and must not be promoted into a gate manifest.

### 6.5 Current completion state

Completed in this working tree:

- The known teacher-prob shortcut in SFT targets is removed.
- Student-visible prompts no longer expose direct `teacher_prob` or
  `teacher_logit` values by default.
- Distillation still has access to the explicit `teacher_prob` soft target.
- Generation loss is assistant-only.
- Formal generation uses the correct assistant generation prompt.
- Hook code-127 from OMX notify was addressed for the Codex plugin flow.
- Unit/static verification is green.

Still not complete:

- Gen-only formal strict schema is not recovered yet:
  `strict_schema_parse_rate=0.0` in all masked smokes.
- The existing `outputs/gated/manifests/gate_manifest.json` is stale and
  should be treated as invalid for current claims; it still reports strict
  schema and student-contribution gates as true despite current gen/fusion
  evidence failing.
- Existing `outputs/formal/gen_only/amazon_smoke_best_ckpt/...` and
  `outputs/formal/fusion/amazon_smoke_best_ckpt/...` remain evidence of the
  pre-fix failure mode, not passing formal artifacts.

### 6.6 Recommended next steps

1. Run a real masked-card structured generation recovery job, not a 4-step
   smoke.  Use the same GPU priority rule (least memory used among available
   GPUs; GPU5/6/7 were usable in this session).
2. Shorten or constrain generated assistant targets.  Current Qwen generations
   can be long and verbose; strict parsing fails even when normalized parsing
   partially recovers.  Consider a compact target rationale/evidence style or
   a schema-constrained decoding/retry path, while keeping strict parse as the
   formal headline metric.
3. Re-run gen-only with larger `max_new_tokens` and inspect raw continuations
   before any formal claim.  The smoke used `max_new_tokens=768`; strict still
   failed.
4. Re-run fusion only after gen-only strict parsing recovers.  Fusion reports
   with `optimal_alpha=0.0` or `student_contribution_pass=False` must remain
   non-promotable.
5. Regenerate gate manifests only from fresh reports after all gates pass; do
   not hand-edit stale gate JSON.

## 7. Amazon end-to-end reproducibility audit — provenance enrichment (current session, 2026-04-24)

### 7.1 Context

User objective: complete a focused reproducibility audit on the **Amazon** dataset, deferring YelpChi to smoke/compatibility only.  Priority order: P0 fix provenance → P1 head-only/fusion/gate/faithfulness closure → P2 Stage 1/2 gaps → P3 generation audit → P4 tests + acceptance report.

### 7.2 Changes made in this working tree (not yet committed)

**Shared provenance helpers** (`scripts/_formal_eval_helpers.py`):
- Added `canonical_json_sha256()`, `selected_node_ids_sha256()`, `selected_records_sha256()` — deterministic SHA256 of sorted/ordered evaluation subsets.
- Added `current_python_command()` — captures `sys.executable + sys.argv`.
- Added `capture_git_state(repo_root)` — `git rev-parse HEAD` + `git status --porcelain` dirty flag (fail-closed on missing git).
- Added `build_subset_runtime_provenance(...)` — reusable assembly of population name, total size, requested subset size, actual subset size, full-pop flag, teacher export path + sha256, selected node-id hash, selected record hash.

**Formal evaluation drivers** — all four updated to embed the same enriched `_runtime_provenance`:
- `scripts/run_formal_head_only_eval.py`
- `scripts/run_formal_fusion_eval.py`
- `scripts/run_formal_gen_only_eval.py`
- `scripts/run_formal_faithfulness.py`

Each now records: `dataset`, `checkpoint_source/type`, `run_id`, `training_commit`, `seed`, `evaluation_command`, `git_commit`, `git_dirty`, plus the per-population subset provenance block above.  Replaced local `_stratified` / `_file_sha256` duplicates with shared helpers.

**Gate manifest schema upgrade** (`schemas/gate_manifest.py`):
- Added `dataset_name: DatasetName` top-level field.
- Added `GateArtifactReference(path, exists, sha256)` with fail-closed validator (`sha256` required when `exists=True`, forbidden when `exists=False`).
- Added `GateManifestProvenance(run_id, data_manifest_path, teacher_baseline_report, head_only_report, fusion_report, gen_only_report, faithfulness_report, generator_command, generator_git_commit, generator_git_dirty)`.
- `GateManifest` now requires `provenance: GateManifestProvenance`.

**Gate manifest generator** (`scripts/generate_gate_manifest.py`):
- Replaced local `_compute_data_manifest_hash` with shared `file_sha256`.
- Populates `dataset_name`, `provenance.*`, artifact path/hash references, generator command, and generator git state from shared helpers.
- Keeps existing real-artifact compatibility logic (head pass from `headline_metrics.auroc`, nested dataset/graph_regime extraction, `_runtime_provenance` stripping before schema validation).

**Test fixtures updated** to satisfy the stricter schema:
- `tests/test_gate_check.py` — `_manifest_payload()` now includes `dataset_name` and full `provenance` block with all required artifact references.
- `tests/test_formal_launcher_gate.py` — `_write_manifest()` updated similarly.
- `tests/test_smoke_pipeline.py` — the two inline manifest payloads (passing + failing) must also be updated before the next pytest run (currently they still lack `dataset_name` and `provenance`).

### 7.3 Immediate next steps (in order)

1. **Run full pytest** to verify schema changes did not break existing tests:
   ```bash
   /data1/anaconda3/bin/python -m pytest tests/ -q
   ```
   If `test_smoke_pipeline.py` fails because of the missing `dataset_name` / `provenance` fields, patch the two inline payloads there first.

2. **Update `tests/test_generate_gate_manifest.py`** to test the new provenance fields (artifact references exist/hash alignment, generator command/git state presence).

3. **P0 closure** — once tests are green, the provenance contract is complete.  Proceed to P1:
   - Re-run Amazon head-only evaluation with the enriched driver on the existing `run_v2` checkpoint.
   - Re-run fusion, gen-only, faithfulness on Amazon (reuse existing teacher exports and checkpoint artifacts).
   - Regenerate `gate_manifest_amazon.json` with the upgraded generator.

4. **P2 Stage 1 gap** — `train/stage1_sft.py` is still missing (README references `python -m train.stage1_sft` which does not exist).  Decide whether to implement a minimal Stage 1 SFT loop (TRL SFTTrainer or custom) or document the fallback that Stage 2 joint training is the effective first training phase for acceptance.

5. **P3 generation audit** — inspect raw Amazon generation outputs for syntax and semantic consistency; confirm strict schema parse rate is at or near the previously observed ~0.99–1.0 on validation subset 128 / 512.

6. **P4 tests + acceptance report** — run targeted tests, full pytest, touched-files ruff; confirm YelpChi smoke tests still pass; produce final Amazon acceptance report.

### 7.4 Files touched in this session (for ruff / diff review)
 
- `scripts/_formal_eval_helpers.py`
- `scripts/run_formal_head_only_eval.py`
- `scripts/run_formal_fusion_eval.py`
- `scripts/run_formal_gen_only_eval.py`
- `scripts/run_formal_faithfulness.py`
- `schemas/gate_manifest.py`
- `scripts/generate_gate_manifest.py`
- `tests/test_gate_check.py`
- `tests/test_formal_launcher_gate.py`
- `tests/test_smoke_pipeline.py`

### 7.5 Current execution status after provenance upgrade (2026-04-24 21:50 UTC+8)

**Verification completed**:

- Full pytest after schema/provenance changes:
  ```bash
  /data1/anaconda3/bin/python -m pytest tests/ -q
  # 241 passed
  ```
- Amazon formal reruns on `outputs/gated/stage2/amazon/run_v2/` completed successfully using the upgraded drivers:
  - `outputs/formal/head_only/amazon_run_v2_final/formal_head_only_report_amazon.json`
    - validation-fit temperature = `0.7586`
    - validation-frozen threshold = `0.0914`
    - final_test AUROC = `0.9892`
    - final_test AUPRC = `0.8765`
  - `outputs/formal/fusion/amazon_run_v2_final/formal_fusion_report_amazon.json`
    - optimal alpha = `0.8000`
    - `student_contribution_pass = True`
    - report fusion AUPRC = `0.9227608135502422`
  - `outputs/formal/gen_only/amazon_run_v2_final/formal_gen_only_report_amazon.json`
    - `strict_schema_parse_rate = 1.0`
    - `normalized_parse_rate = 1.0`
    - AUROC = `0.46825396825396826`
    - AUPRC = `0.07074365286041137`
  - `outputs/formal/faithfulness/amazon_run_v2_final/faithfulness_report_amazon.json`
    - `mean_sufficiency = 0.0309`
    - `mean_comprehensiveness = 0.0242`
    - `teacher_prob_flip_rate = 0.0703`

**Gate closure completed**:

- Because `scripts/generate_gate_manifest.py` resolves reports at flat paths like `outputs/formal/head_only/formal_head_only_report_amazon.json`, the rerun reports from the namespaced `amazon_run_v2_final/` directories were also copied into the flat `outputs/formal/{head_only,fusion,gen_only,faithfulness}/` roots.
- Fresh manifest generated successfully:
  - `outputs/gated/manifests/gate_manifest_amazon.json`
- Gate status from generator:
  - `teacher_baseline_pass=True`
  - `subset_head_gate_pass=True`
  - `strict_schema_parse_pass=True`
  - `student_contribution_pass=True`
  - `teacher_prob_ablation_pass=True`
  - `population_contract_pass=True`
- Gate check passes:
  ```bash
  /data1/anaconda3/bin/python scripts/gate_check.py \
    --manifest-path outputs/gated/manifests/gate_manifest_amazon.json
  # PASS
  ```

**Important provenance caveats still to resolve before final acceptance wording**:

- `outputs/gated/stage2/amazon/run_v2/run_record.json` does **not** currently expose an explicit training git commit.  It does expose `_runtime_provenance.config_fingerprint_sha256 = 74dab540cd9dbfda80265e0005d141551d9dfcb9bc8b26b66cf53a1030a11efb`, but no original training `git_commit` was recoverable from that file during this session.
- The rerun evals were therefore invoked with `--commit 661b89ff7a4e3c180ae4940a8395a3a70b04c31c` (current HEAD of the modified working tree).  This is acceptable as **evaluation provenance**, but may not be the exact original **training provenance** for `run_v2`.  If the acceptance bar requires exact train/eval commit separation, recover the original training commit from older logs or historical artifacts.
- The new formal reports show `git_dirty=True` in `_runtime_provenance`, because the provenance/schema upgrade work was uncommitted at evaluation time.  For a pristine final acceptance bundle, commit the current changes first and re-run the four formal eval drivers plus `generate_gate_manifest.py` so `git_dirty=False`.

**Recommended next steps from here**:

1. Commit the provenance/schema/test changes, then rerun the four Amazon formal drivers and `generate_gate_manifest.py` once more so the final reproducibility artifacts have a clean git state.
2. Write the final Amazon acceptance report using the above metrics and the fresh passing gate manifest.
3. Decide whether to implement minimal Stage-1 SFT or document Stage-2-only fallback explicitly, since `train.stage1_sft` is still missing.

### 7.6 Acceptance-round continuation (2026-04-24 22:40 UTC+8)

The missing Stage-1 gap is now closed:

- Added `train/stage1_sft.py` as a minimal runnable structured-SFT driver.
- Added targeted regression coverage in `tests/test_stage1_sft.py`.
- Updated `README.md` and `docs/040_operator_runbook.md` to reflect that Stage 1 now exists, while Stage 2 warm-start from Stage 1 is still *not* claimed in this acceptance round.

Generation audit closure also progressed:

- Added `scripts/run_generation_audit.py` plus `tests/test_generation_audit.py`.
- Fresh audit artifact written at:
  - `outputs/audits/generation/amazon_run_v2_final_subset32/generation_audit_amazon_final_test.json`
  - `outputs/audits/generation/amazon_run_v2_final_subset32/raw_generations_amazon_final_test.jsonl`
- Audit result on fixed subset-32 final_test batch:
  - `strict_schema_parse_rate = 1.0`
  - `normalized_parse_rate = 1.0`
  - `auroc = 0.18333333333333335`
  - `auprc = 0.05622188905547226`
  - `brier_score = 0.250889379417216`
- Semantic boundary from the 12 reviewed samples:
  - label / score directionality was internally consistent,
  - but rationale / evidence / pattern_hint collapsed to generic placeholder text,
  - therefore syntax pass does **not** imply semantic usefulness.

Important correctness fix discovered during the audit:

- `eval/eval_gen_only.py` had been interpreting `StrictOutput.score` as label-confidence (`benign -> 1-score`), but Stage 1 / Stage 2 SFT targets supervise `score` as **fraud probability** (`fraud=0.95`, `benign=0.05`).
- Fixed `_predicted_positive_probability()` to always return `output.score`.
- Added explicit regression coverage in `tests/test_eval_gen_only.py`.
- Re-ran `outputs/formal/gen_only/amazon_run_v2_final/formal_gen_only_report_amazon.json` after the fix:
  - `strict_schema_parse_rate = 1.0`
  - `normalized_parse_rate = 1.0`
  - `auroc = 0.19701213818860877`
  - `auprc = 0.04673922093376829`
- Copied the refreshed gen-only report to the flat path expected by `scripts/generate_gate_manifest.py`, regenerated `outputs/gated/manifests/gate_manifest_amazon.json`, and verified `scripts/gate_check.py` still passes.

Acceptance report now exists:

- `docs/060_amazon_acceptance_report.md`
- Current conclusion in that report:
  - reproducibility / pipeline closure = pass,
  - gate closure = pass,
  - gen-only syntax compliance = pass,
  - gen-only semantic usefulness / standalone discrimination = not demonstrated.

Verification status after the latest code changes:

- `/data1/anaconda3/bin/python -m pytest tests/test_stage1_sft.py tests/test_generation_audit.py tests/test_eval_gen_only.py -q --tb=short` -> pass
- `/data1/anaconda3/bin/python -m pytest tests/ -q --tb=short` -> `249 passed`
- touched-files ruff including the new audit / eval files -> pass

### 7.7 Immediate next step if you resume from here

1. For a pristine archival bundle, commit the repo and rerun the Amazon formal drivers plus `generate_gate_manifest.py` once more so `_runtime_provenance.git_dirty=False`.
2. If desired, enrich `scripts/run_generation_audit.py` with more explicit semantic-collapse summary fields (for example placeholder-rate or distinct-text counts), although the current audit already captures the key conclusion in the report.

### 7.8 Warm-start recovery follow-up (2026-04-24 23:45 UTC+8)

Focused Amazon follow-up experiments after the acceptance report established a cleaner root-cause boundary for the weak gen-only lane:

- Baseline re-audit of the existing `run_v2` adapter on validation subset 64 remained syntax-only:
  - artifact: `outputs/verification/generation_audit/amazon_run_v2_validation64/generation_audit_amazon_validation.json`
  - `strict_schema_parse_rate = 1.0`
  - `normalized_parse_rate = 1.0`
  - semantic-usefulness rates all `0.0`
  - `auroc = 0.13958333333333334`
  - `auprc = 0.04424802187361748`
- Shortening numeric rendering in `evidence/prompt_builder.py` SFT targets materially improved learnability.
- New Stage-1 compact-target smoke:
  - artifact root: `outputs/verification/stage1_sft/amazon_compact_smoke`
  - validation64 audit: `outputs/verification/generation_audit/amazon_stage1_compact_smoke_validation64/generation_audit_amazon_validation.json`
  - result: strict / normalized parse `1.0`, semantic-usefulness rates all `1.0`, `auroc = 0.75`, `auprc = 0.53125`
- Stage-2 driver now supports warm-starting from a Stage-1 PEFT adapter via `--warm-start-peft-adapter`.
- Stage-2 warm-start smoke:
  - artifact root: `outputs/verification/stage2/amazon_warm_start_smoke`
  - head validation on the 1024-sample / 200-step smoke remained underfit (`best validation_auroc ≈ 0.598973`, low `prob_std`), but generation retention was preserved.
  - final quick validation12 audit: `outputs/verification/generation_audit/amazon_stage2_warm_start_quick_validation12/generation_audit_amazon_validation.json`
  - best-checkpoint quick validation12 audit: `outputs/verification/generation_audit/amazon_stage2_warm_start_best_quick_validation12/generation_audit_amazon_validation.json`
  - both quick audits showed strict / normalized parse `1.0`, semantic-usefulness rates all `1.0`, and `auroc = 1.0`, `auprc = 1.0` on the tiny diagnostic subset.

Practical conclusion at this checkpoint:

- The previous gen-only failure was not primarily a parser problem.
- `compact Stage-1 target + Stage-1 warm-start into Stage-2` is sufficient to recover evidence-grounded generation behavior.
- The main unresolved issue is now head recovery under realistic Stage-2 training budget, not generation collapse.

Observability / ETA improvements added in this follow-up:

- `train/stage1_sft.py`
  - prints graph regime, label counts, trainable / total params, warmup steps
  - periodic logs now include epoch progress, elapsed time, ETA, and steps/sec
- `scripts/run_stage2_train.py`
  - prints graph regime, label counts, warm-start adapter path, elapsed time, ETA, steps/sec
  - periodic validation logs now include validation elapsed time
- `scripts/_formal_eval_helpers.py`
  - `generate_structured_outputs(...)` now supports optional progress logging with elapsed / ETA / records-per-second
- `scripts/run_generation_audit.py`
  - prints label counts and passes `progress_label="audit-gen"`
- `scripts/run_formal_gen_only_eval.py`
  - prints label counts and passes `progress_label="formal-gen"`

Targeted verification after the observability changes:

- `/data1/anaconda3/bin/python -m ruff check scripts/_formal_eval_helpers.py scripts/run_generation_audit.py scripts/run_formal_gen_only_eval.py scripts/run_stage2_train.py train/stage1_sft.py tests/test_formal_eval_driver_smoke.py` -> pass
- `/data1/anaconda3/bin/python -m pytest tests/test_formal_eval_driver_smoke.py tests/test_generation_audit.py tests/test_stage1_sft.py tests/test_stage2_train_driver.py -q --tb=short` -> `27 passed`

Active long-running job at the time of this handoff update:

- Session / log:
  - `tmux session = amazon-stage2-warm-full`
  - log file = `/tmp/amazon_stage2_warm_full500.log`
- Command intent:
  - full Amazon Stage-2 warm-start run using `outputs/verification/stage1_sft/amazon_compact_smoke/peft_adapter`
  - output dir = `outputs/verification/stage2/amazon_warm_start_full500`
  - config = full TRAIN, `max_steps=500`, `batch_size=2`, `lambda_cls=1.0`, `lambda_distill=0.3`, validation subset `256`, validation every `50` steps
- Startup status observed:
  - step `1/500` reached successfully with the new ETA logging enabled
  - initial log line reported `L_gen = 0.0264`, confirming the warm-start path is active rather than cold-starting generation from scratch

Recommended next step from here:

1. Let `amazon-stage2-warm-full` finish and compare:
   - best validation AUROC / AUPRC
   - validation `prob_std`
   - generation audit on both `best_checkpoint/peft_adapter` and final `peft_adapter`
2. If head recovery still lags despite full-data warm-start, move to optimizer / loss-balance diagnosis rather than changing the generation stack again.

### 7.9 Current Amazon status (2026-04-27)

Hard project conventions that the next operator must respect:

- **Primary metric**: `AUPRC`.  AUROC is *informational only* on this severely
  imbalanced data (Amazon train ≈ 13.5 : 1, YelpChi train ≈ 6 : 1).
  `scripts/run_stage2_train.py --best-checkpoint-metric` defaults to
  `validation_auprc`.  Any best-checkpoint, fusion, or report comparison
  must use AUPRC (and `prob_std` / `Brier`) as the headline; AUROC is
  only used as a secondary sanity number.
- **Stratified sampling**: every population subset and every TRAIN batch
  must preserve the positive / negative ratio.
  - validation / final-test subsets in formal eval already use
    `scripts/_formal_eval_helpers.stratified_records(...)`.
  - Stage-2 in-loop monitor and post-training validation now use the
    same stratified helper (`_stratified_record_subset`) inside
    `scripts/run_stage2_train.py`.
  - Stage-2 TRAIN batches now use `_next_stratified_batch_indices(...)`
    so each batch contains at least one positive and one negative.
- **Failure modes**: random-permutation batching plus
  AUROC-driven best-checkpoint selection (the previous default) are
  the two structural reasons earlier runs collapsed gen-only label
  prediction to the majority class while still scoring fine on AUROC.

Code changes that locked these conventions in (all in this session,
2026-04-27):

- `scripts/run_stage2_train.py`
  - default `--best-checkpoint-metric = validation_auprc`
  - per-batch sampling via `_next_stratified_batch_indices(...)`
  - in-loop monitor and post-training validation subsets via
    `_stratified_record_subset(...)` with explicit pos / neg counts
    in the log line
- `tests/test_stage2_train_driver.py`
  - new regressions: AUPRC default, stratified subset proportions,
    per-batch helper always includes both classes, rejects single-class
    pools and `batch_size < 2`
- `train/_tb_logger.py` (new) + wired into both
  `scripts/run_stage2_train.py` and `train/stage1_sft.py`
  - per-step training scalars (`train/lr`, `train/L_gen`, `train/L_cls`,
    `train/L_distill`, `train/total`, `train/steps_per_sec`,
    `train/samples_seen`)
  - in-loop validation scalars (`validation/auroc`, `validation/auprc`,
    `validation/brier`, `validation/prob_std`, `validation/elapsed_seconds`)
  - post-training validation scalars under `post_train_validation/*`
  - default log dir = `<output-dir>/tb`; disable with `--no-tensorboard`
  - inspect with `tensorboard --logdir <output-dir>/tb`
- `tests/test_tb_logger.py` (new) - smoke tests confirming events are
  written, parsed by `EventAccumulator`, and that disabled mode is a
  filesystem no-op (4 tests, all pass)

Verification:

- `/data1/anaconda3/bin/python -m ruff check scripts/run_stage2_train.py train/stage1_sft.py train/_tb_logger.py tests/test_stage2_train_driver.py tests/test_tb_logger.py` -> pass
- `/data1/anaconda3/bin/python -m pytest tests/ -q --tb=short` -> `262 passed`

#### 7.9.1 Status of older Amazon training runs

Everything below was produced **before** the AUPRC + stratified-batching
fix, so they are kept only as historical references and should not be
used as Amazon baselines going forward:

- `outputs/gated/stage2/amazon/run_v2/` - last accepted Stage-2 run from
  the original acceptance bundle; head is still well-ordered (final-test
  AUPRC 0.9051) but it predates stratified batching.  Keep for archival.
- `outputs/verification/stage2/amazon_warm_start_full500/` - warm-start
  diagnostic that showed the gen-only label collapse to majority class
  even after compact Stage 1 + warm start.  This run's data motivated the
  AUPRC + stratified-batch fix above; it is *not* a baseline.
- `outputs/verification/stage2/amazon_warm_start_smoke/` - earlier
  warm-start smoke that the full500 run already supersedes.
- `outputs/gated/stage2/amazon/{masked_shortcut_smoke_*,shortcut_fix_smoke_*_escalated,smoke,smoke_best_ckpt}/` -
  shortcut / escalation diagnostic runs from the masked-eval session;
  conclusions already documented in earlier sections of this file.

A concrete cleanup-candidate list with sizes is recorded in §7.9.3.

#### 7.9.2 Next step from here

1. Re-run a single Amazon Stage-2 run end-to-end with the new defaults
   so the next acceptance baseline is on AUPRC + stratified batches:

   ```bash
   tmux new-session -d -s amazon-stage2-strat-v1 '
     CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
     PYTHONPATH=/data1/mq/codes/graph-detection-main/PriorF-Reasoner \
     /data1/mq/conda_envs/priorfgnn/bin/python scripts/run_stage2_train.py \
       --dataset amazon \
       --qwen-path /data1/mq/models/Qwen3-4B-Instruct-2507 \
       --teacher-export-train  outputs/gated/teacher_exports/amazon/train/teacher_export.parquet \
       --teacher-export-validation outputs/gated/teacher_exports/amazon/validation/teacher_export.parquet \
       --data-manifest manifests/amazon/data_manifest.json \
       --output-dir outputs/gated/stage2/amazon/strat_v1 \
       --warm-start-peft-adapter outputs/verification/stage1_sft/amazon_compact_smoke/peft_adapter \
       --max-steps 500 --batch-size 2 \
       --learning-rate 1e-4 --warmup-ratio 0.1 --lr-scheduler cosine \
       --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
       --lambda-cls 1.0 --lambda-distill 0.3 \
       --validation-subset 256 --validation-every-n-steps 50 \
       --gpu-index 0 2>&1 | tee outputs/gated/stage2/logs/amazon_strat_v1.log'
   ```

   Live-monitor with TensorBoard from a separate shell:

   ```bash
   tensorboard --logdir outputs/gated/stage2/amazon/strat_v1/tb --port 6006
   ```

2. After it finishes, run the generation audit on
   `outputs/gated/stage2/amazon/strat_v1/best_checkpoint/peft_adapter`
   at validation subset 64 with `--max-new-tokens 256` and confirm:
   - `strict_schema_parse_rate = 1.0`,
   - `discriminative_power.auprc` is the headline (AUROC reported only
     for context),
   - generated `label` is no longer all-`benign` on the audit subset.

3. Only if AUPRC still lags after step 2, vary `--lambda-cls` (e.g.
   `2.0`) or class-balance the generation loss directly inside
   `train/train_stage2_canonical.run_canonical_train_step`.  Do **not**
   change `evidence/prompt_builder.py` SFT targets again before those
   knobs are tried.

#### 7.9.3 Cleanup candidates (awaiting operator confirmation)

The directories below total roughly **2.4 GiB** and are no longer
needed as baselines under the new conventions.  A separate operator
should review the list and delete them in one batch; this handoff does
not delete anything automatically.

| path | size | reason |
|---|---:|---|
| `outputs/verification/stage2/amazon_warm_start_smoke/` | 379M | superseded by full500; full500 itself is now also a historical diagnostic |
| `outputs/verification/stage2/amazon_warm_start_full500/` | 379M | predates stratified batching + AUPRC default |
| `outputs/gated/stage2/amazon/masked_shortcut_smoke_gpu5_base_20260424/` | 190M | masked-eval shortcut diagnostic, conclusions in §6 |
| `outputs/gated/stage2/amazon/masked_shortcut_smoke_gpu6_lowdistill_20260424/` | 190M | same |
| `outputs/gated/stage2/amazon/masked_shortcut_smoke_gpu7_lowlr_20260424/` | 190M | same |
| `outputs/gated/stage2/amazon/shortcut_fix_smoke_gpu5_base_20260424_escalated/` | 190M | shortcut fix escalation diagnostic |
| `outputs/gated/stage2/amazon/shortcut_fix_smoke_gpu6_lowdistill_20260424_escalated/` | 190M | same |
| `outputs/gated/stage2/amazon/shortcut_fix_smoke_gpu7_lowlr_20260424_escalated/` | 190M | same |
| `outputs/gated/stage2/amazon/smoke_best_ckpt/` | 379M | early best-checkpoint smoke |
| `outputs/gated/stage2/amazon/smoke/` | 12M | early smoke |
| `outputs/gated/stage2/amazon/run_v1/` | 12M | superseded by run_v2; only kept previously for diff narratives |
| `outputs/gated/stage2/amazon/shortcut_fix_smoke_gpu{5,6,7}_lowlr_20260424/` (non-escalated) | 12K total | empty / placeholder dirs |
| `/tmp/amazon_*.log` (≈ 17 files) | ~ 80K | old session logs from previous diagnostic runs |

`outputs/gated/stage2/amazon/run_v2/` is **not** in the cleanup list:
the existing acceptance bundle (`docs/060_amazon_acceptance_report.md`
and the gate manifest) still references it.  Removing it would
invalidate those artifacts.
