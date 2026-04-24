# PriorF-Reasoner Handoff (2026-04-24)

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

### 2.1 Full-TRAIN / best-checkpoint training (high priority)

`run_v2` still caps at 4 096 TRAIN samples over 500 optimizer steps.
That saturates Amazon at AUROC 0.99 but YelpChi still has headroom.
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

### 2.2 Temperature calibration (high priority for fusion)

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

### 2.3 CLI drivers for the remaining three formal runners

The library functions are implemented + tested + green:

- `eval/eval_gen_only.py` (strict-schema parse rate + normalized parse
  rate; see `self-review/task_08_formal_launcher_gate_integration.md`
  if the file exists in this repo copy),
- `eval/eval_fusion.py` (validation-only alpha search + student
  contribution gate; `llm/fusion.py` provides the math),
- `eval/faithfulness.py` (pinned to `score_head`; small-sample diag).

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

### 2.4 Gate manifest auto-assembly (medium priority)

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
