# PriorF-Reasoner

Fail-closed training and evaluation harness for a compact language model
that reads teacher-derived structural Evidence Cards and produces
auditable fraud-detection probabilities + strict-schema rationales.

This README is the top-level orientation.  Operators looking for a
concrete runbook go to `docs/040_operator_runbook.md`; developers
looking for contracts go to `docs/010_project_contract.md`,
`docs/020_top_level_design.md`, `docs/030_harness_agent_execution_guide.md`,
and `docs/050_fail_closed_guardrails.md`.

## What the system is (and is not)

**Is.** A harness around a compact student LM + classification head that:

* consumes schema-validated Evidence Cards built from a pretrained
  PriorF teacher,
* is trained canonically with generation + classification + distillation
  losses under Accelerate,
* is evaluated formally with frozen validation thresholds / alpha and
  gated by an executable `gate_manifest.json`.

**Is not.** A drop-in LLM for raw graph input, a generic "LLM-replaces-GNN"
claim, or a system whose head metrics can be trusted without the formal
eval path in this repo.

## Three output namespaces

Every launcher writes under exactly one of:

* `outputs/diagnostic/` — smoke runs, exploratory probes, paraphrased
  parse audits. Never counts as a formal result.
* `outputs/gated/` — canonical training checkpoints that have not yet
  cleared the full gate manifest. Used for parity checks and gate
  evidence generation; not reported as formal metrics.
* `outputs/formal/` — only reachable via `scripts/run_full_pipeline.sh
  --mode formal --gate-manifest PATH` with a passing
  `scripts/gate_check.py`. Every field of `GateManifest` must be `True`.

Silent fallbacks between namespaces are forbidden. A failing gate
manifest causes the formal launcher to exit non-zero without creating
any `outputs/formal/` directory.

## What counts as success

A green run is only considered valid if **all** of the following hold:

* `scripts/gate_check.py` passes a complete `gate_manifest.json`
  (every required gate is `True`),
* head / fusion / gen-only / faithfulness reports are produced via the
  modules in `eval/` (not reimplemented downstream),
* teacher-probability ablation audit (`evidence.ablations`) does **not**
  flag `teacher_prob_dependency_high` above the agreed threshold for
  the reported population,
* fusion reports `student_contribution_pass = True` at the selected
  `optimal_alpha`,
* strict-schema parse rate (`eval/eval_gen_only.py`) is the headline
  parse metric; normalized parse rate is diagnostic only.

## What explicitly does not count as success

* **Teacher fallback is not student success.** A fusion whose
  `optimal_alpha` collapses to `0.0` means the student contributes
  nothing; the system is reporting the teacher, not the student.
  `student_contribution_pass` must be `True`.
* **Low-alpha fusion is not success.** Even an `optimal_alpha` near
  zero that nominally beats either branch alone is not a valid "fusion
  works" claim; `min_student_alpha` in `FusionEvalConfig` is there to
  enforce this.
* **Normalized parse is not schema compliance.** Only
  `strict_schema_parse_rate` counts as schema compliance. Normalized /
  alias parsers live exclusively in the diagnostic path.
* **Oracle same-population thresholds are not formal metrics.** Any
  report that tunes a decision threshold on the rows it is reporting
  is, by construction, diagnostic.
* **Small-sample faithfulness is not formal faithfulness.** Faithfulness
  below the minimum sample size is marked as smoke / diagnostic and
  does not enter the gate manifest.

## One-line quick starts

The launchers are thin wrappers over `scripts/run_full_pipeline.sh`.
Each forwards to the right output namespace:

```bash
# Fail-closed smoke check on canonical plumbing (diagnostic namespace only).
scripts/run_smoke.sh

# Stage 1 (TRL SFT structured-generation training) in gated namespace.
scripts/run_stage1.sh -- python -m train.stage1_sft ...

# Stage 2 (canonical Accelerate trainer: L_gen + L_cls + L_distill) in gated namespace.
scripts/run_stage2.sh -- python -m train.train_stage2_canonical ...

# Formal evaluation.  Requires a valid gate manifest; gate_check runs first.
scripts/run_eval.sh --gate-manifest outputs/gated/gate_manifest.json -- \
    python -m eval.eval_head_only ...
```

Every launcher respects `--output-root DIR` (default `outputs`).

## End-to-end pipeline on real Qwen3-4B + Amazon / YelpChi

The `scripts/run_full_stage2_pipeline.sh` driver chains the five stages
required to go from raw legacy `.mat` files to a formal head-only eval
report on one dataset:

1. `scripts/legacy_mat_to_canonical.py` - legacy CIKM 2020 `.mat` ->
   canonical `x`/`ground_truth_label`/`split_vector`/`relation_*` `.mat`.
2. `scripts/generate_teacher_exports.py` - runs `LGHGCLNetV2` inference
   on the canonical `.mat`, writes validated
   `outputs/gated/teacher_exports/<ds>/<pop>/teacher_export.parquet`
   (TRAIN / VALIDATION / FINAL_TEST) + `DataManifest` +
   `TeacherBaselineReport`.  Fail-closed `_git_head_sha_or_fail` pins a
   real commit sha into every `TeacherProvenance`.
3. `scripts/run_stage2_train.py` - multi-step canonical joint trainer:
   Qwen3-4B backbone + PEFT LoRA adapter on `{q_proj, v_proj}` +
   activation checkpointing + trainable linear `cls_head`, running
   `run_canonical_train_step` in a loop.  Saves `peft_adapter/` +
   `cls_head.pt` + `CanonicalTrainerRunRecord` + per-step JSONL log.
4. `scripts/run_stage2_inference.py` - head-only scoring on
   `final_test`: loads PEFT adapter + `cls_head.pt`, builds canonical
   `EvidenceCards` via `build_evidence_card`, runs `score_head`, writes
   a `ScorerReport` under `outputs/gated/eval/<ds>/`.
5. `scripts/run_formal_head_only_eval.py` - formal head-only eval:
   validation-only F1 threshold freezing + frozen-threshold headline
   metrics on `final_test` + calibration summary + optional oracle
   diagnostics.  Writes `FormalHeadOnlyReport` under
   `outputs/formal/head_only/<ds>/`.

Run the whole chain for one dataset:

```bash
bash scripts/run_full_stage2_pipeline.sh \
  --dataset amazon \
  --qwen-path /data1/mq/models/Qwen3-4B-Instruct-2507 \
  --max-steps 40 --batch-size 2 \
  --max-train-samples 1024 \
  --validation-subset 128 --final-test-subset 256 \
  --gpu-index 0
```

Use `--skip-teacher-exports` to reuse cached parquets for rapid
experimentation over different training recipes.

For a quick plumbing smoke that does **not** require canonical teacher
exports, `scripts/run_head_only_smoke.py` and
`scripts/run_stage2_train_smoke.py` operate on the legacy
`assets/teacher_exports/*.parquet` and write only under
`outputs/diagnostic/`.

## Prompt / task bundle

Historical prompt bundle used to bring the project up to the current
15-task contract:

## 建议使用方式

1. 先把仓库内 source-of-truth 文档落盘：
   - `docs/010_project_contract.md`
   - `docs/020_top_level_design.md`
   - `docs/030_harness_agent_execution_guide.md`
   - `docs/050_fail_closed_guardrails.md`
   - `read.md`
   - `CLAUDE.md`

2. 先用只读模式让 Codex 审阅文档，再按 Task 逐个执行。

3. 每个 Task 都要：
   - 先贴 `prompts/00_master_control_prompt.txt`
   - 再贴对应 Task prompt
   - 写完后贴 `prompts/98_self_review_prompt.txt`
   - 审核通过后贴 `prompts/99_wrapup_prompt.txt`

## 文件目录

- `codex_operation_guide.md`
- `prompts/00_master_control_prompt.txt`
- `prompts/01_task_01_data_foundations.txt`
- `prompts/02_task_02_teacher_contract_and_baseline_gate.txt`
- `prompts/03_task_03_evidence_card_and_output_schema.txt`
- `prompts/04_task_04_hidden_state_pooling.txt`
- `prompts/05_task_05_unified_head_scoring.txt`
- `prompts/06_task_06_eval_head_parity_test.txt`
- `prompts/07_task_07_gate_manifest_and_gate_check.txt`
- `prompts/08_task_08_formal_launcher_gate_integration.txt`
- `prompts/09_task_09_canonical_joint_trainer.txt`
- `prompts/10_task_10_formal_head_only_eval.txt`
- `prompts/11_task_11_teacher_prob_ablation_audit.txt`
- `prompts/12_task_12_formal_fusion_eval.txt`
- `prompts/13_task_13_formal_generation_eval.txt`
- `prompts/14_task_14_formal_faithfulness_eval.txt`
- `prompts/15_task_15_readme_runbook_and_launcher_closure.txt`
- `prompts/98_self_review_prompt.txt`
- `prompts/99_wrapup_prompt.txt`
