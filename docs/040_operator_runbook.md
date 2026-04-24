# `docs/040_operator_runbook.md` — Operator Runbook

This runbook is the **shortest valid execution path** for operating the
PriorF-Reasoner harness end-to-end under the fail-closed contract.

It complements:

* `read.md` — higher-level rules about success and failure,
* `docs/010_project_contract.md` — goals and constraints,
* `docs/020_top_level_design.md` — architecture,
* `docs/030_harness_agent_execution_guide.md` — task-by-task execution plan,
* `docs/050_fail_closed_guardrails.md` — formal guardrails catalogue.

Operators reading this document should not need to invent behavior; any
ambiguity is a bug to file, not a decision to make.

## 1. Output namespaces

There are exactly three output namespaces, keyed by the `--mode` argument
of `scripts/run_full_pipeline.sh`:

| Namespace | Directory | When |
|---|---|---|
| `diagnostic` | `outputs/diagnostic/` | smoke, paraphrase / alias parse audits, teacher-prob ablation probes below sample-size floor |
| `gated`      | `outputs/gated/`      | canonical training checkpoints before the full gate manifest clears |
| `formal`     | `outputs/formal/`     | gate-cleared formal eval reports only |

Silent fallback from one namespace to another is forbidden at the
contract level.  Any launcher that would otherwise downgrade must exit
non-zero instead.

## 2. Shortest end-to-end path

These steps assume a clean checkout on the target machine with the
dependencies pinned by the project's dependency manager already
installed. None of the steps reach into formal namespace until the gate
manifest clears.

1. **Preflight smoke** (diagnostic only):

   ```bash
   scripts/run_smoke.sh
   ```

   Runs the canonical data → Evidence Card → prompt path plus the
   launcher fail-closed contract tests. Green smoke proves plumbing,
   not formal validity.

2. **Prepare data** (gated namespace):

   ```bash
   scripts/run_stage1.sh --output-root outputs -- \
       python -m graph_data.build_manifest ...
   ```

   Produces a `data_manifest.json` with complete `sha256` provenance
   for every artifact. Without this manifest the gate cannot pass.

3. **Stage 1 — structured-generation SFT** (gated namespace):

   ```bash
   scripts/run_stage1.sh -- python -m train.stage1_sft ...
   ```

   Uses TRL `SFTTrainer` on Evidence-Card → strict-JSON prompt /
   completion pairs. Emits a Stage 1 checkpoint; still gated.

4. **Stage 2 — canonical joint trainer** (gated namespace):

   ```bash
   scripts/run_stage2.sh -- python -m train.train_stage2_canonical ...
   ```

   Runs the Accelerate loop with all three required losses:
   `L_gen + L_cls + L_distill`. Refuses to start if any of the three
   is disabled. Validation uses the shared `eval.head_scoring.score_head`
   so epoch-end and offline head eval agree.

5. **Parity check** (before any formal eval):

   ```bash
   PYTHONPATH=. python -m pytest tests/test_eval_head_parity.py -q
   ```

   The single source of truth for validation parity; must pass before
   you fill in `validation_eval_parity_pass = True` in the manifest.

6. **Assemble the gate manifest.**

   Fill `gate_manifest.json` (schema pinned by `schemas/gate_manifest.py`)
   with all required gates set to `True` and the real `commit`,
   `config_fingerprint`, `data_manifest_hash`, and UTC `generated_at`.
   Store it under `outputs/gated/gate_manifest.json`.

7. **Formal evaluation** (formal namespace):

   ```bash
   scripts/run_eval.sh \
       --gate-manifest outputs/gated/gate_manifest.json \
       -- python -m eval.eval_head_only ...

   scripts/run_eval.sh \
       --gate-manifest outputs/gated/gate_manifest.json \
       -- python -m eval.eval_fusion ...

   scripts/run_eval.sh \
       --gate-manifest outputs/gated/gate_manifest.json \
       -- python -m eval.eval_gen_only ...

   scripts/run_eval.sh \
       --gate-manifest outputs/gated/gate_manifest.json \
       -- python -m eval.faithfulness ...
   ```

   `scripts/gate_check.py` runs first; any failing gate aborts before
   the evaluation command is invoked.

8. **Teacher-probability dependency audit.**

   After `eval_head_only` on both full and teacher-prob-ablated
   populations, run `evidence.ablations.run_teacher_prob_ablation_audit`
   with an explicit `dependency_threshold` (recommended: `0.05` on
   AUROC) and record `teacher_prob_dependency_high` next to the head
   report.

## 3. Fail conditions and what to check first

When anything goes wrong, check these in order:

1. **Missing manifest.** Any launcher invoked in `formal` mode without
   `--gate-manifest` exits immediately. Fix: supply the manifest path.

2. **Gate check failure.** Any `*_pass` field that is not exactly
   `True` aborts `gate_check.py`. Fix: re-run the upstream step that
   owns the gate (see the mapping in `schemas/gate_manifest.py`
   comments) and regenerate the manifest. Do not edit the gate value
   by hand.

3. **Parity mismatch.** If `tests/test_eval_head_parity.py` fails, the
   canonical `score_head` path is drifting from epoch-end validation.
   Fix this before any formal report.

4. **Population overlap.** `graph_data.validators` rejects overlapping
   node-id sets across populations. Fix the manifest; never suppress
   the validator.

5. **Single-class population.** Threshold-dependent metrics are
   intentionally `None`. Fix the input split; never fabricate AUROC.

6. **Silent teacher fallback.** `fusion` reports with
   `optimal_alpha <= min_student_alpha` fail
   `student_contribution_pass`. This is a modeling problem, not a
   reporting problem; fix the student or the distillation target.

7. **Strict parse below contract.** Only strict parse counts. If
   normalized parse succeeds but strict fails, the model output is
   schema-non-compliant. Fix the generation path; do not migrate
   normalized metrics into formal reports.

8. **Teacher-prob dependency flagged.** A `True`
   `teacher_prob_dependency_high` means the student is effectively
   reading teacher_prob off the card. Either tune training to reduce
   that dependency or report the finding honestly and do not claim
   independent student detection.

## 4. What never belongs in formal reports

* accuracy / F1 tuned on the rows being reported
* normalized / alias parse rates
* small-sample faithfulness (below the contract minimum)
* oracle thresholds from the reported population
* any metric produced outside `eval/` modules
* any metric computed before `gate_check.py` passed

## 5. Command cheat-sheet

```bash
# Help for each launcher:
scripts/run_smoke.sh --help
scripts/run_stage1.sh --help
scripts/run_stage2.sh --help
scripts/run_eval.sh   --help
scripts/run_full_pipeline.sh --help

# Gate check on an existing manifest:
python scripts/gate_check.py --manifest-path outputs/gated/gate_manifest.json

# Full pytest suite (must be green before any gate manifest is written):
PYTHONPATH=. python -m pytest tests/ -q
```
