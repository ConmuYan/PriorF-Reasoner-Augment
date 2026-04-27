# Amazon Reproducibility Acceptance Report

> **Superseded / diagnostic-only notice (Stage 9.0c, 2026-04-27):** Pre-fix run_v2 / strat_v1 artifacts are diagnostic-only and superseded due to Evidence Card neighbor-label leakage discovered in Stage 4.5. Do not cite pre-fix run_v2 / strat_v1 metrics, gate manifests, or acceptance artifacts as clean formal results; regenerate under the current leakage-policy and prompt-audit gate before making formal claims.

All acceptance/pass language below is retained only as historical diagnostic context. It is not a current clean accepted result and must not be used for Stage 9.1 or later formal claims until regenerated under the current post-fix provenance and prompt-audit policy.


## Scope

This report covers the Amazon acceptance round only.

YelpChi was left in smoke / compatibility scope and is not used for Amazon tuning conclusions in this round.

## Acceptance outcome

Amazon now has a reproducible end-to-end evaluation bundle with:

- Formal head-only, fusion, gen-only, and faithfulness reports.
- A passing gate manifest.
- A minimal runnable Stage 1 structured SFT driver.
- Targeted regressions, full pytest, and touched-files ruff green.
- A raw-generation audit that distinguishes syntax pass from semantic usefulness and discriminative power.

The main acceptance conclusion is:

- **Reproducibility / pipeline closure**: pass.
- **Gate closure**: pass.
- **Gen-only syntax compliance**: pass.
- **Gen-only semantic usefulness / standalone discriminative value**: fail to demonstrate useful behavior in the audited subset.

So the Amazon pipeline is acceptable as a reproducible evaluation bundle, but the current generator should **not** be promoted as a semantically useful standalone decision module.

## Canonical artifacts

- `outputs/formal/head_only/amazon_run_v2_final/formal_head_only_report_amazon.json`
- `outputs/formal/fusion/amazon_run_v2_final/formal_fusion_report_amazon.json`
- `outputs/formal/gen_only/amazon_run_v2_final/formal_gen_only_report_amazon.json`
- `outputs/formal/faithfulness/amazon_run_v2_final/faithfulness_report_amazon.json`
- `outputs/gated/manifests/gate_manifest_amazon.json`
- `outputs/audits/generation/amazon_run_v2_final_subset32/generation_audit_amazon_final_test.json`
- `outputs/audits/generation/amazon_run_v2_final_subset32/raw_generations_amazon_final_test.jsonl`
- `train/stage1_sft.py`

## Commands used in this closure round

### Formal evaluation bundle

```bash
/data1/mq/conda_envs/priorfgnn/bin/python scripts/run_formal_head_only_eval.py \
  --dataset amazon \
  --qwen-path /data1/mq/models/Qwen3-4B-Instruct-2507 \
  --peft-adapter outputs/gated/stage2/amazon/run_v2/peft_adapter \
  --cls-head outputs/gated/stage2/amazon/run_v2/cls_head.pt \
  --checkpoint-source final_checkpoint \
  --teacher-export-validation outputs/gated/teacher_exports/amazon/validation/teacher_export.parquet \
  --teacher-export-final-test outputs/gated/teacher_exports/amazon/final_test/teacher_export.parquet \
  --data-manifest manifests/amazon/data_manifest.json \
  --output-dir outputs/formal/head_only/amazon_run_v2_final \
  --run-id amazon_run_v2_final \
  --commit 661b89ff7a4e3c180ae4940a8395a3a70b04c31c \
  --config-fingerprint 74dab540cd9dbfda80265e0005d141551d9dfcb9bc8b26b66cf53a1030a11efb \
  --step 500 \
  --validation-subset 256 \
  --final-test-subset 512 \
  --seed 0 \
  --include-oracle-diagnostics \
  --apply-temperature-scaling
```

```bash
/data1/mq/conda_envs/priorfgnn/bin/python scripts/run_formal_gen_only_eval.py \
  --dataset amazon \
  --qwen-path /data1/mq/models/Qwen3-4B-Instruct-2507 \
  --peft-adapter outputs/gated/stage2/amazon/run_v2/peft_adapter \
  --cls-head outputs/gated/stage2/amazon/run_v2/cls_head.pt \
  --checkpoint-source final_checkpoint \
  --teacher-export outputs/gated/teacher_exports/amazon/final_test/teacher_export.parquet \
  --data-manifest manifests/amazon/data_manifest.json \
  --population final_test \
  --output-dir outputs/formal/gen_only/amazon_run_v2_final \
  --run-id amazon_run_v2_final \
  --commit 661b89ff7a4e3c180ae4940a8395a3a70b04c31c \
  --config-fingerprint 74dab540cd9dbfda80265e0005d141551d9dfcb9bc8b26b66cf53a1030a11efb \
  --step 500 \
  --subset-size 128 \
  --max-new-tokens 768 \
  --seed 0
```

### Generation audit

```bash
/data1/mq/conda_envs/priorfgnn/bin/python scripts/run_generation_audit.py \
  --dataset amazon \
  --qwen-path /data1/mq/models/Qwen3-4B-Instruct-2507 \
  --peft-adapter outputs/gated/stage2/amazon/run_v2/peft_adapter \
  --teacher-export outputs/gated/teacher_exports/amazon/final_test/teacher_export.parquet \
  --data-manifest manifests/amazon/data_manifest.json \
  --population final_test \
  --checkpoint-source final_checkpoint \
  --output-dir outputs/audits/generation/amazon_run_v2_final_subset32 \
  --subset-size 32 \
  --review-sample-size 12 \
  --max-new-tokens 768 \
  --seed 0 \
  --run-id amazon_run_v2_final_subset32
```

### Verification

```bash
/data1/anaconda3/bin/python -m pytest tests/test_stage1_sft.py tests/test_generation_audit.py tests/test_eval_gen_only.py -q --tb=short
/data1/anaconda3/bin/python -m pytest tests/ -q --tb=short
/data1/anaconda3/bin/python -m ruff check scripts/_formal_eval_helpers.py scripts/run_formal_head_only_eval.py scripts/run_formal_fusion_eval.py scripts/run_formal_gen_only_eval.py scripts/run_formal_faithfulness.py scripts/generate_gate_manifest.py scripts/run_generation_audit.py schemas/gate_manifest.py train/stage1_sft.py tests/test_gate_check.py tests/test_formal_launcher_gate.py tests/test_smoke_pipeline.py tests/test_stage1_sft.py tests/test_generation_audit.py tests/test_eval_gen_only.py
```

## Formal metrics summary

### Head-only

- **Population**: `final_test`
- **Subset size**: `512`
- **AUROC**: `0.9892482779275232`
- **AUPRC**: `0.8765276001152786`
- **Temperature**: `0.7585775750291838`
- **Frozen threshold**: `0.09140074869594589`
- **Subset node hash**: `401c25a13b9df5a9725acaca3f13a3fab79a3ec0d19a7317139a49a11cb55459`
- **Subset record hash**: `d3709d4bd3a8594be2a87e6298bcd8d3e5f3d1a5eef7f69678d17dc755af0ffa`

### Fusion

- **Optimal alpha**: `0.8`
- **Student contribution pass**: `True`
- **AUROC**: `0.9869421982629529`
- **AUPRC**: `0.9227608135502422`

### Gen-only formal report

- **Subset size**: `128`
- **Strict schema parse rate**: `1.0`
- **Normalized parse rate**: `1.0`
- **AUROC**: `0.19701213818860877`
- **AUPRC**: `0.04673922093376829`
- **Brier score**: `0.36108689716135256`
- **Subset node hash**: `c673c24293a6be15f730b94a0f16b822539850ea162dd19586c8cf1b4a4d7aff`
- **Subset record hash**: `6613d7b7a8a46399868cb273ab8712566185c601c8b692368617e1addd23f13c`

Important note:

- These values were regenerated **after** fixing `eval_gen_only` so `StrictOutput.score` is interpreted as fraud probability, consistent with Stage 1 / Stage 2 SFT targets.
- The formal 128-subset report and the raw-generation audit now agree on the same qualitative conclusion: syntax is excellent, but standalone discrimination remains weak.

### Faithfulness

- **Mean sufficiency**: `0.030896042560925707`
- **Mean comprehensiveness**: `0.02416626049671322`
- **Teacher-prob ablation flip rate**: `0.0703125`

## Gate manifest summary

`outputs/gated/manifests/gate_manifest_amazon.json`

- **teacher_baseline_pass**: `True`
- **subset_head_gate_pass**: `True`
- **strict_schema_parse_pass**: `True`
- **student_contribution_pass**: `True`
- **teacher_prob_ablation_pass**: `True`
- **population_contract_pass**: `True`
- **smoke_pipeline_pass**: `True`

## Stage 1 closure

`train/stage1_sft.py` was implemented as a minimal runnable Stage 1 structured SFT driver.

It now:

- Builds Evidence Card -> strict JSON prompt/completion pairs.
- Trains LoRA adapters with generation-only loss.
- Saves adapter artifacts, config, and run record with provenance.
- Exposes a CLI aligned with the existing driver conventions.

Current boundary for this acceptance round:

- Stage 2 is still the canonical joint-training path used by the accepted Amazon bundle.
- The acceptance round does **not** claim that Stage 2 currently warm-starts from a Stage 1 adapter.

## Generation audit findings

### Audit artifact provenance

- **Audit report**: `outputs/audits/generation/amazon_run_v2_final_subset32/generation_audit_amazon_final_test.json`
- **Raw generations**: `outputs/audits/generation/amazon_run_v2_final_subset32/raw_generations_amazon_final_test.jsonl`
- **Raw generations sha256**: `d8c97b0add0c412111f6104be235c8917516f63b6dcc72961e011dbf76acd9b4`
- **Audit subset node hash**: `a5faabb2bc59cab986dbad5a3139e5dfd25405ba95657052327c80bf7c357689`
- **Audit subset record hash**: `48c773d4041806143aa758ea973d2af7b9a1111128370fcaca1821eb2ce99f1b`
- **Review sample count**: `12`

### Syntax pass

- **Strict schema parse rate**: `1.0`
- **Normalized parse rate**: `1.0`
- **Strict parse failure count**: `0`

### Semantic usefulness audit

The audit shows a clear gap between syntax compliance and useful content.

In the fixed review sample of 12 generations:

- `label` / `score` directionality was internally consistent in all reviewed samples.
- The reviewed generations were template-like rather than evidence-grounded.
- The generated `rationale`, `evidence`, and `pattern_hint` fields collapsed to generic placeholder text in the reviewed batch.
- The audit summary recorded zero structural-signal mentions under the current heuristic:
  - `rationale_structural_signal_rate = 0.0`
  - `evidence_structural_signal_rate = 0.0`
  - `pattern_hint_mentions_graph_regime_rate = 0.0`

Representative reviewed outputs showed the same generic pattern:

- `rationale`: `The structured graph evidence should be considered before assigning the fraud label.`
- `evidence`: `Evidence Card fields are the only permitted basis for this structured decision.`
- `pattern_hint`: `Use the declared graph-regime structural pattern; no raw text or raw graph matrix is available.`

### Discriminative power

On the audited 32-sample final_test subset:

- **AUROC**: `0.18333333333333335`
- **AUPRC**: `0.05622188905547226`
- **Brier score**: `0.250889379417216`

Interpretation:

- The generator is excellent at emitting valid strict JSON.
- It is **not** showing useful standalone fraud discrimination on the audited subset.
- Therefore, `strict_schema_parse_pass=True` should be interpreted only as a **syntax contract pass**, not as evidence of semantic reasoning quality.

## Provenance summary

### Shared inputs

- **Backbone**: `/data1/mq/models/Qwen3-4B-Instruct-2507`
- **PEFT adapter**: `outputs/gated/stage2/amazon/run_v2/peft_adapter`
- **PEFT adapter sha256**: `5b95e6b75e557115712fd15d3de39d3d6ceb743ef03172a215bcabfb189ec7cb`
- **CLS head**: `outputs/gated/stage2/amazon/run_v2/cls_head.pt`
- **CLS head sha256**: `e5d3ab5d6931b3fab9c810c333987734c7ab17a7bb02ac604cd5d64c1c3cfe33`
- **Data manifest**: `manifests/amazon/data_manifest.json`
- **Data manifest sha256**: `b73c7fe773775ad08e0060e6e0771126d92f72064c00eb349b32ce0b5c8d1100`
- **Teacher export final_test sha256**: `dd3a31502ad236f0ff3514880da566711d21ad3d0bdb99e3a48fc1ae7a9b6cc3`
- **Seed**: `0`
- **Checkpoint source**: `final_checkpoint`
- **Config fingerprint**: `74dab540cd9dbfda80265e0005d141551d9dfcb9bc8b26b66cf53a1030a11efb`
- **Step**: `500`

### Git caveats

The current reproducibility bundle still has two explicit caveats:

- The evaluation artifacts were produced with `git_dirty=True` in `_runtime_provenance`.
- `outputs/gated/stage2/amazon/run_v2/run_record.json` does not expose a recoverable original training commit; the current bundle uses the evaluation-time commit `661b89ff7a4e3c180ae4940a8395a3a70b04c31c` for evaluation provenance.

These caveats do **not** invalidate the reproduced metrics in this round, but they should be cleaned up before claiming a pristine archival bundle.

## Verification status

- **Targeted regressions**: pass.
- **Full pytest**: `249 passed`.
- **Touched-files ruff**: pass.
- **YelpChi status in this round**: smoke / compatibility only; no tuning conclusions mixed into Amazon acceptance.

## Final decision

### What is accepted now

- The Amazon evaluation pipeline is reproducible end-to-end.
- The Amazon formal report set and gate manifest are complete enough for audit closure.
- The Stage 1 missing-driver gap is closed at the implementation level.
- The repo now makes the key distinction between generation syntax pass and actual generation usefulness.

### What is explicitly not accepted as a positive claim

- The current generator is **not** accepted as semantically useful or discriminatively strong based on the audited raw generations.
- The current bundle is **not yet** a pristine archival package because evaluation provenance still records `git_dirty=True` and original Stage 2 training commit provenance remains incomplete.

### Recommended final cleanup

- Commit the current repo state.
- Re-run the Amazon formal evaluation drivers plus gate-manifest generation once more on the clean tree.
- Preserve the current generation audit as the capability boundary note for the generator.
