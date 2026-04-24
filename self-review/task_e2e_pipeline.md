# Self-review: End-to-end training / inference / formal eval on real data

This self-review covers the three scripts added to close the "end-to-end
on real Qwen3-4B + canonical teacher exports for Amazon and YelpChi"
delivery item:

- `scripts/run_stage2_train.py` (canonical trainer driver)
- `scripts/run_stage2_inference.py` (head-only ScorerReport driver)
- `scripts/run_formal_head_only_eval.py` (formal head-only runner)
- `scripts/run_full_stage2_pipeline.sh` (chains steps 1-5)

plus the regenerated canonical teacher exports that land under
`outputs/gated/teacher_exports/<ds>/<pop>/` and the per-dataset
data manifests under `manifests/<ds>/data_manifest.json`.

## 1. Scope kept out

None of the existing canonical modules were edited.  Every script
consumes the public contracts:

- `graph_data.manifests.load_data_manifest`
- `priorf_teacher.export_pipeline.read_teacher_export_artifact`
- `evidence.evidence_schema.build_evidence_card`
- `train.train_stage2_canonical.run_canonical_train_step`,
  `run_validation_with_unified_scorer`
- `eval.head_scoring.score_head`
- `eval.eval_head_only.run_formal_head_only_eval`

No new Pydantic schemas are introduced; every report that leaves the
driver is a re-dumped existing model (`CanonicalTrainerRunRecord`,
`ScorerReport`, `FormalHeadOnlyReport`) with an additional
`_runtime_provenance` side-block for operator auditing.

## 2. Silent-default / schema-drift audits

Trainer driver (`run_stage2_train.py`):

- `CanonicalTrainerConfig` is built with `Literal[True]` /
  `Literal[False]` guards still in place (`require_generation_loss`,
  `require_classification_loss`, `require_distillation_loss`,
  `use_accelerate`, `diagnostic_mode`, `frozen_backbone_probe`,
  `class_imbalance_recipe`).  Any attempt to flip them fails Pydantic.
- `CanonicalTrainingBatch` is built from samples whose
  `evidence_card.population_name` is guaranteed to be `TRAIN` by the
  upstream `read_teacher_export_artifact` + `build_evidence_card`
  contract.  The driver does not mutate populations.
- Gradient checkpointing is enabled via the PEFT API
  (`gradient_checkpointing_enable(use_reentrant=False)`) + explicit
  `enable_input_require_grads()` and `model.config.use_cache = False`,
  exclusively to keep bf16 training on a single 48 GiB card at B=2
  feasible.  No schema or objective changes.
- `_LinearClsHead.forward` returns `[B]` via `.squeeze(-1)` to match the
  1-D logit contract enforced by `score_head`.

Inference driver (`run_stage2_inference.py`):

- `PeftModel.from_pretrained` on a bf16 base produces a bf16 PEFT-merged
  forward path; the same `pool_last_valid_token` / sigmoid / `extra="forbid"`
  guards inside `score_head` are in effect.
- `CheckpointProvenance.content_hash` is the sha256 of the saved
  `cls_head.pt` on disk, so downstream formal eval can cross-check that
  the scorer and the checkpoint bundle point to the exact same bytes.

Formal head-only eval (`run_formal_head_only_eval.py`):

- `FormalHeadOnlyCheckpointBundle` requires all three components
  (`llm_backbone`, `peft_adapter`, `cls_head`) to share `run_id`,
  `checkpoint_step`, `commit`, `config_fingerprint`,
  `data_manifest_hash`, and `graph_regime`.  The driver supplies them
  from one operator-provided set of args, and `content_hash` comes
  from per-file/per-dir sha256 over real bytes (deterministic hash over
  sorted adapter entries).
- `validation_inputs.checkpoint_provenance` equals
  `report_inputs.checkpoint_provenance` equals the shared
  `CheckpointProvenance` computed from the cls_head checkpoint, which
  the formal runner cross-checks against
  `FormalHeadOnlyCheckpointBundle.cls_head.to_shared_checkpoint_provenance()`.
- The driver passes `threshold_selection_metric="f1"` and does not
  attempt to reselect a threshold on the report population; the formal
  runner enforces that anyway.

## 3. Runtime evidence (commit `eaefd59`, Qwen3-4B-Instruct-2507)

### 3.1 canonical teacher exports

```
outputs/gated/teacher_exports/amazon/{train,validation,final_test}/teacher_export.parquet
  records: 8360 / 1194 / 2390     validation AUROC = 0.9744 passed = True
outputs/gated/teacher_exports/yelpchi/{train,validation,final_test}/teacher_export.parquet
  records: 32167 / 4595 / 9192    validation AUROC = 0.9867 passed = True
TeacherProvenance.code_git_sha = eaefd5946c823ec22c8f6191553fc1cdaa5a89d1
```

### 3.2 Stage 2 training (40 steps, B=2, 1024-sample cap, LoRA r=8 alpha=16)

```
outputs/gated/stage2/amazon/run_v1/
  losses: total 5.53 -> 2.05    (L_gen 2.77 -> 1.81, L_cls 2.12 -> 0.06, L_distill 1.28 -> 0.36)
  post-training validation (n=128): AUROC = 0.5756
outputs/gated/stage2/yelpchi/run_v1/
  losses: total 5.37 -> 2.17    (L_gen 2.73 -> 1.83, L_cls 1.68 -> 0.16, L_distill 0.63 -> 0.38)
  post-training validation (n=128): AUROC = 0.6328
```

`CanonicalStepReport._losses_finite` validator (1e-5 equality between
`total_loss` and the weighted sum) held for every logged step.

### 3.3 Head-only inference on final_test

```
outputs/gated/eval/amazon/scorer_report_amazon_final_test.json
  n_total=256 auroc=0.5093 auprc=0.0882 brier=0.0654
outputs/gated/eval/yelpchi/scorer_report_yelpchi_final_test.json
  n_total=256 auroc=0.6287 auprc=0.1889 brier=0.1229
```

### 3.4 Formal head-only eval

```
outputs/formal/head_only/amazon/formal_head_only_report_amazon.json
  validation-frozen threshold = 0.0647
  final_test: auroc=0.5093 auprc=0.0882 f1=0.1053 prec=0.0690 rec=0.2222
outputs/formal/head_only/yelpchi/formal_head_only_report_yelpchi.json
  validation-frozen threshold = 0.1394
  final_test: auroc=0.6287 auprc=0.1889 f1=0.1818 prec=0.1750 rec=0.1892
```

All four `FormalHeadOnlyReport` cross-checks cleared:

- `population_metadata.population_name == "final_test"` and
  `contains_tuning_rows == False`,
- `validation_population_metadata.population_name == "validation"` and
  `contains_tuning_rows == True`,
- `calibration.population_name == "final_test"`,
- `scorer_checkpoint_provenance == checkpoint_bundle.cls_head.to_shared_checkpoint_provenance()`,
- `validation_threshold.source_population_name == validation`.

### 3.5 Test suite

`PYTHONPATH=. pytest tests/ -q` -> 216 passed, 0 regressions.

## 4. Risks / follow-ups (carried over into status_package)

- **Training is a pipeline proof-of-run, not a converged model.**  40
  steps on a 1024-sample cap is intentionally small; numbers above
  should not be interpreted as "this is what the student can do".  A
  production run needs TRAIN full-size, more steps, a learning-rate
  schedule, and best-checkpoint selection on the validation scorer.
- **Only the head-only formal runner is wrapped in a CLI.**
  `eval_gen_only`, `eval_fusion`, and `faithfulness` still require
  LLM text generation + parsing plumbing outside the head-only path;
  the library functions are fully tested but the CLIs are not yet
  written.  Documented as explicit follow-ups.
- **Gate manifest auto-assembly is still manual.** A
  `scripts/generate_gate_manifest.py` that reads the produced reports
  and composes a `GateManifest` is a natural next step.
