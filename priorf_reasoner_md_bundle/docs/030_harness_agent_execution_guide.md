# `docs/030_harness_agent_execution_guide.md`

## Harness Agent Execution Guide

### Mission

You are not here to maximize benchmark numbers quickly.
You are here to preserve contract correctness while rebuilding the system into a fail-closed research pipeline.

## 1. Operational Principle

At every task boundary, ask:

1. Is this canonical or diagnostic?
2. Does this code touch a contract boundary?
3. Does this change require a new test?
4. Would this allow silent leakage or silent objective mutation?
5. Would this allow a formal-looking run without real gates?

If any answer is unsafe, stop and narrow the scope.

## 2. Mandatory Work Sequence

Harness Agent must execute tasks in this order:

### Task 1. Data foundations

Deliver:

* `graph_data/mat_loader.py`
* `graph_data/validators.py`
* `graph_data/manifests.py`

Acceptance:

* validated data manifest emitted
* overlap assertions implemented
* shape / NaN / relation count checks pass

### Task 2. Teacher contract

Deliver:

* `priorf_teacher/schema.py`
* `priorf_teacher/teacher_baseline_gate.py`
* `priorf_teacher/export_pipeline.py`

Acceptance:

* teacher report emitted
* export artifacts contain graph regime and provenance
* downstream launch blocks when teacher gate fails

### Task 3. Evidence contract

Deliver:

* `evidence/evidence_schema.py`
* `evidence/prompt_builder.py`
* `evidence/output_schema.py`

Acceptance:

* Evidence Card validates under Pydantic
* prompt builder emits canonical ordering
* output schema strictly validates reversed serialization order

### Task 4. Hidden-state contract

Deliver:

* `llm/hidden_state_pooling.py`

Acceptance:

* mask-aware absolute indexing
* no use of `[:, -1, :]`
* padding edge cases covered

### Task 5. Unified head scoring

Deliver:

* `eval/head_scoring.py`

Acceptance:

* shared by validation and offline eval
* threshold-free metrics only at scorer layer
* returns probabilities, labels, metadata

### Task 6. Parity test

Deliver:

* `tests/test_eval_head_parity.py`

Acceptance:

* same prompt builder
* same chat template
* same token indices
* same hidden-state indices
* same probabilities
* same predictions between validation and offline eval

### Task 7. Canonical trainer

Deliver:

* `train/train_stage2_canonical.py`

Acceptance:

* `L_gen`, `L_cls`, `L_distill` all present
* all backprop correctly
* no NaN
* no silent mutation
* checkpoint provenance recorded

### Task 8. Formal eval

Deliver:

* `eval/eval_head_only.py`
* `eval/eval_gen_only.py`
* `eval/eval_fusion.py`
* `eval/faithfulness.py`

Acceptance:

* validation-only thresholding
* validation-only alpha tuning
* teacher-prob ablation supported
* formal and diagnostic outputs separated

### Task 9. Gate system

Deliver:

* `scripts/gate_check.py`
* `gate_manifest.json` schema
* launcher integration

Acceptance:

* formal launch blocked without manifest
* failure is fatal
* no manual bypass without override provenance

## 3. Coding Rules

### Pydantic contracts first

Never write downstream logic before formalizing the schema.

### Test before expanding

If you implement a contract module and there is no test for it, stop and write the test.

### No multi-purpose trainer

Canonical and diagnostic trainers must be separate modules.

### No hidden behavior

Do not encode formal behavior in defaults and diagnostic behavior in environment variables.

Formal behavior must be explicit.

## 4. Prompt Builder Rules

Prompt builder must:

* produce one and only one canonical message structure
* serialize output examples in canonical reversed order
* support generation and head evaluation modes without changing semantic content
* preserve schema under ablation

## 5. Hidden-State Pooling Rules

The pooling implementation must satisfy all of the following:

* derive indices from the attention mask
* use absolute token positions
* work correctly under left padding
* fail on all-pad rows
* be tested on left and right padding
* be used identically in train and eval

The SAC sample implementation must not be copied as-is because `sum(mask)-1` is incorrect for left padding.

## 6. Parity Rules

Parity means more than “metrics are close”.

Parity must verify:

* identical message serialization
* identical token spans
* identical hidden-state extraction
* identical classifier input tensor
* identical logits
* identical probabilities

If parity fails, stop and fix upstream logic.
Do not re-tune thresholds or loss weights.

## 7. Distillation Rules

Canonical distillation must use teacher probability as a soft target with BCEWithLogitsLoss.

This is the only default canonical distillation mode.

Alternative distillation losses go to diagnostics.

## 8. Gate Rules

Formal launch requires explicit success for:

* data validation
* teacher baseline
* head quality
* parity
* student contribution
* strict schema parsing
* smoke pipeline

If any are false, formal launch must exit non-zero.

## 9. Output Rules

Harness Agent must never write:

* diagnostic runs into formal folders
* formal-looking names for probe experiments
* mixed-population reports without metadata

## 10. Review Rules

Before finishing any task, Harness Agent must self-review against this checklist:

* Did I introduce a new silent default?
* Did I create a path that can mutate canonical behavior?
* Did I add a script that can bypass gates?
* Did I allow thresholds or alpha to look at test-like labels?
* Did I preserve population naming?
* Did I preserve label naming?
* Did I add or update tests?

If any answer is unsafe, revise before returning.
