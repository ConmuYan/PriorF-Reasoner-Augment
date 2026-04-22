# `docs/010_project_contract.md`

## PriorF-Reasoner Project Contract

### Purpose

This file defines the hard implementation contract.
All code, scripts, configs, and evaluations must satisfy this document.

## 1. Scope

The project uses:

* standard Amazon benchmark `.mat`
* standard YelpChi benchmark `.mat`
* PriorF teacher artifacts
* Evidence Cards
* Qwen3-4B student
* structured generation training
* canonical joint training
* head-only evaluation
* fusion evaluation
* faithfulness evaluation

The project does not use:

* raw review text
* raw graph matrices as LLM input
* multimodal inputs
* tool calling
* pseudo-label augmentation
* ranking loss
* reinforcement learning fine-tuning
* model scaling as a substitute for correctness

## 2. Canonical Paths

There are three permitted execution paths.

### Canonical

For formalizable implementation and gated runs.

### Diagnostic

For probes, ablations, and repairs only.

### Formal

For archival and reporting only after all gates pass.

## 3. Artifact Contracts

### 3.1 Data manifest

All train and eval entrypoints must require a validated data manifest.

### 3.2 Teacher export

Teacher export must include graph regime and provenance.

### 3.3 Evidence Card

Evidence Cards must be schema-validated and ablation-compatible.

### 3.4 Output schema

Strict schema output is the only formal parse metric.

### 3.5 Gate manifest

Formal launches require a valid gate manifest.

## 4. Population Contracts

Every artifact that contains examples or predictions must record:

* `population_name`
* `split_values`
* `node_ids_hash`
* `contains_tuning_rows`
* `contains_final_test_rows`

Allowed population names include:

* `train`
* `validation`
* `unused_holdout`
* `diagnostic_holdout`
* `final_test`

The name `test` must not be used ambiguously.

## 5. Label Contracts

The system must not overload one field called `label`.

Separate fields are mandatory:

* `ground_truth_label`
* `teacher_label`
* `teacher_prob`
* `sft_target_label`
* `pred_label`
* `pred_score`

## 6. Prompt Contract

One prompt builder only.

All train / validation / eval / generation paths must call the same builder.

## 7. Tokenization Contract

One tokenization path only.

All formatted messages must flow through the same chat-template serialization logic.

## 8. Hidden-State Contract

One hidden-state pooling implementation only.

Forbidden:

```python id="dme4wk"
hidden_states[:, -1, :]
```

Required:

* mask-aware absolute token indexing
* padding-side consistency
* parity-tested extraction

## 9. Canonical Loss Contract

Canonical joint training must contain:

* generation loss
* classification loss
* distillation loss

Disabling any one of these invalidates the canonical run.

## 10. Distillation Contract

The canonical distillation target is the teacher probability, not raw teacher logits.

The default canonical distillation loss is:

* BCEWithLogitsLoss against clipped `teacher_prob`

KL and MSE are diagnostics only.

## 11. Validation Contract

Validation must use the same scorer path as offline head-only evaluation.

Validation-selected checkpoints are only valid if offline parity is proven.

## 12. Threshold Contract

Thresholds may be selected on validation only.

Test-like populations may report:

* threshold-free metrics
* metrics at a frozen validation threshold

Oracle-threshold metrics are diagnostics only.

## 13. Fusion Contract

Fusion alpha may be selected on validation only.

Formal fusion evaluation must accept distinct validation and test prediction inputs.

## 14. Teacher-Probability Audit Contract

If Evidence Cards include teacher probability, formal reporting must include a teacher-prob-ablated student evaluation.

## 15. Faithfulness Contract

Faithfulness must use the same prompt, tokenization, and inference path as formal evaluation.

Schema-preserving ablations only.

## 16. Fail-Closed Contract

Any of the following must halt execution:

* missing manifest
* invalid schema
* gate failure
* parity failure
* data overlap
* training NaN
* eval NaN
* silent objective mutation
* formal run without gates
