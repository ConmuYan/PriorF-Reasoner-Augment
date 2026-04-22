# `docs/020_top_level_design.md`

## PriorF-Reasoner Top-Level Design

### Document Status

This file is the top-level architectural contract for the project.

If implementation details, scripts, or historical notes conflict with this file, this file wins.

## 1. Project Intent

PriorF-Reasoner is not a generic LLM fraud classifier and not a replacement for PriorF-GNN.

It is a structured teacher-student system:

```text id="vjlwmg"
PriorF-GNN teacher
  -> structural evidence export
  -> Evidence Card
  -> Qwen3-4B student reasoner
  -> head_only / gen_only / fusion
  -> faithful explanation + useful student probability
```

The student learns from structured structural evidence, not from raw graph matrices and not from raw text.

## 2. Core Claim

The project claims:

> Structural priors extracted by PriorF-GNN can be serialized into Evidence Cards that allow a compact language model to learn auditable fraud reasoning, produce stable structured explanations, and contribute useful classification signal through a head and teacher-guided distillation.

## 3. System Components

### 3.1 PriorF-GNN teacher

Responsibilities:

* produce baseline fraud detection performance
* export structural signals
* define the graph-aware evidence space

Required outputs include:

* `teacher_prob`
* `teacher_logit`
* `hsd`
* `hsd_quantile`
* `asda_switch`
* `mlp_logit`
* `gnn_logit`
* `branch_gap`
* relation discrepancy profile
* neighbor statistics
* routing and discrepancy flags
* graph regime metadata

### 3.2 Evidence Card

The Evidence Card is the bridge between graph structure and the language model.

It must be:

* schema-stable
* non-leaking
* serializable
* auditable
* ablatable

It must summarize:

* teacher summary
* discrepancy summary
* relation profile
* neighborhood structure summary
* explicit task instruction

### 3.3 Qwen3-4B student

The student must learn two things:

1. structured reasoning from Evidence Cards
2. hidden representations that support a reliable fraud classification head

Qwen3 is selected because the family supports switching between thinking and non-thinking modes, which is important for stable schema-constrained generation. ([Hugging Face][1])

### 3.4 Classification head

The classification head is the independent student detector.

Formal definition:

```text id="t3t2n8"
p_cls = cls_head(hidden_prompt_only)
```

This is evaluated separately as `head_only`.

### 3.5 Fusion

Fusion is defined as:

```text id="cpdft7"
p_final = alpha * p_cls + (1 - alpha) * p_teacher
```

Alpha is selected on validation only and then frozen.

Interpretation:

* alpha near 0 → teacher fallback
* moderate alpha with stable gains → real student contribution

## 4. Canonical Output Semantics

The formal model output schema is:

```json id="j7tvjj"
{
  "rationale": "...",
  "evidence": ["..."],
  "pattern_hint": "...",
  "label": "fraud",
  "score": 0.93
}
```

Important distinction:

* JSON semantics do not depend on field order
* generation training does depend on token order

Therefore, canonical serialization order must be:

1. `rationale`
2. `evidence`
3. `pattern_hint`
4. `label`
5. `score`

This forces “reason first, conclude last”.

## 5. Classification Contract Is Independent From Generation Order

The classification head must not depend on whether the model has already generated label tokens.

The classification path is defined over prompt-only hidden states and must remain independent from generation order.

This decouples:

* auto-regressive generation behavior
* classification hidden-state extraction

## 6. Graph Regime Contract

The system must explicitly declare the graph regime used for teacher export and evaluation.

Allowed values:

* `transductive_standard`
* `inductive_masked`

This field must be propagated into:

* teacher export
* Evidence Card artifacts
* evaluation reports
* gate manifest

Formal results are only comparable when teacher, student, and evaluation share the same graph regime.

## 7. Leakage Policy

### Allowed

* structural information derived from the graph under the declared graph regime
* teacher probability if explicitly represented and audited

### Forbidden

* any feature derived from non-training labels
* validation/test labels inside Evidence Cards
* hidden pseudo-label propagation into formal evaluation
* threshold or alpha tuning on evaluation/test-like populations

## 8. Teacher-Probability Dependency Audit

If Evidence Cards contain `teacher_summary.teacher_prob`, then student contribution claims must be audited with two variants:

1. full Evidence Card
2. teacher-prob-ablated Evidence Card

If ablation collapses `head_only`, student must be reported as teacher-prob dependent.

## 9. Canonical Training Phases

### 9.1 Structured generation training

Purpose:

* learn strict schema-valid structured outputs
* stabilize reasoning text generation

### 9.2 Canonical joint training

Purpose:

* retain structured generation behavior
* learn a fraud classification head
* align head confidence with teacher probabilities

Formal objective:

```text id="zjlwmz"
L = L_gen + λ_cls * L_cls + λ_distill * L_distill
```

All three terms are mandatory in canonical training.

### 9.3 Diagnostic probes

Probe-style experiments exist outside canonical training and do not count as system completion.

## 10. Formal Evaluation Dimensions

The system is evaluated along three axes:

### 10.1 `head_only`

Independent student quality.

### 10.2 `fusion`

Teacher-student cooperation under frozen validation-selected alpha.

### 10.3 explanation faithfulness

At minimum:

* sufficiency
* comprehensiveness
* evidence-ablation impact

## 11. Formal Failure Conditions

The system is considered failed if any of the following is true:

* `head_only` collapses toward random or near-constant probabilities
* `fusion` only looks strong because alpha is near zero
* validation and offline eval disagree
* output schema compliance is achieved only via forgiving alias normalization
* explanation quality looks good but faithfulness is empty or broken
* thresholds or alpha are tuned on test-like data

## 12. Launch Policy

The project may scale only after:

* data contract passes
* teacher baseline gate passes
* validation/eval parity passes
* student contribution is non-trivial
* formal gate manifest is valid

Until then, only canonical gated runs and diagnostic runs are allowed.

[1]: https://huggingface.co/Qwen/Qwen3-8B/blob/main/README.md?utm_source=chatgpt.com "README.md · Qwen/Qwen3-8B at main"
