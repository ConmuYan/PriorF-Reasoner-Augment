# `docs/050_fail_closed_guardrails.md`

## Fail-Closed Guardrails

This file lists behaviors that must block execution rather than degrade gracefully.

## 1. Formal launch blockers

The following must block a formal launch:

* missing gate manifest
* invalid gate manifest schema
* failed data validation
* failed teacher baseline
* failed head gate
* failed validation/eval parity
* failed student contribution gate
* failed strict schema parse gate

## 2. Canonical trainer blockers

The following must block canonical joint training:

* generation loss disabled
* classification loss disabled
* distillation loss disabled
* frozen-backbone probe mode invoked inside canonical trainer
* prompt-only probe mode silently changing the objective
* diagnostic imbalance recipe routed through canonical code

## 3. Evaluation blockers

The following must block formal evaluation:

* threshold selected on the same rows being reported
* alpha selected on the same rows being reported
* population metadata missing
* unknown graph regime
* teacher-prob ablation missing when teacher probability is present in Evidence Card

## 4. Faithfulness blockers

The following must block formal faithfulness:

* ablation changes the overall prompt schema
* alpha differs from the frozen formal alpha
* different prompt builder is used
* different tokenization path is used
* evaluation set too small for formal reporting

## 5. Script blockers

The following script behaviors are forbidden:

* continue-after-failure warnings in smoke
* full-pipeline launch without gate check
* implicit fallback to final checkpoint when formal path requires best checkpoint
* reuse of diagnostic output namespace by canonical launchers

## 6. Documentation blockers

Formal README or alignment claims are forbidden unless executable gates pass.

No checklist or README may claim:

* “implemented”
* “verified”
* “aligned”

unless backed by actual gate artifacts.
