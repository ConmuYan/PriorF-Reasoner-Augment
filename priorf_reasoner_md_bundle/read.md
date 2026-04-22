# `read.md`

## PriorF-Reasoner Execution Readme

This file is the operator-facing execution manual.

If you need to know what to run, what to check, and what counts as success or failure, use this file.

## 1. What this project is trying to prove

The project is trying to prove that a compact language model can read teacher-derived structural evidence and produce:

* useful student fraud probabilities
* structured, auditable explanations
* meaningful teacher-student fusion

It is not trying to prove that a language model can replace PriorF-GNN on raw graph input.

## 2. What counts as success

The system is successful only if all of the following hold:

* the student head is meaningfully above random
* validation and offline head eval agree
* fusion is not teacher fallback
* strict JSON schema compliance is stable
* faithfulness artifacts are valid and non-empty

## 3. What does not count as success

The following do not count:

* pretty generation with a collapsed head
* high fusion with alpha near zero
* metrics tuned on evaluation/test-like labels
* passing tests that do not exercise the canonical path
* diagnostic probes saved under formal names

## 4. Required Run Order

### Step A. Prepare and validate data

Run the data preparation path first.
If no data manifest is produced, stop.

### Step B. Validate teacher baseline

Run the teacher baseline check.
If the teacher is not aligned with expected benchmark quality, stop.

### Step C. Export teacher evidence

Export the structural evidence with full provenance and graph regime declaration.

### Step D. Build Evidence Cards and datasets

Create schema-validated Evidence Cards and training/eval datasets.

### Step E. Run structured generation training

Use TRL `SFTTrainer` for schema-stable generation training. TRL’s SFT trainer supports prompt-completion and conversational formats and can automatically apply the chat template to conversational data, which matches the structured generation phase of this project. ([Hugging Face][1])

### Step F. Run canonical joint training

Use the custom Accelerate training loop.

Accelerate is selected here because the canonical path needs explicit control over multi-loss training and distributed evaluation, including deduplicated metric gathering with `gather_for_metrics()`. ([Hugging Face][2])

### Step G. Run parity checks

Before any formal head report:

* run `tests/test_eval_head_parity.py`
* confirm epoch-end validation and offline eval match

### Step H. Run formal evaluation

Run:

* head-only evaluation
* generation evaluation
* fusion evaluation
* faithfulness evaluation

### Step I. Run gate check

Formal runs require a valid gate manifest.

## 5. Structured Generation Training

The structured generation phase uses the student to produce strict JSON outputs from Evidence Cards.

Required properties:

* strict output schema
* canonical serialization order
* no alias fallback in formal metrics
* schema-valid examples in training data

## 6. Canonical Joint Training

Canonical joint training is the only accepted main training path.

It must include:

* generation loss
* classification loss
* distillation loss

It must not silently mutate into:

* frozen head probe
* no-generation objective
* no-distillation objective
* class-imbalance-only experiment

## 7. Hidden-State Rules

The classification head consumes a single canonical hidden state.

The extraction function must:

* be mask-aware
* be absolute-index correct
* avoid direct `[:, -1, :]`
* be used identically in train/val/eval

## 8. Threshold Rules

Formal test-like reports cannot choose thresholds on the rows being reported.

Allowed:

* AUROC
* AUPRC
* metrics at frozen validation threshold

Diagnostics may include oracle thresholds but must be clearly separated.

## 9. Fusion Rules

Fusion alpha is selected on validation only.

Formal fusion reports must include:

* `optimal_alpha`
* student-only metrics
* teacher-only metrics
* fusion metrics
* student contribution pass/fail

## 10. Teacher-Probability Dependency Audit

If teacher probability is present in the Evidence Card, report both:

* full-card student performance
* teacher-prob-ablated student performance

Do not claim independent student detection without this comparison.

## 11. Faithfulness Rules

Faithfulness must use schema-preserving ablations and the same inference path as formal evaluation.

Required outputs:

* sufficiency
* comprehensiveness
* evidence-ablation impact

Small-sample faithfulness runs are smoke only.

## 12. Fail Conditions

Stop immediately if any of the following occur:

* missing manifest
* invalid schema
* overlap between populations
* parity mismatch
* NaN in training or eval
* gate failure
* formal launch without gate manifest

## 13. Framework Reuse Policy

Use the ecosystem tactically, not blindly.

### Primary implementation

* Transformers
* PEFT
* TRL
* Accelerate

### Borrow engineering patterns from

* torchtune for hackable recipes and minimal abstraction. PyTorch describes torchtune as PyTorch-native, modular, and built around readable, self-contained recipes. ([PyTorch Docs][3])
* LitGPT for readable, low-abstraction code and validated YAML recipes; its README explicitly highlights “No abstractions”, YAML recipes, LoRA, QLoRA, and readable code. ([GitHub][4])

### Use as sanity baselines only

* LLaMA-Factory supports Qwen3, Gemma, LoRA, QLoRA, and vLLM deployment, which makes it useful as a baseline or fallback recipe check, not as the canonical codebase. ([GitHub][5])
* Axolotl also has an official Qwen3 guide and can be used for heavyweight fallback experimentation, but not as the canonical implementation. ([Axolotl][6])

### Inference acceleration only

* vLLM belongs after correctness, for batch inference or service acceleration only.

## 14. Operator Summary

If the head is broken, stop.
If parity is broken, stop.
If alpha is near zero, stop calling fusion a success.
If the schema is only passing under normalization, stop calling it compliant.
If a script continues after failure, fix the script before running again.

[1]: https://huggingface.co/docs/trl/v0.25.1/en/sft_trainer?utm_source=chatgpt.com "SFT Trainer · Hugging Face"
[2]: https://huggingface.co/docs/accelerate/v1.9.0/en/quicktour?utm_source=chatgpt.com "Quicktour · Hugging Face"
[3]: https://docs.pytorch.org/blog/torchtune-fine-tune-llms/?utm_source=chatgpt.com "torchtune: Easily fine-tune LLMs using PyTorch | PyTorch"
[4]: https://github.com/Lightning-AI/litgpt?utm_source=chatgpt.com "GitHub - Lightning-AI/litgpt: 20+ high-performance LLMs with recipes to pretrain, finetune and deploy at scale. · GitHub"
[5]: https://github.com/hiyouga/LlamaFactory/blob/main/README.md?utm_source=chatgpt.com "LlamaFactory/README.md at main · hiyouga/LlamaFactory · GitHub"
[6]: https://docs.axolotl.ai/docs/models/qwen3.html?utm_source=chatgpt.com "Qwen 3 – Axolotl"
