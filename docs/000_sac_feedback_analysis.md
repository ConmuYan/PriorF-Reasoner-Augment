# `docs/000_sac_feedback_analysis.md`

## SAC Feedback Analysis Conclusion

### Overall judgment

The direction is correct, and the code skeleton has value, but it should not be accepted as-is.

The single most important conclusion is:

**Among SAC’s four infrastructure files, `gate_check.py` can serve as a draft, `head_scoring.py` can serve as a minimal skeleton, while `hidden_state_pooling.py` and `test_eval_head_parity.py` cannot be used directly and must be corrected before entering the canonical path.**

## 1. Most valuable points to formally absorb from the feedback

The following should enter the final contract as formal requirements:

* **Reverse generation order**: generate `rationale` / `evidence` first, then `label` / `score`. This reduces the auto-regressive bias of “deciding first, justifying later.”
* **Independent hidden-state contract for the classification head**: `head_only` remains defined on prompt-only hidden states and must not depend on generation order.
* **Use soft-target BCEWithLogitsLoss for distillation**: more stable than MSE on teacher logits and more robust than default KL in this setting.
* **Explicit `graph_regime` plus label-leak audit**: more correct than force-cutting everything into inductive mode, and preserves comparability with the original baseline.

## 2. Critical mathematical error in SAC’s `hidden_state_pooling.py`

SAC uses:

```python
sequence_lengths = attention_mask.sum(dim=1) - 1
cls_hidden = hidden_states[batch_indices, sequence_lengths]
```

This is only correct under **right padding**.
It is incorrect under **left padding**.

Example:

If `attention_mask = [0,0,1,1,1,1,1,1,1,1]`, the effective length is 8, so `sum(mask)-1 = 7`.
But the true index of the last valid token is **9**, not 7.

That means SAC’s implementation **takes the wrong token exactly in the left-padding setting it emphasizes most**.

This must be made explicit in the Harness Agent guidance text:
**do not copy this implementation as-is.**

There are two safe formal approaches:

* **Option A**: force left padding and assert no right-end pad, in which case the last valid token is always `seq_len - 1`
* **Option B**: make it padding-side agnostic by computing the absolute position of the last `1` from the mask

The formal path should use **Option B**, because it is more robust and easier to parity test.

A safe pattern looks like:

```python
positions = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0).expand_as(attention_mask)
last_token_idx = (positions * attention_mask.long()).max(dim=1).values
```

and then gather from that index.

## 3. Why SAC’s `test_eval_head_parity.py` also cannot be accepted directly

It has at least four problems:

* the example code itself has formatting issues and is easy to copy into a syntax error
* its left-padding assertion contradicts its own `sum(mask)-1` formula
* after `head.train()` it does not actually disable dropout; it only comments that this should be done in reality, which is not formal parity
* it only validates pooling and a bare `Linear`, but does not yet validate:

  * identical prompt builder
  * identical tokenizer/chat template
  * identical hidden extraction
  * identical `predict_proba`
  * sample-wise identity between train-loop validation and offline eval

Therefore, the example may be kept as part of **operator-level unit testing**, but it cannot replace the formal `tests/test_eval_head_parity.py`.

## 4. `head_scoring.py` is directionally correct, but not yet formal

Its strengths:

* explicitly requires train/eval to share one scorer
* uses only threshold-free metrics
* emphasizes the prompt-only path

But it still lacks:

* distributed evaluation via `gather_for_metrics()`; the Accelerate quicktour explicitly recommends this for deduplicated distributed evaluation. ([Hugging Face][1])
* AUROC/AUPRC edge protection for single-class batch or single-class population
* probability spread / calibration outputs
* frozen validation threshold as a separate artifact
* population metadata
* checkpoint provenance

So it can be used as a **minimal scorer draft**, but not yet as the formal evaluator.

## 5. `gate_check.py` is the closest of the four to being directly absorbable

Its direction is basically correct:

* use Pydantic modeling
* fail immediately on missing fields
* exit non-zero if any gate fails

But the formal version still needs:

* `schema_version`
* `graph_regime`
* `strict_schema_parse_pass`
* `teacher_prob_ablation_pass`
* `smoke_pipeline_pass`
* `population_contract_pass`
* strict `generated_at` time format
* `config_fingerprint`
* `data_manifest_hash`

So:

**it is a good first brick, but not yet the final gate system.**

## 6. SAC feedback is consistent with the current logs

The current logs you uploaded already show that the nominal cooperative training currently degenerates into:

* `compute_gen_loss=False`
* `prompt_only_cls=True`
* `freeze_backbone=True`
* `lambda_distill=0.00`
* `use_weighted_sampler=True`
* `cls_head_lr=1e-2`

This combination already co-occurs with a validation threshold collapsing to `1.0000` and a significantly reduced variance, which indicates that the objective and validation semantics are indeed drifting.

So the correct action is not “paste SAC’s four files directly into the repo.”

It is:

**absorb SAC’s design intent into the formal contract, while treating its example code as a draft that requires correction.**

## Community Scaffold Reuse Path

This is not an invitation to create yet another framework branch.
It exists to prevent Harness Agent from drifting.

The fixed reuse route is:

### Implement the main repository yourself

This project has too many contracts that generic training frameworks will not protect automatically:

* graph regime
* Evidence Card
* teacher-prob audit
* parity
* frozen alpha
* population naming
* fail-closed gate

### Reuse TRL for structured generation training

TRL’s `SFTTrainer` supports standard and conversational dataset formats, prompt-completion formats, and can automatically apply chat templates to conversational datasets; assistant-only loss is also supported. ([Hugging Face][2])

### Reuse Accelerate for joint training, not Trainer

Because you need:

* explicit control over three loss terms
* explicit control over the hidden-state path
* explicit control over validation/offline parity
* explicit control over distributed eval gathering

This is a better fit for a custom Accelerate loop. ([Hugging Face][1])

### Borrow torchtune’s recipe style, not its main repo

PyTorch describes torchtune as PyTorch-native, composable, hackable, recipe-oriented, and low-abstraction. That style is worth borrowing for the canonical trainer. ([PyTorch Docs][3])

### Borrow LitGPT’s directory and config style

LitGPT explicitly emphasizes “No abstractions,” LoRA, QLoRA, YAML recipes, and easily modifiable code, which makes it useful as scaffolding inspiration. ([GitHub][4])

### Use LLaMA-Factory only as a sanity baseline

LLaMA-Factory supports Qwen3, Gemma, LoRA, QLoRA, FlashAttention, and vLLM workers, which makes it useful as a “can an off-the-shelf recipe also run?” sanity check, but not as the canonical codebase. ([GitHub][5])

### Use Axolotl only as a heavyweight fallback

Axolotl has an existing Qwen3 guide and is suitable for heavier YAML-driven experimentation, but for this project it should remain a fallback tool only. ([Axolotl][6])

## Whether to adopt SAC’s four files directly

The final recommendation is very clear:

### Keep as idea sources

* `eval/head_scoring.py`
* `scripts/gate_check.py`

### Rewrite before entering the canonical path

* `llm/hidden_state_pooling.py`
* `tests/test_eval_head_parity.py`

### Why they must not be copied directly

The core indexing formula in `hidden_state_pooling.py` is wrong for left padding.
The example in `test_eval_head_parity.py` contains both implementation contradictions and incomplete formal parity coverage.

## First landing tasks for Harness Agent

The order should be fixed as follows:

1. rewrite `llm/hidden_state_pooling.py`
2. write `tests/test_hidden_state_pooling.py`
3. write `eval/head_scoring.py`
4. write `tests/test_eval_head_parity.py`
5. write a formal `GateManifest` Pydantic model
6. write `scripts/gate_check.py`
7. make `run_full_pipeline.sh` fail immediately when no gate manifest exists

[1]: https://huggingface.co/docs/accelerate/v1.9.0/en/quicktour?utm_source=chatgpt.com "Quicktour · Hugging Face"
[2]: https://huggingface.co/docs/trl/v0.25.1/en/sft_trainer?utm_source=chatgpt.com "SFT Trainer · Hugging Face"
[3]: https://docs.pytorch.org/blog/torchtune-fine-tune-llms/?utm_source=chatgpt.com "torchtune: Easily fine-tune LLMs using PyTorch | PyTorch"
[4]: https://github.com/Lightning-AI/litgpt?utm_source=chatgpt.com "GitHub - Lightning-AI/litgpt: 20+ high-performance LLMs with recipes to pretrain, finetune and deploy at scale. · GitHub"
[5]: https://github.com/hiyouga/LlamaFactory/blob/main/README.md?utm_source=chatgpt.com "LlamaFactory/README.md at main · hiyouga/LlamaFactory · GitHub"
[6]: https://docs.axolotl.ai/docs/models/qwen3.html?utm_source=chatgpt.com "Qwen 3 – Axolotl"
