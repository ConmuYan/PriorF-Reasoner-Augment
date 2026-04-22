# `docs/040_framework_reuse_and_adoption.md`

## Community Scaffold Reuse Path

This is not “open yet another framework branch.”
It exists to keep Harness Agent from drifting.

The final reuse route is fixed as follows.

### Main repository implemented in-house

The project should be implemented in the main repository because there are too many contracts that generic training frameworks will not protect for you:

* graph regime
* Evidence Card
* teacher-prob audit
* parity
* frozen alpha
* population naming
* fail-closed gate

### Reuse TRL for structured generation training

TRL’s `SFTTrainer` supports standard and conversational data formats, prompt-completion data, and can automatically apply the chat template to conversational datasets; assistant-only loss is also explicitly supported. ([Hugging Face][1])

### Reuse Accelerate for joint training, not Trainer

Because this project needs:

* explicit control over the three loss terms
* explicit control over the hidden-state path
* explicit control over validation/offline parity
* explicit control over distributed evaluation gathering

This makes a custom Accelerate loop the correct fit. ([Hugging Face][2])

### Borrow torchtune’s recipe style, not its main repository

PyTorch describes torchtune as PyTorch-native, modular, hackable, recipe-oriented, and built around readable self-contained recipes. This style is worth borrowing when designing the canonical trainer. ([PyTorch Docs][3])

### Borrow LitGPT’s directory and config style

LitGPT explicitly emphasizes “No abstractions,” LoRA, QLoRA, YAML recipes, and editable readable code, which makes it useful as a directory and scaffolding reference. ([GitHub][4])

### Use LLaMA-Factory as a baseline sanity check only

LLaMA-Factory supports Qwen3, Gemma, LoRA, QLoRA, FlashAttention, and vLLM workers, which makes it useful for quick recipe sanity checks, but not as the canonical codebase. ([GitHub][5])

### Use Axolotl only as a heavyweight fallback

Axolotl has an official Qwen3 guide and is suitable for heavier YAML-driven fallback experimentation, but it should not be the canonical path for this project. ([Axolotl][6])

## Adoption Decision for SAC’s Four Files

### Can be kept as idea sources

* `eval/head_scoring.py`
* `scripts/gate_check.py`

### Must be rewritten before entering the canonical path

* `llm/hidden_state_pooling.py`
* `tests/test_eval_head_parity.py`

## Immediate Landing Tasks

The next implementation steps should be executed in this order:

1. rewrite `llm/hidden_state_pooling.py`
2. write `tests/test_hidden_state_pooling.py`
3. write `eval/head_scoring.py`
4. write `tests/test_eval_head_parity.py`
5. write a formal `GateManifest` Pydantic model
6. write `scripts/gate_check.py`
7. make `run_full_pipeline.sh` fail immediately when no gate manifest exists

[1]: https://huggingface.co/docs/trl/v0.25.1/en/sft_trainer?utm_source=chatgpt.com "SFT Trainer · Hugging Face"
[2]: https://huggingface.co/docs/accelerate/v1.9.0/en/quicktour?utm_source=chatgpt.com "Quicktour · Hugging Face"
[3]: https://docs.pytorch.org/blog/torchtune-fine-tune-llms/?utm_source=chatgpt.com "torchtune: Easily fine-tune LLMs using PyTorch | PyTorch"
[4]: https://github.com/Lightning-AI/litgpt?utm_source=chatgpt.com "GitHub - Lightning-AI/litgpt: 20+ high-performance LLMs with recipes to pretrain, finetune and deploy at scale. · GitHub"
[5]: https://github.com/hiyouga/LlamaFactory/blob/main/README.md?utm_source=chatgpt.com "LlamaFactory/README.md at main · hiyouga/LlamaFactory · GitHub"
[6]: https://docs.axolotl.ai/docs/models/qwen3.html?utm_source=chatgpt.com "Qwen 3 – Axolotl"
