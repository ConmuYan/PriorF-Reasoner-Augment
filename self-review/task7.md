# Task 7 Fail-Closed 自审（`train/train_stage2_canonical.py`）

范围：本轮新增/修改：

- `train/__init__.py`
- `train/train_stage2_canonical.py`
- `tests/test_canonical_trainer.py`
- `self-review/task7.md`

未实现 Task 8 formal eval / fusion / faithfulness，也未实现 Task 9 gate manifest / launcher。

## 1. Trainer 输入 contract

新增 Pydantic contract：

- `CanonicalTrainerConfig`
  - 强制记录 `model_name_or_path`，当前测试路径为 `/data1/mq/models/Qwen3-4B-Instruct-2507`
  - 强制记录 `dataset_name`、`graph_regime`、`output_dir`、`thinking_mode`
  - 强制显式训练超参：`learning_rate`、`train_batch_size`、`max_steps`、`gradient_accumulation_steps`
  - 强制 `lambda_cls > 0`、`lambda_distill > 0`
  - 强制 `teacher_prob_clip_min < teacher_prob_clip_max`
  - `require_generation_loss` / `require_classification_loss` / `require_distillation_loss` 均为 `Literal[True]`
  - `diagnostic_mode` / `frozen_backbone_probe` / `class_imbalance_recipe` 均为 `Literal[False]`

- `CanonicalTrainingSample`
  - 必须来自 `PopulationName.TRAIN`
  - 必须带 `ground_truth_label`、`sft_target_label`、`sft_target_score`、`teacher_prob`
  - `teacher_prob` 必须与 `EvidenceCard.teacher_summary.teacher_prob` 精确一致
  - 禁止 teacher-prob-ablated Evidence Card 进入 canonical distillation

- `CanonicalTrainingBatch`
  - 至少一个样本
  - batch 内 dataset / graph_regime 一致

- `CanonicalTrainerRunRecord`
  - 记录 config、checkpoint provenance、train/validation population metadata、graph_regime、last_step、validation_report
  - 不是 formal report，也不是 gate manifest

## 2. 三个 loss 的计算路径

`run_canonical_train_step()` 每个样本执行两条 forward：

1. `L_gen`
   - `build_prompt(mode=PromptMode.TRAIN, ...)`
   - `tokenizer.apply_chat_template(..., tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=False)`
   - `model(..., labels=labels, output_hidden_states=True, use_cache=False)`
   - 要求 `outputs.loss` 存在且为有限 scalar

2. `L_cls`
   - `build_prompt(mode=PromptMode.EVAL_HEAD, ...)`
   - 同一 chat-template 参数集合
   - `model(..., output_hidden_states=True, use_cache=False)`
   - `pool_last_valid_token(hidden_states[-1], attention_mask)`
   - `cls_head(prompt_only_hidden)`
   - `BCEWithLogitsLoss(student_logits, ground_truth_label)`

3. `L_distill`
   - 与 `L_cls` 使用同一 `student_logits`
   - teacher target 为 `teacher_prob.clamp(config.teacher_prob_clip_min, config.teacher_prob_clip_max)`
   - `BCEWithLogitsLoss(student_logits, clipped_teacher_prob)`

总目标固定为：

```text
L = L_gen + lambda_cls * L_cls + lambda_distill * L_distill
```

反向传播通过 `Accelerator.backward(total_loss)` 执行。

## 3. 哪些配置组合会 fail closed

以下情况直接报错：

- `require_generation_loss=False`
- `require_classification_loss=False`
- `require_distillation_loss=False`
- `lambda_cls <= 0`
- `lambda_distill <= 0`
- `teacher_prob_clip_min >= teacher_prob_clip_max`
- `diagnostic_mode=True`
- `frozen_backbone_probe=True`
- `class_imbalance_recipe=True`
- model 或 cls_head 无 trainable parameters
- `outputs.loss` 缺失
- `L_gen` / `L_cls` / `L_distill` / `total_loss` 非有限
- cls logits 非有限
- sample graph_regime 与 config 不一致
- sample 不是 train population
- sample.teacher_prob 与 Evidence Card teacher_prob 不一致

## 4. Validation checkpoint selection 如何依赖统一 scorer

新增 `run_validation_with_unified_scorer()`：

- 它只调用 `eval.head_scoring.score_head()`；
- 不实现第二套 validation scorer；
- 不选择 threshold；
- 不选择 alpha；
- 返回 `ScorerReport`，供后续 checkpoint selection / gate 层消费。

测试 `test_validation_checkpoint_path_must_call_unified_scorer` monkeypatch `score_head` 并断言 validation path 必须委托该共享 scorer。

## 5. 本轮没有做什么

- 未实现 full dataloader loop。
- 未加载真实 Qwen3 4B 权重训练。
- 未写 checkpoint 文件。
- 未实现 validation-selected best checkpoint policy。
- 未实现 formal head-only/gen-only/fusion/faithfulness eval。
- 未实现 gate manifest 或 launcher closure。
- 未引入 diagnostic trainer / frozen probe / oracle threshold。

## 6. 验证证据

已通过：

```bash
PYTHONPATH=. pytest -q tests/test_canonical_trainer.py
# 5 passed

PYTHONPATH=. ruff check train/train_stage2_canonical.py train/__init__.py tests/test_canonical_trainer.py
# All checks passed

PYTHONPATH=. mypy train/train_stage2_canonical.py tests/test_canonical_trainer.py --ignore-missing-imports
# Success: no issues found

PYTHONPATH=. pytest -q tests/
# 157 passed
```

## 7. 剩余风险

1. 当前是 canonical train step / contract 层，不是完整 epoch trainer；后续需要 dataloader、checkpoint write、validation schedule 包装，但不得绕过本 step。
2. `L_gen` 当前依赖模型返回 `outputs.loss`，真实 assistant-only label masking / TRL SFT 阶段仍属于后续训练集成工作。
3. checkpoint selection 尚未实现，只锁定 validation 必须通过统一 scorer；best checkpoint policy 需后续任务继续实现并记录 provenance。
