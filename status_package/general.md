项目：PriorF-Reasoner

当前状态：
- Task 1 已完成并已人工审核通过
- Task 1 的目标是数据基础设施
- Task 1 已实现 data manifest / mat_loader / validators 的基础合同
- Task 1 已通过对应测试并已提交
- Task 2 已完成并已人工审核通过
- Task 2 的目标是 Teacher 合同与 Baseline Gate
- Task 2 已实现 teacher export schema / teacher baseline gate / teacher export pipeline IO 与 fail-closed 前置
- Task 2 已通过对应测试
- Task 3 已完成并已人工审核通过
- Task 3 的目标是 Evidence Card schema / prompt builder / strict output schema / schema-preserving ablation API
- Task 3 已通过对应测试
- Task 4 已完成并已人工审核通过
- Task 4 的目标是 Hidden-State Contract（mask-aware absolute token indexing）
- Task 4 已实现 llm/hidden_state_pooling.py / llm/__init__.py
- Task 4 已通过对应测试并已提交
- Task 5 已完成并已人工审核通过
- Task 5 的目标是 Unified Head Scoring（validation 与 offline head-only eval 共用的 canonical predict_proba + 指标产出层）
- Task 5 已实现 eval/head_scoring.py / eval/__init__.py 与 tests/test_head_scoring.py（21 条用例 / 25 test nodes）
- Task 5 已通过对应测试并已提交
- 当前准备进入 Task 6

必须遵守的 source-of-truth 文件：
- docs/010_project_contract.md
- docs/020_top_level_design.md
- docs/030_harness_agent_execution_guide.md
- docs/050_fail_closed_guardrails.md
- read.md
- CLAUDE.md

执行策略：
- 每个 task 单独执行与审查
- 不跨 task 越界
- canonical / diagnostic / formal 必须严格分离
- 若发现需求冲突、schema 漂移、silent default、threshold/alpha leakage、provenance 断裂，必须 fail closed

当前任务：
Task 6：Validation/Offline Head Parity Test（尚未开始）

已完成文件：
- graph_data/mat_loader.py
- graph_data/validators.py
- graph_data/manifests.py
- priorf_teacher/schema.py
- priorf_teacher/teacher_baseline_gate.py
- priorf_teacher/export_pipeline.py
- evidence/__init__.py
- evidence/evidence_schema.py
- evidence/prompt_builder.py
- evidence/output_schema.py
- llm/__init__.py
- llm/hidden_state_pooling.py
- eval/__init__.py
- eval/head_scoring.py
- tests/test_mat_loader.py
- tests/test_manifests.py
- tests/test_teacher_export.py
- tests/test_teacher_baseline_gate.py
- tests/test_evidence_schema.py
- tests/test_prompt_builder.py
- tests/test_output_schema.py
- tests/test_hidden_state_pooling.py
- tests/test_head_scoring.py
- self-review/task2.md
- self-review/task3.md
- self-review/task4.md
- self-review/task5.md

Task 3 验收证据：
- PYTHONPATH=. pytest -q tests/test_output_schema.py tests/test_evidence_schema.py tests/test_prompt_builder.py
- 结果：40 passed
- PYTHONPATH=. pytest -q tests/test_manifests.py tests/test_mat_loader.py tests/test_teacher_export.py tests/test_teacher_baseline_gate.py tests/test_output_schema.py tests/test_evidence_schema.py tests/test_prompt_builder.py
- 结果：103 passed

Task 3 已完成内容：
- StrictOutput / PredLabel formal output schema
- canonical_serialize byte-exact JSON serialization
- parse_strict JSON + Pydantic-only strict parser
- EvidenceCard schema
- TeacherSummary / DiscrepancySummary typed schema
- EvidenceAblationMask explicit enum
- build_evidence_card consistency checks
- PromptMode / ThinkingMode / ChatMessage / PromptBundle / FewShotExample
- build_prompt cross-mode stable message structure
- schema-preserving ablation prompt rendering with `Masked / Not Available` sentinel
- prompt-related env var fail-closed guard

当前 canonical 路径已包含：
- graph_data/*
- priorf_teacher/{schema,teacher_baseline_gate,export_pipeline}.py
- evidence/*
- llm/*
- eval/{head_scoring,__init__}.py

当前 diagnostic 路径已包含：
- 暂无正式 diagnostic implementation 文件
- Task 5 未对 canonical scorer 开任何 diagnostic 后门（无 oracle_threshold / probe mode / env-var switch）

当前 formal 路径已包含：
- 暂无 formal execution path 文件
- evidence/output_schema.py 仅提供 formal strict output schema；尚未接入 formal eval runner / gate manifest / launcher
- eval/head_scoring.py 提供 canonical predict_proba + threshold-free 指标层，formal eval runner 将在 Task 8 基于此构建

Task 4 验收证据：
- PYTHONPATH=. pytest -q tests/test_hidden_state_pooling.py
- 结果：21 passed
- PYTHONPATH=. pytest -q tests/
- 结果：124 passed

Task 4 已完成内容：
- pool_last_valid_token 单一公开函数，__all__ 仅含该函数
- 10 项 fail-closed 校验顺序（TypeError/ValueError）
- Path A 绝对位置索引（masked_fill + argmax），禁止 [:, -1, :] 与 cumsum/sum-1
- left-pad / right-pad / all-1s / T=1 / B=1 全覆盖
- padding-side invariance 核心防御测试（pad 位噪声不影响 pooled 结果）
- autograd 保留测试（backward + grad 非零位置验证）
- dtype/device 保真测试（fp32 / fp16 / bf16）
- 模块导出限制测试（无 mean / first / cls / max pooling）

Task 5 验收证据：
- python -m pytest tests/test_head_scoring.py -q
- 结果：25 passed（21 条用例 / 25 test nodes，含参数化展开）
- python -m pytest tests/ -q
- 结果：149 passed（Task 1–5 全量，零回归）

Task 5 已完成内容：
- eval/head_scoring.py 唯一公开入口 score_head；__all__ 严格等于 ("score_head", "ScorerReport", "ClsHead", "HeadScoringInputs", "HeadScoringSample", "CheckpointProvenance")
- Pydantic schema 全部 extra="forbid" + frozen=True：CheckpointProvenance / HeadScoringSample / HeadScoringInputs / ScorerReport
- HeadScoringInputs model_validator：每个 sample 的 evidence_card 必须与顶层 dataset_name / population_name / graph_regime 一致
- ScorerReport 字段集合 pin 死：provenance + counts + is_single_class_population + auroc / auprc / brier_score + prob 分布 7 项 + probs/labels/node_ids 三元组 + path audit (prompt_mode/thinking_mode/pooling_path/uses_inference_mode) + distributed_gather
- 禁止字段 schema 级封死：accuracy / f1 / precision / recall / threshold / optimal_threshold / fixed_threshold_* / alpha / selected_checkpoint / checkpoint_policy / oracle_threshold / faithfulness_* / fusion_*
- ClsHead Protocol（runtime_checkable）同时声明 __call__ 与 eval，isinstance 可拒裸 callable
- canonical predict_proba 链路硬编码：PromptMode.EVAL_HEAD → apply_chat_template(tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=False) → model(output_hidden_states=True, use_cache=False) → outputs.hidden_states[-1] → pool_last_valid_token → cls_head → .to(fp32) → torch.sigmoid
- 严格 B=1 逐样本 forward，attention_mask 恒为全 1，scorer 不传 padding、不碰 padding_side
- 全流程 torch.inference_mode()，probs.requires_grad is False
- 单类 population：auroc / auprc = None，is_single_class_population = True，brier_score 仍按公式计算
- 分布式合同：accelerator=None → distributed_gather="none"；accelerator 非 None → gather_for_metrics 在 probs/labels/node_ids 上各 1 次；caller 预分片，report 为 world-level
- 静态检查通过：无 hidden_states[:, -1, :] / 无 .train() 调用 / 无 trl|peft|datasets 导入 / 无 .generate( 调用 / 无 torch.distributed 属性访问与导入 / 无 os.environ|os.getenv / 无 open(|write_text|write_bytes / 无 padding_side 属性访问或赋值

Task 5 下游约束（由本 Task 选择固化，但归属 Task 6/7 承接，Task 5 本身不实现）：
- Task 7 canonical trainer 的 L_cls 头前向必须使用 PromptMode.EVAL_HEAD prompt；若用 PromptMode.TRAIN，pool 到的 hidden 会落在 JSON/assistant 末尾之后，违反 docs/020 §5 的 prompt-only hidden states 合同
- Task 6 parity test 必须用真 Qwen3 tokenizer 至少覆盖一条样本，以验证 apply_chat_template 的 4-kwarg 组合在真模型下行为与 DummyTokenizer 一致
- Task 9 gate_check 必须校验 report.n_total == expected_population_size，以兜住 caller 忘记 pre-shard 导致 n_total 被 world_size 倍放大的情况

Task 6 及以后尚未完成且不得视为已完成的事项：
- tests/test_eval_head_parity.py（Task 6 parity：same prompt builder / chat template / token spans / hidden-state indices / probabilities / predictions）
- train/train_stage2_canonical.py（Task 7 canonical joint trainer：L_gen + L_cls + L_distill 三项必须并存）
- cls head 架构与持久化
- canonical tokenization 模块（目前仅以 apply_chat_template 4-kwarg 形式 pin 在 score_head 内）
- eval/eval_head_only.py / eval/eval_gen_only.py / eval/eval_fusion.py / eval/faithfulness.py（Task 8 formal eval runner）
- validation-only threshold / alpha 选择
- teacher-prob ablation 审计
- scripts/gate_check.py + gate_manifest.json schema + launcher integration（Task 9）
