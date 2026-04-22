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
- 当前准备进入 Task 4

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
Task 5：Unified Head Scoring（尚未开始）

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
- tests/test_mat_loader.py
- tests/test_manifests.py
- tests/test_teacher_export.py
- tests/test_teacher_baseline_gate.py
- tests/test_evidence_schema.py
- tests/test_prompt_builder.py
- tests/test_output_schema.py
- tests/test_hidden_state_pooling.py
- self-review/task2.md
- self-review/task3.md
- self-review/task4.md

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
- priorf_teacher/*
- evidence/*
- llm/*

当前 diagnostic 路径已包含：
- 暂无正式 diagnostic implementation 文件

当前 formal 路径已包含：
- 暂无 formal execution path 文件
- evidence/output_schema.py 仅提供 formal strict output schema；尚未接入 formal eval runner / gate manifest / launcher

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

Task 5 尚未完成且不得视为已完成的事项：
- eval/head_scoring.py
- tests/test_eval_head_parity.py
- canonical trainer
- fusion
- faithfulness
- formal eval
- formal gate manifest / full launcher restoration
