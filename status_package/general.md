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
- 当前准备进入 Task 3

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
Task 3：Evidence Card 与输出 Schema

已完成文件：
- graph_data/mat_loader.py
- graph_data/validators.py
- graph_data/manifests.py
- priorf_teacher/schema.py
- priorf_teacher/teacher_baseline_gate.py
- priorf_teacher/export_pipeline.py
- tests/test_mat_loader.py
- tests/test_manifests.py
- tests/test_teacher_export.py
- tests/test_teacher_baseline_gate.py
- self-review/task2.md

Task 2 验收证据：
- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. pytest -q tests/test_teacher_export.py tests/test_teacher_baseline_gate.py
- 结果：32 passed
- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. pytest -q tests/test_mat_loader.py tests/test_manifests.py tests/test_teacher_export.py tests/test_teacher_baseline_gate.py
- 结果：63 passed

Task 2 未完成且不得视为已完成的事项：
- teacher 前向推理
- pseudo inference
- Evidence Card schema
- prompt builder
- output schema
- student 训练
- fusion
- faithfulness
- head scoring
- hidden-state pooling
- canonical trainer
- formal eval
- formal gate manifest / full launcher restoration
