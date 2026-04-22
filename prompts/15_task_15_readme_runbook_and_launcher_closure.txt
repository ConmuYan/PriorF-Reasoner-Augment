现在执行 Task 15。只做这一个任务，不要越界。

前置条件：
- data manifest 已实现
- teacher baseline gate 已实现
- Evidence Card schema 已实现
- hidden_state_pooling 已实现
- head_scoring 与 parity test 已实现
- canonical trainer 已实现
- eval_head_only / eval_fusion / eval_gen_only / faithfulness 已实现
- gate_check 已实现
- formal launcher 已接 gate

目标：
更新 README、read.md、launcher 与 operator-facing 文档，使其与 canonical / diagnostic / formal 三层结构完全一致。

必须创建或修改的文件：
- README.md
- read.md
- docs/040_operator_runbook.md（如合适）
- scripts/run_smoke.sh
- scripts/run_stage1.sh
- scripts/run_stage2.sh
- scripts/run_eval.sh
- scripts/run_full_pipeline.sh
- tests/test_smoke_pipeline.py

硬性要求：
1. README 不允许再过度宣称“系统已完全正确”，除非 formal gates 已经存在并可执行
2. 所有 launcher 都必须清晰区分：
   - diagnostic
   - gated
   - formal
3. smoke 必须 fail closed
4. smoke 不允许再“训练失败但继续 metric 检查”
5. run_full_pipeline.sh 必须在无 gate manifest 或 gate 不通过时直接退出
6. README 必须明确说明：
   - teacher fallback 不是 success
   - low alpha fusion 不是 success
   - strict parse 才是正式 schema compliance
7. 不要修改 trainer 核心逻辑
8. 不要新增新的实验分支

额外要求：
- 给出 operator-facing 的最短运行路径
- 给出 canonical / diagnostic / formal 输出目录说明
- 给出最常见 fail-closed 原因和排查顺序

交付时必须输出：
1. 修改文件列表
2. README / read.md 是如何避免误导的
3. smoke 是如何变成 fail-closed 的
4. formal launcher 是如何被 gate 严格拦截的
5. 还没有做什么

完成后停下，等我审核。