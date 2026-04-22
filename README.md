# PriorF-Reasoner Codex / OMX Prompt Bundle

本压缩包包含两类内容：

1. `codex_operation_guide.md`
   - 面向配置了 Codex / OMX（oh-my-codex）的实际操作教程
   - 包括建议工作模式、任务切片方式、审核节奏、建议命令

2. `prompts/`
   - 可直接复制给 Codex / Harness Agent 的 prompts
   - 包括总控启动 prompt、Task 1~15、审查 prompt、收尾 prompt

## 建议使用方式

1. 先把仓库内 source-of-truth 文档落盘：
   - `docs/010_project_contract.md`
   - `docs/020_top_level_design.md`
   - `docs/030_harness_agent_execution_guide.md`
   - `docs/050_fail_closed_guardrails.md`
   - `read.md`
   - `CLAUDE.md`

2. 先用只读模式让 Codex 审阅文档，再按 Task 逐个执行。

3. 每个 Task 都要：
   - 先贴 `prompts/00_master_control_prompt.txt`
   - 再贴对应 Task prompt
   - 写完后贴 `prompts/98_self_review_prompt.txt`
   - 审核通过后贴 `prompts/99_wrapup_prompt.txt`

## 文件目录

- `codex_operation_guide.md`
- `prompts/00_master_control_prompt.txt`
- `prompts/01_task_01_data_foundations.txt`
- `prompts/02_task_02_teacher_contract_and_baseline_gate.txt`
- `prompts/03_task_03_evidence_card_and_output_schema.txt`
- `prompts/04_task_04_hidden_state_pooling.txt`
- `prompts/05_task_05_unified_head_scoring.txt`
- `prompts/06_task_06_eval_head_parity_test.txt`
- `prompts/07_task_07_gate_manifest_and_gate_check.txt`
- `prompts/08_task_08_formal_launcher_gate_integration.txt`
- `prompts/09_task_09_canonical_joint_trainer.txt`
- `prompts/10_task_10_formal_head_only_eval.txt`
- `prompts/11_task_11_teacher_prob_ablation_audit.txt`
- `prompts/12_task_12_formal_fusion_eval.txt`
- `prompts/13_task_13_formal_generation_eval.txt`
- `prompts/14_task_14_formal_faithfulness_eval.txt`
- `prompts/15_task_15_readme_runbook_and_launcher_closure.txt`
- `prompts/98_self_review_prompt.txt`
- `prompts/99_wrapup_prompt.txt`
