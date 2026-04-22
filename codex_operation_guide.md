# Codex / OMX（oh-my-codex）操作教程

这份教程用于指导你如何把 PriorF-Reasoner 的 fail-closed 合同稳定地落到仓库里。

## 一、先定工作方式

不要一开始就让 agent “把整个仓库做完”。

正确方式是：

1. 先锁文档
2. 再锁目录
3. 再锁 contracts
4. 再锁 tests
5. 最后才写 trainer / eval

也就是：文档先行，测试先行，主路径和诊断路径从第一天就拆开。

## 二、先把 source-of-truth 文档写进仓库

先在项目根目录里创建这些文件，并把定稿内容写进去：

- `docs/010_project_contract.md`
- `docs/020_top_level_design.md`
- `docs/030_harness_agent_execution_guide.md`
- `docs/050_fail_closed_guardrails.md`
- `read.md`
- `CLAUDE.md`

最推荐的方式是你手动写入这 6 个文件。

## 三、第一次启动 Codex：只做“文档锁定”，不写业务代码

在项目根目录执行：

```bash
codex --suggest --sandbox read-only
```

进入后，先让它只读审文档，不要写代码。建议先贴总控启动 prompt，再贴“只读审查型需求”。

## 四、如果你用了 OMX / oh-my-codex

先检查运行环境和 hooks：

```bash
omx doctor
omx hooks install --presets=workspace-context,memory,safety,review
omx hooks status
```

如果你的变体命令不同，就先运行：

```bash
omx --help
omx hooks --help
```

目标是确保：
- workspace context 自动注入
- memory 会保留上下文
- safety 会拦危险操作
- review 会把输出带回审查队列

## 五、正式开始实现时，永远按“一个任务一个提交”推进

你应该告诉 agent 的方式不是：

> 帮我把 PriorF-Reasoner 完整实现出来。

而是：

> 只实现 Task N；只改动指定文件；做完后停下；汇报你新增了什么、哪些测试通过了、哪些没有做。

## 六、建议的 Codex 模式分配

- Suggest + read-only sandbox：读文档、审计划、做 contract review
- Auto Edit + workspace-write sandbox：创建文件、补代码、写测试
- 不要一开始就用 Full Auto

## 七、推荐的任务切片顺序

1. 数据基础设施
2. Teacher 合同与 baseline gate
3. Evidence Card 与输出 schema
4. Hidden state pooling
5. 统一 head scoring
6. Parity test
7. GateManifest 与 gate_check
8. Formal launcher 接 gate
9. Canonical joint trainer
10. Head-only 正式评估
11. Teacher-probability ablation 正式审计
12. Fusion 正式评估
13. Generation 正式评估
14. Faithfulness 正式评估
15. README / runbook / launcher 收口

## 八、推荐执行节奏

每一轮：

1. 贴总控启动 prompt
2. 贴本轮 Task prompt
3. 看它先输出 A/B/C/D
4. 你确认后让它写
5. 写完后贴自审 prompt
6. 你审核
7. 贴收尾 prompt
8. commit
9. 再进下一 task

## 九、如果你想用 OMX 多 agent，最稳的方式

- Agent A：Architect
  - 只读 docs
  - 生成 task graph
  - 不写代码

- Agent B：Executor
  - 只做当前单一 task
  - 限定改动文件
  - 写代码和测试
  - 做完即停

- Agent C：Reviewer
  - 审 diff
  - 审 contracts
  - 审测试覆盖
  - 审 fail-closed 逻辑

## 十、最短可执行启动清单

```bash
cd priorf_reasoner_slm

# 先让 Codex 只读审文档
codex --suggest --sandbox read-only

# 然后用可写模式做 Task 1
codex --auto-edit --sandbox workspace-write

# 如果你用了 OMX
omx doctor
omx hooks install --presets=workspace-context,memory,safety,review
omx hooks status

# 每完成一个任务跑对应测试
pytest tests/test_mat_loader.py
pytest tests/test_teacher_baseline_gate.py
pytest tests/test_evidence_schema.py
pytest tests/test_hidden_state_pooling.py
pytest tests/test_eval_head_parity.py
```

## 十一、最重要的三个习惯

1. 每做完一个 task 就 commit
2. 每个 task 都要求“停下等审核”
3. 看到它想顺手改别的模块，就立刻打断

## 十二、给 agent 的总控提示词使用法

建议每个新任务开始时，都先贴：
- `prompts/00_master_control_prompt.txt`

然后再贴对应 Task prompt。

任务完成后，贴：
- `prompts/98_self_review_prompt.txt`

审核通过后，贴：
- `prompts/99_wrapup_prompt.txt`
