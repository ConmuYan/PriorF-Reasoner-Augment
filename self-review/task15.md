# Task 15 Fail-Closed 自审（README + operator runbook + stage launchers + smoke tests）

覆盖 `prompts/15_task_15_readme_runbook_and_launcher_closure.txt`。

范围：本轮新增/修改：

- `README.md`（重写为 operator-facing orientation，旧 prompt-bundle 段落作为 "Prompt / task bundle" 子章节保留）
- `docs/040_operator_runbook.md`（新增）
- `scripts/run_smoke.sh`（新增，diagnostic-only）
- `scripts/run_stage1.sh`（新增，gated 默认 / 带 manifest 提 formal）
- `scripts/run_stage2.sh`（新增，gated 默认 / 带 manifest 提 formal）
- `scripts/run_eval.sh`（新增，formal-only，强制 --gate-manifest）
- `tests/test_smoke_pipeline.py`（新增）
- `self-review/task15.md`（本文件）

未修改 trainer / eval 核心逻辑；未修改 `schemas.gate_manifest` / `scripts.gate_check` / `scripts.run_full_pipeline.sh`。

## 1. README / read.md 如何避免误导

README 显式改写了四处关键表述：

- "**Is not.** A drop-in LLM for raw graph input, a generic 'LLM-replaces-GNN' claim, or a system whose head metrics can be trusted without the formal eval path in this repo." — 明确避免过度宣称。
- "What counts as success" 列出五条硬性条件（gate manifest 全 pass、`eval/` 模块全部出品、teacher-prob ablation 不触发 high flag、`student_contribution_pass=True`、strict parse 是 headline）。
- "What explicitly does not count as success" 列出五条反面陈述：
  - teacher fallback 不是 student success（`optimal_alpha→0` 即只剩 teacher）；
  - low-alpha fusion 不是 success（`min_student_alpha` 是下限）；
  - normalized parse 不是 schema compliance（只 strict）；
  - oracle same-population thresholds 不是 formal；
  - 小样本 faithfulness 不是 formal。
- "Three output namespaces" 段落解释 `diagnostic` / `gated` / `formal` 的硬分流语义，禁止 silent fallback。

`read.md` 的既有章节（"What counts as success" / "What does not count as success" / "Threshold Rules" / "Fusion Rules" / "Teacher-Probability Dependency Audit" / "Fail Conditions"）与 README 保持语义一致，不在本任务内重写，避免引入冗余信息。

## 2. smoke 如何变成 fail-closed

- `scripts/run_smoke.sh`：`set -euo pipefail`；不接受 `--mode`；强制 `outputs/diagnostic/smoke/`；对 `run_full_pipeline.sh` 的 formal 路径不可达；stage 链条靠 `python -m pytest tests/test_smoke_pipeline.py -q` 的 exit code 决定，任何测试失败直接阻断脚本。
- `tests/test_smoke_pipeline.py`：
  - canonical plumbing：`build_evidence_card` + `build_prompt` 在最小 subset 上 round-trip 成功；
  - 反面：伪造一个 `teacher_model_name=""` 的 `TeacherExportRecord` 必须触发 `pydantic.ValidationError`。正面 + 反面并置，防"green smoke 什么也没验"这种静默退化。
- `run_smoke.sh` 自己不"训练失败但继续 metric 检查"。它不调用 trainer；它执行一组 Python 层的 schema + 路径断言。Trainer 的 fail-closed 属于 Task 7 `train_stage2_canonical.py` + Task 9 gate。

## 3. formal launcher 如何被 gate 严格拦截

- `scripts/run_eval.sh`：
  - 无 `--gate-manifest` ⇒ "run_eval: --gate-manifest is required for formal evaluation" + `exit 1`，绝不进入 `run_full_pipeline.sh`。
  - 有 `--gate-manifest` ⇒ `exec scripts/run_full_pipeline.sh --mode formal --gate-manifest ...`，由 `run_full_pipeline.sh` 第一行执行 `python scripts/gate_check.py --manifest-path`；任一 gate fail 就 `set -e` 短路，用户命令从不被启动。
- `scripts/run_stage1.sh` / `run_stage2.sh`：
  - 默认 `gated` namespace；用户命令写在 `outputs/gated/`。
  - 传 `--gate-manifest` 后提升到 `formal`；但 gate 失败同样短路，不退回 `gated`。`tests/test_smoke_pipeline.py::test_run_stage1_sh_rejects_failing_gate_manifest` 专门验证这条。

## 4. 是否引入 silent default / canonical-formal 混淆 / threshold-alpha leakage / formal-diagnostic 混用

**四项全无。**

- 所有 launcher 必须指定 namespace；不存在"默认 formal"或"缺 manifest 当 gated"。
- README 与 runbook 明确禁止 "teacher fallback / low-alpha fusion / normalized parse / oracle threshold / 小样本 faithfulness" 进入 formal 报告。
- 本任务未修改 threshold / alpha 选择代码；不可能注入 leakage。
- 本任务未修改任何 Pydantic report 模型；`formal` vs `diagnostic` 字段分离仍由 Task 8 各 report 承担，launcher 只负责输出目录路由。

## 5. 本轮新增的 schema / provenance / gate contract

- `scripts/run_smoke.sh` / `run_stage1.sh` / `run_stage2.sh` / `run_eval.sh`：CLI 契约。`--help` 必须给出用法；未知参数必须 `exit 1`。
- `tests/test_smoke_pipeline.py`：launcher 契约测试 + canonical 最小 subset 测试。

## 6. 本轮新增测试分别在防什么

`tests/test_smoke_pipeline.py`（9 个）：

1. `test_smoke_evidence_card_and_prompt_roundtrip_on_minimal_subset`：防 canonical 路径被后续改动悄悄破坏。
2. `test_smoke_fails_closed_on_invalid_teacher_record`：防 schema validator 被意外放宽。
3. `test_run_smoke_sh_help_succeeds`：防 `--help` 退化成 usage error（从而让 CI 使用者知道脚本还活着）。
4. `test_run_eval_sh_requires_gate_manifest`：防 eval launcher 忘记校验 manifest。
5. `test_run_eval_sh_runs_command_under_formal_namespace_when_gate_passes`：防 gate 通过却没把 namespace 设到 `formal`。
6. `test_run_stage1_sh_defaults_to_gated_namespace`：防 Stage 1 意外变 formal。
7. `test_run_stage2_sh_defaults_to_gated_namespace`：防 Stage 2 意外变 formal。
8. `test_run_stage2_sh_promotes_to_formal_when_gate_manifest_passes`：防带 manifest 却仍留在 gated。
9. `test_run_stage1_sh_rejects_failing_gate_manifest`：防失败 gate 被 wrapper silent-downgrade 成 gated 再继续执行。

## 7. 剩余三个最大风险

1. **stage launcher 目前不绑定具体 Python 入口。** `run_stage1.sh` / `run_stage2.sh` / `run_eval.sh` 把 `-- command ...` 原样透传给 `run_full_pipeline.sh`。若操作者传错模块（例如把 `eval.eval_head_only` 给到 Stage 2 launcher），只会"在 gated namespace 里跑 formal 代码"；namespace 正确但语义错位。可通过 Task 9 gate 的 `subset_head_gate_pass` 作为上游防线，但 launcher 层本身没有阻止。
2. **README 与 read.md 的内容有 partial 重叠。** 本任务只更新 README；`read.md` 的旧有文字保留。二者若未来讲述口径漂移，会让操作者困惑。缓解：README 的 "does not count as success" 五条与 `read.md` §3 保持语义同源。后续任一改动必须同步更新另一份。
3. **`run_smoke.sh` 依赖 pytest 作为执行器。** 若运行环境没装 pytest（极罕见），smoke 会以"pytest 找不到"失败，而非以"canonical 路径断言失败"失败。这依然是 fail-closed，但错误消息会偏技术向；runbook §3 应补一条 "pytest 必须可用"。

## 8. 验证证据

```bash
PYTHONPATH=. pytest -q tests/test_smoke_pipeline.py   # 9 passed
PYTHONPATH=. pytest -q tests/                         # 216 passed
bash scripts/run_smoke.sh --help                      # exit 0
bash scripts/run_stage1.sh --help                     # exit 0
bash scripts/run_stage2.sh --help                     # exit 0
bash scripts/run_eval.sh --help                       # exit 0
```
