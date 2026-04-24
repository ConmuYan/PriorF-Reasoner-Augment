# Task 9 Fail-Closed 自审（gate manifest + gate_check + formal launcher 集成）

覆盖 `prompts/07_task_07_gate_manifest_and_gate_check.txt` 和 `prompts/08_task_08_formal_launcher_gate_integration.txt`，对应 `docs/030_harness_agent_execution_guide.md` 的 Task 9「gate system」。

范围：本轮新增（来源：OMX team `worker-1` 的 `e7d4987` cherry-pick 合入 main）：

- `schemas/__init__.py`
- `schemas/gate_manifest.py`
- `scripts/gate_check.py`
- `scripts/run_full_pipeline.sh`
- `tests/test_gate_check.py`
- `tests/test_formal_launcher_gate.py`
- `self-review/task9.md`（本文件）

未修改 Task 1–8 的 schema / trainer / formal eval 合同；未实现 Task 15 的 stage launchers（Task 15 在 `self-review/task15.md` 中单独结账）。

## 1. 是否引入新的 silent default

**没有。**

- `GateManifest` 字段集是 pinned 的：`schema_version: Literal["gate_manifest/v1"]`、`graph_regime`、`commit` (hex40)、`generated_at` (UTC tz-aware)、`config_fingerprint` (min_length=1)、`data_manifest_hash` (hex64)、加上九个 `*_pass: bool`。缺任何字段或写入多余字段都会被 `extra="forbid"` 拦下；`generated_at` 非 UTC 也会被 `_generated_at_must_be_utc` validator 拒绝。
- `scripts/gate_check.py` 没有默认 manifest 路径；`--manifest-path` 是 required 参数。
- `scripts/run_full_pipeline.sh` 没有默认 `--mode`；未传入会 `exit 1`。

## 2. 是否让 canonical / diagnostic / formal 路径混淆

**没有。**

- `run_full_pipeline.sh` 的 `formal` 分支必须传 `--gate-manifest`，且会在任何用户命令之前执行 `python scripts/gate_check.py --manifest-path "$gate_manifest"`。gate 失败 `set -euo pipefail` 直接让脚本退出非零，用户命令不会被 `exec`。
- `formal` / `gated` / `diagnostic` 三个 namespace 是硬分流：`namespace_dir="$output_root/formal|gated|diagnostic"`。不会在 formal 失败时自动退回 gated。
- `PRIORF_OUTPUT_NAMESPACE` / `PRIORF_OUTPUT_DIR` 显式 export，下游代码只能读取已经确定好的 namespace；不存在"悄悄换 namespace"的入口。

## 3. 是否重新引入 threshold / alpha leakage

**没有。**

- 本任务不触碰 threshold / alpha 选择逻辑；`eval.calibration` 与 `eval.eval_fusion` 仍是唯一选择点。
- `gate_manifest.validation_eval_parity_pass` 只标记 validation 与 formal head 的 parity 是否在上游通过；本层只校验 bool 值为 `True`，不重新计算任何 metric。

## 4. 是否让 formal metrics 和 diagnostic metrics 混用

**没有。**

- `GateManifest` 中的九个 `*_pass` 是独立 gate，任意一个为非 `True` 都会让 `gate_check.py` 抛 `ValueError`；不允许"某项 diagnostic 通过，formal 也自动通过"的 cross-gate 替代。
- `teacher_prob_ablation_pass` 是显式独立字段；无此字段、或字段为 `False` 时 formal launcher 拒绝启动。
- `smoke_pipeline_pass` 是独立字段；一个绿色 smoke 本身不能替代任何其他 gate。

## 5. 本轮新增的 schema / provenance / gate contract

- `schemas.gate_manifest.GateManifest`：强 typed、`extra="forbid"`、`frozen=True` 的 Pydantic 模型。
- `schemas.gate_manifest.load_gate_manifest(path)`：唯一合法的反序列化入口；缺文件抛 `FileNotFoundError`，JSON 解析失败或字段违约都会抛。
- `scripts.gate_check.gate_check(*, manifest_path)`：对 `_REQUIRED_TRUE_FIELDS` 中的九项做严格 True 校验，收集失败字段并在异常消息中列出。
- `scripts/run_full_pipeline.sh` 是唯一带执行态的"formal 前门"；所有后续 formal eval launcher（`run_eval.sh` 等，Task 15 引入）都 exec 进入此脚本。

## 6. 本轮新增测试分别在防什么

- `tests/test_gate_check.py`（多个）：
  - 防缺字段通过；防多余字段通过；防字符串长度 / 正则不匹配通过；
  - 防 `generated_at` 为 naive datetime 或非 UTC tz 通过；
  - 防任意单一 `*_pass=False` 漏网；
  - 防不存在的 manifest 路径被 silently 吞掉。
- `tests/test_formal_launcher_gate.py`（4 个）：
  - `test_formal_launcher_requires_manifest`：`--mode formal` 缺 `--gate-manifest` 必须非零退出；
  - `test_formal_launcher_blocks_failed_gate`：`*_pass=False` 时用户命令绝不被执行，`outputs/formal/` 不得存在；
  - `test_formal_launcher_runs_only_after_gate_pass_and_routes_to_formal_namespace`：gate 通过后 `PRIORF_OUTPUT_NAMESPACE=formal`、`PRIORF_OUTPUT_DIR=.../outputs/formal`；
  - `test_non_formal_modes_do_not_enter_formal_namespace`：`gated` / `diagnostic` 绝不创建 `outputs/formal/`。

## 7. 剩余三个最大风险

1. **manifest 生成端仍靠人工。** `GateManifest` 的字段值（尤其 `*_pass` 九项）当前没有"自动生成器"，需要上游任务各自跑完后填 `True`。风险是操作者在未跑某个 gate 的情况下手填 `True`；缓解策略应是 Task 10+ 阶段加 `generate_gate_manifest.py` 从各 eval 报告自动抽取。
2. **没有 override 机制。** 当前 launcher 没有 `--override-gate-with-provenance` 的逃生通道。prompt 原文允许 override 但要写入 provenance；目前完全禁止 override。若将来确实需要 emergency path，需在此模块新增显式 `OverrideManifest` 而不是放松 `gate_check.py`。
3. **`run_full_pipeline.sh` 与 Task 15 stage launcher 的 exec 链。** Stage1 / Stage2 / eval wrapper（Task 15）依赖 `exec scripts/run_full_pipeline.sh ...` 接力；若未来该脚本的参数签名变更而上游 wrapper 未同步，会出现"看起来 formal、实则 gated"的悄悄退化。缓解：Task 15 的 `tests/test_smoke_pipeline.py` 已经显式断言 wrapper 在 gate 缺失/失败时退非零。

## 8. 验证证据

```bash
PYTHONPATH=. pytest -q tests/test_gate_check.py tests/test_formal_launcher_gate.py  # 17 passed
PYTHONPATH=. pytest -q tests/                                                      # 216 passed（Task 11 + Task 15 合入之后）
python -m ruff check schemas/gate_manifest.py scripts/gate_check.py \
    tests/test_gate_check.py tests/test_formal_launcher_gate.py                     # All checks passed
python -m py_compile schemas/__init__.py schemas/gate_manifest.py scripts/gate_check.py  # PASS
```

原始 commit `e7d4987`（cherry-pick 合入 main 作为 `20316ef`）。
