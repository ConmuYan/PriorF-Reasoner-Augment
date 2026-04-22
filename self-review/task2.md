# Task 2 Fail-Closed 自审

## 1. 本次改动是否引入了新的 silent default

有少量“默认值”，但没有引入会改变 canonical 行为的 silent default：

- `TeacherExportManifest.schema_version` 默认 `"teacher_export/v1"`：这是 schema 标识默认，不影响训练、评估、阈值、metric、population 或 graph regime。
- export artifact 固定文件名：
  - `teacher_export.parquet`
  - `teacher_export_manifest.json`
  这是写入通道约定，不是行为降级。
- `f1_positive_threshold: None`：不是 silent default。若 `metric_name=f1_macro` 且未显式传入，会直接 `raise`。
- baseline gate timestamp 使用 `datetime.now(timezone.utc)`：用于 provenance/report 生成时间，不参与 metric 或 gate 判定。

没有引入默认 threshold、默认 alpha、默认 population、默认 graph regime、默认 metric，也没有 fallback 到 legacy assets。

## 2. 本次改动是否让 canonical 和 diagnostic 路径再次混淆

没有。

本次实现只覆盖 Task 2 的 teacher-side contract：

- teacher export schema
- teacher baseline gate
- teacher export pipeline IO/preflight

没有实现 diagnostic 逻辑，没有 diagnostic namespace 写入，没有 diagnostic metric 分支，没有 oracle threshold，没有 sampled eval，没有 probe 逻辑。

同时 export pipeline 明确拒绝：

- `assets/teacher_exports/*` legacy 输入
- `outputs/diagnostic/...`
- `outputs/formal/...`

只允许 gated teacher export 路径：

```text
outputs/gated/teacher_exports/<dataset>/<population>/
```

## 3. 本次改动是否可能重新引入 threshold / alpha leakage

alpha：没有引入。Task 2 没有 fusion，也没有 alpha 字段或 alpha 选择逻辑。

threshold：风险被限制在 baseline gate 内：

- threshold 必须由调用方显式传入
- 没有默认 threshold
- baseline gate 只允许 `population_name == validation`
- 非 validation population 直接 raise
- env var 干预 threshold / metric / population / graph regime 会 raise
- report 中强制 `passed == (metric_value >= threshold)`

当前实现没有读取 `unused_holdout` / `diagnostic_holdout` / `final_test` 标签计算 baseline gate。

剩余风险：当前函数接收调用方传入的 `validation_ground_truth_label` / `validation_teacher_prob` 序列；它能强制 population 参数为 validation，但还不能独立证明这些数组确实来自 DataManifest validation rows。这个需要后续接入真实 teacher export / row-level source 时继续 fail-closed。

## 4. 本次改动是否可能让 formal metrics 和 diagnostic metrics 混用

目前没有。

原因：

- Task 2 不写 formal results。
- export pipeline 只允许 gated output namespace。
- baseline gate report 不是 formal metric artifact。
- 没有 diagnostic metric mode。
- 没有 oracle threshold。
- 没有 fusion / faithfulness / head scoring / formal eval。

剩余风险：`MetricName` 中包含 `f1_macro`，但这是 teacher baseline gate metric enum，不是 formal student metric。后续若接入 formal eval，必须继续保持 report namespace 与 gate namespace 分离，不能把 Task 2 baseline report 当 formal student metric。

## 5. 本次改动新增了哪些 schema / provenance / gate contract

新增 schema contracts：

- `GraphRegime`
- `PopulationName`
- `DatasetName`
- `MetricName`
- `TeacherProvenance`
- `RelationProfile`
- `NeighborSummary`
- `TeacherExportRecord`
- `TeacherExportManifest`
- `TeacherBaselineReport`

新增 provenance contract：

- teacher export manifest 必须包含完整 `TeacherProvenance`
- provenance pin 住：
  - `code_git_sha`
  - `teacher_checkpoint_path`
  - `teacher_checkpoint_sha256`
  - `data_manifest_path`
  - `data_manifest_sha256`
  - UTC tz-aware `export_timestamp_utc`
  - `random_seed`
  - `graph_regime`

新增 gate contract：

- `run_teacher_baseline_gate(...) -> TeacherBaselineReport`
- baseline gate 只允许 validation population
- metric threshold 显式传入
- `passed == metric_value >= threshold`
- baseline report 必须落盘
- CLI 在 `passed=False` 时非零退出
- CLI 在非法参数 / env 干预 / schema 校验失败时非零退出
- export pipeline 只接受 `passed=True` baseline report
- export pipeline 校验 current data manifest sha 与 report sha 一致
- export pipeline 校验 graph regime、dataset、population、node hash、split values、tuning/final-test flags 一致
- artifact 写入后重新读取并校验 row count / graph regime / population / dataset

## 6. 本次新增测试分别在防什么

### `tests/test_teacher_export.py`

防：

- 非法 `graph_regime` 被接受
- 非法 `population_name` 被接受
- schema extra 字段被接受
- `teacher_prob` / `hsd_quantile` / `node_id` 越界
- manifest provenance 缺字段仍通过
- naive datetime / 非 UTC datetime 被接受
- `label` 字段进入 TeacherExportRecord 或 artifact reader
- 单 artifact 混入多个 population
- baseline report `passed=False` 时 export pipeline 继续运行
- report 的 `data_manifest_sha256` 与当前 manifest 不一致仍运行
- report / manifest / data manifest graph regime 不一致仍运行
- record graph regime 与 export manifest 不一致仍写入
- output_dir 落入非 gated teacher export namespace
- manifest row_count 与实际 artifact 行数不一致后仍被认为成功

### `tests/test_teacher_baseline_gate.py`

防：

- metric 达标却没有生成完整 passed report
- metric 不达标时 CLI 仍零退出
- non-validation population 被用于 baseline gate
- 非法 graph regime 被接受
- 非法 metric name 被接受
- 必需参数缺失仍运行
- env var 干预 threshold / gate 行为
- checkpoint sha 缺失或格式错误
- report 中 `data_manifest_sha256` 与传入 manifest sha 不一致
- baseline report timestamp 接受 naive / 非 UTC datetime
- CLI 参数非法、env 干预、`passed=False` 时仍零退出

## 7. 还剩下的 3 个最大风险点

### 1. baseline metric 输入数组的来源仍需后续绑定到真实 validation rows

当前 gate 能强制 `population_name=validation`，也能校验 manifest 中存在 validation population，但 `validation_ground_truth_label` / `validation_teacher_prob` 是调用方显式传入的数组。Task 2 未实现 teacher 前向和 row-level export source，因此还不能证明这些数组一定来自 manifest validation node ids。后续接入 teacher inference/export 时必须绑定 node ids、population metadata 和 row provenance。

### 2. Task 2 只建立 IO/preflight，不证明 legacy teacher exports 或真实 teacher checkpoint 质量

按本轮要求，`assets/teacher_exports/*.parquet` 全部视为未验证 legacy，未被收编。当前 schema/gate 可以防止未校验 artifact 进入 gated export pipeline，但并不证明已有 legacy teacher 数据可用，也不证明 teacher checkpoint 的真实 baseline 达标。

### 3. export artifact 写入依赖 parquet serialization，对 nested Pydantic structure 的长期兼容还需后续集成验证

当前测试覆盖了 schema round-trip 和 fail-closed mismatch，但真实规模 export 可能暴露 pyarrow nested struct、datetime、enum serialization 的兼容边界。后续在真实 teacher export 接入时，需要增加 integration test，确保大规模 artifact 的 read/write schema、row_count、nested relation/neighbor fields 和 provenance manifest 始终一致。
