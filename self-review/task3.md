# Task 3 Fail-Closed 自审

## 1. 本次改动是否引入了新的 silent default

有少量默认值，但没有引入会静默改变 canonical objective / diagnostic 行为的默认：

- `EvidenceCard.schema_version` 默认 `"evidence_card/v1"`：这是 schema 标识默认，不影响训练、评估、阈值、alpha、metric、population 或 graph regime。
- `build_evidence_card(..., ablation_mask=frozenset())`：默认表示不做 ablation，符合 Task 3 的函数签名要求。
- `build_prompt(..., few_shot_examples=())`：默认空 few-shot，符合 Task 3 的函数签名要求。
- `build_evidence_card()` 内部生成固定 `TaskInstruction.text`：这是新的静态 instruction 文本，不切换路径、不改变目标、不读取环境变量；但后续若要变化，必须显式改 API 或 schema，不能从外部隐式覆盖。

没有引入默认 threshold、默认 alpha、默认 diagnostic mode、默认 graph regime、默认 population、alias parser、retry parser、env var fallback 或 legacy asset fallback。

## 2. 本次改动是否让 canonical 和 diagnostic 路径再次混淆

没有。

本次只覆盖 Task 3 的类型化 contract：

- strict formal output schema
- Evidence Card schema
- prompt builder
- schema-preserving ablation API

没有实现：

- diagnostic parser
- normalized / forgiving / alias parser
- diagnostic eval
- diagnostic output namespace
- oracle threshold
- ablation metric
- trainer
- head scoring
- hidden-state pooling
- fusion
- faithfulness
- formal eval
- gate check
- launcher

因此没有把 canonical 和 diagnostic 路径混在一起。

## 3. 本次改动是否可能重新引入 threshold / alpha leakage

没有。

本次没有新增或修改任何：

- threshold selection
- alpha selection
- metric computation
- validation/test report
- fusion logic
- head scoring
- eval runner

`score_target_for_sft` 只在 `PromptMode.TRAIN` 下用于构造 `StrictOutput.score`，非 train 模式显式禁止传入 `ground_truth_label_for_sft` 和 `score_target_for_sft`。它不读取 eval/test-like labels，也不会选择 threshold 或 alpha。

## 4. 本次改动是否可能让 formal metrics 和 diagnostic metrics 混用

没有。

原因：

- 没有新增 metrics。
- 没有新增 report schema。
- 没有写入 `outputs/formal` / `outputs/diagnostic` / `outputs/gated`。
- `StrictOutput` 只是 formal generation output schema，不产生 parse-rate metric。
- `parse_strict()` 只做 `json.loads + StrictOutput.model_validate`，没有 alias-normalized parse rate 或 forgiving parse path。

剩余注意点：后续 formal eval 如果基于 `parse_strict()` 统计 strict schema compliance，必须继续避免同时报告 alias-normalized / forgiving parse rate 为 formal metric。

## 5. 本次改动新增了哪些 schema / provenance / gate contract

新增 schema contracts：

- `PredLabel`
  - 固定枚举：`fraud` / `benign`
  - 禁止大小写宽松匹配、alias、未知 label

- `StrictOutput`
  - 字段：`rationale`, `evidence`, `pattern_hint`, `label`, `score`
  - 严格长度、类型、范围、有限数校验
  - canonical serialization 顺序固定为 `rationale -> evidence -> pattern_hint -> label -> score`

- `TaskInstruction`
  - `text`
  - `schema_hint_order`
  - 强制等于 canonical output order

- `TeacherSummary`
  - 显式字段：`teacher_prob`, `teacher_logit`, `hsd`, `hsd_quantile`, `asda_switch`, `mlp_logit`, `gnn_logit`, `branch_gap`, `high_hsd_flag`
  - masked 字段必须为 `None`
  - unmasked 字段必须非 `None`

- `DiscrepancySummary`
  - 显式字段：`branch_gap_abs`, `teacher_mlp_agreement`, `teacher_gnn_agreement`, `discrepancy_severity`, `route_hint`
  - 禁止 `dict[str, Any]` 式自由扩展

- `EvidenceAblationMask`
  - 显式枚举 teacher / discrepancy / relation / neighbor 可 mask 字段
  - 禁止自由字符串 mask

- `EvidenceCard`
  - 复用 Task 2 类型：`DatasetName`, `PopulationName`, `GraphRegime`, `RelationProfile`, `NeighborSummary`
  - 禁止 extra 字段
  - 禁止 `label` / `ground_truth_label`
  - schema version 固定为 `evidence_card/v1`

- `PromptMode`
  - `train`, `validation`, `eval_head`, `eval_gen`, `generation`

- `ThinkingMode`
  - `thinking`, `non_thinking`

- `ChatMessage`
  - role 限定为 `system` / `user` / `assistant`

- `PromptBundle`
  - train 模式必须有 `sft_target_label`
  - 非 train 模式必须没有 `sft_target_label`
  - 所有 mode 必须包含 assistant role

- `FewShotExample`
  - `source_population` 必须是 `train`

新增 provenance / consistency contract：

- `build_evidence_card()` 校验：
  - `teacher_record.dataset_name` 与 `data_manifest.dataset_name` 一致
  - `teacher_record.graph_regime` 与 `data_manifest.graph_regime` 一致
  - `teacher_record.node_id` 在 manifest 范围 / node ids 内有效
  - 不读取或写入 `ground_truth_label`

新增 gate-like fail-closed contract：

- `parse_strict()` 只允许 JSON + Pydantic validation。
- prompt builder 检测到以下 env var 前缀即 raise：
  - `PRIORF_PROMPT_*`
  - `EVIDENCE_*`
  - `PROMPT_BUILDER_*`
- `eval_head` / `eval_gen` / `generation` 模式禁止 few-shot。
- train / non-train SFT label 参数严格互斥。

没有新增 formal gate manifest contract。

## 6. 本次新增测试分别在防什么

### `tests/test_output_schema.py`

防：

- label alias / 大小写宽松解析
- score 越界、NaN、inf、字符串 coercion
- 空 rationale / 空 pattern_hint / 空 evidence / 空 evidence item
- canonical JSON 顺序漂移
- `parse_strict()` 退化成 alias parser / forgiving parser / first-block parser
- extra 字段进入 formal output
- `parse_strict(canonical_serialize(x)) != x`

### `tests/test_evidence_schema.py`

防：

- EvidenceCard 缺必需字段仍通过
- extra 字段进入 EvidenceCard
- 非法 dataset / population / graph regime 通过
- `label` / `ground_truth_label` 泄漏进入 EvidenceCard
- masked 字段非 None
- unmasked 字段为 None
- 非枚举 ablation mask 被接受
- teacher/data manifest dataset mismatch 被忽略
- graph regime mismatch 被忽略
- node id 不存在仍构造 EvidenceCard
- TeacherSummary / DiscrepancySummary 变成自由 dict 或宽松值域

### `tests/test_prompt_builder.py`

防：

- 未知 `PromptMode` / `ThinkingMode` 通过
- thinking mode 未显式传入
- train 模式没有 SFT target 仍通过
- non-train 模式携带 SFT label/score
- train assistant content 不是 canonical serialization
- non-train assistant role 缺失或 content 非空
- 跨 mode message 结构、role 顺序、system/user/few-shot 文本漂移
- few-shot 使用非 train population
- `eval_head` / `eval_gen` / `generation` 使用 few-shot
- ablation mask 用 `0.0` / `-1` / `999` 等数值哨兵
- prompt builder 受 env var 静默影响

## 7. 还剩下的 3 个最大风险点

### 1. `DataManifest` 当前没有显式 `node_ids` 字段，node membership 校验只能部分 fail-closed

Task 3 要求 `teacher_record.node_id` 必须存在于 `data_manifest.node_ids`。但现有 Task 1 `DataManifest` 没有 `node_ids` 字段，只有 `num_nodes` 和 population hash。

当前实现若 manifest 暴露 `node_ids` 会做精确 membership；否则 fallback 到 `0 <= node_id < num_nodes`。这比完全不校验安全，但不是完整 node-id membership contract。

### 2. `RelationProfile` / `NeighborSummary` 的 schema-preserving ablation 只能在 prompt 渲染层表达

Task 2 的 `RelationProfile` / `NeighborSummary` 是 frozen 且字段非 Optional，Task 3 又要求复用不得重定义。

因此 relation / neighbor mask 不能在 EvidenceCard 数据模型里把字段置为 `None`，只能在 prompt builder 渲染为 `Masked / Not Available`。这满足 schema-preserving prompt ablation，但如果后续代码误以为 relation/neighbor mask 会改变底层 Pydantic 对象值，会有解释偏差风险。

### 3. `build_prompt()` 中 SFT target 的 rationale/evidence/pattern_hint 是固定模板文本

Task 3 只给了 `ground_truth_label_for_sft` 和 `score_target_for_sft`，没有提供 rationale/evidence/pattern_hint target 参数。

当前实现用固定模板补齐 `StrictOutput`。这保持 schema 合法、没有 label 泄漏扩展，但后续 structured generation training 如果直接依赖这些模板，可能导致 generation target 过于模板化。

后续如果要接真实 SFT targets，必须显式扩展 API，而不能静默从其他字段推断。
