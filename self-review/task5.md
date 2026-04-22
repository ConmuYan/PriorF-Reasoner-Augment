# Task 5 Fail-Closed 自审（`eval/head_scoring.py`）

范围：本轮新增的三文件 —
- `eval/head_scoring.py`
- `eval/__init__.py`
- `tests/test_head_scoring.py`

未修改任何 Task 1–4 的既有代码 / 测试 / 文档。

## 1. 是否引入了新的 silent default

**未发现 canonical 级的 silent default**，但有 2 处"弱 default"需要点名：

- `HeadScoringInputs.scorer_schema_version: Literal["head_scorer/v1"] = "head_scorer/v1"` — Literal 单值默认，调用方无法静默切换到别的 schema 版本；视作"显式 pin 而非 silent default"，合规。
- `score_head(..., accelerator=None)` — `accelerator` 有 `None` 默认。它切换的是 `distributed_gather` 审计字段（`"none"` vs `"accelerate_gather_for_metrics"`），且该字段**强制写入 report**，下游 gate 一眼可审。**不是 silent**：report 里看得见。
- `thinking_mode` **无默认**（keyword-only 无默认值），调用方必须显式传入，符合 CLAUDE.md §"explicit formal behavior"。

无其它隐性默认。`prompt_mode`、`pooling_path`、`uses_inference_mode`、`distributed_gather` 都是从 scorer 实际行为 pin 出来的审计字段，不是"允许调用方 override 的 default"。

## 2. 是否让 canonical 和 diagnostic 路径再次混淆

**未引入混淆。**

- `score_head` 是 canonical predict_proba 的唯一入口，未暴露任何 diagnostic 开关（无 `oracle_*`、无 `mode="probe"`、无 `use_teacher_label_as_input`）。
- Scorer 只接受 `PromptMode.EVAL_HEAD`，硬编码在 `build_prompt(..., mode=PromptMode.EVAL_HEAD, ...)`，调用方**无法**通过入参切换到 `TRAIN` / `GENERATION` / `EVAL_GEN`。
- `pooling_path` 固定为 `"pool_last_valid_token"`，不接受替代 pooling。
- 若将来需要 diagnostic 变体（如 frozen-backbone probe），必须新建 `eval/diagnostic_*.py` 且命名明显，不能复用此模块。

未对 `docs/030 §3 / §4` 的 frozen-backbone probe / objective ablation / oracle threshold / sampled eval / sanity baseline 开任何后门。

## 3. 是否可能重新引入 threshold / alpha leakage

**没有。**

- `ScorerReport` 字段集不含 `threshold`、`optimal_threshold`、`fixed_threshold_*`、`alpha`、`selected_checkpoint`、`checkpoint_policy`、`oracle_threshold`、`accuracy`、`f1`、`precision`、`recall` — `extra="forbid"` 从 schema 层封死（Test 2 显式注入这些 key 全部触发 `ValidationError`）。
- Scorer 只产出 threshold-free 指标：`auroc`、`auprc`、`brier_score` + prob 分布统计。没有任何路径会"先选阈值再报指标"。
- Scorer 不消费 `ground_truth_label` 做阈值选择（labels 只进入指标计算；概率来自 cls_head，与标签无耦合）。
- Fusion / alpha 相关 — 本模块**不触碰**，`ScorerReport` 里也没有 fusion 字段槽位。Task 8/13 做 fusion 时必须用独立 runner，无法从 `score_head` 得到 alpha。

**残留风险点**（非 leakage 引入，但需 Task 9 gate_check 兜底）：调用方若忘记 Q4.A 的预分片合同，会让 `n_total` 吹胖 `world_size`；已在 docstring 第 3 条明确 Task 9 必须验证 `n_total == expected population size`。

## 4. 是否可能让 formal metrics 和 diagnostic metrics 混用

**没有。**

- `ScorerReport` 字段已 pin 死，没有 `accuracy`、`f1` 这些 threshold-依赖指标 → 无法在同一 report 里混写 formal + diagnostic。
- `checkpoint_provenance`、`dataset_name`、`population_name`、`graph_regime`、`scorer_schema_version` 强制回显，report 与 population 不可解耦；下游 formal runner 若要写入 `outputs/formal/`，必须对得上 provenance。
- Scorer 不写文件、不确定 namespace；不会把 diagnostic 跑的结果错写到 formal 目录（Test 17 断言无文件写入）。
- `is_single_class_population` 是独立 bool 字段，单类边界行为（`auroc=None` / `auprc=None`）**不会静默回落到 0.5 / 0.0 / 1.0**（Test 8 双向断言）。这一处过去是 SAC §"single-class collapse masking" 的常见偷懒点，这里封死。

**残留**：Task 8 formal eval runner 自己写 report 时需继续守住这条线；scorer 层已无法发生混用。

## 5. 新增了哪些 schema / provenance / gate contract

**Schema（Pydantic，全部 `extra="forbid"` + `frozen=True`）：**
- `CheckpointProvenance(path, step, content_hash)` — 检查点来源三元组。
- `HeadScoringSample(evidence_card, ground_truth_label∈{0,1}, node_id≥0)`。
- `HeadScoringInputs(samples, dataset_name, population_name, graph_regime, checkpoint_provenance, scorer_schema_version="head_scorer/v1")` + model_validator：每个 sample 的 `evidence_card.{dataset_name|population_name|graph_regime}` 必须与顶层一致。
- `ScorerReport` — 固定字段集合（见 Test 1 的 `expected_fields`）；内置 `n_positive + n_negative == n_total`、`len(probs)==len(labels)==len(node_ids)==n_total`、probs 逐元素 finite ∈ [0,1] 的 model_validator。
- `ClsHead(Protocol, runtime_checkable)` — `__call__` + `eval`；运行时 `isinstance` 可拒 bare callable。

**Provenance：** 每个 report 强制携带 `dataset_name / population_name / graph_regime / checkpoint_provenance / scorer_schema_version`，不可省略。

**Path audit：** `prompt_mode="eval_head"`, `thinking_mode`, `pooling_path="pool_last_valid_token"`, `uses_inference_mode=True`。

**Distributed audit：** `distributed_gather ∈ {"none", "accelerate_gather_for_metrics"}`，无法置空、无法伪造。

**本 Task 未新增 gate contract**（Task 9 的 `gate_manifest.json` 未触碰），但为 Task 9 埋好了挂钩：`scorer_schema_version`、`checkpoint_provenance`、world-level `n_total` 是 gate 层可直接消费的 invariants。

## 6. 新增测试分别在防什么

| 测试 | 在防什么 |
|---|---|
| **1 schema 字段存在 + extra forbid** | 防 report 结构漂移；防未来偷偷塞 `threshold` / `accuracy` 等字段 |
| **2 forbidden names absent** | 防复活 threshold/alpha/fusion/faithfulness/oracle 字段到 formal report |
| **3 build_prompt spy** | 防某天有人把 `mode` 改成 `TRAIN` 或 `EVAL_GEN`；防 `thinking_mode` 隐式默认化 |
| **4 pool_last_valid_token spy + 源码扫描** | 防平行出现第二个 pooling 实现；防 `hidden_states[:, -1, :]` 回潮 |
| **5 model.generate 禁用** | 防 scorer 偷走 generation path；防 head eval 被误实现成 generation-based predict_proba |
| **6 `.eval()` 被调 + Protocol isinstance** | 防 dropout / batchnorm 未真正关闭；防 ClsHead 合同退化成"任意 callable" |
| **7 inference_mode 生效 + no autograd** | 防 SAC §3 坑（"head.train() 未真正关 dropout"）；防 scorer 泄漏 autograd 到下游 |
| **8 单类 population** | 防 auroc/auprc 静默返回 0.5 / 0.0；防"全零 label 混入 formal report 还不知道" |
| **9 空 population** | 防 0 样本时 scorer 静默返回 NaN / 空 report |
| **10 permutation equivariance** | 防 scorer 对样本顺序产生隐性依赖（batched attention 的位置噪声） |
| **11 determinism** | 防"两次 run 结果飘移"；防随机路径渗入 canonical scorer |
| **12 NaN/Inf logit** | 落地 Contract §16 "eval NaN" 的 fail-closed |
| **13 accelerator=None + 无 torch.distributed** | 防 scorer 在非分布式场景下误调用 all_gather / reduce |
| **14 accelerator + world-level consistency** | 防 caller 忘记 pre-shard；防两 rank 产生不一致 report；钉死 `gather_for_metrics` 调用次数 == 3 |
| **15 provenance echo** | 防 report 与 population 解耦、与 checkpoint 解耦 |
| **16 inputs model_validator** | 防 evidence_card 与顶层 population / dataset / regime 不一致造成跨人群样本混入 |
| **17 无文件写入** | 防 scorer 偷偷写 `outputs/`、写 cache、写 tmp artifact |
| **18 无 env 依赖** | 防 env var 被当成隐性 switch（CLAUDE.md §"no hidden behavior"）|
| **19 `__all__` 严格 ==** | 防后续添加 diagnostic helper 到公开 surface |
| **20 logit fp32 hard cast** | 防 bf16 / fp16 logit 直接喂 sigmoid → sklearn，产生静默精度漂移 |
| **21 B=1 + attention_mask 全 1 + tokenizer 无 padding kwarg** | 钉死 Q3.A 的单样本 forward；防 padding / 左右 padding 语义漂移让 parity test 失效 |

## 7. 剩下的 3 个最大风险点

### R1. `L_cls` prompt 路径仅在 plan §L 记录，代码层无强制

Task 7 canonical trainer 在计算 `L_cls` 时若误用 `PromptMode.TRAIN`（assistant slot 含 JSON），pool 到的 hidden 就落在 generation tokens 之后，违反 `docs/020 §5` + 我的 Q1.A 选择，但 Task 5 本身**无法**在代码里拦截 Task 7 的这个错误。现有 Task 6 parity test（未实现）是**唯一**能在代码级兜住这个错误的地方。若 Task 6 的 parity 实现时没有 sample-wise 比对 prompt 构造路径，这条合同就只剩文档保护。

**缓解要求**：Task 6 的 parity 必须包含 "trainer-time cls forward uses `build_prompt(mode=EVAL_HEAD)`" 的断言，而不是只比 logits 数值。

### R2. 调用方忘记 pre-shard 时，`n_total` 会被 `world_size` 倍放大，scorer 不自检

我在 docstring 明确声明 scorer 不 re-validate `n_total == expected population size`，把兜底责任交给 Task 9 `gate_check`。但 Task 9 若实现时漏写这条校验，整个分布式 eval 可能在单机开发时全绿、到 4-GPU formal run 时才暴露 `n_total = 4 × N`，**且 AUROC / AUPRC 依然计算成功**（仅 Brier 会受重复样本的再采样偏移，AUROC 对复制不变）—— 即静默失败。

**缓解要求**：Task 9 gate manifest 必须把 `expected_population_size` 纳入强制字段，且 `gate_check` 必须断言 `report.n_total == expected_population_size`。

### R3. `tokenizer.apply_chat_template` 的输出 schema 在真实 Qwen3 tokenizer 上未被实测

Task 5 内部仅用 `DummyTokenizer` 覆盖，真 tokenizer 合同是**假设**而非验证：

- 若 Qwen3 tokenizer 需要额外 kwarg（例如 `enable_thinking=True/False`）才在 `apply_chat_template` 层吃 `thinking_mode`，而 scorer 只传了 4 个 kwargs，`thinking_mode` 的语义可能**根本没进 tokenizer**（目前仅反映在 Evidence Card 的 system prompt 文本里）。
- 某些 HF tokenizer 在 `return_dict=True` 与 `return_tensors="pt"` 组合下的行为随 `transformers` 版本漂移。

**缓解要求**：Task 6 parity test 必须用**真 Qwen3 tokenizer**跑至少一条 sample，否则本模块的 tokenization 合同在 fake 测试外未被验证。
