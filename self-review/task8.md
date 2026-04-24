# Task 8 Fail-Closed 自审（formal eval 套件：head-only / gen-only / fusion / faithfulness）

覆盖 `prompts/10_task_10_formal_head_only_eval.txt`、`prompts/12_task_12_formal_fusion_eval.txt`、`prompts/13_task_13_formal_generation_eval.txt`、`prompts/14_task_14_formal_faithfulness_eval.txt` 四个 prompt，对应 `docs/030_harness_agent_execution_guide.md` 的 Task 8「formal eval suite」。

范围：本轮新增/修改（由 OMX team 四条通道合并而成）：

- `eval/eval_head_only.py`（prompt 10）
- `eval/calibration.py`（prompt 10 附属 threshold 支持层）
- `eval/eval_fusion.py`（prompt 12）
- `llm/fusion.py`（prompt 12 probability 合成层）
- `eval/eval_gen_only.py`（prompt 13）
- `llm/parsing.py`（prompt 13 strict + normalized parse）
- `eval/faithfulness.py`（prompt 14）
- `tests/test_eval_head_only.py`、`tests/test_eval_fusion.py`、`tests/test_eval_gen_only.py`、`tests/test_faithfulness.py`、`tests/test_strict_output_schema.py`
- `self-review/task8.md`（本文件）

未修改 Task 1–7 的 `graph_data` / `priorf_teacher` schema / `evidence.evidence_schema` / `llm.hidden_state_pooling` / `eval.head_scoring` / `train.train_stage2_canonical`。未产出任何 `outputs/formal/` 路径下的 artifact。

## 1. 是否引入新的 silent default

**没有。**

- `eval_head_only`：threshold 唯一来源是 `eval.calibration.select_validation_threshold(...)`，作用域在 `population=validation` 的 `ScorerReport`；report population 只允许 frozen 复用，没有 `0.5` / accuracy / F1 默认。
- `eval_gen_only`：strict parse 失败不回退 normalized；normalized 只作 diagnostic 分母，不写入 formal report。
- `eval_fusion`：alpha 只来自 `FusionEvalConfig.alpha_candidates`，没有 `alpha=0.5` 等硬编码回退；`_GUARDRAIL_EPS=1e-12` 是 numerical tie-break，不是容差默认。
- `faithfulness`：每个 variant 走同一 `score_head` 调用，不内建 faithfulness 专属 threshold 或 alpha。

## 2. 是否让 canonical / diagnostic / formal 路径混淆

**没有。**

- `eval.head_scoring.score_head` 是 Task 5 canonical 层。`eval_head_only` / `eval_fusion` / `faithfulness` 三个 formal runner 只通过 `score_head` 或其产出的 `ScorerReport` 拿概率，不自建 `predict_proba`。
- `eval_gen_only` 不走 head scoring；走 `llm/parsing.py` 的 `parse_strict_output` + 可选 `parse_normalized_output`。formal 输出只依赖 strict path，normalized 只进 `DiagnosticParseAudit`。
- formal runners 不写 `outputs/` 任何路径；namespace 分流由 Task 15 launcher + `gate_check.py` 前置决定。

## 3. 是否重新引入 threshold / alpha leakage

**没有。**

- `eval_head_only`：threshold 选择**只**接受 `population=validation` 的 `ScorerReport`；schema 级 `model_validator` 拒绝 `final_test` 或混合 population 作为 selection 输入。oracle same-population threshold 指标被显式命名 `oracle_*` 并与 formal headline 指标分离。
- `eval_fusion`：`run_formal_fusion_eval(validation_inputs=..., report_inputs=...)` 两个参数 schema 级互斥（validation vs final_test / report-like），alpha candidate 枚举只在 validation 上排序，test-like 只做 frozen-forward。
- `faithfulness`：所有 variant 共用 frozen alpha / threshold，其来源字段直接引用自 formal head report，不在 faithfulness 层重新 tune。

## 4. 是否让 formal metrics 和 diagnostic metrics 混用

**没有。**

- `HeadOnlyReport` / `FusionEvalReport` / `GenOnlyReport` / `FaithfulnessReport` 均为 `extra="forbid"` + `frozen=True` 的 Pydantic 模型。
- `eval_head_only` 将 `oracle_*` 字段放入 `FormalHeadOnlyDiagnostics`，与 `FormalHeadOnlyHeadlineMetrics` 完全分离；`auroc` / `auprc` / `f1_at_val_threshold` / `precision_at_val_threshold` / `recall_at_val_threshold` / `specificity_at_val_threshold` / `prediction_std` 才是 headline。
- `eval_gen_only` 的 headline 只能是 `strict_schema_parse_rate`；`normalized_parse_rate` 与 alias 统计挂在 `DiagnosticParseAudit` 下。
- `faithfulness` 小样本情况下自动标记为 diagnostic（`full_inputs` 必须达到 `config.min_formal_sample_size`，否则 fail closed），不自动退化为 sampled formal。

## 5. 本轮新增的 schema / provenance / gate contract

- `FormalHeadOnlyReport`（`eval/eval_head_only.py`）：pinned headline + diagnostics + checkpoint provenance + population audit。
- `ThresholdReport`（`eval/calibration.py`）：`selected_threshold`、`selection_method`、`validation_population_name` 显式记录。
- `FusionPopulationInputs` / `FusionEvalReport` / `FusionSelectionSummary` / `StudentContributionVerdict`（`eval/eval_fusion.py`）：validation / report population 字段互斥，`student_contribution_pass` 显式。
- `GenOnlyReport` + `DiagnosticParseAudit`（`eval/eval_gen_only.py`）：strict vs normalized 指标分离；`teacher_conditioned=True` 作为字段写入。
- `FaithfulnessReport`（`eval/faithfulness.py`）：sample size + alpha / threshold provenance + 四个 variant（full / sufficiency / comprehensiveness / teacher-prob-ablated）走同一 `score_head`。

## 6. 本轮新增测试分别在防什么

- `tests/test_eval_head_only.py`（7 个）：防 threshold leakage、single-class 分支、checkpoint provenance 缺失、headline vs oracle 字段混用。
- `tests/test_eval_fusion.py`（9 个）：防 alpha 回退到默认、test-like 上重新选 alpha、guardrail 浮点误判、student_contribution_pass 被 low-alpha 伪装、tolerance trigger 错判。
- `tests/test_eval_gen_only.py`（6 个）：防 strict parse 失败静默 drop、normalized 指标进入 headline、teacher_conditioned 字段缺失。
- `tests/test_strict_output_schema.py`（3 个）：防 alias 字段进入 strict schema、字段顺序被重排。
- `tests/test_faithfulness.py`（6 个）：防 faithfulness 脱离 `score_head`、小样本伪装 formal、ablated card 未通过 schema-preserving 路径生成。

## 7. 剩余三个最大风险

1. **真实 backbone + LoRA 运行未跑通**。所有 ScorerReport 目前来自单元测试用的合成概率，尚未在真实 Qwen3-4B + LoRA checkpoint 上端到端验证 provenance / threshold / alpha 的实际分布。
2. **faithfulness 三个指标的样本量下限需要在真实数据上标定**。当前 `min_formal_sample_size` 是 schema 级 fail-closed 阈值，但具体数字应在真实 eval 次运行中根据 population 大小校准。
3. **alpha candidate 网格的粒度**。validation 选 alpha 目前依赖调用方传入 `alpha_candidates`；若传入过粗的网格（例如只有 `(0.0, 0.5, 1.0)`），optimal_alpha 可能因为离散化而没落在真实最优附近。应与 Task 9 gate 一并约束调用方至少使用 0.1 级别的 grid。

## 8. 验证证据

```bash
PYTHONPATH=. pytest -q tests/  # 216 passed（Task 11 + Task 15 合入之后）
```

相关 commit：`5465d0d` fusion 精修、`3838099` faithfulness 锁定到 `score_head`、以及 OMX team 期间 worker-1 / worker-2 / worker-3 / worker-4 多次 auto-checkpoint。
