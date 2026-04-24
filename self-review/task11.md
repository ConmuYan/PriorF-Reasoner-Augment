# Task 11 Fail-Closed 自审（teacher-probability ablation audit）

覆盖 `prompts/11_task_11_teacher_prob_ablation_audit.txt`。docs/030 的 Task 编号中没有独立的 Task 11；本任务位于 Task 8 formal eval 与 Task 9 gate 之间。

范围：本轮新增：

- `evidence/ablations.py`
- `tests/test_teacher_prob_ablation.py`
- `self-review/task11.md`（本文件）

未修改 `evidence.evidence_schema` / `evidence.prompt_builder`（二者已在 Task 3 中支持 `EvidenceAblationMask.TEACHER_SUMMARY_TEACHER_PROB` 与 `"Masked / Not Available"` sentinel）；未修改 `eval.head_scoring` / `eval.eval_head_only`；未新增依赖。

## 1. ablation 如何保持 schema-preserving

`ablate_teacher_prob(card)` 返回一份 `EvidenceCard.model_copy(update=...)` 的深拷贝，只做两件事：

1. `teacher_summary.teacher_prob = None`；
2. `ablation_mask = card.ablation_mask | {TEACHER_SUMMARY_TEACHER_PROB}`。

所有其它字段（`teacher_summary` 其它条目、`discrepancy_summary`、`relation_profile`、`neighbor_summary`、`task_instruction`、`node_id`、`schema_version` 等）逐字保留。`EvidenceCard` 的 `model_validator(mode="after")` 会立刻对新 card 重新生效：masked 字段必须为 `None`，其它 unmasked 字段必须为 non-None，`schema_version` 必须固定。这意味着任何"把整张 card 置空"或"把别的字段顺带删掉"的退化都会立刻被 schema 层拒绝。

Prompt builder 读到 `TEACHER_SUMMARY_TEACHER_PROB` mask 后会把对应行渲染为 `"Masked / Not Available"`，字面串与其它字段布局完全一致；下游 LLM 看到的 prompt 只差这一行，不会触发 "样本整体长度/结构改变" 的分布漂移。

## 2. full card 与 ablated card 的对比输出长什么样

`run_teacher_prob_ablation_audit(full_report, ablated_report, dependency_threshold)` 返回的 `TeacherProbAblationAudit`（`frozen=True, extra="forbid"`）字段集：

- provenance：`dataset_name` / `population_name` / `graph_regime` / `provenance`（直接引用自 `full_report.checkpoint_provenance`）
- 标识：`ablation_target = "teacher_summary.teacher_prob"`（Literal 类型）
- 样本量：`n_total` / `n_positive` / `n_negative`
- 指标对：`auroc_full / auroc_ablated`、`auprc_full / auprc_ablated`、`brier_full / brier_ablated`
- 差值：`auroc_delta = auroc_full - auroc_ablated`（brier 同理）
- 阈值 & 触发：`dependency_threshold`（`gt=0.0, lt=1.0`）、`teacher_prob_dependency_high`（`StrictBool`）
- path audit：`prompt_mode` / `thinking_mode` / `pooling_path` / `uses_inference_mode` / `distributed_gather` 直接 echo 自 full_report，用于下游审计时核对两次 inference 路径一致。

## 3. teacher_prob_dependency_high 的判定逻辑

纯函数：

```text
dependency_high = (auroc_delta >= dependency_threshold)
```

由 `TeacherProbAblationAudit` 的 `_delta_flag_must_match_auroc_delta` model_validator 再次校验——即使调用方构造 audit 对象时传入不一致的 flag，也会在 pydantic validation 阶段被拒绝。

两类边界：

- single-class population（`is_single_class_population=True`）：`auroc_full` / `auroc_ablated` 必须都是 `None`，`auroc_delta` 必然为 `None`，此时 `teacher_prob_dependency_high` 被强制为 `False`；validator 也校验此一致性。
- non-single-class 但 `auroc_delta is None`：直接抛 `ValueError`，不凭空填数。

## 4. 是否引入新的 silent default / canonical-formal 混淆 / threshold 或 alpha leakage

**全无。**

- `dependency_threshold` 无默认值；调用方必须显式传，且 `gt=0, lt=1` 拒绝 0 / 1 / 负数。
- audit 不写任何 `outputs/` 路径；不 emit gate manifest；不跟 `gate_check.py` 耦合。
- audit 既不触发 inference，也不重算 threshold / alpha；只消费已经在 Task 8 `eval_head_only` 路径上跑出来的两份 `ScorerReport`。
- audit 在发现 `checkpoint_provenance` / `population_name` / `n_total` / `n_positive` / `n_negative` / `is_single_class_population` / `prompt_mode` / `thinking_mode` / `pooling_path` / `uses_inference_mode` / `distributed_gather` 任一项 full vs ablated 不一致时直接 `raise ValueError`，拒绝出 audit。

## 5. 本轮新增的 schema / provenance / gate contract

- `evidence.ablations.TEACHER_PROB_MASK`：ablation 目标 sentinel。
- `evidence.ablations.ablate_teacher_prob(card) -> EvidenceCard`。
- `evidence.ablations.TeacherProbAblationAudit`：唯一的正式 audit 产出模型。
- `evidence.ablations.run_teacher_prob_ablation_audit(*, full_report, ablated_report, dependency_threshold) -> TeacherProbAblationAudit`：唯一合法的入口函数。

## 6. 本轮新增测试分别在防什么

`tests/test_teacher_prob_ablation.py`（10 个）：

1. `test_ablate_teacher_prob_masks_only_prob_and_preserves_rest`：防 ablation 顺带改其它字段。
2. `test_ablate_teacher_prob_rejects_already_masked_card`：防重复 ablation 给出一个语义不清的 card。
3. `test_audit_flags_dependency_high_when_auroc_drop_exceeds_threshold`：保证 flag 真的会触发。
4. `test_audit_does_not_flag_when_auroc_drop_below_threshold`：防阈值被忽略。
5. `test_audit_rejects_checkpoint_provenance_mismatch`：防不同 checkpoint 的两份 report 被跨对比。
6. `test_audit_rejects_population_mismatch`：防 validation 与 final_test 被混合对比。
7. `test_audit_rejects_path_audit_drift`：防 `distributed_gather` / `thinking_mode` 等改动后还被视为同一条 inference 路径。
8. `test_audit_rejects_invalid_threshold`：防 threshold=0 或 1。
9. `test_audit_handles_single_class_population_without_fabricating_metric`：防 single-class 下被凭空凑出 AUROC。
10. `test_audit_report_is_frozen_and_extra_forbid`：防下游通过 duck-typing 新增字段。

## 7. 剩余三个最大风险

1. **dependency_threshold 的选择目前靠业务端自订。** 本任务只给出严格 schema 和 fail-closed 逻辑；真实大小（通常是 AUROC 降幅 0.03–0.10 区间）应在 Task 15 runbook 给出的"fail conditions 排查顺序"里与业务一同决定，不应默认写死 0.05 入模块。
2. **其它 teacher 字段的 ablation 当前尚无专用 audit 模块。** `EvidenceAblationMask` 已经枚举了 teacher_logit / hsd / asda_switch / branch_gap 等多项，但目前 Task 11 contract 只覆盖 teacher_prob 这一项。若将来需要对 teacher_logit / branch_gap 也做 formal audit，应在此模块内并列添加 `run_teacher_logit_ablation_audit` 等函数，而不是扩展现有函数的目标字段。
3. **ablated report 实际生产时需 Task 8 配合。** `run_teacher_prob_ablation_audit` 本身不负责产出 ablated `ScorerReport`；它假设调用方已经对 ablated Evidence Cards 跑过一次 `eval.head_scoring.score_head` 并拿到配对 report。若调用方误用 full inputs 跑 ablated 一次 inference，audit 不会察觉（但 path audit 字段会至少保证 inference 配置一致）。

## 8. 验证证据

```bash
PYTHONPATH=. pytest -q tests/test_teacher_prob_ablation.py  # 10 passed
PYTHONPATH=. pytest -q tests/                               # 216 passed（Task 15 合入之后）
```

Commit：`9e41048`。
