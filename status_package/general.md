项目：PriorF-Reasoner

当前状态：
- Task 1 已完成并已人工审核通过
- Task 1 的目标是数据基础设施
- Task 1 已实现 data manifest / mat_loader / validators 的基础合同
- Task 1 已通过对应测试并已提交
- Task 2 已完成并已人工审核通过
- Task 2 的目标是 Teacher 合同与 Baseline Gate
- Task 2 已实现 teacher export schema / teacher baseline gate / teacher export pipeline IO 与 fail-closed 前置
- Task 2 已通过对应测试
- Task 3 已完成并已人工审核通过
- Task 3 的目标是 Evidence Card schema / prompt builder / strict output schema / schema-preserving ablation API
- Task 3 已通过对应测试
- Task 4 已完成并已人工审核通过
- Task 4 的目标是 Hidden-State Contract（mask-aware absolute token indexing）
- Task 4 已实现 llm/hidden_state_pooling.py / llm/__init__.py
- Task 4 已通过对应测试并已提交
- Task 5 已完成并已人工审核通过
- Task 5 的目标是 Unified Head Scoring（validation 与 offline head-only eval 共用的 canonical predict_proba + 指标产出层）
- Task 5 已实现 eval/head_scoring.py / eval/__init__.py 与 tests/test_head_scoring.py（21 条用例 / 25 test nodes）
- Task 5 已通过对应测试并已提交
- 当前准备进入 Task 6

必须遵守的 source-of-truth 文件：
- docs/010_project_contract.md
- docs/020_top_level_design.md
- docs/030_harness_agent_execution_guide.md
- docs/050_fail_closed_guardrails.md
- read.md
- CLAUDE.md

执行策略：
- 每个 task 单独执行与审查
- 不跨 task 越界
- canonical / diagnostic / formal 必须严格分离
- 若发现需求冲突、schema 漂移、silent default、threshold/alpha leakage、provenance 断裂，必须 fail closed

当前任务：
Task 6：Validation/Offline Head Parity Test（尚未开始）

已完成文件：
- graph_data/mat_loader.py
- graph_data/validators.py
- graph_data/manifests.py
- priorf_teacher/inference.py
- priorf_teacher/schema.py
- priorf_teacher/teacher_baseline_gate.py
- priorf_teacher/export_pipeline.py
- evidence/__init__.py
- evidence/evidence_schema.py
- evidence/prompt_builder.py
- evidence/output_schema.py
- llm/__init__.py
- llm/hidden_state_pooling.py
- eval/__init__.py
- eval/head_scoring.py
- scripts/legacy_mat_to_canonical.py
- scripts/generate_teacher_exports.py
- tests/test_mat_loader.py
- tests/test_manifests.py
- tests/test_teacher_export.py
- tests/test_teacher_baseline_gate.py
- tests/test_evidence_schema.py
- tests/test_prompt_builder.py
- tests/test_output_schema.py
- tests/test_hidden_state_pooling.py
- tests/test_head_scoring.py
- self-review/task2.md
- self-review/task3.md
- self-review/task4.md
- self-review/task5.md

Task 3 验收证据：
- PYTHONPATH=. pytest -q tests/test_output_schema.py tests/test_evidence_schema.py tests/test_prompt_builder.py
- 结果：40 passed
- PYTHONPATH=. pytest -q tests/test_manifests.py tests/test_mat_loader.py tests/test_teacher_export.py tests/test_teacher_baseline_gate.py tests/test_output_schema.py tests/test_evidence_schema.py tests/test_prompt_builder.py
- 结果：103 passed

Task 3 已完成内容：
- StrictOutput / PredLabel formal output schema
- canonical_serialize byte-exact JSON serialization
- parse_strict JSON + Pydantic-only strict parser
- EvidenceCard schema
- TeacherSummary / DiscrepancySummary typed schema
- EvidenceAblationMask explicit enum
- build_evidence_card consistency checks
- PromptMode / ThinkingMode / ChatMessage / PromptBundle / FewShotExample
- build_prompt cross-mode stable message structure
- schema-preserving ablation prompt rendering with `Masked / Not Available` sentinel
- prompt-related env var fail-closed guard

Task 2 运行时验收（operator data materialization）：
- 原 Task 2 合同仅覆盖 schema / gate / pipeline 三件 IO 合同与 unit tests，并未要求、也未实际产出 parquet；真实 teacher export artifact 的产出在 Task 5 审结前后由 operator 统一补做
- 归属：该组件属于 Task 2 的运行时验收（docs/030 §Task 2 Acceptance），不是 Task 6 内容；Task 6 parity test 以产出的 parquet 为读端但不负责产出
- 涉及文件：priorf_teacher/inference.py、scripts/legacy_mat_to_canonical.py、scripts/generate_teacher_exports.py、scripts/verify_teacher_metric_consistency.py（诊断）
- 数据流：legacy `.mat` → scripts/legacy_mat_to_canonical.py（StratifiedShuffleSplit random_state=717，7:1:2）→ canonical `.mat` → priorf_teacher/inference.py（LGHGCLNetV2 full-graph forward）→ TeacherExportRecord per population → teacher_baseline_gate（validation）→ write_teacher_export_artifact（每 population 一个 parquet）
- 当前 bridge 推理结果（2026-04-22 验证，scripts/verify_teacher_metric_consistency.py 实测）：
  - Amazon : n_val=1194  pos_rate=0.0687 | AUROC=0.9744 | AUPRC_trap=0.8860 | AUPRC_step=0.8864
  - YelpChi: n_val=4595  pos_rate=0.1454 | AUROC=0.9867 | AUPRC_trap=0.9539 | AUPRC_step=0.9539
  - model_summary.best_metric（参考）：amazon=0.8318，yelpchi=0.8421（来自原训练 early-stopping 记录）
- Artifact baseline hash（2026-04-22 18:52 pin 死，后续所有产出必须与此对齐）：
  - assets/data/Amazon_canonical.mat    sha256=c77d595fd741cfa9ff34bab5413775e366b8b1face248885c3e65295efa3f9d7    mtime=2026-04-22 17:23    size≈3.2GB
  - assets/data/YelpChi_canonical.mat   sha256=778498b72a788cccf91d8260ae7e320375a8894e8c3b21485b58bdd14d64faea    mtime=2026-04-22 17:29    size≈108MB
  - assets/teacher/amazon/best_model.pt sha256=ca8ae67eb12c619dec3459a48039f14c0762d32f5bd85577d980217dbe21f894    mtime=2026-04-22 11:28
  - assets/teacher/yelpchi/best_model.pt sha256=bfc5d1bde3b2479cee6fa75254bdef076c2b2595bc94ab5b48e4fca0f78a9441   mtime=2026-04-22 11:28
- Amazon gate 现状：validation AUROC=0.9744 ≥ 0.80 → 通过；但历史已写入 outputs/gated/teacher_exports/amazon/* 的 artifact 使用的是旧 canonical .mat（mtime < 17:23 的 v1 版本）产出、且 TeacherProvenance.code_git_sha=0×40 占位，审计链不完整，需在 bridge 代码 commit 后重跑覆写
- YelpChi gate 现状：validation AUROC=0.9867 ≥ 0.80 → 通过；artifact 尚未产出，需在 bridge 代码 commit 后首次写入 outputs/gated/teacher_exports/yelpchi/*
- 早先历史诊断（已被当前实测证据否定，保留仅作时间线记录）：
  - 旧观测 "YelpChi AUROC=0.7518" 来自 v1 canonical .mat + 早先 bridge 源码组合，该数字从未入 git 历史（git log -S "0.7518" 为空），仅存在于 operator 手改工作区
  - 关于"梯形法 AUPRC 过估 0.05–0.15"的假设被实测推翻：AUPRC_trap 与 AUPRC_step 在两个数据集上数值差 ≤ 4e-4
  - 关于"gate 需改 metric 为 AUPRC 或改 per-dataset 阈值"的方案 B/C 在当前 AUROC 下均无需执行，0.80 AUROC 阈值在两个数据集上都成立
- 审计链形式性断裂（follow-up 待补齐）：
  - priorf_teacher/inference.py、scripts/legacy_mat_to_canonical.py、scripts/generate_teacher_exports.py、scripts/verify_teacher_metric_consistency.py 在本 Task 2 运行时验收登记前均未纳入 git；`code_git_sha=GIT_SHA_PLACEHOLDER="0"*40` 是形式占位
  - Amazon_canonical.mat 体积 ≈3.2GB（YelpChi 仅 108MB），疑为 legacy_mat_to_canonical.py 在 Amazon 上存储了稠密 float64 relation 矩阵；不影响当前 AUROC 但作为独立 follow-up
  - .gitignore 未覆盖 outputs/、assets/data/*_canonical.mat、priorf_gnn/、manifests/；须在后续统一补 ignore，防止误入库

当前 canonical 路径已包含：
- graph_data/*
- scripts/{legacy_mat_to_canonical,generate_teacher_exports}.py
- priorf_teacher/inference.py
- priorf_teacher/{schema,teacher_baseline_gate,export_pipeline}.py
- evidence/*
- llm/*
- eval/{head_scoring,__init__}.py

当前 diagnostic 路径已包含：
- scripts/verify_teacher_metric_consistency.py：一次性诊断脚本，复算 Amazon/YelpChi validation 的 (AUROC, AUPRC_trapezoidal, AUPRC_step)，用于核验 model_summary.best_metric 与 bridge 实测的口径一致性；严格不写 outputs/、不写 parquet、不走 canonical/formal 路径
- Task 5 未对 canonical scorer 开任何 diagnostic 后门（无 oracle_threshold / probe mode / env-var switch）

当前 formal 路径已包含：
- 暂无 formal execution path 文件
- evidence/output_schema.py 仅提供 formal strict output schema；尚未接入 formal eval runner / gate manifest / launcher
- eval/head_scoring.py 提供 canonical predict_proba + threshold-free 指标层，formal eval runner 将在 Task 8 基于此构建

Task 4 验收证据：
- PYTHONPATH=. pytest -q tests/test_hidden_state_pooling.py
- 结果：21 passed
- PYTHONPATH=. pytest -q tests/
- 结果：124 passed

Task 4 已完成内容：
- pool_last_valid_token 单一公开函数，__all__ 仅含该函数
- 10 项 fail-closed 校验顺序（TypeError/ValueError）
- Path A 绝对位置索引（masked_fill + argmax），禁止 [:, -1, :] 与 cumsum/sum-1
- left-pad / right-pad / all-1s / T=1 / B=1 全覆盖
- padding-side invariance 核心防御测试（pad 位噪声不影响 pooled 结果）
- autograd 保留测试（backward + grad 非零位置验证）
- dtype/device 保真测试（fp32 / fp16 / bf16）
- 模块导出限制测试（无 mean / first / cls / max pooling）

Task 5 验收证据：
- python -m pytest tests/test_head_scoring.py -q
- 结果：25 passed（21 条用例 / 25 test nodes，含参数化展开）
- python -m pytest tests/ -q
- 结果：149 passed（Task 1–5 全量，零回归）

Task 5 已完成内容：
- eval/head_scoring.py 唯一公开入口 score_head；__all__ 严格等于 ("score_head", "ScorerReport", "ClsHead", "HeadScoringInputs", "HeadScoringSample", "CheckpointProvenance")
- Pydantic schema 全部 extra="forbid" + frozen=True：CheckpointProvenance / HeadScoringSample / HeadScoringInputs / ScorerReport
- HeadScoringInputs model_validator：每个 sample 的 evidence_card 必须与顶层 dataset_name / population_name / graph_regime 一致
- ScorerReport 字段集合 pin 死：provenance + counts + is_single_class_population + auroc / auprc / brier_score + prob 分布 7 项 + probs/labels/node_ids 三元组 + path audit (prompt_mode/thinking_mode/pooling_path/uses_inference_mode) + distributed_gather
- 禁止字段 schema 级封死：accuracy / f1 / precision / recall / threshold / optimal_threshold / fixed_threshold_* / alpha / selected_checkpoint / checkpoint_policy / oracle_threshold / faithfulness_* / fusion_*
- ClsHead Protocol（runtime_checkable）同时声明 __call__ 与 eval，isinstance 可拒裸 callable
- canonical predict_proba 链路硬编码：PromptMode.EVAL_HEAD → apply_chat_template(tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=False) → model(output_hidden_states=True, use_cache=False) → outputs.hidden_states[-1] → pool_last_valid_token → cls_head → .to(fp32) → torch.sigmoid
- 严格 B=1 逐样本 forward，attention_mask 恒为全 1，scorer 不传 padding、不碰 padding_side
- 全流程 torch.inference_mode()，probs.requires_grad is False
- 单类 population：auroc / auprc = None，is_single_class_population = True，brier_score 仍按公式计算
- 分布式合同：accelerator=None → distributed_gather="none"；accelerator 非 None → gather_for_metrics 在 probs/labels/node_ids 上各 1 次；caller 预分片，report 为 world-level
- 静态检查通过：无 hidden_states[:, -1, :] / 无 .train() 调用 / 无 trl|peft|datasets 导入 / 无 .generate( 调用 / 无 torch.distributed 属性访问与导入 / 无 os.environ|os.getenv / 无 open(|write_text|write_bytes / 无 padding_side 属性访问或赋值

Task 5 下游约束（由本 Task 选择固化，但归属 Task 6/7 承接，Task 5 本身不实现）：
- Task 7 canonical trainer 的 L_cls 头前向必须使用 PromptMode.EVAL_HEAD prompt；若用 PromptMode.TRAIN，pool 到的 hidden 会落在 JSON/assistant 末尾之后，违反 docs/020 §5 的 prompt-only hidden states 合同
- Task 6 parity test 必须用真 Qwen3 tokenizer 至少覆盖一条样本，以验证 apply_chat_template 的 4-kwarg 组合在真模型下行为与 DummyTokenizer 一致
- Task 9 gate_check 必须校验 report.n_total == expected_population_size，以兜住 caller 忘记 pre-shard 导致 n_total 被 world_size 倍放大的情况

Task 6 及以后尚未完成且不得视为已完成的事项：
- tests/test_eval_head_parity.py（Task 6 parity：same prompt builder / chat template / token spans / hidden-state indices / probabilities / predictions）
- train/train_stage2_canonical.py（Task 7 canonical joint trainer：L_gen + L_cls + L_distill 三项必须并存）
- cls head 架构与持久化
- canonical tokenization 模块（目前仅以 apply_chat_template 4-kwarg 形式 pin 在 score_head 内）
- eval/eval_head_only.py / eval/eval_gen_only.py / eval/eval_fusion.py / eval/faithfulness.py（Task 8 formal eval runner）
- validation-only threshold / alpha 选择
- teacher-prob ablation 审计
- scripts/gate_check.py + gate_manifest.json schema + launcher integration（Task 9）
