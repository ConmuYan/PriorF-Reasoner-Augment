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
- Task 6 已完成并已人工审核通过（Validation/Offline Head Parity Test）
- Task 7 已完成（canonical joint trainer：L_gen + L_cls + L_distill）
- Task 8 已完成（formal eval suite：eval_head_only / eval_gen_only / eval_fusion / faithfulness）
- Task 9 已完成（gate manifest + gate_check + formal launcher integration）
- Task 11 已完成（teacher-probability ablation audit；prompts/11 专项任务）
- Task 15 已完成（README + operator runbook + stage launchers + smoke tests）
- prompts/ 下 15 个 task 的代码与 self-review 交付物均已落盘

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
- 15-task 总规划已完成实现与自审；如需继续，下一步应落入真实 Qwen3-4B + LoRA 权重的端到端运行验收（docs/030 §Task 2 runtime acceptance 的完整闭环）、以及将"YelpChi 首次 artifact 写入 + Amazon 旧 artifact 覆写"这两项 operator follow-up 执行到位。

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
- self-review/task6.md
- self-review/task7.md
- self-review/task8.md
- self-review/task9.md
- self-review/task11.md
- self-review/task15.md
- tests/test_eval_head_parity.py
- train/__init__.py
- train/train_stage2_canonical.py
- tests/test_canonical_trainer.py
- eval/calibration.py
- eval/eval_head_only.py
- tests/test_eval_head_only.py
- eval/eval_gen_only.py
- llm/parsing.py
- tests/test_eval_gen_only.py
- tests/test_strict_output_schema.py
- eval/eval_fusion.py
- llm/fusion.py
- tests/test_eval_fusion.py
- eval/faithfulness.py
- tests/test_faithfulness.py
- schemas/__init__.py
- schemas/gate_manifest.py
- scripts/gate_check.py
- scripts/run_full_pipeline.sh
- tests/test_gate_check.py
- tests/test_formal_launcher_gate.py
- evidence/ablations.py
- tests/test_teacher_prob_ablation.py
- README.md（rewritten as operator-facing orientation）
- docs/040_operator_runbook.md
- scripts/run_smoke.sh
- scripts/run_stage1.sh
- scripts/run_stage2.sh
- scripts/run_eval.sh
- tests/test_smoke_pipeline.py

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

Task 6 验收证据（Validation/Offline Head Parity Test）：
- tests/test_eval_head_parity.py 已提交（same prompt builder / chat template / token spans / hidden-state indices / probabilities / predictions）
- self-review/task6.md 已提交

Task 7 验收证据（canonical joint trainer：L_gen + L_cls + L_distill）：
- train/__init__.py、train/train_stage2_canonical.py 已提交
- tests/test_canonical_trainer.py 已提交
- self-review/task7.md 已提交

Task 8 验收证据（formal eval suite：prompts/10 + 12 + 13 + 14）：
- eval/eval_head_only.py、eval/calibration.py（+ tests/test_eval_head_only.py 7 passed）
- eval/eval_gen_only.py、llm/parsing.py（+ tests/test_eval_gen_only.py + tests/test_strict_output_schema.py 22 passed）
- eval/eval_fusion.py、llm/fusion.py（+ tests/test_eval_fusion.py，cherry-pick 合入 commit 5465d0d 修复 fusion guardrail epsilon 与 test fixtures，全部通过）
- eval/faithfulness.py（+ tests/test_faithfulness.py 6 passed；commit 3838099 锁定 faithfulness 到 score_head）
- self-review/task8.md 已提交

Task 9 验收证据（gate manifest + gate_check + launcher）：
- schemas/__init__.py、schemas/gate_manifest.py（Pydantic 模型，extra="forbid"，frozen=True，hex40 commit、hex64 data_manifest_hash、UTC generated_at、9 项 *_pass）
- scripts/gate_check.py（fail-closed gate 校验，任一 *_pass != True 抛 ValueError；CLI entrypoint）
- scripts/run_full_pipeline.sh（formal/gated/diagnostic 硬分流；formal 必须 --gate-manifest 且 gate 通过才会 exec 用户命令；set -euo pipefail）
- tests/test_gate_check.py、tests/test_formal_launcher_gate.py（合计 17 passed；原 commit e7d4987，cherry-pick 合入 main 作为 20316ef）
- self-review/task9.md 已提交

Task 11 验收证据（teacher-probability ablation audit；prompts/11）：
- evidence/ablations.py（TEACHER_PROB_MASK sentinel、ablate_teacher_prob(card)、TeacherProbAblationAudit Pydantic 模型、run_teacher_prob_ablation_audit 纯函数；不触发 inference，不重选 threshold/alpha）
- tests/test_teacher_prob_ablation.py（10 passed：schema 保持、重复 ablation 拒绝、checkpoint/population/path-audit 一致性、阈值边界、single-class 无虚构指标、frozen+extra=forbid）
- self-review/task11.md 已提交
- 相关 commit：9e41048

Task 15 验收证据（README + operator runbook + stage launchers + smoke tests；prompts/15）：
- README.md 已重写为 operator-facing 文档（三个 namespace 硬分流、什么算/不算 success、quick starts；旧 prompt-bundle 段落作为 "Prompt / task bundle" 子章节保留）
- docs/040_operator_runbook.md 新增（shortest valid path + fail conditions 排查顺序 + cheat-sheet）
- scripts/run_smoke.sh（diagnostic-only，set -euo pipefail，不接受 --mode）
- scripts/run_stage1.sh / scripts/run_stage2.sh（gated 默认；带 --gate-manifest 提升 formal，gate 失败不退回 gated）
- scripts/run_eval.sh（formal-only，强制 --gate-manifest，gate_check 前置）
- tests/test_smoke_pipeline.py（9 passed：canonical Evidence Card + prompt round-trip、schema fail-closed、4 个 launcher 的 fail-closed 路由行为）
- self-review/task15.md 已提交

整体验收：
- PYTHONPATH=. pytest tests/ -q → 216 passed（Task 1–15 全量，零回归）
- docs/030 Task 1–9 + prompts/ Task 1–15 的实现与自审交付物均已落盘

Pipeline smoke 运行时验收（真实 Qwen3-4B-Instruct-2507 + legacy teacher_exports parquet，diagnostic namespace）：
- scripts/run_head_only_smoke.py（legacy parquet → EvidenceCard adapter → build_prompt → Qwen3-4B forward(output_hidden_states=True) → pool_last_valid_token → random linear cls_head → sigmoid → ScorerReport；outputs/diagnostic/head_only_smoke/）
  - amazon subset_size=8：SMOKE OK，ScorerReport 通过 Pydantic 校验（n_total=8/pos=1/neg=7，is_single_class=False，auroc/auprc/brier 全部有限，prob 分布 7 项齐全）
  - amazon subset_size=64：SMOKE OK（n_total=64/pos=4/neg=60）
  - yelpchi subset_size=64：SMOKE OK（n_total=64/pos=9/neg=55）
  - 说明：cls_head 未经 Stage 2 训练，AUROC/AUPRC/Brier 数值不可作为质量评估，仅证明 prompt→chat_template→forward→pool→head→sigmoid→ScorerReport 端到端路径在真实 Qwen3-4B 权重下打通
- scripts/run_stage2_train_smoke.py（Qwen3-4B + PEFT LoRA r=8/alpha=16 on {q_proj, v_proj} + trainable linear cls_head + AdamW + Accelerator.backward；调用 run_canonical_train_step 1 步）
  - amazon batch_size=2：STAGE2 SMOKE OK，CanonicalStepReport 通过 Pydantic 校验（L_gen=2.8172/L_cls=1.2260/L_distill=1.3588/total=4.7225，total = L_gen + 1.0*L_cls + 0.5*L_distill 的 1e-5 恒等式满足，backward_via_accelerate=True）
  - yelpchi batch_size=2：STAGE2 SMOKE OK（L_gen=2.8496/L_cls=1.1300/L_distill=2.2258/total=5.0926）
  - 说明：trainable/total=2,949,120/4,025,417,216；LoRA 正确冻结 99.93% 参数，L_gen + L_cls + L_distill 反向传播与优化器步都成功，无 NaN
- scripts/run_smoke.sh（diagnostic-only launcher）：9 passed / 0.77s，tests/test_smoke_pipeline.py 全绿
- Gate manifest + formal launcher 通路（outputs/gated/manifests/gate_manifest.json 的合成示例）：
  - scripts/gate_check.py --manifest-path ... → gate_check: PASS（9 个 *_pass 全 True 时）
  - 负向验证：任一 *_pass 置 False 时，gate_check exit=1 且 scripts/run_eval.sh 同步 exit=1（不创建 outputs/formal/，fail-closed 合同生效）
  - scripts/run_eval.sh --gate-manifest ... → gate_check PASS → 建立 outputs/formal/ 命名空间、设置 PRIORF_OUTPUT_NAMESPACE=formal / PRIORF_OUTPUT_DIR=outputs/formal

真实 End-to-End 运行验收（2026-04-24，commit eaefd59，Qwen3-4B-Instruct-2507 + 真实 canonical teacher_exports）：

1) canonical teacher_exports 重新产出：
- scripts/legacy_mat_to_canonical.py → Amazon_canonical.mat / YelpChi_canonical.mat 落盘
- scripts/generate_teacher_exports.py → 双数据集各 3 个 population 的 canonical parquet + data_manifest + baseline_report 全部通过 _git_head_sha_or_fail 的 HEAD sha 校验（code_git_sha=eaefd594...）
  - outputs/gated/teacher_exports/amazon/{train,validation,final_test}/teacher_export.parquet  (8360/1194/2390 records；validation AUROC=0.9744 passed=True)
  - outputs/gated/teacher_exports/yelpchi/{train,validation,final_test}/teacher_export.parquet (32167/4595/9192 records；validation AUROC=0.9867 passed=True)
  - manifests/amazon/data_manifest.json, manifests/yelpchi/data_manifest.json

2) Stage 2 训练（scripts/run_stage2_train.py wrap run_canonical_train_step）：
- Amazon：outputs/gated/stage2/amazon/run_v1/，40 steps bs=2，1024-sample cap，LoRA r=8 α=16 on {q_proj,v_proj} + gradient checkpointing；损失从 L_gen=2.77 L_cls=2.12 L_distill=1.28 total=5.53 降到 L_gen=1.81 L_cls=0.06 L_distill=0.36 total=2.05，post-training validation (n=128) AUROC=0.5756
- YelpChi：outputs/gated/stage2/yelpchi/run_v1/，同配置；损失 total=5.37→2.17，post-training validation (n=128) AUROC=0.6328
- 两次运行都保存了 peft_adapter/ + cls_head.pt + run_record.json + train_log.jsonl；CanonicalStepReport 与 CanonicalTrainerRunRecord 全部通过 Pydantic 校验

3) Head-only inference（scripts/run_stage2_inference.py wrap score_head，加载已保存 adapter + cls_head）：
- Amazon final_test (n=256): auroc=0.5093, auprc=0.0882, brier=0.0654，ScorerReport 写入 outputs/gated/eval/amazon/scorer_report_amazon_final_test.json
- YelpChi final_test (n=256): auroc=0.6287, auprc=0.1889, brier=0.1229，ScorerReport 写入 outputs/gated/eval/yelpchi/scorer_report_yelpchi_final_test.json

4) Formal head-only eval（scripts/run_formal_head_only_eval.py wrap run_formal_head_only_eval）：
- Amazon outputs/formal/head_only/amazon/formal_head_only_report_amazon.json，validation-frozen threshold=0.0647，final_test auroc=0.5093 auprc=0.0882 f1@val_thr=0.1053 precision=0.0690 recall=0.2222
- YelpChi outputs/formal/head_only/yelpchi/formal_head_only_report_yelpchi.json，validation-frozen threshold=0.1394，final_test auroc=0.6287 auprc=0.1889 f1@val_thr=0.1818 precision=0.1750 recall=0.1892
- FormalHeadOnlyReport 全字段（checkpoint_bundle cross-identity + population metadata 对齐 + validation_threshold.source=validation + scorer_checkpoint_provenance==cls_head bundle + calibration population match）均通过 Pydantic model_validator

5) End-to-end driver：scripts/run_full_stage2_pipeline.sh 串联上述 5 步（legacy→canonical→teacher_exports→train→inference→formal eval），接收 --dataset {amazon,yelpchi} + --qwen-path + 训练/评估规模参数；每一步出错立即 set -euo pipefail 退出，可审计

运行时验收仍未覆盖（Follow-up，对真实发布必要但不在当前范围）：
- 大规模 Stage 2 训练（真实发布需要 TRAIN 全量 × 更多 steps × 学习率调度 + best-checkpoint 选择；当前 run_v1 是 40 steps × 1024 samples，仅作为 pipeline proof-of-run，非收敛训练）
- eval_gen_only / eval_fusion / faithfulness 三个 formal runner 的 CLI driver（需要 LLM 文本生成路径 + 解析，当前未实现 CLI；核心 library 函数已就绪）
- gate_manifest 自动组装（需从各 eval report 抽字段，目前仍是人工拼装；建议下一步写 scripts/generate_gate_manifest.py）

当前 canonical 路径（追加于 Task 6/7/8/11/15）：
- tests/test_eval_head_parity.py（Task 6 parity）
- train/__init__.py、train/train_stage2_canonical.py（Task 7 canonical trainer）
- eval/calibration.py、eval/eval_head_only.py（Task 8 head-only formal runner）
- eval/eval_gen_only.py、llm/parsing.py（Task 8 generation formal runner，strict vs normalized）
- eval/eval_fusion.py、llm/fusion.py（Task 8 fusion formal runner，validation-only alpha）
- eval/faithfulness.py（Task 8 faithfulness，复用 score_head）
- evidence/ablations.py（Task 11 teacher-prob ablation audit）

当前 formal 路径（追加于 Task 9/15）：
- schemas/gate_manifest.py（GateManifest Pydantic 模型）
- scripts/gate_check.py（fail-closed gate 校验入口）
- scripts/run_full_pipeline.sh（formal/gated/diagnostic 硬分流 launcher）
- scripts/run_eval.sh（formal-only wrapper）

当前 diagnostic 路径（追加于 Task 15）：
- scripts/run_smoke.sh + tests/test_smoke_pipeline.py（canonical plumbing smoke；不进 formal namespace）

遗留 follow-up（与 15-task 代码交付无直接关联，但对真实发布必要）：
- docs/030 §Task 2 runtime acceptance 的"YelpChi artifact 首次写入"与"Amazon 旧 artifact 以当前 canonical .mat + 已纳入 git 的 bridge 源码覆写"（status_package §Task 2 运行时验收已登记具体证据与原因）
- 真实 Qwen3-4B + LoRA 权重下的端到端运行验收（当前所有 ScorerReport / FusionEvalReport / FaithfulnessReport 尚未在真实 backbone 产出）
- generate_gate_manifest 工具（目前 gate_manifest 仍需人工组装；建议后续新增专用工具自动从各 eval report 抽取字段填充）
- .gitignore 补齐 outputs/、assets/data/*_canonical.mat、priorf_gnn/、manifests/（历史条目）
