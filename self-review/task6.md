# Task 6 Fail-Closed 自审（`tests/test_eval_head_parity.py`）

范围：本轮新增/修改：

- `tests/test_eval_head_parity.py`
- `tests/test_teacher_export_script_smoke.py`
- `self-review/task6.md`
- `scripts/generate_teacher_exports.py`
- `priorf_teacher/inference.py`
- `priorf_gnn/lghgcl/models/asda_layer.py`

未修改 Task 1–5 的 formal schema、prompt builder、scorer、teacher export pipeline schema/read-write 合同。生产侧改动仅限 teacher inference/export operator bridge：让用户提供的 canonical assets 与 teacher source 能被脚本导入、预检、加载，不改变 Task 2 的 `TeacherExportRecord` / `TeacherExportManifest` schema。

## 1. 是否引入新的 silent default

**没有。**

本轮只新增测试，不新增运行时默认值。测试中的固定 `0.5` 只用于 parity equality check：

- 不从 validation / test-like rows 选择；
- 不写入 `ScorerReport`；
- 不作为 formal metric；
- 不改变 `eval/head_scoring.py` 的 threshold-free 合同。

真实 Qwen3 tokenizer 路径固定为 `/data1/mq/models/Qwen3-4B-Instruct-2507`，并使用 `local_files_only=True`，防止测试静默联网 fallback。

`scripts/generate_teacher_exports.py` 只新增显式 repo-root / teacher-source import path bootstrap，避免 operator CLI 在没有手动 `PYTHONPATH` 时导入失败；没有新增 schema 或 metric 默认。

## 2. 是否让 canonical / diagnostic / formal 路径混淆

**没有。**

- 测试只验证 canonical `score_head` 链路。
- 未新增 diagnostic runner、oracle threshold、probe mode、fusion、faithfulness 或 formal output。
- 未写入 `outputs/formal` / `outputs/diagnostic` / `outputs/gated`。
- epoch-end validation 与 offline eval 在测试中均被定义为同一个 canonical scorer call surface。
- teacher inference bridge 只服务 gated teacher export；仍通过既有 `write_teacher_export_artifact()` schema/路径 gate 写出，不绕过 Task 2 合同。

## 3. 是否重新引入 threshold / alpha leakage

**没有。**

测试没有选择 threshold 或 alpha，也没有新增任何生产字段。`score_head` 仍只返回：

- probabilities；
- ground-truth labels；
- threshold-free metrics；
- provenance / path audit。

`0.5` 只用于验证两条路径的 fixed-threshold parity decisions 完全相同，等价于“如果下游拿同一个固定判决规则，两边结果也必须一致”。

## 4. 是否让 formal metrics 和 diagnostic metrics 混用

**没有。**

新增测试不产出 report 文件，不写命名空间，不新增 metrics schema。它只比较两个内存中的 `ScorerReport` 与 trace：

- prompt messages；
- chat-template kwargs；
- input ids / attention mask；
- absolute last-valid hidden index；
- classifier input tensor；
- logits；
- probabilities；
- fixed-threshold parity decisions。

## 5. 本次新增了哪些 schema / provenance / gate contract

**没有新增生产 schema。**

新增的是 Task 6 parity test contract：

- validation 与 offline head-only eval 必须都通过 `score_head`；
- `PromptMode.EVAL_HEAD` 必须用于 head path；
- `ThinkingMode` 必须显式传入；
- tokenizer 必须走固定 `apply_chat_template(..., tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=False)`；
- hidden-state index 必须来自 `pool_last_valid_token` 的 mask-aware absolute index；
- classifier input / logits / probs 必须 sample-wise identical；
- 本地真 Qwen3 tokenizer 必须接受上述四个固定 kwargs。

新增 operator bridge contract：

- `scripts/generate_teacher_exports.py --help` 必须在不手动设置 `PYTHONPATH` 时成功；
- teacher inference bridge 读取 canonical `x` / `ground_truth_label` / `split_vector` / `relation_0..2`，不再依赖 legacy `.mat` 作为 split/population source；
- ASDA export-time scatter-add 使用内置 `Tensor.scatter_add_` fallback，避免 export-only 环境缺 `torch_scatter` 时连 CLI help / checkpoint load 都失败。

## 6. 新增测试分别在防什么

### `test_validation_and_offline_head_eval_are_samplewise_identical`

防：

- validation 和 offline eval 走不同 prompt builder；
- head path 误用 `PromptMode.TRAIN` / assistant JSON suffix；
- chat-template kwargs 漂移；
- token ids / attention mask 漂移；
- hidden index 回退成相对 `sum(mask)-1` 或其它非唯一实现；
- classifier input tensor 与 logits 只在 metrics 层“接近”而非 sample-wise identical；
- probabilities / parity predictions 不一致却被下游阈值或指标掩盖。

### `test_real_local_qwen3_tokenizer_accepts_canonical_eval_head_chat_template`

防：

- Task 5 只在 dummy tokenizer 上成立，真 Qwen3 tokenizer 不接受固定四 kwargs；
- 本地模型路径缺失时测试静默跳过；
- tokenizer 输出形状不满足 scorer 所需的 `[1, T]` input ids / attention mask；
- 真 tokenizer 输出接入 `pool_last_valid_token` 后 hidden index 语义不明确。

### `test_generate_teacher_exports_help_imports_without_pythonpath`

防：

- operator 直接运行 `python scripts/generate_teacher_exports.py --help` 时因 `graph_data` / `priorf_teacher` / teacher source import path 缺失而失败；
- 用户已放置 teacher source/assets 但导出脚本连参数说明都不可用。

## 7. 剩下的 3 个最大风险点

### R1. Task 7 canonical trainer 仍未实现

本测试能锁定 parity surface，但不能阻止尚未存在的 trainer 在实现时绕过 `score_head` 或用 `PromptMode.TRAIN` 计算 `L_cls`。Task 7 必须显式复用同一 prompt/tokenization/pooling/head path。

### R2. 真 Qwen3 模型权重未在 Task 6 中加载

本测试只加载真实 tokenizer，不加载 4B 权重，避免把 Task 6 变成慢速/显存依赖测试。真实模型 forward parity 需要 Task 7/8 的 smoke pipeline 或 gate 层继续覆盖。

### R3. teacher export 全量实跑仍依赖后续操作验证

用户已提供 teacher assets/source 与脚本；本轮已验证 `scripts/generate_teacher_exports.py --help`、YelpChi canonical flat load、teacher checkpoint load。但 Task 6 本身不执行全量 Amazon/YelpChi teacher inference/export；全量导出仍需要按运行手册单独运行并确认耗时 / 显存 / CPU 内存条件。
