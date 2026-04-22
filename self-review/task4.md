# Task 4 Fail-Closed 自审

## 1. 本次改动是否引入了新的 silent default

**否。**

`pool_last_valid_token` 为纯函数，两个参数 `hidden_states`、`attention_mask` 均为无默认值的位置参数。模块内：

- 无模块级全局可变状态
- 无 `lru_cache`
- 无环境变量读取
- 无隐式 `padding_side` 推断
- 无默认 threshold、默认 alpha、默认 diagnostic mode、默认 graph regime、默认 population
- 无 alias parser、retry parser、env var fallback 或 legacy asset fallback

## 2. 本次改动是否让 canonical 和 diagnostic 路径再次混淆

**否。**

本次只覆盖 Task 4 的 Hidden-State Contract：

- `llm/hidden_state_pooling.py` — 唯一公开函数 `pool_last_valid_token`
- `llm/__init__.py` — 仅 re-export `pool_last_valid_token`
- `tests/test_hidden_state_pooling.py` — 21 条边界测试

没有实现：

- diagnostic parser
- diagnostic eval
- diagnostic output namespace
- oracle threshold
- ablation metric
- trainer
- head scoring
- fusion
- faithfulness
- formal eval
- gate check
- launcher

因此没有把 canonical 和 diagnostic 路径混在一起。

## 3. 本次改动是否可能重新引入 threshold / alpha leakage

**否。**

本次没有新增或修改任何：

- threshold selection
- alpha selection
- metric computation
- validation/test report
- fusion logic
- head scoring
- eval runner
- label 字段
- population 命名

`pool_last_valid_token` 不接触标签、概率阈值、融合权重或任何评估指标。输入输出均为原始张量，无 leakage 通道。

## 4. 本次改动是否可能让 formal metrics 和 diagnostic metrics 混用

**否。**

原因：

- 没有新增 metrics。
- 没有新增 report schema。
- 没有写入 `outputs/formal` / `outputs/diagnostic` / `outputs/gated`。
- 不计算 AUROC、AUPRC、accuracy、F1、parse rate 等任何 metric。

## 5. 本次改动新增了哪些 schema / provenance / gate contract

### 新增 runtime contract（无 Pydantic schema，因纯张量运算模块）

- **输入类型合同**：`hidden_states` / `attention_mask` 必须为 `torch.Tensor`，`hidden_states` 为 3-D `[B,T,H]`，`attention_mask` 为 2-D `[B,T]`
- **形状对齐合同**：`hidden_states.shape[:2] == attention_mask.shape`
- **维度非空合同**：`B >= 1, T >= 1, H >= 1`
- **设备一致合同**：两 tensor 必须同 `device`
- **浮点合同**：`hidden_states.dtype.is_floating_point`
- **数值有限合同**：`hidden_states` 不含 NaN / Inf
- **掩码值域合同**：`attention_mask` 严格 `⊆ {0, 1}`（bool / int / exact float 0.0/1.0）
- **掩码连续形态合同**：每行必须为 `0^a 1^b`、`1^b 0^a` 或全 1；否则 ValueError 含违法行索引
- **非全 pad 合同**：不允许 `attention_mask.sum(dim=1) == 0` 的行；否则 ValueError 含违法行索引
- **绝对位置索引合同**：采用 Path A（`masked_fill(..., -1)` + `argmax`），禁止 `[:, -1, :]`、`sum-1`、`cumsum` 等相对索引
- **autograd 保留合同**：返回 tensor 保留 `hidden_states` 计算图，不包裹 `torch.no_grad()`，不 `.detach()`，不原地修改入参
- **输出属性合同**：返回 `[B, H]`，dtype == `hidden_states.dtype`，device == `hidden_states.device`

### 未新增 schema / provenance / gate manifest

- 无 Pydantic model（纯张量运算，不产出需序列化 artifact）
- 无 `population_name`、`split_values`、`node_ids_hash` 等 provenance 元数据（下游责任）
- 无 `gate_manifest.json` 或 gate check script（本 Task 为前置基础设施）

## 6. 本次新增测试分别在防什么

### `tests/test_hidden_state_pooling.py`（21 条，全部 pass）

| 测试 | 防御目标 |
|---|---|
| `test_single_no_pad` | 单条无 pad 序列的基本正确性 |
| `test_single_t1` | T=1 边界 |
| `test_mixed_left_pad` | 多 batch 混合 left-pad 长度正确提取 last-valid-token |
| `test_mixed_right_pad` | 多 batch 混合 right-pad 长度正确提取 last-valid-token |
| `test_batch_all_ones` | 全 1 mask 时 pooled == `hs[:, -1, :]` |
| `test_padding_side_invariance` | **核心防御**：同一内容分别 left-pad / right-pad，pad 位赋噪声，两次 pooled `torch.equal` 成立 |
| `test_all_pad_row` | 全 pad 行必须触发 ValueError，消息含违法行索引 |
| `test_non_contiguous_mask` | 非连续 mask（如 `[1,0,1,1]`）触发 ValueError，消息含违法行索引 |
| `test_mask_values_outside_zero_one` | mask 含 2 / 0.5 等非法值触发 ValueError |
| `test_hidden_states_non_float` | 整数 hidden_states 触发 TypeError |
| `test_wrong_dims` | 2-D hidden_states 或 1-D mask 触发 ValueError |
| `test_shape_mismatch` | batch / seq 维度不一致触发 ValueError |
| `test_device_mismatch` | 跨 device tensor 触发 ValueError |
| `test_nan_inf` | NaN / +Inf / -Inf 触发 ValueError |
| `test_dtype_fidelity` | 输出 dtype / device 与输入一致（fp32 / fp16 / bf16） |
| `test_autograd` | backward 通路完整，grad shape 正确，选中位置梯度非零，未选中位置梯度为零 |
| `test_determinism` | 相同输入两次调用 `torch.equal` 成立 |
| `test_module_exports` | `__all__` 严格仅含 `pool_last_valid_token`，无 mean / first / cls / max pooling 函数 |

## 7. 还剩下的 3 个最大风险点

### 1. `device mismatch` 测试仅覆盖 meta vs cpu

真实 GPU 环境下 `cuda:0` vs `cpu` 的测试未在 CI 中执行。代码逻辑已覆盖该分支，但无 GPU CI 验证。风险等级：**低**（逻辑简单，meta 已触发分支）。

### 2. `bf16` 平台兼容性依赖环境 skip

测试对不支持 bf16 的平台自动 skip，不影响正确性；但可能导致 bf16 相关 dtype fidelity 未实际运行。风险等级：**低**（fp32/fp16 已覆盖，bf16 行为与 fp16 一致）。

### 3. 下游误用 `last_idx` 为 relative index

本模块已采用绝对位置索引（Path A），但 Task 5+ 的下游代码若在其他地方重新计算 `sum(mask)-1` 或 `cumsum` 来推导 last-valid index，会在 left-pad 场景下产生**静默错误偏移**。

该风险必须由 **Task 6 `test_eval_head_parity.py`** 捕获：验证 validation 与 offline eval 使用完全相同的 prompt builder、chat template、token index、hidden-state index、classifier input tensor、probabilities。若下游出现 relative-index 回退，parity test 的 logits/probabilities 将不匹配，从而 fail-closed 阻断。

风险等级：**中**（依赖后续 Task 6 的 parity test 作为最终防线）。
