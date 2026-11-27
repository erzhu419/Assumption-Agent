# M9.8.1 审计闭合报告：证据契约加固、容器化双普通表归因与前缀锁定双生对照（`0` 次付费 LLM 调用）

- **封存证据目录**：`docs/evidence/m9_8_1_audit_closure_and_twotable_attribution_20261128/`（`SHA256SUMS.txt` 全量核验通过；全部历史 `M9.4`–`M9.8` 封存证据 `100%` 原样只读保留）
- **付费模型调用数**：`0`（全部在断网容器 `hegel_clean_scipy_sandbox:v1` 中真实执行，`evidence_provenance = executed`）
- **目标范围**：在进入 `M9.9` 冻结前，完整闭合 `Hegel_M9_8_Audit` 登记的 `R01–R16`。

## 一、P0 证据语义、基础设施异常隔离与内容寻址加固（`R04, R05, R06, R07`）

1. **`R04`（`r2_diag >= 0.96` 未执行快捷分支与组件直接执行证据分离）**：
   - `diagnose_m98_factored_conflict_locus` 在未启动组件容器时显式返回 `component_probe_executed = False`、`execution_status = "not_executed"`、`evidence_origin = "compatibility_inference"`、`chosen_up_holdout_rmse = None`（绝不伪报 `0.0`），且 `run_m98_equal_capability_pre_llm_trial` 绝不向未直接执行的组件写入 `tested_passed` 证人。
   - **`M9.8` 全部 `16` 个 `C1_down`/`C2_down` 实例补充隔离容器实测（`02_m98_r04_16_upstream_direct_verification_and_r11_probe_ledger.json`）**：`16/16` 独立组件容器均 `completed` 且留出集最大 `RMSE = 0.00019837`（远低于 `0.03` 阈值，总耗时 `9,734.25 ms`），同时将原 `M9.8` 记录严格重分类为 `compatibility_inference`。
2. **`R05`（主探针超时与定位容器 `rc=125` / `TimeoutExpired` 基础设施故障隔离）**：
   - 主组合预检触发 `INFRA_FAILURE` 时直接终止科学归因，上下游组件证人保持 `None`。
   - 定位容器 `TimeoutExpired`、`rc=125`、守护进程不可达或 JSON 异常一律返回 `locus = "unknown_infra"`、`chosen_up_holdout_rmse = None`（彻底废除 `999.0` 哨兵值），`0` 科学负例污染（`tests/test_m9_8_1_audit_closure_suite.py`）。
3. **`R06` 与 `R07`（规范 `ecov_` 回执校验、冷恢复缺摘要拒绝与外键验真）**：
   - `EmpiricalCoverageReceipt.verify_integrity()` 无条件校验 29 位规范 `ecov_` 摘要（`ecov_invalid` 立即拒绝）、数组长度一致性及 `supported_on_witnesses` 的 `passed == evaluated >= 1` 约束。
   - `ImmutableBranchLineageController` 在当前 schema 下对缺失 `state_sha256` 或 `witness_content_sha256` 严格失败闭合，校验 `binding_id` 与 `solver_code_sha256` 外键，并将旧 schema 降级记录限制为 `unverified_legacy`（`0` 分叉授权与 `0` 覆盖授权）。

## 二、容器化双普通表已付前缀归因与跨环境 Pickle 复现（`R01, R11, R12`）

| 指标（`01_m98_executed_two_plain_tables_paid_prefix_replay.json`） | 实测结果 |
|---|---:|
| 实验类别标记 | `paid_prefix_representation_control`（`0` 图 / `0` 注册表 / `0` 谱系控制器） |
| 宿主 `SHA256(public_batch.pkl)` 与封存 `M9.8` 逐字节一致数（`Python 3.10 / NumPy 2.2.6`） | **`56/56`**（并新增 `canonical_numeric_sha256`） |
| 上下游特征、选中模块、模式与组合代码 SHA 一致数 | **`56/56`** |
| Docker 公共门控**接受**候选数（`R0, U_iso, M_comb1, M_comb2`） | **`32/56`**（与 `Hegel_FactoredMechanismGraph` `100%` 一致） |
| — 公共接受候选的零 LLM 私判 `strict_pass` | **`32/32`（`100.0%`）** |
| Docker 公共门控**拒绝**候选数（`C1_down: 8, C2_down: 8, C_up: 8`） | **`24/56`**（原在线运行中进入 LLM 修复） |
| 单独核算的原 `M9.8` 已付结构形成成本 | `310` 次持续流调用（`6,177,898 v2`）+ `73` 次 `Stage S` 调用（`1,193,405 v2`） |
| `8` 次 `C_up` 定位容器探针重跑参考墙时（`R11`） | `4,967.12 ms`（均值 `620.89 ms`；`8/8` RMSE 与封存值一致） |

- **`R12` 跨环境 Pickle 哈希差异根因**：在原 `M9.8` 宿主运行时（`Python 3.10.12 / NumPy 2.2.6`）下，全部 `56/56` 个封存 `public_batch.pkl` 哈希逐字节精确一致；跨版本环境（如 `Python 3.12.14 / NumPy 2.3.5`）因 CPython/NumPy `pickle` 帧与 `dtype` 元数据字节布局变化改变原始字节摘要，而底层 IEEE-754 双精度数组值 `100%` 一致（`56/56` 特征与代码哈希匹配）。新增 `compute_canonical_numeric_item_sha256` 提供了与 Python/NumPy 版本无关的规范数值摘要。
- **`R01` 诚实归因结论**：在共享给定 `U/D` 接口与已付前缀模块/守卫条件下，双普通条件表可 `56/56` 复现候选程序并达成 `32/32` 零轮私判通过；故 `M9.8` 证明的是**模块化分解修复与跨因子组合相对单体表的收益**，而**全等工具权限下的自主分解形成与非可分 `U–D` 交互撤回**由下一阶段 `M9.9` 正面验证。

## 三、字节级全等 `(R0, U_iso)` 上的严格在线前缀锁定与固定无历史策略对照（`R08`）

| 对照策略（`03_m98_r08_prefix_locked_and_fixed_policy_twin_ablations.json`，每组 `16` 个 episode） | 使用检查点 | `R0` 通过 | `U_iso` 通过 | 总私判 `strict_pass` | 私判 `wall_ms` |
|---|---|---:|---:|---:|---:|
| `online_prefix_locked_history_aware` | `step_01` (`R0`), `step_04` (`U_iso`) | `8/8` | `8/8` | **`16/16`（`100.0%`）** | `86,397.51` |
| `online_prefix_locked_history_masked`（遮蔽日志 `[]`） | `step_01` (`R0` -> `D1`), `step_04` (`U_iso` -> `D2`) | `0/8` | `0/8` | **`0/16`（`0.0%`）** | `86,272.58` |
| `fixed_current_only_D0_policy`（固定 `u0 o d0`） | 前缀校准 `u0 o d0` | `8/8` | `0/8` | **`8/16`（`50.0%`）** | `86,075.54` |
| `fixed_current_only_D1_policy`（固定 `u0 o d1`） | 前缀校准 `u0 o d1` | `0/8` | `8/8` | **`8/16`（`50.0%`）** | `86,359.05` |
