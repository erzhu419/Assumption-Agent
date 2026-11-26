# `M9.3.1` 勘误、在线 `L2` 代码保留重分析与确定性审计收口附录（`0` 次 LLM 调用）

- **附录范围**：在不修改已封存 `M9.3` 证据包（[`m9_3_online_source_freeze_lifecycle_20261126`](../evidence/m9_3_online_source_freeze_lifecycle_20261126/m9_3_online_lifecycle_master_summary.json)）的前提下，完成全部统计/归因勘误与生产运行时收口（`F01`–`F10`）
- **封存证据目录**：[`docs/evidence/m9_3_1_audit_closure_and_regrade_20261126/`](../evidence/m9_3_1_audit_closure_and_regrade_20261126/SHA256SUMS.txt)
- **新增在线 LLM 调用**：**`0`**（纯确定性代码收口、重判分与反例控制校验）

---

## 1. 对 `docs/48` 与 `docs/49` 的关键科学勘误与补正

1. **此前漏报的关键正结果（在线编译 `Stage S` `L2` 模块的真实调用与 AST 静态保留）**：
   - `docs/48` 当时仅汇总了 `L1` 原语计数（`v7` 旧 `L1` 调用 `3` 次、新编译 `L1` 调用 `0` 次），漏掉了新编译在线 `L2` 模块的真实复用！
   - 对 `04c_full_skill_activations.jsonl` 与提交代码的 AST 归一化重审计（[`m9_3_source_aware_l1_l2_adoption_funnel.json`](../evidence/m9_3_1_audit_closure_and_regrade_20261126/m9_3_source_aware_l1_l2_adoption_funnel.json)）证明：
     - `E`（`executable_source_reuse`）在 `2/8` 个后继 episode 中共执行新编译 `Stage S` `L2` 函数 **`1,021` 次**（其中 `rep_01/suc01_C` 调用 `autogenous_hereditary_spectrum_identification` **`1,018` 次**，`rep_02/suc02_T` 调用 **`3` 次**），并在 `Stage R`（`rec02_R`）调用 **`3` 次**。
     - 在 `E / rep_01 / suc01_C` 中，宿主机 AST 静态打包器在第 4 轮将 `align_reference_panel` 与 `hereditary_design` 内联打包进最终交付文件，归一化 AST 比对确认 **`2/2` 个函数体精确保留且从主入口可达**。
2. **基线 `B` 的精确定义**：
   - `M9.3` 中的 `B`（`bare_no_source_reuse`）精确身份为 **`prior_frozen_v7_no_source_growth`**（挂载已有 `v7` 森林但不接收 `Stage S` 新增资产），而非零资产裸模型基线。
3. **Provider Attempt 与 Token Proxy 核算精度（`F06`）**：
   - `M9.3` 的 `330` 次为逻辑调用数（`324` 次 episode 决策 + `6` 次 `L2/L1` 编译器调用），实际触发 **`>= 333` 次 provider 尝试**（`3` 次约 `360s` CLI 超时重试带 `usage_unknown = true`）。
   - 因 `K` 臂含 `1` 次 `360s` 未知用量超时（`rep_01/suc02_C` 第 6 轮），`E - K = +18,076` proxy 严格属于 **`known-only` 口径**；而 `E` 对 `B`（`1,134,884` vs `1,259,035`，`−124,151` / `−9.86%`）双臂均为 `0` 未知用量，比较完全精确。
   - `src01/01b_provider_attempts/` 中多出的 `3` 个未匹配文件（`010..012`，`88,471` proxy）源于共享目录 `glob` 归档，未混入 `00_episode_summary.json`（`385,093`）；`M9.3.1` 已引入每次调用唯一 UUID 与严格白名单拷贝隔离。
4. **`suc01_T` Verifier CPU 耗时归因修正**：
   - `suc01_T` 平均耗时下降主要来自 `rep_01`（`0.044s` vs `0.487s`，`0` 次技能调用），而非 `rep_02` 的 `3` 次 `v7` `L1` 调用（`0.160s` vs `0.049s`）。
5. **`19` 条已封存 `M9.3` Nogood 重分类（`F03`）**：
   - `M9.3` 中 `validate_public_schema.py` 仅校验公开输出格式，不应将 `19` 条科学/数值残差类 nogood 标记为科学已解决；现已在 [`a5_scientific_probe_vs_schema_and_19_nogood_errata.json`](../evidence/m9_3_1_audit_closure_and_regrade_20261126/a5_scientific_probe_vs_schema_and_19_nogood_errata.json) 中统一重分类为 `scientific_resolution_not_established_by_recorded_receipt`。
6. **`Track A`（`m8_persistent_variational_planner.py`）真实实现数学口径说明**：
   - 当前 `Track A` 代码实现的是尺度协变经验贝叶斯 `NIG` 对数边缘似然（`-log_marginal_likelihood - log_p0`）、调整 `R^2`/尾部集中度残差诊断排序以及预测方差/成本比主动探针排序，报告中严格按真实代码公式表述，不夸大为未接入主循环的 `H_leaf x Rao-score` 或精确熵 `BALD`。

---

## 2. `M9.3.1` 确定性生产收口一览（`F01`–`F10`）

| 编号 | 生产代码收口内容 | 审计回执 |
| :--- | :--- | :--- |
| **`F01`** | `consolidate_autogenous_episode_into_forest` 支持显式 `source_node_id`、`source_version`、`violated_component_id`、`preserved_component_ids`，将反例护栏同步写入 `forest.nodes[parent].claim_contract.preconditions` 且完全不污染无关节点 `node_auto_07`。 | [`c1_c2_component_bound_bifurcation_and_guard_controls.json`](../evidence/m9_3_1_audit_closure_and_regrade_20261126/c1_c2_component_bound_bifurcation_and_guard_controls.json) |
| **`F02`** | 三值（`TRUE / FALSE / UNKNOWN`）可执行 `ObservablePreconditionGuard`：同时检查历史与保载段屈服超应力及 `residual_plastic_state`（防止卸载态误判复位），并用精确二维加性残差比识别真正不可分离交叉势（非零加性势 `V(x,y)=a(x)+b(y)` 残差严格为 `0.0`）。 | [`c1_c2_component_bound_bifurcation_and_guard_controls.json`](../evidence/m9_3_1_audit_closure_and_regrade_20261126/c1_c2_component_bound_bifurcation_and_guard_controls.json) |
| **`F03`** | 严格拆分 `PublicSchemaReceipt` 与 `ScientificProbeReceipt`，执行前重验 `/holdout_task/validate_public_schema.py` SHA-256，新增宿主机只读科学残差/对偶间隙探针执行器 `execute_host_trusted_scientific_probe`。 | [`a5_scientific_probe_vs_schema_and_19_nogood_errata.json`](../evidence/m9_3_1_audit_closure_and_regrade_20261126/a5_scientific_probe_vs_schema_and_19_nogood_errata.json) |
| **`F04`** | 收紧全部输出与下游交接校验（拦截 `src01` `np.zeros(7)`、`rec01_R` `np.array([np.nan])`、`suc01_T` 割线/首达时刻不一致、`Chain 2` `weights` 与 `kkt_certificate['weights']` 不一致）；重判分证明封存 `M9.3` 的 **`28/28`** 份交付全部真实通过！ | [`m9_3_28of28_tightened_handoff_regrade.json`](../evidence/m9_3_1_audit_closure_and_regrade_20261126/m9_3_28of28_tightened_handoff_regrade.json) |
| **`F05`** | 多标签分源分层漏斗 `classify_source_aware_adoption_funnel`，严格区分 `new_source_L2`、`new_source_L1` 与 `prior_v7_L1`。 | [`m9_3_source_aware_l1_l2_adoption_funnel.json`](../evidence/m9_3_1_audit_closure_and_regrade_20261126/m9_3_source_aware_l1_l2_adoption_funnel.json) |
| **`F06`–`F10`** | UUID 级 attempt 账本 + `_LLM_ERROR` 统计 + 白名单归档 + 编译器 attempt 回执（`F06`）；`L1` 可选 KKT 对偶见证生成（`F07`）；双独立 `python3 -I` 进程确定性与输入不可变门禁 + 容器 `.Image` 校验 + 冷启动 `ensure_runtime_qualification_context`（`F08`）；独立开发种子 `88512..88621`（`F09`）；启动生成 `/holdout_task/public_asset_index.json` + AST 纯输入卫语句剥离升格（`F10`）。 | [`f06_f07_f08_f09_ast_and_hipporag_controls.json`](../evidence/m9_3_1_audit_closure_and_regrade_20261126/f06_f07_f08_f09_ast_and_hipporag_controls.json) |
