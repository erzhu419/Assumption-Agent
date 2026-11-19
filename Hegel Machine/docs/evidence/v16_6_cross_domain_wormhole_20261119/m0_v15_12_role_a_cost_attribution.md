# M0 — V15 12 道 Role-A 目标费用归因表（基于已保存 `deliverable_4` 原始事件对账）

- 样本范围：6 个独立前瞻簇 × `{A1, A2}` = **12 道 `Role_A_compatible_focal_memory` 目标**（零强行分摊；7 桶之和与总 `compute_cost` 残差计入 `unallocated_delta_compute`）。
- 严格通过率：`K = 12/12`，`Ordinary = 12/12`。
- 配对均值差异（`K - Ordinary`）：`Δcompute = +59138.7`，`Δobs = +61.83`，`unallocated = +100.8`。

## 七桶均值分解（`K - Ordinary`）

| 计费桶 | 平均 `Δcompute (K - Ordinary)` | 诊断结论 |
|---|---:|---|
| `1_initial_object_and_premise_audit_compute` | `+1691.2` | 实测事件直接对账 |
| `2_historical_instance_rescoring_compute` | `+0.0` | 实测事件直接对账 |
| `3_discriminating_experiment_planning_compute` | `+2052.0` | 实测事件直接对账 |
| `4_parameter_rebinding_and_mechanism_search_compute` | `+9663.3` | 实测事件直接对账 |
| `5_confirmation_and_selection_scoring_compute` | `+44858.9` | 实测事件直接对账 |
| `6_branch_reconciliation_and_index_compute` | `+836.8` | 实测事件直接对账 |
| `7_on_demand_or_raw_history_reconstruction_compute` | `-64.4` | 实测事件直接对账 |
| `unallocated_within_episode_compute` | `+100.8` | 未细分残差（不强行归因） |

## 逐题明细（12 题）

| 簇 | 槽位 | `k` | `K pass` | `Ord pass` | `K compute` | `Ord compute` | `Δcompute` | `Δbucket_4 (search/rebind)` | `Δbucket_6 (reconcile/index)` | `Δobs` |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| `prospective_v15_cluster_1051000` | `A1` | 1 | True | True | 282414 | 142004 | +140410 | +19313 | +6649 | +114 |
| `prospective_v15_cluster_1051000` | `A2` | 4 | True | True | 291248 | 58396 | +232852 | +51979 | +4 | +70 |
| `prospective_v15_cluster_1052000` | `A1` | 3 | True | True | 136154 | 139339 | -3185 | +1607 | +4 | +44 |
| `prospective_v15_cluster_1052000` | `A2` | 6 | True | True | 65451 | 56153 | +9298 | +4554 | +3 | +48 |
| `prospective_v15_cluster_1053000` | `A1` | 2 | True | True | 71050 | 57594 | +13456 | +4041 | +3 | +46 |
| `prospective_v15_cluster_1053000` | `A2` | 5 | True | True | 104338 | 105925 | -1587 | +4681 | +3 | +48 |
| `prospective_v15_cluster_1054000` | `A1` | 1 | True | True | 73157 | 58492 | +14665 | +3869 | +3360 | +48 |
| `prospective_v15_cluster_1054000` | `A2` | 4 | True | True | 66608 | 58297 | +8311 | +4654 | +3 | +48 |
| `prospective_v15_cluster_1055000` | `A1` | 3 | True | True | 368772 | 99000 | +269772 | +6559 | +4 | +136 |
| `prospective_v15_cluster_1055000` | `A2` | 6 | True | True | 59150 | 54055 | +5095 | +2984 | +3 | +46 |
| `prospective_v15_cluster_1056000` | `A1` | 2 | True | True | 71154 | 60559 | +10595 | +3026 | +3 | +46 |
| `prospective_v15_cluster_1056000` | `A2` | 5 | True | True | 69977 | 59995 | +9982 | +8693 | +3 | +48 |
