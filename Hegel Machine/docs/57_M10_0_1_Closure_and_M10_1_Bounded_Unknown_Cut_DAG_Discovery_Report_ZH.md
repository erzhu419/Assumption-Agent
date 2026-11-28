# Hegel `M10.0.1` 审计闭环、`M10.1` 面板辅助有界计算图发现与 `M10.1.1` 校准报告（`U01–U12`）

> **零硬编码派生常数声明（`U10`）**：本报告全部百分比、`K*`、置信区间、双口径零调用计数与结构门状态均由封存 JSON 实时计算并经一致性断言核验。

## 1. 校准后的执行摘要与主张范围（`U01 / U06 / U07 / U08`）

1. **实验性质与曝光溯源（`U01`）**：确认组八个 `Stage S` 源任务在流前中止包中已产生官方私判回执（`source_private_previously_exposed = True`），随后协议 `v1.1` 将公开 `52/20` CV 门统一至 `0.038` 并增加 `1` 次公开门源重试（`protocol_amended_after_source_results = True`），中止时后继流幕运行数为 `0/192`（`downstream_streams_previously_exposed = False`）。因此本实验准确界定为 **“源阶段曝光并修订后冻结的前瞻在线流验证”**（开发树 `67c15bfd4b56...` vs 确认树 `de6853799209...`，单次运行内部零改码）。
2. **信息先验与对照边界（`U06`）**：`M10.1` 取消了强制 `(calibrate_upstream, solve_downstream)` 双函数签名，实现了真实可执行 `K=3/4` 分解、共享前缀单次计算（`a04`）与 `5/5` 免节点重合成新组合；但实际付费提示词包含 `panel_a -> panel_b -> panel_c['v1']/['v2'] -> readout` 分解示例与观测编辑映射，且单体/模块臂调前探针上限分别为 `2` 与 `5`，故准确表述为 **“取消强制两阶段接口、在校准面板与领域提示辅助下的有界图提议与外部模块粒度消融”**（完全去提示与同探针上限对照由 `M10.2` 独立检验）。
3. **统计与非劣裁决（`U07 / U08`）**：`Hegel` 私判通过 **`61/64`（`95.3%`）** vs `Mono` `60/64`（`93.8%`）与 `Fixed2` `60/64`（`93.8%`）；零求解轮通过 `19 / 18 / 22`，**整幕全角色零模型调用通过 `17 / 10 / 11`**（`2 / 8 / 11` 幕含已计入 `Scope-1` 的守卫维护调用）。相对 `Mono`：`Scope-1` 净变 **`+3.32%`**（`p = 0.8281`，非统计等效证明），质量差 `+1.56pp` 的自举 95% CI 为 **`[-6.00pp, +9.00pp]`**（下界 `-6.25pp < -5.00pp`，**相对强单体质量非劣未确证**，`K* = not_reached`）。相对 `Fixed2`：`Scope-1` 点估计净变 **`-15.32%`**（`4/8` 簇更省，`p = 0.1484`，CI 跨零），质量自举 95% CI **`[-3.00pp, +6.00pp]`**（满足 `-5.00pp`，`t(7)` 下界 `-5.13pp` 略越界，有限 8 步 `K* = 4`）。
4. **完整研发总账**：开发集 `829` 调用 / `18,068,350 v2` + 流前中止 `90` 调用 / `2,212,443 v2` + 完成评估 `1878` 调用 / `41,196,913 v2` = **`2,797` 次逻辑调用 / `61,477,706 v2`**（`0` 未知用量）。

## 2. 确认组（`n = 8`）三臂总表（并列零求解与整幕全角色零调用，`U08`）

| 臂 | 私判通过 | 零求解通过 | **整幕全角色零调用** | 总调用数 | 求解轮数 | Scope-1 总 v2 | stream_solve | dag_prop | dag_synth | dag_failed | dag_merge | guard_ind | 容器累计秒 | 最大 K/d/fo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicArchive_Online` | **60/64 (93.8%)** | 19/64 | **17/64** | 546 | 491 | **12,076,813** | 11,094,551 | 0 | 0 | 0 | 0 | 982,262 | 529.36s | 1/1/0 |
| `Fixed_TwoStage_Modular_Online` | **60/64 (93.8%)** | 18/64 | **10/64** | 671 | 517 | **14,735,484** | 2,056,530 | 802,806 | 6,597,620 | 1,055,805 | 2,231,646 | 1,991,077 | 981.00s | 2/2/1 |
| `Hegel_UnknownCut_GraphDiscovery_Online` | **61/64 (95.3%)** | 22/64 | **11/64** | 581 | 440 | **12,477,436** | 1,893,758 | 681,869 | 4,481,736 | 723,732 | 2,935,017 | 1,761,324 | 898.96s | 4/3/2 |

## 3. 分层配对推断（修正分母与三项独立判定，`U07 / U10`）

- 精确分层校核：A层相对 `Mono` 为 **`-14.06%`**（修正旧硬编码 `-12.1%`）；B层（`b01/b02` 新静态拓扑 + `b03/b04` 新到达次序/切换）相对 `Fixed2` 为 **`-18.66%`**（修正旧 `-25.3%`），相对 `Mono` 为 **`+21.12%`**（通过率同为 `31/32`）；去除 `a01` 后的 `n=7` 多节点簇相对 `Mono` 为 **`-0.31%`**（`p = 0.9844`）。

| 对比 | 分层 | 通过幕数 (H vs Base) | 质量差 Bootstrap 95% CI | 成本净变 (总差 v2) | 成本双侧 `p` | 成本均值差 Bootstrap 95% CI | 成本优效 / Bootstrap非劣 |
|---|---|---|---|---:|---:|---|---|
| Hegel − 通用单体 | 全体汇总 (n=8) | `61/64` vs `60/64` | `[-6.00pp, +9.00pp]` | `+3.32%` (`+400,623`) | `0.8281` | `[-358642, +454756]` | `False` / `False` |
| Hegel − 固定两模块 | 全体汇总 (n=8) | `61/64` vs `60/64` | `[-3.00pp, +6.00pp]` | `-15.32%` (`-2,258,048`) | `0.1484` | `[-602847, +21359]` | `False` / `True` |
| Hegel − 通用单体 | A层已见拓扑族 (n=4) | `30/32` vs `29/32` | `[-9.00pp, +19.00pp]` | `-14.06%` (`-859,324`) | `0.5000` | `[-710346, +280684]` | `False` / `False` |
| Hegel − 固定两模块 | A层已见拓扑族 (n=4) | `30/32` vs `30/32` | `[-9.00pp, +9.00pp]` | `-10.25%` (`-600,083`) | `0.5000` | `[-486101, +186060]` | `False` / `False` |
| Hegel − 通用单体 | B层未见新拓扑/新次序 (n=4) | `31/32` vs `31/32` | `[-9.00pp, +9.00pp]` | `+21.12%` (`+1,259,947`) | `0.3750` | `[-199074, +829048]` | `False` / `False` |
| Hegel − 固定两模块 | B层未见新拓扑/新次序 (n=4) | `31/32` vs `30/32` | `[+0.00pp, +9.00pp]` | `-18.66%` (`-1,657,965`) | `0.5000` | `[-880601, +51618]` | `False` / `True` |
| Hegel − 通用单体 | 不可分单体负控 (n=1) | `8/8` vs `6/8` | `[+25.00pp, +25.00pp]` | `+45.36%` (`+435,032`) | `1.0000` | `[+435032, +435032]` | `False` / `False` |
| Hegel − 固定两模块 | 不可分单体负控 (n=1) | `8/8` vs `7/8` | `[+12.00pp, +12.00pp]` | `+1.28%` (`+17,675`) | `1.0000` | `[+17675, +17675]` | `False` / `False` |
| Hegel − 通用单体 | 仅多节点非单体簇 (n=7) | `53/56` vs `54/56` | `[-7.00pp, +4.00pp]` | `-0.31%` (`-34,409`) | `0.9844` | `[-455120, +444035]` | `False` / `False` |
| Hegel − 固定两模块 | 仅多节点非单体簇 (n=7) | `53/56` vs `53/56` | `[-5.00pp, +5.00pp]` | `-17.04%` (`-2,275,723`) | `0.1406` | `[-673408, +17248]` | `False` / `True` |

## 4. 逐簇明细与精确 `K*`（`a04=2, a02=not_reached, b01=1` vs Mono）

| 簇 ID | 分层 | 隐藏拓扑类 | Mono 通过 (v2) | Fixed2 通过 (v2) | **Hegel 通过 (v2)** | H−Mono% | H−Fixed2% | 零求解 | 最大 K/d/fo | 共享 exec=1 | 逐簇 `K*` (vs M / F) |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| `conf_m101_a01` | `stratum_a_seen_topology` | `monolithic_coupled` | 6/8 (959,148) | 7/8 (1,376,505) | **8/8 (1,394,180)** | `+45.36%` | `+1.28%` | 4/8 | K=2,d=2,fo=1 | — | `not_reached / not_reached` |
| `conf_m101_a02` | `stratum_a_seen_topology` | `chain_k3_variable_cut` | 7/8 (1,890,301) | 7/8 (1,697,377) | **6/8 (1,248,092)** | `-33.97%` | `-26.47%` | 1/8 | K=3,d=3,fo=2 | YES | `not_reached / not_reached` |
| `conf_m101_a03` | `stratum_a_seen_topology` | `chain_k3_reversed_cut` | 8/8 (1,480,017) | 8/8 (1,251,909) | **8/8 (1,606,353)** | `+8.54%` | `+28.31%` | 3/8 | K=3,d=3,fo=2 | YES | `not_reached / not_reached` |
| `conf_m101_a04` | `stratum_a_seen_topology` | `diamond_k4_shared_prefix` | 8/8 (1,782,275) | 8/8 (1,526,709) | **8/8 (1,003,792)** | `-43.68%` | `-34.25%` | 4/8 | K=4,d=3,fo=2 | YES | `2 / 1` |
| `conf_m101_b01` | `stratum_b_unseen_topology` | `chain_k4_deep_cascade` | 8/8 (1,747,227) | 7/8 (2,369,553) | **8/8 (1,272,945)** | `-27.14%` | `-46.28%` | 3/8 | K=3,d=3,fo=2 | YES | `1 / 5` |
| `conf_m101_b02` | `stratum_b_unseen_topology` | `asymmetric_skip_dag_k4` | 8/8 (1,464,644) | 7/8 (3,037,549) | **7/8 (2,372,955)** | `+62.02%` | `-21.88%` | 2/8 | K=3,d=3,fo=1 | — | `not_reached / 6` |
| `conf_m101_b03` | `stratum_b_unseen_topology` | `diamond_k4_coupled_first` | 8/8 (1,204,055) | 8/8 (1,868,740) | **8/8 (1,953,840)** | `+62.27%` | `+4.55%` | 3/8 | K=4,d=3,fo=2 | YES | `not_reached / not_reached` |
| `conf_m101_b04` | `stratum_b_unseen_topology` | `hybrid_mono_to_split_shift` | 7/8 (1,549,146) | 8/8 (1,607,142) | **8/8 (1,625,279)** | `+4.91%` | `+1.13%` | 2/8 | K=4,d=3,fo=2 | YES | `not_reached / not_reached` |

## 5. 预登记结构验证门逐项自动裁决表（`U06`）

| 结构门 ID | 预登记判据 | 实测记录 | 裁决 |
|---|---|---|---|
| `monolithic_k1_non_fragmentation` | Retain K=1 on monolithic_coupled (`conf_m101_a01`) without decomposing | strict_pass=8/8, max_active_nodes_observed=2 (adopted K=2 on S5/S7/S8) | **`FAIL`** |
| `multi_stage_unknown_cut_discovery_and_reuse` | Discover K>=3 on K=3/K=4 cascades and pass unseen combinations without node re-synthesis | K>=3 discovered on a02/a03/b01; 5/5 solver-free unseen node compositions passed (with paid guard maintenance calls; prompt had panel scaffolding) | **`PARTIAL_PASS_WITH_SCAFFOLDING`** |
| `shared_prefix_diamond_and_skip_exec_count_1` | Verify max_fanout==2 and exec_count==1 on BOTH diamond_k4 (`a04`) and asymmetric_skip_dag_k4 (`b02`) | a04: max_fanout=2, shared_exec_1=True; b02: max_fanout=1, shared_exec_1=False | **`FAIL`** |
| `cut_withdrawal_and_old_domain_preservation` | 4-link external FK withdrawal receipt + old separable domain preserved on colliding revisits | 2/6 Hegel tickets code-identical (`a02/S2, a04/S6`), 4/6 had prior/fail code-version mismatch; `a02/S7` revisit missed private threshold due to empty-guard tiebreak | **`FAIL`** |
| `plain_dict_substrate_parity_ablation` | 100% candidate SHA-256 routing parity on frozen Step-06 checkpoints | 16/16 (100.0%) | **`PASS`** |

## 6. 开发集（`n = 4`）摘要与 `M10.1.1` 机制故障根因指向

| 臂 | 私判通过 | 零求解通过 | **整幕全角色零调用** | 总调用数 | 求解轮数 | Scope-1 总 v2 | stream_solve | dag_prop | dag_synth | dag_failed | dag_merge | guard_ind | 容器累计秒 | 最大 K/d/fo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicArchive_Online` | **31/32 (96.9%)** | 10/32 | **9/32** | 267 | 243 | **5,978,326** | 5,537,949 | 0 | 0 | 0 | 0 | 440,377 | 245.78s | 1/1/0 |
| `Fixed_TwoStage_Modular_Online` | **29/32 (90.6%)** | 12/32 | **9/32** | 282 | 226 | **6,070,822** | 1,301,544 | 302,408 | 2,306,232 | 640,061 | 905,670 | 614,907 | 409.47s | 2/2/1 |
| `Hegel_UnknownCut_GraphDiscovery_Online` | **31/32 (96.9%)** | 15/32 | **11/32** | 233 | 179 | **4,862,101** | 624,677 | 261,120 | 1,165,766 | 292,806 | 1,929,125 | 588,607 | 367.02s | 4/3/2 |

- **关键机制诊断（详见 `docs/57b_*`）**：`M10.1` 中 Hegel 守卫开销（`1,761,324 v2`）有 **`86` 次调用 / `1,663,837 v2`** 来自同源正证人重叠导致的不可满足互斥约束（`0` 次采纳），另有 `a03/S1` 签票时序异常触发的 `10` 次调用 / `193,777 v2` 回退开销；这些实现故障不能归为图发现固有资本支出，已在 `M10.1.1` 全量修复并进入 `M10.2` 验证。
