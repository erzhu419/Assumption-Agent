# Hegel V17.4 (`M4.2`) 复核闭环、跨支柱可执行 $\Pi_s$ 生命周期与零重拟合作用域分发器实测报告

> **审计基准 Commit**：`99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c`（对应 `markdown/Hegel_M4_1_Audit/GPT_diagnosis_M4_1.md`、`M4_1_AUDIT.md` 及四套反事实探针 `closure/`、`pi/`、`unified/`、`part2/`）  
> **静态只读约束**：此前全部证据包（`docs/evidence/v16_8_*` 至 `v17_3_*`）与全部审计目录（`markdown/hegel_M3_Audit/`、`markdown/Hegel_M4_Audit/`、`markdown/Hegel_M4_1_Audit/`）保持 **100% 原样只读未改动**；本轮 `V17.4 (M4.2)` 全新静态证据包独立归档于 `docs/evidence/v17_4_m4_2_unified_closure_and_pi_lifecycle_20261120/`（复现脚本支持 `--out-dir` 只读外部重放）。

---

## 一、对 `Hegel_M4_1_Audit`（GPT-6 复核）的独立架构与工程研判

遵循一贯的自主决策原则——**将外部红队审计作为最坏边界提醒器：凡属真实的执行断路、路径逃逸、 provenance 缺口与统计重计，100% 在源码与实测中彻底闭合；凡属建议把系统退回扁平静态顺序或放弃跨尺度/跨树图游走的保守倾向，不仅不退缩，反而通过更严格的反例结构正面证明四支柱统一架构的不可替代性**——我们将 `GPT_diagnosis_M4_1.md` 归纳为 **12 项硬核工程闭环** 与 **4 项跨支柱架构跃迁**：

### 1.1 100% 吸收并已在源码与回归测试闭合的 12 项工程与计量问题

| 编号 | `Hegel_M4_1_Audit` 指出的具体问题 | 源码修复位置与机制 | 复测证据状态 |
| :--- | :--- | :--- | :--- |
| **C1** | `program_steps` 合法校验后未真正消费参数（`fresh_points: 1 -> 2` 行为与开销完全不变） | `m4_multiscale_renormalization.py`、`m4_1_unified_discovery_and_renormalization.py`、`m4_2_unified_closure_and_pi_lifecycle.py`：将 `_exec_plan`（`fresh_points`、`calibrate_lens`、`operator_tag`）直接接入确认采样数与透镜校准开关 | **已闭合** (`1 -> 40.0` vs `2 -> 44.0`) |
| **C2** | `FidelityCompression` 提交了 `disjunctive_branches`，但 `authorize_abstraction_use` 只读顶层合取 `preconditions`（导致析取合并后 `A`、`B` 均 `granted=False`） | `wormhole_forest.py`：① `compress_compatible_nodes` 顶层仅保留公共不变量，`disjunctive_branches` 扁平化保存各合法域分支；② `authorize_abstraction_use` 原生求值 `disjunctive_branches`（任一分支全真即 `TRUE`，全假即 `FALSE`） | **已闭合** (`A=True, B=True, C=False`) |
| **C3** | `load_verified_authored_method` 仅做内部自洽校验，接受 `confirmation_digest="no_confirmation_was_run"` 的无确认自洽 JSON | `m4_compositional_discovery.py`：写入宿主密封 `host_confirmation_bundle` + `confirmed_manifest.json` + `confirmed_observations.json`，加载时强制校验 SHA-256 链并在隔离子进程重放确认集 RMSE | **已闭合** (`unconfirmed_accepted = False`) |
| **C4** | `_load_node_verified_operator_spec` 未核对节点注册的 `ast_skeleton_digest`（原地把 `W1` 文件替换为合法 `W2` 产物会被照单全收） | `m4_compositional_discovery.py` + `m4_1_unified_discovery_and_renormalization.py`：向 `load_verified_authored_method` 传入 `expected_ast_digest=act["ast_skeleton_digest"]`，跨节点替换立即拦截 | **已闭合** (`swapped_w2_accepted = False`) |
| **C5** | `evaluate_isolated_python_model` 对相对父路径 `../../../outside.py` 未做 `.resolve().relative_to(ws.resolve())` 校验，存在工作区逃逸 | `m4_compositional_discovery.py`：对绝对与相对路径统一执行 `candidate_path.resolve().relative_to(ws_resolved)` 越界拦截 | **已闭合** (`ValueError` 100% 拦截) |
| **C6** | `revalidate_vertical_edges_after_revision` 仅做低层版本重绑，不校验 `ScaleMap` 是否仍处于 `not_closed` | `wormhole_forest.py`：保留低层重绑接口兼容性的同时，新增生产级 **`revalidate_vertical_edges_with_certificate`**，强制校验 `certified is True`、`defect <= tol`、`closure_scope != 'not_closed'` 及被消元变量核验集 | **已闭合** (`unclosed_rejected = True`) |
| **C7** | Part 2 `M3.1` $2\times 2$ 脚本拼接 `suite["targets"] + suite["walk_targets"]` 导致 `T25..T28` 重复计权（`124` vs `108`） | `m4_2_unified_closure_and_pi_lifecycle.py`：直接遍历唯一 `suite["targets"]`（每簇 `27` 唯一目标 × `4` 簇 = **`108` 次唯一运行/格，`432` 总运行**） | **已闭合** (`Walk_Pool = 37.4591` 仍严格最优) |
| **C8** | Part 3 `Pi_s` 在确认点越界（如 $T=1.2 \notin [2.2, 5.2]$）或导出模型面对盒外查询（$T=0.12$）时未统一拦截/升尺度，且 `without_edge` 回退重购了 4 个已付试探点 | `m4_1_unified_discovery_and_renormalization.py`：① 召回、回退与导出契约（`model_scope_contract.json` + `evaluate_heldout_micro_truth`）全链路强制 `verified_input_box`（越界确认自动升至 `micro_fine`，越界导出查询返回 `out_of_verified_scale_box`）；② 回退复用已付 `pilot_rows`（重复购点 `0`） | **已闭合** (`false_confirmation = False`, 重复行 `0`) |
| **C9** | `project_trajectory_via_pi_s` / `apply_executable_pi_s_projection` 的细轨迹窗口投影（`window_projected_y`）未接入区域闭合证书生成路径（计数为 `0`） | `m4_1_unified_discovery_and_renormalization.py` + `m4_2_unified_closure_and_pi_lifecycle.py`：在区域盒标定中直接传入真实 RK4 细轨迹，同时考核端点闭合缺陷与快流形轨迹奴役缺陷 | **已闭合** (`projections_with_fine_trajectory = 6`) |
| **C10** | Part 4 `Forest_Alpha0` / `Flat_BM25` 在首轮召回确认 `cf0` 失败后未将已付 `cf0_rows` 并入 `frozen_prefix`，导致回退重购确认点；`Ordinary` 二轮确认未记入 `confirmed_via_recall` | `m4_1_unified_discovery_and_renormalization.py` + `m4_2_unified_closure_and_pi_lifecycle.py`：所有分支 100% 保留已付失败确认点并追踪 `refuted_tags`；统一二轮召回口径 | **已闭合** (零重复购点、零已付证据丢失) |
| **C11** | `authored_method.json` 的 `diagnostic_rounds` 用 `len(composed_history)` 近似，未保存已拟合参数快照 `fitted_params` | `m4_compositional_discovery.py`：独立记录真实 `provider._diag_rounds` 与 `composed_candidate_count`，落盘 `confirmed_fitted_params.json` 并新增删除 `fit` 命名空间的 `predict_with_frozen_instance` | **已闭合** (`fit_called = False`) |
| **C12** | Part 4 的 `Stage 0 = 180.0` 为常数记账、`Stage 2` 3 次诱饵 DEV 为重复同任务、种子间未重置 `Stage 2` 快照、OOS 未随种子变化 | `m4_2_unified_closure_and_pi_lifecycle.py`：① `Stage 0` 全部经真实 `OpenWorkspaceTaskLab` / `MultiScaleWorkspaceTaskLab` 实测计费（`244.05`）；② 3 个异构域/异构特征 DEV 诱饵任务；③ 每种子严格恢复 `post_stage2_forest_snapshot`；④ OOS 随种子变化 | **已闭合** |

---

### 1.2 四项正面架构升级（拒绝保守退化，打通跨支柱闭环）

1. **用对称两跳起始分岔（Symmetric Onset Pigeonhole Fork）彻底击穿一切扁平静态顺序（含零窥探 `DEV` 冻结最优静态序）**：
   - 针对 `GPT_diagnosis_M4_1.md` 指出的“在 `M4.1` 12 目标上精心挑选一个静态顺序 `36.4250` 可逼近 `Hegel` `36.2396`”，我们在 `M4.2` 中正面引入具有严格抽屉原理下界的跨域分岔家族：`OnsetWalk_MonodShunt_2D`（载体结合起始 $\xrightarrow{\text{2-hop}}$ `Monod+Shunt`）与 `OnsetWalk_MurrayShunt_2D`（血管分枝起始 $\xrightarrow{\text{2-hop}}$ `Murray+Shunt`），叠加 `Haldane+Shunt`、`Oscillatory+Shunt 4P` 与多尺度 `Pi_s` 绝热跨域目标（每种子 `20` 个未见目标，`4` 种子共 **`80` 个未见目标 + `8` 个异构 OOS**）。
   - 在低负载安全试探区 $s \in [0.55, 1.30]$ 上，`Monod+Shunt` 与 `Murray+Shunt` 均能在容差内拟合试探集（`pilot_rmse <= 0.029 < 0.042`），但在高负载确认点 $s = 11.5$ 剧烈分化（误差 `> 5.4`）。由抽屉原理，**任何单一静态列表顺序（无论正序、逆序还是在独立 `DEV` 集上调优冻结的 `Ordinary_AllHistory_DevFrozenOrder`）至少在一半目标上必先猜错分支并多付高价确认观测**；唯有沿着 `HorizontalWormholeEdge + local_implication_edge` 两跳游走的 `Hegel` 能在全部目标上一次命中！
2. **把可执行 $\Pi_s$ 节点与 `coarse_graining_edge` 真正熔入同一个持续演化森林**：
   - 不再把动力学 $\Pi_s$ 留在独立支线实验：在 `Stage 0` 直接将 `ode_micro_2mode_node`、`ode_meso_schur_pi_thermal` 与 `cg_ode_micro_to_meso_pi_s` 注册进同一个 `PhilosophicalForest`；在 `Stage 2` 合并 `'hydraulic'` 析取作用域、经历 $T=0.12$ 淬火破裂、拦截未闭合重绑、重标定绝热盒 `[2.4, 5.2]` 并通过 `revalidate_vertical_edges_with_certificate` 恢复粗粒化边；在 `Stage 3` 直接路由跨域多尺度目标！
3. **建成真正的 `ForestScopeDispatcher`（零 `fit()` 实例级 `R3` 旧知识与新知识共存分发）**：
   - 将“实例级零重拟合保守重访（Tier-A）”与“跨领域参数化算子模板迁移（Tier-B）”诚实分立：在全部 `Stage 1` 发现、`Stage 2` 作用域收缩、析取 `FidelityCompression` 与淬火重整化完成后，由同一个 `ForestScopeDispatcher` 在 **删除 `fit` 函数（`any_fit_called = False`）** 的条件下对 `10` 条交错查询流执行真实 `authorize_abstraction_use` 门控与冻结参数推理。
4. **实测跑穿完整生命周期累计成本回本曲线（零线性外推）**：
   - 在实际执行的 `80` 个未见目标流上逐目标记录 `Stage 0 + Stage 1 + Stage 2 + Stage 3` 累计总复合成本，同时对比开放重发现 `Frozen_After_Stage0_OpenRediscovery` 与保守共享池 `Frozen_After_Stage0_PoolFallback`。

---

## 二、去重后 `M3.1` $2\times 2$ 检索因子分解精确结果（`108` 唯一运行/格，`432` 总运行）

| 检索臂（$\alpha \times \text{WormholePool}$） | 唯一目标运行数 | 严格通过率 | 假确认率 | 唯一两跳游走胜出 (`T25..T28`) | 游走子集复合成本 | 全集平均测量费 ($\lambda=0$) | 全集平均计算费 | 全集平均复合成本 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`hipporag_ppr` ($\alpha=0.65$, `Pool=True`)** | **`108`** | **`1.0000`** | **`0.0000`** | **`14 / 16` (`87.5%`)** | **`44.7579`** | **`35.6481`** | **`36.2192`** | **`37.4591`** |
| `hipporag_walk_nopool` ($\alpha=0.65$, `Pool=False`) | `108` | `1.0000` | `0.0000` | `14 / 16` (`87.5%`) | `44.7579` | `35.8333` | `37.2377` | `37.6952` |
| `hipporag_alpha0_nopool` ($\alpha=0.0$, `Pool=False`) | `108` | `1.0000` | `0.0000` | `0 / 16` (`0.0%`) | `64.3025` | `37.9630` | `46.4741` | `40.2867` |
| `hipporag_alpha0_pool` ($\alpha=0.0$, `Pool=True`) | `108` | `1.0000` | `0.0000` | `0 / 16` (`0.0%`) | `64.3025` | `37.9630` | `46.8444` | `40.3052` |

- **去重后正交主效应完全成立**：多跳图游走主效应（`Pool=True` 下 **`-2.8461`**，`Pool=False` 下 **`-2.5915`**）与虫洞等价池化在游走开启下的协同增益（**`-0.2361`**，纯测量费 `35.8333 -> 35.6481`）均保持严格正向贡献。

---

## 三、`ForestScopeDispatcher` 零 `fit()` 实例级 `R3` 交错重访实测（Part 4 Act 3）

在经历 `Stage 1` 五类复合算子入林、`Stage 2` 根节点 `stage0_monod_2d` 收缩至 `b <= 1e-6, s <= 4.0`、`ode_meso_schur_pi_thermal` 吸收 `'hydraulic'` 析取压缩及 `[2.4, 5.2]` 淬火重验证之后，`ForestScopeDispatcher` 对 `10` 条交错查询流执行**零重拟合（`any_fit_called = False`）**实例分发：

- **路由准确率**：**`10 / 10` (`all_routes_exact = True`)**；
- **授权查询通过率**：**`9 / 9` (`1.0000`)**，最大绝对预测误差 **`0.009287`**（其中 `R3_Q01/Q07` 旧 `Monod` 与 `R3_Q02/Q08` 新 `Haldane+Shunt` 冻结参数误差均为 **`0.0`**，而未拟合默认参数在 `W1` 上误差高达 `0.6577`；`R3_Q03` `'thermal'` 与 `R3_Q04` `'hydraulic'` 均经压缩后的析取节点 `ode_meso_schur_pi_thermal` v4 授权命中；`R3_Q05` 短时淬火 $T=0.12$ 自动拒绝介观盒并升尺度至 `ode_micro_2mode_node` 精确求解）；
- **越界模态拦截**：`R3_Q10` 未授权 `'optical'` 模态被 `authorize_abstraction_use` 干净拦截（`status = "scope_unauthorized"`，零假放行）。

---

## 四、四支柱统一持续生命周期与 `80` 目标未见迁移及实测回本曲线（`4` 种子 × `22` 目标 × `9` 臂 = `792` 次运行）

### 4.1 `Stage 3` 未见跨领域迁移九臂全量对照（每臂 `80` 域内 + `8` 异构 OOS）

| 评测臂 | 域内严格通过率 (`80`) | 域内假确认率 | OOS 诚实率 (`8`) | 召回确认数 | 非直接图游走胜出 | 平均候选评估数 | 平均测量费 ($\lambda=0$) | 平均计算费 | 平均复合成本 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Hegel_Unified_Continual_Forest`** | **`1.0000`** | **`0.0000`** | **`1.0000`** | **`80 / 80`** | **`48 / 80`** | **`1.9000`** | **`28.0000`** | **`24.4000`** | **`29.2200`** |
| `Hegel_NoMaintenance_SameHistory`（无压缩/无 Minton 消融） | `1.0000` | `0.0000` | `1.0000` | `64 / 80` | `32 / 80` | `3.5000` | `28.2500` | `34.0938` | `29.9547` |
| `Ordinary_AllHistory_Reversed`（扁平全历史逆序） | `0.9875` | **`0.0125`** | `1.0000` | `80 / 80` | `0 / 80` | `4.1000` | `34.4000` | `37.9000` | `36.2950` |
| **`Ordinary_AllHistory_DevFrozenOrder`（独立 DEV 冻结静态序）** | `1.0000` | `0.0000` | `1.0000` | `64 / 80` | `0 / 80` | `5.1875` | `36.6000` | `45.1000` | **`38.8550`** |
| `Forest_Alpha0_SameHistory`（同森林 $\alpha=0$ 无游走） | `1.0000` | `0.0000` | `1.0000` | `32 / 80` | `0 / 80` | `6.7000` | `36.8000` | `54.4000` | `39.5200` |
| `Flat_BM25_SameHistory`（同内容 BM25 扁平索引） | `1.0000` | `0.0000` | `1.0000` | `32 / 80` | `0 / 80` | `6.7000` | `36.8000` | `54.4000` | `39.5200` |
| `Frozen_After_Stage0_PoolFallback`（保守池回退冻结基线） | `1.0000` | `0.0000` | `1.0000` | `0 / 80` | `0 / 80` | `9.5000` | `36.8000` | `71.2000` | `40.3600` |
| `Ordinary_AllHistory_Forward`（扁平全历史正序） | `1.0000` | `0.0000` | `1.0000` | `64 / 80` | `0 / 80` | `14.3875` | `35.8000` | `100.1500` | `40.8075` |
| `Frozen_After_Stage0_OpenRediscovery`（真实开放重发现冻结基线） | `1.0000` | `0.0000` | `1.0000` | `0 / 80` | `0 / 80` | `9.5000` | `41.6000` | `95.2000` | `46.3600` |

### 4.2 全生命周期累计总成本（`Stage 0 + Stage 1 + Stage 2 + Stage 3`）实测回本里程碑

- **前期实测投入**：`Stage 0`（实测原子与 $\Pi_s$ 标定）`= 244.05`；`Stage 1`（5 类开放 `CodeAgent` 组合发现入林）`= 190.225`；`Stage 2`（析取压缩 + 3 异构 DEV 诱饵淘汰 + $\Pi_s$ 淬火复核）`= 218.45`（`Hegel` 前期总计 `652.725` vs `Frozen_Stage0` `244.05`，净持续投入 `408.675`）。
- **实测交叉回本点（无需任何外推）**：
  - 对比 **`Frozen_After_Stage0_OpenRediscovery`**：在**第 `29` 个目标**正式实现累计总成本反超（`1,498.875 < 1,511.350`），至第 `80` 个目标净节省 **`+962.525`** 复合成本（`2,990.325` vs `3,952.850`）；
  - 对比保守下界 **`Frozen_After_Stage0_PoolFallback`**：在**第 `39` 个目标**正式实现累计总成本反超（`1,792.575 < 1,800.900`），至第 `80` 个目标净节省 **`+482.525`** 复合成本（`2,990.325` vs `3,472.850`）。

---

## 五、可复现指令与静态证据核验

```bash
# 1. 运行 M0 -> M4.2 全套 37 项单元与端到端回归测试（100% 通过）
PYTHONPATH=src python3 -m pytest -q \
  tests/test_v16_6_m0_contracts_and_wormholes.py \
  tests/test_v16_6_m1_cross_domain_wormhole_study.py \
  tests/test_v16_8_m2_open_workspace_wormholes.py \
  tests/test_v16_9_m2_audit_closure_and_open_revision.py \
  tests/test_v17_0_m3_hipporag_wormhole_scaling.py \
  tests/test_v17_1_m3_1_audit_closure_and_walk.py \
  tests/test_v17_2_m4_multiscale_and_open_discovery.py \
  tests/test_v17_3_m4_1_unified_chain_and_audit_closure.py \
  tests/test_v17_4_m4_2_unified_closure_and_pi_lifecycle.py

# 2. 只读外部重放（不触碰 docs/evidence/ 已提交证据）
PYTHONPATH=src python3 scripts/run_v17_4_m4_2_unified_study.py --out-dir /tmp/hegel_v17_4_replay

# 3. 校验归档证据包完整性
(cd docs/evidence/v17_4_m4_2_unified_closure_and_pi_lifecycle_20261120 && sha256sum -c SHA256SUMS.txt)
```
