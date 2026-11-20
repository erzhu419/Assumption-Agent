# 17 — Hegel V17.5 (`M4.3`) 红队审计闭环、真实通道绑定多尺度计费、确认实例身份契约与 11 臂全生命周期实测报告

- **对应审计输入**：`markdown/Hegel_M4_2_Audit/`（基线提交 `fb4b23aaf802c20b4e2d0018436abab73b0cca06`，对比提交 `99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c`）
- **核心新增与升级模块**：
  - `src/hegel_machine/cognition/wormhole_forest.py`
  - `src/hegel_machine/cognition/m4_compositional_discovery.py`
  - `src/hegel_machine/cognition/hipporag_wormhole_index.py`
  - `src/hegel_machine/cognition/m4_multiscale_renormalization.py`
  - `src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py`
  - `src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py`
  - `src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py`
  - `scripts/run_v17_5_m4_3_unified_study.py`
  - `tests/test_v17_5_m4_3_channel_bound_lifecycle_and_audit_closure.py`
- **隔离静态证据目录**：`docs/evidence/v17_5_m4_3_channel_bound_lifecycle_20261120/`（`126` 项文件全部通过 `SHA256SUMS.txt` 校验，零 `.pyc` 字节码混入；全部历史 `docs/evidence/v16_8_*..v17_4_*` 与 `markdown/*Audit*/` 100% 保持只读未改）

---

## 一、战略裁决：吸收真实工程断连 vs 拒绝过度保守退缩

针对 GPT-6 对 `V17.4 (M4.2)` 的独立审计（`GPT_diagnosis_M4_2.md` / `M4_2_AUDIT.md`），我们按**“100% 闭合真实执行/计量断连，同时主动把实验台推进到更能体现多精度哲学森林长期价值的真实双通道与有限仪器精度机制”**做出如下架构裁决：

### 1.1 完全吸收并彻底修复的五组核心连接（`20 / 20` 原版审计探针翻转闭合）

1. **测量价格严格绑定实际获得授权并执行的物理通道（闭合 Sec. II / IV）**：
   - 废除检索前按索引类型或“森林中存在任意未失效粗边”决定单价的逻辑。
   - 在 `M4.2` 与全新双通道 `M4.3` 中，粗通道（`coarse_macro`）仅在**实际检索命中对应 `meso` 节点、该节点自身拥有非失效 `coarse_graining_edge`（`fine_scale_children_of(node_id)` 非空）、通过 `authorize_abstraction_use` 模态/前提门控、且规范角色映射后的查询点全部落在 `verified_input_box` 内**时才可激活；在 `M4.3` 中 `coarse_macro`（`3.0`/点，真实 Schur 约化输出）与 `micro_fine`（`10.0`/点，真实双模态 RK4 输出）更是物理输出不同的独立通道。
   - **原版探针复测**：删掉 ODE 粗边、仅留无关 `hydro` 粗边时，执行通道与无粗边完全同为 `micro_fine`，复合成本逐位相等（`67.0995 == 67.0995`）。同时所有检索 `query_compute_cost`、DEV 候选拟合及关系表编译/重编开销全部逐事件记入 `CostMeter`。
2. **完整保留真实 `Stage 0` 边界快照 + 真实执行 `CodeAgent` 开放重发现（闭合 Sec. III / V）**：
   - 在 `Stage 0` 结束、`Stage 1` 开始的精确边界捕获完整森林快照（**`8` 个节点、`2` 条粗粒化边、`1` 条横向虫洞**），不再手工裁剪为无边四节点。
   - 删除 `Stage 3` 中无对应动作的合成记账 `meter.charge("open_compositional_rediscovery_diag")`；`Frozen_After_Stage0_OpenRediscovery` 在召回未命中时通过 `_PrepaidAwareAdapter` 继承已付前缀并**真实运行 `run_compositional_code_agent_episode`**（含真实诊断采样、子进程 `diag_residual.py`、逐候选仿射透镜标定与独立确认）；同时单列 `Frozen_After_Stage0_NoDiscovery`（仅使用 `Stage 0` 原子池，诚实展现 `Stage 0` 的 `16.67%` 能力上限且 `0%` 假确认）。
3. **全路径新鲜独立确认不变量（`max_prefix_confirm_overlap == 0`，闭合 Sec. IV / VI）**：
   - 新增 `select_or_synthesize_disjoint_confirm_points`：继续 100% 保留全部已付失败确认进入训练前缀，但在任何确认前严格校验确认点与 `training_prefix_rows` 零重叠且足额满足 `fresh_points`；预置确认池耗尽时在 `feature_bounds`（及有效盒）内确定性合成未见内点并真实付费采购。
   - **实测结果**：原版 `DEV T13` 四次评估重叠数从 `[0, 0, 1]` 降为 **`[0, 0, 0, 0]`**；`fresh_points = 3` 真实采购 `3` 个独立点；父级耗尽后子级真实采购 `2` 个全新点；`M4.3` 全部 `11` 臂 `1,144` 次运行的最大训练—确认重叠数为 **`0`**。
4. **确认实例密码学绑定与真正全链路零 `fit()` 只读 `R3` 分发（闭合 Sec. V / VII / IX）**：
   - `canonicalize_fitted_params_snapshot` 与 `compute_confirmed_instance_digest` 将源码 SHA、AST 骨架摘要、有序有限 `_PARAMS` 快照哈希与规范角色名合成为唯一实例身份，同时写入 `authored_method.json`、`confirmed_fitted_params.json` 与 `confirmed_manifest.json`。
   - `load_verified_authored_method` 直接用 `predict_with_frozen_instance` 在原 `confirm_rows` 上做零 `fit()` 预测核验并比对外部参数文件与 manifest（`p0 * 3` 仅改 artifact 或同时改外部参数文件均返回 `None`）。
   - `ode_micro_2mode_node` 在 `Stage 2` 通过正式 `empirical_revise_node` 升级为 `thermal | hydraulic` 析取契约（版本号升至 `v2`），使 `run_interleaved_zero_fit_dispatcher_stream` **100% 只读**且在冷加载与目标推理全链路**零次调用 `fit()`**；同时越盒细尺度升级强制依赖 `fine_scale_children_of` 活粗粒化边（删除全部粗边后 `Q05` 返回 `escalation_blocked_missing_coarse_edge`）。
5. **统一尺度证书、规范角色双射映射、最终查询守卫与可重定位快照归档（闭合 Sec. VI / VIII / XI）**：
   - `revalidate_vertical_edges_with_certificate` 强制校验 `requested_box ⊆ certified_box`（合法 `[2.4, 5.2]` 证书扩盒至 `[0.05, 5.2]` 立即拒绝），并在接受时原子同步节点与 `coarse_graining_edge` 的 `ScaleMap`、证书与实例摘要。
   - **主动保留并接通窗口投影门控（战略升级点 1）**：没有把 `window_projected_y` 退化为闲置字段，而是将尾窗准稳态漂移 `|window_projected_y - y_coarse| <= window_drift_tolerance` 正式接入证书与纵向边复核，使**末态快模态偏移与尾窗慢模态瞬态漂移双向因果控制 `certified`**（为 `M5` 非平稳振荡粗粒化铺路）。
   - `Stage 3` 全面继承 `model_scope_contract.json` 最终查询盒守卫（越盒续问返回 `out_of_verified_scale_box` 且 `false_confirmation = False`），并通过 `_find_valid_canonical_role_map_for_box` 先做规范角色双射再验盒（未知字段一律 `fail-closed`）。
   - 静态证据包完整持久化 `Stage 0 / Stage 1 / Stage 2` 产物与可重定位 JSON 快照，并在临时目录销毁后冷加载 `post_stage2_forest_snapshot.json` 实测通过 `10/10` `R3` 验证。

### 1.2 不盲从保守收缩、主动面向后续计划强化的三项实验与架构设计

1. **有限仪器精度分岔（$\sigma_{\text{pilot}} = 0.0020$）与无噪（$\sigma = 0$）双轨并测 + 正面纳入 `BestPilotRMSE`**：
   - GPT-6 指出在完全无噪浮点下，`min-pilot-RMSE` 可凭 $10^{-10}$ vs $10^{-3}$ 残差区分 `MonodShunt` 与 `MurrayShunt`。在真实物理测量与 `M5` 开放世界路线中，低负载试探窗口天然受有限传感器精度限制。
   - 我们在 `M4.3` 中既把全历史最小试探残差选择器 `Ordinary_AllHistory_BestPilotRMSE` 升格为正式主评测臂，又同时运行 $\sigma = 0$ 与有界仪器精度 $\sigma_{\text{pilot}} = 0.0020$（小于容差的 `3.3%`，但高于两曲线在低负载区的 `0.0013` 理论间隔）双轨对照：无噪下诚实验证 GPT-6 指出的测量持平、计算更高（`32.925` vs `29.1623`）；有界仪器噪声下则清晰观测到 `BestPilotRMSE` 因低信噪比产生误判并多付高负载确认费（`37.025` vs `31.6748`）。
2. **正面纳入同信息 `Compiled_Relational_CueTable_SameForest` 对照（解耦“森林关系表示”与“在线 PPR 搜索”）**：
   - 直接把同一座哲学森林的 2 跳关系闭包预编译为查询表（显式计量 `Stage 1` 初编与 `Stage 2` 维护失效重编计算费）。实测证明：**大幅击败扁平/全历史基线的核心因果来源正是哲学森林的关系拓扑与契约门控本身**；在当前 `K ≈ 14` 规模下，编译闭包表与在线 PPR 取得完全一致的物理测量成本（`31.3333`），编译表单次查询计算略轻（均值 `32.3235` vs `32.3941`），而在线 PPR 平均少评估候选（`2.5938` vs `2.7604`）且无需随每次局部修订重建全局闭包表，为 `M5` 大规模动态图与多线索组合查询保留关键扩展性。
3. **补齐全部 6 族测试覆盖（含 `W3 MurrayCapped` 与短时淬火边界 `Pi_s`）并实现 `Stage 2` 维护真实回本**：
   - 采用全新未见随机种子 `20301120, 20311120, 20321120, 20331120`（从未参与 `M4.2` 诊断），每 seed 覆盖全部 6 个机制族（`24` 域内 + `2` OOS = `96` 域内 + `8` OOS/臂），使 `Stage 2` 维护在非 $\Pi_s$ 族与双通道 $\Pi_s$ 族同时产生正向净节省，并在**第 `46` 题正式实现对 `Hegel_NoMaintenance_SameHistory` 的全生命周期回本（第 `96` 题净省 `+278.20`）**！

---

## 二、Part 1：`Hegel_M4_2_Audit` 六组反事实探针闭合总表

证据文件：`docs/evidence/v17_5_m4_3_channel_bound_lifecycle_20261120/part1_m4_2_audit_closure_verification.json`

| 探针组 | 反事实控制项 | `V17.4` 审计暴露状态 | `V17.5 (M4.3)` 实测结果 | 状态 |
|---|---|---|---|---|
| **C1 实例参数绑定** | 只把真实 W1 `authored_method.json` 中 `p0 * 3`（保留源码/manifest） | `loader` 仍接受（冻结预测 RMSE `7.69`） | `mutated_artifact_param_accepted = False`；连改外部参数文件仍 `False` | **闭合** |
| **C2 OR 权限/异质法则与步骤序** | 同法则 `A(bounded_trial)+B(retrieve_candidate)` 合并后对 `B` 申请 `bounded_trial`；异质法则合并；反转 `program_steps` | `B` 被提升权限；异质法则合并通过；逆序计划相同 | `scope_B_bounded_trial_granted = False`（限为 `retrieve_candidate`）；`heterogeneous_law_merge_accepted = False`；`reversed_program_rejected = True` | **闭合** |
| **C3 新鲜独立确认** | `fresh_points=3`（预置 2 点）；父级耗尽 2 点后子级确认；`DEV T13` 回退确认 | 静默截断为 2 点；复用训练点 `q=6.2`；`32` 次训练点重确认 | 真实购满 `3` 点；子级购 `2` 全新点；重叠数全部为 **`0`** | **闭合** |
| **C4 轨迹门控与尺度授权链** | 尾窗慢态 `+100`；窄盒 `[2.4,5.2]` 证书配 `[0.05,5.2]` `ScaleMap`；`Stage 3` `T=0.12` 最终查询；`u_probe/t_probe` 改名越盒 | 尾窗不影响证书；扩盒通过；最终查询 `completed + false_confirmation`；改名跳过盒检查 | 尾窗与末态均使 `certified = False`；扩盒 `accepted = False`；节点与边盒同步；最终越盒返回 `out_of_verified_scale_box`（`false_confirmation = False`）；改名越盒拒绝、盒内通过 | **闭合** |
| **C5 只读零拟合 `R3` 与边升级** | 未跑 stream 前的生产森林测 `optical`；删除全部 `coarse_graining_edges` 测淬火升级；冷加载 `fit()` 计数 | 现场补规则才拒 `optical`；删边路由不变；冷加载重跑 `fit` | 原始 `post_stage2` 快照直接 `scope_unauthorized`；stream `100%` 只读；全链路 `fit_called = False`；删粗边后 `escalation_blocked_missing_coarse_edge` | **闭合** |
| **C6 通道绑定计价、真 `Stage 0` 与开放重发现** | 删 ODE 粗边仅留无关 `hydro` 粗边 vs 删全部粗边；`Frozen` 快照节点/边数；`OpenRediscovery` 动作对比 | 无关边触发半价（`34.825` vs `66.825`）；`Frozen` 仅 4 节点 0 边；仅多记账无新采样 | 两者同走 `micro_fine` 且成本完全相等（`67.0995 == 67.0995`）；真 `Stage 0` 完整保留 `8` 节点 `2` 粗边 `1` 虫洞；真实执行 `CodeAgent` 诊断采样与合成 | **闭合** |

---

## 三、Part 2：`V17.5 (M4.3)` 11 臂全生命周期主实验结果（`96` 未见域内 + `8` OOS / 臂，全新四种子）

证据文件：
- `part2_m4_3_channel_bound_continual_summary.json`
- `part2_m4_3_execution_receipts.jsonl`（共 `1,144` 条逐运行回执）
- `relocatable_forest_snapshots/`（含冷加载重定位验证 `relocated_post_stage2_cold_reload_r3_verified: all_routes_exact = True`）

### 3.1 11 臂主指标与全生命周期账本总表

前期投入（实测事件入账）：`Stage 0 = 244.05`，`Stage 1 = 189.775`，`Stage 2 = 218.45`（`Stage 0+1 = 433.825`，`Stage 0+1+2 = 652.275`；6 族分层 DEV 调序实测成本 `467.55`；关系闭包表编译+维护重编复合费 `1.818`）。

| 评测臂（每臂 `96` 域内 + `8` OOS） | 域内通过率 | 假确认率 | 训练-确认最大重叠 | 召回确认 | 非直接游走胜出 | 平均评估候选 | 平均测量费 | 平均计算费 | **Stage-3 平均复合费** | 前期总投入 | **96 题全生命周期总成本** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`Hegel_Unified_Continual_Forest`** | **`1.0000`** | **`0.0000`** | **`0`** | **`87`** | **`31`** | **`2.5938`** | **`31.3333`** | `21.2150` | **`32.3941`** | `652.275` | **`3,762.1078`** |
| `Hegel_Walk_NoPool_SameHistory` | `1.0000` | `0.0000` | `0` | `87` | `31` | `2.5938` | `31.3333` | `21.2150` | `32.3941` | `652.275` | `3,762.1078` |
| `Compiled_Relational_CueTable_SameForest` | `1.0000` | `0.0000` | `0` | `87` | `31` | `2.7604` | `31.3333` | `19.8042` | `32.3235` | `654.093` | `3,757.1530` |
| `Hegel_NoMaintenance_SameHistory` | `1.0000` | `0.0000` | `0` | `79` | `23` | `3.4479` | `36.2500` | `26.3508` | `37.5675` | `433.825` | `4,040.3097` |
| `Forest_Alpha0_PoolOn_SameHistory` | `0.9896` | `0.0000` | `0` | `64` | `0` | `5.5417` | `37.6250` | `37.7625` | `39.5131` | `652.275` | `4,445.5373` |
| `Forest_Alpha0_PoolOff_SameHistory` | `0.9896` | `0.0000` | `0` | `64` | `0` | `5.5417` | `37.6250` | `37.7625` | `39.5131` | `652.275` | `4,445.5373` |
| `Flat_BM25_SameHistory` | `0.9896` | `0.0000` | `0` | `64` | `0` | `5.5417` | `37.6250` | `38.1300` | `39.5315` | `652.275` | `4,447.2990` |
| `Ordinary_AllHistory_StratifiedDevOrder` | `1.0000` | `0.0000` | `0` | `80` | `0` | `7.5000` | `37.8333` | `47.5000` | `40.2083` | `901.375` | `4,761.3750` |
| `Ordinary_AllHistory_BestPilotRMSE` | `1.0000` | `0.0000` | `0` | `96` | `0` | `17.5000` | `36.1875` | `107.5000` | `41.5625` | `433.825` | `4,423.8250` |
| `Frozen_After_Stage0_OpenRediscovery` | `0.9688` | `0.0104` | `0` | `8` | `0` | `5.4792` | `42.1250` | `46.9588` | `44.4729` | `244.050` | `4,513.4520` |
| `Frozen_After_Stage0_NoDiscovery` | `0.1667` | `0.0000` | `0` | `8` | `0` | `4.3125` | `34.2917` | `30.5317` | `35.8182` | `244.050` | `3,682.6020`（仅解 `Stage 0` 机制） |

（注：全部 `11` 臂在 `8` 个随种子变化的域外不连续目标上均实现 `oos_scoped_unknown_rate = 1.0000`、`oos_false_confirmation_rate = 0.0000`。）

### 3.2 关键科学发现与精确口径

1. **真实动作计费下的全生命周期双回本（同时击败真实 `Frozen_OpenRediscovery` 与 `NoMaintenance`）**：
   - 对比完整保留 8 节点 3 边 `Stage 0` 快照、并真实运行 `CodeAgent` 组合发现的 `Frozen_After_Stage0_OpenRediscovery`（平均复合费 `44.4729` vs `32.3941`，节省 `-27.2%`），`Hegel_Unified_Continual_Forest` 在**第 `41` 个目标实现全生命周期累计回本**，到第 `96` 个目标累计净节省 **`+751.3442`**（`3,762.11` vs `4,513.45`）。
   - 对比不做 `Stage 2` 维护的 `Hegel_NoMaintenance_SameHistory`（平均复合费 `37.5675` vs `32.3941`），`Hegel_Unified_Continual_Forest` 在 `Stage 3` 共节省 `496.6519`（其中双通道 `PiS_MultiScale_Mixed_2D` 节省 `491.1840`，非 $\Pi_s$ 族节省 `5.4679`），完全覆盖 `Stage 2` 的 `218.45` 维护投入，在**第 `46` 个目标实现维护回本**，到第 `96` 个目标净节省 **`+278.2019`**！
2. **关系知识表示 vs 在线 PPR vs 停止规则的清晰拆解**：
   - **单因子分解**：在 `Stage 3` 中已将 `Alpha0` 与 `Pooling` 正交拆开（`Hegel [Walk+Pool] = 32.3941`、`Walk_NoPool = 32.3941` vs `Alpha0_PoolOn = 39.5131`、`Alpha0_PoolOff = 39.5131`），证明本轮两跳桥接与粗粒化边可达性收益来自 `damping_alpha > 0` 的多跳传播而非池化混同。
   - **同信息关系闭包表对照**：`Compiled_Relational_CueTable_SameForest` 基于**完全相同的维护后哲学森林**预编译 2 跳关系闭包，获得与在线 PPR 相同的测量费（`31.3333`）与略低的表查询计算费（`32.3235` vs `32.3941`），但在线 PPR 平均评估候选更少（`2.5938` vs `2.7604`）且无需在每次局部节点修订/压缩后重编全局闭包表。因此准确科学结论为：**相对于扁平检索（`39.53`）和全历史穷举/静态序（`40.21–41.56`）的核心优势来自哲学森林的显式跨树/跨尺度关系结构与作用域契约；小规模 2 跳静态闭包表可近似复现单线索路由，而在线 PPR 提供免重编的动态多线索扩展能力**。
   - **无噪 vs 有界仪器噪声分岔对照**：
     - $\sigma = 0$（理想无噪）：`BestPilotRMSE` 测量费与 `Hegel` 持平（`28.0`），但因全池拟合角色排列导致复合费更高（`32.925` vs `29.1623`）；
     - $\sigma_{\text{pilot}} = 0.0020$（有界仪器噪声）：低负载区两分岔模型差值被传感器精度淹没，`BestPilotRMSE` 测量费升至 `30.0`、复合费升至 `37.025`，`Hegel` 依托跨树同源关系保持 `31.6748`。

---

## 四、可复现指令清单

```bash
# 1. 运行 M0—M4.3 全量 40 项回归测试
PYTHONPATH=src python3 -m pytest -q \
  tests/test_v16_6_m0_contracts_and_wormholes.py \
  tests/test_v16_6_m1_cross_domain_wormhole_study.py \
  tests/test_v16_8_m2_open_workspace_wormholes.py \
  tests/test_v16_9_m2_audit_closure_and_open_revision.py \
  tests/test_v17_0_m3_hipporag_wormhole_scaling.py \
  tests/test_v17_1_m3_1_audit_closure_and_walk.py \
  tests/test_v17_2_m4_multiscale_and_open_discovery.py \
  tests/test_v17_3_m4_1_unified_chain_and_audit_closure.py \
  tests/test_v17_4_m4_2_unified_closure_and_pi_lifecycle.py \
  tests/test_v17_5_m4_3_channel_bound_lifecycle_and_audit_closure.py

# 2. 在只读外部目录重放完整 V17.5 (M4.3) 四种子 11 臂研究
PYTHONPATH=src PYTHONHASHSEED=0 python3 scripts/run_v17_5_m4_3_unified_study.py --out-dir /tmp/hegel_v17_5_replay

# 3. 校验提交证据完整性（126 项）
cd docs/evidence/v17_5_m4_3_channel_bound_lifecycle_20261120 && sha256sum -c SHA256SUMS.txt
```
