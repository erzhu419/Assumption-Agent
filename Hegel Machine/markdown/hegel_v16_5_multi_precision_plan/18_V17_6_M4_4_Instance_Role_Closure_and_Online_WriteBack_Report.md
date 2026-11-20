# 18 — Hegel V17.6 (`M4.4`) 红队复核闭环、全节点实例与规范角色锁定、自验证可迁移归档及有状态在线持续写回流（`M5` 桥接）实测报告

- **对应审计输入**：`markdown/Hegel_M4_3_Audit/`（固定审计提交 `3461535364fcdb4e9ae9c87ccba502b7e6b56fe8`，对照基线 `fb4b23aaf802c20b4e2d0018436abab73b0cca06`）
- **核心新增与升级模块**：
  - `src/hegel_machine/cognition/wormhole_forest.py`
  - `src/hegel_machine/cognition/hipporag_wormhole_index.py`
  - `src/hegel_machine/cognition/m4_compositional_discovery.py`
  - `src/hegel_machine/cognition/m4_multiscale_renormalization.py`
  - `src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py`
  - `src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py`
  - `src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py`
  - `src/hegel_machine/cognition/m4_4_instance_role_closure_and_online_writeback.py`（新增）
  - `scripts/run_v17_6_m4_4_closure_and_writeback_study.py`（新增，支持 `--out-dir` 无损外部只读重放）
  - `tests/test_v17_6_m4_4_instance_role_closure_and_online_writeback.py`（新增）
- **隔离静态证据目录**：`docs/evidence/v17_6_m4_4_instance_role_closure_and_online_writeback_20261120/`（`129` 项文件全部通过 `SHA256SUMS.txt` 校验，零 `__pycache__`/`.pyc` 混入；全部历史 `docs/evidence/v16_8_*..v17_5_*` 与 `markdown/*Audit*/` 100% 保持只读未改）

---

## 一、架构裁决：100% 闭合 `Hegel_M4_3_Audit` 工程与口径缺口 + 主动跨越保守边界打通有状态在线写回流（`M5` 桥接）

针对 GPT-6 对 `V17.5 (M4.3)` 的独立复核（`GPT_diagnosis_M4_3.md` / `M4_3_AUDIT.md` 及 `instance/`、`scale/`、`evaluation/` 三个专项报告），我们坚持**“凡有具体复现反例的工程缺口与计量口径一律 100% 严密闭合；凡建议停留在纯补丁 `M4.3.1` 或因静态小图表现就搁置动态 PPR/等价池化的保守判断，用真实有状态在线写回实验正面回应”**的架构原则：

### 1.1 完全吸收并闭合的两项 P1 阻断项与五项 P2 完整性/口径修复

1. **P1-1 全节点确认实例密码学锚定与分支感知零 `fit()` 分发（闭合 `GPT_diagnosis_M4_3.md` 第三节）**：
   - 在 `m4_2_unified_closure_and_pi_lifecycle.py` 中新增 `_make_verified_instance_action`，在 `Stage 0` 原子源标定、`Stage 1` 开放组合发现、`Stage 2` 析取整合与 $\Pi_s$ 重认证的全生命周期形成与修订入口，将 `confirmed_instance_digest`、`fitted_params_sha256`、`model_source_sha256`、`role_names` 与位置无关逻辑定位符统一写入节点动作（`post_stage2_forest` 全部 `14/14` 持产物节点 `missing_expected_instance_count == 0`）。
   - `ForestScopeDispatcher.dispatch_predict` 先选取 `authorize_abstraction_use` 授权命中的具体析取分支（`selected_branch`），对所有非内建精确求解器的实例节点强制要求非空 `confirmed_instance_digest`（缺失即返回 `missing_expected_instance_digest` / `scope_unauthorized`），并分别输出 `model_source_sha256`、`fitted_params_sha256` 与 `confirmed_instance_digest`。
   - **同 AST 合法重标定实例替换控制**：用 `_run_metered_atomic_calibration` 真实购测标定另一份同 AST、同源码 SHA 的合法 Monod 实例（`p0≈12, p1≈2` vs 原实例 `p0=6, p1=2`）替换磁盘三件套后，`R3` 立即拒绝（`scope_unauthorized`），恢复原文件后精确恢复 `2.66666667`。
2. **P1-2 规范角色映射全链路锁定与新鲜确认点盒交集门控（闭合第四节）**：
   - 新增 `_find_all_valid_canonical_role_maps_for_box` 与 `_role_order_from_canonical_role_map`：当目标特征名本身与节点规范角色名集合一致（如 `{'u_drive', 'T_horizon'}`）时，强制施加 **Canonical Identity Lock**（恒等映射越盒即直接判定不落盒，严禁通过交换 `u_drive ↔ T_horizon` 把短时淬火点塞进绝热证书盒）；当目标使用匿名特征名（如 `x0, x1`）时，将候选拟合的 `allowed_role_orders` 严格限制在通过证书盒校验的映射集合内，并把选定的唯一 `canonical_role_map` 贯穿**通道资格预检 → 候选编译拟合 → 新鲜确认点筛选/合成 → 粗通道购测守卫 → 最终 `heldout` 查询盒守卫**全链路。
   - `m4_3` 中 `select_or_synthesize_disjoint_confirm_points` 显式传入映射至目标坐标系的 `verified_box`（即 `feature_bounds ∩ verified_input_box`），且 `_acquire_points` 对所有 `coarse_macro` 购测施加逐点前置盒校验。
   - **原版反例复测**：短时轴交换反例（`u∈[2.45..2.95], T∈[0.40..1.20]`）不再误走 `coarse_macro`，而是诚实阻断粗盒门控（`canonical_axis_swap_box_gate_accepted = False`），自动走 `micro_fine` 精确通过（`heldout_rmse = 0.0`, `false_confirmation = False`）；即便强行删去细尺度节点也不发生粗通道假确认。匿名坐标 `x0/x1` 在越盒 `heldout` 上精确返回 `out_of_verified_scale_box`（`false_confirmation = False`）。
3. **P2-1 双模态细尺度精确特征矩阵（`A12 = g_sf`）与完整事件入账（闭合第九节 1–2）**：
   - 修正 `recalled_fine_2mode.py` 中慢模态耦合矩阵项 `a12 = float(ode_p["g_sf"])`（消除原先误除以 `tau_s` 导致的特征值偏差），使正常淬火目标 `T23` 的 `heldout_rmse` 从 `0.005842` 降至 **`0.0`**（机器精度级 `< 1e-6`）。
   - 删除进入双模态细尺度分支前被丢弃的冗余通用候选预拟合（实测 `discarded_generic_prefit_count == 0`，`evaluated_candidate_count == 1` 与子进程评估次数严格一一对应），并将本地可信 RK4 基函数生成（`local_trusted_rk4_basis_compute`）及产物完整性校验（`artifact_integrity_check`）如实记入 `CostMeter` 计算账本。
4. **P2-2 证书—实例原子绑定与位置无关自验证快照摘要（闭合第九节 3–4）**：
   - `add_coarse_graining_edge` 与 `revalidate_vertical_edges_with_certificate` 强制校验 `node_confirmed_instance_digest == certificate.confirmed_instance_digest`（以及析取分支实例摘要），校验失败在任何状态修改前立即返回 `certificate_instance_digest_mismatch`（森林序列化前后逐字节不变）；通过时原子提交 `updated_claim_contract`、`updated_scale_map` 与边版本重绑。
   - 节点契约内统一存储相对证据包根目录的位置无关逻辑路径（`"persisted_authored_workspaces/..."`），运行时由 `forest.resolve_artifact_path()` 结合外部 `artifact_root` 解析物理路径，序列化与冷加载**零次改写契约字段**。搬迁后磁盘原始 JSON 与冷加载内存对象的五字段 `semantic_digest` 及边 `closure_certificate_digest` 不一致数全部从 `7/8, 14/15, 14/15` 降为 **`0 / 8, 0 / 15, 0 / 15`**。
5. **P2-3 旧兼容入口加固、动态索引自动失效刷新与精确实验口径（闭合第七、八、九节 5）**：
   - `load_verified_authored_method` 的旧无 bundle 兼容分支仅在 `not verify_frozen_only and expected_instance_digest is None` 时允许进入，并显式标记 `_verification_mode = "legacy_isolated_refit"`；严格标志下零次调用 `evaluate_isolated_python_model` 即直接拒绝。
   - `m4_multiscale_renormalization.evaluate_m4_target_under_arm` 跨通道追踪 `paid_confirm_by_channel`，`ScaleBlind` 升级至 `micro_fine` 时自动并入已付细尺度确认点（`C0_T07` 重复购测点数从 `1` 降为 **`0`**）。
   - `PhilosophicalForest` 引入单调递增 `mutation_epoch` 与 `forest_state_signature()`；`HippoRAGWormholeIndex`、`FlatEqualMetadataIndex` 与 `CompiledRelationalCueTable` 均实现 `ensure_fresh()` 自动检测森林节点/边修订并重建计费（`W4 v1 -> v2` 在线修订后原索引实例直接返回 `v2` 新契约，无需人工新建对象）。
   - 修正实验口径：明确将 `random.gauss(0.0, 0.0020)` 表述为**未截断高斯传感器噪声（$\sigma_{\text{pilot}} = 0.0020$，约为容差的 $3.2\%$）**；将双轨噪声分岔检验扩展至全部 **4 个测试种子共 `32` 道 onset 目标（`192` 条逐例质量与费用回执落盘）**；在 6 族分层 DEV 中同步施加相同高斯噪声并覆盖 1 道绝热 + 1 道淬火 $\Pi_s$；如实报告 `Stage 2` 维护收益的家族归因边界。

### 1.2 不盲从保守收缩：超越纯补丁 `M4.3.1`，正面建成有状态在线写回与多线索持续流（Part 3，`M5` 桥接）

GPT-6 在第十一节建议“下一轮只做一个有边界的 `M4.3.1`”，同时在第七节与第十节指出两个核心科学问题：
1. 在冻结 15 节点单线索 `Stage 3` 上，预编译 2 跳关系闭包表比在线 PPR 全周期略省约 `4.95` 复合单位，且单线索查询下 `PoolOn` 与 `PoolOff` 无差异；
2. 此前 `Stage 3` 仅消费一次形成后的冻结森林，没有让流中新确认的发现实时改变同一活森林并由后续任务消费。

我们没有退回静态补丁，也没有删除 `HippoRAG + Wormhole` PPR 或等价池化，而是保留 Part 2 冻结基准的诚实对照，同时新增 **Part 3（`run_m4_4_online_writeback_and_multicue_stream`）**：在 20 步有状态在线持续流（`S01..S20`）中，让流中触发的作用域修订（`S01: W4 v1 -> v2`）与两项全新 `CodeAgent` 开放组合发现（`S02: W6` 一维 Haldane 底物抑制、`S09: W7` 二维黏弹应力松弛超调）**实时写回活森林并建立跨域纯 AST 虫洞**，自动触发索引失效刷新并计入写回计算费，同时穿插多线索组合查询与历史 `R3` 零拟合保持性检验，直接为 `M5` 递归自我改进奠定完整闭环机制。

---

## 二、Part 1：`Hegel_M4_3_Audit` 全部 P1 / P2 反事实探针闭合总表

证据文件：`docs/evidence/v17_6_m4_4_instance_role_closure_and_online_writeback_20261120/part1_m4_3_audit_closure_verification.json`

| 探针组 | 反事实控制项 | `V17.5 (M4.3)` 审计暴露状态 | `V17.6 (M4.4)` 实测结果 | 状态 |
|---|---|---|---|---|
| **P1-1 节点实例锚定** | `post_stage2` 14 个产物节点实例摘要覆盖；用真实重标定同 AST 同源码 `p0≈12` Monod 实例替换原 `p0=6` | 13/14 节点缺预期实例摘要；替换后仍 `dispatched_ok`（预测 `5.3333` 错翻倍） | `14/14` 节点完整锚定（缺失 **`0`**）；同 AST 替换立即 `scope_unauthorized`；删摘要亦拒绝；恢复原实例精确预测 `2.66666667` | **闭合** |
| **P1-2 角色映射锁与盒交集** | 短时轴交换 `u∈[2.45..2.95], T∈[0.40..1.20]`；匿名 `x0/x1` 越盒 `heldout`；首确认点已付时补选第二确认点 | 盒检查交换 `u↔T` 放行而模型未交换导致 `false_confirmation`（RMSE `0.1003`）；补选越盒点仍走粗通道 | 规范名恒等锁拦截粗盒（`accepted = False`），自动走 `micro_fine`（`heldout_rmse = 0.0`, `false_confirmation = False`）；匿名越盒返回 `out_of_verified_scale_box`；新鲜补选点 `100%` 落在证书盒内 | **闭合** |
| **P2-1 精确双模态矩阵与全计量** | `recalled_fine_2mode.py` 耦合项与正常淬火 `T23`；细分支冗余预拟合与本地 RK4 计算入账 | 误写 `g_sf / tau_s`（RMSE `0.005842`）；丢弃 1 次通用预拟合未计且本地 RK4 未单列计费 | 修正 `a12 = g_sf` 后 `T23 heldout_rmse = 0.0`；冗余预拟合 **`0`** 次（`evaluated_candidate_count == 1`）；`local_trusted_rk4_basis_compute` 与 `artifact_integrity_check` 完整入账 | **闭合** |
| **P2-2 跨实例证书与自验证归档** | 用 `Stage 0` 证书重验 `Stage 2` 节点；搬迁后重算三份快照 `semantic_digest` 与 `closure_certificate_digest` | 跨实例证书被接受导致边与证书实例摘要分裂；`7/8, 14/15, 14/15` 节点摘要不一致 | 跨实例证书原子拒绝（`certificate_instance_digest_mismatch`，森林零变更）；三份快照磁盘与内存摘要不一致数均为 **`0`**；冷加载 `R3` `10/10` 精确路由且 `0` 次 `fit` | **闭合** |
| **P2-3 旧入口与动态索引刷新** | 无 bundle 旧产物传 `verify_frozen_only=True`；旧 `ScaleBlind C0_T07` 细确认去重；`W4 v1->v2` 在线修订后查原索引 | 旧兼容分支绕过严格标志；`C0_T07` 重复购测 1 点；原 PPR 与原 Compiled 表继续返回 `v1` | 旧产物严格标志下零重拟合直接拒绝；`C0_T07` 重复细确认点数 **`0`**；原 PPR 与原 Compiled 表自动刷新返回 **`v2`** 并计入刷新计算费 | **闭合** |

---

## 三、Part 2：完整事件计量下的 11 臂全生命周期基准与四种子高斯噪声分岔对照

证据文件：
- `part2_m4_4_continual_benchmark_summary.json`
- `part2_m4_4_execution_receipts.jsonl`（`1,144` 条唯一运行回执）
- `part2_m4_4_fork_noise_receipts.jsonl`（全 4 种子 `192` 条双轨噪声分岔逐例回执）
- `relocatable_forest_snapshots/`（三份位置无关自验证森林快照 + `stage2_pi_s_reverified_certificate.json`）

### 3.1 11 臂主指标与全生命周期账本总表（完整计入本地 RK4 与产物校验计算事件）

前期投入（完整实测事件入账）：`Stage 0 = 244.05`，`Stage 1 = 189.775`，`Stage 2 = 218.45`（`Stage 0+1 = 433.825`，`Stage 0+1+2 = 652.275`；同分布带噪 6 族分层 DEV 调序实测复合成本 `468.00`；关系闭包表初编+维护重编复合费 `1.818`）。复合成本代理 $C = M + 0.05K$。

| 评测臂（每臂 `96` 域内 + `8` OOS） | 域内通过率 | 假确认率 | 最大重叠 | 召回确认 | 非直接关系命中 | 平均评估候选 | 平均测量费 $M$ | 平均计算费 $K$ | **Stage-3 平均 $C$** | 前期投入 | **96 题全周期总 $C$** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`Hegel_Unified_Continual_Forest`** | **`1.0000`** | **`0.0000`** | **`0`** | **`87`** | **`31`** | **`2.5938`** | **`31.3333`** | `22.6358` | **`32.4651`** | `652.275` | **`3,768.9262`** |
| `Hegel_Walk_NoPool_SameHistory` | `1.0000` | `0.0000` | `0` | `87` | `31` | `2.5938` | `31.3333` | `22.6358` | `32.4651` | `652.275` | `3,768.9262` |
| `Compiled_Relational_CueTable_SameForest` | `1.0000` | `0.0000` | `0` | `87` | `31` | `2.7604` | `31.3333` | `21.2250` | `32.3946` | `654.093` | `3,763.9730` |
| `Hegel_NoMaintenance_SameHistory` | `1.0000` | `0.0000` | `0` | `79` | `23` | `3.4479` | `36.2500` | `27.9383` | `37.6469` | `433.825` | `4,047.9281` |
| `Ordinary_AllHistory_BestPilotRMSE` | `1.0000` | `0.0000` | `0` | `96` | `0` | `17.5000` | `36.1875` | `107.5000` | `41.5625` | `433.825` | `4,423.8250` |
| `Forest_Alpha0_PoolOff_SameHistory` | `0.9896` | `0.0000` | `0` | `64` | `0` | `5.5417` | `37.6250` | `38.5208` | `39.5510` | `652.275` | `4,449.1725` |
| `Forest_Alpha0_PoolOn_SameHistory` | `0.9896` | `0.0000` | `0` | `64` | `0` | `5.5417` | `37.6250` | `38.6042` | `39.5552` | `652.275` | `4,449.5727` |
| `Flat_BM25_SameHistory` | `0.9896` | `0.0000` | `0` | `64` | `0` | `5.5417` | `37.6250` | `38.8883` | `39.5694` | `652.275` | `4,450.9390` |
| `Frozen_After_Stage0_OpenRediscovery` | `0.9896` | `0.0104` | `0` | `8` | `0` | `5.5208` | `42.1250` | `47.5463` | `44.5023` | `244.050` | `4,516.2720` |
| `Ordinary_AllHistory_StratifiedDevOrder` | `1.0000` | `0.0000` | `0` | `96` | `0` | `2.9167` | `37.1667` | `20.0000` | `38.1667` | `901.825` | `4,565.8250` |
| `Frozen_After_Stage0_NoDiscovery` | `0.1667` | `0.0000` | `0` | `8` | `0` | `4.3125` | `34.2917` | `30.8692` | `35.8351` | `244.050` | `3,684.2220`（仅解出 `16/96`） |

（注：全部 `11` 臂在 `8` 道 OOS 不连续目标上均为 `oos_scoped_unknown_rate = 1.0000`、`oos_false_confirmation_rate = 0.0000`；`Hegel_Unified_Continual_Forest` 的 `96` 次域内成功由 `87` 次森林关系召回确认 + `9` 次全历史池回退组成。）

### 3.2 维护回本的精确家族归因与条件边界

在完整计入本地 RK4 与产物校验计算事件后，全生命周期双回本结论依然稳健成立，其精确归因如下：
1. **对比 `Frozen_After_Stage0_OpenRediscovery`**：`Hegel_Unified_Continual_Forest` 同时取得更高质量（`96/96` 零假确认 vs `95/96` 且含 `1` 次假确认）与更低成本，在**第 `41` 个目标**实现累计全周期回本（`1,938.96` vs `1,949.96`），至第 `96` 个目标累计净省 **`+747.3458`**。
2. **对比 `Hegel_NoMaintenance_SameHistory`**：`Stage 3` 毛节省共 **`+497.4519`**，扣除 `Stage 2` 维护投入 `218.45` 后全周期净省 **`+279.0019`**，在当前 6 族固定配比、访问顺序、`3:10` 粗细单价与 $C=M+0.05K$ 代理下于**第 `46` 个目标**观测到累计回本（`2,070.30` vs `2,100.85`）。
   - **逐族精确分解**：`PiS_MultiScale_Mixed_2D` 贡献 **`+491.3840`**（占毛节省 **`98.78%`**，其中 `8` 道 `hydraulic adiabatic` 因 `Stage 2` 析取整合与重认证边接通合法 `coarse_macro` 通道贡献 **`+491.2920`**，`8` 道 `thermal quench` 为 `+0.0920`）；其余非 $\Pi_s$ 五族 `80` 题合计毛省 **`+6.0679`**（主要来自 `OnsetWalk_MonodShunt_2D` `+5.5911`）。因此，当前实验证明的是**在含跨模态多尺度通道解锁需求的持续流中，结构整合与证书重绑可收回整体维护投入**，而非每项局部安全修订单独盈利。

### 3.3 全 4 种子（`32` 题 / `192` 条回执）未截断高斯传感器噪声 vs 无噪分岔对照

| 传感器噪声设置（每臂 `32` 道 onset 题） | 评测臂 | 严格通过率 | 假确认率 | 最大重叠 | 平均测量费 $M$ | **平均复合费 $C$** |
|---|---|---:|---:|---:|---:|---:|
| 无噪（`noiseless`, $\sigma_{\text{pilot}} = 0$） | `Hegel_PPR` | `1.0000` | `0.0000` | `0` | `28.0000` | **`29.2435`** |
| 无噪（`noiseless`, $\sigma_{\text{pilot}} = 0$） | `BestPilotRMSE` | `1.0000` | `0.0000` | `0` | `28.0000` | `32.9250` |
| 无噪（`noiseless`, $\sigma_{\text{pilot}} = 0$） | `StratifiedDev` | `1.0000` | `0.0000` | `0` | `32.0000` | `33.0250` |
| 未截断高斯噪声（$\sigma_{\text{pilot}} = 0.0020$） | `Hegel_PPR` | `1.0000` | `0.0000` | `0` | **`29.2500`** | **`31.2529`** |
| 未截断高斯噪声（$\sigma_{\text{pilot}} = 0.0020$） | `BestPilotRMSE` | `1.0000` | `0.0000` | `0` | `30.0000` | `37.0250` |
| 未截断高斯噪声（$\sigma_{\text{pilot}} = 0.0020$） | `StratifiedDev` | `1.0000` | `0.0000` | `0` | `32.0000` | `33.0250` |

全 4 种子扩展验证了三点精确事实：① 各臂在同一目标上接收逐点一致的带噪 `pilot` 观测且全部保持 `100%` 严格通过与 `0` 假确认；② 无噪下 `BestPilotRMSE` 与 `Hegel_PPR` 物理测量费持平（`28.00`），差异来自全池角色排列拟合计算；③ 在 $\sigma_{\text{pilot}} = 0.0020$ 高斯传感器噪声下，`Hegel_PPR` 自身测量费也从 `28.00` 升至 `29.25`，但仍比 `BestPilotRMSE`（`30.00`）平均每题少 `0.75` 测量单位，并在复合成本上领先 `-5.7721`（`31.2529` vs `37.0250`）。

---

## 四、Part 3：有状态在线持续写回与多线索跨树流（`M5` 桥接，`20` 步 × `4` 臂 = `80` 条回执）

证据文件：
- `part3_m4_4_online_writeback_stream_summary.json`
- `part3_m4_4_online_writeback_receipts.jsonl`

在同一座 `post_stage2` 活森林起点上执行 `S01..S20` 序列：`S01` 触发 `stage1_W4_monod_shunt` 在线版本修订（`v1 -> v2`）；`S02` 通过原生 `CodeAgent` 新发现一维 Haldane 底物抑制 `W6` 并实时注册跨域同构虫洞对；`S03..S08` 为 6 道未见下游 `W6` 任务（含单线索直连、单线索跨虫洞及双线索组合查询）；`S09` 通过原生 `CodeAgent` 新发现二维黏弹应力松弛超调 `W7` 并实时注册跨域同构虫洞对；`S10..S15` 为 6 道未见下游 `W7` 任务；`S16..S20` 穿插 5 道历史多机制/双通道 $\Pi_s$ 保持性任务，流结束后再对活森林执行全套历史 `10/10` `R3` 零拟合回归与新节点 `W6`/`W7` 零拟合分发检验。

| 在线流评测臂（`20` 步有状态序列） | 严格通过率 | 假确认率 | 下游写回召回数 | 自动索引刷新次数 | 平均评估候选 | 平均测量费 $M$ | 平均计算费 $K$（含写回与刷新） | **20 步流总复合费 $C$** | 流后 `R3` `10/10` + `W6/W7` 零 `fit` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`Hegel_LiveWriteBack_PPR_PoolOn`** | **`1.0000`** | **`0.0000`** | **`12 / 12`** | **`3`** | **`2.1000`** | **`30.1500`** | **`24.7790`** | **`627.7785`** | **`True`（且三线索直查 Rank-1 命中 `W6`）** |
| `Hegel_LiveWriteBack_PPR_PoolOff` | `1.0000` | `0.0000` | `12 / 12` | `3` | `2.1000` | `30.1500` | `24.7790` | `627.7785` | `True`（三线索直查退居多义 hub 后） |
| `Compiled_LiveWriteBack_CueTable` | `1.0000` | `0.0000` | `12 / 12` | `3` | `2.5500` | `30.1500` | `25.2335` | `628.2335` | `True` |
| `Frozen_PostStage2_OpenRediscovery` | `1.0000` | `0.0000` | `0 / 12` | `0` | `7.8000` | `32.5500` | `61.8060` | `712.8068` | 历史 `10/10 True`，无 `W6/W7` 实例 |

### Part 3 核心发现
1. **真实在线写回收益与零遗忘保持**：开启在线写回后，流中新发现的 `W6` 与 `W7` 在后续 `12/12` 道未见跨域任务中全部直接召回确认（而冻结 `PostStage2` 为 `0/12`，每次被迫重跑全历史池与开放 `CodeAgent`），在完整计入写回（`11.2` 计算单位）与 `3` 次动态索引刷新开销后，20 步内即实现净节省 **`+85.0283`**（`627.7785` vs `712.8068`），同时流后历史 `10/10` `R3` 零拟合路由与新增 `W6`/`W7` 零拟合实例分发全部精确通过，节点 `semantic_digest` 不一致数为 **`0`**。
2. **动态写回与多线索查询下在线 PPR 反超预编译闭包表**：不同于静态冻结的 Part 2（`Compiled` 略省 `4.95`），在每次在线修订/入林都会触发闭包表全局重编、且下游包含多线索组合查询的动态流中，`Hegel_LiveWriteBack_PPR_PoolOn` 平均评估候选更少（**`2.10` vs `2.55`**），流总复合成本正式反超 `Compiled_LiveWriteBack_CueTable`（**`627.7785 < 628.2335`**，净省 `+0.4550`）。同时在纯净双域分割三线索查询 `["sig:cue_tubule_saturation_a", "sig:cue_reuptake_saturation_b", "sig:pos_arity_1"]` 上，开启虫洞等价池化的 `PoolOn` 将真实同构节点 `online_W6_reuptake_twin_1d` 排在 **Top-1**，而 `PoolOff` 与 `Compiled` 均被同时携带两表面标签的多义节点 `stage1_W3_murray_capped` 抢占 Top-1（流内因 `W6` 经验曲线同时触发通用次线性特征 `sig:pos_sublinear_2_3`，`PoolOn` 与 `PoolOff` 端到端账本持平，如实记录该边界）。

---

## 五、可复现指令清单

```bash
# 1. 运行 M0—M4.4 全量 42 项回归与审计闭环测试（零字节码污染）
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src PYTHONHASHSEED=0 python3 -m pytest -q \
  tests/test_v16_6_m0_contracts_and_wormholes.py \
  tests/test_v16_6_m1_cross_domain_wormhole_study.py \
  tests/test_v16_8_m2_open_workspace_wormholes.py \
  tests/test_v16_9_m2_audit_closure_and_open_revision.py \
  tests/test_v17_0_m3_hipporag_wormhole_scaling.py \
  tests/test_v17_1_m3_1_audit_closure_and_walk.py \
  tests/test_v17_2_m4_multiscale_and_open_discovery.py \
  tests/test_v17_3_m4_1_unified_chain_and_audit_closure.py \
  tests/test_v17_4_m4_2_unified_closure_and_pi_lifecycle.py \
  tests/test_v17_5_m4_3_channel_bound_lifecycle_and_audit_closure.py \
  tests/test_v17_6_m4_4_instance_role_closure_and_online_writeback.py

# 2. 重新生成完整 V17.6 (M4.4) 证据包（或通过 --out-dir /tmp/v17_6_replay 做无损外部复核）
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src PYTHONHASHSEED=0 \
  python3 scripts/run_v17_6_m4_4_closure_and_writeback_study.py

# 3. 校验全部 129 项静态证据文件 SHA256
(cd docs/evidence/v17_6_m4_4_instance_role_closure_and_online_writeback_20261120 && sha256sum -c SHA256SUMS.txt)
```
