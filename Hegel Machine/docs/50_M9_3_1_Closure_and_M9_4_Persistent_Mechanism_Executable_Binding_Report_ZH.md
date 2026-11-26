# Hegel M9.3.1 确定性审计闭环与 M9.4 持久机制—可执行绑定持续世界报告

## 1. 架构路径裁定（`Path 1 / Path 2 / Path 3`）

基于 `Hegel_M9_3_Audit`（`GPT_diagnosis_M9_3.md`、`00_M93_FULL_AUDIT_ZH.md`、`01_M93_1_M94_IMPLEMENTATION_PLAN_ZH.md`）给出的完整审计信息，我们对三条后续演进路径做出如下明确裁定：

- **拒绝 `Path 1`（仅把 `M9` 作为孤立容器技能库封存）**：若不将 `M9` 的可执行算子接回 `M8` 的 `PersistentMechanismGraph`，则 `M8`（持久机制图谱、`AncestorReweight` 祖先重加权、`LocalSplit` 局部分裂、`sentinel` 零附带漂移隔离）与 `M9`（隔离容器 `L1/L2` 算子、四元组准入收据、`ScientificProbeReceipt` 宿主科学探针）将长期割裂为两套互不闭环的子系统。
- **采纳 `Path 2`（在现有 `M8 + M9` 对象之上建立 `MechanismExecutableBinding` 统一闭环）**：无需另起炉灶重写第三套元框架（拒绝 `Path 3`），也**完全不需要**要求任意黑盒数值 Python 代码（`SciPy` `minimize`、`splu`、IRLS、FVM 离散格式）可微——通过在 `PersistentMechanismGraph` 内持久化显式版本化的 **`MechanismExecutableBinding`** 记录，让 `M8` 负责跨任务共享上游校准版本演进（`AncestorReweight`）与按组件归因的谱系分叉（`LocalSplit`），让三值可观测前置守卫（`ObservablePreconditionGuard`）与宿主科学探针（`ScientificProbeReceipt`）驱动真实容器可执行模块的零轮次直通复用或受控回退综合。

### 关于 `M8`（`Track A`）与 `M9`（`Track B`）真实实现边界的诚实声明

1. **`M8`（`Track A`）的精确数学实现范围**：
   - `m8_persistent_variational_planner.py` 中的 `PersistentMechanismGraph`、`TypedMechanism`、`TaskModel` 与 `M8VariationalPlanner` 实现的是尺度协变经验贝叶斯正态—逆伽马（`NIG`）共轭线性/基函数后验推断（`fit_scale_covariant_nig_linear_model` 与 Student-$t$ 预测对数得分 `predict_nig_student_t_log_score`）、残差结构诊断排序（调整 $R^2$ / 偏移 $R^2$ / 重尾集中度）、跨动作族 `{NoChange, ReuseMechanism, LocalRefit, AncestorReweight, LocalSplit, ConceptLift, ChangeMemory}` 的单位成本变分自由能/预测增益仲裁，以及无关 `sentinel` 根节点的 `0.0` 附带漂移隔离。
   - 它**不对任意黑盒 Python 数值求解器求导**；在 `M9.4` 中，`M8` 负责管理共享上游校准参数版本（`param_v1 -> param_v2`）与分叉谱系，`MechanismExecutableBinding` 则将该参数版本与三值可观测守卫绑定到 `Stage S` 在线生成的容器 Python 可执行模块上。
2. **`M9.3` 基线命名与留存率口径勘误**：
   - `M9.3` 中的 `B`（`prior_frozen_v7_no_source_growth`）是冻结 `v7` 先验库无在线源生长消融基线，**不是**无记忆裸模型基线。
   - `M9.3` 中 `E` 臂除 `L1` 调用外还真实执行了 `1,021` 次运行时 `L2` 模块调用；在引入 `M9.3.1` `F07` 可选证书接口前，最终提交文件内的严格 AST 静态留存率为 `1/8`（`12.5%`）。

---

## 2. `M9.3.1` 十项确定性审计闭环（`F01–F10`，提交 `239f08248`）

`Hegel_M9_3_Audit` 指出的全部 10 项问题已在 `0` 次 LLM 调用下完成确定性修复、重评与跨里程碑回归验证（封存于 `docs/evidence/m9_3_1_audit_closure_and_regrade_20261126/`）：

| 闭环编号 | 审计问题摘要 | 确定性修复方案与封存收据验证 |
| :--- | :--- | :--- |
| **`F01` (`C1`)** | `Stage R` 分叉节点挂到了 `HippoRAG` 邻近节点（`node_auto_03`）而非真实源父节点（`node_auto_09`） | 为 `consolidate_autogenous_episode_into_forest` 增加显式 `source_node_id`、`source_version`、`violated_component_id`、`preserved_component_ids`，强制精确挂接真实父节点并同步反向守卫（`c1_c2_component_bound_bifurcation_and_guard_controls.json`）。 |
| **`F02` (`C2`)** | 散文式前提字符串（如 `"V_pocket == 0"`）不可执行，且把非零可加分离势误判为违例 | 实现结构化 `ObservablePreconditionGuard` 与三值执行器 `evaluate_observable_precondition_guard`（`TRUE / FALSE / UNKNOWN`）：同时检测历史与保持段应力超限及 `residual_plastic_state`，并用精确二维可加分离残差比 $\lVert V - \bar{V}_{x\cdot} - \bar{V}_{\cdot y} + \bar{V}_{\cdot\cdot}\rVert_F / \max(\lVert V\rVert_F, 1)$ 判定可分离性（对非零可加势 $V(x,y)=a(x)+b(y)$ 精确给出 `0.0`）。 |
| **`F03` (`A5`)** | 容器内可写的 `/holdout_task/validate_public_schema.py` 被用于将科学类 nogood 升级为 `host_probe_verified` | 用 `shlex.quote` 在运行时重验 `/holdout_task` SHA-256；严格拆分 `public_schema_receipt`（仅允许升级 `schema_contract / output_format / interface_schema`）与宿主可信科学探针 `execute_host_trusted_scientific_probe`（`scientific_probe_receipt`）；对 `M9.3` 历史 `19` 条越权升级完成重分类勘误（`a5_scientific_probe_vs_schema_and_19_nogood_errata.json`）。 |
| **`F04` (`A4`)** | `M9.3` 验证器存在 `hereditary_weights` / `secant_compliance` 交接脱钩漏洞 | 紧化 `verify_m93_output_contract_and_handoff` 与内联私有验证器（强校验正定非退化 Prony 权重、曲线—初速一致性及 `secant_compliance` 公式恒等式）；对封存 `M9.3` 全部 `28/28` 个 episode 重评后 **`28/28` 仍严格通过（`0` 翻转）**（`m9_3_28of28_tightened_handoff_regrade.json`）。 |
| **`F05` (`A1/A2/A6`)** | 单一 `L1` 漏斗漏计 `1,021` 次 `L2` 运行时调用 | 实现多标签源感知漏斗 `classify_source_aware_adoption_funnel`（区分 `new_source_L2`、`new_source_L1`、`prior_v7_L1` 四阶段漏斗），并在 `docs/48b_*` 明确勘误基线定义（`m9_3_source_aware_l1_l2_adoption_funnel.json`）。 |
| **`F06` (`A7/A8`)** | `_LLM_ERROR` 尝试漏计与 `attempts_*.json` Stage S 碰撞（`88,471` 未匹配代理开销） | 为每次调用分配唯一 `attempts_<seq>_<uuid8>.json`，统计 `_LLM_ERROR` 与编译器 `provider_attempt_receipts`，并仅归档白名单尝试文件（`quarantined_unmatched_attempt_files == 0`）。 |
| **`F07` (`A3`)** | 强制耦合 `(w, cert)` 返回签名阻碍最终文件静态留存 | 增加 `build_box_simplex_kkt_certificate_from_weights` 与 `include_optional_witness_guidance`（默认返回核心解，`return_certificate=True` 时附带证书）。 |
| **`F08` (`A9/A10`)** | 冷启动进程 `_VERIFIED_RUNTIME_L1_ENV_DIGESTS` 未初始化及双进程复现实验口径 | 增加 `ensure_runtime_qualification_context`、运行态容器 `.Image` 反查与双独立 `python3 -I` 进程（`PYTHONHASHSEED=101` vs `102`）同输入重放 + 输入不可变校验（拒绝无种子随机函数）。 |
| **`F09` (`A11/A12`)** | 定义域守卫 AST 提升（`if cond: raise ValueError(...)`）与不相交开发种子插件对照 | 更新 `lift_python_scientific_expressions_to_programs` 保留前置定义域守卫，并在不相交开发种子 `[88512, 88521, 88611, 88621]` 上验证直接浮点插件全数失败（`f06_f07_f08_f09_ast_and_hipporag_controls.json`）。 |
| **`F10`** | 代理首轮探索 `pickle` 截断浪费轮次 | 容器启动阶段对所有臂统一生成 `/holdout_task/public_asset_index.json` 并写入 `public_setup_receipt["public_asset_index_path"]`。 |

---

## 3. `M9.4` 持久机制—可执行绑定持续世界设计（提交 `4817cc2a1`）

### 3.1 隐藏日程持续流设计（中性 Task ID，零阶段/公式剧透）

构建两条独立的持续科学任务流，每条流先盲跑 `Stage S` 源发现并冻结 `v9` 森林与初始绑定图谱，随后在持久状态上顺序经历 `ep01 -> ep02 -> ep03 -> ep04`（对应隐藏日程 `T -> G -> C -> R`，所有任务名采用完全中性的业务批次命名）：

- **Chain Alpha（多实验室聚合物试样遗传蠕变 + 共享载荷传感器校准流）**：
  - `alpha_s00__polymer_coupon_creep_baseline`（`S`）：名义载荷校准（`param_v1`），双支路变步长遗传记忆 + 参照试样去混杂 + Cauchy 脉冲抗差反演。
  - `alpha_ep01__preloaded_coupon_compliance_batch`（`T`）：非零初始记忆预载 + 几何时间步 + `secant_compliance` 与 `hazard_crossing_times` 输出契约迁移（`critical_strain` 与未来末端应变严格独立）。
  - `alpha_ep02__facility_recalibrated_coupon_batch`（`G`）：全设施四实验室载荷传感器增益/偏置漂移（`gains=(1.28, 0.82, 1.18, 0.88)`，`biases=(0.14, -0.12, 0.09, -0.11)` -> 触发 `M8` `AncestorReweight` `param_v1 -> param_v2`）。
  - `alpha_ep03__high_load_structural_coupon_batch`（`C`）：超屈服二次过应力与塑性功累积反例（`guard(C0) == FALSE` -> 在 `linear_viscoelastic_memory` 组件上分裂 `C1` 分支）。
  - `alpha_ep04__standard_load_fresh_coupon_batch`（`R`）：**仍处于 `G` 校准纪元（`param_v2`）内**的全新亚屈服试样批次（`specimen_reset=True`，`yield_stress=2.85`；`guard(C1) == FALSE` 且 `guard(C0) == TRUE` -> 精准路由 `C0@param_v2`；若误用 `Stage S` 旧参数 `param_v1` 则私有验证器必定失败）。

- **Chain Beta（二维多频移反应—输运层析 + 带证书 Box-Simplex KKT 流）**：
  - `beta_s00__tracer_transport_tomography_baseline`（`S`）：名义传感阵列校准（`param_v1`），一维调和扩散/Robin/迎风对流/反应算子 Kronecker Sylvester 频移求解 + 干扰子空间投影 + Box-Simplex KKT 对偶证书。
  - `beta_ep01__anisotropic_multibudget_survey_batch`（`T`）：各向异性 `48 x 36` 反向对流网格 + 三预算帕累托 KKT 证书列表迁移。
  - `beta_ep02__recalibrated_sensor_array_survey_batch`（`G`）：四传感组电子增益/零点重标定（`gains=(1.24, 0.84, 1.15, 0.89)`，`offsets=(0.08, -0.06, 0.05, -0.07)` -> 触发 `M8` `AncestorReweight` `param_v1 -> param_v2`）。
  - `beta_ep03__dual_tracer_reactive_survey_batch`（`C`）：含强非可分离二维耦合势 `V_channel_b(x,y)` 的双通道反例（`nonseparable_2d_additive_residual_ratio = 0.5455 > 1e-6` -> `guard(C0) == FALSE` -> 在 `separable_kronecker_transport` 组件上分裂 `C1` 分支）。
  - `beta_ep04__single_tracer_background_survey_batch`（`R`）：**仍处于 `G` 校准纪元（`param_v2`）内**、含**非零可加分离背景势** $V_{\text{bg}}(x,y)=v_x(x)+v_y(y)>0$ 的单示踪勘测批次（`F02` 守卫实测残差比 `1.53e-16 <= 1e-6` -> `guard(C1) == FALSE` 且 `guard(C0) == TRUE` -> 将一维边际精确并入 `(reaction_x, reaction_y)` 并路由 `C0@param_v2`；旧 `param_v1` 必败）。

### 3.2 零宿主求解器不变量与 100% 等信息强基线对等性

- **零宿主手写数值求解器**：`render_bound_executable_adapter` 不包含任何 IRLS、Nelder-Mead、试样配准、FVM 离散格式、Sylvester 特征分解或 KKT 求解逻辑，而是通过 Base64 原样装载并调用 **`gpt-6.2-sol` 在 `Stage S` 盲跑中自主生成的真实 Python 交付文件**（`alpha_s00_solver.py` / `beta_s00_solver.py`），仅做由 `PersistentMechanismGraph` `param_v2` 驱动的输入角色校准映射。
- **100% 等信息对等（`Strong_Archive_Reconstruct` vs `Mechanism_Bound_Continual`）**：双臂共享同一份 `Stage S` 冻结 `v9` 森林包（`b3febcd82e76...`），在容器 `/tmp/hegel_continual_archive/` 中挂载完全相同的历史已验证 `.py` 源码归档，读取相同的公开校准表与 `/holdout_task/public_asset_index.json`，并共享相同的 `18` 轮 `gpt-6.2-sol` 预算与全局 32 路并发池。

---

## 4. `M9.4` 真实持续流实验结果（`docs/evidence/m9_4_continual_mechanism_binding_world_20261126/`）

### 4.1 溯源完整性与计费审计收据

- **运行标识**：`m9_4_continual_20261126T143254Z`（模型 `gpt-6.2-sol`，`reasoning_effort="medium"`，全局 `BoundedSemaphore(32)`）
- **干净冻结提交**：`code_commit_at_start == code_commit_at_end == "4817cc2a1d049ab7240e105fcefdb442c15a1ff4"`，`code_dirty_at_start == false`
- **SHA-256 校验**：`SHA256SUMS.txt` 共 `756/756` 个文件全部 `: OK`（`0` 失配）
- **全局调用台账（`all_provider_call_receipts.json`）**：全流程 `321` 次逻辑调用 = `321` 次底层 Provider 尝试（`0` 重试，**`usage_unknown_attempt_count = 0`（`"complete"`）**，`quarantined_unmatched_attempt_files = 0`）

### 4.2 分阶段持续流核心指标对比（`2 Replicates × 2 Chains = 16 Episodes / Arm`）

| 评估指标 | `Strong_Archive_Reconstruct` (`A_raw`) | `Mechanism_Bound_Continual` (`M_hegel`) | 净差值（`M_hegel - A_raw`） |
| :--- | :---: | :---: | :---: |
| **持续流严格通过率（`Strict Pass`）** | **`16 / 16` (`100.0%`)** | **`16 / 16` (`100.0%`)** | `0.0 pp`（同达满分天花板） |
| `step 1`（`ep01` `[T]`：接口/算子迁移） | `4/4`，均值 `12.00` 轮，`960,553` proxy | `4/4`，均值 `10.50` 轮，`792,225` proxy | **`-1.50` 轮（`-17.5%` proxy）** |
| `step 2`（`ep02` `[G]`：全局共享校准漂移） | `4/4`，均值 `12.25` 轮，`927,735` proxy | **`4/4`，均值 `0.00` 轮，`0` proxy（`4/4` 绑定直通）** | **`-12.25` 轮（`-100.0%` LLM 开销）** |
| `step 3`（`ep03` `[C]`：前提反例与组件分裂） | `4/4`，均值 `11.25` 轮，`843,239` proxy | `4/4`，均值 `10.75` 轮，`824,833` proxy | **`-0.50` 轮（`-2.2%` proxy）** |
| `step 4`（`ep04` `[R]`：`G` 纪元内原机制恢复） | `4/4`，均值 `10.75` 轮，`804,482` proxy | **`4/4`，均值 `0.00` 轮，`0` proxy（`4/4` 绑定直通）** | **`-10.75` 轮（`-100.0%` LLM 开销）** |
| **流内总交互轮次（`16` Episodes）** | `185` 轮（均值 `11.56` 轮） | **`85` 轮（均值 `5.31` 轮）** | **`-100` 轮（`-54.05%`）** |
| **流内计费代理 Token（`w_cached = 0.25`）** | `3,536,009` | **`1,617,058`** | **`-1,918,951`（`-54.27%`）** |
| **单流平均端到端墙钟耗时（`4` 步合计）** | `2,496.28 s`（`41.6` 分钟） | **`1,044.70 s`（`17.4` 分钟）** | **`-1,451.58 s`（`-58.15%`，提速 `2.39x`）** |

### 4.3 全生命周期逐笔成本对账与盈亏平衡验证（`all_provider_call_receipts.json`）

| 生命周期阶段 / 成本桶 | 逻辑调用数 | Provider 尝试数 | 未知用量数 | Input Tokens | Cached Input Tokens | Output Tokens | 计费代理开销 (`w=0.25`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. 共享 `Stage S` 盲源探索（`alpha_s00`, `beta_s00`）** | `21` | `21` | `0` | `500,929` | `146,304` | `20,100` | `411,301` |
| **2. 流前 `Stage S` `L2` 扬弃编译（`alpha`, `beta`）** | `2` | `2` | `0` | `47,816` | `15,744` | `13,474` | `49,482` |
| **3. 流前 `Stage S` `L1` 原语提炼（生成 `2` 个四元组 `L1`）** | `2` | `2` | `0` | `47,202` | `14,976` | `11,740` | `47,710` |
| **4. `A_raw` 持续流求解（`16` Episodes）** | `185` | `185` | `0` | `4,413,609` | `1,369,344` | `149,408` | `3,536,009` |
| **5. `A_raw` 在线回合后 `L2` 归档编译** | `20` | `20` | `0` | `1,262,766` | `637,312` | `137,026` | `921,808` |
| **6. `M_hegel` 持续流求解（`16` Episodes，含 `8` 次零轮直通）** | `85` | `85` | `0` | `2,044,698` | `664,448` | `70,696` | `1,617,058` |
| **7. `M_hegel` `ep03` `C1` 组件分裂 `L2` 编译** | `6` | `6` | `0` | `278,799` | `102,144` | `40,409` | `242,600` |
| **全实验总计（`1..7` 精确相加）** | **`321`** | **`321`** | **`0`** | **`8,595,819`** | **`2,950,272`** | **`442,853`** | **`6,825,968`** |

- **全口径生命周期净收益验证**：
  - 按流内求解 + 在线巩固同口径对比（`4+5` vs `6+7`）：`M_hegel` 总计 **`1,859,658`** vs `A_raw` **`4,457,817`**，净省 **`2,598,159` 计费代理 Token（`-58.28%`）**。
  - 即便采用最严苛的不对称计费口径——将流前 `Stage S` 的 `L2 + L1` 编译成本（`2 + 3 = 97,192`）与流内 `C1` 分裂编译成本（`242,600`）**100% 计入 `M_hegel`**（合计 `1,956,850`），同时免除 `A_raw` 的一切编译开销（仅计 `4`：`3,536,009`）——`M_hegel` 仅在 `2` 个复现流内便已实现 **`1,579,159` 计费代理 Token（`-44.66%`）的净正向摊销盈余**！

### 4.4 `M8 + M9` 机制耦合链路的四项取证核验

1. **`M8` `AncestorReweight` 祖先重加权与 `0.0` 哨兵漂移**：
   - 全部 `4` 条 `M_hegel` 持续流均精确走出 `NoChange (v1) -> AncestorReweight (v2) -> NoChange (v2) -> NoChange (v2)`（Alpha `ep02` 验证集对数预测增益 `+0.8583` nats，Beta `ep02` 增益 `+0.6732` nats），且无关任务 `sentinel_acoustic_calibration` 在全部 `16/16` 个回合中的最大绝对预测漂移恒为 **`0.0`**。
2. **`F01/F02` 显式父节点与组件级 `C1` 分叉**：
   - 全部 `4` 份最终检查点 `graph_ckpt_after_step_04.json` 均通过 `local_implication_edges` 将 `C1` 节点精确挂接在真实 `Stage S` 父节点上（Alpha：`node_auto_11_split_yield_gated_hereditary_creep -> node_auto_07_control_panel_deconfounding`，`violated_component_id="linear_viscoelastic_memory"`，`guard="> 0.0"`；Beta：`node_auto_11_split_* -> node_auto_10_calibrated_transport_inversi`，`violated_component_id="separable_kronecker_transport"`，`guard="> 1e-6"`）。
3. **`Stage R`（`ep04`）在 `{C0, C1}` 之间的三值守卫精准路由**：
   - `alpha_ep04`：`guard(bind_alpha_c0) == TRUE`（`0.0 <= 0.0`）且 `guard(bind_alpha_c1_alpha_ep03) == FALSE`，直通调用 `bind_alpha_c0@param_v2`（宿主科学探针 `public_strain_mae = 0.007696`、`panel_affine_gauge_max_diff = 0.0`，`0` 轮通过私有验证器）。
   - `beta_ep04`（含非零可加分离势 $V_{\text{bg}}(x,y)=v_x(x)+v_y(y)$）：`guard(bind_beta_c0) == TRUE`（实测二维非可加残差比 `1.5324e-16 <= 1e-6`）且 `guard(bind_beta_c1_beta_ep03) == FALSE`，直通调用 `bind_beta_c0@param_v2`（`public_w_rel_l2 = 1e-8`、`kkt_duality_gap = 0.0`，`0` 轮通过私有验证器）。
4. **多标签技能漏斗透明说明（`F05`）**：
   - 在 `M_hegel` 的 `16` 个持续流回合中，`8` 个回合（`step 2` 与 `step 4`）通过 `MechanismExecutableBinding` 直接零轮次执行闭合（`bound_exec = 8/16`）；其余需要接口改造或新物理定律综合的 `8` 个回合（`step 1` 与 `step 3`）中，由于 `/tmp/hegel_continual_archive/` 已按等信息原则直接挂载了完整历史 `.py` 求解器源码，双臂模型均直接读取并改造归档 `.py` 源码，而未额外通过 `/tmp/hegel_wormhole_skills` CLI 导入包（`insp = 0, exec = 0`）。
