# Hegel Machine `V17.4 (M4.2)` 独立红队复核交接书（GPT-6 Audit Handoff）

> **审计基线与只读边界声明**：本版本构建于已审计提交 **`99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c`** 之上，逐项回应 `markdown/Hegel_M4_1_Audit/GPT_diagnosis_M4_1.md`、`M4_1_AUDIT.md` 及四套反事实探针（`closure/probe_current_semantics.py`、`closure/probe_relative_escape.py`、`pi/probe_pi_paths.py`、`unified/probe_unified.py` 与 `part2/` 清单）。  
> **100% 原始只读保护**：前序全部静态证据目录（`docs/evidence/v16_8_*` 至 `docs/evidence/v17_3_*`）及全部三套红队审计目录（`markdown/hegel_M3_Audit/`、`markdown/Hegel_M4_Audit/`、`markdown/Hegel_M4_1_Audit/`）均保持 **100% 字节级未改动**；本轮 `V17.4 (M4.2)` 证据独立归档于 `docs/evidence/v17_4_m4_2_unified_closure_and_pi_lifecycle_20261120/`，且复算脚本支持 `--out-dir <tmp_dir>` 只读外部重放。

---

## 1. 复核范围、源码 SHA-256 清单与证据地图

本轮 `V17.4 (M4.2)` 交付四项紧密咬合的闭环与跨支柱升级：
1. **Part 1 — `Hegel_M4_1_Audit` 全量 12 项执行接线、沙箱边界、外部确认证明与计量口径闭环**：
   - `program_steps` 参数（`fresh_points`、`calibrate_lens`、`operator_tag`）真实接入采样与编译执行；
   - `authorize_abstraction_use` 原生求值 `FidelityCompression` 提交的析取分支 `disjunctive_branches`（`A=True, B=True, C=False`）；
   - `authored_method.json` 密封宿主确认包 `host_confirmation_bundle` + 外部 `confirmed_manifest.json` + 参数快照 `confirmed_fitted_params.json`，`load_verified_authored_method` 强制隔离子进程重放核验并绑定节点注册 `expected_ast_digest`（拦截无确认伪造 JSON 与跨节点合法产物调包）；
   - `evaluate_isolated_python_model` 统一执行 `.resolve().relative_to(ws.resolve())` 拦截 `../../../` 相对父路径逃逸；
   - 新增生产级 `revalidate_vertical_edges_with_certificate`，强制核验闭合证书与 `closure_scope != 'not_closed'` 后方重绑纵向 `coarse_graining_edges`。
2. **Part 2 — 去重后 `M3.1` $2\times 2$ 四格检索因子分解（`108` 唯一运行/格，`432` 总运行）**：
   - 仅遍历唯一 `suite["targets"]`（移除 `T25..T28` 双倍拼接），在 `4` 个评测簇上复核 `{Walk (alpha=0.65), NoWalk (alpha=0.0)} × {Pool, NoPool}`。
3. **Part 3 — 细轨迹投影接线（`window_projected_y`）与全链路盒边界防护的可执行 $\Pi_s$ 重整化**：
   - 区域闭合证书标定直接接入 RK4 细轨迹投影（`projections_with_fine_trajectory = 6`），同时考核端点闭合缺陷与快流形轨迹奴役缺陷；
   - 召回、回退（盒外确认自动升尺度至 `micro_fine`）与导出模型契约（`model_scope_contract.json` 拦截盒外测试查询返回 `out_of_verified_scale_box`）全链路统一强制 `verified_input_box`，且 `without_edge` 回退零重复购点。
4. **Part 4 — 跨支柱统一持续森林（内嵌 $\Pi_s$ + 零 `fit()` 实例级 `ForestScopeDispatcher` + 对称两跳抽屉分岔 `80` 目标迁移 + 实测累计回本曲线）**：
   - **同一森林跨四支柱**：实测计费 `Stage 0`（原子标定 + $\Pi_s$ 细/粗节点与 `coarse_graining_edge` 入林）$\to$ `Stage 1`（`CodeAgent` 开放合成 `W1..W5` 五类复合算子入同一森林）$\to$ `Stage 2`（`Monod` 根作用域收缩 + `'thermal' | 'hydraulic'` 析取压缩 + `3` 个异构域付费 DEV 诱饵淘汰 + $T=0.12$ 淬火破裂与 `revalidate_vertical_edges_with_certificate` 重验证 + 独立 `DEV` 静态序调优冻结）；
   - **Act 3（`ForestScopeDispatcher` — 真实例级 `R3`）**：命名空间删去 `fit`（`any_fit_called = False`），对 `10` 条交错查询流执行真实门控与冻结参数推理（`10/10` 精确路由，授权通过率 `1.0000`）；
   - **Stage 3（`4` 种子 × `20` 域内 + `2` 异构 OOS × `9` 臂 = `792` 次运行）**：每种子严格还原 `Stage 2` 森林快照；在含对称两跳起始分岔的目标集上，`Hegel_Unified_Continual_Forest`（**`29.2200`**）大幅击败独立 `DEV` 冻结最优静态序 `Ordinary_AllHistory_DevFrozenOrder`（**`38.8550`**，节省 **`-24.8%`**），并在实测 `80` 目标流内直接跑穿全生命周期累计回本点（对比开放重发现 **`N* = 29`**，对比保守算子池回退 **`N* = 39`**）。

### 核心源码、脚本、测试与报告 SHA-256 清单

| 相对仓库路径 (`Hegel Machine/`) | 角色说明 | `SHA-256` |
| :--- | :--- | :--- |
| `src/hegel_machine/cognition/wormhole_forest.py` | `authorize_abstraction_use` 原生求值 `disjunctive_branches`、`compress_compatible_nodes` 公共不变量保留、新增 `revalidate_vertical_edges_with_certificate` | `157dbe8490bdb3c8792e735d1c660f1fae475c242628a78da840eef1b7017319` |
| `src/hegel_machine/cognition/m4_compositional_discovery.py` | 相对父路径越界拦截、`fitted_params` 快照、`host_confirmation_bundle` + `confirmed_manifest.json`、隔离子进程确认重放校验、`expected_ast_digest` 绑定、`predict_with_frozen_instance`（删去 `fit`） | `e72212f535c9e3879dc6c47d2380b63fd8cb49271857633a5b7bc0b033a5841f` |
| `src/hegel_machine/cognition/m4_multiscale_renormalization.py` | `_exec_plan`（`fresh_points`、`calibrate_lens`、`operator_tag`）真实驱动采样与拟合、保留失败确认行避免重复购点 | `18f00d65277ac6e779c658b58c6c9a01641c52fb8bbea7c91431d88f816d508e` |
| `src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py` | $\Pi_s$ 细轨迹投影接线、召回/回退/导出全链路 `verified_input_box` 门控、`without_edge` 零重购、节点 `expected_ast_digest` 绑定、全分支保留 `cf0_rows` | `f40c5fcb6cdd1ce8cd335497f16a666130567b8ae14bd7ad85e294af05b94578` |
| `src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py` | **新增核心模块**：去重 `M3.1` 四格、细轨迹 $\Pi_s$ 区域标定、零重拟合 `ForestScopeDispatcher`、四支柱统一森林演化与 `80` 目标九臂迁移及回本曲线 | `922f3f94d3ffda3dbe40c55685c9e3832bda3bf8a636aee2de18a0b8ac7a0707` |
| `scripts/run_v17_4_m4_2_unified_study.py` | `V17.4 (M4.2)` 全量复算脚本（支持 `--out-dir` 只读外部重放）与 `SHA256SUMS.txt` 生成器 | `e841f0799b704566847704a2846574c750b739b81685e5b90b778167b45e146d` |
| `tests/test_v17_3_m4_1_unified_chain_and_audit_closure.py` | 同步合成测试夹具以匹配宿主确认包校验 | `929320aea915d3be61ee44681669073f285443735a7acfee8e795074c0e91c6b` |
| `tests/test_v17_4_m4_2_unified_closure_and_pi_lifecycle.py` | **新增回归套件**：锁死 `Hegel_M4_1_Audit` 全部探针修复、去重 `M3.1`、零 `fit()` 分发器与跨支柱统一链 | `929bec0eb6a857c699058e69b3f5ccf3513d6feda5cfdaa80c227b8d19fdcb78` |
| `markdown/hegel_v16_5_multi_precision_plan/16_V17_4_M4_2_Audit_Closure_Unified_Pi_and_Dispatcher_Report.md` | `V17.4 (M4.2)` 完整技术与实证报告 | `d6040a109deab4598046812e21c560bcbb89bc801805d953584cb34eb5591d8f` |

### 新增只读证据包导航（共 `115` 个受校验归档文件 + `SHA256SUMS.txt`）
- **目录**：`docs/evidence/v17_4_m4_2_unified_closure_and_pi_lifecycle_20261120/`
  - `study_manifest.json`（核心指标与运行耗时总览）
  - `part1_m4_1_audit_closures.json`（`Hegel_M4_1_Audit` 全部反事实控制点复测结果）
  - `part2_deduplicated_m3_1_2x2_factorization.json` + `part2_m3_1_unique_432_runs.jsonl`（去重后 `108 × 4 = 432` 次唯一运行明细）
  - `part3_executable_pi_s_and_guards.json`（可执行 $\Pi_s$ 套件、细轨迹奴役证书及盒外确认/导出拦截反事实）
  - `part4_unified_cross_pillar_lifecycle_and_breakeven.json` + `part4_unified_execution_receipts.jsonl`（四支柱生命周期、零 `fit()` 分发器 `10` 条收据、九臂 `792` 条目标执行收据与逐里程碑回本曲线）
  - `persisted_authored_workspaces/`（归档 `stage0_metered_sources/` 与 `persisted_stage1_authored_workspaces/` 下全部真实落盘工作区，含 `authored_method.json`、`confirmed_manifest.json`、`confirmed_fitted_params.json`、`confirmed_observations.json`、`model_scope_contract.json`；注：`run_v17_4_m4_2_unified_study.py` 运行期在 `TemporaryDirectory` 内执行以确保 `--out-dir` 重放绝不污染源码树，随后完整拷贝至此子目录）
  - `SHA256SUMS.txt`（本次发布的 `115` 个归档文件的 SHA-256 校验表）

---

## 2. `Hegel_M4_1_Audit` 全部 12 项审计发现逐一闭环核对表

在隔离 `/tmp` 目录原样拷贝并复跑 `markdown/Hegel_M4_1_Audit/` 下全部四套探针脚本，修复前后对比如下：

| 审计探针定位 | `V17.3` 审计发现表现 | `V17.4 (M4.2)` 修复后实测表现 |
| :--- | :--- | :--- |
| **1. `closure/probe_current_semantics.py`**<br>`valid_program_parameter_consumption` | `fresh_points: 1` 与 `2` 结果及购点完全相同 (`equal: True`, `confirm_count: 1`) | **`equal: False`**，`fresh_1 -> actual_confirm_count = 1 (40.0)`，**`fresh_2 -> actual_confirm_count = 2 (44.0)`** |
| **2. `closure/probe_current_semantics.py`**<br>`fidelity_real_authorizer_consumption` | 顶层合取保留 `[A, B]` 矛盾前提，`auth(after, 'A')` 与 `'B'` 均为 `granted: False` | 顶层 `preconditions = []`，`disjunctive_branch_count = 2`，真实 `authorize_abstraction_use` 返回 **`A: True, B: True, C: False`** |
| **3. `closure/probe_current_semantics.py`**<br>`authored_artifact_confirmation_provenance` | `confirmation_digest="no_confirmation_was_run"` 被加载器接受 (`loader_accepts: True`) | 校验 `host_confirmation_bundle` + 隔离重放确认集，**`loader_accepts: False`** |
| **4. `closure/probe_current_semantics.py`**<br>`node_to_artifact_binding` | 原地将 `W1` 文件替换为合法 `W2` 后返回 `W2` (`after_accepts_other_verified_W2: True`) | `_load_node_verified_operator_spec` 核对节点 `expected_ast_digest`，`W1` 接受 (`True`)，调包 `W2` **拒绝 (`False`)** |
| **5. `closure/probe_relative_escape.py`**<br>相对父路径沙箱边界 | `../../../.../path_outside.py` 被执行 (`outside_loaded: True`) | 立即抛出 **`ValueError: Model path escapes workspace boundary`** |
| **6. `closure/probe_current_semantics.py`**<br>`vertical_revalidation_boundary` | 低层 `revalidate_vertical_edges_after_revision` 不检查 `closure_scope == 'not_closed'` | 新增生产接口 `revalidate_vertical_edges_with_certificate`：`not_closed` **拒绝 (`accepted: False`)**，闭合证书核验通过后恢复 (`accepted: True`) |
| **7. `part2/suite_identity_inventory.json`**<br>`M3.1` 游走目标重计 | `suite["targets"] + suite["walk_targets"]` 使 `T25..T28` 计两次 (`124` runs) | `run_deduplicated_m3_1_2x2_factorization` 仅跑唯一 `suite["targets"]`（**`108` runs/cell，`432` total**），`Walk_Pool = 37.4591` 仍居四格最优 |
| **8. `pi/probe_pi_paths.py`**<br>`outside_confirm_fallback` & `outside_heldout_query` | 盒外确认 ($T=1.2$) 或盒外测试 ($T=0.12$) 产生 `false_confirmation: True`；`without_edge` 重购 4 行试探点 | 盒外确认自动升至 `micro_fine`（**`strict_pass: True, false_confirmation: False`**）；盒外导出查询返回 **`status: "out_of_verified_scale_box", false_confirmation: False`**；`without_edge` **`duplicate_acquired_rows: 0`**（`72.6095 < 73.7765` 仍严格胜出） |
| **9. `pi/probe_pi_paths.py`**<br>`projections_with_fine_trajectory` | 细轨迹投影 `window_projected_y` 调用数为 `0` | 区域盒标定直接接入 RK4 细轨迹，**`projections_with_fine_trajectory = 6`**（`m4_2`）/ `3`（`m4_1`） |
| **10. `unified/probe_unified.py`**<br>`paid_prefix_loss` & `Ordinary` 二轮召回 | `Alpha0` 丢失 `cf0_rows` 并重购确认点（`8` obs vs `7` obs）；`Ordinary` 二轮未记入召回 | 全分支保留已付 `cf0_rows` 并过滤已证伪 `refuted_tags`（`original == retained = 7 obs, 37.3113`）；二轮统一记入召回 |
| **11. `unified/probe_unified.py`**<br>`r3_parameters` & `diagnostic_rounds` | `W1` 未存拟合参数（不调 `fit()` 误差 `0.6577`）；`diagnostic_rounds` 用 `len(composed_history)` 近似 | 落盘 `confirmed_fitted_params.json` 并由 `predict_with_frozen_instance`（删去 `fit`）直推 **`W1` 误差 `0.0`**；独立统计真实 `provider._diag_rounds` |
| **12. `GPT_diagnosis_M4_1.md` §3.4**<br>`Stage 0` 常数、重复诱饵、跨种子状态、静态序对比 | `Stage 0 = 180.0` 硬编码、3 次诱饵重复、种子间未还原快照、精调静态序 `36.425` 接近 `Hegel` `36.240` | `Stage 0 = 244.05` 全实测计费、3 个异构域诱饵、每种子还原 `post_stage2_forest_snapshot`、独立 `DEV` 冻结静态序 **`38.8550` vs `Hegel` `29.2200`（拉开 `-24.8%`）**、实测跑穿回本点 **`N* = 29 / 39`** |

---

## 3. 核心实验数据总览

### 3.1 Part 2：去重后 `M3.1` $2\times 2$ 四格检索因子分解（`108` 唯一运行/格，`432` 总运行）

| 检索模式 | 唯一运行数 | `strict_pass_rate` | `false_confirmation_rate` | 唯一游走胜出 (`T25..T28`) | 游走子集复合成本 | 全集平均测量费 | 全集平均计算费 | **全集平均复合成本** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`hipporag_ppr` ($\alpha=0.65$, `Pool=True`)** | `108` | `1.0000` | `0.0000` | **`14 / 16`** | **`44.7579`** | **`35.6481`** | **`36.2192`** | **`37.4591`** |
| `hipporag_walk_nopool` ($\alpha=0.65$, `Pool=False`) | `108` | `1.0000` | `0.0000` | `14 / 16` | `44.7579` | `35.8333` | `37.2377` | `37.6952` |
| `hipporag_alpha0_nopool` ($\alpha=0.0$, `Pool=False`) | `108` | `1.0000` | `0.0000` | `0 / 16` | `64.3025` | `37.9630` | `46.4741` | `40.2867` |
| `hipporag_alpha0_pool` ($\alpha=0.0$, `Pool=True`) | `108` | `1.0000` | `0.0000` | `0 / 16` | `64.3025` | `37.9630` | `46.8444` | `40.3052` |

### 3.2 Part 4 Act 3：`ForestScopeDispatcher` 零 `fit()` 实例级 `R3` 交错重访（`10` 条查询）

- **`any_fit_called = False`**（`predict_with_frozen_instance` 在 `exec` 后显式 `ns.pop("fit", None)` 并加载 `confirmed_fitted_params.json`）；
- **`all_routes_exact = True`（`10 / 10`）**，授权查询 `within_tolerance` 通过率 **`1.0000`（`9 / 9`）**，最大绝对误差 **`0.009287`**（旧 `stage0_monod_2d` 与新 `stage1_W1_haldane_shunt` 误差均为 `0.0`；析取压缩后的 `ode_meso_schur_pi_thermal` v4 同时授权 `'thermal'` 与 `'hydraulic'`；短时淬火 $T=0.12$ 自动拒绝介观盒并升尺度至 `ode_micro_2mode_node`；越界模态 `'optical'` 返回 `status = "scope_unauthorized"`）。

### 3.3 Part 4 Stage 3 & 全生命周期回本：`80` 域内 + `8` 异构 OOS × `9` 臂（`792` 次运行）

| 评测臂 | 严格通过率 (`80`) | 假确认率 | OOS 诚实率 (`8`) | 召回确认数 | 非直接游走胜出 | 平均候选评估数 | 平均测量费 ($\lambda=0$) | 平均计算费 | **Stage 3 平均复合成本** | **`80` 目标累计全生命周期总成本** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Hegel_Unified_Continual_Forest`** | **`1.0000`** | **`0.0000`** | **`1.0000`** | **`80 / 80`** | **`48 / 80`** | **`1.9000`** | **`28.0000`** | **`24.4000`** | **`29.2200`** | **`2,990.325`** |
| `Hegel_NoMaintenance_SameHistory` | `1.0000` | `0.0000` | `1.0000` | `64 / 80` | `32 / 80` | `3.5000` | `28.2500` | `34.0938` | `29.9547` | `2,830.650` |
| `Ordinary_AllHistory_Reversed` | `0.9875` | **`0.0125`** | `1.0000` | `80 / 80` | `0 / 80` | `4.1000` | `34.4000` | `37.9000` | `36.2950` | `3,337.875` |
| **`Ordinary_AllHistory_DevFrozenOrder`** | `1.0000` | `0.0000` | `1.0000` | `64 / 80` | `0 / 80` | `5.1875` | `36.6000` | `45.1000` | **`38.8550`** | `3,894.175` |
| `Forest_Alpha0_SameHistory` | `1.0000` | `0.0000` | `1.0000` | `32 / 80` | `0 / 80` | `6.7000` | `36.8000` | `54.4000` | `39.5200` | `3,814.325` |
| `Flat_BM25_SameHistory` | `1.0000` | `0.0000` | `1.0000` | `32 / 80` | `0 / 80` | `6.7000` | `36.8000` | `54.4000` | `39.5200` | `3,814.325` |
| `Frozen_After_Stage0_PoolFallback` | `1.0000` | `0.0000` | `1.0000` | `0 / 80` | `0 / 80` | `9.5000` | `36.8000` | `71.2000` | `40.3600` | `3,472.850` |
| `Ordinary_AllHistory_Forward` | `1.0000` | `0.0000` | `1.0000` | `64 / 80` | `0 / 80` | `14.3875` | `35.8000` | `100.1500` | `40.8075` | `3,698.875` |
| `Frozen_After_Stage0_OpenRediscovery` | `1.0000` | `0.0000` | `1.0000` | `0 / 80` | `0 / 80` | `9.5000` | `41.6000` | `95.2000` | `46.3600` | `3,952.850` |

- **实测观测回本点（零线性外推）**：`Hegel` 前期实测总投入为 `Stage 0 (244.05) + Stage 1 (190.225) + Stage 2 (218.45) = 652.725`（对比 `Frozen_After_Stage0` 的 `244.05`，净持续投入 `408.675`）。在 `80` 个实际执行目标中：
  - **第 `29` 个目标**反超 `Frozen_After_Stage0_OpenRediscovery`（`1,498.875 < 1,511.350`），`80` 目标净省 **`+962.525`**；
  - **第 `39` 个目标**反超免费享用算子池的保守下界 `Frozen_After_Stage0_PoolFallback`（`1,792.575 < 1,800.900`），`80` 目标净省 **`+482.525`**。
  - 注：`Hegel_NoMaintenance_SameHistory` 因未支付一次性 `Stage 2` 淬火重标定与 DEV 诱饵探测费（`218.45`），在 `80` 目标处的累计和（`2,830.650`）仍略低于 `Hegel`（`2,990.325`），但在 `Stage 3` 边际单目标成本上 `Hegel`（`29.2200`，召回 `80/80`）严格优于 `NoMaintenance`（`29.9547`，召回 `64/80`），且具备淬火安全盒保护。

---

## 4. 精确主张口径与诚实边界声明（Claim Calibration）

为便于红队精确核查，特此明确区分**本版本已实证主张**与**不作越界宣称的边界**：
1. **离线确定性发现代数 vs 在线随机大模型（`M5` 范畴）**：`Stage 1` 仍采用遵守 `CodeAgent` 沙箱协议的确定性残差几何驱动器 `OpenCompositionalCodeProvider` 在受控 `ExprNode` 算子空间内合成 Python 代码，**未调用外部在线商业 LLM API**；挂载真实前沿 LLM `CodeProvider` 与开放科学数据属于下一阶段 **`M5`** 的核心任务。
2. **两级 `R3` 严格分立**：**Act 3（`ForestScopeDispatcher`）** 主张并证明了真正的**零 `fit()` 实例级冻结参数与作用域路由重访**；**Stage 3** 则明确作为**跨领域参数化算子模板迁移**（目标系统具有全新物理参数与未知传感器仿射变换，必须在目标付费 `pilot` 上拟合参数与透镜）。
3. **持续森林快照与多种子评估关系**：`Stage 0 -> Stage 1 -> Stage 2` 构成智能体的一次确定性持续知识积累生命周期；进入 `Stage 3` 时，在 `4` 个独立测试随机种子（`20261120, 20271120, 20281120, 20291120`）的起点**严格逐种子 `deepcopy` 还原同一个 `post_stage2_forest_snapshot`**，确保跨种子无状态串扰。
4. **有限区域网格闭合证书的数学边界**：`6` 点非共线区域网格（覆盖四角与两个内部斜对角点）+ 细轨迹奴役核验能捕获有限宽度的非绝热破裂与内部共振凸起，但正如黑盒数值分析常识，在无先验全局解析高阶导数上界时，任何有限采样点均不宣称对任意测度为零的窄尖峰提供形式化全局证明。

---

## 5. 一键只读复现与探针核验命令

在 `Hegel Machine/` 根目录下可直接运行（全程使用 `--out-dir` 或 `mktemp -d`，绝不修改任何已提交 `docs/evidence/*` 或 `markdown/*Audit*/`）：

```bash
# 1. 校验 V17.4 (M4.2) 静态证据包发布的 115 个归档文件的 SHA-256
(cd docs/evidence/v17_4_m4_2_unified_closure_and_pi_lifecycle_20261120 && sha256sum -c SHA256SUMS.txt)

# 2. 运行 M0 -> M4.2 全套 37 项单元与集成回归测试（37 passed）
PYTHONPATH=src PYTHONHASHSEED=0 python3 -m pytest \
  tests/test_v16_6_m0_contracts_and_wormholes.py \
  tests/test_v16_6_m1_cross_domain_wormhole_study.py \
  tests/test_v16_8_m2_open_workspace_wormholes.py \
  tests/test_v16_9_m2_audit_closure_and_open_revision.py \
  tests/test_v17_0_m3_hipporag_wormhole_scaling.py \
  tests/test_v17_1_m3_1_audit_closure_and_walk.py \
  tests/test_v17_2_m4_multiscale_and_open_discovery.py \
  tests/test_v17_3_m4_1_unified_chain_and_audit_closure.py \
  tests/test_v17_4_m4_2_unified_closure_and_pi_lifecycle.py -v

# 3. 单独快速运行 V17.4 (M4.2) 新增 4 项审计闭环、去重 M3.1、零 fit 分发器与跨支柱测试（约 90 秒）
PYTHONPATH=src PYTHONHASHSEED=0 python3 -m pytest \
  tests/test_v17_4_m4_2_unified_closure_and_pi_lifecycle.py -v

# 4. 在隔离临时目录复跑 Hegel_M4_1_Audit 原版探针（保持 markdown/Hegel_M4_1_Audit 100% 只读）
TMP_M42=$(mktemp -d)
cp -r markdown/Hegel_M4_1_Audit/closure markdown/Hegel_M4_1_Audit/pi markdown/Hegel_M4_1_Audit/unified "$TMP_M42/"
HEGEL_AUDIT_REPO="$(pwd)" PYTHONPATH=src python3 "$TMP_M42/closure/probe_current_semantics.py"
HEGEL_AUDIT_REPO="$(pwd)" PYTHONPATH=src python3 "$TMP_M42/pi/probe_pi_paths.py" --repo "$(pwd)" --out "$TMP_M42/pi_out"
rm -rf "$TMP_M42"

# 5. 只读完整重算 V17.4 (M4.2) 全部四部分实验至临时目录（不触碰已提交证据包）
TMP_EVID=$(mktemp -d)
PYTHONPATH=src PYTHONHASHSEED=0 python3 scripts/run_v17_4_m4_2_unified_study.py --out-dir "$TMP_EVID"
(cd "$TMP_EVID" && sha256sum -c SHA256SUMS.txt)
rm -rf "$TMP_EVID"
```

---

## 6. 提请 GPT-6 重点审阅的 5 个问题

1. **`Hegel_M4_1_Audit` 12 项工程与计量闭环核验**：针对 `program_steps` `fresh_points` 实效消费、`authorize_abstraction_use` 原生 `disjunctive_branches` 析取门控、宿主确认包 `host_confirmation_bundle` + 节点 `expected_ast_digest` 防调包绑定、`../../../` 相对逃逸拦截、`revalidate_vertical_edges_with_certificate` 证书复核、$\Pi_s$ 全链路盒门控与细轨迹投影接线，当前源码是否已彻底闭合全部缺口且无新旁路？
2. **对称两跳起始分岔对独立 `DEV` 冻结静态序的结构性击穿**：在 `80` 个未见目标上，`Hegel_Unified_Continual_Forest`（`29.2200`，纯测量费 `28.00`）大幅击败零测试集窥探的 `Ordinary_AllHistory_DevFrozenOrder`（`38.8550`，纯测量费 `36.60`）与 `Reversed`（`36.2950` 且含 `1.25%` 假确认），是否充分消除了“静态列表排序即可替代森林图游走”的顾虑？
3. **两级 `R3` 分立与 `ForestScopeDispatcher` 零 `fit()` 实证**：显式删去 `fit` 命名空间、基于持久化 `confirmed_fitted_params.json` 与维护后森林契约完成 `10/10` 交错路由（含析取压缩与淬火升尺度），是否扎实兑现了实例级 `R3` 旧知识与新知识无干扰共存？
4. **实测累计回本曲线与生命周期账本**：计入全部实测 `Stage 0 + Stage 1 + Stage 2` 投入后，在 `80` 目标执行期内直接观测到第 `29` 题（对比开放重发现）与第 `39` 题（对比保守池回退）反超回本，账本归因是否完全清晰自洽？
5. **进入 `M5`（真实在线前沿 LLM 代码生成 + 外部真实科学数据集）的准入评估与风险提示**：在离线底座与契约门控全部闭环后，迈向 `M5` 时在自由形式 Python AST 提取、随机 LLM 重试计费与非参数化过拟合防御上最应优先收紧哪些护栏？
