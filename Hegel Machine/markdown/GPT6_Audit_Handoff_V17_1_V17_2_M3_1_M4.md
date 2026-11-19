# Hegel Machine `V17.1 (M3.1)` + `V17.2 (M4)` 独立红队复核交接书（GPT-6 Audit Handoff）

> **仓库基线与静态证据保护说明**：本复核包构建于前序已审计提交 **`612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8`** 之上。前序全部静态证据目录（`docs/evidence/v16_8_*`、`docs/evidence/v16_9_*`、`docs/evidence/v17_0_*`）及红队审计目录 `markdown/hegel_M3_Audit/` 均保持 **100% 原始未改动**；本轮新增两套只读证据包独立归档于 `docs/evidence/v17_1_m3_1_audit_closure_and_walk_20261119/` 与 `docs/evidence/v17_2_m4_multiscale_open_discovery_20261119/`。

---

## 1. 复核范围、模块 SHA-256 清单与证据地图

本次复核覆盖前后紧密衔接的两个里程碑：
1. **`V17.1 (M3.1)`：`hegel_M3_Audit` 全量红队闭环 + 虫洞池化 vs 多跳游走 $2\times 2$ 因子分解 + 单因子 $K$ 扩容**
   - 逐项闭合 `M3_AUDIT.md` 及 `closure/`、`mechanism/`、`results/` 指出的全部技术漏洞与计费偏差（确认阶段文件通道隔离 `F11`、`36/36` 对称纯 AST 规范化与遮蔽拦截、四重收据强绑定、诊断 `stdout` 因果驱动与公平初始 Pilot、合成器单调工作量计费、真实解释执行 `program_steps` 与去标识符内容哈希平局决胜）；
   - 新增 **安全低负载初始试探体制（`T_WalkBridge`，`T25..T28`，$Q_{\text{safe}} \cap \text{sigs}(\text{Child}) = \emptyset$）**，在 `10` 个受控因子臂 × `108` 个目标工作区 = **`1,080` 个目标工作区 + `8` 个域外工作区** 及 **$K \in \{20, 36, 52\}$ 单因子扩容**上完成实证归因。
2. **`V17.2 (M4)`：开放组合式 AST 科学发现 + 主动多尺度粗粒化重整化（$\Pi_s$）与尺度索引虫洞 + $K=64$ 自生长保真压缩森林**
   - **支柱一（开放组合算子演化）**：零预设复合公式表，从原子基元出发由 `CodeAgent` 运行残差几何诊断 `diag_residual.py` 动态施加结构组合子（支持多轮迭代复合），完成 5 组反事实见证结构分叉、原作用域向后保持及持久化产物**删除劣化 / 恢复复原**因果验证；
   - **支柱二（主动多尺度重整化与尺度索引虫洞）**：建立双通道付费实验台（`coarse_macro` `4.0/点` vs `micro_fine` `12.0/点`），让同一系统树在微观极性/快弛豫 $S_{\text{micro}}$、介观六边形变分 $S_{\text{meso}}$、宏观 Murray $2/3$ 分形输运 $S_{\text{macro}}$ 打开完全不同的跨领域虫洞，并在 `7` 个受控臂 × `72` 道目标 = **`504` 个目标工作区 + `8` 个域外工作区**上验证主动闭合探针与微观辩证下钻；
   - **支柱三（$K = 16 \to 32 \to 48 \to 64$ 持续生命周期治理）**：4 阶段持续流在线执行零冲突 `FidelityCompression`、过期虫洞重验与效用账本归档，检验节点体积有界性与历史零遗忘。

### 核心源码、脚本与测试文件 SHA-256 清单

> *注：`wormhole_forest.py` 与 `hipporag_wormhole_index.py` 在 `V17.1` 运行完成后追加了面向 `V17.2 (M4)` 的向后兼容可选扩展（`coarse_graining_edges`、`scale_regime`、`scale_aware`；默认参数下对 `M0–M3.1` 行为零影响，重跑 `run_v17_1_m3_1_factorized_walk_study.py` 指标完全一致）。下表所列为当前工作树最终提交态 SHA-256（与 `V17.2` 的 `summary.json["module_sha256"]` 100% 一致）。*

| 相对仓库路径 (`Hegel Machine/`) | 角色说明 | `SHA-256` |
| :--- | :--- | :--- |
| `src/hegel_machine/cognition/wormhole_forest.py` | 四重收据强制绑定、子规律前置条件反自排斥、`coarse_graining_edges`、`scale_regime` 与 `compress_compatible_nodes` | `8f690a9266264ecde7835b4bfc7ed378ab97de1c175ec5d1bf904888e3f5cd12` |
| `src/hegel_machine/cognition/open_workspace_labs.py` | `TemporaryDirectory` 隔离确认沙箱（严格仅写 `isolated_prefix`，闭合 `F11` 文件通道） | `f3a1e6a84c96e0a4d0153050f55360509cb4a6ba74d467113713aec90f62035b` |
| `src/hegel_machine/cognition/open_workspace_wormhole.py` | 对称 `_strip_outer_affine`、局部遮蔽调用拦截、`round` 精度保留与隔离前缀确认沙箱 | `6ec392b41142048163dbeab03bbfa51630d800cae431b0753a9b234a6804225f` |
| `src/hegel_machine/cognition/open_python_revision.py` | 真实解析诊断 `stdout` 重排/合成算子、移除普通臂初始 `extra_q`（拉平初始测量费 `50 == 50`） | `b2d4deeba662d3e4cb96fa5682374c0105d7015e9bd8bdf869e747e13c988d7c` |
| `src/hegel_machine/cognition/hipporag_wormhole_index.py` | 纯内容哈希 `_node_content_tie_key`、`wormhole_equivalence_pooling` 开关、$\alpha=0$ 支持、`coarse_graining_edges` 与 `scale_aware` | `4cfd306850cc3b9b7dd8e21c7c080c0643b1b1270cac85de7ce3fccf443abdb8` |
| `src/hegel_machine/cognition/m3_scale_universe.py` | 单调工作量合成器计费、`36/36` 纯 AST 虫洞门控、桥接成本入账、真实微步振荡探针、解释执行 `program_steps` 与条件匹配 `mistake_bank` | `29d6daa8b121e0be7252ba34800c88971a88781661479b1d69835a2411dfaab5` |
| `src/hegel_machine/cognition/m3_1_walk_factorization.py` | `T25..T28` 安全低负载试探目标 `T_WalkBridge`、十臂 $2\times 2$ 因子分解与单因子 $K \in \{20, 36, 52\}$ 扩容 | `36c0c21ddd783858f4edce9bde81b1944b88906513fe2dbbcf78f7efb54b1d7e` |
| `src/hegel_machine/cognition/m4_compositional_discovery.py` | 符号算子表达式树 `ExprNode`、6 类结构组合子、多轮残差诊断驱动 `OpenCompositionalCodeProvider` 与持久化删除/恢复对照 | `878779c5c0e19264d354d050016a60fbe024df6067b2052842707d26b9b031c4` |
| `src/hegel_machine/cognition/m4_multiscale_renormalization.py` | 双通道 `MultiScaleWorkspaceTaskLab`、主动尺度闭合探针、三尺度尺度索引虫洞森林、7 臂多尺度评测与 $K=64$ 持续压缩流 | `a42abf4b6795681323502cfcfd5a22758471f0ccde1d83891aa47f0468b65165` |
| `scripts/run_v17_1_m3_1_factorized_walk_study.py` | `V17.1 (M3.1)` 全量复算与归档脚本 | `322f13b9e1f6ccae3c29d77e0aa8e0f311133dff80b7fd4c26bbf9d6886f50a2` |
| `scripts/run_v17_2_m4_multiscale_open_discovery_study.py` | `V17.2 (M4)` 三大支柱全量复算与持久化代码归档脚本 | `481a2c302e1dd349b05c073312f2dc861f2ce9898bf187ba41da5a3de4c28de4` |
| `tests/test_v17_1_m3_1_audit_closure_and_walk.py` | `M3.1` 红队探针回归、标识符无关排序与两跳游走测试 | `cf5ef801af19ed3770ecad83da4f02deed074d4ab24da1a9b763a5b77b3e0b6f` |
| `tests/test_v17_2_m4_multiscale_and_open_discovery.py` | `M4` 反事实见证、删除/恢复因果性、尺度索引虫洞、$S_2$ 置换不变性与 $K=64$ 零遗忘测试 | `4de637a1ad812a6e60ca7fa6829f4dbf720f60ee82cd8299d595e610a65553a0` |

### 新增只读证据包与工程报告导航
- **`V17.1 (M3.1)` 证据目录**：`docs/evidence/v17_1_m3_1_audit_closure_and_walk_20261119/`
  - 含 `summary.json`、`closure_verification.json`、`open_revision_fair_results.json`、`isolated_k_scaling.json`、`dev_gate_calibration.json`、`forest_snapshots.json`、`source_episodes.jsonl`、`target_episodes.jsonl`。
- **`V17.2 (M4)` 证据目录**：`docs/evidence/v17_2_m4_multiscale_open_discovery_20261119/`
  - 含 `summary.json`、`multiscale_7arm_table.csv`、`multiscale_target_runs.json`、`continual_k64_stages.json` 以及完整落盘的 **`persisted_open_author_artifacts/`**（保留全部 10 个开放作者工作区的 `authored_model.py`、`authored_method.json`、`observations.json` 与 `confirmed_training_observations.json` 供逐字节核查）。
- **详细工程报告**：
  - `markdown/hegel_v16_5_multi_precision_plan/13_V17_1_M3_1_Audit_Closure_Factorization_and_M4_Plan.md`
  - `markdown/hegel_v16_5_multi_precision_plan/14_V17_2_M4_MultiScale_OpenDiscovery_and_Lifecycle_Report.md`

---

## 2. 第一部分：`hegel_M3_Audit` 红队审计闭环与 $2\times 2$ 游走归因（`V17.1 / M3.1`）

### 2.1 七项技术漏洞与计费偏差逐项闭环表

| 审计事项与对应原探针 | 根因修复机制 | 未改动探针与回归复验结果 |
| :--- | :--- | :--- |
| **C1. 确认文件通道泄漏 (`F11`)**<br>`closure/probe_closure.py` | `validate_workspace_model_on_rows` 与 `_evaluate_workspace_submission` 改为在独立 `TemporaryDirectory` 沙箱运行，沙箱内 `observations.json` 与 `model_observations.json` 严格只写 `isolated_prefix`（绝不含确认点或留出点）。 | 复跑原版 `probe_closure.py`：三种记忆器确认 `RMSE = 9.0`，`CodeAgent` 全生命周期返回 `scoped_unknown`、`false_confirmation = False`。 |
| **C2. AST `9/36` 误拒 `mismatch:` 与双碰撞**<br>`probe_m3_ast_rejection.py` & `probe_closure.py` | `_CORE_LAW_EXPR` 与 `predict` 对称剥离非零外层仿射壳，森林建桥强制要求 `skeleton_status == "VERIFIED_PURE_AST"`；拦截局部遮蔽定义（`def exp(x):`）并保留 `round(..., ndigits=k)` 精度令牌 `PREC_k`。 | `M3` 全部 **`36/36` 达成 `VERIFIED_PURE_AST`（`0` 个 `mismatch:`）**，横向虫洞恢复完整 `14` 条；两项碰撞 `same_digest` 均为 `False`。 |
| **C3. 收据字段省略与布尔旁路**<br>`probe_closure.py` | `_validate_evidence_receipt` 无条件校验非空 `target_id` 与整型 `node_version`，开放工作区强制核验 `model_sha256` 与 `confirmation_digest`，禁止 `heldout_confirmed=True` 覆盖无效收据。 | `probe_closure.py` 全部 4 种绕过测试均为 `granted = False`。 |
| **C4. 开放修订 `stdout` 无效与普通臂无条件多买点**<br>`probe_open_forced_queries.py` | `ScriptedOpenGrammarCodeProvider` 解析诊断 `stdout` 动态合成/重排算子；普通臂初始 `extra_q` 移至拟合超限才触发的 `fallback_query_batches`。 | `provider_ignores_diagnostic_stdout = False`；两臂初始测量费拉平（`50.0 == 50.0`），`Hegel_OpenRevision_Forest` 复合成本仍严格胜出（`53.23 < 74.70`）。 |
| **C5. `M3` 未消费 `program_steps` 与 `node_id` 平局决胜**<br>`MECHANISM_AUDIT_M3.md` | `M3ForestResearcher` 真实解释执行 `program_steps`；清理 `aa_*` 命名，`HippoRAGWormholeIndex` 与 `FlatEqualMetadataIndex` 统一改用纯内容哈希 `_node_content_tie_key`。 | 清空 `program_steps` 会使 `confirmed_via_recall` 退化并增加实测开销；逆序重命名全部 `node_id` 后召回排序 **`0` 变化**。 |
| **C6. 子集算力倒挂、桥接漏账与尺度常数桩**<br>`probe_cost_accounting.py` | 合成器改为单调工作量计费 `4.0 + 6.0 * evaluated_candidate_count`；`m3_bridge_fit` 写入 `source_episodes`；`scale_probe_fn` 真实积分快振荡载波。 | 子集算力严格低于全集，显式全列表与 `None` 算力 `100%` 相等。 |
| **C7. 拟合/确认预算混同与 `HExA` 错误库不可达**<br>`results_audit.md` | 分离廉价训练拟合淘汰上限（`max_fit_trials = 4`）与昂贵买点确认上限（`max_confirm_purchases = 2`）；`M3HExACodeSkills` 按观测签名优先匹配 `mistake_bank`。 | 删除 `mistake_bank` 会真实劣化 `O_HExA` 目标成本。 |

### 2.2 为什么保留多跳图游走（$\alpha > 0$）：安全低负载试探体制 `T_WalkBridge` 与十臂因子分解

- **机理解释**：`V17.0` 中 $\alpha = 0$ 之所以能命中子节点，仅仅因为其初始 Pilot 直接覆盖了 `[0.06, 0.88]` 高负载区（$Q \cap \text{sigs}(\text{Child}) \neq \emptyset$）。在真实工程与多尺度科学的安全低负载初始试探（`T25..T28` `T_WalkBridge`，初始试探严格在 `[0.005, 0.22]`，$Q_{\text{safe}} \cap \text{sigs}(\text{NM}_A) = \emptyset$）下，所有无游走或无横向虫洞基线对跨树细化子规律 $\text{NM}_A$ 的得分**恒等于 `0.0`**，唯有 $\alpha = 0.65$ 随机游走能沿 $Q_{\text{safe}} \to S_B^{(-1)} \leftrightarrow S_A^{(+1)} \to \text{NM}_A^{(+1)}$ 完成跨树两跳输送：

| 受控实验臂（每臂 `108` 个目标工作区） | `strict_pass` | 假确认 | 总召回确认 (`108`) | **`T_WalkBridge` 两跳游走确认 (`16`)** | 全集平均测量费 | **`T_WalkBridge` 测量费** | 全集平均算力费 | **全集复合成本** | **`T_WalkBridge` 复合成本** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Hegel_M3_1_HippoPPR_Walk_Pool`** | **`1.000`** | **`0`** | **`102/108`** | **`14/16` (`87.5%`)** | **`35.648`** | **`41.875`** | **`36.219`** | **`37.459`** | **`44.758`** |
| `HippoPPR_Walk_NoPool` ($\alpha=0.65$, 无池化) | `1.000` | `0` | `102/108` | **`14/16` (`87.5%`)** | `35.833` | `41.875` | `37.238` | `37.695` | `44.758` |
| `HippoPPR_Alpha0_WormholePool` ($\alpha=0.0$, 虫洞池化) | `1.000` | `0` | `88/108` | **`0/16` (`0.0%`)** | `37.963` | `57.500` | `46.844` | `40.305` | `64.303` |
| `Flat_BM25_WormholePool` (扁平 BM25 + 虫洞池化) | `1.000` | `0` | `84/108` | **`0/16` (`0.0%`)** | `38.704` | `57.500` | `54.440` | `41.426` | `64.382` |
| `Flat_BM25_EqualMetadata` | `1.000` | `0` | `88/108` | **`0/16` (`0.0%`)** | `37.963` | `57.500` | `49.477` | `40.437` | `64.382` |
| `Flat_Cosine_EqualMetadata` | `1.000` | `0` | `88/108` | **`0/16` (`0.0%`)** | `37.963` | `57.500` | `48.366` | `40.381` | `64.382` |
| `Forest_VerticalPPR_NoWormholes` ($\alpha=0.65$, 无虫洞) | `1.000` | `0` | `88/108` | **`0/16` (`0.0%`)** | `37.963` | `57.500` | `47.757` | `40.351` | `64.304` |
| `HippoPPR_UncheckedRecall` | `1.000` | `0` | `102/108` | `14/16` | `35.833` | `41.875` | `38.464` | `37.757` | `44.758` |
| `O_strong_AllValidatedPriors` | `1.000` | `0` | `0/108` | `0/16` | `37.639` | `47.500` | `83.569` | `41.817` | `53.269` |
| `O_HExA_CodeSkills` | `1.000` | `0` | `0/108` | `0/16` | `38.333` | `58.750` | `50.481` | `40.857` | `65.381` |

---

## 3. 第二部分：里程碑 `M4 (V17.2)` 实验主张精确边界、实测结果与主动披露

### 3.1 `M4` 主张的精确科学边界与诚实披露（供红队重点核查）

1. **支柱一（组合式算子演化器的语法边界）**：
   - `m4_compositional_discovery.py` 的初始候选池**严格只包含原子算子基元**（线性、单调有理式、一阶指数弛豫、幂律、面积-周长基元），**绝不包含任何预制复合答案表**；复合定律完全依赖 `CodeAgent` 执行 `diag_residual.py` 读取真实残差 `stdout` 后动态调用结构组合子（`apply_structural_combinator`，支持 2 轮迭代复合）生成新 `ExprNode` 树并编译成含纯 Python 阻尼 Levenberg-Marquardt `fit(observations)` 的真实 `model.py`。
   - **主动边界说明**：离线 CI 评测使用确定性残差驱动 `OpenCompositionalCodeProvider`（以便 `100%` 可复现核验 `R1–R6` 反事实因果性与计费），**不声称在本次离线脚本中发起了在线商业 LLM API 网络请求**；但实验台完整实现了 `CodeAgent` 标准工具协议（`OpenWorkspaceCodeLabAdapter`），可直接无缝挂载仓库内既有的 `ChatCompletionsCodeProvider` / `CodexCodeProvider`。
2. **支柱二（多尺度通道计价与 `ScaleBlind` 真实风险权衡披露）**：
   - `MultiScaleWorkspaceTaskLab` 显式区分廉价宏观粗粒化通道 `coarse_macro`（`4.0 / 点`）与昂贵微观精细通道 `micro_fine`（`12.0 / 点`），所有最终留出集评估（`evaluate_heldout_micro_truth`）一律对照底层真实微观物理 `_micro_fn` 并在隔离前缀沙箱内运行。
   - **主动披露真实权衡**：`HippoPPR_ScaleBlind_Walk` 因为跳过了 `16.0` 费用的主动尺度闭合探针（`1` 个宏观 + `1` 个微观边界比对点），在绝热闭合成立区 `T_Adiabatic_Coarse` 上测量费比 `Hegel_M4` 更低（`34.0` vs `42.0`）！但正因缺乏闭合验证，它在快淬火非绝热破裂区 `T_Closure_Breakdown` 误用宏观投影导致确认失败并被迫重采微观通道（复合成本飙升至 `133.556` vs `Hegel_M4` 的 `100.656`），从而在全集平均复合成本上被 `Hegel_M4` 反超（`56.565 < 59.432`）。这一正反双向表现证明 `ScaleBlind` 并未被人为加罚，而是真实呈现了“省探针费 vs 跨尺度破裂返工费”的物理权衡。
3. **支柱三（持续流效用反馈与 DEV 独立记账）**：
   - `V17.2` 的一次性 DEV 门控校准总复合成本 `1382.872` 已在 `summary.json["dev_gate_calibration"]` 独立单列；
   - 在 $K = 16 \to 64$ 持续流中，每阶段流入的 6 个同构副本经真实 `fidelity_compress_nodes`（核查 `ACC / REJ / UNK` 零冲突、零假新拒绝）在线合并，8 个诱饵干扰源经 3 次开发验证负收益反馈触发 `record_utility_feedback` 自动归档，2 个中间节点修订触发 `stale_endpoint_revision = True` 与 `revalidate_wormhole_after_revision` 重验。

### 3.2 `V17.2 (M4)` 核心结果汇总

1. **支柱一（反事实见证分叉 `R2` + 向后保持 `R3` + 删除/恢复因果性 `R1`）**：
   - 同源 `S1` 面对 `W_haldane_shunt`、`W_variational_cap`、`W_pure_shunt` 合成 **3 个互异 `VERIFIED_PURE_AST` 摘要**（`c2ced29ba2c345`, `176ae5aef4e4e4`, `fa3cf16ad597b5`），时间族 `W_underdamped_ringing` 与 `W_fast_mode_excess` 合成 **2 个互异 `VERIFIED_PURE_AST` 摘要**（`5ab363f56cb6c7`, `a6eb5d9ace81d3`），全部 `5/5` 达成 `confirm_rmse = 0.0`、`heldout_rmse = 0.0`、`strict_pass = True` 且 `S1` 低底物作用域 `100%` 保持。
   - 未见逆向重排目标迁移：`Hegel_Intact_Artifact`（`ran_diagnostic=False`, 评估 `1` 次, 算力 `22.0`, 复合 `41.10`）$\to$ **删除持久化文件 `Hegel_Deleted_Artifact`**（`ran_diagnostic=True`, 评估 `4` 次, 算力 `46.0`, 复合 `42.30`）$\to$ **恢复持久化文件 `Hegel_Restored_Artifact`**（`ran_diagnostic=False`, 评估 `1` 次, 算力 `22.0`, 复合 `41.10`），且严格优于同初始 Pilot 的 `Ordinary_AllHistory_EqualPilot`（评估 `6` 次, 算力 `52.0`, 复合 `42.60`）。
2. **支柱二（7 臂 × 72 未见多尺度目标 = `504` 工作区 + `8` 域外工作区）**：
   - `Hegel_M4_MultiScale_HippoWalk` 取得 **`72/72` `strict_pass`、`0` 假确认、`8/8` 域外 `scoped_unknown`、`72/72` 召回确认、`16/16` 两跳跨树游走（对比 `Alpha=0`、`Flat_BM25_WormholePool`、`VerticalOnly` 的 `0/16`）与 `16/16` 闭合破裂下钻**；全集平均测量费 **`55.778`**、算力费 **`15.747`**、复合成本 **`56.565`** 三项全场最低（相对 `MicroOnly` `88.871` 省 `-36.4%`，相对 `Continuing_AllHistory` `63.428` 省 `-10.8%`）。
3. **支柱三（$K = 16 \to 32 \to 48 \to 64$ 持续流）**：
   - 在 $K=64$ 时，`Hegel_M4_FullLifecycle` 将活跃节点从 `59` 压缩至 **`17`（`-71.2%`）**，`0` 条未重验过期虫洞，**`0.0%` 历史遗忘率**，复合成本保持 `57.797` 不膨胀（全面优于 `NoPruning` `57.849`、`Frozen_Stage1` `61.010`、`Flat_BM25_Continual` `61.161` 与 `Continuing_AllHistory` `64.950`）。

---

## 4. 一键独立复现与红队核验命令

在仓库 `Hegel Machine/` 根目录下可直接运行以下命令完成全量复核：

```bash
# 1. 运行 M0 -> M1 -> M2 -> M2.5 -> M3 -> M3.1 -> M4 全套 28 项回归测试（约 2 分钟，全绿）
PYTHONPATH=src PYTHONHASHSEED=0 python3 -m pytest \
  tests/test_v16_6_m0_contracts_and_wormholes.py \
  tests/test_v16_6_m1_cross_domain_wormhole_study.py \
  tests/test_v16_8_m2_open_workspace_wormholes.py \
  tests/test_v16_9_m2_audit_closure_and_open_revision.py \
  tests/test_v17_0_m3_hipporag_wormhole_scaling.py \
  tests/test_v17_1_m3_1_audit_closure_and_walk.py \
  tests/test_v17_2_m4_multiscale_and_open_discovery.py -v

# 2. 复跑上一轮红队审计原始未改动的闭环探针（在临时目录运行以保持审计目录只读清洁）
TMP_AUDIT=$(mktemp -d)
HEGEL_REPO="$(pwd)" PYTHONPATH=src PYTHONHASHSEED=0 python3 -c "
import os, pathlib, runpy
os.chdir('$TMP_AUDIT')
import hegel_machine
" && PYTHONPATH=src PYTHONHASHSEED=0 python3 -c "
import json, pathlib, sys
sys.path.insert(0, 'markdown/hegel_M3_Audit/closure')
import probe_closure as pc
pc.OUT = pathlib.Path('$TMP_AUDIT/closure_recheck.json')
pc.main()
print(pc.OUT.read_text())
"
rm -rf "$TMP_AUDIT"

# 3. 完整重算 V17.1 (M3.1) 十臂 2x2 因子分解与孤立 K={20,36,52} 扩容（约 6 分钟）
PYTHONPATH=src PYTHONHASHSEED=0 python3 scripts/run_v17_1_m3_1_factorized_walk_study.py

# 4. 完整重算 V17.2 (M4) 开放组合发现、多尺度重整化与 K=64 持续森林（约 1 分 45 秒）
PYTHONPATH=src PYTHONHASHSEED=0 python3 scripts/run_v17_2_m4_multiscale_open_discovery_study.py
```

---

## 5. 提请 GPT-6 重点审阅的 5 个问题

1. **`M3.1` 红队闭环彻底性**：确认阶段 `TemporaryDirectory` 纯前缀沙箱（闭合 `F11`）、对称 `_strip_outer_affine` + 遮蔽符号拦截 + `PREC_k`（`36/36` `VERIFIED_PURE_AST`）、四重收据强制校验及去标识符内容哈希平局决胜，是否已彻底消除此前指出的全部绕过路径？
2. **$2\times 2$ 检索因子分解与安全低负载试探体制（`T_WalkBridge`）**：在初始查询限于安全低负载窗口（$Q_{\text{safe}} \cap \text{sigs}(\text{Child}) = \emptyset$）的真实约束下，`Walk`（$\alpha=0.65$ vs $\alpha=0.0$）与 `Pool`（`WormholePool` vs `NoPool`）的正交边际归因是否严密且公平？
3. **`M4` 开放组合式 AST 演化与 `§10.2` 六判据符合度**：零预设复合公式表的多轮残差诊断组合子（`m4_compositional_discovery.py`）、5 组反事实见证分叉、原作用域向后保持以及持久化文件删除/恢复控制，是否扎实满足了 `M3_AUDIT.md §10.2` 提出的“历史产物有实际作用、反例驱动结构变化”要求？
4. **主动多尺度重整化（`ScaleClosureProbe`）与尺度索引虫洞**：同一系统树在 $S_{\text{micro}} / S_{\text{meso}} / S_{\text{macro}}$ 绑定不同跨领域同构邻居并通过主动闭合探针切换付费通道的设计，其数学契约与对 `ScaleBlind` / `MicroOnly` 的双向权衡刻画是否严谨？
5. **迈向真实外部科学基准与在线大模型闭环的下一步建议**：在当前离线确定性可复现实验底座全部闭环后，下一步挂载真实在线 `ChatCompletionsCodeProvider` / `CodexCodeProvider` 并接入外部真实多尺度物理/生物/工程实验数据时，最应优先防范的评测偏差与接口风险是什么？
