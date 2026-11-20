# 19 — Hegel V17.7 (`M4.5` 审计收尾闭环 + `M5.0` 反例驱动持续扬弃、双域实证虫洞、端到端池化收益与跨进程冷恢复流) 实测报告

- **对应审计输入**：`markdown/Hegel_M44_Audit/`（固定审计提交 `b965f022b214f47a1341559f20041da6a224d66d`，对照基线 `3461535364fcdb4e9ae9c87ccba502b7e6b56fe8`）
- **核心新增与升级模块**：
  - `src/hegel_machine/cognition/wormhole_forest.py`
  - `src/hegel_machine/cognition/hipporag_wormhole_index.py`
  - `src/hegel_machine/cognition/m4_multiscale_renormalization.py`
  - `src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py`
  - `src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py`
  - `src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py`
  - `src/hegel_machine/cognition/m4_4_instance_role_closure_and_online_writeback.py`
  - `src/hegel_machine/cognition/m5_0_closure_and_continual_sublation.py`（新增）
  - `scripts/run_v17_7_m5_0_closure_and_continual_study.py`（新增，支持 `--out-dir` 无损外部只读重放与独立子进程冷恢复校验）
  - `tests/test_v17_7_m5_0_closure_and_continual_sublation.py`（新增）
- **隔离静态证据目录**：`docs/evidence/v17_7_m5_0_closure_and_continual_sublation_20261120/`（`144` 项文件全部通过 `SHA256SUMS.txt` 校验，零 `__pycache__`/`.pyc` 混入；全部历史 `docs/evidence/v16_8_*..v17_6_*` 与 `markdown/*Audit*/` 100% 保持只读未改）

---

## 一、战略裁决：100% 完成 `Hegel_M44_Audit` 四大收尾修复包 + 一步建成 `M5.0` 反例驱动持续扬弃与跨进程冷恢复完整链

GPT-6 在 `GPT_diagnosis_M4_4.md` 与 `AUDIT_REPORT_ZH.md` 中确认：`V17.6 (M4.4)` 已彻底闭合上一轮两项主 P1 反例与全部 P2 计量/索引/归档问题，且正常主表路径零 P1 假确认，进程内 `W6/W7` 在线发现与后续复用真实成立；同时定位了五项具体 P2 接口或交付缺口，并建议“先完成四个有限修复包，再做 `M5` 的一条完整链”。

我们采取独立进取裁决：**不拆成两轮碎片补丁，而是在 `V17.7` 中一次性 100% 闭合全部四个修复包（五项 P2 控制），并直接建成由运行结果自动触发（零步骤号硬编码）的 `M5.0` 完整持续学习链（`28` 步流 + `3` 步跨进程冷恢复续跑）**，正面回答“经验反例改变后续合法行为”“双域各自独立实证与负例拒绝”“端到端付费任务池化收益”及“图拓扑相对扁平动态结构记忆的独立价值”四大核心科学问题。

### 1.1 四个收尾修复包（五项 P2 缺口）100% 工程闭合

1. **修复包 1 — 新建、重验证、消费与归档四端共享 `node / OR-branch / edge / certificate` 实例关系一致性（闭合 3.1）**：
   - 在 `wormhole_forest.py` 抽出无副作用纯校验器 `validate_claim_certificate_instance_identity`，由 `add_coarse_graining_edge`、`revalidate_vertical_edges_with_certificate`、`fine_scale_children_of` / `coarse_scale_parents_of` 以及 `verify_forest_snapshot_digests` 共同消费。
   - 只要节点动作、任一 OR 分支或闭合证书提供了 `confirmed_instance_digest`，新建粗粒化边在修改列表与 `mutation_epoch` 之前强制校验全部相等；将真实 `Stage 0` 证书接到 `Stage 2` 节点直接抛出 `certificate_instance_digest_mismatch`，全森林 `__dict__` 状态逐字节不变；即使在内存中强行篡改已有粗边的证书，`fine_scale_children_of` 亦拒绝将其视为活跃边（`T21` 保持 `micro_fine` 收费 `70.0`），且 `verify_forest_snapshot_digests` 准确检出 `edge_certificate_instance_mismatch_count == 1`（同时补齐全部 `node_history` 五字段 `semantic_digest` 校验）。
2. **修复包 2 — 析取 OR `selected_action` 全消费者统一与规范/半规范坐标锁（闭合 3.4 & 3.5）**：
   - 重新拟合召回链（`m4_3`、`m4_2`、`m4_1`）与通道预检全部改为**先调用 `authorize_abstraction_use` 获取 `selected_action`，再对命中的具体分支动作校验 `program_steps`、加载 `operator_spec`、核验盒约束并在 Tier-3 复核同一动作视图**。在普通两实例 Monod OR 控制中（分支 A 增益 `6` 产物缺席、分支 B 增益 `12` 完整且匹配），`R3` 与 `M4.3` 重新拟合召回均 `100%` 成功通过（`heldout_rmse = 0.0`）。
   - 新增公共规范化入口 `validate_canonical_role_map_contract` 并接入 `ForestScopeDispatcher.dispatch_predict`（`R3`）与盒映射枚举：凡输入坐标名已属于规范角色集合（无论全规范 `('u_drive','T_horizon')` 还是半规范 `('u_drive','x1')`），一律强制恒等锁定 `tf -> tf`。`R3` 显式传入矛盾映射 `{'u_drive':'T_horizon','T_horizon':'u_drive'}` 立即返回 `invalid_contradictory_role_map`（`prediction = None`），合法匿名映射 `{'x0':'u_drive','x1':'T_horizon'}` 正常通过；半规范短时目标在通道预检阶段即锁定 `u_drive`，零次浪费粗通道试探购测（测量费由 `88` 降至 `70.0`）。
3. **修复包 3 — 结果驱动的经验反例作用域扬弃与三层跨域知识表示（闭合 3.2 & 第四节 4）**：
   - 在 `m4_4` 兼容流中为 `S01` 补上 `rec["confirmed_ok"] and rec["strict_pass"]` 前置条件并正名为 `protocol_metadata_update`（失败回执下 `version_incremented_on_failure == False`）。
   - 在 `m5_0_closure_and_continual_sublation.py` 建成零步骤号硬编码的 **`OutcomeDrivenForestConsolidator`**：
     - **经验反例驱动的作用域扬弃**：当已支持节点（如 `stage1_W4_monod_shunt`）在后继任务新鲜确认中被反例击穿，整合器对比历史/当前低残差支持坐标与高残差反例坐标，直接从已付数据合成经验边界 `theta_emp = 0.5 * (max_supported + min_refuted)`（如 `b <= 1.85`），写入可执行 `preconditions` 并重绑纵向边，使后续域内任务（`b <= 1.45`）继续授权复用、越界任务（`b >= 1.95`）在门控层直接拦截（`scope_unauthorized`）。
     - **三层跨域知识分层（源确认实例 vs 结构假设桥 vs 第二域独立确认实例）**：新结构在源域 A 首次确认时，仅域 A 节点持有 `SUPPORTED_IN_SCOPE`、`local_solver_guidance` 与 `confirmed_instance_digest`；候选跨域槽位严格注册为 `STRUCTURAL_HYPOTHESIS_ONLY`（`bounded_trial`，无冻结实例摘要，`R3` 零拟合分发直接拒绝将其冒充已确认异域实例）。仅当第二域真实付费任务完成重拟合与独立新鲜确认后，才持久化第二域自有参数包、晋升为 `SUPPORTED_IN_SCOPE`（独立 `confirmed_instance_digest`），并以两域实测误差上限 `max(err_A, err_B)` 和双域 `source_episode_refs=[ep_A, ep_B]` 升级横向虫洞；若异域映射不成立（负例域 C），确认失败后立即记入失败证人账本并降级为 `REFUTED_IN_SCOPE`。
4. **修复包 4 — 在线学习终态完整归档、全字段 `R3` 保持记录与跨进程冷恢复续跑（闭合 3.3 & 第四节 5）**：
   - 官方 runner 完整持久化 `persisted_authored_workspaces/online_writebacks/`（`W6` 与 `W7` 在域 A、域 B 各自确认的 `4` 套三件套实例包）、第四张位置无关快照 `post_m5_online_stream_forest_snapshot.json`（含 `mutation_epoch`、`questions`、`raw_episode_archive`）及 `continual_session_checkpoint.json`（游标、已付观测历史、失败证人账本、累计预算账本与写回日志），同时保存完整 `10` 条历史 `R3` 回执及 `authorized_pass_rate`、`max_authorized_abs_error`、`unauthorized_cleanly_blocked`、`W4` 边界与 `W6_A/B`、`W7_A/B` 异实例零拟合验证。
   - **真实跨进程冷恢复**：在主构建 `TemporaryDirectory` 彻底销毁后，启动**全新独立 OS Python 子进程**从归档目录冷加载终态森林与 checkpoint，零重发现、零重复购测继续完成 `S29..S31` 三道后继任务（`cursor: 28 -> 31`）。

---

## 二、Part 1：`Hegel_M44_Audit` 五项 P2 反事实控制闭合总表

证据文件：`docs/evidence/v17_7_m5_0_closure_and_continual_sublation_20261120/part1_m4_4_audit_closure_verification.json`

| 控制包 | 反事实测试内容 | `V17.6 (M4.4)` 审计暴露行为 | `V17.7 (M4.5 + M5.0)` 实测结果 | 状态 |
|---|---|---|---|---|
| **P2-1 新建粗边与归档关系一致性** | 将真实 `Stage 0` $\Pi_s$ 证书通过 `add_coarse_graining_edge` 接到 `Stage 2` 节点；内存篡改边证书后测消费与归档校验 | 新建成功并开通 `coarse_macro`（`21 M`）；快照校验 `mismatch = 0` | 新建在状态变更前抛 `ValueError(certificate_instance_digest_mismatch)`，全森林 `__dict__` 不变；篡改边 `fine_scale_children_of == []`，`T21` 走 `micro_fine`（`70.0 M`），归档检出 `edge_certificate_instance_mismatch_count == 1` | **闭合** |
| **P2-2 OR `selected_action` 与角色锁** | 普通两实例 Monod OR（未选中分支 A 缺席、适用分支 B 完整）；`R3` 规范轴传矛盾 `u↔T` 映射；半规范轴 `('u_drive','x1')` 短时目标 | `M4.3` 在 gate 前因顶层 A 缺席跳过整个节点返回 `scoped_unknown`；`R3` 按矛盾映射走粗节点；半规范轴浪费 `18 M` 粗试探 | `R3` 与 `M4.3` 均成功加载分支 B（`confirmed_via_recall = True`, `heldout_rmse = 0.0`）；矛盾映射返回 `invalid_contradictory_role_map`（`prediction = None`），合法匿名映射通过；半规范轴零粗试探浪费（`70.0 M`） | **闭合** |
| **P2-3 失败守卫、反例扬弃与三层 Twin** | 强制 `S01 confirmed_ok=False`；高渗漏反例观测触发 `W4` 经验修订后测 `b=1.45` vs `b=2.10`；第二域未确认前测 Twin `R3` vs 确认后 vs 负例域 C | `S01` 失败仍升 `v2` 且仅改未消费协议字段；未确认 Twin 直接获 `local_solver_guidance` | 失败回执零版本递增；反例观测自动学出 `b <= 1.85`（`max_supported=1.45, min_refuted=2.25`），`b=1.45` 授权通过、`b=2.10` 返回 `scope_unauthorized`；未确认槽与负例域 C 均 `scope_unauthorized`，域 B 确认后晋升独立实例并通过零拟合校验 | **闭合** |

---

## 三、Part 2：统一 `selected_action` 与关系守卫下的 4 种子 11 臂主基准复核

证据文件：
- `part2_m4_5_continual_benchmark_summary.json`
- `part2_m4_5_execution_receipts.jsonl`（`1,144` 条唯一回执）
- `part2_m4_5_fork_noise_receipts.jsonl`（全 4 种子 `192` 条高斯噪声分岔回执）

在将 `authorize_abstraction_use` 前置至产物加载之前（消除对未授权候选节点的无效磁盘校验计费 `-0.375 K`/题）并启用全量关系一致性守卫后，11 臂主基准全部核心结论完全保持：
- **`Hegel_Unified_Continual_Forest`**：域内通过率 **`1.0000`**（`96/96`：`87` 次关系召回确认 + `9` 次全历史池回退），假确认率 **`0.0000`**，OOS `scoped_unknown` 率 **`1.0000`**（`8/8`），训练—确认最大重叠 **`0`**，平均测量费 **`31.3333`**，Stage-3 平均复合费 **`32.4464`**，96 题全周期总成本 **`3,767.1254`**。
- **静态基准对照与条件性双回本**：冻结静态森林上 `Compiled_Relational_CueTable_SameForest` 仍略低 `4.9524`（`3,762.1730` vs `3,767.1254`）；`Hegel_Unified_Continual_Forest` 在当前 6 族固定流与 `3:10` 通道单价下继续于**第 `41` 题**对 `Frozen_After_Stage0_OpenRediscovery`（`4,516.1720`）回本（终点净省 **`+749.0466`**），于**第 `46` 题**对 `Hegel_NoMaintenance_SameHistory`（`4,046.0290`）回本（终点净省 **`+278.9036`**，主要来自液压绝热 $\Pi_s$ 解锁合法 `coarse_macro` 通道）。

---

## 四、Part 3：`M5.0` 反例驱动持续扬弃、双域实证虫洞、端到端池化收益与跨进程冷恢复完整链

证据文件：
- `part3_m5_0_continual_sublation_summary.json`
- `part3_m5_0_stream_receipts.jsonl`（`28` 步 × `5` 臂 = `140` 条流内回执，首步显式计入 `initial_index_build` 初建计算费！）
- `part3_m5_0_cold_continuation_receipts.jsonl`（独立 OS 子进程冷恢复续跑 `S29..S31` 回执）
- `persisted_authored_workspaces/online_writebacks/`（`online_W6_tubule_domain_a`、`online_W6_reuptake_domain_b`、`online_W7_photobio_domain_a`、`online_W7_allosteric_domain_b` 四套持久化确认实例包）
- `relocatable_forest_snapshots/post_m5_online_stream_forest_snapshot.json`（`20` 节点、`18` 持产物节点，五字段与跨对象关系摘要零偏差）
- `continual_session_checkpoint.json`

### 4.1 28 步有状态结果驱动持续流（`S01..S28`）五臂完整账本对比

流结构（全程零 `wb_meta` 硬编码，完全由运行回执与确认/反例观测自动触发）：
- **Phase A（`S01..S04`，经验反例扬弃）**：`S01` `W4` 域内确认 $\to$ `S02` 高渗漏非线性突增反例触发 `W4` 合成可执行边界 `b <= 1.85`（`v1 -> v2`） $\to$ `S03` 低渗漏域内继续复用通过 $\to$ `S04` 超界高渗漏任务在门控层直接拦截 `W4`；
- **Phase B（`S05..S11`，`W6` 双域晋升 + 负例域 C 拒绝）**：`S05` 域 A 开放发现 `W6`（一维 Haldane）并建立两个 `STRUCTURAL_HYPOTHESIS_ONLY` 候选桥 $\to$ `S06` 域 B（不同物理参数 `11.8, 2.15, 0.145`）独立重拟合确认后晋升自有实例并升级双域虫洞 $\to$ `S07` 不兼容分段跳变域 C 确认失败后记入证人账本并降级为 `REFUTED_IN_SCOPE` $\to$ `S08..S11` 下游复用；
- **Phase C（`S12..S23`，`W7` 双域晋升 + 端到端多线索池化压力检验）**：`S12` 域 A 开放发现 `W7`（二维应力松弛超调） $\to$ `S13` 域 B（不同物理参数 `1.92, 0.56, 2.25`）独立确认晋升双域虫洞 $\to$ `S14..S23` 10 道下游复用（含 `8` 道双域分割多线索任务，与活跃二维多义节点 `stage1_W5_murray_shunt` 正面竞争）；
- **Phase D（`S24..S28`，历史多机制与双通道 $\Pi_s$ 保持检验）**：穿插 `W1`、`W2`、`W3` 及液压绝热/热学淬火双通道 $\Pi_s$ 任务。

| 评测臂（`25` 域内 + `3` 反例/域外 = `28` 步，含初建/刷新/写回全计费） | 域内通过 | 域外诚实拒答 | 假确认 | 下游复用召回 | `W7` 多线索评估候选 | 总评估候选 | 初建+写回 $K$ | 自动刷新 | 总测量费 $M$ | 总计算费 $K$ | **28 步总复合费 $C$** | 流后双域独立实例 `R3` + `W4` 边界拦截 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`Hegel_M5_LiveSublation_PPR_PoolOn`** | **`1.0000`** | **`1.0000`** | **`0.0000`** | **`16 / 16`** | **`8`** (`1`/题) | **`55`** | `19.55 + 23.50` | `7` | **`819.0`** | `730.53` | **`855.5273`** | **`True`（且拦截 `b=2.10` 与域 C）** |
| `Compiled_M5_LiveSublation_CueTable` | `1.0000` | `1.0000` | `0.0000` | `16 / 16` | `24` (`3`/题) | `69` | `19.56 + 23.50` | `7` | `819.0` | `735.86` | `855.7930` | `True`（且拦截 `b=2.10` 与域 C） |
| `Hegel_M5_LiveSublation_PPR_PoolOff` | `1.0000` | `1.0000` | `0.0000` | `16 / 16` | `24` (`3`/题) | `79` | `19.55 + 23.50` | `7` | `819.0` | `874.53` | `862.7265` | `True`（且拦截 `b=2.10` 与域 C） |
| `Dynamic_Flat_StructureMemory_LiveWriteBack` | `1.0000` | `1.0000` | `0.0000` | `16 / 16` | `24` (`3`/题) | `63` | `2.60 + 7.00` | `3` | `868.0` | `567.74` | `896.3870` | 仅源域单实例（无双域虫洞/粗边） |
| `Frozen_PostStage2_OpenRediscovery` | `1.0000` | `1.0000` | `0.0000` | `0 / 16` | `128` (`16`/题) | `238` | `19.55 + 0.00` | `0` | `883.0` | `1828.45` | `974.4229` | `False`（且对 `b=2.10` 仍放行旧 `W4`） |

### 4.2 五项关键科学结论与严谨归因

1. **虫洞等价池化首次产生端到端付费任务净收益（`PoolOn` 净省 `+7.1992 C` 击败 `PoolOff`）**：
   - 在 `8` 道二维双线索跨域任务上，活跃多义节点 `stage1_W5_murray_shunt` 同时携带两表面线索，`PoolOff` 因跨树双生节点概率质量分散而先试拟合 `W5` 的两种角色排列（每题评 `3` 候选，共 `24` 次），`PoolOn` 则通过同 AST 虫洞等价类池化将 `{W7_A, W7_B}` 直接推上 Rank-1（每题仅评 `1` 候选，共 `8` 次），全流少评 `24` 个候选（`55` vs `79`），直接转化为 **`+7.1992 C`** 的真实端到端复合成本优势。
2. **隔离“保存新结构”与“哲学森林关系拓扑”（`Hegel_M5` 净省 `+40.8597 C` 与 `-49.0 M` 击败 `Dynamic_Flat_StructureMemory`）**：
   - `Dynamic_Flat_StructureMemory_LiveWriteBack` 同样实时写回 `W6`、`W7` AST 结构与 `W4` 经验边界，完成了 `16/16` 结构复用（相对 `Frozen` 省 `+78.0359 C`）；但由于缺乏跨尺度 `coarse_graining_edges` 与跨树同构虫洞池化，它在多尺度 $\Pi_s$ 保持题上被迫走昂贵 `micro_fine` 通道（多付 **`+49.0 M`** 物理测量费），在双线索任务上同样落后于多义节点（`24` vs `8` 候选），且无法形成经确认的第二域实例。这严格证明图拓扑与尺度/同构边本身在“动态保存 AST”之外贡献了 **`+40.8597 C`** 的独立净收益。
3. **补齐初始索引构建入账后，`PPR_PoolOn` 在频繁动态更新与多线索流中稳健优于 `Compiled_CueTable`（`855.5273 < 855.7930`）**：
   - 在首步显式计入初建费（`19.55 K` vs `19.56 K`）与流内 `7` 次真实演化刷新后，`PoolOn` 凭借少评 `14` 个候选（`55` vs `69`）抵消在线查询计算差，取得 `+0.2657 C` 净优势（与 Part 2 冻结单线索流 `Compiled` 略省 `4.95 C` 形成清晰可解释的任务流分界）。
4. **反例只撤回越界部分、双域独立实例与负例隔离全部通过零拟合核验**：
   - 流后综合 `R3` 验证确认：历史 `10/10` 路由与数值容差（`authorized_pass_rate = 1.0`, `max_authorized_abs_error = 0.009287`, `any_fit_called = False`）完整保持；`W4 v2` 对 `b=1.40` 精确预测（`dispatched_ok`），对越界 `b=2.10` 严格拦截（`scope_unauthorized`，而未更新的 `Frozen` 仍错误放行）；`W6_A`（`14.5, 1.60, 0.18`）与 `W6_B`（`11.8, 2.15, 0.145`）、`W7_A` 与 `W7_B` 四套独立物理实例各自精确命中对应域真值且 `confirmed_instance_digest` 互不相同；负例域 C 保持 `scope_unauthorized`。
5. **跨进程冷恢复续跑（`S29..S31`）验证成立**：
   - 主构建临时目录销毁后，独立 OS Python 子进程从静态证据包恢复 `post_m5_online_stream_forest_snapshot.json` 与 `continual_session_checkpoint.json`（`cursor: 28 -> 31`），3 道续跑任务（含域 B `W6`、双线索 `W7` 单候选命中及域内 `W4`）全部 `100%` 经召回严格通过，触发源重发现次数为 **`0`**，训练—确认重叠为 **`0`**，累计预算账本平滑延续至 `941.6807`。

---

## 五、可复现指令清单

```bash
# 1. 运行 M0—M5.0 全量 44 项回归与审计闭环测试（12 个测试文件，零字节码污染）
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
  tests/test_v17_6_m4_4_instance_role_closure_and_online_writeback.py \
  tests/test_v17_7_m5_0_closure_and_continual_sublation.py

# 2. 重新生成完整 V17.7 (M4.5 + M5.0) 证据包（或通过 --out-dir /tmp/v17_7_replay 做无损外部复核）
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src PYTHONHASHSEED=0 \
  python3 scripts/run_v17_7_m5_0_closure_and_continual_study.py

# 3. 校验全部 144 项静态证据文件 SHA256
(cd docs/evidence/v17_7_m5_0_closure_and_continual_sublation_20261120 && sha256sum -c SHA256SUMS.txt)
```
