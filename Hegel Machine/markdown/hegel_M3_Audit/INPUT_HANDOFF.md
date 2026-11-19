# Hegel Machine `V16.9 (M2.5)` + `V17.0 (M3)` 独立红队复核交接书（GPT-6 Audit Handoff）

> **远端已推送状态说明**：本复核包对应的全部源码、单元测试、评测脚本、两套新增只读证据包及工程报告均已推送至远端仓库提交 **`612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8`**（运行实验时的前序基线提交 `git_base_commit` 为 `9115f9adf9dad1f52f106a2ab14babe8b68243dd`）。历史证据目录 `docs/evidence/v16_8_m2_open_workspace_wormhole_20261119/` 保持 100% 原始未改动，本轮新增证据独立落盘于 `v16_9` 与 `v17_0` 两个新目录。

---

## 1. 复核范围、模块 SHA-256 与静态证据地图

本次复核包含前后衔接的两个完整阶段：
1. **`V16.9`（M2.5 红队闭环 + 开放算子修订链）**：逐项闭合 `markdown/Hegel_M2_Audit/GPT_diagnosis_m2.md` 指出的全部 `F1`–`F11` 问题，并打通接入原生 `CodeAgent` + `scipy.optimize.least_squares` 的两条非模板开放算子反例修订与跨领域逆序迁移链（`384` 个目标工作区）。
2. **`V17.0`（里程碑 `M3` 大规模森林扩展）**：将开放工作区哲学森林扩展至每簇 **`K = 36` 个跨学科源与反例近失工作区（`23` 类互不相同的可执行 Python 机理）**，接入概率守恒的 **HippoRAG 特异性加权个性化 PageRank（`HippoRAGWormholeIndex`）**，在 `4` 个冻结评测簇 × `24` 道未见目标 × `7` 个严格单因子受控臂 = **`672` 个未见目标工作区 + `8` 个域外诚实性工作区**上完成受控评测。

### 核心源码与测试文件 SHA-256（与 `612d8eea5` 完全一致）

| 相对仓库路径 (`Hegel Machine/`) | 角色说明 | `SHA-256` |
| :--- | :--- | :--- |
| `src/hegel_machine/cognition/wormhole_forest.py` | 四级权限门控、收据身份强绑定、`PhilosophicalForest` 与 `HorizontalWormholeEdge` | `417b443ae06be4f8277e780f83e299a14f0326fb8675fbf92aa1bec352c58d54` |
| `src/hegel_machine/cognition/open_workspace_labs.py` | 独立克隆噪声 RNG、前缀验证加载、子进程实测墙钟记录与 M2 套件 | `6282c73791252f3a3e8af6538701807ea96b1d0b7dd8a72376923f9ae844234e` |
| `src/hegel_machine/cognition/open_workspace_wormhole.py` | 纯数据流 AST 规范化互验、前缀冻结确认、`scoped_unknown`、$S_d$ 对称探针与全历史基线 | `d1d2e938995609c4f880593112e16cafda7ec67aae205f126b7ed4169ea23794` |
| `src/hegel_machine/cognition/open_python_revision.py` | 原生 `CodeAgent` + 有状态 `scipy.optimize.least_squares` 开放 Python 反例修订链 | `3836fee94f08939a73d9c3b538d36d54f53bc17b5fc5dbc5185217b1b62f2336` |
| `src/hegel_machine/cognition/hipporag_wormhole_index.py` | 零泄漏 `sig:*` 校验、`HippoRAGWormholeIndex` 概率守恒 PPR 与同元数据 `FlatEqualMetadataIndex` | `e41998ea7399e440980a9d557b514234dbc712936d866aa4ffe32bdf5766285d` |
| `src/hegel_machine/cognition/m3_scale_universe.py` | `K=36` 源/近失工作区（`23` 类 Python 机理）、`24` 未见目标、`2` 域外对照与 7 臂单因子求解器 | `7fc3b7fd46b34612915ace7d6e99e32fea9c1db7b65409e504ba901a542ae039` |
| `scripts/run_v16_9_m2_audited_open_study.py` | V16.9 聚焦反例复验 + 开放修订链 + 9 臂 M2.5 全量评测脚本 | `1271c3f9870e8efe4c0de7304c0885231b01c904a76c440f87a9b63cba31ae6c` |
| `scripts/run_v17_0_m3_hipporag_scale_study.py` | V17.0 M3 7 臂 `680` 工作区全量评测、$\lambda$ 网格与 `N=1..24` 摊销曲线脚本 | `6605972c1ad9bfb314d40962294e5636c8766104f3fe92d9a289d3851ffff238` |
| `tests/test_v16_9_m2_audit_closure_and_open_revision.py` | `F1`–`F11` 红队反例回归与开放修订链测试 | 已纳入 `pytest` 全绿套件 |
| `tests/test_v17_0_m3_hipporag_wormhole_scaling.py` | 零泄漏校验、同元数据受控图分离、召回 $\neq$ 授权、坐标逆序与域外退出测试 | `3de08ef3f19538d751be69304f81806caf1b9379bc6f8847218d80c052ecf880` |

### 证据包与报告路径
- **V16.9 证据目录**：`docs/evidence/v16_9_m2_audited_open_workspace_20261119/`（含 `focused_audit_recheck.json`、`open_author_revision_results.json`、`paired_results.json` 等 14 个文件）
- **V17.0 (M3) 证据目录**：`docs/evidence/v17_0_m3_hipporag_wormhole_scale_20261119/`（含 `paired_results.json`、`index_and_retrieval_diagnostics.json`、`out_of_scope_honesty.json`、`wormhole_forest_graph.json`、`cost_ledger.jsonl` 等 13 个文件）
- **详细工程报告**：
  - `markdown/hegel_v16_5_multi_precision_plan/11_V16_9_M2_Audit_Closure_and_Open_Author_Report.md`
  - `markdown/hegel_v16_5_multi_precision_plan/12_V17_0_M3_HippoRAG_Scale_Wormhole_Report.md`

---

## 2. 第一部分：`Hegel_M2_Audit` 红队问题逐项闭环对照表（`V16.9`）

| 红队编号与问题摘要 | 代码修复机制与落点 | 实测复核证据 (`focused_audit_recheck.json`) |
| :--- | :--- | :--- |
| **`F1 / F3`**：M2 合成器局限在 10 个预置模板、`fit` 为空操作、无开放算子修订 | 新增 `open_python_revision.py`：接入 `CodeAgent`（`query/run/write/evaluate/finish`）+ `scipy.optimize.least_squares` 有状态参数重拟合，跑通 **Chain O1（Haldane 底物抑制 + 旁路分流）** 与 **Chain O2（欠阻尼阶跃振荡 `1 - exp(-t/tau)*cos(w*t)`）** 两条非 10 模板算子反例修订与跨领域逆序复用链 | 两条链旧模型在反例上 `RMSE > 0.6`，新合成算子向后兼容原域 `RMSE < 0.005`，未见逆序反号目标 `T_O1 / T_O2` `4/4` 簇全通，复合成本严格低于移除新节点及持续普通会话 |
| **`F2`**：AST 规范化盲信 `_CORE_LAW_EXPR` 旁注、未内联局部绑定、`x**2` 与 `x**-2` 碰撞 | 重构 `canonicalize_python_model_ast`：从 `predict(row)` 真实返回表达式递归内联局部纯赋值、剥离外层仿射读出壳、保留有符号幂指数（`POS_INT_EXP_2` vs `NEG_INT_EXP_2`），若与 `_CORE_LAW_EXPR` 不一致直接标记 `ANNOTATION_CODE_MISMATCH` | 3 组红队碰撞探针 `digests_equal` 全为 `False`；化学 `+1` 与电磁 `-1` 真实代码仍收敛至同一 `VERIFIED_PURE_AST` 摘要 |
| **`F4`**：`O_strong` 同维只留 1 个 `>=2` 域槽位；`HExA` 频率平票受 `PYTHONHASHSEED` 影响且未消费 `mistake_bank` | 新增 `O_strong_AllValidatedPriors` 保留同维**全部**验证通过历史候选（含单域与近失例，按候选数计入逻辑计算）；`HExA` 改为确定性键 `(-count, tag)` 并主动拼接消费 `mistake_bank` | 在 M2.5 小库（`K=7`）如实呈现 `O_strong` 测量费打平（`ΔMeas = 0`）、目标复合差 `-8.95`、单批 10 目标端到端 `+6.05`；同时保留旧压缩版作独立对照 |
| **`F5`**：确认点失败仍退回 `completed`，在 `5 + sin(u)` 上产生假确认 | 确认失败一律返回 `status="scoped_unknown"`、`model_path=None`（候选另存 `unconfirmed_candidate.py` 供审计），绝不伪造通过 | `5 + sin(u)` 与 M3 的 `OOS01 / OOS02` 共 `100%` 返回 `scoped_unknown`，`false_confirmation = 0` |
| **`F6`**：2D/3D 非对称 pilot 依赖特征声明顺序 | 全面改为 $S_2$（6 点）与 $S_3$（6 点）完全置换对称 pilot 设计，并由响应几何对称性自动识别角色顺序 | `T_M2A1` 与 M3 `T01` 在原特征顺序与 `reversed(feature_names)` 下通过率、测量费与计算费 **100% 恒等** |
| **`F7`**：证据收据接受 `claim_id="*"`、缺失/非有限 `observed_error` 及跨目标跨版本旧收据 | `wormhole_forest._validate_evidence_receipt` 强制核验非通配 `claim_id`、有限 `0 <= observed_error <= tol`、`target_id`、`node_version`、`model_sha256` 与 `confirmation_digest` | 5 类伪造/过期/`NaN`/`*` 收据在测试与探针中 100% 返回 `granted = False` |
| **`F8`**：节点 `program_steps` 仅为装饰性元数据 | 求解器改为真实逐条解释执行 `node["program_steps"]`（`calibrate_observation_lens`、`synthesize_scoped_python_schema`、`subprocess_confirm_and_refit`） | 删除校准或确认步骤后真实改变执行分支、测量开销与授权结果 |
| **`F9`**：`deepcopy(lab)` 共享同一 `_noise_rng` 闭包 | `OpenWorkspaceTaskLab.__deepcopy__` 独立克隆底层 `Random` 状态并重新绑定闭包 | 两个 `deepcopy` 分支首次采样噪声观测值完全一致且互不污染 |
| **`F10 / F11`**：混合成本缺乏实测墙钟与 $\lambda$ 灵敏度；`fit(observations)` 在 `acquire(confirm)` 后可偷看确认点标签 | 实测记录子进程 `measured_python_wall_seconds` 并输出 $\lambda \in [0, 0.40]$ 网格；在 `acquire(confirm)` 前冻结 `training_prefix` 传入 `validate_workspace_model_on_rows` | 有状态 `memorize.py` 探针确认点 `RMSE = 9.0`（与仅传前缀完全一致，彻底阻断偷看） |

---

## 3. 第二部分：里程碑 `M3 (V17.0)` 架构边界、实测结果与主动披露事项

### 3.1 实验主张的精确边界（请审阅时重点区分两项正交声明）
1. **声明边界 A（合成器受控不变量）**：在 `V17.0` 的 `672` 个未见目标工作区矩阵中，为严格隔离 **“大规模图检索拓扑 + 契约门控”** 的单因子因果效应并保证 7 臂跨簇可复现对账，全部 7 个受控臂共享同一个 23 机理子进程 Python 合成器（`_M3_SYNTHESIZER_SCRIPT`）；开放自由表达式合成能力已在 `V16.9` 的 `open_python_revision.py` 中独立验证，**`V17.0` 不声称 `_M3_SYNTHESIZER_SCRIPT` 本身是无界程序空间搜索**，其核心主张是在 **完全相同的合成器与完全相同的冻结森林元数据** 下，检验 `HippoRAGWormholeIndex` 相对 `Flat_BM25`、`Flat_Cosine`、`VerticalPPR_NoWormholes`、`UncheckedRecall`、`O_strong` 与 `O_HExA` 的规模化检索与成本优势。
2. **声明边界 B（零阉割同元数据公平性）**：`Flat_BM25_EqualMetadata` 与 `Flat_Overlap_EqualMetadata` 通过 `hegel_base.clone_with_mode(...)` 直接读取 **同一个冻结 `PhilosophicalForest` 的全部节点契约、`ALIGN` 排除条件、两跳累积误差界、`ScaleMap` 证书与 `Stage-1` 门控**，唯一差异仅为 Tier-1 排序算子（概率守恒马尔可夫游走 vs Okapi BM25 / IDF 余弦）。

### 3.2 `V17.0 (M3)` 冻结 4 评测簇（每簇 `36` 源/近失 + `24` 未见目标）核心结果汇总

| 实验臂 | 严格通过 | 假确认 | 授权 `Recall@1` | 授权 `Recall@2` | 错误试算/簇 | 源期+建索引 | 目标测量费 (`λ=0`) | 目标逻辑计算 | 目标复合 (`λ=0.05`) | 端到端总复合 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Hegel_M3_HippoPPR`** | **`96/96`** | **`0`** | `0.6957` | **`1.0000`** | **`5.0`** | `1980.96` | **`830.00`** | **`817.49`** | **`870.87`** | **`2851.83`** |
| `Flat_BM25_EqualMetadata` | `96/96` | `0` | **`0.7826`** | `0.9130` | `6.0` | `1978.76` | `840.00` | `867.24` | `883.36` | `2862.12` |
| `Flat_Overlap_EqualMetadata` | `96/96` | `0` | **`0.7826`** | `0.9130` | `6.0` | `1978.76` | `840.00` | `867.24` | `883.36` | `2862.12` |
| `Forest_VerticalPPR_NoWormholes` | `96/96` | `0` | `0.5652` | `0.8696` | `13.0` | `1980.88` | `890.00` | `1015.48` | `940.77` | `2921.65` |
| `HippoPPR_UncheckedRecall` | `96/96` | `0` | `0.5833` | `0.8750` | `9.0` | `1980.96` | `855.00` | `912.79` | `900.64` | `2881.60` |
| `O_strong_AllValidatedPriors` | `96/96` | `0` | — | — | — | `1955.00` | `856.25` | `2904.38` | `1001.47` | `2956.47` |
| `O_HExA_CodeSkills` | `96/96` | `0` | — | — | — | `1966.20` | `1095.00` | `1750.50` | `1182.53` | `3148.73` |

### 3.3 主动披露的真实权衡与成本边界（供红队重点复核）
1. **`Recall@1` vs `Recall@2` 的真实图扩散权衡**：
   - `Flat_BM25_EqualMetadata` 在一跳词面直接匹配上具有更高 `Recall@1`（`0.7826` vs `0.6957`），因此在 `T_Assoc` 子类上比 `Hegel_M3_HippoPPR` 低 `6.3270` 复合成本/簇。
   - 但在两跳非传递跨协议桥接目标（如 `T13`）上，`Flat_BM25` 与 `Flat_Cosine` 被单域反号局部节点（`S24, S29..S32`）挤出 `max_recalled_trials=2` 窗口（`Recall@2 = 0.9130`），被迫退回高开销全量探索；`Hegel_M3_HippoPPR` 则通过跨树虫洞随机游走实现 **授权 `Recall@2 = 1.0000`（`92/92`）**，在 `T_MultiHopGuard` 净省 `-17.6705`/簇，从而在纯物理测量费（`-10.00`/簇，`λ=0`）与计入建图开销的端到端总成本（`-10.29`/簇，`4/4` 负，平衡点 `N* = 13`）上均实现净胜。
2. **从 `K=7` 到 `K=36` 的规模反转效应**：
   - 在 V16.9（`K=7`）中，`O_strong_AllValidatedPriors` 测量费与 Hegel 打平且 10 道目标内端到端占优（`+6.05`）；扩展至 `K=36`（`23` 类机理）后，同维未条件化全量候选池不仅使逻辑计算暴涨（`ΔCompute = -2086.89`），更因多参数历史候选偶发拟合 6 点 pilot 而多耗物理确认点（`ΔMeas = -26.25`），使 `Hegel_M3_HippoPPR` 在第 **`N* = 7`** 道目标即实现端到端摊销反超，24 道目标累计端到端净省 **`-104.64`**（`4/4` 簇全负）。
3. **开发集校准成本独立披露**：`2` 个冻结开发集（`m3_dev_01..02`）一次性门控校准总投入 `6931.84` 复合成本已在 `paired_results.json` 顶层 `dev_calibration_summary` 单独列出；若将一次性开发集成本硬摊入单个评测簇，则需跨多簇持续服役（约 $\ge 67$ 个同规模目标簇或约 $1600$ 道目标）摊平。

---

## 4. 一键独立复现与红队核验命令

在仓库 `Hegel Machine/` 根目录下可直接运行以下标准命令复核全部结论：

```bash
# 1. 运行 M0 -> M1 -> M2 -> V16.9 -> V17.0 (M3) 全套 20 项里程碑回归测试（零警告全绿）
PYTHONPATH=src PYTHONHASHSEED=0 python3 -m pytest \
  tests/test_v16* tests/test_v17_0_m3_hipporag_wormhole_scaling.py -q

# 2. 复跑原红队全部聚焦反例探针（AST 碰撞、前缀偷看、运行时间、收据重算）
PYTHONPATH=src PYTHONHASHSEED=0 python3 markdown/Hegel_M2_Audit/probe_ast.py
PYTHONPATH=src PYTHONHASHSEED=0 python3 markdown/Hegel_M2_Audit/probe_confirmation_training_leak.py

# 3. 完整重算 V16.9（M2.5 审计闭环 + 原生 CodeAgent 开放修订链）
PYTHONPATH=src PYTHONHASHSEED=0 python3 scripts/run_v16_9_m2_audited_open_study.py

# 4. 完整重算 V17.0（M3 K=36 HippoRAG + 横向虫洞 7 臂 680 工作区大规模实验）
PYTHONPATH=src PYTHONHASHSEED=0 python3 scripts/run_v17_0_m3_hipporag_scale_study.py
```

---

## 5. 提请 GPT-6 重点审阅的 5 个问题

1. **红队闭环完整性（`V16.9`）**：`F1`–`F11`（尤其是 `predict(row)` 纯数据流内联互验、`training_prefix` 冻结隔离、`scoped_unknown` 诚实退出与收据五元组强绑定）是否仍存在任何残余绕过路径或边界漏洞？
2. **M3 单因子检索公平性（`V17.0`）**：`validate_observable_query_signatures` 与 `extract_m3_observable_signatures` 是否存在任何隐式领域/真值标签泄漏？`Flat_BM25_EqualMetadata` 与 `Flat_Overlap_EqualMetadata` 共享完整冻结森林契约的设计是否构成严格公平的单因子对照？
3. **`HippoRAGWormholeIndex` 数学机制与 `Recall@1` vs `Recall@2` 权衡**：对于 `α = 0.80` 下牺牲少量一跳 `Recall@1`（`0.6957` vs `0.7826`）换取多跳 `Recall@2 = 1.0000` 的扩散权衡，在不引入领域特判的前提下，是否存在更优的自适应阻尼（如 query-dependent $\alpha$ 或 HippoRAG 2 的 passage-node 双层重置）以兼顾一跳锐度与多跳桥接？
4. **“召回 $\neq$ 授权”与多尺度安全边界**：Tier-2 契约预筛（`ALIGN` 近失排除 + `compose_wormhole_chain` 非传递两跳误差界 + `ScaleMap` 视图保持验证）在大规模异质库下是否还有尚未覆盖的负迁移失效模式？
5. **下一阶段 `M4` 路线建议**：目前 `M0 -> M1 -> M2.5 -> M3` 已分别独立跑通了 **“开放 `CodeAgent` 算子合成修订（V16.9）”** 与 **“大规模 `K=36` 异质森林 HippoRAG 虫洞检索（V17.0）”**。下一阶段 `M4` 计划将二者合流（在无预置机理列表的大规模真实科学/工程表格与动力学数据集上，由开放 `CodeAgent` / LLM 自主发现算子并持续生长虫洞森林），最关键的实验陷阱与成功判据应如何设定？
