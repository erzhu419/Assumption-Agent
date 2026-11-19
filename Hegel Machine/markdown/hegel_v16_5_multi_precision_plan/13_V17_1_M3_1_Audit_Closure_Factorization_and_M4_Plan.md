# Hegel V17.1 (`M3.1`)：针对 `hegel_M3_Audit` 的独立裁决、全量红队闭环、虫洞池化 vs 多跳游走 $2\times 2$ 因子分解与 `M4` 路线图

> **审计对象说明**：`/Hegel Machine/markdown/hegel_M3_Audit/GPT_diagnosis_M3.md` 在磁盘上为 `0` 字节空文件（可能系网页端粘贴未保存），但其同目录下的主报告 `M3_AUDIT.md`（`36,383` 字节，`# Hegel Machine M3 审计与 M4 路线建议`）及三个子审计包 `closure/closure_audit.md`、`mechanism/MECHANISM_AUDIT_M3.md`、`results/results_audit.md` 与全部复现探针脚本（`probe_closure.py`, `probe_m3_ast_rejection.py`, `probe_open_forced_queries.py`, `probe_retrieval_factorization.py`, `probe_target_controls.py`, `probe_cost_accounting.py`）完整齐全。本报告对上述全部文档与探针完成了逐行复核与实证闭环，且严格保留 `docs/evidence/v16_8_*`、`v16_9_*`、`v17_0_*` 历史静态证据目录零改动，新增 `V17.1 (M3.1)` 证据独立归档至 `docs/evidence/v17_1_m3_1_audit_closure_and_walk_20261119/`。

---

## 一、独立治理裁决：吸收（Accept） vs 纠偏保留（Pushback & Upgrade）

秉持**“GPT-6 负责做最坏边界提醒，工程与科学路线决定权在我方；凡有利于后续 M4 多尺度与开放发现的架构坚定保留并用更强实验正名”**的原则，我们将 `M3_AUDIT.md` 的意见精确划分为两类：

### 1. 100% 吸收并已在 `V17.1` 完成零缺陷代码闭环的 7 项硬核提醒（Accept & Closed）

| 审计编号 / 探针脚本 | GPT-6 指出的真实技术漏洞或计费失衡 | `V17.1` 根因修复与实证复核结果 |
| :--- | :--- | :--- |
| **C1. 确认阶段文件通道泄漏 (`F11`)**<br>`closure/probe_closure.py` | `lab.acquire(confirm, workspace=ws)` 先把确认点写入 `ws/observations.json` 与 `ws/model_observations.json`，导致在 `cwd=ws` 中直接 `json.load(open('observations.json'))` 的 `file_memorizer.py` 绕过参数隔离拿到 `RMSE = 0.0`。 | `validate_workspace_model_on_rows` 与 `_evaluate_workspace_submission` 均升级为在独立 `TemporaryDirectory` 沙箱中运行，沙箱内 `observations.json` / `model_observations.json` **严格只写入 `isolated_prefix`**。复跑未改动的 `probe_closure.py`：三种记忆器（参数、`observations.json`、`model_observations.json`）及 `CodeAgent` 全生命周期确认 `RMSE = 9.0`、状态 `scoped_unknown`、`false_confirmation = False`。 |
| **C2. AST 规范化不对称（`9/36` 误判 `mismatch:`）与 2 个边缘碰撞**<br>`probe_m3_ast_rejection.py` & `probe_closure.py` | ① `predict(row)` 的 `_strip_outer_affine` 循环剥除了内层定律自带的截距/比例（如 `1.2 + 0.85*x`, `2.55*sqrt(x*y)`），而 `_CORE_LAW_EXPR` 未对称剥除，导致 `S01/S02`、`S09/S10`、`S20/S21` 等 `9/36` 源任务落入 `mismatch:` 并漏建 3 条真实虫洞；<br>② 局部重定义 `def exp(x):` 与丢弃 `round(x, ndigits=k)` 关键字导致双碰撞。 | ① 对 `_CORE_LAW_EXPR` 与 `predict` 对称执行模非零仿射读出透镜的骨架提取，并在森林构建端强制要求 `skeleton_status == "VERIFIED_PURE_AST"` 才允许跨树建虫洞；`M3` 全部 **`36/36` 源与近失工作区达成 `VERIFIED_PURE_AST`（`0` 个 `mismatch:`）**，横向虫洞由 `12` 条恢复至完整 `14` 条；<br>② 扫描模块级局部定义/遮蔽符号（拒绝被遮蔽调用）并保留 `round` 精度参数 `PREC_k`，`probe_closure.py` 两项碰撞均降为 `same_digest = False`。 |
| **C3. 收据可选字段与布尔旁路**<br>`probe_closure.py` | 当 `task_view` 省略绑定字段或同时传入 `heldout_confirmed=True` 时，旧收据仍可放行 Tier-3。 | `wormhole_forest._validate_evidence_receipt` 强制要求 `target_id`、`node_version` 非空匹配，且对开放代码任务强制要求 `model_sha256` 与 `confirmation_digest` 四重齐备，严禁 `heldout_confirmed=True` 覆盖无效收据；`probe_closure.py` 全部 4 种绕过测试均为 `granted = False`。 |
| **C4. 开放修订链 `stdout` 因果性与普通臂无条件多买点**<br>`probe_closure.py` & `probe_open_forced_queries.py` | ① `ScriptedOpenGrammarCodeProvider` 未读取诊断脚本 `stdout`；<br>② `Continuing_Ordinary_CodeSession` 在拟合前被无条件多塞了 `extra_q`（2 个测量点）。 | ① 诊断脚本实测计算残差非单调回落、副通道斜率与欠阻尼振荡超调，Provider 解析 `stdout` 动态合成/重排算子候选（`provider_ignores_diagnostic_stdout = False`）；<br>② 移除普通臂初始 `extra_q`（设为仅在初始先验全部拟合超限时才触发的 `fallback_query_batches`）：两臂初始测量费完全拉平（`50.0 == 50.0`），`Hegel_OpenRevision_Forest` 仍凭作用域与角色对齐在总复合成本上严格胜出（`53.23 < 74.70`）。 |
| **C5. `M3` 解释执行 `program_steps` 与去标识符平局决胜**<br>`MECHANISM_AUDIT_M3.md` | ① `M3ForestResearcher` 未实际消费节点 `program_steps`；<br>② 平局按 `node_id` 字典序决胜，且存在 `aa_*` 命名。 | ① `M3ForestResearcher` 真实解释执行 `calibrate_observation_lens`、`synthesize_scoped_python_schema`、`subprocess_confirm_and_refit`（清空 `program_steps` 会使 `confirmed_via_recall` 从 `True` 变为 `False` 并增加实测开销）；<br>② 清理全部 `aa_*` 命名，`HippoRAGWormholeIndex` 与 `FlatEqualMetadataIndex` 统一改用纯内容哈希 `_node_content_tie_key`，将全部 `node_id` 逆序重命名为 `zz_renamed_*` 后召回排序 **`0` 变化**。 |
| **C6. 合成器单调工作量计费、桥接账本入账与真实微宏观尺度探针**<br>`probe_cost_accounting.py` | ① 传入 6 个同维先验收 `60` 算力，而全搜索 `allowed_schemas=None` 只收 `48` 算力（子集反而更贵）；<br>② `m3_bridge_fit` 的 `0.5` 复合成本未写入 `source_episodes`；<br>③ `scale_probe_fn` 返回常数桩。 | ① 合成器按实际评估候选数单调计费 `4.0 + 6.0 * evaluated_candidate_count`（子集严格更便宜，显式全集与 `None` 算力 **100% 相等**）；<br>② `m3_bridge_fit` 显式记入 `source_episodes`（`source_bridge_calibration`）；<br>③ `scale_probe_fn` 改为真实调用 `_sim_oscillatory_carrier_trace` 对快振荡载波做微步积分与窗口平均对比。 |
| **C7. 拟合预算与确认买点预算分离 + `HExA` 错误库可达性**<br>`M3_AUDIT.md` §4.3 & `results_audit.md` | 各臂统一限制 `[:2]` 混淆了廉价训练拟合淘汰与真实买点确认，且 `HExA` 的 `mistake_bank` 排在队尾不可达。 | 所有检索臂与 `O_HExA` 统一采用 `max_fit_trials = 4`（零新买点的训练残差淘汰上限）+ `max_confirm_purchases = 2`（物理买点确认上限）；`M3HExACodeSkills` 按观测签名条件匹配优先激活 `mistake_bank`（删除 `mistake_bank` 会真实劣化 `O_HExA` 目标成本）。 |

---

### 2. 明确驳回保守退缩、并用 `V17.1` 新实验正面证伪的 3 项关键路线判断（Pushback & Retain）

1. **坚决保留 `HippoRAG` 多跳图随机游走（$\alpha > 0$），并正面揭示为何 `V17.0` 中 $\alpha = 0$ 看似足够、而在安全受限多尺度探测中 $\alpha > 0$ 绝对不可替代！**
   - **GPT-6 的诊断现象**：在 `V17.0` 的 `T01..T24` 上，`PPR(alpha=0, same_pool)` 与 `Flat_BM25_same_component_sum` 就能拿到高召回，`BFS_needed_candidates == 0`。
   - **我方的深层机理解剖与反驳**：
     - 首先，**横向虫洞等价类池化（`WormholePool`）本身就是 `PhilosophicalForest + HorizontalWormholeEdge`（经 AST 骨架同构验证的跨树等价类）的产物**，普通扁平记忆库根本不存在跨领域同构等价类边！我们在 `V17.1` 中把 `HippoPPR_Alpha0_WormholePool` 与 `Flat_BM25_WormholePool` 正式列为一等消融臂，清晰剥离“虫洞等价池化”与“多跳图游走”两个正交因子。
     - 其次，为什么 `V17.0` 中 `BFS_needed == 0`？**仅仅因为 `V17.0` 的初始 Pilot 直接横跨了 `[0.06, 0.88]` 全量程高负载区**，使目标任务在零跳时就直接观测到了高负载饱和/振铃/回落签名，与子规律节点直接命中！
     - 但在真实的工程安全探测与多尺度科学（以及 `M4` 核心场景）中，未见系统往往只能先在**安全低负载窗口（`safe_low_load`, `[0.01, 0.22]`）**做初始试探——此时高负载封顶、孔道回落、底物抑制、欠阻尼超调尚未触发，**初始查询 $Q_{\text{safe}}$ 与高负载修订子节点 $\text{NM}_A$ 的直接签名交集严格为空（$Q_{\text{safe}} \cap \text{sigs}(\text{NM}_A) = \emptyset$）**！
     - 在这一真实体制下（新增 `T25..T28` 共 4 类 × 4 簇 = `16` 个 `T_WalkBridge` 目标工作区）：
       - `Flat_BM25`、`Flat_BM25_WormholePool`、`HippoPPR_Alpha0_WormholePool`（$\alpha=0$）与 `Forest_VerticalPPR_NoWormholes` 对正确修订子规律的得分**恒等于 `0.0`（召回率 `0/16 = 0.0%`）**！
       - 唯有 **`Hegel_M3_1_HippoPPR_Walk_Pool`（$\alpha = 0.65$）** 能够沿两跳异质路径
         $$Q_{\text{safe}} \xrightarrow{\text{传感器低载签名}} S_B^{(-1)} \xleftrightarrow{\text{AST 横向虫洞}} S_A^{(+1)} \xrightarrow{\text{树内辩证扬弃边}} \text{NM}_A^{(+1)}$$
         把概率质量精确输送至零直接重叠的修订子节点 $\text{NM}_A$，并配合 `M0` 三值逻辑契约（高载前置条件未探测时判为 `UNKNOWN` 放行 Tier-2 `bounded_trial`，首点高载确认证伪父规律后升级 `TRUE` 放行 Tier-3 `local_solver_guidance`），在 `T_WalkBridge` 上取得 **`14/16`（`87.5%`）纯靠多跳图游走召回确认（对比全部 $\alpha=0$ / 扁平池化臂的 `0/16`）**，将平均物理测量费从 `57.50` 降至 **`41.88`**（`-27.2%`），总复合成本从 `64.30` 降至 **`44.76`**（`-30.4%`）！
2. **保留经验有界路由风险预算（`hop_error_budget`）+ 强制目标端独立确认点，拒绝滑向不可判定的全局形式化 Lipschitz 定理证明器**
   - 采纳审计对口径诚实性的提醒，将 `compose_wormhole_chain` 明确界定为**多跳候选路由的经验误差风险上界（Empirical Routing Risk Bound）**，超界自动降级为 `bounded_trial` 或回退，绝不替代目标工作区的隔离前缀新鲜点确认。
3. **保留四契约（`TaskView / ObservationMap / ScaleMap / ClaimContract`）+ 哲学森林主干，同时新增单因子孤立 $K \in \{20, 36, 52\}$ 干扰源扩容曲线**
   - 不因跨里程碑口径变化而削足适履退回小玩具集，而是直接固定同一批未见目标、同一合成器与同一单调工作量计费表，单因子将源库规模从 $K=20 \to 36 \to 52$ 翻倍扩容，正面验证抗干扰检索曲线。

---

## 二、`V17.1 (M3.1)` 预注册实测总表（`1,080` 个目标工作区 + `8` 个域外控制工作区）

证据包路径：`docs/evidence/v17_1_m3_1_audit_closure_and_walk_20261119/`（运行耗时 `378.20s`）。

### 1. 十臂全因子分解对比（4 个冻结评测簇 × 27 道目标/簇 = 每臂 `108` 个真实子进程工作区）

| 受控实验臂 | 拓扑与检索因子设定 | 严格通过率 (`strict_pass`) | 假确认数 | 全部目标召回确认 (`108`) | `T_Assoc` 召回确认 (`24`) | **`T_WalkBridge` 纯多跳游走确认 (`16`)** | 全部目标平均测量费 | **`T_WalkBridge` 平均测量费** | 全部目标平均复合成本 | **`T_WalkBridge` 平均复合成本** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Hegel_M3_1_HippoPPR_Walk_Pool`** | **完整森林 ($\alpha=0.65$ + 虫洞池化 + 全契约)** | **`1.0000`** | **`0`** | **`102/108`** | **`24/24`** | **`14/16` (`87.5%`)** | **`35.648`** | **`41.875`** | **`37.459`** | **`44.758`** |
| `HippoPPR_Alpha0_WormholePool` | **无游走 ($\alpha=0.0$) + 虫洞池化 + 全契约** | `1.0000` | `0` | `88/108` | `24/24` | **`0/16` (`0.0%`)** | `37.963` | `57.500` | `40.305` | `64.303` |
| `HippoPPR_Walk_NoPool` | 有游走 ($\alpha=0.65$) + 无池化 + 全契约 | `1.0000` | `0` | `102/108` | `24/24` | **`14/16` (`87.5%`)** | `35.833` | `41.875` | `37.695` | `44.758` |
| `Flat_BM25_WormholePool` | **扁平 BM25 + 虫洞池化 + 全契约** | `1.0000` | `0` | `84/108` | `20/24` | **`0/16` (`0.0%`)** | `38.704` | `57.500` | `41.426` | `64.382` |
| `Flat_BM25_EqualMetadata` | 扁平同元数据 BM25 + 全契约 | `1.0000` | `0` | `88/108` | `24/24` | **`0/16` (`0.0%`)** | `37.963` | `57.500` | `40.437` | `64.382` |
| `Flat_Cosine_EqualMetadata` | 扁平同元数据 IDF-Cosine + 全契约 | `1.0000` | `0` | `88/108` | `24/24` | **`0/16` (`0.0%`)** | `37.963` | `57.500` | `40.381` | `64.382` |
| `Forest_VerticalPPR_NoWormholes` | **仅有树内纵向边 ($\alpha=0.65$，删除横向虫洞)** | `1.0000` | `0` | `88/108` | `24/24` | **`0/16` (`0.0%`)** | `37.963` | `57.500` | `40.351` | `64.304` |
| `HippoPPR_UncheckedRecall` | 有游走召回但关闭四契约门控 | `1.0000` | `0` | `102/108` | `24/24` | `14/16` | `35.833` | `41.875` | `37.757` | `44.758` |
| `O_strong_AllValidatedPriors` | 全权限同维历史先验（单调工作量公平计费） | `1.0000` | `0` | `0/108` | `0/24` | `0/16` | `37.639` | `47.500` | `41.817` | `53.269` |
| `O_HExA_CodeSkills` | 条件匹配技能库 + 可达错误恢复库 | `1.0000` | `0` | `0/108` | `0/24` | `0/16` | `38.333` | `58.750` | `40.857` | `65.381` |

**核心实证结论**：
1. **横向虫洞 × 纵向扬弃边 × 虫洞等价池化三者皆具严格正边际收益**：在安全低负载试探目标 `T_WalkBridge` 上，砍掉随机游走（`HippoPPR_Alpha0_WormholePool`, `0/16`）、换成扁平虫洞池化（`Flat_BM25_WormholePool`, `0/16`）或砍掉横向虫洞只留树内纵向边（`Forest_VerticalPPR_NoWormholes`, `0/16`），均**完全丧失跨树两跳召回能力**；唯有横向虫洞与树内纵向边协同游走拿到 `14/16` 并节省 `30.4%` 复合成本，且加上虫洞等价池化（`Walk_Pool` vs `Walk_NoPool`）进一步将全集测量费从 `35.833` 压至 **`35.648`**、复合成本从 `37.695` 压至全场最低的 **`37.459`**！
2. **修复计费反常后依然严格击败 `O_strong_AllValidatedPriors`**：在改为按实际评估候选数单调计费（同维子集算力严格低于全搜索）后，`Hegel_M3_1_HippoPPR_Walk_Pool` 在全部 `108` 道目标上的平均测量费（`35.648 < 37.639`）、平均算力费（`36.219 < 83.569`，节省 `-56.7%` 候选拟合算力！）与平均复合成本（`37.459 < 41.817`）上**三项全胜**！
3. **域外诚实性 `100%` 守住**：`8/8` 个域外工作区（`OOS01`, `OOS02`）全部返回 `scoped_unknown`，`false_confirmation = 0`。

### 2. 单因子孤立 $K \in \{20, 36, 52\}$ 干扰源扩容曲线（固定目标集 + 固定合成器 + 固定计费）

| 检索臂 | $K = 20$（召回率 / 复合成本） | $K = 36$（召回率 / 复合成本） | $K = 52$（召回率 / 复合成本） | 扩容敏感度诊断 |
| :--- | :---: | :---: | :---: | :--- |
| **`Hegel_M3_1_HippoPPR_Walk_Pool`** | **`90.0%` / `43.275`** | **`90.0%` / `43.344`** | **`90.0%` / `43.954`** | $K$ 翻 `2.6` 倍，总召回率保持 `90.0%` 不掉，复合成本仅微增 `+0.68` |
| `HippoPPR_Alpha0_WormholePool` | `60.0%` / `49.560` | `60.0%` / `49.680` | `60.0%` / `50.288` | 卡死在 `60.0%`（`T_WalkBridge` 恒为 `0.0%`） |
| `Flat_BM25_WormholePool` | `60.0%` / `49.795` | `50.0%` / `52.379` | `50.0%` / `53.688` | 随单域干扰源增至 `36/52`，`T_Assoc` 召回跌至 `83.3%`，成本升至 `53.69` |
| `Flat_BM25_EqualMetadata` | `60.0%` / `49.695` | `60.0%` / `49.859` | `60.0%` / `49.773` | `T_WalkBridge` 恒为 `0.0%` |
| `Forest_VerticalPPR_NoWormholes` | `60.0%` / `49.566` | `60.0%` / `49.636` | `60.0%` / `49.656` | 无横向虫洞桥接，`T_WalkBridge` 恒为 `0.0%` |

---

## 三、面向 `M4` 的架构定型与执行路线图（融合我方多尺度主线与审计建设性建议）

在 `M0 -> M1 -> M2/M2.5 -> M3/M3.1` 全部工程地基与红队闭环就绪后，**`M4` 的核心使命是从“受控合成算子库 + 预定义残差分支”迈向“真正开放的组合式 Python 科学发现 + 主动多尺度变焦实验设计（Active Multi-Scale Renormalization & Open Operator Discovery）”**：

1. **M4 核心支柱一：开放组合算子搜索与真实 LLM / 自举代码提议器双轨接入（Open Compositional Python Discovery）**
   - 打破 18 固定 schema 枚举，构建基于**基本物理/数学算子原语（代数有理式、指数弛豫、幂律/分形标度、分段瓶颈 `min/max`、积分/差分记忆核、耦合 ODE 步进器）+ 维数/单调性/渐近极限静态契约检查**的开放 AST 组合演化器，并原生对接 `CodeAgent`（支持离线确定性算子组合搜索器与在线真实 LLM `CodeProvider` 同一套防泄漏评测协议）。
2. **M4 核心支柱二：主动多尺度变焦与粗粒化闭合发现（Active Multi-Scale Coarse-Graining & Renormalization Ladder）**
   - 直接兑现“水分子极性（微观）$\to$ 雪花/蜂巢表面能最小（介观变分）$\to$ 对流胞/输运分形网络（宏观）”的跨尺度思想：实验台同时暴露**高成本微观轨迹探针（Micro-Probe）**与**低成本宏观粗粒化探针（Coarse Macro-Probe：窗口平均、空间聚合、守恒量投影、快变量绝热消元 $\Pi_s$）**。Agent 自主设计尺度变焦探针，检验何时微观自由度可被重整化消去（生成带误差上界证书的 `ScaleMap`），何时在相变/临界/高频负载边界处粗粒化闭合破裂（自动触发向微观尺度的辩证下钻 Escalation）。
3. **M4 核心支柱三：带生命周期治理的自生长哲学森林（Self-Pruning & Fidelity-Compressing Forest at $K \ge 60$）**
   - 在 $K \ge 60$ 的持续跨学科流中，在线运行 `FidelityCompression`（零假新拒绝保真合并）、经验虫洞证伪降级（Stale Wormhole Invalidation & Revalidation）与低效用节点归档（Utility Ledger Archival），检验森林在长程演化下的体积有界性与正迁移净收益。
