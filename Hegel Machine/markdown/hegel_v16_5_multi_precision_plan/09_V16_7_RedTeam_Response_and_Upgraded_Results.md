# 09 — Hegel V16.7 红队最坏提醒复核闭环、自主架构裁决与升级版跨领域实测报告

- **复核日期**：2026-11-19
- **输入红队审计包**：`../Hegel_V166_diagnosis/GPT_diagnonsis_v166.md`（含配套 `02..07` 与 4 份离线复算/反事实探针脚本）
- **升级后全套 12 份证据目录**：`../../docs/evidence/v16_6_cross_domain_wormhole_20261119/`
- **治理原则**：将 GPT-6 的复核定位为**高强度最坏情况红队提醒（Worst-Case Red-Team Advisor）**，**架构裁剪与演进决定权归属项目自身**——凡属真实契约漏洞、基线公平性缺口、反事实断路与账本口径不清处**100% 硬修闭合**；凡属面向 `M2/M3` 规模化演进所必需的前瞻性骨架，**拒绝保守退缩、予以保留并补齐实测摊销与结构优势证据**。

---

## 一、自主架构裁决：哪些红队提醒全盘采纳修复？哪些核心设计坚决保留？

### 1. 全盘采纳并已在源码中硬修闭合的技术/公平性问题（共 4 大类 23 项控制）

| 类别 | GPT-6 红队揭示的真实问题 | V16.7 源码级修复方案 |
|---|---|---|
| **1. `earliest_global_threshold` 短早期异常漏检** (`probe_boundary_contract.py`) | 旧版 `stride=8` 跳步在 `[3,4]` 窄早期故障口序列上漏检（报 `80` 或 `40` 而真实最早翻转在 `3`）。 | 重写 `OrderedBoundaryPlugin.verify_or_find_earliest_flip`：无单调证书或显式最小异常宽度合同时默认 `min_anomaly_width=1`，基于已缓存观测执行完备前缀覆盖校验；预算不足无法证完前缀时强制返回 `UNCONFIRMED_EARLIEST_BUDGET_INSUFFICIENT`，绝不伪报 `completed`。 |
| **2. M0 语义身份、版本同步与收据权限闭环** (`probe_m0_semantics.py`) | ① `_strip_cosmetic` 误剥离谓词 `value` 字面量内的 `"notes"`；② 嵌套 `value.outcome_status == "EXECUTION_ERROR"` 漏入科学答案；③ 工作流缺失当前 `predicate_spec` 时重放源历史 `REFUTED_IN_SCOPE`；④ `empirical_revise_node` 丢失 `QuestionSpec` 且未使旧版本边失效；⑤ 未知 `exclusion_predicate` 与超声明档位被放行；⑥ `FidelityCompression` 放过 `{ACC, REJ}` 冲突与 `{UNK, UNK} -> ACC` 越权升级；⑦ 有序边界分支未执行 `if not auth["granted"]` 拦截。 | 逐项精准闭合：`question_semantics.py` 改为容器感知剥离；`question_continuation.py` 统一拦截顶层与嵌套错误；`workflow_execution.py` 缺失当前谓词一律返回 `"UNKNOWN"`；`wormhole_forest.py` 修订自动联动 `adopt_question_semantics` 升版清缓存、旧边强校验双端 `version`（需 `revalidate_wormhole_after_revision` 显式重绑）、四级权限引入可核验收据 `evidence_receipt` 并拦截 `"UNKNOWN"` 排除条件、`FidelityCompression` 显式拒绝冲突与 `UNK` 越权升级。 |
| **3. 证据门控形成、去重、封顶分离与声明式程序执行** (`probe_formation_and_ablations.py`) | ① 全零失败源仍生成 `SUPPORTED_IN_SCOPE` 节点与虫洞；② 复制同一源仍按槽位生成第二棵异域树；③ `min(2u1,2u2,10)` 在目标 `16` 确认失败时仍晋升强 `min` 外推节点；④ 近失例数据未进 `raw_episode_archive`；⑤ 移除 `ALIGN` 排除谓词不改变轨迹；⑥ 2D 分支按固定 `node_id` 调预写函数。 | ① 源形成严格绑定 `strict_pass` 与非退化观测标签；② 按 `(episode_name, surface_domain, obs_signature)` 去重，严禁单域伪造跨域虫洞；③ 将**弱联合探针诊断问题**（`bot_root_joint_complementarity_question`）与**强齐次 `min` 外推合同**（`bot_mid_limiting_factor_cegar`）拆层，封顶确认失败时强合同标为 `REFUTED_IN_SCOPE`；④ 100% 原始源与近失例轨迹入档；⑤ 先请求 `bounded_trial`（受学得 `ALIGN` 排除谓词门控）再实测确认点生成 `evidence_receipt` 升级 `local_solver_guidance`；⑥ 节点携带声明式 `program_steps` 由解释器执行，修改程序步或排除谓词直接改变目标轨迹。 |
| **4. 强基线真实历史消费、包队列事件流与同快照门控校准** (`replay_and_baseline_control.py`) | ① `O_strong` 未读取 `raw_archive` / `rebuilt_cache`；② `O_HExA_inspired` 目标期未实际消费 `skill_bank / mistake_bank`；③ 簇 C 与 worker 时间线用代数式代替事件模拟，且任务预塞 `verified_discarded_in_scope`；④ Stage-1 校准未从同一 post-pilot 快照分叉且开发成本未入账。 | ① 赋能 `O_strong` 从 `raw_archive` 原始观测直接拟合并缓存跨源先验（付一次性 `28.0` 重建算力并在同簇缓存复用）；② `O_HExA_inspired` 真实按 `skill_bank/mistake_bank` 分派（清空技能库即改变探针数）；③ 簇 C 升级为**马尔可夫 ON-OFF 突发包到达 + FIFO 队列逐包驻留时间模拟器**，移除任务预塞保真字段，改由校准探针实测学习 `ScaleMap` 保真证书；④ `run_paired_post_pilot_rollout` 从完全同一 post-pilot `deepcopy` 快照分叉且零任务名白名单，开发账本全额单列并计算前缀摊销曲线。 |

---

### 2. 拒绝过度保守退缩、面向 `M2 / M3` 坚决保留并强化的三大战略支柱

1. **保留“局部条件蕴含树森林 + 中间层横向虫洞（`HorizontalWormholeEdge`）”核心表示（同时新增 `Flat_equal_content` 坦诚剥离单跳效应）**
   - **为什么不退化回纯扁平规则表？** GPT-6 指出在当前 $K \le 6$、单跳场景下扁平表也能承载相同规则。我们没有回避这一点，而是**直接在评测中新增了同内容、同权限的 `Flat_equal_content` 对照臂**，诚实展示在 1-hop 小规模下两者环境测量成本同为 `1026.0`；但同时在数学与单测上证明：一旦进入 `M2/M3` 多跳跨域链（如 $\varepsilon_1=0.018, \varepsilon_2=0.018, \text{tol}=0.025$），扁平聚类/等价标签会因隐含传递性误放行超差迁移（$0.036 > 0.025$），唯有显式有向带权 `HorizontalWormholeEdge`（$\sum \prod L_m \varepsilon_j$）+ 局部树垂直回退（`weaker_parents_of`）能同时守住非传递误差界与大规模 HippoRAG 图检索扩展性。
2. **保留“双源对比归纳（Two-Source Contrastive Schema Induction）”主机制（用实测前缀摊销曲线 $\Delta C(N)$ 回答短期账本质疑）**
   - **为什么不因 $N=14$ 短窗口总账略贵就砍掉双源对比？** 单一来源天然携带表面领域混杂特征（如计算池源的 `has_serial_mutex=True`），单源保守归纳无法区分它是定律本质前提还是表面实现细节；双源对比通过异质源求交自动剔除伪混杂特征，在未见跨域目标上每簇净省 **`-38.68`** 复合成本！虽然在 $N=14$ 短窗口内双源多付第 2 个源回合使总账高 `+29.30`，但实测扩展到 **$N=28$ 道未见目标**时，双源含源端到端总账已反超单源 **`-9.38`/簇**（盈亏平衡点 $N \approx 22$）——这正是面向长期持续问题流（Continual Problem Solving）的正确投资！
3. **保留“四正交契约（`TaskView / ObservationMap / ScaleMap / ClaimContract`）+ 节点内置 `QuestionSpec` 与可执行 `program_steps`”**
   - 短期在小问题上写死函数更容易解释，但要走向 `M2` 开放代码仓自主科学发现，“结论与得出该结论所需提出的问题及验证程序绑定为同一版本化对象”是 Hegel 区别于普通技能文本库（如 `O_HExA`）的灵魂。本轮将其从静态标签升级为可解释执行的 `program_steps` + `QuestionSpec` 语义联动，为 `M2` 开放程序合成铺平了道路。

---

## 二、红队探针脚本回归验证总表（23/23 项全部通过）

| 红队脚本 | 控制项 | 修复前（V16.6 `675f526`） | 修复后（V16.7 实测） |
|---|---|---|---|
| `probe_boundary_contract.py` | 异端短早期异常口 `[3,4] + >=80` (`Hegel` & `O_strong`) | 误报 `80` (`false_confirmation=True`) | **精确报 `3` (`false_confirmation=False`)** |
| `probe_boundary_contract.py` | 同端短早期异常口 `[3,4] + 40..55` (`Hegel` & `O_strong`) | 误报 `40` (`false_confirmation=True`) | **精确报 `3` (`false_confirmation=False`)** |
| `probe_m0_semantics.py` | `cosmetic_key_overstrips_semantic_literal` (`value.notes`) | `meaning_unchanged=True`（漏检） | **`meaning_unchanged=False`, `answer_retained=False`** |
| `probe_m0_semantics.py` | `nested_execution_error_result_accepted` | 接受错误为答案 | **`return_status="scoped_unknown"`, `semantic_answer=None`** |
| `probe_m0_semantics.py` | `workflow_historical_outcome_without_predicate` | 重放 `REFUTED_IN_SCOPE` 多买探针 | **`node_outcomes={"step_1": "UNKNOWN"}`, 不误买支线探针** |
| `probe_m0_semantics.py` | `forest_revision_identity_gap` | 丢 `question_id`、召回旧版边、保留旧父边 | **`question_id="fq"`, `meaning_version=2`, 旧边召回数 `0`, 旧父边 `False`** |
| `probe_m0_semantics.py` | `unknown_exclusion_and_declared_tier` | 越权授予 `validated_pruning` | **`granted=False`, `granted_tier="retrieve_candidate"`** |
| `probe_m0_semantics.py` | `m1_ordered_gate_denial_not_enforced` | `auth=False` 仍报 `used_wormhole` | **`auth=False` 严格阻断虫洞分支，转直接求解** |
| `probe_m0_semantics.py` | `fidelity_accepts_conflicting_verdicts` & `unknown_promotion` | `accepted=True`（漏检冲突与升级） | **两者均 `accepted=False`，精确定位冲突与越权索引** |
| `probe_formation_and_ablations.py` | `archive_vs_experiences`（原始档案完整性） | `[4, 7]`（缺 3 个近失例） | **`[7, 7]`（正式评测含尺度探针为 `[8, 8]`，100% 入档）** |
| `probe_formation_and_ablations.py` | `all_source_failures_still_edge`（全零源反事实） | `True`（伪支持） | **`False`（标为 `NOT_DETECTED_UNDER_PROTOCOL`，零虫洞）** |
| `probe_formation_and_ablations.py` | `duplicate_source_still_edge`（同源复制反事实） | `True`（伪造生化树与边） | **`False`（源指纹+域去重，零伪造节点与边）** |
| `probe_formation_and_ablations.py` | `failed_confirmation_promoted`（cap=10 vs target=16） | `"SUPPORTED_IN_SCOPE"` | **`"REFUTED_IN_SCOPE"`（保留弱联合探针问题，拒绝强外推）** |
| `probe_formation_and_ablations.py` | `ordinary_history_same_trace`（`O_strong` 历史消费） | `True`（未消费历史） | **`False`（`O_strong` 重建缓存先验，`T_B1` 测量费 `42.0 -> 24.0`）** |
| `probe_formation_and_ablations.py` | `remove_align_exclusion_same_trace`（排除谓词因果杠杆） | `True`（被预写布尔遮蔽） | **`False`（移除排除谓词在 `T_B5` 多耗费 `6.0` 确认探针费）** |

---

## 三、V16.7 升级版跨领域虫洞实验实测结果（赋能强历史重建基线后）

### 1. 核心配对结果总表（`N=4` 独立冻结评测簇，每簇 14 道未见目标 = 每臂 `56` 道目标；全部 8 臂严格通过率均为 **`56/56 = 100%`**，假确认 **`0`**）

| 配对比较指标（负值代表前者更省成本） | `eval_01` | `eval_02` | `eval_03` | `eval_04` | 跨簇均值 `mean_diff` | 4/4 簇一致为负？ |
|---|---:|---:|---:|---:|---:|---|
| **给定冻结门控端到端总成本 `Hegel - O_strong` [含历史重建缓存]** | `-23.33` | `-23.03` | `-23.33` | `-23.33` | **`-23.25`** | **是（4/4）** |
| **给定冻结门控端到端总成本 `Hegel - O_HExA_inspired` [真读技能库]** | `-63.98` | `-63.68` | `-63.98` | `-63.98` | **`-63.90`** | **是（4/4）** |
| **未见目标总复合成本 `Hegel - O_strong` [含历史重建缓存]** | `-20.18` | `-19.88` | `-20.18` | `-20.18` | **`-20.10`** | **是（4/4）** |
| **未见目标总复合成本 `Hegel - O_HExA_inspired`** | `-57.73` | `-57.43` | `-57.73` | `-57.73` | **`-57.65`** | **是（4/4）** |
| **未见目标纯环境测量费 `Hegel - O_strong` [含历史重建缓存]** | `-18.00` | `-18.00` | `-18.00` | `-18.00` | **`-18.00`** | **是（4/4）** |
| **未见目标纯环境测量费 `Hegel - O_HExA_inspired`** | `-54.00` | `-54.00` | `-54.00` | `-54.00` | **`-54.00`** | **是（4/4）** |
| **同历史完整档案对照 `Hegel - O_strong_SameHist_RebuildOpt`** | `-20.18` | `-19.88` | `-20.18` | `-20.18` | **`-20.10`** | **是（4/4）** |
| **簇 A（有序离散边界，含完备前缀证书）`Hegel - O_strong`** | `+0.75` | `+0.75` | `+0.75` | `+0.75` | **`+0.75`（诚实持平略贵）** | 否（双方共享最优原语） |
| **簇 B（饱和与多资源瓶颈）`Hegel - O_strong`** | `-20.43` | `-20.13` | `-20.43` | `-20.43` | **`-20.35`** | **是（4/4）** |
| **簇 C（真实包事件流队列多视图/多尺度）`Hegel - O_strong`** | `-0.50` | `-0.50` | `-0.50` | `-0.50` | **`-0.50`** | **是（4/4）** |
| **簇 C（真实包事件流队列多视图/多尺度）`Hegel - O_HExA_inspired`** | `-6.83` | `-6.83` | `-6.83` | `-6.83` | **`-6.83`** | **是（4/4）** |
| **单因素跨域虫洞消融 `Hegel - Hegel_no_horizontal_wormhole`** | `-57.58` | `-57.28` | `-57.58` | `-57.58` | **`-57.50`** | **是（4/4）** |
| **表示结构纯消融 `Hegel - Flat_equal_content`（1-hop 同内容同权限）** | `+0.20` | `+0.20` | `+0.20` | `+0.20` | **`+0.20`（诚实分离单跳同分）** | 否（1-hop 下由合同内容主导） |
| **版本化对象移除+合法重建干预 `Hegel - ObjectRemovedRebuild`** | `-21.93` | `-21.63` | `-21.93` | `-21.93` | **`-21.85`** | **是（4/4）** |
| **双源对比 vs 单源形成（未见目标净差，`N=14`）** | `-38.75` | `-38.45` | `-38.75` | `-38.75` | **`-38.68`** | **是（4/4）** |

### 2. 前缀摊销地平线曲线 $\Delta C(N)$（`N = 1 .. 28` 道未见目标/簇，4 簇均值）

| 未见目标地平线 $N$ | 目标净差 `Hegel - O_strong` | 含源端到端 `Hegel - O_strong` (给定门控) | 含源+开发校准分摊 `Hegel - O_strong` | 目标净差 `2-Source - 1-Source` | 含源端到端 `2-Source - 1-Source` |
|---:|---:|---:|---:|---:|---:|
| **`N = 1`** | `+0.15` | `-3.00` | `+316.56` | `0.00` | `+67.98` |
| **`N = 4`** | `+0.60` | `-2.55` | `+317.01` | `0.00` | `+67.98` |
| **`N = 8`** | **`-19.53`** | **`-22.68`** | `+296.89` | **`-19.53`** | `+48.45` |
| **`N = 14`** | **`-20.10`** | **`-23.25`** | `+296.31` | **`-38.68`** | `+29.30` |
| **`N = 28`** | **`-38.80`** | **`-41.95`** | `+277.61` | **`-77.35`** | **`-9.38`（跨过总账盈亏平衡！）** |

> **关键科学解读**：
> 1. **正面回答 MathRSI/GPT-6 核心之问（“当强普通方法能从原始历史重建同一知识时，Hegel 价值何在？”）**：本轮赋能 `O_strong` 直接从 `raw_archive` 拟合并缓存 `rational_saturation` 先验后，`O_strong` 在 `T_B1` 与 `T_C1` 上的探针数确实追平了 `Hegel`（同为 `4` 次探针、`24.0` 测量费），差距从 V16.6 的 `-55.35` 收敛至真实硬核净优势 **`-20.10`/簇（端到端 `-23.25`/簇，4/4 簇全胜）**——这 `-20.10` 的净价值 100% 来自**免除目标期原始轨迹重建税**以及**从近失例结构化合成的多资源 `CEGAR` 最小探针程序**（在 `T_B2` 省 `6.0` 测量费、在 `T_B3` 省 `12.0` 测量费），无任何基线削弱水分！
> 2. **清晰量化长期摊销逻辑**：双源对比学习多付一个异质源回合（`+67.98` 前期成本），但每遇到一轮跨域目标即省下 `38.68` 成本，在 $N \approx 22$ 道未见目标处实现全周期总账反超（$N=28$ 时净赚 `-9.38`/簇）；一次性开发门控校准成本（两开发簇合计 `1278.25`）亦单列透明披露，随问题流增长稳步摊销。

---

## 四、下一步推进路线（`M2 -> M3`）

1. **`M2`（开放代码工作区与在线 LLM 自主合成接线）**：将当前已验证的 `PhilosophicalForest` + `HorizontalWormholeEdge` + 收据式四级权限门接入开放 Python 实验工作区，让候选 `ClaimContract`、`ObservationMap`、`ScaleMap` 与 `program_steps` 由受门控的在线代理从真实实验残差中自主提议并经反例验证入库（替代固定插件语法枚举）。
2. **`M3`（大规模多簇森林 $K \ge 32$ 与多跳虫洞检索评测）**：扩充异质机制簇规模至 $K \ge 32$ 棵局部树并引入真实 2-hop / 3-hop 跨学科虫洞路径与近似角色干扰项，正面评测 `PhilosophicalForest + HippoRAG PPR + 非传递误差界` 相对 `Flat_equal_content` 在检索精度、多跳防漂移与 sub-linear 匹配开销上的独立图结构优势。
