# v18.1–v18.4 独立诊断：计划、实现、逐题证据与有限改进方案

审计固定提交：`922b95e194d17c9259f1993706440e0602a43db9`。M6 的 `b1d0c3e99` 是从 `63b940a03` 分出的兄弟提交，不能把两个分支的新增能力直接叠加为一个已运行系统。

## 1. 总判断

这几版的实际进步很明确：真实模型会提出探针、提交可执行 Python、在隔离于拟合输入的确认点和最终审计点上接受检查；账本完整性和逐步费用保存也显著改善。v18.4 保存了完整的 2,160 个 live episode，报告中的通过数和加权成本可以从提交逐行复算。不能把这一批工作归结成“只有展示”，也没有证据说日志是伪造的。

但目前仍不能据此验收“F01–F11 全部闭合”“在 240 个独立新机制上证明自主哲学树 RSI”“通过自动学习形成了 45→48 节点的科学知识”“在同等资源条件下稳定优于 Strong”。几个决定性障碍有可执行反例：

1. **后继新公式由 subfamily/task_id 分派器写死，未由成功作者程序抽取。** 保持成功状态不变，把作者程序换成 `predict=-999`，写回出的公式完全不变；删除程序也能完成 45→48。
2. **机制标签仍直接从生成器进 live 提示，Hegel 另获固定的组合公式和决策处方。** 这能证明有结构指导的 agent 的行为，但尚不能分离自动抽象、图组织、内容丰富度、手写策略各自的贡献。
3. **同一行观测的 live 计价仍随实验臂变化。** 只对原动作轨迹重计同价，不改成功率、不重新实验，三个模型的 Hegel 相对 Strong 的总 C 优势都翻转。
4. **独立复制的 traced 执行器落后于修复过的真实 pipeline。** 强制 HEC stop 时，真实 pipeline 花 M=12，traced 仍花 M=14，而且记录链校验通过。
5. **已出现的损失远未说明算法达到潜力上限。** 一个 Luna 真正失败的题，仅接通已有 `fit` 返回值到 `predict`，确认 RMSE 就从 3.348349 降至约 2.2×10⁻⁸；一个假确认仅放开已有残差项的符号，heldout RMSE 从 0.057020 降至约 4.15×10⁻⁸。它们不需要新哲学，也不需要牺牲多尺度和部分共性结构。

建议保持“跨时刻动态 × 多尺度”和“跨领域部分共性结构逐层组合”的复杂度，先补接真实作者、真实历史、权限结果、后继消费与反事实归因；这比继续增加同类参数实例更能增强现有论证。

## 2. 审阅与复算范围

按要求先读修改后的计划索引及 22、23、24、25 四篇工作报告，再核对实现。`20` 主计划在本次分支差异中没有修改。检查了四个新运行模块、M5/forest/旧灰区与摘要相关改动、五个新脚本和 v18 测试，以及四批新证据。使用 `AGENTS.md` 的历史复用入口，检索了 reuse hub、181 版本 catalogue 和现存研究/重组/付费历史模块。没有运行付费模型，没有修改仓库源码，没有把当次未运行的既有能力称为算法缺失。

所有 SHA 条目及所有 live DAG 均已机械验证，全部 v18.4 失败 episode 均已抽取决策、确认记录、费用和工作区路径；对关键失败及因果断点做了局部源码/保存程序复现。这是“全记录扫描 + 关键路径语义检查 + 有界反事实”，不应被误解为对 12,701 个文件所有科学含义的形式证明。

| 证据根 | SHA 条目 | live episodes | live 决策记录 | SHA / DAG 错误 |
|---|---:|---:|---:|---:|
| `v18_1_m5_live_gpt_and_decision_provenance_20261120` | 378 | 54 | 395 | 0 |
| `v18_2_m5_sovereign_closure_and_live_subscription_20261121` | 95 | 12 | 108 | 0 |
| `v18_3_m5_complete_3model_continual_complex_and_amortization_20261121` | 584 | 84 | 754 | 0 |
| `v18_4_hegel_discovery_240_20261121` | 11,644 | 2,160 | 15,679 | 0 |
| 合计 | **12,701** | **2,310** | **16,936** | **0** |

另验证 v18.1 的 80 条 deterministic episode 链、630 条记录，以及 v18.2 的 22 条 deterministic episode 链、170 条记录，均通过现有 verifier。**链完整性成立，不自动推出这些记录对应同一执行器，也不自动证明“被拒绝的权限真的改变了执行”。**

v18.2 报告中的 800 次十臂主基准与 32 次 stress，新的证据根保存了汇总，未像 live 那样保存全部逐题原始 receipts；所以上述“逐行复算”严格指 2,310 个 live episodes 和提交中的 deterministic provenance，不冒称已经从 800/32 条原始数据独立复算其全部汇总。已有 J 流 receipts 和局部闭合记录仍可检查。

复现文件：`v18_recompute.py`、`v18_recomputed.json`、`v18_all_episode_metrics.csv`；最小反事实：`v18_targeted_controls.py`、`v18_targeted_controls_results.json`。脚本只读原提交，在内存中运行少量已保存程序或改变控制变量，不调用模型。环境未安装 pytest，所以不把报告中自述的 pytest 数量当作本轮重新运行的结果。

## 3. 各版到底增加了什么

| 版本 | 可以成立的新增证据 | 应限制的解释 |
|---|---|---|
| v18.1 | 3 模型 × 3 臂 × 6 题共 54 次 live；作者代码、工具动作、token 与确认链可查 | 6 题是有限机制夹具；提示中有机制签名和明示方程；无法单凭 rationale 证明独立推理过程 |
| v18.2 | 原 M5 的同类宽域回退、真实选探针、冷快照消费、PS​​D/lineage 治理、时间原点修复；增加同价与 compiled-subtree deterministic 对照 | 闭合测试 9 个检查族，不等于 F01–F11 的所有语义义务；同价/同内容对照没有接入 live 主榜 |
| v18.3 | 36 次新 L01–L08 补跑，48 次 CX 多波 live；有持续状态和固定写回动作 | 72 格 L01–L08 总表混合旧 baseline 与新 Hegel；CX 缺 sol Flat 六格；写回仍从题号分派 |
| v18.4 | 真正完成 2,160 次 live；12 类 × 18 参数实例 + 24 OOS；K=45 的混淆检索压力 | 240 个 task_id 不等于 240 种独立发现；F09/F10 的变换后完整方程已在 source note，不能据此认定完成外部反记忆 benchmark |

v18.3 原始新结果与合并表要分开。新 CX 部分：Luna Hegel 5/5、Flat 1/5、Strong 2/5；Terra Hegel 5/5、Flat 4/5、Strong 5/5；Sol Hegel 与 Strong 都 5/5，各另有 1 个 OOS。这个小样本有价值，尤其提示结构处方能帮助较弱模型；但它不足以给出后续 v18.4 一定胜出的结论。[S3]

## 4. 上轮 F01–F11 的闭合核对

| 原发现 | 本轮判断 | 具体理由 |
|---|---|---|
| F01 原始历史入口/机制信息 | **未闭合** | deterministic 把经验签名与 `ds.probe_signatures` 合并；live 仍直接用后者。M4 又存在 `_raw_prefix`/`raw_prefix` 错接；helper 被调用不等于真实历史参与预测 |
| F02 源经历 → 可执行共享结构 | **有实质修复，但只部分成立** | 3 个源真的 least-squares 拟合并 anti-unify 出执行 AST；但共享 C1/C2 关系手填，源 confirm 只收费未执行，K45 全档案未按此形成 |
| F03 VOI 动作控制观测 | **真实 pipeline 已修，traced copy 未闭合** | 前者尊重 HEC/selected probe；后者仍在 stop 后无条件加 open probe |
| F04 学得内容/冷恢复因果依赖 | **旧 J 流修复成立，新 live 流重新断开** | 旧 J19 删除 leak 槽位会改变后继代价；v18.3/4 新公式由 task metadata 分派，忽略作者 AST |
| F05 对照归因 | **deterministic 改善，live 未闭合** | deterministic 新增同价 Strong 与同内容 compiled Flat；主 live 沿用 2/1.5/1 M 和不匹配的 branch guidance |
| F06 同模型宽域回退 | **确定性局部错误已修** | 已有 confirm-triggered same-class wide escape；live 是给 LLM 一条建议，实际采纳仍受作者代码/参数界影响 |
| F07 联合绑定/部分激活 | **有修复但仍有可复现遗漏** | 空条件、显式空边、角色折叠已处理；省略边或无关边仍拿完整授权，live 不消费数值权限结果 |
| F08 旧灰区写回/恢复 | **已改实际旧路径；范围有限** | 新写回存 max_supported/min_refuted，旧 snapshot dispatcher 补推导灰区；新 F07 live 灰区端点却由常数给定，不能称自动实测边界 |
| F09 多样性 | **所列旧反例有实质修复** | lineage 内保守合并、相关矩阵 PSD 投影；本轮没发现足以否认这些局部修复的新证据 |
| F10 时间原点 | **数学 helper 修复成立** | 改成相对时间差；但 raw history 接线仍应单独验收 |
| F11 治理/历史摘要 | **存储覆盖有实质修复；live 消费未闭合** | v2 digest 和 history 验证扩大了覆盖；数值拒绝返回后 live guidance 不变 |

“100% closed” 的程序谓词仅是 `all(v['passed'] for v in checks.values())`。F01 对应检查只是 monkeypatch helper 抛异常被捕获，F02 只检查 3 个 estimates、正费用、3 槽位、AST 类型；它们不能承担完整结论。[S2:88–468]

## 5. 逐项诊断

### V18-R01 — P1：持续学习写回由语义题号/类别触发，作者发现未参与科学内容

**位置**：`m5_v18_3_continual_and_complex_live.py:392–552`；`m5_v18_4_hegel_discovery_240.py:785–936`。

v18.3 遍历 `confirmed_ok and strict_pass` receipt，再用 task_id 包含 `CX01…` 或 `CX02…` 判断写入哪条固定 ODE/avalanche。v18.4 用 `confirmed_subfams` 选择 F07/F11/F12，再写死完整公式、参数个数、0.52 feedback、灰区范围和下次应先买的探针。函数没有读取作者代码、code SHA、拟合参数、残差结构、归纳后的 AST 或实测边界点。

**最小反事实已执行**：保持三个 receipt 的状态、strict_pass 和任务元数据不变：

- 没有任何作者代码：仍生成 3 个新节点，K45→48，成本 5.4。
- 加入与原科学式完全矛盾的 `def predict(row): return -999`：新节点内容 digest 与无代码情形完全相同。
- 只把 subfamily 改为匿名名：K 仍为 45，没有事件。
- v18.3 只把 `CX01/CX02` 题号改成无语义名，保持成功状态：K15 不增长。

因此问题是**科学内容的因果来源错误**，而不是反对使用已完成源任务的评估反馈。在线训练协议可以在任务结束后使用反馈，但要明示其预算和训练身份。即便允许所有这些反馈，当前代码也没有从作者成功程序提炼出它后来宣称学到的内容。

**最小修复**：把成功模型的 immutable code/AST、fit state、公开观测、确认范围、读取角色/单位、失败残差交给现有 consolidation/reconciliation 路径。`branch_reconciliation.py` 的 anti-unification、付费证据裁决、scope 收缩及 `OutcomeDrivenForestConsolidator` 已存在；这里缺的是 Python 作者输出到这些入口的适配和消费证明。对任意 Python 的完备 AST 归纳不是现成保证，可先保留可执行程序/明确声明的结构候选，再经原证据确认，不能用题号替代。

**验收**：匿名改名不改变科学写回；改变成功作者程序/范围会改变写回内容或阻止写回；去掉一个学得分支会有可解释的后继行为差异。严格保留局部共性和多个时间尺度，不要把它们退化成单一公式分类器。

### V18-R02 — P1：真实观测、生成器机制注释与组合处方没有分离

**位置**：`m5_temporal_multiscale_and_base_subtree.py:1776–1864,2012–2021`；`m5_live_gpt_and_provenance.py:1309–1330,1347–1354,1405–1490`。

deterministic 新 helper 的确根据 pilot 拟合出一些经验特征，但随后 `active_sigs=[*ds.probe_signatures,*emp_sigs]` 保留原本的机制信息。live 根本没有调用这个经验签名入口：直接把带机制语义的 task_id、真实 tau scale 和 `ds.probe_signatures` 交给作者及检索器。

Hegel 的 `branch_rule` 又直接从这些签名给出正确组合方程：memory×siphon、dual-horizon×sink、Hill-2、barrier flux 等。第一波尚未“学到” O/S 节点时，F11/F12 的五参数组合式已经在初始提示。随后自动写回的节点不是这些组合知识首次进入系统的时刻。

Flat 的 top3 notes、Strong 的全档案与 Hegel 的组合处方不是同一内容编码；允许 Hegel 这样设计为一种整体系统本身没有问题，但不能把胜差只记作图结构或自动归纳的贡献。[S1]

**最小修复/复用**：保留现有 rich representations，建立“可观测特征计算 → 有不确定性的 feature receipt → 多层匹配/组合计划”的入口；所有原始公共信息对各臂相同。提供 lossless compiled guidance 基线，使 Strong/Flat 也能拿到同一已获得结构，只去掉索引/树组织；另作移除组合计划但保留内容的消融。生成器元数据放在 auditor 侧，题号改成匿名 ID。若给定 tau 是合法仪器校准，应明确列为共同输入，不能称作从历史估计的结果。

### V18-R03 — P1：OOS 的全通过是显式标签遵循，尚非未知机制识别

**位置**：live system rule 4，`m5_live_gpt_and_provenance.py:1309–1311`；OOS generator，`m5_v18_4_hegel_discovery_240.py:710–738`。

系统提示明确写出遇到 `sig:high_freq_sign_oscillation` 要 abstain；全部 OOS 都含此签名，task_id 本身也有 `OOS_Adversarial_NonStationary`。逐 DAG 复算：9 个模型/臂组合的 216 次 OOS 运行全部只有一次 `abstain_scoped_unknown` 工具决策，没有提交候选。

它证明所有臂遵守了声明的范围退出规则；不能证明模型从这五点数据中识别出新型非平稳/混沌过程，更不能用它排除在无标记未知机制上错误泛化的风险。生成式本身是已知平滑正弦余弦函数，“chaotic”也不是代码中已验证的动力学性质。

**最小修复**：保留这些 protocol compliance 题，另加入去掉任务语义标签、只保留同样数据的成对控制；让有限候选在目标相关区间出现共同盲点，检查原有 mandatory confirmation/CEGAR 能否诚实退出。此处不需删掉多尺度模型或追求单一可解释式。

### V18-R04 — P1：live 同样观测行计价不一致，核心成本胜差依赖协议设定

**位置**：`m5_live_gpt_and_provenance.py:1599–1604,1813–1818`。

Strong 每行 2 M，Hegel/Flat 常规每行 1.5 M，带 coarse 标记每行 1 M，但各臂从同一个 `ds.pilot_rows`/`confirm_rows` 获取同样 payload。v18.2 deterministic 的 equal-unit-price 对照没有迁移到 2,160 次 live 主表。

只对原来的收费行重计 1.5 M，**不改变任何行数、动作、成功率，也不是一次新的公平实验**，得到：

| 模型 | 原 Hegel C | 原 Strong C | 同价敏感性 Hegel C | 同价敏感性 Strong C | Hegel − Strong |
|---|---:|---:|---:|---:|---:|
| Luna | 3738.70860 | 4680.69330 | 3810.70860 | 3759.69330 | +51.01530 |
| Terra | 3603.72690 | 4452.14440 | 3675.72690 | 3531.14440 | +144.58250 |
| Sol | 3549.15835 | 4433.01135 | 3621.15835 | 3512.01135 | +109.14700 |

即便保留 Hegel coarse 优惠，只让 Strong 常规行 2→1.5，Terra/Sol 的 Hegel 仍更贵；Luna 暂便宜 20.9847，若再单独给 Hegel 加报告中的 33.008 源形成/写回费用，则该优势也消失。此分析仅说明“低 C”的归因很敏感，不能用它反向宣称同价重新运行后 Strong 一定获胜。

**最小修复**：复用已有 deterministic 同价臂，把价格作为观测类型/仪器成本，不作为 arm 身份；coarse channel 若确实提供更便宜的实际数据，给出不同采样/读出机制和证书成本。继续报告 M、K、实际 wall time、token 各项，不只报告 `M+0.05K`。各模型相同 token 权重只是研究代理成本，不能直接当作真实订阅费用或“弱模型美元更便宜”的证据。

### V18-R05 — P1：provider 故障消耗科学轮数，混入算法失败和成本统计

**位置**：`CodexSubscriptionClient`，`m5_live_gpt_and_provenance.py:852–1086`；错误回路 `1642–1661`；v18.4 worker `1164–1181`。

全部 2,160 DAG 有 **259 条 provider error**：164 次 `codex_subscription_timeout_135s`，95 次 `Unknown feature flag: code_mode_host`。错误被作为一次 scientific turn 消耗，六轮耗尽就成 `scoped_unknown`/域内失败。Sol Hegel 的 8 个失分、Sol Strong 的 2 个失分、Terra Hegel 的 2 个失分都没有提交任何候选程序，不能按科学求解失败诊断。

timeout 返回 token=0 是“没有拿到计量”，不能推出没有消耗；最多四次内部 provider retry 只解析最后一个 proc 的 usage，也不能保证覆盖先前 attempt 成本。已经记录的失败要保留，不能把这一批事实抹掉重跑后替换旧分数。

**最小修复**：复用 `experiment_runtime`、`paid_score_history` 和可恢复 episode 状态。单独计 provider attempts/unknown usage/scientific turns；环境特性检查失败应直接标记 infrastructure_invalid，恢复后续跑同一已付费 episode。保留当前 ITT 总成绩，同时并列可执行 episode 条件诊断、预算敏感性及超时率。模型科学能力的比较需要同等实际计算机会。

### V18-R06 — P1：live 未消费联合绑定的数值权限；边覆盖验证也有 fail-open

**位置**：`wormhole_forest.py:2361–2552`；live caller `m5_live_gpt_and_provenance.py:1395–1501`。

判定 `explicit_edges_present = 'edge_bindings' in candidate_binding` 后，只拒绝“显式传入空边”；没有传该字段则自动绕过此检查。角色全部存在且 C1/C2 标记满足，就把 causal relations 计入 active，而不要求提交边覆盖正确端点。

**控制结果**：相同合法角色与 C1/C2 前提，省略 edge bindings → 完整数值 warm 授权；放一个与真实结构无关的 edge → 同样授权；显式空列表才被拒绝。live caller 正好省略 edge bindings。

即使假定 validator 完全正确，live 只读取 `merged_role_bindings`，直接拿 `priors_bundle` 的 warm box。将 validator 返回值单独换成 `numeric_warm_start_authorized=False / retrieve_candidate_only`，给作者的 `shared_base_subtree_guidance` 完全不变。因此 F11 的存储治理改进尚未在当前 live 路径变成实际约束。

**最小修复**：完整关系匹配要检查 required causal edges 的端点/类型/readout/时间锚；缺边时保留 partial candidate 的合理价值，禁止完整数值 prior。live 必须消费 tier 和 authorized 字段，并从已核验 persisted subtree 重构 prior，不能靠独立 bundle 旁路。增加由有效降权/撤回产生的端到端权限拒绝测试，而不只改 hash。

### V18-R07 — P1：traced deterministic executor 与被审计 pipeline 是两份不同实现

**位置**：`m5_live_gpt_and_provenance.py:257–700` 对比 `m5_temporal_multiscale_and_base_subtree.py:1867–2340`，尤其 live copy `430–451`。

traced 函数没有 instrumentation 包裹真实执行器，而是复制算法。它仍保留旧 Strong 全量初始购买、原生成器签名、权限旁路和 C 分支无条件 open-probe 购买。已有 parity 测试只遍历 `tasks[:8]`，覆盖不到 C 类 HEC 反例。

**控制已执行**：固定数据/RNG，强制 `select_m5_discriminating_probe` 返回 HEC stop。真实 pipeline M=12；traced M=14，并增加 probe row；后者 verifier 仍 valid。两者都通过当前题，因而只看成功率会漏掉动作/成本语义差异。

**最小修复**：将 ledger 作为真实执行器动作边界的 observer，同一份执行代码产生账本；不要维护第三份“为审计而重写”的 solver。对 stop、不相关 probe、权限下降、确认失败宽域回退各做一项轨迹等价控制即可，无需再扩大通用测试量。

### V18-R08 — P1：M4 真实原始历史被字段错拼旁路

**位置**：`m5_temporal_multiscale_and_base_subtree.py:1184–1286`。

reader 用 `r.get('_raw_prefix')`，但 L08 与 F01 producer 写 `raw_prefix`。匹配失败后 helper 只收到 t=0 的单点锚。后续只使用 `0.52*raw_prefix[0]['x']` 作为 latent initial，整个中段历史没有参与。

**控制已执行**：L08 实际 9 点 prefix；对 extractor 加 spy，所有调用的输入长度都是 1。仅把中间历史各 x 加 100，保留首尾锚和当前观测，其预测仍完全相等，都是 1.59210649。

这是确定性 M4 路径的明确错接；**不能把它直接当作所有 live F01 结果失效**，live 给了另一条明确的积分提示，保存作者程序可以真正读 raw_prefix。恰恰说明两个路径必须分开验收。

**最小修复/复用**：统一 row schema，接回已有 causal temporal extractor 和 memory propagator；增加“首尾相同、中间历史不同”的消费控制以及时间平移控制。旧 `language.py`/world commitment 中已有记忆、AR2、POMDP 能力，当前问题不等于需要重造一个时序算法。

### V18-R09 — P1：作者的 fit/predict 接口错误被当成科学模型反证

**位置**：live history 组装 `1058–1083`；程序执行 `1090–1180`；失败处理 `1810–1920`；实际样例见失败附件。

host 的合同是先 `fit(train_rows)`，再调用 `predict(row)`；它不自动把 fit 返回值传给 predict。有些 Luna 作者返回正确参数，却在 predict 中引用未定义 `_FIT_PARAMS`/`_train_rows`；另一些每个预测又调用 `fit([dict(row,y=0)])`，把正确公式 refit 成近零。history 又不保留上一轮完整 `python_code`，只记动作名、hypothesis 和 rationale，之后作者拿不到准确的当前待修代码。

**实证局部修复**：Luna Hegel `W1R3__F06_NarrowPilot_HeteroParam_WideEscape/model_step_02.py` 已拟合出 `[7.5509994,0.9140001,0.3650002]`。没有改变方程、训练数据、初始化、bounds 和拟合参数，只把该结果保存并给 predict 用：

| 指标 | 原保存程序 | 仅修接口 |
|---|---:|---:|
| train RMSE | 3.2080505 | 2.29×10⁻⁸ |
| confirm RMSE | 3.3483490 | 2.20×10⁻⁸ |
| heldout RMSE | 3.4758080 | 2.19×10⁻⁸ |

原轨迹把这个正确结构判为 refuted，转去 Hill 和 barrier 等其它机制，最终六轮耗尽。这是可避免的损失，绝不是“现有哲学知识无法覆盖此题”。本地修复只是已保存候选的诊断，不是把主榜某个失败自动改成成功。

没有 Python exception 也不保证是科学反证：Luna Hegel `W1R3 F12/model_step_02.py` 的正确五参数式在 predict 中又调用 `fit([dict(row,y=0.0)])`；`W5R2 F11/model_step_02.py` 的 predict 重拟合 `_FIT_ROWS=[]`，因而实际上用了初始化参数。后者另把 `tau_fast` 上界设为 0.5，生成器允许 0.48–0.65；这里有接口和参数域两层问题，不能仅凭保存的 `CONFIRM_REFUTED` 标签定为模型类不适用。

**最小修复/复用**：借用原 `code_agent.py` 保留已评价 Python、既有 `python_evidence_repair`/`code_provider_context` 路线；明确一种稳定 ABI（例如 `fit -> model_state; predict(row,state)`），训练/预测运行前做 schema、状态持久性、无 y 依赖检查。错误要区分 `implementation_invalid`、`optimization_failed`、`scientific_refuted`；先修实现再推翻机制。把上一轮确切代码或 immutable reference 返回作者。

### V18-R10 — P1：错误符号域与薄确认集产生可重复的假确认

**位置**：v18.4 F10 generator `617–649`；live acceptance `1857–1899`；保存程序 `W2R1__F10…/model_step_01.py:20–25`。

F10 生成器允许 `c_sqrt` 正负；多个作者把它限制非负。补偿的其它参数在 3 个 confirm 点内可过 0.045，但在 heldout 交叉输入上露出误差。Luna Hegel 的 6 次 false confirmation 中，4 次属于 F10，另 2 次 F04；Flat 同类 F10 假确认反复出现。

**反事实修复**：Luna Hegel W2R1 F10 只把 `c_sqrt` 下界从 0 改为 −2，其余不变：confirm 0.02355066→3.37×10⁻⁸；heldout 0.05702019→4.15×10⁻⁸；恢复 `c_sqrt≈−0.344`。未用 confirm/heldout 标签拟合。

问题不止“多买几行”。当前 pipeline 见 confirm≤tol 就接受，即使训练残差已显示系统性误配；同一三点 confirm 还被多个候选反复查询，却一直叫 fresh。数据最初与训练分离是真的，但反复反馈后它是 validation channel，最终 heldout 才是独立审计。

**最小修复**：source/slot 需携带带符号的合法参数域而非让 LLM凭机制名臆测；复用 `probing.py` 的候选分歧/目标风险选择，重点查询能分辨正负残差或 equilibrium/contrast 的点。确认权限按噪声与残差结构校准，不应只看三点平均 RMSE；固定验证集可保留并声明为自适应 validation，最终冻结后使用独立审计集。灰区与 failed source 也应入学习路径，以免不同 wave 重犯同类错误。

### V18-R11 — P1：源形成成本不是整棵森林的自主获得成本，且有“确认收费未执行”

**位置**：`m5_temporal_multiscale_and_base_subtree.py:1645–1743`；v18.4 forest `70–290`；CPU timing `1077–1084`。

三条 source 的参数拟合是真的，但训练式已经给定，三种跨领域身份也是人造 metadata。每个 source 只生成 5 个训练 y 并 least-squares，随后直接记 `5*1.5 + 1*1.5 # 5 pilot + 1 confirm`；不存在第六点的独立确认计算和接受门禁。C1/C2 causal schema、verified/stability 标签仍手写。新增 K45 的正确公式、桥接节点及 25 distractor 更没有共同经历同一 source 形成/验证流水线。

“<10ms CPU / zero GPU / 27.608C”测的是执行已有程序进行三次已知结构的合成拟合及组装，不包括基础模型预训练，也不包括发现/校准全部 45 个知识节点、组合规则和 readout 映射。作为**增量符号处理成本**可以报告，作为替代神经预训练的总形成成本则口径过大。

**最小修复**：源 ledger 保存真实 pilot、独立 confirm、程序 hash、拟合参数与判定；已注入人类给定 archive 标注为 common endowed knowledge，不伪装成当轮自动发现。源历史如是外部给定，比较双方共享继承成本；如要证明从零学得，接回仓库原有 source episode 形成过程并计账。

### V18-R12 — P2：灰区端点与时间尺度测试的语义比标题窄

**位置**：v18.4 `524–551,838–851`；F04/F05/F09/F10/F11 generators `450–706`。

F07 第一波 pilot 最大 s_grad 是 2.34，high-regime probe 从 2.78 起，而写回声称 verified 的 max_supported=2.36、min_refuted=2.68。这两个端点不是当前函数从带标签证据中得到的；它们是为后续 wave 最大 pilot=2.44 预设的常数。灰区逻辑的执行可见，但自动边界推断证据不足。

若干“跨时间尺度”题把 `u_past/u_now` 乘 `t_scale`，公开 `tau_scale` 让作者除回去，却保持 H、relaxation tau、prefix 时间表不变。它们验证的是输入幅度/单位归一化的某种迁移，不等于把真实时间过程快慢整体缩放后保持预测关系的检验。F01/F02 另有实际动态过程，这一限制不应抹掉它们。

**最小修复/复用**：gray zone 引用真实 supported/refuted 点及单调性前提；直接接既有 threshold boundary probe/curved hypersurface probing，多个条件/尺度可以继续保留。增加时间坐标、H、intrinsic tau、观测间隔与 readout 一起变换的合法 gauge 对照；幅度尺度和时间尺度分别报告。

### V18-R13 — P2：240 题及持续收益的外推范围，需要按机制族和学习状态报告

**位置**：v18.4 generator `342–778`；v18.3/4 wave runner；reports 24–25。

12 个 in-scope 名称也不全是不同方程：F05/F06/F08 都是同一 siphon+sqrt，只改参数、pilot 范围或桥接标签。很多行位置在 18 个任务间固定；F09/F10 的完整目标式已事先放在 Q/R archive。所以任务总量增加是真实的，但统计单位有共享机制、共享模板和共享 prompt 相关性，不能把每个参数重采样当作独立的新科学领域。对齐 LLM-SRBench/DiscoveryBench 的题量不自动对齐它们的输入、未知结构、评分或泛化协议。

按模型维护一份森林，三臂共用；每波只用 Hegel receipt 写回，下一波 Flat/Strong 也收到新 notes。这可以定义为“共同训练内容下的组织比较”，甚至会帮助 baseline；但它不是三个独立 self-improving learners。只写回 F07/F11/F12 后其余波几乎不再发生结构成长，随后准确率提高可能是公共内容放入 archive，而不等于递归提升了学习算法。

**最小修复**：明确研究问题是同内容组织收益、从独立反馈学习，还是两者；分别锁定对应 state 协议。保留现有 12 族作为机制单元测试，按 family/wave 报告；在未用过的结构组合/角色读出/时间尺度上做少量冻结迁移验收，而非继续同模板倍增。不能以题数或 tree node 数替代“学得的新科学内容被后继因果使用”的证明。

### V18-R14 — P2：持久化、恢复和摊销代码仍未遵守共享实验契约

**位置**：v18.4 `1149–1181,1220–1293`；v18.3 `560–636,730–755,856–870`；`AGENTS.md` Experiment storage。

恢复缓存只验证已有 DAG 的链，不核对 task/prompt/code/config/forest-at-start 的完整 run identity；修改源码后同路径 resume 可能混入旧 receipt。quota exception 时把没有最终 DAG 的 workspace 整目录删除，已经完成的探针/作者响应/失败成本不会作为中断状态恢复。这是潜在可重现风险；最终 manifest 的 paused_models 为空，本报告不声称这批运行一定发生了 quota 丢失。

摊销函数把所有 writeback C 在 `step_n>=10` 一次加上。它对应 v18.3 特定排列表尚能解释，v18.4 第一波是 40 题，不是第 10 题。break-even 取第一次短暂交叉，也不检查之后是否再次亏损。所有这些 ledger/线程/checkpoint 又直接用 `write_text` 和新 ThreadPoolExecutor，未走 AGENTS 指定的共享 checkpoint_storage、experiment_runtime、parallel_prediction 接口。

**最小修复**：复用现有共享实现保存每个唯一已付费状态、不可变响应和 resume_origin，run identity 绑定冻结配置/源状态；按实际 consolidation event timestamp 计成本，区分 first crossing 与 sustained break-even。现有 inspector 也只识别 v18.1 两个固定 ledger 文件名，跨版本 --verify-all 不能默默报告 0/0 链成功；本轮自带复算脚本补做了实际所有 DAG 检查。

## 6. v18.4 主表复算与模型分解

下面 C 是仓库原计价，含每臂 240 题；成功率分母只含 216 个 in-scope，OOS 列为另外 24 题。

| 模型 | 臂 | in-scope | OOS退出 | 假确认 | 原 C | API calls |
|---|---|---:|---:|---:|---:|---:|
| Luna | Hegel | 199/216 | 24/24 | 6 | 3738.70860 | 605 |
| Luna | Flat | 191/216 | 24/24 | 7 | 3842.73325 | 662 |
| Luna | Strong | 201/216 | 24/24 | 0 | 4680.69330 | 596 |
| Terra | Hegel | 214/216 | 24/24 | 0 | 3603.72690 | 455 |
| Terra | Flat | 203/216 | 24/24 | 4 | 3642.66035 | 488 |
| Terra | Strong | 211/216 | 24/24 | 0 | 4452.14440 | 442 |
| Sol | Hegel | 208/216 | 24/24 | 0 | 3549.15835 | 431 |
| Sol | Flat | 199/216 | 24/24 | 1 | 3707.09190 | 503 |
| Sol | Strong | 214/216 | 24/24 | 0 | 4433.01135 | 382 |

不能用总榜说三个模型都最大程度发挥了算法。Hegel 对 Flat 都有净提升（+8/+11/+9），但 Hegel 对 Strong 的通过数是 Luna −2、Terra +3、Sol −6，最后一个主要是基础设施机会不同。

### 6.1 各机制族的通过数（每格 18 题）

| 族 | Luna H/F/S | Terra H/F/S | Sol H/F/S |
|---|---:|---:|---:|
| F01 原始不规则记忆 | 18/18/17 | 18/18/18 | 18/18/18 |
| F02 绝热折叠 | 18/18/18 | 18/18/18 | 18/18/18 |
| F03 双时域反转 | 17/18/18 | 18/18/18 | 18/18/18 |
| F04 before/after | 15/18/18 | 18/18/18 | 18/18/18 |
| F05 合取 siphon | 17/16/18 | 18/18/18 | 18/17/18 |
| F06 窄 pilot 异质参数 | 14/15/15 | 16/15/16 | 17/17/16 |
| F07 阈值 avalanche | 17/15/14 | 18/15/17 | 18/16/18 |
| F08 bridge 检索 | 18/15/18 | 18/18/18 | 11/9/18 |
| F09 Hill inhibition | 18/18/17 | 18/18/18 | 18/18/18 |
| F10 barrier siphon | 13/12/18 | 18/15/18 | 18/18/18 |
| F11 siphon×memory | 17/15/15 | 18/17/18 | 18/15/18 |
| F12 sink×dual horizon | 17/13/15 | 18/15/16 | 18/17/18 |

F08 的 Sol 11/18 不应解释为“最强模型不懂跨领域结构”：Hegel 的七个 F08 失败都没进入候选执行。相反，Luna 的 F04/F10 差距是实质性的作者参数域/错误机制/确认设计问题，值得优先修复。

### 6.2 机会匹配后的诊断切片

这里只看**同一题双方都至少提交过一个候选**，用于区分调用机会与解题，不改变主分数，也不作为删失败后的新排行榜。

| 模型与比较 | 配对数 | 双方通过 | 仅 Hegel 通过 | 仅 baseline 通过 | 双方未过 |
|---|---:|---:|---:|---:|---:|
| Luna vs Flat | 213 | 182 | 17 | 8 | 6 |
| Luna vs Strong | 213 | 188 | 11 | 12 | 2 |
| Terra vs Flat | 213 | 203 | 10 | 0 | 0 |
| Terra vs Strong | 214 | 211 | 3 | 0 | 0 |
| Sol vs Flat | 198 | 197 | 1 | 0 | 0 |
| Sol vs Strong | 207 | 207 | 0 | 0 | 0 |

它支持一个更精确的假设：当前结构化指导在弱/中模型上可能减少找错机制或选错探针的负担；对 Sol，这套合成题在真正获得执行机会时大多已接近饱和。Luna 对 Strong 的优劣交错说明更强的 hand guidance 也可能把作者锁进错误参数域/历史模式，不能只看平均 token 节省。

### 6.3 全部 104 个域内失分的观测分类

分类规则优先记录 false confirmation，其次无候选的 provider error，再执行错误、确认拒绝和直接 abstain；“执行错误”是轨迹中出现过，不自动等于唯一终极原因。逐题完整条目见 `v18_failure_appendix.md`/JSON。

| 模型/臂 | 假确认 | provider 且无候选 | 有执行错误 | 只见确认拒绝 | 直接 abstain | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| Luna H | 6 | 2 | 6 | 2 | 1 | 17 |
| Luna F | 7 | 2 | 13 | 3 | 0 | 25 |
| Luna S | 0 | 2 | 9 | 4 | 0 | 15 |
| Terra H | 0 | 2 | 0 | 0 | 0 | 2 |
| Terra F | 4 | 2 | 2 | 4 | 1 | 13 |
| Terra S | 0 | 2 | 0 | 3 | 0 | 5 |
| Sol H | 0 | 8 | 0 | 0 | 0 | 8 |
| Sol F | 1 | 16 | 0 | 0 | 0 | 17 |
| Sol S | 0 | 2 | 0 | 0 | 0 | 2 |

### 6.4 有代表性的实际错误链

- **W1R3 F06 / Luna Hegel**：探针已买对、正确三参数式已拟合成功；predict 逐点 fit 到 y=0 →巨大误差→被错误当作模型反证→换 Hill/barrier→耗尽。这是最有确定性的可挽回损失。
- **W2R1 F10 / Luna Hegel**：方程正确但 `c_sqrt>=0`；其它参数补偿后三点确认 0.02355，四点 heldout 0.05702。只放开符号就几乎精确拟合。不需要加一棵新“哲学树”。
- **W2R1、W5R1 F04 / Luna Hegel**：有候选把 equilibrium 项写成 `alpha*x_now`，再加 before-after transient，忽略真正的当前输入平衡项 `k_eq*u_now/tau_scale`；二者在有限点上近似但不是同一长期规律。分别出现确认通过、heldout 0.054206/0.064898。
- **W1 F07 / Terra Flat/Strong**：源 smooth siphon 在 subcritical pilot 上几乎精确，confirm 进入超临界后 RMSE 0.6–0.7；部分作者未正确保留 avalanche 的 `*u_n` 或阈值结构。拒绝是诚实的，但不能说已有边界学习算法不存在；它没有被当前作者/执行路由充分采用。
- **W1 F11/F12 / Luna Flat/Strong**：试 simple memory ODE/dual exponential，漏掉 sink 与输入饱和的组合，再伴随 fit-state 错误。Hegel 此时已得到完整组合方程，表现更好；这恰好是组合知识有用的迹象，但要分离“知识本身”与“自动形成该知识”的证据。
- **W6R2 F04 / Luna Hegel**：第一轮直接 abstain，理由是召回的 Maxwell 等模式与证据冲突而额外轨迹不值得买；没有作者分析/模型，不能定性为 solver 失败。它是检索→研究行动采纳失败。

这些失败同时指向实现、表达域、检索采纳与确认设计四层，不能统一塞进“LLM 不够强”或“哲学框架失效”。

### 6.5 保存的 runtime-error 候选局部重放

为补足原 DAG 截断在前 240 字符的错误摘要，本轮对这 104 个失分 episode 内的 **60 份已保存 runtime-error 作者提交**做了只读本地重放。全部 60 份源码 SHA 与原记录一致，没有修改程序或训练数据：35 个 NameError、7 个 ValueError、5 个 RuntimeError、5 个 KeyError、2 个 SyntaxError、1 个 AttributeError、1 个 TypeError；另 4 个在本轮 15 秒诊断上限停止。

35 个 NameError 大多是 `_TRAIN_ROWS`、`_FIT_PARAMS`、`_PARAMS` 等从未创建的拟合状态；RuntimeError 多数由作者自己明确抛出“请调用 fit 并使用其返回参数”；KeyError 中 4 个是在 y 已剥离的预测阶段仍读取 y。这个重复模式进一步支持接口层修复优先于推翻机制。

完整本地 traceback、原记录摘要、源码路径/双 SHA 在 `v18_replayed_runtime_errors.json`；脚本为 `v18_replay_runtime_errors.py`。这是本轮环境的回放，不是恢复出原机器完整 stderr：本轮 Python/依赖环境可能不同，且本轮超时 15 秒而原 host 是 25 秒，因此四个 timeout 不能声称精确复现原超时条件，也不能据此给它们重新分类或改主榜。逐题附件已关联相应回放结果。

## 7. 有限改进顺序与已有能力复用

### 第一包：先恢复实际计算机会和候选程序合同

修 code_mode_host 兼容性检查、超时/配额恢复和付费历史保存；科学轮数与 provider attempt 分账。接通稳定 fit/predict 状态，保留上一轮代码，运行不看 y 的预测合同检查。复用 `code_agent.py` 的 evaluated-model retention、`code_provider_context.py`、`python_evidence_repair` catalogue 路由，以及共享 checkpoint/runtime。

验收只需本报告的实际 F06 错误、一个未定义参数错误、一次中断后恢复控制。它解决已发生损失，不应变成新造一套 solver。

### 第二包：统一执行路径并让权限/历史产生真实效果

去掉 traced 复制实现，ledger 挂到 genuine pipeline 的动作处。修 raw-prefix schema，修 edge coverage，live 使用授权 tier 和 persisted prior。保留 layer1 partial 的候选价值及后续探针，不因完整 warm 被拒绝就丢弃局部共性。

验收：本报告 HEC、raw-prefix、denied numeric permission、missing causal edge 四个反例全部改变实际动作/预测且 trace 与执行一致。

### 第三包：让已完成源 episode 真正形成后继内容

从已保存成功作者程序/观测提取结构和范围，接已有 `branch_reconciliation`、`world_structure_commitments`、`OutcomeDrivenForestConsolidator`。原 `theory_induction`/`theory_consolidation`、scientific generalization、paid evidence retention 都能查到；本轮不把这些已存在的研究循环重命名为“下一代新能力”。具体不兼容点是当前 M5 只有 scalar row+任意 Python 的 contract，而旧路线常需 typed Program、object schema、source fit 与 paid Observation；应写清 adapter 可承诺什么，保留不能可靠转换的可执行实例。

验收：metadata rename、作者程序改变、分支 erasure、真实边界读回四类最小因果对照。只有确认数据参与筛选时，报告它是训练反馈而非未见最终审计。

### 第四包：重新界定比较与外推

先在原 240 题上保留原结果，并给出同价敏感性/配对机会诊断，不删除旧失败。下一次有限 live 比较使用共同观测价格、相同已获得内容、不同组织，或另开清楚声明的 independent learners；加入匿名签名、真正时间缩放/角色映射、一个未见组合族，并冻结 source 形成过程。

对于“RSI”，目前能较稳地说：**已经有真实模型执行的外部记忆/结构指导科学发现循环；存在部分能力积累和在后继任务上复用的工程路径。当前 v18 的新 live 写回还不能证明自主产生改进算法，更不能由节点增长和代理成本回本推出开放式递归改进已经实现。** 这个结论不要求降低设计野心，只要求把真正的研究主体、来源与改进对象放回证据链里。

## 8. 固定提交源码与报告入口

以下链接均固定到本轮 SHA，行号见各发现的“位置”；证据 case 的完整 repo-relative 路径在 JSON/逐题附件。

- [S1 — live harness、prompt、traced solver 与 provider](https://github.com/erzhu419/Assumption-Agent/blob/922b95e194d17c9259f1993706440e0602a43db9/Hegel%20Machine/src/hegel_machine/cognition/m5_live_gpt_and_provenance.py)
- [S2 — v18.2 closure 与压力测试](https://github.com/erzhu419/Assumption-Agent/blob/922b95e194d17c9259f1993706440e0602a43db9/Hegel%20Machine/src/hegel_machine/cognition/m5_v18_2_sovereign_closure.py)
- [S3 — v18.3 三波写回与摊销](https://github.com/erzhu419/Assumption-Agent/blob/922b95e194d17c9259f1993706440e0602a43db9/Hegel%20Machine/src/hegel_machine/cognition/m5_v18_3_continual_and_complex_live.py)
- [S4 — v18.4 K45、生成器、写回与恢复](https://github.com/erzhu419/Assumption-Agent/blob/922b95e194d17c9259f1993706440e0602a43db9/Hegel%20Machine/src/hegel_machine/cognition/m5_v18_4_hegel_discovery_240.py)
- [S5 — M5 deterministic、source formation 与 raw-prefix](https://github.com/erzhu419/Assumption-Agent/blob/922b95e194d17c9259f1993706440e0602a43db9/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py)
- [S6 — forest binding、diversity 与 digest](https://github.com/erzhu419/Assumption-Agent/blob/922b95e194d17c9259f1993706440e0602a43db9/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py)
- [S7 — v18.4 原始总 receipts](https://github.com/erzhu419/Assumption-Agent/blob/922b95e194d17c9259f1993706440e0602a43db9/Hegel%20Machine/docs/evidence/v18_4_hegel_discovery_240_20261121/all_completed_episode_receipts.jsonl)
- [S8 — 计划与 22–25 工作报告目录](https://github.com/erzhu419/Assumption-Agent/tree/922b95e194d17c9259f1993706440e0602a43db9/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan)
- [S9 — historical capability catalogue](https://github.com/erzhu419/Assumption-Agent/blob/922b95e194d17c9259f1993706440e0602a43db9/Hegel%20Machine/src/hegel_machine/cognition/historical_capabilities.json)
- [S10 — 已有可执行重组与反事实边界研究](https://github.com/erzhu419/Assumption-Agent/blob/922b95e194d17c9259f1993706440e0602a43db9/Hegel%20Machine/src/hegel_machine/cognition/branch_reconciliation.py)

本报告不重复 M6 子审计负责的 Python 文件系统隔离控制；两分支共享的 live harness 仍把 repo PYTHONPATH 提供给作者 subprocess，该隔离边界应在总报告统一说明。任何本地控制展示的可访问性都不能被改写成“原模型已经使用隐藏真值”，原日志未发现此类利用证据。
