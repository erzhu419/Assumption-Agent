# `GPT_advice&roadmap.md` 向 Assumption Agent 的迁移评估

日期：2026-07-29

审计依据：

- [`GPT_advice&roadmap.md`](../markdown/GPT_advice&roadmap.md)
- [`red_queen_architecture_diagnosis_20260711.md`](red_queen_architecture_diagnosis_20260711.md)
- [`HypothesisProgram`](../assumption_agent/models.py)
- [`proposer`](../assumption_agent/proposer.py)
- [`typed operator grammar`](../assumption_agent/typed_operator_grammar.py)
- [本地参考资料 manifest](../reference/gpt_advice_roadmap_20260729/metadata/MANIFEST.md)

## 1. 结论

最后一条 GPT 回答抓住了当前架构最重要的缺口，方向约有八成正确：

> 不应把 13 条或 22 条“通用假设”直接写进 prompt；应把它们变成假设搜索空间的类型系统，并把
> `MetaAssumptionTemplate`、问题实例化的 `HypothesisClaim` 与可执行的 `TreatmentProgram`
> 分开。

但这套方案不能被理解成“增加 22 个候选或 22 个 gate”。对当前项目最合理的用法是：

1. 22 条形成一个有角色类型、适用条件和编译器的 catalog，而不是互斥且完备的真理本体；
2. 用 TRAIN-only、固定预算的 probe 形成少数 prediction-distinct claims；
3. 由 harness-owned compiler 把 claim 编译到现有 typed recipe 与 `HypothesisProgram`；
4. 把 RAW no-op 作为同一决策空间中的正式 action，而不是在看到 held-out harm 后增加阈值；
5. 只在全新 source/cohort 上进行一次 Agent / RAW / official-core HippoRAG 三臂离线测量。

该方向适合作为新 study，而且比继续手写 RRF、关键词、family switch 或 promotion gate 更接近
Assumption Agent 的核心命题。不过它不能保证补上最后缺口：ontology 只能改善机制候选的形成和
选择，真正的因果收益仍取决于 compiler 能否把被支持的 claim 忠实地变成有用 action。

## 2. 当前项目状态与真正瓶颈

当前全局状态应以最新 QuAC formal 结果为准：

- L5 已成立：A_hold 中 E1 相对 E0 晋升，且在预冻结、untouched 的 M_search 上保持正改善；
- E1 相对 HippoRAG 在 A_hold 三个 relation family 都为正；
- 现实域三-family双基线 primary 仍未成立。

最后一点有两个并列原因，而不只是 FOLLOW：

1. E1−RAW 的 family delta 为 `+13 / −4 / +2`，FOLLOW 为负；
2. E1−RAW aggregate exact tail 为 `374/2048 ≈ 0.1826 > 0.10`。

原主诊断中“失败点只有 FOLLOW”的文字解释不完整，本次审计已将其修正；`reality_primary=false`
这个布尔终态本身一直是正确的。当前唯一剩余目标是：在 fresh A_hold 中，同一冻结 Agent 对 RAW
与可审计的 official-core HippoRAG comparator 在 aggregate 和三个预注册 relation family 上都满足
原 primary。

现有证据已经把原因定位得比“假设不够多”更具体：

- 当前 `HypothesisProgram` 保存 statement、trigger、anti-trigger、action DAG、expected effect、
  verifier、fallback 和 lineage，本质上已经是一个很好的可执行 treatment 容器，但没有独立保存
  assumption family、竞争预测、counter-prediction、probe 或 claim→treatment 编译绑定。
- `HypothesisKind={task, policy, evaluator}` 描述作用位置，而不是世界结构的认识论类型。
- `family_slot_contract.target_failure_family` 指向任务失败 family，不等于 assumption family。
- HybridQA marginal-replacement 证明 RAW top-5 外存在大量有正因果 utility 的 typed action；
  oracle−RAW 在所有 block×family 单元都为正，但 additive marginal evaluator 不能稳定识别它们。
- HybridQA whole-set interaction 把 pooled、三个 held block 和两个 family 大幅改善，却选择了
  210 次 replacement，而 oracle 只选择 51 次；主要问题已变成过度行动、表示迁移与 no-op 决策，
  不是候选空间不存在。
- QuAC RJMC 证明 set-level treatment、A_hold promotion 和 untouched downstream retention 可以真实
  发生，但没有做“有/无 ontology”这一单变量随机消融，不能把成功完全因果归因为某个通用假设。

所以当前最需要的新能力是：

> 从 residual 中形成具有不同可观测预测的机制 claims，识别有益 action 的稀疏性和关系条件，并在
> 不确定时保留 RAW，而不是再增加一个事后 family gate。

## 3. 早期 13 条假设：逐项判断

| # | 原假设 | 对当前项目的判断 | 正确用法 |
|---|---|---|---|
| 1 | 对称 / 不变 / 等变 | 高工程价值、低新增效果价值。当前 set evaluator 已需要 permutation/ID invariance，但这主要防实现错误。 | 编译为 metamorphic tests 和 set/graph equivariant contract，不作为 held-out efficacy gate。 |
| 2 | 局部性 / Markov blanket | 中等价值。typed edge、turn/section neighborhood 与当前机制相容，但没有独立 locality 因果消融。 | 预注册 0/1/2-hop compiler 与 distance-conditioned TRAIN probe；不可看到失败 family 后改半径。 |
| 3 | 流形 / 低内禀维 | 过于宽泛，直接价值低。对离散 evidence-set 决策，“最小充分表示”比流形叙事更可检验。 | 改写为 conditional predictive sufficiency，而不是假设数据天然位于低维流形。 |
| 4 | 低秩 / 可分离 | 条件性、当前证据弱。可以压缩 set interaction，但没有奇异谱或 rank 稳定性证据。 | 只在 TRAIN 谱和 held-fold reconstruction 支持时使用固定秩；不得遍历 rank 到成功。 |
| 5 | 单调 / shape constraint | 只在 source 给出真实偏序时有价值。通用“更多证据更好”在 distractor 场景通常不成立。 | 对明确 coordinate 做 pairwise order probe；不要对整体 utility 强加单调。 |
| 6 | 次模 / 边际收益递减 | 对当前主要机制可能有害。已观察的证据互补性恰恰包含超加性 synergy。 | 作为可证伪候选而非默认先验；先做 unary/pair interaction probe。 |
| 7 | 守恒 / 平衡 | 当前 retrieval study 基本不适用。没有真实 conserved quantity 时容易产生“伪物理”叙事。 | 仅在 source schema 明示流量、预算或概率质量时启用。 |
| 8 | 稳定 / 收缩 / 耗散 | 中等价值，适合约束 evaluator 对微扰、fold 和组件删除的敏感度。 | 编译成连续 regularizer 或 bounded update；不要追加新的 held-out stability gate。 |
| 9 | 交换性 / 层级 Bayes | set permutation 有用；更强的 iid/exchangeability 声明通常不成立。 | 把置换不变性与统计交换性分开，后者必须有 source-level 证据。 |
| 10 | 最大熵 / 最小承诺 | 对当前过度行动问题有直接价值，但应解释为 calibrated no-op/abstention，而不是自动假设高斯 residual。 | RAW no-op 与其他 action 同池竞争；不确定性规则在 TRAIN 前固定。 |
| 11 | MDL / 压缩 / Occam | 适合候选排序和复杂度审计，不能直接生成有效 mechanism。 | 使用固定编码或程序长度作 tie-break/regularizer；避免任意自然语言长度。 |
| 12 | 信息瓶颈 / 最小充分状态 | 高价值。它能把表面文档特征商掉，只保留角色、关系、覆盖和风险。 | 检验给定 typed representation 后 raw lexical/item/family feature 是否仍带来 held-fold 增益。 |
| 13 | PAC-Bayes / data-dependent prior | 对 generalization audit 有理论价值，对 action formation 的直接价值低。 | 只有在先验、后验、损失和数据依赖条件都明确时报告 bound；不能把任意 support score 称为 posterior。 |

早期回答把“对称/交换、局部、单调”列为最高优先级，对 SC-OLH-KG 可能合理，但不符合当前
Assumption Agent 已定位的 over-action 瓶颈。当前优先级应改为：

1. 有益 action 稀疏；
2. 角色/关系的最小充分表示；
3. set-level 低阶交互；
4. 典型机制 + 污染鲁棒性；
5. 决策相关性与正式 no-op。

## 4. 后续 22 条：逐项角色、价值与边界

这 22 条不是同质对象。至少应增加
`role={world_claim, representation_prior, regularizer, governance_rule, decision_rule}`；
同一个 claim 也可能是多标签，而不是唯一叶类。

| # | 假设 | 角色 | 对当前最后缺口的增量价值 | 建议的 compiler / probe 与边界 |
|---|---|---|---|---|
| 1 | 简洁 / 可压缩 | 模型选择元原则 | 低 | 固定代码长度与 cross-fit utility；只能排序，不能成为新 promotion gate。 |
| 2 | 稀疏性 | world + decision | 高 | group sparsity、固定 action budget、stability selection；不能看 harm 后加 allowlist。 |
| 3 | 低秩 / 可分离 | representation prior | 低 | 固定秩 bilinear set factors；一次 TRAIN 谱/重构 probe，禁止 rank search。 |
| 4 | 最小充分表示 / 商空间 | representation prior | 高 | role/relationship quotient encoder；检验 raw feature conditional increment。 |
| 5 | 加性 / 低阶交互 | world claim | 中，已部分实现 | functional-ANOVA unary/pair probe；保留 synergy，不能硬设加性。 |
| 6 | 正交 / 非冗余 | search governance | 中 | 比较 prediction vectors 与 residualized incremental effect；余弦只作连续偏好，不能作真理 gate。 |
| 7 | 模块化 / 独立机制 | world + architecture | 中 | typed module interface 与 intervention-locality probe；不可演变为事后 family expert。 |
| 8 | 局部性 / Markov blanket | world claim | 中 | 固定 hop compiler 与 distance-conditioned action value；不根据失败 family 改 radius。 |
| 9 | 可组合 / 机制复用 | architecture prior | 中 | typed composition DAG 与 held-combination test；禁止穷举组合直到成功。 |
| 10 | 低频 / 平滑 | representation prior | 低或可能为负 | graph-Laplacian probe；错误平滑会抹去当前已知的 complementarity。 |
| 11 | 分段低频 / 稀疏变化 | world claim | 中但高风险 | input-derived latent regimes；不能把已观察的 TABLE/FOLLOW 直接变成 switch。 |
| 12 | 尺度分离 / 慢变量 | representation prior | 低 | structural/lexical two-stream 与 perturbation test；静态 retrieval 没有天然时间尺度。 |
| 13 | 稳定 / 收缩 / 耗散 | regularizer | 中 | bounded/Lipschitz update、fold consensus；作为训练约束而非又一 held-out gate。 |
| 14 | 对称 / 不变 / 等变 | implementation contract | 新增效果低、工程必要 | permutation、ID rename、paraphrase metamorphic tests；所有候选共享。 |
| 15 | 守恒 / 平衡 / 流 | world claim | 当前为负 | 只有真实 schema 提供 conserved quantity 才可启用。 |
| 16 | 拓扑持久性 | representation prior | 低至中 | path/connectivity persistence；不能从已评分成功 motif 反写规则。 |
| 17 | 单调 / 偏序 / 次模 | world claim | 低且条件性可能为负 | 只对 TRAIN probe 支持的具体 coordinate 施加；普遍次模性会排除 synergy。 |
| 18 | 典型规律 + 稀疏污染 | robust world claim | 高 | robust loss、influence/leave-small-set-out probe；绝不能删除正式 item/family。 |
| 19 | 最大熵 / 最小承诺 | uncertainty + decision | 中高 | calibrated no-op/abstention；必须是预冻结决策函数，不是看到 harm 后补阈值。 |
| 20 | 可证伪 / 主动区分 | governance | 对发现中等、对最终效用间接 | 固定 probe budget，按 prediction disagreement 选 probe；不得持续 probe 到成功。 |
| 21 | 证据三角测量 | governance | 因果增量低、证据价值高 | 独立 evidence ledger 和 leave-channel-out；不要无限追加“还缺一种验证”。 |
| 22 | 决策相关 / 边界充分 | decision rule | 高 | counterfactual action value 与 value-of-information；只保留会改变 policy/utility 的 claim。 |

最值得进入当前 gap-closing study 的竞争性 world claims 是 `#2/#5/#8/#18`；`#4/#14/#19/#20/#21/#22`
更适合作为所有候选共享的 representation、decision 和 governance contract。`#10/#12/#15` 与强形式
`#17` 不宜进入当前默认 retrieval portfolio。

第二段回答提出的 “Piecewise-Smooth Sufficient Modular Mechanism Prior” 很有启发性，但作为单一总先验
太强：它同时押注平滑、分段、充分性、模块化和稀疏污染，失败时无法识别是哪一个成分错误。更好的做法是
把它拆成可竞争 claims，再由固定 probe 形成一个稀疏、多标签组合。

## 5. 下载文献逐篇核对

本地 bundle 位于：

`reconstruction_v2/reference/gpt_advice_roadmap_20260729/`

原 Markdown 有 26 个 reference 条目、25 个唯一来源。现已保存 18 篇学术论文 PDF、1 份 tutorial
slides、7 份网页/产品说明快照，并 clone 4 个经作者或论文确认的论文代码仓库以及 OpenAI Codex
仓库。论文 25 的匿名代码镜像只取得不完整快照，按用户指示不再追取；不影响论文内容分析。精确文件、
哈希和 commit 见 bundle 内 `metadata/MANIFEST.md`。

| ID | 文献 | 实际能支持的主张 | 不能据此推出的结论 |
|---|---|---|---|
| 01 | Bronstein et al., *Geometric Deep Learning* | 对称性、群作用、等变架构是统一的归纳偏置语言。 | 不能证明任意 retrieval evaluator 使用对称性都会提高效用。 |
| 02 | Allen et al., *Learning Markov State Abstractions* | 在 RL 中给出保持 Markov 性的抽象状态条件。 | 不能把所有局部任务都视为 Markov，也不直接支持通用 Markov blanket。 |
| 04 | Zhang, *CROSS* | 在特定采样与可辨识条件下进行低秩 tensor completion。 | 不能证明现实 evidence utility 天然低秩。 |
| 05 | Li et al., *Bayesian Optimization with Monotonicity Information* | 当 monotonicity 是正确 domain knowledge 时，可改善特定 BO。 | 不能把 monotonicity 当作 universal default。 |
| 06 | Krause & Guestrin tutorial | 解释次模性、贪心算法和机器学习用例。 | 不是当前 set utility 具有 diminishing returns 的证据；当前 synergy 甚至可能反对它。 |
| 07 | Djeumou et al., physics-informed architectures | 对动力系统，把已知物理约束嵌入架构可改善可行性与样本效率。 | 不支持没有物理方程的任务硬套守恒或耗散。 |
| 08 | Davydov & Bullo, contractivity perspectives | 收缩理论可统一分析动态系统的收敛、鲁棒性和模块性。 | 这是 perspective，不证明现实系统“通常收缩”。 |
| 12 | Alquier, PAC-Bayes introduction | 系统给出 PAC-Bayes bounds、假设条件和 data-dependent prior 的处理。 | 不能把任意 source/target KL 或经验 support score直接解释成泛化保证。 |
| 13 | Hutter, generalized universal priors | 支持算法信息论中的 universal prior 和特定收敛结果。 | Solomonoff prior 不可直接变成可计算的自然语言假设排序；编码/参考机不可忽略。 |
| 14 | Lippl & Stachenfeld, compositional kernel theory | 组合结构只有在训练覆盖等条件成立时才产生组合泛化。 | “结构是组合的”本身不足以保证 transfer；这是本 roadmap 最重要的负面护栏之一。 |
| 15 | Müller et al., independent causal mechanisms | 在多环境分布偏移下利用 ICM 形成更鲁棒模型。 | 从 causal modules 到 task/policy/evaluator modules 仍是架构类比，不是已证等价。 |
| 16 | Luketina et al., compositional interfaces | 模块接口有助于跨 observation/action 组合的迁移。 | 不证明自然语言 hypothesis ontology 自动形成可复用机制。 |
| 17 | Xu et al., change-point review | 支持 piecewise regime 与 change-point 工具的存在和适用范围。 | 当前 family 异质性不是 change-point 的直接证据。 |
| 18 | Hong & Wang, scale separation in μP | 讨论 maximal update parameterization 中优化尺度与超参数迁移。 | 与“现实系统由 slow variables 支配、fast variables 是 residual”并非同一主张；这是明显的引用错配。 |
| 19 | Trigka & Dritsas, symmetry-aware learning review | 汇总 symmetry-aware architecture 和部署权衡。 | 综述不能替代当前任务的效用消融；且本地版本是早期未编辑稿。 |
| 20 | Meng et al., physics-informed ML survey | 广泛总结 physics-informed learning 的方法和边界。 | 不能为非物理 evidence retrieval 提供普遍守恒先验。 |
| 21 | Su et al., topological data/deep learning review | 支持 topology/persistence 作为结构表征工具。 | “图连通稳定”不等同于严格 persistent homology，也不保证 action utility。 |
| 22 | Agarwal & Yamada, *Hypothesis-Driven Reasoning* | 在特定多模态 episode task 中，显式语义 hypothesis memory 和 generate/verify 有效。 | 任务范围较窄，不验证 22-family ontology 或现实检索中的 autonomous scientist。 |
| 25 | Xiong, LLM hypothesis generation/updating | 在 number-game 中展示 Occam 偏好、生成与评价差距及外推局限。 | 不能直接推断通用科学发现能力；但强烈支持把 Generator 与 Falsifier 分开。 |

另外四类来源只能作为背景材料，不能作为论文级关键证据：

- manifold hypothesis 与 MDL 使用 Wikipedia；
- exchangeability 使用技术博客；
- maximum entropy 使用 Jaynes 历史性摘录；
- 两篇 OpenAI 文章说明公开的 Codex agent loop 和产品训练/使用方式，不是完整内部训练机制披露。

若正式写论文，应为 manifold/MDL/exchangeability/maximum-entropy 补一手或权威综述引用；本次评估没有
把这些网页来源冒充成论文。

## 6. 对最后一条 GPT 回答的精确审计

### 6.1 正确且应保留

1. 22 条不应平铺进 prompt。
2. 当前 `HypothesisProgram` 更接近 treatment，而不是完整的机制 claim。
3. `task/policy/evaluator` 是作用位置，不是认识论 family。
4. 每个 family 需要 support signature、counter-signature 和 discriminating probe。
5. 机制应先产生可观测预测，随后才允许编译 action。
6. 正交性在这里应主要表示 residual explanation 和预测的非冗余，而不是基函数 Gram penalty。
7. 现场“自我改进”主要是上下文内 generate–test–revise，不是模型权重在线更新。
8. Generator、Falsifier、Evaluator 和 Promotion authority 的信息与权限必须分离。
9. ontology 新实验必须独立，不能追溯性改写 QuAC 或 HybridQA 的既有证据。

### 6.2 必须修正

1. **计数矛盾**：六个根类实际为 `4+5+4+4+2+3=22`；前五类有 19 项，治理类有 3 项。
   文中“前18/后4”与“治理三条”不能同时成立。
2. **不是严格树**：稀疏、局部、模块、鲁棒性和决策相关性常同时成立，应允许多标签和 typed role。
3. **不要叫 posterior**：没有显式 prior、likelihood 和校准时，字段应叫 `support_score`，而不是
   `posterior_weight` 或 “assumption-family posterior”。
4. **现实域没有 family selection truth**：family selection accuracy 只能在机制已知的 synthetic/injected
   suite 上定义；现实 study 应测 prediction distinctness、probe calibration、selection regret 和 held-out utility。
5. **information gain 不是免费量**：`I(H;Y_q|D)` 需要真实或估计的 outcome model；当前离线系统只能使用
   预注册的 disagreement/expected-elimination proxy。
6. **强制四根多样性可能制造噪声**：应冻结可适用 family 集，再按 prediction distinctness 和固定 probe
   选 K 个，而不是机械地每个 residual 取四个根类。
7. **QuAC 因果表述过强**：结果与关系型 set mechanism 一致并证明 treatment retained effect，但没有
   “有/无 ontology”随机消融。
8. **四个角色不等于四个 LLM**：可以由同一模型在不同 sealed views 下执行，只要信息、写权限和最终
   promotion authority 被 harness 隔离。
9. **缺少 CompilationReceipt**：claim 与 action 分开后，还必须证明 action 忠实实现了 claim；否则只是
   把语义说明和 treatment 放在两个文件里。
10. **“四个不同根类”的示例自相矛盾**：示例中的 relational 与 local 都属于“分解与机制复用”根类，
    并没有形成四个不同根。
11. **POPPER 被错误归引到 HDR**：HDR 论文不包含 POPPER、顺序证伪或错误率控制；这些主张需要独立
    引用，不能由该 AAAI 论文背书。
12. **#17 不是一个数学 family**：monotonicity、convexity/concavity、submodularity 与 stochastic
    dominance 是不同性质，必须有不同的变量类型、probe 和 compiler。

### 6.3 数学表述需收紧

roadmap 的公式适合作为设计草图，不能原样进入论文或实现合同：

- 对称性除了 `A(gx)=Q_gA(x)`、`N(gx)=P_gN(x)`，还需
  `Q_g^TΛQ_g=Λ`、`P_g^TBP_g=B`、`P_g^Tω=ω` 等兼容条件；任意 rename 也不一定是语义保持的 group action。
- temporal Markov、spatial locality、Markov blanket 与 covariance decay 是四个不同假设，不能互相推出。
- 低内禀维不推出 decision sufficiency 或 graph low-frequency；应分别测表示维数和条件充分性。
- `B=UU^T` 同时强制 PSD，且 U 有旋转不可辨识性；自由度不是简单 `Kr`。
- 单凭 `B⪰0` 不能推出 coordinate monotonicity；即使 `B` entrywise nonnegative、`N,ω≥0`，
  若 A 随 N 改变，整体目标仍未必单调。
- 对二次可微连续函数，仅令 `i≠j` 的 cross partial 非正通常是 lattice submodularity，不足以得到
  DR-submodularity；DR 还要求 coordinatewise diminishing returns。
- MaxEnt 依赖 base measure 和约束；Gaussian 需要固定均值/方差，不能把未知 residual 自动写成
  Gaussian 或 sub-Gaussian。
- 标准 two-part MDL 是 `L(M)+L(D|M)`；任意
  `−loglik + λ·CodeLength` 只能称 regularized objective，除非给出合法 code、单位和 λ 的解释。
- Information bottleneck 应明确为随机变量和编码器；对连续 deterministic encoder，
  `I(X;Z)` 可能为无穷，`dim(Z)` 或 `||ψ||_0` 也不等于 mutual information。
- PAC-Bayes 必须写完整 empirical-risk、sample size、confidence 与 `KL(Q||P)` 项；
  source prior 还需和 bound sample 独立，或使用专门的 data-dependent/meta-PAC-Bayes theorem。
- `I(H;Y_q|D)` 只有在 hypothesis weights 与 outcome model 都定义后才是信息增益；否则使用预注册
  disagreement proxy。
- 对离散 argmax，`∂V*/∂f(x)` 常不存在；decision relevance 宜用 counterfactual regret、margin
  或 value-of-information。

## 7. 最小正确架构迁移

不要删除或重写现有 `HypothesisProgram`。在它前面增加不可变对象：

```text
OntologyVersion
  -> MetaAssumptionTemplate
  -> HypothesisClaim
  -> ProbeReceipt
  -> CompilationReceipt
  -> existing HypothesisProgram / TreatmentProgram
  -> paired runtime + offline score + promotion
```

建议的最小字段为：

```text
MetaAssumptionTemplate
  template_id
  parent_ids
  role
  admissible_variables
  support_signature
  counter_signature
  probe_schema
  prediction_schema
  compiler_id
  not_applicable_conditions
  invariances

HypothesisClaim
  claim_id
  template_ids
  scope
  bound_variables
  observable_predictions
  counter_predictions
  competing_claim_ids
  description_length

ProbeReceipt
  claim_hash
  train_split_hash
  fixed_probe_budget
  observations_hash
  support_score
  counter_score
  falsified

CompilationReceipt
  ontology_hash
  template_hashes
  claim_hash
  probe_receipt_hash
  compiler_hash
  recipe_ids
  status_independent_treatment_hash
```

必须把四个名称空间分开：

- `task_family`：数据本身的 relation/error family；
- `assumption_family`：机制 claim 的类型，可多标签；
- `action_family`：compiler 产生的 typed recipe；
- `evaluation_stratum`：只用于报告与 primary，不进入 formal policy。

现有 typed operator registry、closed recipe compiler、`HypothesisProgram.validate()`、archive、
late-label barrier 和 paired offline evaluation 都可以直接复用。LLM 只应选择 template/recipe ID 和
绑定变量；可执行语义及 compilation receipt 由 harness 拥有。

这些对象应作为版本化、内容寻址的 sidecars，而不是给现有 `HypothesisProgram` 直接加字段：

- `HypothesisProgram.payload_hash` 对完整 dataclass 哈希，schema 增字段会改变全部旧 program identity；
- status/lineage 的变化也会改变 payload，所以 claim binding 应指向 status-independent treatment hash；
- archive、dedup、typed executable identity 和旧 parser 都依赖当前 v1 schema；
- 当前 `statement` 在部分 compiler 中会进入实际 skill description，并非纯 metadata。

最小集成不必先改 `EvolutionKernel`：上游 meta pipeline 可以独立完成 family selection、probe 和
compilation，再把最终 frozen programs 作为预编译 candidates 注入现有 validation/evaluation 链。

## 8. 推荐的新 study

建议名称：

`UAO_P1_SPARSE_RELATIONAL_DECISION_COMPILER_V1`

它不是完整 ontology 论文，而是直接针对当前唯一缺口的最小 study。

### 8.1 事前科学假设

该假设可由 QuAC FOLLOW 结果出现之前的证据独立形成：

> 现实检索中的有益 typed action 是稀疏的，并依赖低阶 set context；可迁移表示应以证据角色和关系
> 而非 source family 为单位，在不确定时应让 RAW no-op 与其他 action 共同竞争。

### 8.2 一次性协议

1. 选择完全未消费、具有三个原生 relation family、可离线构造 RAW 与 official-core HippoRAG 的现实 source。
2. 在打开效果 source/label 前冻结 ontology version、compiler、四个竞争 world claims：
   sparse action、pair/set interaction、local modular relation、contamination-robust selection。
3. 把 minimal sufficient representation、set symmetry、minimal commitment、falsifiability、
   evidence ledger 和 decision relevance 固定为所有候选共享合同，不参与结果后选择。
4. 只打开 TRAIN formation。每个 claim 运行一次固定预算 probe；用预先固定的
   `support − counter − MDL − redundancy` 分数选择恰好 K 个 prediction-distinct claims。
   没有“通过才继续”的新增效果 gate，也不重复 probe。
5. harness 把 claims 编译成同一个跨-family policy：
   source-native typed candidate expansion、unary+pair set utility、固定 robust loss，
   RAW no-op 在同一 argmax 中竞争。
6. 只做一次 TRAIN multi-fold fit；随后冻结候选、参数、编译收据、metric 和 A_hold。
7. fresh A_hold 最大并行执行 `Agent / RAW / official HippoRAG = 3×n`；先密封 action，后打开 label，
   统一离线评分。
8. primary 沿用唯一剩余目标：Agent 对两条 baseline 的 aggregate 均为正且 exact one-sided
   `p≤0.1`，三个预注册 family 对两条 baseline 都严格为正。
9. 不再重复 M_search：L5 已经成立，这轮只补 reality primary。
10. 无论结果正负都终止；不改 K、template、probe、loss、family、threshold、source 或 gate。

这个设计不是针对 FOLLOW=`−4` 增加 switch。它针对 HybridQA 在看到 QuAC 结果前就已暴露的
“typed candidate 有上界、selector 过度行动”问题，因此可以形成合规的独立研究边界。

### 8.3 与 GPT 所提五系统实验的关系

`free proposer / flat-22 / hierarchical ontology / random selector / oracle selector` 五系统比较很有价值，
但不应塞进本次最小 gap-closing study：

- oracle family 只在机制已知 synthetic suite 中有真值；
- 五臂会把“补一个现实 primary”扩成另一篇 ontology 方法论文；
- 现实数据上应比较 treatment utility/regret，而非不存在真值的 family accuracy。
- 如果执行五臂，必须匹配 token、probe、compiler 和运行预算，并把 ontology selection 的贡献与
  人工 compiler 的贡献分开。

如果后续单独写 ontology 论文，可在 source-free、mechanism-known synthetic suite 上做五系统实验；
其结果不得成为反复尝试现实 A_hold 的许可 gate。

## 9. 能否解决当前问题

分三层回答：

| 问题 | 判断 |
|---|---|
| 能否解决现有 proposer 假设类型集中在 trigger/action 层？ | 很可能。三层对象和 typed compiler 直接补上 mechanism/representation 层。 |
| 能否解决 whole-set evaluator 的 over-action？ | 有合理机制路径。稀疏 action、robust selection、decision relevance 与正式 no-op 都直接对应 210 vs 51 的差距。 |
| 能否保证现实三-family双基线 primary 通过？ | 不能保证。RAW 与 HippoRAG 是不同强项的 baseline，ontology 只提高成功概率，不构成效果证明。 |

因此本方案适合继续，但论文中应预先写成“检验 ontology-guided claim selection 是否改善 treatment
形成”，不能写成“使用 universal assumptions 必然达到 oracle”。

## 10. 如果不补新 study，当前结果够不够论文

够形成一篇边界清楚的架构/审计论文，但不够支撑最强的通用性能主张。

可以支持：

- typed、falsifiable、executable、auditable 的 assumption program 架构；
- frozen formation、late-label、paired offline evaluation、promotion authority 与证据 archive；
- QuAC 中真实的 A_hold evaluator promotion 和 untouched M_search retained improvement；
- 多个合法 non-promotion、implementation-invalid 与 architecture-stop 结果；
- typed candidate space 有真实 oracle utility，而 marginal 与简单 linear set evaluator 存在可定位边界。

其中最强、仍可守住的单一实证句应精确写成：

> 在一个 derived QuAC conversational evidence-retrieval study 中，一个 human-specified、
> prospectively frozen、typed/auditable relational set-level evaluator 在独立 A_hold 相对 E0 晋升
> （`+51`，exact `29467/33554432`），并在不再拟合或修改后于 untouched M_search 保留 E1−E0
> 正收益（`+29`，exact `980/131072`）。

不能支持：

- Agent 已自动发现 universal assumptions；
- 对现实域 RAW 和 HippoRAG 的三-family稳定普遍优越性；
- full official benchmark/SOTA 或 compute-matched superiority；
- ontology、关系假设或递归本身是 QuAC 收益的唯一原因；
- 模型进行了在线权重自我更新。

还必须披露：A_hold E1−RAW 虽为 `+11`，exact tail 为 `374/2048` 且 FOLLOW 为 `−4`。项目中的
HippoRAG 是同 corpus、candidate-restricted 的 official-core implementation，并非 official QuAC
leaderboard baseline；不能写成 full-corpus、等算力或 SOTA 对照。

最诚实的论文结论是：

> Assumption Agent 已证明可审计的 hypothesis-to-treatment 与 evaluator promotion/retention 链能够工作；
> 但稳定的现实双基线优势仍未建立，自动选择机制假设 family 仍是下一项独立研究问题。

## 11. 最终决策

建议继续这一方向，但采用第 8 节的最小 fresh study，而不是把 22 条全部加入 prompt，也不是再围绕
QuAC/HybridQA 的失败 family 补规则。实施前只需完成 ontology/claim/probe/compilation receipt 的
source-free 代码与测试冻结；之后一次 TRAIN formation、一次 fresh A_hold 三臂测量、统一离线评分并
硬停止。
