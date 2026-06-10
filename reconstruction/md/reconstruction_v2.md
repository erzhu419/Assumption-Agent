# 我：
https://github.com/erzhu419/Assumption-Agent
这是我自己的“递归式提出假设并自我论证”的agent的项目，下一步AI认为我的项目存在以下问题，在recursive_agent_gap_world_model_role_20260610.md 里，Claude.md里是我原本的构思过程，reconstruction.md 是我重构过一次之后的构思。
我的问题是，我其实有点困惑我现在的假设是以什么形式表示的，以及如果新增一个假设，又是以什么形式在原有架构上扩展的。我记得刚开始的版本是graph，后面加入了范畴论，我就不太清楚了。
关于世界模型，我在想Discrete World Models via Regularization这篇论文或者其他离散化状态表示的世界模型也许可以有帮助，因为关于选取哪个假设的问题，或者说从语义问题而提取的决策问题，本质是个离散的博弈树（世界）模型，而这刚好又和我们的把assumption表示成graph的理念一样：每个假设就是个新的分支或者是一条边，表示两个范畴之间是有联系的，每个回合我们尝试增加一条边或者一个新的节点，以表示一个新的假设（两个范畴直接有关联）或者表示发现了一个全新的范畴。然后每回合环境尝试去掉一条边或者节点以表示一些干扰，去掉一条边或节点后性能下降的最多的那条边或节点就表示最重要的相关性（可以认为是因果性甚至，甚至你可以查查因果世界模型的相关部分），并且已经建立的这个世界模型要和之前LLM吐出过的所有文本信息能印证（逻辑一致，就像哲学的发展一样，所有新哲学概念需要能兼容之前所有的哲学概念）。
我昨天的想法是，需要继续解构这个假设过程，例如勒夏特烈原理，不能像范畴论里一样只用一个节点来表示，然后楞次定律也用一个节点表示，两个节点之间一条边就表示两者同态/同构。而是需要用graph或者其他结构，表示出勒夏特烈原理的整个过程，可能包括n个反应物，m个生成物，并且表示出随着反应进行，n随时间的导数/偏导数会随着m（随时间的导数）的增加而减小，同理在楞次定律那边一样用这套结构表示出这个过程，当然细节可能会变化，然后我们度量的是2个过程之间的相似性（我其实也不知道如何度量）。然后假设即包含n随时间的导数会因为m增大而减小，也可能反过来，即n随时间的导数会因为m增大而增大；也包含两个过程之间是否是范畴论里的同构/同态这种大问题，这样表示这样才能精细化的表示一个假设。
两个认为相似的过程之间的相似性，信息几何是否能解决这个问题？就像刚刚说的如果勒夏特烈原理和楞次定律的反应过程都被一个graph或者一个更精细的方程或者NN或者矩阵表示，那么我们认为这两个graph/NN/matrix之间在范畴论里是同态/同构的时候，又应该如何只添加一条边就表示出这种假设？或者去掉一条边就认为这种假设不成立？
结果就是，agent每次负责加上一条边以表示构造一个假设，然后环境负责去掉这条边以检测这个假设的adaptation适应度。
思路有点乱，要解决的问题依然很多，帮我仔细整理，探索，规划下后面的路

# GPT:
根据 2026-06-10 的 gap 文档和你当前 repo 的 `assumption_os` 状态，我认为你现在不是“从 graph 改成了范畴论”，而是已经自然演化成了一个**分层假设操作系统**：

> **Graph 是运行时记忆和生命周期容器；范畴论是其中一类 formal alignment 插件；信息几何是形式化之后的度量工具；世界模型是对 graph 操作后果的廉价模拟器。**

你现在的困惑，核心来自把五种东西压成了一个词“假设”：
**自然语言猜想、图上的节点/边、过程模型、同构/同态证明、世界模型里的动作。**
它们应该同时存在，但不能混成同一个对象。

---

## 1. 你现在的“假设”应该以什么形式表示？

最稳的定义不是“假设 = graph edge”，也不是“假设 = 范畴论态射”，而是：

```text
Assumption
= 在上下文 C 中，若采用解释/关系/策略/模型/行动 H，
  则未来观测 O 或任务表现 M 应该发生某种可预期变化；
  若没有发生，则产生 residual，并触发修正、降权或新假设。
```

你之前的 reconstruction 已经把这件事说得很清楚：卖布朗尼、东汉王室遗传病、E=mc²、RL 奖励函数和格林函数同构、最小原型替换核心模块、v16 case+reflect scaffold，都可以统一成“在某个上下文中押注某个可失败结构”的假设。

所以一个假设在工程里至少应该有四层表示：

```text
H = Manifest + Graph Projection + Optional Formal Payload + Verifier Contract
```

其中：

**Manifest** 是最低层，也是所有假设必须有的形式。它回答：这条假设说了什么，适用于什么上下文，预期改善什么，可能伤害什么，怎么验证，失败后怎么归因。你之前文档里已经建议把每次 LLM 调用、检索、wisdom 选择、audit、judge 都记录成 assumption manifest；这是对的，应该成为系统不可绕过的底层格式。

**Graph Projection** 是假设在 Assumption Graph 里的投影。它可以表现为节点、边，或者更推荐的“关系节点”。你 reconstruction 里给出的 `AssumptionNode` 字段已经很接近最终形态：`type`、`claim`、`formal_form`、`context_conditions`、`predicted_effects`、`risk_predictions`、`verifier`、`evidence`、`residuals`、`confidence`、`metaproductivity`、`status`。边也不该只是普通 triple，而应该包括 `supports`、`contradicts`、`specializes`、`generalizes`、`is_analogy_of`、`is_formal_isomorphism_of`、`generated_from_residual`、`failed_because`、`replaces`、`depends_on` 等生命周期关系。

**Optional Formal Payload** 是可选的形式化对象。它可以是程序、方程、Markov kernel、SCM、ODE、process graph、NN、矩阵、prompt policy、test suite、proof sketch。范畴论和信息几何只作用在这层，不应该强迫所有假设一开始就进入这层。你之前的文档也已经把范畴论/信息几何降级成 `Formal Alignment Layer`，只负责发现已形式化假设之间的同构、合并重复家族、支持跨域迁移，而不是统一万物。

**Verifier Contract** 是假设的“证伪合约”。它规定这条假设怎么被 cheap check、world model、cross-judge、fresh split、placebo、objective benchmark 或 human review 验证。你文档里已经明确不要只用 same-family A/B gate，而要做分层 verifier stack。

因此最终你应该把“假设”理解成一个**生命周期对象**，不是单一数学对象。

---

## 2. 新增一个假设时，怎么在原有架构上扩展？

新增假设不应该直接改主图。正确流程应该是：

```text
residual / opportunity
  -> candidate manifest
  -> graph overlay diff
  -> optional formal payload
  -> predicted effects / risks
  -> world-model screening
  -> live ablation / verifier stack
  -> accept: merge into graph
  -> reject: keep as failed evidence / residual
```

也就是说，新假设首先进入一个 **candidate overlay**，而不是直接写入 committed Assumption Graph。

最小新增格式可以是：

```json
{
  "id": "H_20260610_001",
  "claim": "Le Chatelier principle and Lenz's law share an opposition-to-perturbation process schema.",
  "type": "alignment_hypothesis",
  "context_conditions": [
    "systems with identifiable perturbation and compensatory response",
    "local response can be compared by sign, potential reduction, or deviation minimization"
  ],
  "graph_ops": [
    {
      "op": "add_relation_hypothesis",
      "source": "Process_LeChatelier",
      "target": "Process_Lenz",
      "relation": "shares_process_schema"
    }
  ],
  "formal_refs": ["ProcessModel_LeChatelier_v1", "ProcessModel_Lenz_v1"],
  "predicted_effects": [
    "helps transfer negative-feedback reasoning across chemistry and electromagnetism examples",
    "improves analogy explanation tasks"
  ],
  "risk_predictions": [
    "may overstate isomorphism where thermodynamic equilibrium and electromagnetic induction differ",
    "may produce superficial analogy without mechanistic mapping"
  ],
  "verifier": {
    "cheap": "check mapping preserves perturbation-response sign relation",
    "world_model": "predict downstream analogy-task improvement and regression risk",
    "live": "fresh heldout analogy/explanation tasks with placebo relation controls"
  },
  "status": "candidate"
}
```

这里最关键的一点是：**新增“两个过程相似”这种假设时，不要只加一条普通边。**
应该加一个“关系假设节点”，也就是 reified edge：

```text
Process_LeChatelier
   ── participates_in ──>
AlignmentHypothesis_LeChatelier_Lenz
   <── participates_in ──
Process_Lenz
```

这条 `AlignmentHypothesis` 本身有 mapping、证据、反例、验证器、置信度、失败记录。高层图上可以显示成一条边，但工程底层必须把它当成一个可被验证、可被部分推翻、可被细化的对象。

这样你就不会被“一条边到底怎么表示这么复杂的假设”卡住了。答案是：

> **一条边只是索引，不是证明。真正的假设内容在 relation node / formal payload / verifier contract 里。**

就像“诺特定理”可以在知识图谱上是一个节点，但它背后的证明、适用条件、物理解释、数学形式不是那个节点本身。

---

## 3. 你的当前系统其实已经在朝这个方向走

你的 gap 文档显示，当前实现已经不是纯概念，而是有一个递归闭环：

```text
failure / residual
  -> generate multiple hypotheses
  -> novelty / orthogonal / morphism classification
  -> preflight
  -> world-model or pre-live budget screen
  -> live ablation / judge
  -> acceptance gate
  -> recursive resume
  -> gated apply / reject
  -> next-generation descendants
```

文档里还给出保守估计：递归自治假设验证原型约 75–80%，一般 long-running hypothesis OS 约 50–60%。缺的不是“完全没有架构”，而是 generator、world model、daemon、benchmark、observability、formal/morphism boundary 还不够强。

repo 的 `assumption_os/README.md` 也显示，代码层已经有 `schema.py`、`graph_memory.py`、metaproductivity selector、residual handling、candidate evaluation、candidate acceptance、recursive runner、evolution context、assumption bench、manifest logger、world model、trajectory search、verifier stack、recursive daemon 等模块。这说明你现在要做的不是推翻，而是**把语义统一、schema 冻结、实验闭环做硬**。([GitHub][1])

---

## 4. Graph、范畴论、信息几何、世界模型分别应该站在哪里？

你可以把系统想成四层：

```text
L0: Assumption Manifest
    每个行动背后的可失败押注。

L1: Assumption Graph
    记录假设、证据、残差、验证器、继承、冲突、替代、适用条件。

L2: Formal / Process Payload
    对某些假设给出过程图、方程、Markov kernel、SCM、程序、NN、矩阵等形式化内容。

L3: World Model
    模拟在当前 graph/state 下执行某个假设操作后的后果。
```

Graph 是**文件系统/记忆系统**。
范畴论是**结构保持关系的类型检查器**。
信息几何是**形式化对象之间的距离度量器**。
世界模型是**搜索控制器/模拟器/廉价验证器**。

你原来觉得“假设 = graph edge”是对的，但只对 L1 层的简化情况成立。后来加入范畴论之后混乱，是因为你让范畴论去承担 L0–L3 全部职责。它承担不了。它最多在 L2/L3 的某些已形式化对象上工作。

---

## 5. 勒夏特列原理和楞次定律：不要做“节点相似”，要做“过程相似”

你昨天的新想法是对的：不能把勒夏特列原理和楞次定律各自压成一个节点，然后用一条边说“它们同构”。那太粗。

你应该先把每个原理表示成一个 **ProcessModel**：

```json
{
  "id": "Process_LeChatelier_v1",
  "domain": "chemical_thermodynamic_equilibrium",
  "state_variables": ["concentration_reactants", "concentration_products", "temperature", "pressure"],
  "control_parameters": ["external_concentration_change", "external_temperature_change", "external_pressure_change"],
  "perturbation": "delta_external_condition",
  "response": "equilibrium_shift",
  "objective_or_potential": "reduce free-energy / restore equilibrium constraint",
  "dynamics_or_transition": "optional ODE / Markov kernel / symbolic rule",
  "invariants": [
    "response partially counteracts imposed perturbation",
    "system moves toward constrained equilibrium"
  ],
  "failure_cases": [
    "far-from-equilibrium kinetics",
    "irreversible reactions",
    "ambiguous reaction coordinate"
  ]
}
```

```json
{
  "id": "Process_Lenz_v1",
  "domain": "electromagnetic_induction",
  "state_variables": ["magnetic_flux", "induced_emf", "induced_current", "induced_magnetic_field"],
  "control_parameters": ["external_flux_change"],
  "perturbation": "dPhi_dt",
  "response": "induced_current_creates_opposing_flux",
  "objective_or_potential": "opposes change in magnetic flux",
  "dynamics_or_transition": "Faraday-Lenz relation / circuit dynamics",
  "invariants": [
    "induced response has sign opposing flux change"
  ],
  "failure_cases": [
    "open circuit with no current path",
    "nonlinear materials",
    "radiative or high-frequency regimes"
  ]
}
```

然后新增一个 alignment hypothesis：

```json
{
  "id": "Align_LeChatelier_Lenz_v1",
  "type": "process_alignment_hypothesis",
  "source": "Process_LeChatelier_v1",
  "target": "Process_Lenz_v1",
  "mapping": {
    "external perturbation": "external flux change",
    "compensatory shift": "induced current / opposing magnetic field",
    "opposition relation": "negative response sign",
    "stability intuition": "local negative feedback"
  },
  "preserved_structure": [
    "perturbation-response-opposition schema",
    "local counteraction relation"
  ],
  "not_preserved": [
    "thermodynamic equilibrium formalism",
    "electromagnetic induction equations",
    "state spaces and conservation laws"
  ],
  "tests": [
    "check sign of response under positive/negative perturbation",
    "check whether response reduces deviation under local linearization",
    "compare trajectory-level response curves after normalization"
  ]
}
```

这样，“勒夏特列 ↔ 楞次”的边就不是一句“同构”，而是一个可展开对象。

你可以在 UI 或高层 graph 中显示：

```text
Le Chatelier Principle -- analogous_negative_feedback_schema --> Lenz's Law
```

但底层应该是：

```text
Process A
  -> AlignmentHypothesis H
  -> Process B
  -> Mapping / Invariants / Tests / Evidence / Residuals
```

---

## 6. 信息几何能不能解决“两个过程相似”的问题？

可以，但只能解决一部分。

信息几何要求你先把两个过程变成某种概率分布族、轨迹分布、Markov kernel、参数流形，或者可比较的统计对象。比如：

```text
Process A -> trajectory distribution P_A(τ | intervention)
Process B -> trajectory distribution P_B(τ | mapped intervention)
```

然后你可以比较：

```text
D_KL(P_A || P_B)
Wasserstein(P_A, P_B)
Fisher-Rao distance
Jensen-Shannon divergence
trajectory embedding distance
```

但是，信息几何不能自动告诉你：

```text
Le Chatelier 的 perturbation 应该对应 Lenz 的 dΦ/dt
```

这个 mapping 是结构归纳问题，不是单纯度量问题。

所以更准确的分工是：

```text
LLM / symbolic parser / domain model:
  生成候选变量映射。

范畴论 / typed graph morphism:
  检查映射是否保结构。

信息几何 / distribution metric:
  度量映射后两个过程的轨迹分布有多接近。

verifier:
  检查这种相似性是否真的提升任务表现或解释质量。
```

换句话说：

> **范畴论回答“哪些结构必须被保留”；信息几何回答“保留之后还差多远”；世界模型回答“用这个相似性会不会让 agent 做得更好”。**

---

## 7. “去掉一条边/节点”是什么意思？不要把它等同于“假设不成立”

你提出的环境动作很有启发性：

```text
agent 每回合加一条边/节点；
environment 每回合去掉一条边/节点；
去掉后性能下降最多的就是最重要的相关性/因果性。
```

这个想法可以做，但要修正语义。

在工程上，“去掉边/节点”应该叫：

```text
counterfactual mask / ablation
```

不是从 committed graph 里物理删除，而是在一次实验中临时 mask：

```text
do(mask edge e)
do(mask node n)
do(mask retrieval of assumption h)
do(mask formal alignment payload p)
```

然后比较：

```text
score_with_h - score_without_h
regression_with_h - regression_without_h
cost_with_h - cost_without_h
```

如果 mask 后性能下降最大，说明这条边/节点对当前任务分布有高**功能贡献**。但这不自动等于哲学意义或科学意义上的“真因果”。

你至少要区分三种东西：

```text
truth:       这个假设是否符合外部世界？
utility:     这个假设是否帮助当前 agent 完成任务？
causal role: 在控制其他变量后，它是否造成性能变化？
```

一条边可能不真，但因为 exemplar boost 让 judge 喜欢，所以有 utility。
一条边可能很真，但当前 benchmark 不需要，所以 utility 很低。
一条边可能在当前 graph 里贡献大，但只是因为替代假设还没被激活。

你 reconstruction 里已经指出，naive self-validation gate 会被 selection bias、judge style、exemplar boost、小样本噪声污染；所以新假设不能只是 append 到 KB，也不能只靠一次 A/B gate。

因此，edge/node ablation 应该进入 verifier stack：

```text
V0: cheap self-check
V1: world-model prediction
V2: matched ablation
V3: placebo edge / length-matched context
V4: cross-domain / cross-judge
V5: fresh heldout
V6: human or objective validation
```

POPPER 的价值也在这里：它不是只让 LLM judge 喜不喜欢，而是让 agent 设计并执行 falsification experiments，并用 sequential testing 控制 Type-I error；但 POPPER 目前偏自由形式科学假设，你要把它扩展到 agentic assumption lifecycle。([arXiv][2])

---

## 8. 世界模型应该怎么进入你的架构？

你现在的世界模型还不是“完整世界模拟器”，而是一个 cheap verifier / budget gate。gap 文档说得很明确：它主要预测 proposal acceptance probability、regression risk、下一步该 ablate/collect/repair/reject、是否低收益 tie，并减少 live API/judge 调用。`pre_live_tie_screen_20260609` 已经把 live calls 从 7 降到 3，失败 live calls 节省 4，live call reduction 57.14%，且 accepted positive blocked 为 0。

这很好，但下一步要把它从“预算门控器”升级成“前瞻模拟器”。

你的 state 可以先不是物理世界状态，而是：

```text
S_t = {
  active_assumption_subgraph,
  task_features,
  residual_cluster,
  selected_strategy,
  retrieved_cases,
  verifier_state,
  current_confidence,
  cost_budget,
  previous_failures
}
```

action 可以是：

```text
A_t = {
  add_node,
  add_edge,
  add_alignment_hypothesis,
  add_process_model,
  mask_node,
  mask_edge,
  run_ablation,
  collect_more_evidence,
  repair_scope,
  reject_candidate
}
```

world model 预测：

```text
P(accept | S_t, A_t)
P(regression | S_t, A_t)
P(failure_type | S_t, A_t)
expected_score_delta
expected_cost
expected_information_gain
```

这和你提到的 **Discrete World Models via Regularization** 很契合。DWMR 的目标是学习 Boolean latent world model，强调离散/布尔状态表示对 search heuristics、symbolic reasoning、planning 有用；它通过 latent prediction 加 entropy/independence/sparsity-locality regularization 学布尔状态和转移。你的 Assumption Graph 操作天然是离散的：加节点、加边、mask edge、accept/reject、residual cluster 迁移。([arXiv][3])

但注意：DWMR 类模型不应该一开始拿来模拟“真实物理世界”。它更适合作为：

```text
Assumption-Graph Transition Model
```

也就是模拟：

```text
在当前假设图状态下，如果新增/删除/验证某个候选假设，
系统表现、风险、残差类型、后代生产力会怎么变。
```

如果你要引入因果世界模型，也不要先做宏大的“世界因果模型”，而是先做 pipeline-level causal model：

```text
task_features
  -> retrieved_assumptions
  -> selected_hypothesis
  -> generated_answer/process_model
  -> verifier_outcome
  -> final_score/cost/regression
```

相关的 causal world model for language agents 已经在做“把 causal variables 与自然语言连接，并让 CWM 作为 simulator 支持长程规划”的方向；这说明你的“语义问题投影到可干预世界模型”的直觉是合理的，但应该先从 pipeline causality 而不是真实世界 causality 做起。([John Gkountouras][4])

Dreamer 系列也给了你一个模型式 RL 参照：DreamerV2/V3 都是学习 latent world model，再通过 imagined futures 改进行为；区别是它们面向 Atari/Minecraft/连续控制，而你的 latent state 是 assumption graph + residual + verifier state。([OpenReview][5]) ([arXiv][6])

---

## 9. 你真正需要的是“离散假设博弈树”，但节点要分型

你说“语义问题提取的决策问题，本质是离散博弈树”，这非常对。建议把它定义成：

```text
Hypothesis Game State:
  G_t: 当前 Assumption Graph
  R_t: 当前 residual buffer
  B_t: budget / cost
  W_t: world-model belief
  V_t: verifier state
  M_t: memory / evidence state

Agent action:
  propose / specialize / generalize / align / ablate / verify / merge / reject

Environment action:
  perturb task distribution
  reveal contradiction
  mask assumption
  introduce counterexample
  shift judge / evaluator
  expose regression
```

但是 graph 里的节点不要只有一种。至少要分：

```text
ConceptNode:       概念，如“勒夏特列原理”
ProcessNode:       过程模型，如“化学平衡扰动-响应过程”
AssumptionNode:    可失败声明，如“该过程符合负反馈 schema”
AlignmentNode:     两个过程/模型之间的映射假设
EvidenceNode:      支持/反对案例
ResidualNode:      未解释失败
VerifierNode:      验证协议
WorldModelNode:    模拟器或预测器
PolicyNode:        策略/方法，如“控制变量法”
```

这会解决你现在的“一个节点到底表示原理、过程、假设、还是证明？”的混乱。

---

## 10. 方法层假设：控制变量法应该是一个 assumption family

“控制变量法”不是一个普通 wisdom 文本。它应该是一个高层 method-family：

```text
AssumptionFamily_ControlVariables
  ├── EngineeringIncrementalReplacement
  ├── ML_Ablation
  ├── Business_AB_Test
  ├── PromptScaffoldAblation
  ├── ScientificControlledExperiment
```

每个 child 都有自己的适用条件和失败条件。

比如“增量替换核心模块”不是简单等于控制变量法，而是：

```text
在组件耦合度未知、已有 baseline 可运行、模块接口可隔离时，
一次只替换一个核心模块，
比一次性重写全系统更可能收敛。
```

Claude.md 里说得对：方法层假设的反馈回路很长，你要跑完整个项目才知道“一次性构建”还是“增量替换”更好；所以它更接近 algorithm selection / meta-learning / regret，而不是单题回答技巧。

现有 ReMA、MEL、Meta-RL 方向已经能让 LLM/RL 学到一些隐式元策略，但还没做到你要的那一步：从具体失败中显式抽出“控制变量法”这种跨域方法论规则，并在新领域有意识调用。

所以你的贡献点依然很清楚：

> **不是让 LLM 背诵哲学原则，而是把哲学原则变成可选择、可模拟、可验证、可迁移、可演化的假设家族。**

---

## 11. 你需要的开发路线：不要再大改，按“增量替换”推进

### Phase 0：冻结 schema

先把四个核心对象固定下来：

```text
AssumptionManifest
ProcessModel
AlignmentHypothesis
WorldModelTrial
```

这一步不需要训练，不需要大模型，不需要范畴论。只要让所有新增候选都必须落到这四类对象之一。

建议 schema：

```python
class AssumptionManifest:
    id: str
    type: Literal[
        "object", "method", "evaluator", "memory",
        "world_model", "alignment", "self_modification"
    ]
    claim: str
    context_conditions: list[str]
    predicted_effects: list[str]
    risk_predictions: list[str]
    formal_refs: list[str]
    graph_ops: list[dict]
    verifier_plan: dict
    evidence_refs: list[str]
    residual_refs: list[str]
    confidence: float
    metaproductivity: float | None
    status: Literal["candidate", "active", "rejected", "deprecated", "contradicted"]
```

```python
class ProcessModel:
    id: str
    domain: str
    state_variables: list[str]
    parameters: list[str]
    interventions: list[str]
    dynamics: str | dict
    observation_map: str | dict
    invariants: list[str]
    failure_cases: list[str]
    traces: list[str]
```

```python
class AlignmentHypothesis:
    id: str
    source_process: str
    target_process: str
    mapping: dict[str, str]
    preserved_structure: list[str]
    broken_structure: list[str]
    metric_scores: dict[str, float]
    verifier_tests: list[str]
    status: str
```

```python
class WorldModelTrial:
    id: str
    state_summary: dict
    action: dict
    predicted_accept_prob: float
    predicted_regression_prob: float
    predicted_failure_type: str
    predicted_value_delta: float
    actual_outcome: dict | None
    calibration_error: float | None
```

成功标准：任何新 hypothesis 都能被保存、检索、ablate、rollback。

---

### Phase 1：把“一条边”升级成 relation node / overlay diff

你现在最大的表示风险是 edge overloaded。先实现：

```text
committed_graph
candidate_overlay
graph_diff
rollback_ref
```

新增假设只写 overlay：

```text
Overlay_H123:
  + node AlignmentHypothesis_LeChatelier_Lenz
  + edge Process_LeChatelier -> AlignmentHypothesis
  + edge AlignmentHypothesis -> Process_Lenz
  + verifier contract
```

只有 verifier 通过才 merge。

这和你当前 gap 文档里的 gated apply/reject 一致，只是把 graph diff 显式化。

---

### Phase 2：做 10 个 process model，不要一上来做通用世界模型

从最小“过程动物园”开始：

```text
1. Le Chatelier principle
2. Lenz's law
3. thermostat negative feedback
4. predator-prey local stabilization
5. chemical first-order decay
6. radioactive decay
7. RC circuit discharge
8. logistic growth
9. damped oscillator
10. supply-demand equilibrium response
```

每个 process 都写成：

```text
variables + perturbation + response + dynamics/rule + invariants + failure cases
```

然后做 alignment hypotheses：

```text
first-order decay family
negative feedback family
equilibrium restoration family
oscillation family
saturation family
```

这就是你自己的 HypothesisGym / AssumptionBench 起点。Claude.md 也指出现在缺少类似 Gym/MuJoCo 的“广义假设论证实验环境”，可以从 FinStoch / 离散马尔可夫链 / 时间序列 law zoo 做起。

---

### Phase 3：先训练 graph-action world model，不训练真实世界模型

收集数据：

```text
(state, candidate_hypothesis, graph_diff, world_model_prediction, live_result, residual_type)
```

预测：

```text
accept?
regress?
failure_type?
value_delta?
cost?
information_gain?
```

这和 reconstruction 里提出的世界模型蒸馏方向一致：目标不是生成最终答案，而是预测“这条 assumption 会不会帮、最可能失败在哪里、是否 worth live execution”。

成功标准不要看最终问答准确率，先看：

```text
AUROC_accept
Brier_accept
AUROC_regression
failure_type_F1
cost_saved_without_blocking_true_positive
```

你自己文档也提醒：cheap world model prompt 曾经 AUROC 0.40、Brier 0.262，说明 world model 必须模块化评估，不能混在总分里。

---

### Phase 4：加入 causal ablation，但只做 pipeline causality

实现：

```text
do(mask_assumption h)
do(mask_edge e)
do(mask_process_payload p)
do(mask_retrieval_case c)
do(mask_verifier v)
```

对每个 mask 记录：

```text
delta_score
delta_cost
delta_regression
delta_residual_type
delta_downstream_transfer
```

输出一个 causal contribution estimate：

```text
contribution(h, task_distribution)
```

不要声称“证明真实因果”。只声称：

```text
在当前 agent pipeline 和任务分布下，
该 assumption 对表现有可干预贡献。
```

这是严谨且可发表的。

---

### Phase 5：Formal Alignment Layer 最后接入

当你已经有 ProcessModel 和 AlignmentHypothesis 之后，再引入范畴论/信息几何。

先做三种 baseline：

```text
LLM semantic aligner
graph edit / WL / Gromov-Wasserstein style process similarity
trajectory distribution metric / information geometry metric
```

再做 formal checker：

```text
typed variable mapping
invariant preservation
commutative diagram check for finite cases
negative controls
```

你的 current gap 文档也明确说，当前 category-theory component 是 bounded structural morphism layer，支持 objects、morphisms、invariants、finite diagram checks、negative controls、orthogonal family gates，但不是完整 category-theory theorem prover。这个边界要保留。

---

### Phase 6：新假设生成只能从 systematic residual 触发

不要让 LLM 随机发明 wisdom。正确流程：

```text
collect last N trials
filter: active assumptions applied but failed
cluster residuals
detect systematic gap
ask LLM synthesize missing method/process/alignment hypothesis
check duplicate/conflict in graph
make candidate overlay
world-model screen
fresh verifier stack
merge or reject
```

这和 reconstruction 里“不要 LLM，请发明一条新 wisdom；从最近 500 trials 中筛 residual cluster 再生成 candidate”的路线一致。

---

### Phase 7：最后才做 downstream paper claim

你的 frozen benchmark 应该比较：

```text
ordinary RAG
HippoRAG-style graph retrieval
v16/v20 case-backed reflection
no world model
no recursive runner
no formal alignment
full recursive assumption graph
```

gap 文档也说，paper-level claim 需要 frozen end-to-end benchmark on unseen tasks，并比较 HippoRAG、ordinary RAG、one-shot self-improve、no world model、no recursive runner。

HippoRAG 2 / From RAG to Memory 的价值是把普通 RAG 的 isolated passage retrieval 变成 KG + Personalized PageRank + recognition memory 的长期记忆机制；你要借的不是“普通 KG”，而是“从 query 激活相关子图”的机制，再把实体图升级为 Assumption Graph。([OpenReview][7])

---

## 12. 对你当前问题的直接答案

你问：“我的假设现在到底是什么形式？新增一个假设又怎么扩展？”

我的答案是：

```text
假设的最小形式 = AssumptionManifest
假设的记忆形式 = Assumption Graph node / relation node
假设的可计算形式 = optional ProcessModel / Program / Equation / Markov Kernel / SCM / NN / Matrix
假设的跨域形式 = AlignmentHypothesis / MorphismSpec
假设的验证形式 = Verifier Contract + Trial Evidence
假设的搜索形式 = WorldModel action over graph state
```

新增假设时，不是“加一条边就完事”，而是：

```text
1. 生成 candidate manifest
2. 生成 graph overlay diff
3. 如需要，附加 process/formal payload
4. 如果是假设两个对象有关联，创建 relation node，而非裸 edge
5. 写入 predicted effect / risk / verifier
6. world model 预测是否值得 live test
7. live ablation / judge / verifier stack
8. accept 后 merge；reject 后作为 residual/evidence 保留
```

而关于“每回合加边，环境删边”：

```text
可以做，但语义应是 counterfactual mask / ablation，
不是逻辑删除，也不是直接证明假设真伪。
```

性能下降最多的边/节点表示：

```text
它在当前任务分布和当前 agent pipeline 下有高功能贡献；
进一步配合 controls、placebo、fresh split、causal model，
才可以更谨慎地讨论 causal contribution。
```

---

## 13. 最后把你的方向压成一句话

你后面的路不应该是“graph 还是范畴论”。正确表述是：

> **用 Assumption Graph 管理所有可失败押注；用 ProcessModel 表示复杂机制；用 AlignmentHypothesis 表示跨域同构/同态猜想；用信息几何度量已形式化对象的距离；用离散/因果世界模型预测 graph 操作后果；用 verifier stack 和 residual clustering 决定假设的生死、细化和后代生成。**

这样，你的“勒夏特列—楞次—控制变量法—世界模型—范畴论—信息几何—递归自我论断”就不再互相打架了。它们是同一个系统里的不同层。

[1]: https://raw.githubusercontent.com/erzhu419/Assumption-Agent/main/assumption_os/README.md "raw.githubusercontent.com"
[2]: https://arxiv.org/abs/2502.09858 "[2502.09858] Automated Hypothesis Validation with Agentic Sequential Falsifications"
[3]: https://arxiv.org/abs/2603.01748 "[2603.01748] Discrete World Models via Regularization"
[4]: https://j0hngou.github.io/LLMCWM/ "Language Agents Meet Causality -- Bridging LLMs and Causal World Models"
[5]: https://openreview.net/forum?id=0oabwyZbOu "Mastering Atari with Discrete World Models | OpenReview"
[6]: https://arxiv.org/abs/2301.04104 "[2301.04104] Mastering Diverse Domains through World Models"
[7]: https://openreview.net/forum?id=LWH8yn4HS2 "From RAG to Memory: Non-Parametric Continual Learning for Large Language Models | OpenReview"
