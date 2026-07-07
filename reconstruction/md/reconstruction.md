我：
这是我自己的自发现假设并自我递归式论证的算法，gemini.md和claude.md是最开始的构想。 /manual 里是具体开发手册，代码也基本是这样做的。 现在有点问题，因为我发现了两个新方向： 一个是HippoRAG这两篇论文，尤其是From RAG to Memory，我发现我最初的构想——用范畴论表示一个新假设（一个假设的核心其实就是两个变量或者说feature之间是否有相关性，然后就是这种相关性是什么函数形式，在LLM的视角，这个函数可能非常复杂，在RL的视角，这个函数就是个NN），在From RAG to Memory这篇论文里，它用的是知识图谱来表示两个词之间的关系（这就涉及由模型扫描新问题的时候用offline data的模型进行推断，推断其实就是假设），而我应该是可以把知识图谱替换成范畴论里两个概念/实体之间的关系，这会让这个模型更general 另一个是，今年出现了一批自进化智能体，包括bengio的自我递归式推断模型，这都和我的设想一样，但是好像目前的代码里没体现出来。 目前的效果很弱，帮我好好设计下新，看看我最初的设想：claude.md和gemini.md，然后结合最新的方向的几篇论文（你自己也可以搜索查找最相关的），然后看看怎么把这个方向重新做一下，给个系统的路线图。这种提出新假设，目的不止是充当科学家，而是其实万物都是假设。
只是用一种科学家式的方式去提出假设，自我递归论证的方式，去证明某个假设（我感觉自己饿了，我吃饭，得到满足的多巴胺，证明我这种饥饿的感觉和吃饭之间构成保证我生存的机制，但其实我该不该吃饭，什么感觉下吃，都是假设，只是有些假设被验证的次数太多，且毫无任何苛刻的附加条件，就被当作真理，有些由于狭窄，还只是假设，假设和公理没任何区别形式上以及作用上）
GPT：
对，这一句把方向真正钉住了：**提出新假设不是“科学家专用功能”，而是智能体处理世界的最小单元。科学假设只是其中一个高可验证、高形式化的特例。**

更准确地说，不是“万物都是命题式 hypothesis”，而是：

> **智能体的每一次理解、检索、计划、执行、评价、修正，都隐含一个可失败的假设。**

比如：

“这个问题属于商业竞争分析”是假设。
“应该检索巴菲特卖柠檬水这个案例”是假设。
“控制变量法适用于当前工程调试”是假设。
“这个测试能代表真实成功”是假设。
“这个 judge 的偏好等价于人类偏好”是假设。
“把这个 memory 放进上下文会帮助而不是干扰”是假设。
“当前失败说明策略错了，而不是执行没 follow 策略”也是假设。

所以你的项目不该被框成 “AI scientist”。它更像是一个 **Assumption Operating System / Hypothesis Operating System**：让 agent 把自己所有行动背后的隐式假设显式化、记录、检索、模拟、验证、修正、合并、继承、淘汰。

---

## 1. 你现在的核心命题应该改成这样

我建议把项目的第一性原理写成：

> **智能行为 = 在不确定世界中不断生成、选择、执行、验证、修正假设的递归过程。**

其中“假设”不是狭义的科学命题，而是一个结构：

```text
Assumption = 在上下文 C 下，若采用解释/策略/模型/行动 H，
则未来观测 O 应该以某种方式变化；
若不变化，则产生 residual，并触发修正。
```

这可以覆盖你说的所有例子：

“在地铁口卖布朗尼能赚钱”是一个商业世界模型假设。
“东汉皇室短寿可能由遗传病导致”是一个历史-医学因果假设。
“E=mc²”是一个物理理论假设。
“RL 奖励函数和场论点源函数存在同构”是一个跨形式系统的结构假设。
“先最小原型、再替换核心模块”是一个方法论假设。
“v16 的 case+reflect 会优于单纯 wisdom 文本”是一个 scaffold 设计假设。

这和你仓库里现在的五阶段设计其实已经高度一致：你原来把阶段零定义为“递归假设-验证智能体”的地基，后续阶段依赖方法论知识库、调度器、经验反馈、形式化对齐和新假设生成；阶段零文档也明确说知识库从第一天起就必须为持续演化设计，而不是冻结成静态规则。

---

## 2. 你之前的系统弱在哪里：不是 “没有假设”，而是 “假设没有成为一等公民”

你现在的 v16 其实已经证明了一件很关键的事：**case-backed reflection 是有效的执行 scaffold**。v16 的核心结构是“跨域判例 + 同域判例 + 第二轮 audit-revise”，并且在 100 题 benchmark 上显著超过 baseline_long、Self-Discover 和 v13-reflect。 代码里也很清楚：Turn 1 用 2 条 wisdom，每条配 3 个跨域判例和 1 个同域判例生成 draft；Turn 2 再检查 draft 是否真的应用了 prior/wisdom，找出 1–2 个 blindspot 并改写。

但这个系统目前仍然主要把 wisdom 当作 **prompt 内容**，而不是把每个 wisdom、每次检索、每个案例、每次 audit 判断都当成一个可追踪、可证伪、可继承的假设节点。

这也是为什么你后面的 paper 草稿会走到一个很重要的负结果：最简单的“提出新 wisdom → n=50 same-family A/B gate → KEEP/REVERT”闭环并不可靠。你的草稿里已经写得很清楚：3/12 个候选被 gate 接受，但后续通过跨家族重判、位置重随机、扩展样本、跨 solver、fresh-domain 等审计后，结果是 replicated clean null；cached signal 很可能来自 exemplar boost、top-3-of-12 selection bias 和小样本噪声，而不是新 wisdom 的真实效用。

这说明：**新假设生成不是简单往 KB 里 append 新规则，也不是一次 A/B gate 能解决的。真正要做的是建立“假设生命周期管理”。**

---

## 3. 新方向的统一设计：Assumption Graph，而不是普通 Knowledge Graph

你提到 HippoRAG / From RAG to Memory，这个方向非常对，但我会把它再推进一步：

> 不要只把 HippoRAG 的 KG 换成范畴论关系；应该把 KG 换成 **Assumption Graph**。

HippoRAG 的关键不是“知识图谱”本身，而是它把长期记忆从孤立向量检索变成了可多跳联想的结构检索：LLM 抽取 triples，构建图，用 Personalized PageRank 从 query seed 节点扩散，从而完成单步多跳检索。HippoRAG 2 又加入 phrase node、passage node、query-to-triple、recognition memory、dense-sparse integration，解决纯实体图丢失上下文的问题。 早期 HippoRAG 论文也强调，普通 RAG 把 passage 独立编码，难以跨 passage 做知识整合，而 PPR over KG 可以让相关子图被激活，近似 hippocampal associative memory。

你的 Assumption-Agent 应该借这个机制，但节点不是普通实体，而是：

```text
AssumptionNode:
  id
  type: object / method / evaluator / memory / world_model / alignment / self_modification
  claim: 自然语言假设
  formal_form: 可选，程序 / 方程 / Markov kernel / test / prompt / policy
  context_conditions: 适用条件
  predicted_effects: 预期会改善什么
  risk_predictions: 可能破坏什么
  verifier: 如何验证
  evidence: 支持/反对案例
  residuals: 失败时留下的结构化残差
  confidence: 当前置信度
  metaproductivity: 其后代假设带来的长期收益
  status: active / candidate / deprecated / contradicted
```

边也不是普通 `subject-relation-object`，而是：

```text
supports
contradicts
specializes
generalizes
is_analogy_of
is_formal_isomorphism_of
uses_evaluator
generated_from_residual
failed_because
execution_lapse_of
replaces
depends_on
```

这样，“控制变量法”“消融实验”“最小可运行原型”“A/B test”不再只是几条 wisdom 文本，而是同一个 assumption family 的多个表面实现。范畴论/信息几何只在 `is_formal_isomorphism_of` 这条边上发挥作用，而不是强行统一所有东西。

---

## 4. 最新论文怎么嵌进你的系统

### 4.1 HippoRAG 给你记忆结构：从 case retrieval 到 assumption retrieval

你 v16 的“判例法”已经是 HippoRAG 思想的一个手工版本：每条 wisdom 预先选 3 个跨域判例，运行时再检索 1 个同域判例。代码里明确写了“3 个跨域案例用于抽象不变原则，1 个同领域案例用于落回当前语境”，并且注释里直接把它比作 law-school/case-law 机制。

现在要升级成 Assumption Graph：

当前问题进来后，不是直接从 75 wisdom 里 LLM 选 2 条，而是：

```text
problem → extract problem signals
        → seed assumption graph
        → PPR 激活相关 assumption / case / residual / verifier
        → recognition memory 过滤无关 assumption
        → 取 top-k assumption families
        → 给 executor 作为结构化 context
```

这会解决一个关键问题：v16 是 “wisdom → case”，但新系统应该是 “problem/residual → assumption subgraph”。

---

### 4.2 HGM 给你选择准则：不要看单次 performance，要看假设后代的生产力

Huxley-Gödel Machine 最重要的不是 coding agent，而是它把“当前 benchmark 分数”与“长期自我改进潜力”区分开。HGM 定义 clade-metaproductivity，衡量一个节点的后代子树能产生多好的 agent，并指出 immediate benchmark performance 可能和长期改进潜力错配。

你可以把它直接迁移成：

```text
Assumption Clade Productivity, ACP(h)
= 从假设 h 出发，未来生成的子假设/修正/应用路径中，
  能带来的最大或平均长期任务收益。
```

这比“这条 wisdom 本次 A/B 赢了多少”强得多。

例子：
“先做最小可运行原型”在某个单题回答里可能不加分，但它会产生大量可调试、可验证、可复用的后代轨迹，因此 ACP 高。
“说得更全面一些”可能单次 judge 喜欢，但不会产生可继承结构，因此 ACP 低。

你的新调度器不应该问：

```text
which assumption maximizes current answer win-rate?
```

而应该问：

```text
which assumption family, if expanded/tested now,
will produce the most useful future assumptions?
```

这就是把 HGM 的 CMP 思想迁移到“万物皆假设”的世界。

---

### 4.3 AHE 给你工程治理：每个假设必须有 manifest

Agentic Harness Engineering 的核心是 observability：component observability、experience observability、decision observability。它让每个 harness edit 都带上自声明预测，并在下一轮 task-level outcome 中验证，相当于每个 edit 都是一个 falsifiable contract。 AHE 的结果也提醒我们：agent 能较好预测自己会修复什么，但预测 regression 很弱，fix precision/recall 明显好于随机，而 regression precision/recall 只比随机好一点。

这对你非常重要。你现在所有新增 wisdom / trigger / case / prompt 改动，都应该带一个 manifest：

```json
{
  "assumption_id": "A_1023",
  "claim": "给跨域判例能帮助 LLM 抽象不变原则",
  "expected_fixes": ["business analogical reasoning", "engineering diagnosis"],
  "expected_regressions": ["pure math proof", "token budget pressure"],
  "evidence": ["v16 vs baseline_long", "v15/v16 ablation"],
  "verification_plan": "fresh split, cross-judge, placebo context, cross-solver",
  "rollback_condition": "fresh n>=100 mean win-rate <= baseline"
}
```

这会把“假设”从一句想法变成可治理对象。

---

### 4.4 EmbodiSkill 给你 residual 分类：失败不一定说明假设错了

EmbodiSkill 的洞察和你项目高度同构：失败轨迹不一定说明 skill 错了，也可能是 executor 没有 follow valid skill。它把 reflection 分成 Discovery、Optimization、SkillDefect、ExecutionLapse；ExecutionLapse 不修改 skill body，只更新 appendix，提醒 executor 下次遵守已有有效 skill。

这可以直接变成 Assumption-Agent 的 residual taxonomy：

```text
1. Discovery:
   轨迹暴露了 KB 没有的新假设。

2. Optimization:
   旧假设方向对，但操作方式不够好。

3. Assumption Defect:
   旧假设本身错了、缺条件、过度泛化。

4. Execution Lapse:
   假设是对的，但 agent 没真正执行它。
```

这会修正你现在 Phase 4 的一个大问题：看到失败就想“生成新 wisdom”，但很多失败其实只是 **没有真正 apply 已有 wisdom**。v16 audit pass 已经在单题层面做了这个事：检查 draft 是真的塑形答案，还是只是引用/装饰。 EmbodiSkill 给了你一个更完整的跨任务版本。

---

### 4.5 GRAM / recursive reasoning 给你搜索形态：假设不是一条链，而是分布

Generative Recursive Reasoning 的关键是：确定性 recursive model 只沿一条 latent trajectory 收敛，容易 mode collapse；GRAM 把 recursive reasoning 变成随机 latent trajectories，从而维护多条假设路径，并同时支持 depth scaling 和 width scaling。

你现在的 prompt-based audit 是单路径：

```text
draft → audit → revise
```

新系统应该是多路径：

```text
problem
  → hypothesis trajectory 1
  → hypothesis trajectory 2
  → hypothesis trajectory 3
  ...
  → world-model / verifier / judge ranking
  → keep top branches
  → recurse
```

这和 Nature Methods 那篇关于假设的社论非常贴：AI co-scientist 的价值不只是提出一个 hypothesis，而是可以并行生成、竞争、排序、丢弃大量 hypothesis；那篇社论也强调，这种速度会带来 hallucination 和错误追逐风险，所以必须保留验证机制。

---

## 5. 新系统的总架构：Recursive Assumption Agent

我建议你把系统重构成 7 个模块，而不是继续按原来的 Phase 0–4 线性做。

### 模块 A：Assumption Graph Memory

这是 HippoRAG-style 的长期记忆层。存储所有对象层、方法层、评价层、自修改层假设。

输入：

```text
problem description
current trajectory
failure residual
current goal
```

输出：

```text
activated assumption subgraph
relevant cases
known failure modes
known verifiers
similar historical residuals
```

它不只是 RAG，而是“假设联想系统”。

---

### 模块 B：Hypothesis Generator

生成候选假设，但不只生成科学假设。它生成：

```text
problem framing hypothesis
strategy hypothesis
decomposition hypothesis
retrieval hypothesis
tool-use hypothesis
evaluator hypothesis
world-model hypothesis
self-modification hypothesis
```

这比现有 AI Scientist-v2 更一般。AI Scientist-v2 已经能迭代提出科学假设、设计实验、执行、分析和写作，但它主要还在 ML paper generation 工作流里。([arXiv][1]) Agent Laboratory 也覆盖文献综述、实验、报告写作，但仍然从 human-provided research idea 出发。([arXiv][2]) 你的系统目标是：连“当前应该把问题投影到哪个世界模型”也由假设机制处理。

---

### 模块 C：World Model / Simulator

这是你最缺的硬模块。你的 phase 0.5 文档已经意识到世界模型是“假设的廉价验证器”，目标是让递归假设-验证循环从小时级降到秒级。 但 v16 的真实情况是：它已经有一个“大而慢的世界模型”——LLM 内部推理 + 静态案例缓存 + 运行时同域检索；每次 inference 是一次“模拟 + 评估”的 world-model query，只是成本约 16 秒。

你下一步不是从零训练世界模型，而是：

```text
v16 / v20 expensive simulator
→ 跑 1000+ 问题
→ 收集 (problem, assumption subgraph, draft, audit, outcome, residual)
→ 蒸馏一个 cheap predictor
```

你的 phase 0.5 文档已经写出这个路径：用 v16 在 1000+ 问题上收集三元组，再训练 small forward predictor，让它 ms 级 rollout，服务于 RL/搜索。

---

### 模块 D：Verifier Stack

不要再只用 same-family A/B gate。你的 paper 已经证明这会被 selection bias、judge style、exemplar boost、小样本噪声污染。

新 verifier 应该分层：

```text
V0: cheap self-check / programmatic test
V1: world-model predicted success
V2: cross-judge / cross-family
V3: placebo context / length-matched control
V4: fresh split / distribution shift
V5: objective benchmark if available
V6: human review for high-stakes assumptions
```

POPPER 在“自由形式假设验证”上值得借鉴，因为它不是让 LLM judge 喜不喜欢，而是让 agent 设计 falsification experiments，并用 sequential testing 控制 Type-I error。([arXiv][3]) 但 POPPER 的范围仍偏科学/统计假设；你的 verifier stack 要覆盖所有 agentic assumption。

---

### 模块 E：Residual Analyzer

每次失败后，不直接生成新假设，而是先判断 residual 类型：

```text
existing assumption not applied?      → ExecutionLapse
existing assumption partly right?      → Optimization
existing assumption wrong?             → Defect
no existing assumption explains this?  → Discovery
judge/verifier unreliable?             → EvaluatorDefect
retrieved wrong memory?                → MemoryDefect
world model predicted wrong?           → SimulatorDefect
```

这一步是“万物皆假设”真正落地的关键，因为它允许系统反思的不只是答案，还包括检索、评价器、世界模型、调度器本身。

---

### 模块 F：Selector / Scheduler with Metaproductivity

调度器不再只是选 wisdom，而是选“下一步应该展开哪个假设节点”。

目标函数：

```text
score(h) =
  immediate_expected_utility(h)
  + λ * clade_metaproductivity(h)
  - cost(h)
  - risk(h)
  + novelty_bonus(h)
```

这里的 `clade_metaproductivity` 直接借 HGM：一个假设当前可能没赢，但它的后代可能会产生大量有用假设。

---

### 模块 G：Formal Alignment Layer

范畴论 + 信息几何放在这里，不再承担“统一万物”的任务。

它只做三件事：

1. 发现已形式化假设之间的同构。
2. 合并重复假设家族。
3. 支持跨域迁移。

你 phase 3 文档已经得出一个关键反思：v16 没走 Markov kernel + Fisher 路线，但用 GPT-5.4 选择跨域判例，实际完成了“语义同构的经验归纳”；也就是说，LLM 已经能在一定规模下替代纯形式化工具做语义同构检测。 文档也很清楚地建议：形式化路线可保留为理论研究，工程上可以升级 “LLM-semantic isomorphism” 为正式 Phase 3，并验证同构检测质量与下游性能的相关性。

---

## 6. 递归执行循环应该长这样

```python
def solve(problem, goal):
    state = init_state(problem, goal)

    while not done(state):
        # 1. 把当前状态转成 residual / subproblem
        subproblem = formulate_current_gap(state)

        # 2. 从 Assumption Graph 激活相关假设子图
        subgraph = assumption_memory.retrieve(
            query=subproblem,
            seeds=[state.residuals, state.goal, state.domain]
        )

        # 3. 生成多条候选假设轨迹
        candidates = generator.sample_hypotheses(
            subproblem=subproblem,
            assumption_subgraph=subgraph,
            width=k
        )

        # 4. 用 world model 预测每条轨迹后果
        rollouts = [
            world_model.simulate(state, h)
            for h in candidates
        ]

        # 5. 用 metaproductivity-aware selector 选下一步
        chosen = selector.choose(
            candidates,
            rollouts,
            criterion="utility + assumption_clade_productivity - cost - risk"
        )

        # 6. 执行最小可验证行动
        result = executor.act(chosen)

        # 7. 记录 manifest：该假设预言了什么，实际发生了什么
        manifest = log_assumption_trial(chosen, result)

        # 8. 分析 residual，不急着改规则
        residual_type = residual_analyzer.classify(manifest)

        # 9. 更新图：置信度、适用条件、失败模式、后代假设
        assumption_memory.update(manifest, residual_type)

        # 10. 如果发现系统性 residual，触发新假设生成
        if residual_buffer.has_systematic_cluster():
            new_h = generator.propose_new_assumption(residual_cluster)
            validator.run_protocol(new_h)
```

这就是你说的“递归式自我论断”。但每一层都显式化了：当前行动背后的假设是什么，预期是什么，失败说明什么，是否需要生成新假设。

---

## 7. 具体重构路线图

### 第一阶段：先别训练，先把“假设日志”做出来

你现在最应该做的是改日志格式，而不是先改模型。

每次调用 LLM、检索 memory、选择 wisdom、执行 audit、调用 judge，都记录一条 assumption manifest：

```json
{
  "trial_id": "...",
  "problem_id": "...",
  "assumption_type": "retrieval / strategy / evaluator / answer / audit / world_model",
  "assumption": "...",
  "why_selected": "...",
  "expected_effect": "...",
  "observed_effect": "...",
  "residual": "...",
  "residual_type": null,
  "verifier": "...",
  "cost": "...",
  "status": "pending"
}
```

这一步会立刻提升你所有后续实验质量。AHE 的经验就是：没有 observability，自进化会退化成不可解释 trial-and-error；有了 component / experience / decision observability，每个改动才会变成可验证 contract。

---

### 第二阶段：把 v16 改成 Assumption Graph-RAG

当前 v16 是：

```text
problem → v3 wisdom selection → 2 wisdoms → cases → draft → audit
```

改成：

```text
problem/residual → Assumption Graph PPR → activated assumption subgraph
                 → cases + known failure modes + verifier hints
                 → draft → audit
```

不要一开始就追求复杂图数据库。可以先用 JSON + NetworkX / igraph：

```text
nodes.jsonl
edges.jsonl
evidence.jsonl
trials.jsonl
```

HippoRAG 2 用 PPR 时也就是把 phrase nodes / passage nodes / synonym edges / context edges 组织起来；你可以类比成 assumption nodes / case nodes / residual nodes / verifier nodes。

---

### 第三阶段：把 residual taxonomy 接入 audit

v16 audit 现在问的是：

```text
草稿是否真的应用了 prior/wisdom？
```

升级成：

```text
失败来自哪里？
A. 没有应用已有假设
B. 应用方式不佳
C. 假设适用条件错了
D. 缺少新假设
E. 检索错了
F. judge/verifier 错了
G. world model 错了
```

这会让 Phase 4 的“新假设生成”不再乱触发。你的 phase 4 文档已经把方向改成 residual-driven paradigm shift：从多个失败中收集 residual，聚类，如果发现系统性 cluster，就让 GPT 提炼 missing orientation，再在 held-out 上验证。 这正是正确方向。

---

### 第四阶段：把 HGM 的 clade idea 接到 hypothesis selection

每个 assumption 都有后代：

```text
A: 控制变量法
  A1: 工程调试中的最小替换
  A2: ML ablation
  A3: 商业实验中的 A/B test
  A4: prompt scaffold 消融
```

不要只记每个节点的单次胜率，而要记它整棵子树带来的收益：

```text
ACP(A) = descendants of A 的累计有效验证 / 总成本
```

选择新问题的 prior 时，不只看相似度，还看：

```text
retrieval_score * confidence * ACP
```

这会让系统更像“会学习方法论”的 agent，而不是“查词典的 agent”。

---

### 第五阶段：世界模型蒸馏

你现在不能直接用 RL，因为 v16 世界模型太贵。先做数据集：

```text
(problem, activated_assumptions, draft, audit, final, judge, residual_type)
```

目标不是生成最终答案，而是预测：

```text
1. 这条 assumption 会不会帮？
2. 最可能失败在哪里？
3. 是 execution lapse 还是 assumption defect？
4. 是否值得真实执行？
```

这比训练一个通用 world model 容易得多。

---

### 第六阶段：新假设生成只从 systematic residual 触发

不要再 “LLM，请发明一条新 wisdom”。

新流程：

```text
收集最近 500 个 trials
筛出：已有 assumption 激活但仍失败的样本
按 residual embedding 聚类
找出显著 cluster
让 LLM 解释 cluster 的共同缺口
生成 candidate assumption
查 Assumption Graph 是否重复
构造正负验证集
跑 verifier stack
通过才入库
```

这和科学史上的范式转换机制同构：旧理论解释大部分现象，但留下系统性 residual；新理论必须解释 residual，同时保留旧理论成功。你的 phase 4 文档已经用水星近日点、Michelson-Morley、黑体辐射这些例子写出了这个模式。

---

## 8. 评价体系也要重做：不要只看 answer win-rate

“万物皆假设”以后，评价指标至少有 8 类：

| 能力               | 指标                                        |
| ---------------- | ----------------------------------------- |
| 假设显式化            | 每次关键行动是否生成 manifest                       |
| 假设选择             | 选中的 assumption 是否真 relevant               |
| 假设执行             | 是否真的 apply，而非引用装饰                         |
| residual 分类      | ExecutionLapse / Defect / Discovery 分类准确率 |
| 记忆迁移             | 一个假设在新 domain 是否可复用                       |
| metaproductivity | 一个假设 family 是否产生有用后代                      |
| verifier 可靠性     | cross-judge / placebo / fresh split 稳定性   |
| world model      | 预测失败位置和成功概率的 AUROC/Brier                  |

你 paper 里已经做过一个很好的 warning：最简单的 cheap world model prompt 在实验里 AUROC 只有 0.40，Brier 0.262，甚至不如 uniform，而 Stage 4 “直接让 LLM 发明新 prior”在 5 个 hold-out 中 0/5 胜过 baseline。  这说明：**评价必须拆模块，否则整个系统看起来像在进步，其实某些关键部件是反向的。**

---

## 9. 最终论文定位

我建议你不要再把主论文写成：

> “我们提出一个能自动生成科学假设的 agent。”

而是写成：

> **“Everything is an Assumption: A Recursive Assumption Graph for Self-Hypothesizing Agents.”**

核心贡献可以是：

1. **理论定义**：把 agent 的理解、检索、计划、执行、评价、自修改统一为 assumption lifecycle。
2. **系统架构**：Assumption Graph Memory + Residual Analyzer + Metaproductivity Selector + Verifier Stack。
3. **经验发现**：case-backed reflection 有效，但 naive self-validation gate 失败；这证明 hypothesis lifecycle 需要 observability、residual taxonomy 和 metaproductivity。
4. **实验环境**：HypothesisGym / AssumptionBench，专门测“提出—执行—失败归因—修正—迁移”。
5. **开放接口**：范畴论/信息几何只是 formal alignment plugin，不是总框架。

这会比“范畴论 + 信息几何统一假设”稳得多，也比“AI scientist”范围更大。

---

## 10. 一句话总结新的方向

你现在真正要做的不是“让 AI 像科学家一样提出假设”。

而是：

> **让 agent 意识到自己每一步都在押注一个假设，并建立一套机制，让这些押注可以被记录、检索、模拟、验证、归因、继承和进化。**

这就是“万物都是假设”的工程化版本。科学发现只是这个系统在高形式化世界里的一个应用；写代码、做生意、调试世界模型、选择 prompt、相信某个 judge，也全都是同一套递归假设机制的实例。

[1]: https://arxiv.org/abs/2504.08066?utm_source=chatgpt.com "The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search"
[2]: https://arxiv.org/abs/2501.04227?utm_source=chatgpt.com "Agent Laboratory: Using LLM Agents as Research Assistants"
[3]: https://arxiv.org/abs/2502.09858?utm_source=chatgpt.com "Automated Hypothesis Validation with Agentic Sequential Falsifications"

我：
这个方案，你确定充分参考了所有能找到的资料，以及我上传的这些pdf，外加老的claude.md以及gemini.md对吧
GPT"
不，我不能诚实地说“已经充分参考了**所有能找到**的资料”。这个说法太满了。更准确的说法是：

**上一版方案已经参考并吸收了你上传材料里的主轴，包括 Claude.md / Gemini.md、HippoRAG 两篇、HGM、AHE、EmbodiSkill、GRAM、Self-Evolving Agents Survey、Nature Methods 那篇 hypothesis editorial，以及你 repo 里的阶段文档和 v16 结果；但它还不是一份穷尽式 literature review，也不是对整个 GitHub repo 每个文件、每段代码、每个 manual 的逐行审计。**

我现在可以更精确地拆开说。

---

## 我确实参考进去了的部分

你的原始构想来自 Claude.md / Gemini.md：一开始是“假设是否可以表示为范畴论中的关系/态射，再用信息几何度量假设间距离”，Gemini.md 里反复把范畴论定位成结构骨架、把信息几何定位成度量层，也把 Markov 范畴、概率态射、SPD 流形、Fisher/KL 类度量等都拉进了“假设空间”的讨论。  但我上一版没有把“范畴论 + 信息几何”继续当作总框架，而是把它降级成 **Formal Alignment Layer**，这是基于你后来自己修正出来的观点：它只能处理已经被形式化的假设，不能覆盖“卖布朗尼”“控制变量法”“我现在该不该先做最小原型”这类概念性/方法层假设。

Claude.md 里更接近我上一版的核心：它指出现有工作往往能学到隐式元策略，但没有做到从不同任务失败中显式抽出通用方法论规则，例如“复杂任务应该增量构建”“控制变量法”，再跨域迁移到新任务。 这正是我把系统改成 **Assumption Operating System / Recursive Assumption Agent** 的原因：假设不只是科学命题，而是 agent 每一步行动背后的可失败押注。

HippoRAG / From RAG to Memory 被我用于“记忆层”的设计。HippoRAG 2 的关键是把普通向量检索升级为带有 phrase node、passage node、relation edge、synonym edge、context edge 的图式记忆，并用 query-to-triple、recognition memory 和 Personalized PageRank 做图搜索。  所以我提出的不是普通 KG，而是 **Assumption Graph**：节点从“实体/词”升级成“假设/方法/残差/验证器/案例”，边从“实体关系”升级成 supports、contradicts、generalizes、failed_because、generated_from_residual 等。

HGM 被我用于“选择准则”。HGM 的核心不是“coding agent”本身，而是指出 immediate benchmark performance 和长期 self-improvement potential 会错配；它用 Clade-Metaproductivity 衡量一个节点后代 clade 的长期生产力，并说在特定假设下 CMP oracle 足以实现 Gödel Machine。  所以我把你的系统选择函数从“这条 wisdom 当前有没有赢”改成了“这个 assumption family 的后代能不能产生长期有效的新假设”。

AHE 被我用于“工程治理”。AHE 的三层 observability 很关键：component observability、experience observability、decision observability；尤其是每个 edit 都要带 self-declared prediction，并在下一轮 task-level outcome 中验证，成为 falsifiable contract。 这就是我建议你所有新 wisdom、新 memory、新 prompt、新 verifier、新 retrieval policy 都必须有 manifest 的来源。

EmbodiSkill 被我用于“失败归因”。它明确区分 Discovery、Optimization、SkillDefect、ExecutionLapse；尤其是 ExecutionLapse 表示 skill 本身有效，只是 executor 没按它执行，所以不该改 skill body，而应更新 appendix/提醒。  这直接对应你的系统：一次失败不一定说明假设错了，也可能是检索错了、执行没跟上、judge 错了、world model 预测错了。

GRAM 被我用于“递归搜索形态”。它批评传统 recursive reasoning models 只走单一 deterministic trajectory，提出 stochastic latent trajectories，让模型保留多条假设路径，并支持 depth scaling 和 width scaling。  所以我上一版强调：你的递归假设论证不应该是 draft → audit → revise 的单路径，而应该是多 hypothesis trajectory 并行生成、模拟、竞争、裁剪、递归。

Nature Methods 那篇 hypothesis editorial 也被吸收了。它强调 hypothesis generation 是科学核心，AI co-scientist 能并行生成、排序、丢弃假设；但也提醒 AI 生成假设会带来 hallucination、错误追逐和过度信任风险，需要 human-in-the-loop 和实验验证。 这就是我为什么把 verifier stack 和 residual analysis 放在核心，而不是只强调“生成新假设”。

Self-Evolving Agents Survey 被我用于大框架定位。它把 self-evolving agents 分成 what/when/how/where to evolve，并把可进化对象拆成 model、context/memory/prompt、tools、architecture；还把 self-evolving strategy 定义为基于 trajectory 和 feedback 把当前 agent system 映射到新 agent system。 这和我给你的“假设不仅包括答案，也包括检索策略、工具选择、评价器、世界模型、自修改策略”是一致的。

你 GitHub repo 我也看了结构，不只是用户上传的 md。仓库里确实包含 manual、paper、phase zero 到 phase five、phase half、Claude.md、Gemini.md、v16_final_results、world_model_thinking_layer 等材料。([GitHub][1]) v16_final_results 里写得很清楚：v16 是 cross-domain exemplars + same-domain exemplar + audit-revise，两轮结构在 100 题 benchmark 上显著优于 baseline_long、Self-Discover 和 v13-reflect。([GitHub][2]) 这也是我把“案例法 + 反思”看成你现有系统最强实验证据的原因。

---

## 我没有完全做到的部分

我没有逐行审计整个 `/manual`、所有 phase 目录、所有代码实现。上一版主要是基于你上传的 PDF、Claude/Gemini、repo 的关键报告文件和你描述的现状做系统重构，而不是完整 code review。

我也没有穷尽外部所有最新论文。刚才又补查了一下，除了你上传的材料，2026 年还有至少两个非常相关的新方向应该加入 v2 版路线图：一个是 **AEvo / Harnessing Agentic Evolution**，它把 agentic evolution 视为一个 interactive environment，并让 meta-agent 编辑“控制未来演化的 procedure / context”，这和我说的“调度假设生成过程本身也是假设”高度一致。([arXiv][3]) 另一个是 **AI Harness Engineering**，它把软件 agent 能力定位为 model–harness–environment 系统，并列出 task specification、context selection、project memory、observability、failure attribution、verification、intervention recording 等 runtime substrate 责任；这可以增强你系统里的工程层设计。([arXiv][4])

POPPER 我在上一版提到了，但还没有把它充分工程化进你的 verifier stack。它的重点是让 agent 为自然语言假设设计 falsification experiments，并用 sequential testing 控制 Type-I error，这对“假设验证”很关键。([arXiv][5])

---

## 所以结论是

**方向上，上一版已经充分吸收了你上传材料的主线；完整性上，还不能说已经穷尽。**

我现在对上一版方案的置信度是：
**作为系统架构方向，比较稳。**
**作为博士论文级 literature review，还不够。**
**作为可执行研发路线图，需要再补一次“repo 逐模块审计 + 最新论文矩阵 + v2 architecture spec”。**

最重要的修正不是推翻上一版，而是在上一版上补两层：

1. **Evolution-context layer**：来自 AEvo，把整个演化过程的上下文也当作可编辑对象，而不是只编辑单个假设/单个 wisdom。
2. **Harness-responsibility layer**：来自 AI Harness Engineering，把 task spec、context selection、memory、observability、failure attribution、verification、permissions、intervention log 明确列成系统责任。

这样你的方向会更完整：
**Assumption Graph 负责记忆；Residual Analyzer 负责归因；Verifier Stack 负责证伪；Metaproductivity Selector 负责长期选择；Evolution Context / Harness Layer 负责让整个自我进化过程可观察、可编辑、可回滚。**

[1]: https://github.com/erzhu419/Assumption-Agent "GitHub - erzhu419/Assumption-Agent · GitHub"
[2]: https://github.com/erzhu419/Assumption-Agent/blob/main/v16_final_results.md "Assumption-Agent/v16_final_results.md at main · erzhu419/Assumption-Agent · GitHub"
[3]: https://arxiv.org/abs/2605.13821?utm_source=chatgpt.com "Harnessing Agentic Evolution"
[4]: https://arxiv.org/abs/2605.13357?utm_source=chatgpt.com "AI Harness Engineering: A Runtime Substrate for Foundation-Model Software Agents"
[5]: https://arxiv.org/abs/2502.09858?utm_source=chatgpt.com "Automated Hypothesis Validation with Agentic Sequential Falsifications"
