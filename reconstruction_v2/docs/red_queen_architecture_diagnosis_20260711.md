# Assumption Agent 与 Red Queen Gödel Machine 架构诊断快照

> 保存日期：2026-07-11
> 性质：从本任务保留的完整诊断上下文重建的历史快照。这里的实验数字、
> 文件规模和行号都对应诊断当时的 `reconstruction` 实现，不应被解释为
> `reconstruction_v2` 的当前结果。

## 一、总判断

Assumption Agent 的核心想法和 Red Queen Gödel Machine 确实同源：两者都
把系统改进变成“提出候选假设，再递归验证”。Assumption + Verifier
Contract 甚至可能比整块代码修改更可解释，因为它给搜索对象增加了明确的
中间语义。

但诊断时的实现还不是由这种机制真正驱动的 self-evolving agent。更准确地
说，当时系统包含：

1. 一个有价值的研究假设；
2. 一套很强的审计和 verifier 基础设施；
3. 一个庞大的 HLE 推理系统。

三者尚未闭成同一个学习环。

当时的真实链路是：

```text
Assumption Graph
  -> retrieval / OperatorSpec / morphism
  -> multi-prompt candidate generation
  -> source search / span / comparator / many verifiers
  -> fallback / selector
  -> final answer

final trace
  -> transition dataset
  -> fast-policy miner
  -> candidate/shadow policy
  -X-> does not actually control the next HLE runtime
```

因此，当时真正产生行为变化的主要是 prompt ensemble、手写 domain rule、
source/verifier 和 fallback，而不是“系统自己提出、验证并保留的假设”。

## 二、关键代码证据

### 2.1 隐式策略网络已经失去可辨识性

诊断时，`hle_smoke_eval.py` 已有约 116,808 行、1,401 个函数、770 个 HLE
环境开关、126 个 verifier 函数和 56 个 fallback 函数。这不再是一个可以
逐模块归因的 agent graph，而是长期逐题修补后形成的隐式策略网络。

这会带来四个直接问题：

- 很难知道一次提升来自 assumption、source、selector 还是某个 fallback；
- 局部规则很容易覆盖另一个局部规则；
- 同一个 seed 的模型波动会被误认为代码改进；
- 每次“下一刀”都增加新的自由度，增加过拟合和不可复现风险。

### 2.2 递归没有形成跨题学习

HLE recursive runner 主要递归展开已有 graph 节点。HLE 调用使用
`writeback=False`，所以系统在解决一道题时不会把新提出且通过验证的假设
沉淀为下一道题可以调用的结构化程序。

因此，当时的“递归”更接近同题内的推理展开，而不是：

```text
propose hypothesis
  -> validate hypothesis
  -> estimate benefit and harm
  -> archive or reject
  -> alter future runtime behavior
```

### 2.3 fast policy 没有接入行为闭环

`fast_policy_memory.py` 和 policy miner 已经存在，但 HLE 的四处 router 调用
没有传入 `fast_policy_decision`。即使 lane router 收到 policy，它也主要把
policy 写入日志，不会根据 `selected_actions` 启停 candidate generation、
source lane、solver lane、verifier lane 或 final selection。

这意味着 policy 在数据层存在，但在执行层没有 effectful semantics。一个
只被记录、不会改变 action graph 的 policy 不能构成可验证的学习结果。

### 2.4 miner 学的是故障频率，不是因果收益

当时 miner 的 `expected_utility` 主要来自某个 failure bucket 出现得多不多。
它没有回答更关键的问题：

```text
在同一道题、同一预算和同一 evaluator 下：
启用 policy P 相比关闭 P，修正了多少题，又伤害了多少题？
```

故障出现频率不是 policy 的因果收益。高频故障可以对应一个无效修复，也可以
对应一个副作用更大的修复。没有 policy-off/on counterfactual，就无法可靠地
估计净收益。

### 2.5 transition dataset 缺少可泛化触发特征

transition dataset 为防止泄漏保存了大量 hash，这是正确的审计方向；但当时
没有保存足够的结构化语义特征，例如关系类型、约束结构、可验证条件、候选
差异、反触发条件和 verifier 适用域。

结果是数据可以证明“发生过一次 transition”，却很难支持 router 学会“什么
新题应触发哪条 policy”。

## 三、self-evolution 文献的共同要求

当时对 `reference/self_evo_continual_20260707` 中 22 篇论文和相关仓库做了
机制归纳。它们给出的共同约束如下。

| 文献组 | 核心机制 | 当时项目的缺口 |
|---|---|---|
| SkillLearnBench、Voyager、LifelongAgentBench、Fast/Slow | 经验必须编译成可执行、可复用的 skill 或 fast weight | OperatorSpec 最接近，但 application coverage 曾为 0 |
| Reflexion、ExpeL、FLEX、EvolveR、AgentEvolver | 需要成功/失败对照、经验抽象、任务生成和 credit assignment | 主要从 error bucket 生成固定修补建议 |
| DSPy、GEPA、TextGrad、OPRO、GPTSwarm | 必须有小而明确的可优化计算图 | 优化表面是 116k 行隐式分支 |
| MemGPT、MemoryBank、A-MEM、HippoRAG 2 | 记忆组织和检索有价值，但 retrieval 不等于 learning | 大量精力投入 source coverage，policy learning 没闭环 |
| Agent-as-a-Judge、Self-Rewarding LM | 中间轨迹评价重要，但同模型自评会复制偏差 | 多数 verifier 仍由同一模型承担 |
| AI Agents That Matter | 必须使用 heldout、简单基线、预算匹配和可复现控制 | harness 接近要求，但主实验仍不完整 |

SkillLearnBench 尤其重要：多轮 self-feedback 只有在外部反馈可靠时才稳定提高；
纯 self-feedback 容易递归漂移。这解释了为什么继续叠加同模型 verifier 没有
带来稳定提升。

## 四、与 Red Queen Gödel Machine 的关键差别

Red Queen 的价值不只是“递归”两个字，而是以下制度设计。

### 4.1 搜索对象是 archive tree

它搜索的是 agent/evaluator workspace 的候选档案树，不是只保留一条最新代码
主线。多个 clade 可以同时存在，系统可以比较不同演化路线的
metaproductivity。

旧 Assumption Agent 更接近单谱系人工 hill climbing：同一批 debug seeds
持续影响代码、verifier prompt 和 gate，每一次局部胜利都被编译进全局复杂度。

### 4.2 evaluator 在 epoch 内冻结

候选 agent 在固定 evaluator epoch 中比较，避免优化目标与被优化对象同时漂移。
evaluator 只有在固定 heldout anchor 上通过保守下界后才能晋级。

### 4.3 数据角色严格隔离

产生候选的数据、选择候选的 validation、最终 test 必须隔离。旧系统曾反复用
固定 HLE seeds 调参，因此这些 seeds 只能作为 regression/debug cohort，不能再
作为泛化证据。

### 4.4 selective erasure

evaluator 更新后，只清除依赖旧 evaluator 的记录，而不是抹掉整个 archive。
每个结论都需要记录 evaluator epoch 和依赖关系。

### 4.5 对论文主张保持谨慎

Red Queen 论文自身的实验范围有限，主要结果依赖强模型，理论保证主要是
epoch-local，anchor 偏差仍可能导致 evaluator 漂移。因此应该借鉴它的隔离、
archive 和晋级机制，而不应直接继承其性能主张。

## 五、当时的效果证据

诊断时没有可靠证据证明 agent 已稳定优于 raw 或 HippoRAG：

- 旧 `f577d1a7` 的 6/12 是 fixed/debug regression，不能代表 unseen；
- 一个 agent-only “unseen12”曾为 5/12，但没有同题 controls，且 Operator
  application coverage 为 0；
- 一次完整 triad promotion report 中 agent/raw/HippoRAG 都是 2/12；
- 一次 controls-only n=12 中 raw 2/12、HippoRAG 3/12、budget-matched raw
  4/12、budget-matched HippoRAG 3/12，但有四个顶层 endpoint error，而且没有
  agent arm。

所以当时既没有主性能结论，也没有 attribution 证据表明答对的题来自
Assumption 机制。

## 六、潜力判断

### 6.1 能否超过单次 raw

有现实潜力，尤其适合结构重复、可验证、需要实验控制、关系绑定或跨题技能
复用的任务。

### 6.2 能否超过 budget-matched raw

可能，但当时架构尚未证明。必须把 extra compute 转换成可复用 policy，并用
counterfactual 证明净收益，而不是仅增加 prompt、source 和 verifier 调用。

### 6.3 broad random HLE 是否合适

不理想。HLE 包含大量知识瓶颈和一次性长尾题，跨题 assumption 很难复用，
source availability 往往比 policy quality 更决定结果。HLE 可保留为外部 transfer
或 stress test，但不应作为唯一主战场。

### 6.4 更合适的主测试

优先级如下：

1. SkillLearnBench：任务 family 重复、反馈可程序化、适合 policy/skill
   before-after 和 family-out；
2. LifelongAgentBench：检验跨任务持续学习和遗忘；
3. 预注册的 HLE assumption-transfer slice：只用于验证机制能否迁移到高难
   knowledge/reasoning 任务；
4. 非 HLE 的 sealed transfer set：验证 learned policy 是否只适用于开发域。

泛化实验必须采用 instance-out、family-out、operator-family-out 和 leave-domain-out，
并冻结 train/validation/sealed-test manifests。

## 七、应重构成什么

### 7.1 三层假设

1. `TaskHypothesis`：一道题里哪个关系、约束、因果解释或操作步骤成立；
2. `PolicyHypothesis`：什么题应调用哪个 operator/source/solver、使用多强、何时
   abstain；
3. `EvaluatorHypothesis`：什么验证标准能可靠预测最终正确性，在哪些条件下会
   失效。

### 7.2 统一编译形式

每个假设都应编译成 `HypothesisProgram`：

```text
trigger
anti-trigger
action graph
expected effect
verifier contract
fallback / abstain
lineage
evaluator_epoch
```

Archive 节点保存的不只是一条 assumption，而是一组活跃假设、router 版本、
verifier 版本和依赖 epoch 的完整配置。

### 7.3 第一刀

最重要的第一刀不是继续修 source span，而是：

1. 把 fast policy 真正接入 runtime；
2. 让 `selected_actions` 可以启停具体 lane；
3. 对同题执行 policy-off/on shadow counterfactual；
4. 从真实 gain、harm、cost 和 abstention 结果学习 expected utility；
5. 将大文件拆成声明式模块图，使每个模块可独立替换、消融和归因。

### 7.4 evaluator 和 promotion

- evaluator 在 epoch 内冻结；
- evaluator 候选必须在固定 anchors 上通过 lower confidence bound；
- agent 候选必须在 validation 上超过 incumbent，或同分但在预注册稳定性/成本
  指标上更好；
- evaluator 更新触发 dependency-aware selective erasure；
- sealed test 只用于最终报告，不参与 proposer、miner 或 promotion。

### 7.5 最低实验矩阵

真正的主结论至少需要：

```text
raw
raw_budget_matched
fixed agent without learning
fixed agent with hand-authored policy
evolving agent with hypothesis proposal and validation
```

同时报告：

- task accuracy / reward；
- hypothesis proposal precision；
- policy activation precision and recall；
- policy-off/on causal gain and harm；
- evaluator fidelity and calibration；
- latency、token、model-call 和 error stability；
- archive growth、retention、forgetting 和 cross-family transfer。

## 八、什么才算“真正的自我提出并验证假设”

以下条件必须同时成立：

1. 假设不是人工预先写死的唯一候选，而是由系统从 train evidence 提出；
2. 假设被编译成会改变 runtime action graph 的程序；
3. 系统执行同题或匹配题的 off/on counterfactual；
4. verifier 独立记录支持、反例、成本和适用域；
5. promotion 不读取 sealed test；
6. 通过的假设进入 archive，并在未来未见题上被 router 调用；
7. 失效假设能够被降级、abstain 或 selective erasure；
8. 最终提升可以归因到该假设，而不是无法区分的 fallback 或额外预算。

满足这些条件后，“自我提出并递归验证”才不只是 prompt 中的叙述，而是系统
可观察、可消融、可证伪的学习机制。

## 九、一句话结论

研究方向成立，显式 assumption 也可能是比 Red Queen 中整块 workspace mutation
更可解释的中间表示；但旧项目优化的是“怎样更复杂地回答 HLE”，尚未真正优化
“哪些假设值得跨题保留、何时触发，以及它们是否因果性地改善未来行为”。
