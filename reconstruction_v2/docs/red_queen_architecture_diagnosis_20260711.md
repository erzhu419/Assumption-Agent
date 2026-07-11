# Assumption Agent × Red Queen Gödel Machine：架构诊断与 Reconstruction V2 复核

> - 初版日期：2026-07-11
> - 本次复核：2026-07-11
> - 代码审计基线 revision：`6224bb5a279f50fbcf1f8b36d19cb4ce6cc6c882`
> - 本次实现复核：receipt/runtime provenance 修复提交 `e43670f6`、`18ff3417`；v3.3 execution-policy 提交 `e0b1a33b`；v3.4 model-only/action-budget revision 已通过零模型 wire 与进程监督诊断，尚待 clean lock、prewarm 和真实 train canary
> - RQGM 版本：arXiv:2606.26294v2，2026-06-29
> - legacy 代码范围：`assumption_os/`；legacy 报告范围：`reconstruction/md/` 与对应 artifacts
> - v2 范围：`reconstruction_v2/`

本文从已故障的旧任务“继续调试 assumption”的本地完整记录中恢复了 2026-07-10
的 Red Queen 原始诊断，并用当前代码、测试、实验报告和本地论文重新核验。旧任务
中的凭证、网络连接参数和与架构无关的敏感内容没有复制到本文。

本文同时保留两个时间切片：

1. **legacy 诊断**：解释旧 HLE 系统为什么不是由假设学习闭环驱动；
2. **v2 复核**：判断上述缺口哪些已经修复，哪些只是有了接口或测试，哪些仍会
   阻断论文级结论。

除非明确写成当前状态，legacy 的实验数字和行号只描述诊断时的旧实现，不代表
`reconstruction_v2` 的最新性能。

## 一、执行摘要

### 1.1 最准确的当前结论

旧诊断的主结论仍成立：legacy HLE 把最有研究价值的“假设学习”放在了旁路，
真正控制答案的是一个由 prompt、检索、手写规则、verifier、fallback 和 selector
组成的高维控制面。`trace -> transition -> miner` 虽然存在，却没有可靠地改变下一题
的 runtime，也没有用 policy-off/on 反事实估计因果收益。

但这句话不能原样套到 v2。`reconstruction_v2` 已经在**接口和实验 harness 层**接通：

- 三类结构化 `HypothesisProgram`；
- 内部 effectful runtime；
- 递归 proposal repair；
- paired policy-off/on evaluation；
- train/validation/sealed-test guard；
- archive node、evaluator epoch 和 selective invalidation 骨架；
- SkillLearnBench instance-out/family-out 协议。

因此，当前诊断应更新为：

> **学习闭环在 harness 层已接通；promotion 所有权、外部 backend action/fallback
> 边界和 86-item 离线可运行协议已经闭合。尚未成立的是 clean development promotion、
> contrastive trigger learning、跨 family 泛化，以及 Red Queen 式多谱系搜索和
> evaluator co-evolution。v3.3 已把 low reasoning/verbosity、32,768-token
> `body_after_prefix` compaction、10,000-token tool-output limit 和 request-compression
> 变成 protocol-owned treatment；`video-object-counting-1` 从 v3.2 的 71.1 MB hard-cap
> failure 降为 19.69 MB valid failure，full train 的最大流量为 40.6 MB，38/38 均未触发
> cap/provider error，说明本次 batch 未再被 fuse 直接阻塞；但 canary/full 的 1.47/19.69 MB
> 波动也说明跨运行稳定性尚未建立。full train 仍只有 37 valid、
> 9 success、1 invalid：`offer-letter-generator-1` 的真实 Codex JSONL 返回了一次
> `web_search` item，违反冻结的 model-only contract。因 `all_valid_before_proposal_v1`，
> proposal/validation 仍为 0。现已定位根因：Codex 0.144.1 会把兼容键
> `tools.web_search=false` 解析后丢弃，未设置顶层 `web_search` 时默认仍为 `cached`；因此
> v3.3 的 38 个请求都暴露 hosted web search，仅一条实际调用。v3.4 改用权威顶层
> `web_search="disabled"`。零模型 loopback 捕获证明，canonical 请求有 7 个本地工具且
> 0 个 `web_search*`，同配置仅换回旧布尔键的阳性对照有 8 个工具并明确包含
> `web_search(external_web_access=true)`。`max_steps=100` 也已定义为可观测的
> `codex_action_start_v1`：每个 `item.started` 都占一单位，由容器内 supervisor
> 在第 100 个 start 终止，并按 task/TID 清理专用 trial 容器基线后新增的所有 live task；它不是 semantic turn。异常退出、畸形
> start、残留 descendant 和 receipt/trace 不一致均 fail closed。v3.4 仍未执行正式模型
> canary 或 full development；不重试 v3.3 样本、不补评分 gate、不触碰 sealed。**

### 1.2 结论分层

| 命题 | 当前状态 | 证据层级 |
|---|---|---|
| legacy HLE 是高维手写控制面，学习 policy 没有闭环 | 支持 | 代码审计 + 历史 artifacts |
| v2 的 proposal -> repair -> off/on -> gate -> archive 接口已连通 | 支持 | 183/183 离线测试 + 小型 live probes |
| v2 的内部 runtime action 能改变 lane plan | 支持 | 代码 + 单元测试 |
| v2 主 SkillLearn 路径执行了每个 typed action/verifier/fallback 的强语义 | **不支持，且协议已停止这样声称** | 只接受四类显式 prompt/self-check lowering；其余 fail closed |
| promotion threshold 完全由冻结 protocol 所有 | 支持 | protocol-bound spec + 宽松 candidate 对抗测试 |
| 86-item offline-ready runtime 已预验 | 部分支持 | readiness/preflight `blockers=[]`；v3.3 历史 cache-only prewarm 86/86、model 未执行；v3.4 v4 receipt 待生成 |
| v2 已产生可保留的 promoted incumbent | **不支持** | available mixed-protocol artifact scan 中 23 份 archive 均 `incumbent_id=null`，22 份 report 无 promotion |
| v2 稳定优于 raw 或 budget-matched raw | **不支持** | v3.3 full train 为 37 valid / 1 remote-tool-policy invalid；没有 proposal、paired validation 或 main result |
| v2 已实现 Red Queen 式多 clade 搜索和 evaluator co-evolution | **不支持** | 目前是单 incumbent；evaluator 路径未接主实验 |

### 1.3 潜力判断

研究问题是连贯且可证伪的。显式 `HypothesisProgram` 可能比整块 workspace mutation
更容易做 lineage、activation 和 off/on attribution；但“更可解释”目前仍是待验证
假设，而不是既成事实。它至少需要以下操作化证据：

- schema fidelity；
- action lowering 成功率；
- lineage completeness；
- prospective activation precision；
- paired gain/harm attribution；
- cross-instance 与 cross-family retention。

在结构重复、可程序验证、能复用操作步骤或约束模式的任务上，超过单次 raw 有现实
潜力；在 broad random HLE 上稳定领先的先验较低，因为知识瓶颈、一次性长尾和 source
availability 会与 policy quality 混杂。HLE 更适合作外部 transfer/stress test，而不应
继续作为唯一开发靶子。

## 二、术语、三种“递归”与证据标签

### 2.1 核心术语

| 术语 | 本文含义 |
|---|---|
| assumption / hypothesis | 可证伪的关系、策略或 evaluator 命题，不等同于任意 prompt 建议 |
| `HypothesisProgram` | trigger、anti-trigger、action graph、expected effect、verifier、fallback、lineage 与 evaluator epoch 的结构化程序 |
| activation | 程序在运行前由可用特征命中，且实际改变 treatment 或 execution plan |
| promotion | 只依据冻结 validation 与预注册 gate，把 candidate 变成未来 runtime incumbent |
| archive node | 一组 active programs、runtime version、evaluator epoch 与证据依赖的完整配置 |
| evaluator epoch | 一个 evaluator、artifact protocol 和 scoring rule 保持不变的时期 |
| selective erasure | evaluator 被替换后，仅使依赖旧 evaluator 的 utility/score records 失效；不是删除失败假设的同义词 |
| clean external evidence | split、provider、预算、runtime、verifier、invalid-row policy 和 protocol lock 都满足预注册约束的外部结果 |

### 2.2 三种“递归”必须分开

1. **同题推理递归**：在一道题内展开 assumption tree 或多轮验证；
2. **假设修复递归**：候选未通过静态/训练检查后，生成有 lineage 的 child；
3. **跨代演化递归**：被 promotion 的程序改变 incumbent，再影响下一代 train residual、
   proposal 和未来题的 runtime。

legacy 主要有第 1 种；v2 已实现第 2 种的机制和第 3 种的 harness，但尚未出现真实
promotion，因此还没有观察到完整的跨代能力积累。RQGM 的核心则是跨任务 archive
tree search，不应被简化成“多调用几次模型”。

### 2.3 证据标签

本文使用以下强度顺序：

- **[CODE]**：源码直接可见的事实；
- **[TEST]**：离线测试验证的 wiring/invariant；
- **[ARTIFACT]**：真实运行留下的报告或 event；
- **[INFERENCE]**：由代码和结果支持、但尚无 controlled ablation 的解释；
- **[PROPOSAL]**：建议或验收标准。

## 三、legacy Assumption Agent 的架构诊断

### 3.1 真实行为链路

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
  -X-> did not reliably control the next HLE runtime
```

所以 legacy 中真正产生行为变化的主要是 prompt ensemble、手写 domain rule、source、
verifier、fallback 和 selector，而不是“系统自己提出、验证、保留并复用的假设”。

### 3.2 高维控制面的复杂度证据

**[CODE]** 在审计 revision 上，
[`hle_smoke_eval.py`](../../assumption_os/hle_smoke_eval.py) 共 116,808 行；AST 可见
1,401 个顶层函数定义，包含嵌套定义时为 1,604。源码中出现 770 个唯一的 `HLE_*`
配置名。按顶层函数名统计，126 个含 `verifier`，56 个含 `fallback`。

这些数字是复杂度代理指标，不等于 770 个布尔开关、126 个独立 verifier 或 56 个
独立 fallback 行为。它们能直接证明的是：旧系统有很大的配置面和函数面；“归因困难”
则是由此产生、并与历史反复局部回归相一致的**诊断推断**。若要严格量化，仍需要
调用图、交互覆盖与模块消融。

该控制面带来四个可观察风险：

- 一次提升难以归因到 assumption、source、selector、fallback 或额外预算；
- 局部规则可能覆盖另一个局部规则；
- 同 seed 的模型波动可能被误认为代码改进；
- 每次“下一刀”增加自由度，扩大 adaptive overfitting 与不可复现风险。

### 3.3 同题递归没有形成跨题学习

**[CODE]** HLE 在
[`hle_smoke_eval.py:L2963-L2973`](../../assumption_os/hle_smoke_eval.py#L2963-L2973)
调用
[`build_recursive_assumption_run`](../../assumption_os/recursive_runner.py#L73)
时使用 `writeback=False`。该 runner 会构造可审计的同题 assumption tree，但不会把
一道题中新提出并通过验证的程序写回为下一题可调用的 incumbent。

因此 legacy 的“递归”主要是同题内展开，而不是：

```text
propose hypothesis
  -> validate and repair
  -> estimate benefit, harm, and cost
  -> promote or reject
  -> alter future runtime behavior
```

### 3.4 fast policy 没有 effectful semantics

**[CODE]** 旧代码已有
[`fast_policy_memory.py`](../../assumption_os/fast_policy_memory.py) 和
[`hle_fast_policy_miner.py`](../../assumption_os/hle_fast_policy_miner.py)，但 HLE 主文件
对 [`route_option_lanes`](../../assumption_os/hle_lane_router.py#L105) 的四处调用并未把
完整 `fast_policy_decision` 接成动作控制。router 即使收到 policy，也主要把
`selected_policy_ids` / `selected_actions` 写入 metadata，不会据此启停 candidate、
source、solver、verifier 或 final-selection lane。

因此 policy 当时是可审计的 data object，不是可消融的 behavior program。

### 3.5 miner 学的是故障支持度，不是因果收益

**[CODE]** 旧 miner 在
[`_make_policy`](../../assumption_os/hle_fast_policy_miner.py#L185-L207)
里用 `support_count / wrong_count` 构造 `expected_utility`。这回答的是“某个 failure
bucket 出现得多不多”，而不是：

```text
同一题、同一 evaluator、同一预算下，
policy_on 相比 policy_off 修正了多少题，又伤害了多少题？
```

高频故障可以对应无效修复，也可以对应副作用更大的修复。没有 matched off/on，
故障频率不能被解释为净因果收益。

### 3.6 transition data 可审计，但缺 prospective trigger semantics

**[CODE]** [`hle_transition_dataset.py`](../../assumption_os/hle_transition_dataset.py)
为防止泄漏保存了 hash、label、failure bucket、cost 和 path metadata，这是正确的审计
方向；但当时缺少足够的关系类型、约束结构、输出 schema、可验证条件、候选差异和
反触发条件。数据可以证明“发生过一次 transition”，却很难支持 router 学会“什么
新题应触发哪条 policy”。

## 四、从 self-evolution 文献抽取的项目设计约束

本地材料实际包含
[`21 篇 self-evolution/continual-learning PDF`](../reference/self_evo_continual_20260707/papers/)、
2 个背景页面、21 个相关 repo，以及单独保存的
[`RQGM 论文`](<../reference/The Red Queen Gödel Machine Co-Evolving Agents and Their Evaluators.pdf>)。
下表是面向本项目的机制综合，不表示每篇论文都逐字主张所有约束。

| 文献组 | 可迁移机制 | legacy 缺口 |
|---|---|---|
| SkillLearnBench、Voyager、LifelongAgentBench、Fast/Slow | 经验需要编译成可执行、可复用的 skill/fast state，并在未来任务中调用 | OperatorSpec 最接近，但历史 application evidence 不足 |
| Reflexion、ExpeL、FLEX、EvolveR、AgentEvolver | 成功/失败对照、经验抽象、credit assignment 与生命周期 | 主要从 error bucket 生成修补建议 |
| DSPy、GEPA、TextGrad、OPRO、GPTSwarm | 需要小而明确、可优化和可消融的计算图 | 优化表面是 116k 行隐式分支 |
| MemGPT、MemoryBank、A-MEM、HippoRAG 2 | 记忆组织与检索有价值，但 retrieval 不等于 learning | source coverage 投入没有闭合 policy learning |
| Agent-as-a-Judge、Self-Rewarding LM | 中间轨迹评价有价值，但自评需要外部 anchor 与漂移控制 | 多数 verifier 与被评系统共享模型偏差 |
| AI Agents That Matter | held-out、简单基线、预算匹配、成本与复现性必须同时报告 | harness 逐渐接近，历史主实验仍不完整 |

SkillLearnBench 关于 external feedback 优于纯 self-feedback 的结果，为旧项目的递归漂移
提供了一个与观察一致的候选解释；它不是项目内因果证明，仍需 recursive/no-recursive
和 external/self-feedback controlled ablation。

## 五、与 Red Queen Gödel Machine 的精确对照

两者在“生成候选 -> 固定标准评估 -> 保留有效变体 -> 影响后续搜索”的抽象上同构，
但搜索单位不同：Assumption Agent 试图搜索显式 assumption/operator；RQGM 搜索的是
可修改的多代理 workspace/archive node。不能把 RQGM 说成也在直接搜索显式假设。

### 5.1 archive tree 与 clade search

RQGM 把 self-improvement 表示成增长的 archive tree，并按 clade metaproductivity 进行
Thompson sampling；一个节点是共享的多代理 workspace。见
[`RQGM PDF pp. 4-5`](<../reference/The Red Queen Gödel Machine Co-Evolving Agents and Their Evaluators.pdf#page=4>)。

legacy 更接近同一代码主线上的人工 hill climbing：固定/debug seeds 持续影响代码、
prompt、verifier 和 gate，每次局部胜利都被编译进全局复杂度。

### 5.2 evaluator 在 epoch 内冻结

RQGM 在一个 epoch 内冻结 evaluator、artifact-generation protocol 和 binary scoring rule；
只有 epoch boundary 才允许 utility transition。这使每个 epoch 保持为固定标准的搜索
问题，而不是 agent 与 judge 同时漂移。

### 5.3 产生候选与选择候选的数据隔离

RQGM 明确分开创建 node 的 training feedback、驱动 node selection 的 validation，以及
最终 held-out test。训练结果可以指导 meta-agent 修改 node，但不能直接构成 search
utility。见 [`RQGM PDF p. 5`](<../reference/The Red Queen Gödel Machine Co-Evolving Agents and Their Evaluators.pdf#page=5>)。

legacy 中反复使用的 HLE fixed/debug seeds 只能继续作为 regression cohort，不能再作为
泛化证据。

### 5.4 anchor lower bound 与 selective erasure

RQGM challenger evaluator 在固定 ground-truth anchor 上按保守 best-belief/lower-bound
标准与 incumbent 比较。替换 evaluator 后，只清除依赖旧 evaluator slot 的 utility
records，并按需重排 archive；无关信息保留。见
[`RQGM PDF pp. 5-6`](<../reference/The Red Queen Gödel Machine Co-Evolving Agents and Their Evaluators.pdf#page=5>)。

因此：

- 失败 hypothesis 的降级、停用或归档不是 selective erasure；
- selective erasure 只描述 evaluator/utility dependency invalidation。

### 5.5 对 RQGM 性能主张保持边界

RQGM 是 preliminary empirical investigation；主要结果依赖强模型，理论保证是
epoch-local，anchor 偏差仍可能导致 evaluator 漂移。适合借鉴的是 archive、数据隔离、
冻结标准、anchor promotion 和 dependency-aware invalidation，而不是直接继承其性能
或开放式自我改进主张。

## 六、legacy HLE 效果证据 ledger

下表把点估计与 validity 同时报告。它不是 benchmark 排名表，而是说明为什么历史
结果不能支持“Assumption Agent 优于 raw”的主结论。

| 证据 | 点估计 | validity 与归因边界 |
|---|---:|---|
| [`fixed/debug agent-only 12`](../../reconstruction/md/hle_candidateconflictresolve_baseline12_agentonly_mini_20260629.md) | agent 6/12 | `pass=False`、paper-clean false、无同题 raw/Hippo、无 budget controls、operator application verifier 0 |
| [`所谓 unseen12 agent-only`](../../reconstruction/md/hle_docrank_mathbinding_current_unseen12_cacheonly_mini_20260708.md) | agent 5/12 | paper-clean false、无 controls；operator selected/activated 12/12，但 application verifier 0，不能归因于 operator/assumption |
| [`fresh triad promotion report`](../../reconstruction/md/hle_parallelrun_unseen_mc12_fair_policy_promotion_20260707.md) | agent/raw/HippoRAG 均 2/12 | 12 个 triad 齐全，但 promotion `pass=False`；有 control errors、无 selector gain、低于 24-triad gate，缺 budget-matched controls |
| [`controls-only 12`](../../reconstruction/md/hle_freshunseen12_controls_multiglob_fair_cacheonly_mini_20260709.md) | raw 2/12、Hippo 3/12、budget raw 4/12、budget Hippo 3/12 | 4 个不对称 endpoint errors，无 agent；clean-shared n=10 分别为 2/10、3/10、4/10、3/10 |

本表没有把 fixed cohort 的后续更高点估计当成反例隐藏掉：同一 cohort 后续经过更多
adaptive debugging 后出现过更高分，但这只提高 regression value，不提高 generalization
evidence。固定 cohort 越被反复用于决策，越不能承担 sealed claim。

历史证据只能支持：

- 系统有工程性能力和若干有效局部模块；
- 某些改动能改变单题或固定 cohort 的行为；
- 尚无可靠证据证明 agent 稳定优于 raw、HippoRAG 或 budget-matched raw；
- 尚无 attribution 证明正确题来自可迁移 Assumption 机制。

## 七、legacy 缺口到 reconstruction_v2 的 closure delta

### 7.1 已明显改善的部分

| legacy 缺口 | v2 状态 | 证据 | 尚缺 |
|---|---|---|---|
| assumption 没有统一可执行 schema | 已实现三类 `HypothesisProgram` | [`models.py:L221-L275`](../assumption_agent/models.py#L221-L275) | 外部 backend 的 typed lowering |
| policy 不改变 runtime | 内部 `PolicyRuntime` 可启停、排序 lane、设参数和执行 operator step | [`runtime.py:L72-L226`](../assumption_agent/runtime.py#L72-L226) | 主 SkillLearn 路径仍主要是 skill 注入 |
| 无 hypothesis repair lineage | 已实现 failed-check -> child repair tree | [`validation.py`](../assumption_agent/validation.py) | empirical repair benefit |
| utility 来自 failure frequency | promotion 已使用 protocol-owned paired gain/harm/cost/LCB，candidate 只能收紧 | [`evaluation.py`](../assumption_agent/evaluation.py) | 尚缺真实 promotion 与 retained gain |
| train/validation/test 混用 | split guard 与 archive-freeze gate 已实现 | [`splits.py:L220-L267`](../assumption_agent/splits.py#L220-L267) | 一次完整 current-protocol sealed run |
| evaluator 变更无依赖失效 | controller/anchor lower bound/selective invalidation 已实现 | [`archive.py:L291-L370`](../assumption_agent/archive.py#L291-L370) | 尚未接入主 evolution 或真实 challenger |
| HLE 是唯一主战场 | 已转向 86-item offline-ready SkillLearnBench instance-out/family-out | [`BENCHMARK_PROTOCOL.md`](../BENCHMARK_PROTOCOL.md) | 尚缺 clean development/family-out 结果 |

### 7.2 当前证据到哪一层

**[TEST]** 当前 `reconstruction_v2` 离线 suite 为 **183/183 通过**。这证明 schema、
wiring、guard、replay、failure handling 和若干 invariant；不证明真实 benchmark
improvement。新增覆盖包括 protocol threshold ownership、candidate 宽松阈值攻击、
backend action lowering、真实/声明 fallback 分离、offline-ready split 不重抽样，以及
离线 verifier receipt 必须绑定 proxy 实际执行的 frozen runtime profile/command。

**[ARTIFACT]** 对 `reconstruction_v2/artifacts` 中可读的 v1/v2/v3 smoke、diagnostics
和 development runs 做混合扫描得到：

- 23 份 `*.archive.json`；
- 22 份 `*.report.json`；
- 非空 incumbent：0；
- 这些 report 中 `promoted=true`：0。

这 23/22 不是 23 次 current-protocol 独立实验，也不能作为样本量；它只是 available
artifact tree 的状态审计。结果不是说 gate “失败”；恰恰说明现存 artifacts 没有把
诊断信号包装成 incumbent。但它也意味着系统尚未完成“promoted program 改变下一代
runtime”的实证闭环。

**[ARTIFACT]** 曾有一次 full replay-locked development 出现 raw 4/18、candidate 7/18、
3 gain/0 harm、cost ratio 0.914，但一条 baseline trial 无效，gate 正确拒绝；该结果只能
视为 promising but inadmissible diagnostic，见
[`STATUS.md:L96`](../STATUS.md#L96)。后续 pre-network-hardening 的 685a run 第一代是
raw 4/18、candidate 5/18、2 gain/1 harm，LCB 为负并被拒绝；第二代未完整收束，见
[`development_recursive.events.jsonl`](../artifacts/paper_primary_v3_ruoli_gpt54mini/runs/685a4482_full_development_20260711/development_recursive.events.jsonl)。
这些结果都不能形成性能主张。

**[ARTIFACT]** clean commit `e07913f9` 上当时的 v3.1 protocol smoke 已完成机制验收：
两臂均为 2 个有效 pair、0 invalid、0 provider/budget mismatch，且 behavior-identical
validation 被精确 replay；两臂 candidate/raw 都是 0/2，因此没有 promotion。它只证明
运输、lowering、paired replay 和 fail-closed promotion 能协同工作，不是性能证据，见
[`smoke_recursive.report.json`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/smoke_recursive.report.json)
和
[`smoke_no_recursive.report.json`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/smoke_no_recursive.report.json)。

随后第一次 v3.1 full development 在完整 38-item train 上严格中止，见
[`development_recursive.events.jsonl`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/development_recursive.events.jsonl)：

- 26 个本地 verifier 有效 observation，其中 9 pass、17 fail；
- 2 个已启动 trial 收到 `429 Too Many Requests`，provider circuit 随即打开；
- 9 个尚未启动的 trial 按同一 circuit 在本地跳过，没有继续消耗模型请求；
- `court-form-filling-6` 的长轨迹累计模型流量为 33,730,000 bytes，超过冻结的
  33,554,432-byte fuse，作为 hard-budget invalid 处理；
- 因 12/38 training observations 无效，training evidence 没有写入 replay cache，
  proposal、repair、validation、archive 和 promotion 均未执行；report/archive 也没有落盘；
- sealed split 保持未访问。

这次失败没有调用 online evaluator：task payload 与 verifier 均来自冻结的本地
SkillLearnBench checkout，evaluation 仍由 post-agent offline verifier 完成；唯一在线流量
是预注册的 agent model inference。因而缺口不是“再下载一个 evaluator”或“再补一个
readiness gate”，而是恢复预注册 provider transport 后取得一份完整、0-invalid 的
development evidence。当前进程内 training replay 也不能跨失败进程复用这 26 条有效结果，
所以它们只能作为 transport diagnostic，不能与后续 run 拼接成 claim。

全 run 退出后的单题、5-step、非 claim transport canary 已在同一 provider route 上恢复：
模型请求完成、offline verifier 正常执行，observation 为 `evaluation_valid=1`、
`task_success=0`。这说明 429 已冷却；任务失败不等于 transport 失败。canary 没有读取
validation/sealed，也不进入任何性能汇总，见
[`transport recovery canary`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/transport_recovery_canary/report.json)。

29 个实际启动 trial 的 network receipt 显示总流量中位数约 2.40 MB、p90 约 5.53 MB、
p95 约 22.03 MB；唯一超过 32 MiB 的就是被右删失的 `court-form-filling-6`，且只超出
175,568 bytes（0.52%）。因此 v3.1 没有因单个 train diagnostic 原地抬 cap，而获得最多
一次同协议、全新 run-root 的 clean rerun。该 rerun 在同一 item 上再次触发 hard cap，
这次观测到 38,599,999 bytes；进程在 stop condition 已不可逆后主动中止，没有继续烧完
余下 train。v3.1 因此正式判为 execution-infeasible，而不是继续重跑到碰巧通过。

唯一允许的资源修订已版本化为
[`v3.2 protocol`](../manifests/skilllearn_paper_protocol_v3_2_ruoli_gpt54mini.json)：
统一 provider-only fuse 从 32 MiB 提高到 64 MiB；model、86-item subset、offline evaluator、
dependency policy、4 workers、search budget 和 promotion contract 均不变。64 MiB 是
train-only 已观察最大流量向上取下一 2 的幂，不读取 validation/test，也不复用 v3.1
observations。后续不再允许第二次按题调 cap。

v3.2 在 clean commit `748469b2` 上重新得到 claim-eligible lock 和 86/86 cache-only
prewarm，随后 full development 的 64 MiB budget 没有触发；`court-form-filling-6` 本轮以
4.70 MB 正常完成并通过 verifier-validity 检查。然而同一路由在 8 个有效 train
observation（1 pass、7 fail）后发生独立 transport outage：一个 trial 明确以 429 终止，
其余三个在途 trial 的原始 Codex JSONL 先连续报告 `gpt-5.4-mini` 没有可用 distributor
channel（503），随后也以 turn failure/429 收尾；provider circuit 打开后，29 个尚未启动
请求被本地跳过。training evidence 因 30/38 invalid 再次 fail-closed，未进入 proposal、
validation 或 sealed，见
[`v3.2 development events`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini/development_recursive.events.jsonl)。

随后使用用户提供、仅注入进程且未落盘的新 GPT Pro credential 对同一
`https://ruoli.dev` / `gpt-5.4-mini` route 做最小 canary：models、Responses、chat
端点均为 HTTP 200，真实 Codex container 也完成 agent -> offline verifier，证明
`ruoli` 本身并非永久不可用。这里没有切 model/provider contract，也没有使用 online
evaluator。

第一个新渠道 full root `gptpro01` 完成 38/38 train：36 valid、12 success、2 invalid；
provider/circuit 与 64 MiB cap 错误均为 0，最大 trial 流量 62.9 MB。两个 invalid 都来自
`temperature-simulation-2/3` 的确定性 receipt false-negative：proxy 实际执行的离线
profile 已生成完整 CTRF（各 7 tests、5 pass/2 fail、reward 0），但旧 auditor 错误检查
了未被执行的 upstream `test.sh` 是否含 `--ctrf`，把本应是“有效任务失败”的结果标成
`verifier_execution_receipt_unsupported`。这不是增加 gate 的理由，而是 evidence source
绑定错误。

提交 `e43670f6`、`18ff3417` 之后，receipt 直接绑定 proxy 实际执行的 frozen runtime
profile ID/hash/command hash；profile-backed CTRF 缺失或畸形仍 fail closed，reward 0
仍是 valid failure。该历史提交当时的全套 136/136 tests 通过。最终 clean root `gptpro03` 的 lock 绑定
`18ff3417` 且 claim eligible，prewarm 再次为 86/86、0 failed、无 online build。
真实 run 中两项 temperature receipt 均成为 `pytest_ctrf`、`test_count=7`、valid=true，
证明修复生效而未把失败改成成功。

`gptpro03` 最终 38/38 train 返回：37 valid、9 success、1 invalid；provider error 为 0。
唯一 invalid 是 `video-object-counting-1` 的真实 hard-cap：TX 66.5 MB、RX 4.6 MB、总计
71.1 MB，超过冻结 67,108,864-byte limit，容器被监控终止且禁止 retry。因
`all_valid_before_proposal_v1`，proposal/generation/validation/no-recursive/promotion 均为 0，
report/archive 未生成；sealed/test content 仍未访问，见
[`gptpro03 development events`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini_gptpro03/development_recursive.events.jsonl)。

该 71.1 MB 全部发生在只允许模型端点的 egress 域内。trace 只有 23 次本地命令和 10 次
agent message，可见 shell 输出合计约 13 KB，没有 web/image、包安装或视频上传命令；
15.4 MB 本地视频只被 ffmpeg/OpenCV 读取。同族 `video-object-counting-3` 用 16 次命令、
约 8.79 MB 即完成。因此最可信解释不是依赖缺失或数据外泄，而是 Responses 多轮工具
往返反复发送累计 context/envelope，叠加数次无效能力探测和调试循环，造成 TX 放大。

据此，**v3.2 也正式判为 execution-infeasible**。本项目不会第三次抬 cap、删除该题、
降低验收标准或重跑到碰巧通过。本地 Codex 0.144.1 审计确认请求压缩已经默认开启；
`previous_response_id` 没有公开 ConfigToml 开关，固定 mini route 的最小
`/responses/compact` canary 又返回 503 `model_not_found`。可用且必须 protocol-owned 的
节流手段是 `model_reasoning_effort=low`、更早的本地 auto-compaction，以及低 verbosity；
tool-output cap 对本题约 13 KB 可见输出预计是低杠杆。

下一工作流只允许改进 model transport/trajectory 效率：统一冻结上述配置，并把视频类
轨迹收敛为“一次能力探测、一个整合脚本、一次最终校验”；先在 train-only 长轨迹
canary 上证明仍低于现有 64 MiB，再冻结一次新的 execution-policy revision。
promotion/subset/evaluator/cap 不变，也不拼接任何失败 run 的有效 observation。

该工作流已由 v3.3 完整执行。提交 `e0b1a33b` 新增独立
`codex_low_reasoning_early_local_compaction_v1` policy；v3.1/v3.2 仍解析为旧 catalog-default
treatment，不会被静默套用新配置。v3.3 与 v3.2 删除 protocol ID/version 和这一 policy
字段后逐项相同：64 MiB、86-item subset、4 workers、search/promotion/evaluator/sealed
合同均未改变。该 v3.3 提交当时的全套 150/150 tests 及 Codex 0.144.1 `--strict-config` 的断网解析通过。
claim lock 绑定 clean commit `e0b1a33b`、policy hash 和 67,108,864-byte cap，
`claim_eligible=true`；cache-only prewarm 为 86/86、47 images、7 verifier runtimes，
`online_build_attempted=false`。其中历史字段 `test_content_accessed=false` 表示未执行/评分
test split，也未向模型暴露 test bytes；prewarm 的 infrastructure 路径实际会读取并哈希
test task/image/verifier 文件，v3.4 receipt 已改为显式记录这一区别。

train-only `video-object-counting-1` canary 先得到 valid task failure：总流量 1.47 MB、
`error_type=null`、本地 `common-pytest-ctrf-py312-v1` verifier、0 provider/cap/sealed event。
随后同一 lock/prewarm 的 full development 返回 38/38 train：37 valid、9 success、1 invalid；
38 个 verifier execution receipt 全部 valid，37 个 model-only audit valid、1 个 violated；
38 个 network monitor 均 finalized，最大为 `temperature-simulation-3` 的 40.6 MB，
`video-object-counting-1` 为 19.69 MB，provider/circuit 和 hard-cap error 都为 0。因此
**本次 v3.3 batch 已排除 fuse 作为未进入 proposal 的直接原因**。这不等于跨运行稳定性
已经成立：同一 video task 在 canary/full 中为 1.47/19.69 MB，后续仍须报告这一波动。

唯一 invalid 是 `offer-letter-generator-1`。其 trace 明确包含
`item.started(type=web_search)` 和 `item.completed(type=web_search, query=placeholder)`；
auditor 记录 `remote_tool_call_count=1` 并正确产生
`model_remote_tool_policy_violation`。该 trial 已同时使用 `--ignore-user-config`、
`tools.web_search=false`、disabled `standalone_web_search` 和禁止联网工具的 developer
instruction；容器 egress 又只允许模型端点。故它不是 online evaluator，也不是 benchmark
dependency 下载，而是 Codex/Responses execution boundary 出现了禁止的远程工具 item；
现有证据还不能区分该 item 源于 Codex CLI mapping、provider 还是 model。
按现有合同这是不可重试的真实 invalid，不应通过删 trace、白名单 placeholder 或重采样
改写成普通任务失败。`all_valid_before_proposal_v1` 随即阻止 residual/proposal；0 proposal、
0 counterfactual、0 sealed event，recursive/no-recursive report/archive 均未生成。

本轮还暴露一个与评分 gate 无关的执行预算缺口：`max_steps=100` 目前只参与 request/backend
锁定与一致性校验，上游 Codex run template 没有把它传给 `codex exec` 作为可执行限制。
有效的 `temperature-simulation-3` 运行 2,485 秒、累计约 334 万 token，并产生 241 行 JSONL；
其中只有 93 行是 `item.started`，且它们不能直接等同于 protocol 的 semantic step 或 turn，
所以本 run **不能**证明越过了 100-step/turn cap。它证明的是该 cap 在当前执行链中没有
可审计的 enforcement。该结果本身不授权直接启动新 paper run 或添加 ad-hoc gate；若要
声称 budget-matched，必须先定义并离线验证一个可观测的 action budget。以下 v3.4
revision 正是在完成该非评分诊断后建立，而不是对 v3.3 invalid 的重试。

#### v3.4 零模型定位与最小执行修复

上述两个来源不明的问题现已在不调用模型、不评分、不访问 sealed 的条件下定位并实现
最小修复。

首先，精确审阅 Codex 0.144.1 源码与 loopback 出站请求后确认：旧配置
`tools.web_search=false` 是兼容性 no-op；布尔 `false` 被解析成空值，随后未设置的顶层
`web_search` 回落到 `cached`。自定义 Responses provider 支持 hosted web search，因此
v3.3 的 38 个请求都曾把该工具暴露给模型，不能把其中 37 个“未调用” observation 复用为
合规证据。根因在本地 CLI 配置语义，不是 Ruoli 注入，也无需换 provider、升级 Codex、
重试样本或扩充评分 gate。

v3.4 只合并两项 execution-treatment 修复：使用顶层 `web_search="disabled"`，以及把
`max_steps` 落为 `codex_action_start_v1`。独立的
[`codex_model_only_wire_probe_v1`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/diagnostics/codex_model_only_wire.json)
直接复用实际 provider argv，并用本地 canned Responses server 做阴性/阳性对照：

- canonical v3.4：`POST /v1/responses` 一次，`stream=true`，7 个工具，0 个
  `web_search*`/`web.run`，Codex 正常产生 `turn.completed`；
- stale boolean 对照：同一配置只把 canonical 顶层键换回 `tools.web_search=false`，8 个
  工具中出现 `web_search` 且 `external_web_access=true`；
- 两次请求都只到 loopback，模型推理与评分均为 0；raw request、instruction、Authorization
  和 raw trace 均不落盘。

预算单位不是模糊的 turn：每一条可解析的 `item.started` 都计数，缺失 item/id/type 的
malformed start 也保守占一单位并使 evidence invalid，不能靠畸形事件绕过上限。容器内
Node supervisor 为 Codex 建独立进程组；第 N 个 start 先发 TERM，15 秒为 KILL 的上限。
它直接持有并截断当前 attempt 的 trace，写入随机 nonce，并在 verifier 开始前扫描专用 trial
容器的 `/proc/<tgid>/task/<tid>`，清除基线之后新增的全部 live task；因此 `setsid`/new
session，以及 thread-group leader 已为 zombie 但 worker 仍存活的情况也不能逃逸。任何无法
完成的 task scan 都 fail closed。receipt
绑定 supervisor hash、nonce、full trace hash、action projection hash、limit、实际 steps、
spawn/exit/signal、post-trigger count、严格 token usage、process-group 与 container-scope
confirmation。离线注入已覆盖正常完成、无/空 usage 的 `turn.completed`、exit 42、spawn 失败、
同 chunk N+1、旧 receipt 替换当前 trace、background descendant、`setsid` escape、zombie
leader + live pthread worker，以及忽略 TERM 后的 KILL。自然完成若遗留 task 会清理并
invalid；受控 budget truncation 在所有 live task 清空后可保持 valid。完整 trace 中较晚的
429/鉴权/限额等 fatal provider 事件优先于较早的 generic stream error，避免被误分为普通
receipt invalid 并继续请求 provider。

截断通常没有完整 token usage。为避免 promotion cost ratio 把一臂的 token 与另一臂的
action count 静默混算，v3.4 明确对所有 arms 统一使用 action starts 作为 promotion cost；
token usage 继续作为二级报告指标，并逐 trial 持久化 completeness/truncation。64 MiB cap、
subset、workers、retries、evaluator、promotion、statistics、recursive/no-recursive 定义和
sealed policy 均不变。当前剩余执行顺序只有：clean commit/lock、新 runtime cache 与 86/86
cache-only prewarm、一个 `max_steps=2` 的 non-claim/train-only canary（只用本地 offline
verifier 评分；在线仅 Ruoli 模型推理），然后才允许 fresh-root development。

### 7.3 当前 infrastructure/protocol 状态

全 inventory 的
[`offline verifier coverage audit`](skilllearn_offline_verifier_matrix.md)
给出 credential-independent 任务 **86/95 可运行、9 项 blocked**。本次没有继续追逐
大体积、异构依赖，而是在任何新模型调用前冻结了保留原 split assignment 的 86-item
offline-ready subset：

- instance holdout：38 train / 16 validation / 32 sealed test，16 families；
- family out：48 train / 11 validation / 27 sealed test，9/2/5 families；
- 排除 3 个完整 infrastructure-blocked families、缺权威 verifier 的 GDP item 2，以及
  原先需要 `GH_TOKEN` 的 family；
- 不复用相邻 GDP item 的 oracle，也不把 online evaluator 当作替代品。

新的
[`offline86 verifier matrix`](../artifacts/offline_verifier_matrix_offline86_20260711_v1/matrix.json)
实际得到：7/7 active profiles、15/15 train-family representatives、
`blockers=[]`、`manifest_execution_ready=true`、`passed=true`。随后完整本地 preflight
同样得到 `blockers=[]`、`selected_item_count=86`、`ready_for_live_skill_generation=true`。
两次检查均未执行模型，sealed-test 语义也未暴露给模型。该结果另有受版本控制的精简
[`offline readiness receipt`](../manifests/skilllearn_offline_readiness_receipt_v1.json)，
供 protocol/lock 绑定；不再把 `.gitignore` 下的 matrix artifact 当作唯一证据入口。

这里必须区分三层证据。readiness receipt 绑定的是 7/7 profile contract、15/15 train-family
动态代表探针和 86-item 静态 preflight；它不声称逐项执行了 86 个 verifier。独立的
all-manifest runtime prewarm 才覆盖 train、validation、sealed-test 的全部 86 个 image/runtime。
本次第一次 cache-only 检查暴露 14 个未建镜像；它们在独立准备阶段有界构建后，第二次
cache-only 验收为 **86/86 passed、0 failed、47 个唯一镜像、7 个离线 verifier runtime**，
且最终 receipt 记录 `online_build_attempted=false`。这仍只是零模型、零 sealed scoring 的
基础设施证据，不是 86 项任务准确率。

9 项未进入主协议的原因已经分型，而不是统称“缺缓存”：GDP item 2 在当前官方主分支
仍缺权威 `test_outputs.py` 和 solution；Druid 已有零下载 direct-`javac` 参考 patch 路线，
但缺 vulnerable negative control 与 arbitrary-edit generality；Scala 还需要固定 SBT/Maven
闭包和 CLI verifier adapter；NLP 则需要 Python 3.10 CPU runtime 与约 0.5--1.2 GB 的
最小 ML closure。它们是后续独立 infrastructure workstream，不再阻塞主学习实验。

旧 development lock 仍声明 `network_scope_audit=v1`，v3.1 已升级为 hard-egress v2、
offline-verifier v3、32 MiB/题 hard fuse、prompt-action lowering v1 和 protocol-owned
promotion v2；v3.2 只把同一 fuse 版本化为 64 MiB。旧 live 与 v3.1 observations 因此
都只能作诊断，不能与 v3.2 直接合并。

最新 `gptpro03` protocol lock 绑定 clean commit `18ff3417`、`validation_issues=[]`，对应
prewarm 为 86/86。该 run 的 37 条 valid train observation 与 1 条 hard-cap invalid 也不能
被事后修补或跨进程 replay；它只证明 offline receipt 修复和 provider 稳定性，同时否证
当前 64 MiB execution contract 对所有 train trajectory 的可行性。

sealed test 仍未访问，这是正确状态。

## 八、v2 当前最关键的架构缺口

### 8.1 已关闭的 P0：promotion 标准所有权

**[CODE + TEST]** `PromotionGateSpec` 现在是唯一的 evaluator-owned contract。pairs、
confidence、net gain、activation、minimum effect LCB、maximum harm 和 maximum cost
全部由 [`PaperProtocol`](../assumption_agent/benchmarks/paper_protocol.py) 严格解析；实验 CLI
已移除 `--minimum-pairs` 旁路，recursive/no-recursive 两臂共享同一个 immutable spec，
protocol lock 和 freeze report 都复核完整 promotion mapping。

candidate 的 `ExpectedEffect` 仍可表达更保守的自我约束，但 effective threshold 只能收紧：

```text
effective_min_delta = max(protocol_min_delta, candidate_min_delta)
effective_max_harm = min(protocol_max_harm, candidate_max_harm)
effective_max_cost = min(protocol_max_cost, candidate_max_cost)
```

新增对抗测试证明 `minimum_delta=-1`、`maximum_harm_rate=1`、超大 cost ratio 都不能放宽
protocol 及格线；更严格 candidate 则会生效。promotion decision/event 同时记录 protocol、
candidate 和 effective thresholds。这是一次收敛现有判断所有权的修复，不是新增 gate。

### 8.2 已收紧的 P0：外部 action/verifier/fallback contract

内部 `PolicyRuntime` 的 typed lane action 仍保持 effectful。外部 SkillLearn backend 则不再
冒充同等语义：compiler 只接受 `execute_step`、`produce_artifact`、`request_evidence` 三类
`prompt_directive` 和 `check_condition` 这一类 `agent_local_self_check`。`enable_lane`、
`disable_lane`、`prioritize_lane`、`set_parameter`、`require_verifier`、`abstain` 没有外部
lowering，因而 fail closed；proposal/repair capabilities 也只广告这四类。

编译后的 `SKILL.md` 不再暴露 benchmark external verifier、policy-off/on evidence 或
expected-effect 阈值，并明确 external verifier 只在 agent 退出后运行。compile manifest
绑定完整 program set、每项 lowered treatment、实际 `SKILL.md` content hash、
`external_verifier_exposed_to_agent=false` 和 `baseline_on_nonactivation_only_v1`。输出目录
由 staging tree 整体替换，旧 skill 不会残留；action target/value 引用 hidden verifier、
required evidence 或 policy-off/on 时在 validator/compiler 共用的结构检查中 fail closed。

novelty、counterfactual replay、training replay 和 proposal prior context 现在统一使用实际
lowered-treatment identity，而不是 raw program metadata。只改 expected-effect/verifier 元数据
不能获得新行为身份或重新采样；真正改变注入内容才会改变 treatment hash。generation report
同时绑定实际评价的 candidate treatment；freeze 使用 archive program 重算该 hash，并用与
runtime gate 相同的 summary-blocker 函数重算 promotion，防止替换 treatment 或伪造 allowed。

fallback 的伪证据链也已删除：activated candidate 是独立 treatment，
`baseline_preserved=false`；只有 trigger miss 时 candidate observation 直接 alias baseline，
才记为 observed baseline。SkillLearn promotion 依据冻结的 paired harm/LCB/cost contract，
不再要求一个由字符串声明伪造的 post-verifier rollback。

这没有把 prompt directive 变成 typed program；它做的是把宏观“注入 skill”与细粒度
agent instruction 的证据层级说清楚。真正强类型外部 operator 仍可作为后续研究方向，
但不再是当前论文协议的隐含主张。

### 8.3 P1：proposal selection 仍存在 failure-only support bias

v2 promotion 已不再按错误频率；但
[`SkillLearnResidualMiner`](../assumption_agent/benchmarks/skilllearn_lifecycle.py)
会跳过所有成功 train rows，只把失败变成 residual。same-generation candidate selection
又在 [`evolution.py`](../assumption_agent/evolution.py)
优先最大化失败 residual 的 trigger support。

因此系统没有用成功样本估计：

- trigger false-positive rate；
- anti-trigger precision；
- train-side potential harm；
- “不应激活”的对照域。

广触发器天然容易得到更高 support。这是与 legacy frequency bias 相关、但不完全等价
的上游选择偏差；当前证据只能证明 failure-only support bias 仍存在，不能证明旧偏差
被原样“移动”到了上游。

### 8.4 P1：prospective runtime features 仍过粗

当前 SkillLearn feature catalog 主要只有
[`family/category/difficulty/tags/environment_file_count/has_container_environment`](../assumption_agent/benchmarks/skilllearnbench.py)。
完整 train instruction 可以帮助设计 action，但被正确禁止作为 runtime trigger。

这比 legacy hash-only data 有进步，却仍缺：

- artifact/output schema；
- constraint signature；
- required capability；
- verification class；
- input modality与工具需求；
- 可在未见 family 上计算的语义特征。

当前很多 candidate 因此只能形成宽泛 family/env-count gate，不足以支撑可信 family-out
prospective routing。

### 8.5 P2：archive 是树形数据结构，不是多 clade 搜索算法

[`ArchiveNode`](../assumption_agent/archive.py#L23-L31) 有 parent、active programs、epoch
和 generation；但 [`PolicyArchive`](../assumption_agent/archive.py#L96-L102) 只有一个
`incumbent_id`。每代只把 train-only 排名第一的 `eligible[0]` 送入 validation，下一节点
只从当前 incumbent 扩展：
[`evolution.py:L325-L453`](../assumption_agent/evolution.py#L325-L453)。

因此当前算法仍是保守的单 incumbent hill climbing，而不是 RQGM 中多 clade 并行保留、
按 metaproductivity 继续扩展的 archive search。

此外，`ScoreRecord` 只存 candidate successes/total 和 item-set hash，未直接绑定完整 pair
bundle、gain/harm/cost、promotion decision 与 protocol hash。archive 的 provenance 还不够
承担跨 epoch、多分支重排。

### 8.6 P2：evaluator co-evolution 还是独立骨架

`EvaluatorEpochController`、anchor lower bound 和 selective invalidation 有代码与测试；
但主 SkillLearn 实验明确只允许 task/policy hypothesis，evaluator hypothesis 不能编译为
agent skill。当前没有真实 evaluator challenger、epoch transition 或 incumbent re-ranking
artifact。

所以 v2 可以声称“有 evaluator-epoch mechanism skeleton”，不能声称“已经实现 Red Queen
式 agent/evaluator co-evolution”。

### 8.7 P1：递归修复被触发过，但没有因果收益证据

v2 recursive validation 主要修复 schema、trigger support、action vocabulary 和 epoch 等
静态/训练检查。已有 live run 观察到 repair child 被提出和选中，但尚无完整、clean、
behavior-different 的 recursive/no-recursive held-out comparison 显示 repair 带来净收益。

因此当前可说“递归修复机制会运行”，不能说“递归验证已经改善性能”。

### 8.8 文档与协议漂移正在本次收口

[`ARCHITECTURE.md`](../ARCHITECTURE.md) 和
[`BENCHMARK_PROTOCOL.md`](../BENCHMARK_PROTOCOL.md) 此前曾有段落声称 destination allowlist /
dependency-cache-only 尚未强制；但当前
[`docker_egress.py`](../assumption_agent/benchmarks/docker_egress.py) 和 protocol manifest 已
实现 provider-only hard egress、offline package mode 与 network fuse。本次已同步主
README、benchmark protocol、offline-verifier matrix 和 status 摘要；本轮又把 receipt
runtime provenance、v3.4 execution-policy binding 与 test 状态更新为 183/183。历史段落仍
保留为 diagnostic ledger，不能当作当前协议。

这种文档漂移本身会破坏 protocol review；重新跑论文实验前必须同步。

## 九、下一步优先级与硬验收标准

| 优先级 | 工作 | 硬验收标准 |
|---|---|---|
| 完成 | 冻结 evaluator-owned promotion policy | 已由 protocol 绑定完整 spec；candidate 只能收紧；对抗测试通过 |
| 完成 | 收紧外部 action/fallback contract | 4 类 prompt/self-check lowering；6 类 unsupported op fail closed；observed fallback 不再由字符串伪造 |
| 完成 | 冻结 offline-ready 范围 | 86-item manifests 保留旧 split；readiness matrix/static preflight 均 `blockers=[]`，无模型调用 |
| 完成（v3.3 历史） | 全 manifest runtime prewarm | cache-only 86/86、47 images、7 verifier runtimes；无 agent、无 sealed scoring；不作为 v3.4 receipt |
| 待完成（v3.4 active） | clean commit、lock 与 v4 prewarm | scoped Git clean；claim-eligible lock 无 validation issue；新 supervisor/runtime 的 post-commit prewarm 86/86 |
| 本次排除（非稳定性结论） | 64 MiB fuse 作为本 batch 的直接 blocker | v3.3 38/38 均低于 64 MiB；最大 40.6 MB，video-1 为 19.69 MB；0 cap/provider error。canary/full 波动为 1.47/19.69 MB，尚无跨运行稳定性证据 |
| 完成（零模型） | 定位 model-only execution boundary | 根因为 Codex 0.144.1 丢弃 `tools.web_search=false`；canonical 顶层 disabled 的 loopback 为 7 tools / 0 web，旧键阳性对照为 8 tools / 1 hosted web；未调用模型、未评分 |
| 完成（实现与离线注入） | 执行 action budget | `codex_action_start_v1` 在第 N 个 `item.started` 终止 PGID，并按 task/TID 清除 dedicated-container 基线后的 live task；异常退出、malformed、N+1、`setsid`、zombie leader/live worker、残留 descendant 与 evidence tamper 均 fail closed；所有 arms 统一 action-step cost，不再混合 token/step |
| P1（执行阻塞） | v3.4 clean runtime canary 与 fresh development | 先完成 clean lock、new-runtime 86/86 cache-only prewarm、`max_steps=2` non-claim/train-only/local-offline-verifier canary；通过后只启动一次 fresh develop：同一 invocation 的 38/38 all-valid train 才解锁 proposal 和 paired validation，生命周期末端必须产出 recursive/no-recursive 两份 report/archive、0 invalid、sealed evaluation=false；不复用 v3.3 的 37 条 observation |
| P1 | 递归因果归因 | 两臂共享 train evidence 和 roots，唯一差异是 repair；behavior-identical 时 effect 报 N/A，不重采样 |
| P1 | contrastive trigger learning | train successes 进入 anti-trigger/precision；candidate selection 不只最大化 failure support；报告 activation precision、harm、abstention |
| P1 | prospective family-out routing | trigger 不依赖已知 family 或预编译 item ID，只使用冻结、无 gold、运行时可得语义特征 |
| P2 | 多 clade archive | 同 epoch 至少两个 clade 可继续扩展；node 绑定 protocol/evidence/promotion hashes，并报告 retention 与 branch productivity |
| P2 | evaluator co-evolution | 独立 anchor challenger、epoch transition、selective invalidation 和旧 incumbent re-evaluation 实际执行后再作主张 |

近期顺序应是：

1. 已完成：审阅并提交 protocol/action/subset 改动以及 3 个新 manifest/receipt 文件；
2. 已完成：在 clean scoped commit 上重建 claim-eligible lock 和 86-item content-hashed prewarm receipt；
3. 已执行但未形成性能证据：第一次 full development 在 26 个有效 train observation 后，
   被 provider 429/circuit 与一个既有 hard-byte fuse fail-closed；未进入 proposal/validation；
4. 已完成：一个单题、5-step、非 claim transport canary 得到有效 offline-verifier
   observation，确认 provider 已从 429 恢复；
5. 已完成：同协议 fresh-root rerun 再次在 `court-form-filling-6` 超过 32 MiB；按 stop
   rule 中止，v3.1 判 execution-infeasible；
6. 已完成设计：新建 v3.2，仅把统一 fuse 一次性版本化为 64 MiB，其余实验合同不变；
7. 已完成：v3.2 clean lock/prewarm 均通过，64 MiB 未触发；full run 在 8 个有效 train
   observation 后被 provider 的“无可用 distributor channel”503/429 熔断；
8. 已完成：新 GPT Pro credential 的 API/Codex canary 证明同一路由恢复；`gptpro01`
   跑完 38 train，暴露两个 deterministic receipt false-negative，而非 provider/cap 问题；
9. 已完成：receipt auditor 改为绑定实际 runtime profile/command，136/136 tests 通过；
   `gptpro03` 中两项 temperature 均以 7-test CTRF valid failure 完成；
10. 已完成：`gptpro03` 跑完 38 train，但 `video-object-counting-1` 以 71.1 MB 超过
    冻结 64 MiB；37 valid / 1 invalid，proposal 被 fail-closed，v3.2 判 execution-infeasible；
11. 已完成：v3.3 只版本化 Codex execution treatment，v3.1/v3.2 保持旧 mapping；
    150/150 tests、strict-config、claim lock 与 86/86 prewarm 通过；
12. 已完成：video-1 canary 为 1.47 MB valid failure；full run 中 video-1 为 19.69 MB，
    38 个 trial 最大 40.6 MB、0 cap/provider，本次排除 fuse 作为直接 blocker，但尚未证明
    跨运行稳定性；
13. 已执行并 fail-closed：full train 为 37 valid / 1 `web_search` policy invalid；
    proposal/counterfactual/sealed 均为 0，四份 report/archive 未生成。不得重试或通过新 gate
    洗掉 invalid；v3.3 已冻结为不可复用的诊断证据；
    v3.1–v3.3 仅作为 immutable evidence，当前代码仍可按其声明的 legacy schema 验证历史
    receipt，但不承诺这些协议在当前 commit backward-executable；
14. 已完成零模型定位：Codex 0.144.1 把旧 boolean key 丢弃并默认暴露 cached hosted
    search；canonical 顶层 disabled 的真实 wire 捕获无 web，旧键阳性对照稳定检出 web；
15. 已完成 v3.4 最小实现与离线注入：同一 execution policy 同时冻结 model-only tool
    exposure、可执行 action budget、dedicated-container task/TID 清理 receipt、token completeness 和统一 action-step
    promotion cost；未改变 cap/subset/evaluator/promotion/sealed；
16. 待完成：clean commit/lock、新 shared runtime 与 86/86 cache-only prewarm；随后只做一个
    `max_steps=2` non-claim、train-only、仅本地 offline verifier 评分的 canary。二者通过后才启动
    一次 fresh-root development；
17. 若 clean development 没有 promotion，直接转 contrastive trigger learning，不先扩
   family-out、multi-clade 或 evaluator mutation；
18. 有 retained validation gain 后再做 family-out，最后才增加多 clade 与 evaluator mutation。

这比立刻扩展 archive 或继续补 HLE source span 更能降低研究风险。

## 十、建议的实验协议与 claim ladder

### 10.1 主 benchmark

当前最合适的主战场仍是 SkillLearnBench：

- instance holdout：38 train / 16 validation / 32 sealed test；
- family out：48 train / 11 validation / 27 sealed test；
- HLE：只作冻结的 external transfer/stress slice。

如果选择 86-item infrastructure subset，必须在任何模型调用前冻结新 manifests、重新计算
split counts，并说明 9 项排除只由 verifier/toolchain availability 决定，而非 task outcome。

### 10.2 必要 controls

最低矩阵应包含：

```text
raw_no_skill
static_generic_v2              # fixed, no learning
v2_no_recursive_repair         # same evolution budget, repair disabled
promoted_v2                    # frozen incumbent from recursive loop
skilllearn_b1_sonnet           # upstream static reference
human_authored                 # upper reference, not budget matched
```

raw、static、no-recursive 与 promoted arms 必须共享 model、provider policy、step budget、
runtime、verifier isolation 和 invalid-row policy。外部参考与 human upper reference 不应被
伪装成 budget-matched primary controls。

### 10.3 两个不同的因果问题

1. **same-item paired off/on**：相同 item、runtime 和 evaluator 下，candidate treatment
   是否造成 gain/harm；
2. **prospective transfer**：在未见 instance/family 上，router 是否在看不到 outcome 时
   正确激活，并保持净收益。

第 1 个回答局部因果 effect；第 2 个回答假设是否可复用。只做第 1 个不能证明 continual
learning，只做第 2 个而没有 matched controls 又无法归因。

### 10.4 预注册指标

- task success / executable reward；
- gain、harm、net gain 与 exact McNemar；
- effect LCB 与 item-clustered interval；
- prospective activation rate、precision 与 abstention；
- behavior-changing repair count；
- cost ratio、token、latency 与 model calls；
- invalid/error rate、provider/budget/runtime mismatch；
- archive retention、duplicate rate、forgetting 与 cross-family transfer；
- 多比较 Holm correction 与预注册 early stopping。

“hypothesis proposal precision”不能只按 schema pass 定义；更可操作的定义是：候选先通过
train-only static contract，再在 prospective matched validation 中产生正净效应且不超过
harm/cost gate。activation precision 的分母应是所有实际激活，正例应来自独立 paired
outcome，而不是模型自评。

### 10.5 claim ladder

| 层级 | 可声明内容 | 当前状态 |
|---|---|---|
| L0 wiring | schema、repair、off/on、guard、archive transition 的机械链路已连接 | 达到：183 tests、protocol ownership、backend lowering、runtime receipt provenance 与 offline preflight 均通过 |
| L1 mechanism live | 真实外部任务中 proposal/repair/treatment/gate 全链路完成 | 部分达到 |
| L2 validation learning | clean held-out validation 上有可晋级净收益 | 未达到 |
| L3 prospective generalization | frozen incumbent 在 unseen instance/family 上保持收益 | 未达到 |
| L4 self-evolution | 多代 retained improvement，且 recursion ablation 有因果贡献 | 未达到 |
| L5 evaluator co-evolution | anchor-guided evaluator replacement 与 selective erasure 改善搜索 | 未达到 |

## 十一、什么才算“真正自我提出并递归验证假设”

以下条件需要同时满足：

1. 候选不是人工预写的唯一答案，而是系统只从 train evidence 提出；
2. candidate selection 同时利用失败与成功对照，不能只奖励 failure support；
3. 假设被编译为当前 backend 能强制或明确审计的程序；
4. activation 在 outcome 前决定，并实际改变 execution treatment；
5. 同一 item/runtime/evaluator 有 policy-off/on paired counterfactual；
6. promotion gate 完全由冻结 protocol/evaluator 所有，candidate 不能放宽阈值；
7. promotion 不读取 sealed test；
8. 通过的程序进入 archive，并在未来未见题上被 prospective router 调用；
9. 失败程序能降级、停用或归档；evaluator epoch 改变时只使旧依赖证据失效；
10. recursive repair 的收益用共享 root/evidence 的 no-recursive arm 做因果消融；
11. 最终提升能归因到该程序，而不是额外预算、fallback、provider 或重采样；
12. 至少一次真实 promotion 改变下一代 incumbent，并在后续任务上保留净收益。

在满足这些条件前，“自我提出并递归验证”仍应被称为研究机制或 harness，而不是已经
证实的 self-evolving capability。

## 十二、最终结论

旧 Assumption Agent 的主要问题不是“没有足够多假设”，而是假设没有稳定编译成可
执行、可路由、可反事实验证并可跨题保留的 policy。legacy 优化的是“怎样更复杂地
回答 HLE”，而不是“哪些假设值得在未来任务中保留，以及它们是否因果性地改善行为”。

`reconstruction_v2` 已经完成了重要转向：它把三层 hypothesis、paired evaluation、split
guard、archive 和 evaluator epoch 做成了清晰的小型系统。这使研究问题第一次真正可
证伪，也比继续给 legacy HLE monolith 加规则更有价值。

本次已经关闭四个会让后续结果先天不可解释的 P0：candidate 不能控制 promotion
及格线；外部 backend 不再把 prompt/verifier/fallback 声明伪装成 typed/observed 事实；
86-item offline-ready manifests 通过 readiness，且 v3.3 历史 prewarm 为 86/86（v3.4 v4
receipt 仍待生成）；verifier receipt 现在绑定 proxy
实际执行的 runtime profile/command，完整 CTRF 的任务失败不再被误标成 infrastructure
failure。新 GPT Pro route 证明 Ruoli 模型调用在该 v3.3 batch 中可用，但不能外推持续稳定；离线 evaluator 从未需要
替换为 online evaluator。

v3.3 在本次 batch 中排除了原先的 64 MiB 直接阻塞：38/38 train 的最大流量为 40.6 MB，
`video-object-counting-1` 从 71.1 MB invalid 降为 19.69 MB valid；没有 provider 或 hard-cap
error。代价不是修改 evaluator、子集或 promotion gate，而是把低 reasoning / verbosity
与更早的本地 history compaction 作为 protocol-owned agent treatment。canary/full 的
1.47/19.69 MB 差异表明重复稳定性仍未建立，不能把单次 batch 外推成稳定完成。

最近的两个 execution blocker 已在零模型层定位并修复。`web_search` 不是 provider/model
注入，而是 Codex 0.144.1 对旧 boolean config 的兼容性 no-op；canonical 顶层 disabled 已由
真实 wire 阴性/阳性对照确认。nominal `max_steps` 也不再被称为 semantic turn，而是冻结为
可流式观察的 `codex_action_start_v1`，由容器内 supervisor 终止 PGID、清理 dedicated-container
基线后的全部 live task 并生成 receipt。
这不是通过重试或放松 auditor 把 v3.3 invalid 洗成 performance evidence；v3.3 的 37 条
valid observation 仍全部不可复用，sealed 仍未触碰。

因此当前距离目标的第一段只剩执行验证，而不是继续补 gate：v3.4 clean lock、新 runtime
cache、86/86 cache-only prewarm 和一个 `max_steps=2` 的 non-claim、train-only、仅本地
offline verifier 评分的 canary。只有 canary
同时证明 actual wire 无 web、budget receipt valid、全部 agent task 已退出且本地 verifier 继续执行，
才从 fresh root 启动一次 develop；同一 invocation 的 38/38 all-valid train 会解锁 proposal、
paired validation 与 promotion，recursive/no-recursive 两份 report/archive 是该生命周期的
末端产物。之后才有资格做 family-out、
sealed test、多 clade 和 evaluator co-evolution。本次结果不支持再次抬 cap，也不支持声称
transport 已跨运行稳定；任何失败 run 的 valid observation 都不能跨进程拼接。最诚实的
论文级表述是：

> **显式 HypothesisProgram 是一个有希望、可能更易归因的 self-evolution 搜索表示；
> v2 已证明协议所有权、离线 evaluator 和学习环 wiring 可运行，但尚未证明它在冻结、
> 干净的外部 benchmark 上产生稳定净收益，更未证明 Red Queen 式多谱系或 evaluator
> 共演化。**

## 附录 A：关键证据索引

- legacy 代码：[`assumption_os/`](../../assumption_os/)
- legacy 自我演化评估：
  [`codex_gpt_advice_assessment_20260707.md`](../../reconstruction/md/codex_gpt_advice_assessment_20260707.md)
- self-evolution bundle：
  [`reference/self_evo_continual_20260707/`](../reference/self_evo_continual_20260707/)
- RQGM PDF：
  [`The Red Queen Gödel Machine`](<../reference/The Red Queen Gödel Machine Co-Evolving Agents and Their Evaluators.pdf>)
- v2 architecture：[`ARCHITECTURE.md`](../ARCHITECTURE.md)
- v2 benchmark protocol：[`BENCHMARK_PROTOCOL.md`](../BENCHMARK_PROTOCOL.md)
- v2 current status：[`STATUS.md`](../STATUS.md)
- active paper protocol：
  [`skilllearn_paper_protocol_v3_4_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_4_ruoli_gpt54mini.json)
- immutable v3.3 execution diagnostic protocol：
  [`skilllearn_paper_protocol_v3_3_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_3_ruoli_gpt54mini.json)
- immutable v3.2 diagnostic protocol：
  [`skilllearn_paper_protocol_v3_2_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_2_ruoli_gpt54mini.json)
- immutable v3.1 diagnostic protocol：
  [`skilllearn_paper_protocol_v3_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_ruoli_gpt54mini.json)
- frozen offline-ready manifests：
  [`instance holdout`](../manifests/skilllearnbench_instance_holdout_offline_ready_v1.json)；
  [`family out`](../manifests/skilllearnbench_family_out_offline_ready_v1.json)
- version-controlled readiness evidence：
  [`skilllearn_offline_readiness_receipt_v1.json`](../manifests/skilllearn_offline_readiness_receipt_v1.json)
- local ignored diagnostics（非 clone 中的主证据）：
  [`offline verifier matrix`](../artifacts/offline_verifier_matrix_offline86_20260711_v1/matrix.json)；
  [`86-item runtime prewarm receipt`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/development_prewarm.json)；
  [`mechanism smoke`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/smoke_recursive.report.json)；
  [`full-development fail-closed events`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/development_recursive.events.jsonl)；
  [`v3.1 clean-rerun cap recurrence`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini_rerun01/development_recursive.events.jsonl)；
  [`v3.2 claim lock`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini/protocol_lock.json)；
  [`v3.2 86-item prewarm`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini/development_prewarm.json)；
  [`v3.2 provider-capacity failure`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini/development_recursive.events.jsonl)；
  [`GPT Pro Codex canary`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini/gpt_pro_transport_canary/report.json)；
  [`gptpro01 receipt false-negative run`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini_gptpro01/development_recursive.events.jsonl)；
  [`gptpro03 final 64 MiB hard-cap run`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini_gptpro03/development_recursive.events.jsonl)；
  [`video-object-counting-1 failed trace`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini_gptpro03/development_recursive/upstream_trials/no_skill/video-object-counting/video-object-counting-1/v2_policy_off_66599fccf924efd4c6/agent/codex.txt)；
  [`v3.3 claim lock`](../artifacts/paper_primary_v3_3_offline86_ruoli_gpt54mini/protocol_lock.json)；
  [`v3.3 86-item prewarm`](../artifacts/paper_primary_v3_3_offline86_ruoli_gpt54mini/development_prewarm.json)；
  [`v3.3 video-1 canary`](../artifacts/paper_execution_policy_v3_3_video_object_counting_1_canary01/report.json)；
  [`v3.3 full-train events`](../artifacts/paper_primary_v3_3_offline86_ruoli_gpt54mini/development_recursive.events.jsonl)；
  [`forbidden web-search trace`](../artifacts/paper_primary_v3_3_offline86_ruoli_gpt54mini/development_recursive/upstream_trials/no_skill/offer-letter-generator/offer-letter-generator-1/v2_policy_off_a99904ddf5496bed16/agent/codex.txt)；
  [`v3.3 video-1 valid trace`](../artifacts/paper_primary_v3_3_offline86_ruoli_gpt54mini/development_recursive/upstream_trials/no_skill/video-object-counting/video-object-counting-1/v2_policy_off_26f7d1bd8b10776c43/agent/codex.txt)；
  [`long temperature-3 trace`](../artifacts/paper_primary_v3_3_offline86_ruoli_gpt54mini/development_recursive/upstream_trials/no_skill/temperature-simulation/temperature-simulation-3/v2_policy_off_970cfa8b6418033bd2/agent/codex.txt)
  [`v3.4 zero-model wire probe`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/diagnostics/codex_model_only_wire.json)

## 附录 B：复杂度统计口径

legacy 数字按以下口径复核：

```text
lines:
  Python source splitlines

functions:
  AST module.body 中 FunctionDef / AsyncFunctionDef
  nested-inclusive count 使用 ast.walk

HLE configuration surface:
  source 中唯一正则 token HLE_[A-Z0-9_]+

verifier / fallback proxy:
  顶层函数名分别包含 verifier / fallback
```

这些统计用于描述控制面规模，不应被当作独立行为数量或性能指标。
