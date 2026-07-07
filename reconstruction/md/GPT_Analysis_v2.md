我看了你新分支 `codex/operator-fidelity-repair-20260621`。结论先说清楚：

> **这版确实补上了“假设没有被执行，只是当 context”的核心缺口的一部分；但它现在只是在一个很窄的触发子集上有效，本质上还是高精度低覆盖，不是泛化型 operator system。**

这不是坏事。它说明你终于把问题定位对了：不是再加更多哲学/框架，而是让假设变成 **answer-time operator**。但现在的 operator 还太少、太硬编码、触发太窄、验证太表面，所以换题泛化会崩。

---

# 1. 这版真正修了什么？

这版新增了两个关键模块：

```text
assumption_os/operator_specs.py
assumption_os/application_fidelity.py
```

`operator_specs.py` 的目标非常正确：把 graph 中的 assumption claim 编译成 solver prompt 可以执行的形式，也就是 trigger、steps、output slots、negative controls、checks。文件开头已经明确说：graph 存的是 auditable claims，但 solver prompt 需要 procedural shape；实验要测 retrieved assumption 是不是真的变成 executable，而不是仅仅被检索出来。

它定义的 `OperatorSpec` 也正是我前面建议的形式：

```text
source_id
source_type
source_claim
trigger_conditions
execution_steps
required_output_slots
negative_controls
verifier_checks
fallback_policy
confidence
```

`application_fidelity.py` 则做了一个 cheap verifier，用 required slots 是否在最终答案中“物质性出现”来判断 operator 是否真的被执行，目标是减少 retrieved assumptions 变成 decorative context 的问题。

而且你已经把它接进了 `phase2_v20_framework.py`：

```text
Turn 1 prompt 里加入 Operatorized Assumption Constraints
Turn 2 reflect prompt 里也加入 OperatorSpec
如果 application_fidelity 不过，还会做一次 bounded repair pass
```

prompt 里明确写了：OperatorSpec 是 execution constraints，不是 background context；如果触发，最终答案必须体现 required slots，不能只复述 Claim 或原则名称。 Turn 2 也检查了 OperatorSpec required slots 有没有被填，避免假设只是装饰性建议。

这一步是对的。它把系统从：

```text
Assumption as retrieved text
```

推进到了：

```text
Assumption as executable constraint
```

---

# 2. 为什么还是泛化差？

因为当前 operatorization 只是第一层补丁，还没有变成泛化机制。

## 2.1 默认只在 daily_life 启用

这是最直接的原因。`phase2_v20_framework.py` 新增的 CLI 默认是：

```text
--assumption-operator-domains daily_life
```

也就是 OperatorSpec 默认只在 `daily_life` 域启用。

你的 live summary 也确认了这点：

```text
daily_life: enabled, 2 operators each, n=3
business: not selected, 0 operators, n=3
engineering: not selected, 0 operators, n=3
software_engineering: graph skipped, 0 operators, n=3
```

所以当前系统泛化差，部分原因非常简单：**你现在根本没有让 operator 系统在多数 domain 上工作。**

它不是失败于泛化，而是还没进入泛化阶段。

---

## 2.2 n=12 太小，而且是 triggered sample

A/B 的样本只有 12 个：

```text
sample: codex_operator_ab_non_bypass_n12_20260621.json
```

在这个 n=12 上，operator vs ctx-only 一开始其实没有整体赢：

```text
forward: ctxonly 5, operator 5, tie 2
reverse: ctxonly 5, operator 5, tie 2
combined: ctxonly 5, operator 4, mixed_tie 2, tie 1
```

也就是说，**单纯加 OperatorSpec 并没有稳定提升**。真正提升来自后面的 repair 机制和 daily-life selective activation。

---

## 2.3 repair 机制局部有效，但它证明的是“能修槽位”，不是“能泛化”

repair summary 显示 daily-life operator fidelity 很好：

```text
programmatic daily-life operator pass rate: 1.0
fidelity mean: 0.9
decorative use rate: 0.0
repair attempts: 1
daily_life_0216 before 0.675 -> after 1.0
```

pairwise utility 也局部变好：

```text
repair vs selective-live combined: repair 6, selective_live 1, tie 3, mixed_tie 1, direction_conflict 1
repair vs ctx-only combined: repair 6, ctxonly 2, tie 2, mixed_tie 1, direction_conflict 1
```

这说明 repair pass 对“已选中且适合 operator 的小样本”有效。
但这还不是泛化，因为它只是：

```text
如果已选 operator 且 slot 缺失，就让 LLM 补 slot。
```

它没有解决：

```text
新题该选哪个 operator？
operator trigger 是否真的匹配？
这个 operator 是否适合该 domain？
slot 填了是否真的提高答案质量？
```

所以你看到“少数题好，换题不行”很符合这个机制。

---

## 2.4 当前 OperatorSpec 是少数硬编码 heuristic，不是学出来的泛化 operator

`operator_specs.py` 里 `_heuristic_operator` 主要靠关键词决定 operator 类型：

```text
control variable / causal / ablation
incremental / legacy / adapter / mvp
morphism / analogy / framework / generalization
retrieval / evidence
default generic operator
```

这是一版很好的 bootstrap，但它不是泛化模型。它的问题是：

```text
1. 只有 4–5 类 operator。
2. 触发靠关键词，不靠语义结构。
3. required slots 是模板化的。
4. 新 domain 的 operator 只能落到 generic fallback。
5. 多个不同问题可能被强行塞进同一 operator。
```

所以它在 daily_life 这种开放建议类问题上容易见效，但到 business、engineering、software、science 等 domain 就会不稳定。

---

## 2.5 application fidelity verifier 太浅，容易奖励“填关键词”

`application_fidelity.py` 的 programmatic audit 本质是 slot cue matching。比如 `observed_metric` 的 cues 包括“指标、衡量、观察、测量、数据、转化、成本、率、metric”等；只要答案包含这些词，就可能被算作 slot present。

这能检查“有没有填槽位”，但不能检查：

```text
槽位是否正确？
变量是否真的被固定？
metric 是否与问题目标一致？
control 是否真的构成对照？
decision rule 是否可执行？
negative control 是否真的排除了错误方案？
```

所以现在的 fidelity 更像：

```text
surface application fidelity
```

不是：

```text
semantic/operator correctness fidelity
```

它能降低 decorative use，但不保证泛化质量。

---

## 2.6 operator block 太长，容易干扰强模型

A/B summary 里 operator 平均长度很大：

```text
avg_operator_chars_when_on: 2076.75
avg_context_chars: 6649.67
```

这对 GPT-5.5 可能是负担。尤其如果选错 operator，长的 required slots 会把回答拉向不必要结构，导致“过结构化”。你之前 Phase9 broad repair 也遇到过类似现象：强行显式 frame/morphism 会过度结构化答案。

所以泛化差的一个原因是：

```text
operator 有时不是帮助，而是把答案拽进错误模板。
```

---

## 2.7 business 输，可能不是 operator 本身，而是生成噪声/框架不适配

Selective daily-life live summary 里写得很清楚：

```text
daily-life benefits more often than not
business still loses under fresh generation even without operators
remaining non-daily variance is mostly answer-generation noise rather than operator injection
```

这说明你现在不是单纯“operator 泛化失败”，而是：

```text
某些 domain 的基础生成链路本身不稳；
operator 没启用也输；
这些 domain 需要专门 operator / prompt / verifier，而不是套 daily-life operator。
```

---

# 3. 当前状态怎么定位？

我会这样定位这版：

```text
Operator Fidelity Repair v0:
  成功把假设从 context 推向 executable constraints；
  在 daily_life triggered subset 上有局部 positive signal；
  repair pass 能提升 slot fidelity 并带来小样本 utility gain；
  但 operator 类型少、触发窄、验证浅、样本小，尚未解决泛化。
```

换句话说：

> **这版不是泛化方案本身，而是证明“operatorization 是正确方向”的局部证据。**

---

# 4. 下一步泛化方向：从 OperatorSpec v0 到 Operator Policy

你现在不应该继续手写更多 prompt。下一步应该做：

```text
Operator Policy Learning
```

也就是让系统学会：

```text
什么题该启用 operator？
启用哪一个 operator？
operator 应该用哪个强度？
如果低置信度，是否 abstain？
operator 执行后是否真的提升？
失败后是 trigger 错、operator 错、slot verifier 错，还是 answer generator 错？
```

我建议分 6 步。

---

## 方向 1：建立 Operator Taxonomy，不要只靠 4 个 heuristic

现在 heuristic operator 太少。你需要一个更系统的 operator library。

初始可以分 12 类：

```text
O1 causal_control_variable
O2 dependency_aware_intervention
O3 incremental_replacement
O4 adapter_boundary_discovery
O5 evidence_grounding
O6 analogy_role_mapping
O7 limiting_case_reduction
O8 negative_control_abstention
O9 multi_objective_tradeoff
O10 stakeholder_constraint_mapping
O11 failure_mode_triage
O12 verification_plan_construction
```

每个 operator 都有：

```text
trigger
anti_trigger
steps
slots
verifier
fallback
domain examples
negative examples
```

当前 `_heuristic_operator` 里的 control-variable、incremental、morphism、retrieval 可以作为 O1/O3/O6/O5 的 seed。

---

## 方向 2：训练或构造 Trigger Classifier，而不是关键词触发

现在 trigger 靠关键词。泛化要改成：

```text
problem -> operator trigger probabilities
```

每个 operator 输出：

```json
{
  "operator_id": "O2_dependency_aware_intervention",
  "p_trigger": 0.83,
  "p_harm": 0.12,
  "reason": "...",
  "evidence": ["problem mentions coupled variables", "multiple interventions"]
}
```

第一版不用训练大模型，可以用 LLM judge + small labeled set。

标注数据来自你已有 A/B：

```text
operator won
operator lost
operator tied
fidelity pass
fidelity fail
domain
problem tags
operator ids
```

尤其要把失败样本标出来：

```text
business_0097: operator lost
engineering_0244: operator lost
business_0192: operator lost
software_engineering_0265: operator lost
software_engineering_0337: operator lost
```

A/B summary 已经给了 per-problem operator wins/losses。

目标不是全启用，而是：

```text
high precision trigger classifier
```

---

## 方向 3：Operator 强度分级，而不是开/关

现在 operator gate 基本是：

```text
enabled / disabled
```

但真正泛化需要 4 个强度：

```text
0. off
1. soft hint
2. required slots
3. strict template
4. repair-enforced template
```

daily_life 可能适合 3/4。
business 可能只适合 1/2。
engineering 可能需要 2，但 slots 要换成工程验收 slots。
software 需要专门 SE operator，不是 generic operator。

策略：

```text
if p_trigger high and p_harm low:
    strict + repair
elif p_trigger moderate:
    soft required slots
elif p_harm high:
    abstain
else:
    no operator
```

---

## 方向 4：把 application fidelity 从 lexical cue 升级为 semantic fidelity

当前 programmatic fidelity 可以保留，但要加一个 LLM/expert/semantic verifier。你的 `operator_fidelity_eval.py` 里其实已经有 LLM fidelity judge prompt：它要求判断 required slots 是否 substantively filled，不给“只是提到 claim 或用 generic words”的答案 credit。

下一步应该把这个 LLM fidelity judge 纳入正式评估，而不只是可选 `--llm-judge`。

需要两个分数：

```text
programmatic_fidelity
semantic_fidelity
```

只有两者都过，才算 operator 真被执行。

尤其对泛化要看：

```text
semantic_fidelity 是否预测 utility improvement？
```

如果 slot 填满但 utility 不涨，说明 operator slot 设计错。

---

## 方向 5：做 Leave-Operator-Family-Out，而不是 n=12 triggered A/B

现在 n=12 太小。下一步需要一个真正的泛化实验：

```text
train/operator development:
  已知 operator families

test:
  unseen problem family
  unseen domain
  unseen operator combinations
```

实验切法：

```text
leave-domain-out
leave-problem-family-out
leave-operator-family-out
leave-source-strategy-out
```

指标：

```text
trigger precision
trigger recall
utility lift on triggered subset
harm on non-trigger subset
semantic fidelity
control loss
abstention correctness
```

目标不是全局平均，而是：

```text
operator policy selected subset 上赢；
non-selected subset 不伤害。
```

---

## 方向 6：失败归因：operator 输时到底输在哪里？

每个 operator-lost case 要分类：

```text
TriggerFalsePositive:
  不该启用 operator。

WrongOperator:
  选错 operator。

OverStructured:
  operator 强度太高，答案僵硬。

SlotWrong:
  slot 填了，但 slot 不是问题真正需要的。

VerifierFalsePositive:
  fidelity 检查过了，但实际没用。

BaseGenerationNoise:
  operator 没启用也输，是基础生成问题。

DomainOperatorMissing:
  需要专门 domain operator。
```

你现在 business 的结论就像 `BaseGenerationNoise` 或 `DomainOperatorMissing`：business fresh generation 即使不用 operator 也输。

这个 residual taxonomy 非常关键。否则你会错误地继续调 operator prompt。

---

# 5. 具体下一版该做什么？

我建议下一版不要再叫“operator repair”，而叫：

```text
operator_policy_generalization_v1
```

包含 5 个 artifact。

---

## 5.1 `operator_policy.py`

输入：

```text
problem
domain
difficulty
retrieved nodes
frame meta
history residuals
```

输出：

```json
{
  "selected_operators": [],
  "operator_strength": "off|soft|required|strict|repair",
  "abstain_reason": null,
  "p_trigger": 0.0,
  "p_harm": 0.0
}
```

---

## 5.2 `operator_failure_diagnostics.py`

对所有 A/B loser 自动归因：

```text
TriggerFalsePositive
WrongOperator
OverStructured
SlotWrong
VerifierFalsePositive
BaseGenerationNoise
DomainOperatorMissing
```

---

## 5.3 `operator_semantic_fidelity.py`

用 LLM judge 或更强 rubric 判断：

```text
slot 是否实质满足
slot 是否与问题相关
negative control 是否有效
operator 是否改变了答案决策
```

---

## 5.4 `operator_generalization_benchmark.py`

跑：

```text
native GPT-5.5
ctx-only
operator v0
operator policy v1
operator policy v1 without repair
```

切分：

```text
daily_life
business
engineering
software_engineering
leave-domain-out
leave-family-out
```

---

## 5.5 `operator_policy_writeback.py`

把成功 operator 写回 graph：

```text
operator_spec payload
trigger evidence
anti-trigger evidence
loss cases
semantic fidelity
utility delta
```

这样假设模型才会真的“越测越丰满”，而且是 operator-level 丰满，而不是普通文本节点变多。

---

# 6. 你现在如何解读“少数题好，其他题不行”？

我会这样解释：

```text
1. 少数题好：
   因为这些题命中了 daily_life operator，且 required slots 确实改善了答案结构。

2. 其他题不行：
   因为 operator gate 没开，或开了但 operator 不适配，或基础生成噪声比 operator 影响更大。

3. 这不是方向错：
   因为 repair 后局部 utility 和 fidelity 都改善了。

4. 但这还不是泛化：
   因为 operator family 少、触发机制硬、样本小、semantic verifier 弱。
```

---

# 7. 最重要的下一步实验

我建议你做一个 60–100 题的实验，不要太大：

```text
20 daily_life
20 business
20 engineering
20 software_engineering
可选 20 science-decision / research-planning
```

每题跑：

```text
A. ctx-only
B. operator v0 all domains
C. operator policy v1 selective
D. operator policy v1 + repair
```

指标：

```text
overall utility
triggered subset utility
non-trigger harm
semantic fidelity
operator activation rate
operator false positive rate
operator false negative rate
control loss
```

真正目标：

```text
C/D 在 triggered subset 上显著赢；
non-trigger harm 接近 0；
activation coverage 比 daily_life-only 高；
business 不再被硬套 daily-life operator。
```

---

# 8. 最终建议

你这版是正确方向，但不要过度解读。它证明的是：

> **把假设 operator 化，并用 fidelity repair 防止 decorative use，可以在小触发集上提升。**

它还没有证明：

> **operator 能跨 domain 泛化。**

下一步泛化的核心不是再写更长 prompt，而是：

```text
operator taxonomy
+ trigger classifier
+ operator strength policy
+ semantic fidelity verifier
+ failure diagnostics
+ leave-domain/family-out benchmark
+ operator-level writeback
```

一句话总结：

> **你已经从“假设作为 context”走到了“假设作为 operator”。现在要从“手写 operator”走到“可学习的 operator policy”。**

这一步做成后，Assumption-Agent 才会真正从少数题局部强，变成在新题上能判断“该不该用、用哪个、用多强、怎么验证、失败如何归因”的泛化系统。
