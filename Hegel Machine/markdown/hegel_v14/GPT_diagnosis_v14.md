**我的判断是：V13 确实修好了上一轮的几项关键问题，尤其“真实失败 → 子问题 → 新研究 → 改写父程序 → 新确认”已经实际发生；但你最关心的完整目标，仍然不能判定为已经完成。**

这次最重要的发现是：**报告中表现最好的知识组，与强普通基线使用了不同的 H* 搜索策略。**因此，报告中的成本节省是真实的组间差异，但还不能完整归因于“自主形成并维护的假设，使未见问题的研究更有效”。

同时，代码还存在几处影响主线的缺口：无预算搜索可能被记为家族反驳，延期子问题没有跨轮恢复入口，普通基线没有实际重放额外调和证据，新增边界探针尚未接进普通研究循环。

我把详细诊断、下一轮施工合同和可运行复核脚本整理成了文件：

- **完整包：**Hegel_c71e65f7_V13_Audit_and_Next_Plan.zip[Hegel_c71e65f7_V13_Audit_and_Next_Plan.zip](sandbox:/workspace/scratch/9d1e58286220/Hegel_c71e65f7_V13_Audit_and_Next_Plan.zip)
- **系统审计：**01_V13_系统审计与主目标判定.md[01_V13_系统审计与主目标判定.md](sandbox:/workspace/scratch/9d1e58286220/hegel_c71e65f7_review/01_V13_系统审计与主目标判定.md)
- **可直接交给 Gemini 的下一轮合同：**02_下一轮最小施工与实验合同.md[02_下一轮最小施工与实验合同.md](sandbox:/workspace/scratch/9d1e58286220/hegel_c71e65f7_review/02_下一轮最小施工与实验合同.md)
- **复核范围与数据：**03_来源与复核范围.md[03_来源与复核范围.md](sandbox:/workspace/scratch/9d1e58286220/hegel_c71e65f7_review/03_来源与复核范围.md) · recomputed_results.json[recomputed_results.json](sandbox:/workspace/scratch/9d1e58286220/hegel_c71e65f7_review/recomputed_results.json)

---

## 一、这次哪些进步是真的，应该明确认可

我核对了指定提交与上传包的对应关系：**4 个修改的运行时源码、PR-C 脚本和 3 份新回执，其 Git blob SHA 均与 `c71e65f7` 一致**；运行了作者的独立校验器，61 项文件哈希检查通过；读取逐题外部评价值，独立重算了 **96 个目标任务、32 个普通源任务和 8 个共同锚点**的完成、预算与汇总指标。

本次没有重新运行完整科学实验或作者报告的 62 项全仓库测试。下面会区分原始实验结果、本次算术复核和本次源码探针，不把三者混在一起。03_来源与复核范围.md[复核范围](sandbox:/workspace/scratch/9d1e58286220/hegel_c71e65f7_review/03_来源与复核范围.md)

### 1. 上一轮的六个问题，已有多项实质修复

| 上一轮问题 | V13 的实际改进 | 本轮判断 |
|---|---|---|
| 取得新 probe 失败，却拿旧数据冒充新执行 | 已区分 `executed_new / reused_exact_protocol / unavailable` | **原始伪执行错误已修复**；同协议复用另有接口问题 |
| 双方假设失败后，子问题只生成而不执行 | 已进入普通 `run()`，可触发研究并替换父程序 | **同次运行内的递归返回已实现** |
| 没有 pilot 或分母为零，被当成统计值 0 | 改成 `None`，guard 使用三态，依赖从完整 AST 导出 | **核心修复成立** |
| 结构签名不同就当世界反例；旧范围补集直接确认 | 加入交叉预测检验；区分路由候选；补集降为未确认提案 | **方向正确，仍有反驳层级和消费问题** |
| branch 与 commitment 身份错配 | 建立显式来源映射，保留多源独立支持 | **原 ID 错配已修复** |
| 普通档案不断重复构造，没有缓存完整结果 | 实现三层产物缓存，重复调用确实复用 | **缓存明显改善**；完整历史重放仍未落实 |

所以，不能再按上一轮的口径说“仍然只是加日志、没有真实递归”。这些进步在[普通研究入口](https://github.com/erzhu419/Assumption-Agent/blob/c71e65f7f10863ea68f2ccecac2cd5a86e1e1ba4/Hegel%20Machine/src/hegel_machine/cognition/unified_research.py)和[分支调和实现](https://github.com/erzhu419/Assumption-Agent/blob/c71e65f7f10863ea68f2ccecac2cd5a86e1e1ba4/Hegel%20Machine/src/hegel_machine/cognition/branch_reconciliation.py)中都有实际代码支持。

### 2. 已经有一个真正修好父问题的例子

PR-A 的 `focal_memory / seed=9315` 中，历史马尔可夫分支与当前候选都失败后：

- 系统生成子问题；
- 调用已有的潜变量研究能力；
- 父程序发生实际变化，新程序出现 `accumulate` 结构；
- 用 **24 行新确认数据**得到损失约 **\(2.54\times10^{-17}\)**；
- 另外 **48 行外部评价**的 MSE 约 **\(1.61\times10^{-17}\)**。

这已经支持：

> **系统根据普通研究中的失败，调用已有研究能力，形成不同的可执行假设，并修好了当前父任务。**

固定 DSL、已有 latent 搜索器，并不使这个结果失去意义。我们不需要把标准抬高到“必须先发明新原语或改写全部研究算法”。这个具体进步应当保留。[PR-A 原始回执](https://github.com/erzhu419/Assumption-Agent/blob/c71e65f7f10863ea68f2ccecac2cd5a86e1e1ba4/Hegel%20Machine/docs/evidence/v13_evidence_recursion_and_routing_20261117/pr_a_evidence_and_recursion_controls_20261117.json)

---

## 二、最影响结论的问题：主比较同时改变了知识组织和 H* 调度

### 1. 原始成绩基本能复核出来

| 条件 | 严格通过 | 假确认 | 目标计算成本均值 | 墙钟均值 |
|---|---:|---:|---:|---:|
| `K_cold` | **23/24** | 1 | 226,253.33 | 25.31 秒 |
| `R_ordinary_optimized_lazy` | 21/24 | 2 | 226,943.08 | 34.44 秒 |
| `K_routing_no_H_star` | **22/24** | 2 | **184,004.29** | **23.59 秒** |
| `K_routing_plus_H_star` | 20/24 | 2 | 249,564.33 | 39.84 秒 |

这里的“严格通过”是本实验的有限样本标准：内部完成、确认损失及外部 48 行 MSE 均不高于 `1e-7`，并满足预算。计算成本是项目的计费单位，墙钟包含脚本中的外部评价时间。`K_cold` 也有共同校准锚点，并非完全没有先验。[PR-C 原始回执](https://github.com/erzhu419/Assumption-Agent/blob/c71e65f7f10863ea68f2ccecac2cd5a86e1e1ba4/Hegel%20Machine/docs/evidence/v13_evidence_recursion_and_routing_20261117/pr_c_tightened_source_paired_study_20261117.json)

但分组代码明确写了：

```python
lib_r._prioritize_generalized_families = True
lib_no_h._prioritize_generalized_families = False
lib_plus_h._prioritize_generalized_families = True
```

即：

| 比较组 | 是否调度 H* 家族前置搜索 |
|---|---|
| 普通 R | 开 |
| 最佳 K-noH | 关 |
| K-plusH | 开 |

这个开关会改变真实候选生成、搜索顺序和预算占用。普通入口先处理调度到的家族，再处理普通 commitment。[PR-C 分组代码](https://github.com/erzhu419/Assumption-Agent/blob/c71e65f7f10863ea68f2ccecac2cd5a86e1e1ba4/Hegel%20Machine/scripts/run_v13_pr_c_tightened_study.py#L216)

**因此，R 与 K-noH 的差异同时包含：**

1. 是否提前组织来源知识；
2. 是否启用本次已经表现有害的 H* 搜索路径。

18.92% 不能直接写成第一项的独立贡献。

### 2. 同样开启 H* 时，已有结果反而是预组织组更差

现有数据还提供了一个必须一起看的比较：

- **普通 R：21/24，均值计算成本 226,943。**
- **预组织 K-plusH：20/24，均值计算成本 249,564。**

后者平均贵 **9.97%**，少完成一题。

而同一预组织库中，K-noH 相对 K-plusH：

- 多完成 2 题；
- 平均少用 **65,560** 计算单位。

这说明当前比较中较清楚的发现是：

> **当前实现的 H* 前置调用策略没有增量收益，而且会在部分任务上消耗预算、损害结果。**

这不等于 H* 这种抽象表示永远无用；它限定的是当前实现与使用策略。recomputed_results.json[独立复算结果](sandbox:/workspace/scratch/9d1e58286220/hegel_c71e65f7_review/recomputed_results.json)

### 3. 唯一新增通过的目标，更直接暴露了 H* 的伤害

在 `1024000/T1_unseen_family` 上：

| 条件 | 结果 | 计算成本 | 外部 MSE |
|---|---|---:|---:|
| 普通 R，H* 开 | 未决 | 679,993 | 0.001642 |
| 预组织 K，H* 开 | 未决 | 679,995 | 0.001642 |
| 预组织 K，H* 关 | 通过 | 254,538 | \(1.688\times10^{-18}\) |
| cold | 通过 | 265,600 | \(1.688\times10^{-18}\) |

两个 H* 开启组选中了同一个错误家族绑定 `theta_0_0.273:r_0.937`。关闭组成功，cold 也成功。

所以这个案例确实有诊断价值，但它还没有回答：

> **如果普通档案也关闭同一条有害 H* 路径，它会不会同样成功，以及成本是多少？**

缺失的最小对照正是 **`R_lazy_noH`**。不能通过在旧账单上减掉几项费用，代替实际运行这个条件。[该案例原始记录](https://github.com/erzhu419/Assumption-Agent/blob/c71e65f7f10863ea68f2ccecac2cd5a86e1e1ba4/Hegel%20Machine/docs/evidence/v13_evidence_recursion_and_routing_20261117/pr_c_tightened_source_paired_study_20261117.json)

---

## 三、代码层面还有哪些实质缺口

### 1. 最需要优先修复：搜索失败可能被升级为科学反驳

`_cross_evaluate_refittable_family_on_neighbor()` 遇到家族实例化返回空列表时，直接给出：

```python
{
    "evaluated": True,
    "cross_mse": inf,
    "empirically_refuted": True,
}
```

但实例化可以因为预算不足、证据缺失或不支持的槽位而返回空列表。

本次最小源码复现使用真实 `SearchBudget(0)`：

- 实际计算计费为 0；
- 出现一次预算拒绝；
- 最终仍被判为 `empirically_refuted=True`。

这个结论还可能被用于生成硬作用域 guard。[家族交叉检验代码](https://github.com/erzhu419/Assumption-Agent/blob/c71e65f7f10863ea68f2ccecac2cd5a86e1e1ba4/Hegel%20Machine/src/hegel_machine/cognition/branch_reconciliation.py#L3235)

必须分开四种事实：

| 实际发生的事 | 合理结论 |
|---|---|
| 一个固定实例实际预测错误 | 该实例在这些证据上被反驳 |
| 有限搜索没有找到低误差绑定 | 本次搜索未找到解 |
| 没有预算或实现不支持 | 未知、延期或不支持 |
| 声明参数域被完整检查且无可行解 | 该数据和参数域下的家族不可行 |

**有限搜索失败，不能自动否定整个可重拟合家族。**

这是一处确定的控制流缺陷；我没有据此声称 96 个完整目标都触发了它，也没有把所有成绩判为无效。

### 2. 固定实例失败，仍可能错误限制可重拟合模板

当前交叉检验保留源半径和边条件，只重新拟合部分线性系数；失败后收缩 branch scope。dispatcher 又依据这个 scope 排除对应 commitment。

然而，下游 commitment 本来可能允许对新目标重新拟合半径等参数。

也就是说：

> “旧参数实例不适合新环境”的证据，有可能传播成“这个机制模板不应再尝试”。

本次隔离示例中，旧门函数 `1[x<0.5]` 对新环境失败，但合法改绑成 `1[x<0.75]` 可以精确解释；前者生成的 guard 仍能使模板被排除。这验证了对象层级风险，属于明确使用简化依赖的控制流示例，不是完整物理任务回放。[分支交叉检验与范围修订实现](https://github.com/erzhu419/Assumption-Agent/blob/c71e65f7f10863ea68f2ccecac2cd5a86e1e1ba4/Hegel%20Machine/src/hegel_machine/cognition/branch_reconciliation.py)

下一步需要明确：

- **固定绑定假设实例；**
- **允许合法重拟合的机制模板；**
- **经验路由策略。**

反例更新哪一层，必须与实际被测对象一致。

### 3. “路由只改变排序”尚未贯彻到消费者

这里有两处具体问题：

- 一部分 aggregate commitment 进入尾队列前已加入 `seen`，尾队列拼接时又因为“已经见过”被跳过；
- 家族路由不匹配后放入 `deferred_router_families`，但普通入口只消费 `scheduled_families`，没有消费 deferred 列表的路径。

因此，某些所谓“稍后尝试”的候选，实际不会被该复用路径尝试。

这不代表所有知识都被删除——其他 branch 或普通搜索可能补救——但它使“路由仅改变排序”的语义承诺不成立。[dispatcher 实现](https://github.com/erzhu419/Assumption-Agent/blob/c71e65f7f10863ea68f2ccecac2cd5a86e1e1ba4/Hegel%20Machine/src/hegel_machine/cognition/branch_reconciliation.py#L3644) · [普通消费入口](https://github.com/erzhu419/Assumption-Agent/blob/c71e65f7f10863ea68f2ccecac2cd5a86e1e1ba4/Hegel%20Machine/src/hegel_machine/cognition/unified_research.py#L637)

### 4. 证据复用和普通重建，仍有生产—消费接口错位

**第一处：同协议复用的数据形状不一致。**

真实 `acquire` 存储的是：

```python
{
    "probe": asdict(probe),
    "observations": [asdict(row), ...],
}
```

复用 callback 却读取 batch 顶层的 `sequence/reset/probe_id`，并把 observations 当作对象列表。测试恰好构造了平铺字段，因此测试可以通过，生产形状却不能触发这条复用分支。

本次探针中，生产形状会再次尝试 acquisition；拒付后返回 unavailable。测试形状则直接 reused。[实际 callback](https://github.com/erzhu419/Assumption-Agent/blob/c71e65f7f10863ea68f2ccecac2cd5a86e1e1ba4/Hegel%20Machine/src/hegel_machine/cognition/unified_research.py#L8089)

**第二处：普通组获得了调和证据容器，但没有重放其内容。**

`_raw_reconciliation_evidence_archive` 的确被保存并复制给普通组，这是进步。但普通重建函数对它的读取只有 `len()`，用于缓存键，没有把额外 probe 的实际证据重放到分支、反例和 scope 重建中。

而 archive 本身也没有完整保存 transitions、动作/reset、schema/绑定等必要信息。

所以目前可以说“普通组可访问这个容器”，还不能说“普通组实际从相同完整经历重建了语义等价的知识”。[普通档案重建函数](https://github.com/erzhu419/Assumption-Agent/blob/c71e65f7f10863ea68f2ccecac2cd5a86e1e1ba4/Hegel%20Machine/src/hegel_machine/cognition/branch_reconciliation.py#L3359)

### 5. 递归修复已有当前任务成功，但持久生命周期仍缺后半段

当前实现还有三处明确边界：

1. **延期问题没有跨轮恢复入口。**  
   `response_questions` 每次从空列表开始，只装入本轮的新问题。持久队列主要用于追加和状态更新。PR-A funded/unfunded 是两个独立 deepcopy，不是先延期再继续同一个父研究。

2. **范围识别问题进入队列后被过滤。**  
   handler 只处理 `both_competing_branches_refuted`；`scope_boundary_unidentified_after_refutation` 尚无对应执行路径。

3. **子研究修好父程序后，没有再次进入分支调和。**  
   新概念被记忆保存，但没有完整走完“与其余历史分支重新比较、更新范围与依赖”的步骤。成功处理第一个问题后，另一个问题仍可能 pending。[子问题 handler](https://github.com/erzhu419/Assumption-Agent/blob/c71e65f7f10863ea68f2ccecac2cd5a86e1e1ba4/Hegel%20Machine/src/hegel_machine/cognition/unified_research.py#L8211)

这意味着，**当前任务的修复成功已经成立；同一个修复对象之后如何维护和改善未见任务，尚未被这组证据完成。**

### 6. 新边界探针目前只完成了“能够提出”

新增的 `propose_threshold_interval_boundary_probe()` 只在 PR-AB 控制脚本和单测中调用，普通入口没有调用它。

控制脚本手工传入区间 `[0.34, 0.43]` 和半径 `0.89`，记录能生成有分歧的 probe；没有接着执行购测、更新区间、修订父程序和重新独立确认。[边界控制脚本](https://github.com/erzhu419/Assumption-Agent/blob/c71e65f7f10863ea68f2ccecac2cd5a86e1e1ba4/Hegel%20Machine/scripts/run_v13_pr_ab_controls.py#L443)

这与当前仍有的假确认相呼应：

- `1023000/T3`：三个知识组假确认，cold 正确通过；
- `1026000/T3`：四组共同假确认。

这里应当写“新增边界生成器已实现”，不能写“这条边界识别闭环已修好”。也不能把它夸张为旧系统从未做过任何主动阈值测试；缺的是这次合同要求的具体通路。

---

## 四、你最关心的完整目标，现在到底处于什么状态

我会这样分级：

| 目标 | 当前判断 |
|---|---|
| 从普通经历产生新的可执行假设 | **已有有限语言内的真实实例** |
| 真实冲突导致声明变化 | **已有，但反驳层级与范围语义还需修正** |
| 子问题答案实际改变父研究 | **有明确同次运行成功** |
| 延期继续、修复后再与历史分支检查 | **尚未完整接通** |
| 强普通方法从相同历史实际重建知识 | **缓存改善明确，完整证据重放尚缺** |
| 同一个新假设改善某个未见问题 | **尚无干净的连续对象因果证据** |
| 整体 self-recursive assumption 稳定优于使用前或强普通方法 | **当前不能宣称** |

关键在于：**PR-A 的潜变量递归修复，与 PR-C 的阈值家族迁移，是两套证据。**

把前者的“修好了父程序”与后者的“平均成本较低”放在一起，仍然不能代替：

> 这个具体修复对象，如何经过范围和历史分支检查，又如何改变了后继未见任务；允许普通方法重建它以后，差异还剩多少。

此外，本轮的“未见”主要是既定机制族中的新阈值、半径、命名/布线与对象数变化。它们是有效的未见实例测试，但还不是新物理机制族或开放研究问题的证据。[实验定义与冻结方式](https://github.com/erzhu419/Assumption-Agent/blob/c71e65f7f10863ea68f2ccecac2cd5a86e1e1ba4/Hegel%20Machine/scripts/run_v13_pr_c_tightened_study.py)

---

## 五、报告还有三处解释需要收窄

### 1. 源任务不是 32/32 严格通过

准确表述是：

- **32/32 内部确认；**
- **31/32 严格通过；**
- **1 次源期假确认。**

原始失败保留在回执中，这是好的；问题在汇总字段和文字把内部确认称成严格确认。recomputed_results.json[独立复算](sandbox:/workspace/scratch/9d1e58286220/hegel_c71e65f7_review/recomputed_results.json)

### 2. “约 1.02 题回本”不能成为通用结论

这个比值的算术成立，但它有两个限制：

- 分母是当前存在 H* 策略混杂的平均成本差；
- 目标按 T1→T2→T3 运行，普通组后续缓存变暖，每题节省不是恒定的。

即使沿用当前数据，按维护计算费作逐链前缀核算，仍有 **3/8 条链在三题后没有回本**。

另外，R 的直接重建费仅占两组总目标计算差的 **32.68%**。其余涉及候选搜索与预算路径，不能全部归为“把重建开销前移到源期”。recomputed_results.json[成本复算](sandbox:/workspace/scratch/9d1e58286220/hegel_c71e65f7_review/recomputed_results.json)

### 3. `p=1` 不证明等价；19/24→21/24 不证明缓存导致提升

完成率检验没有识别出差异，不等于证明两组等效或非劣。

V12 与 V13 还用了不同来源链和整体代码。可以分别说“缓存错误已修复”“当前差距缩小”，但不能据此量化“先前差距相当一部分由缓存造成”。这需要匹配样本的单因子回放。

---

## 六、下一轮具体应该做什么

我建议下一轮集中完成 **三项交付**，不再扩展物理家族或另建总框架。

### 交付一：修正证据、反驳和消费语义

优先修改现有函数：

- `acquire` / `_exec_reconcile_probe`；
- `adjudicate_branches_on_paid_evidence`；
- 两种 `_cross_evaluate_*_on_neighbor`；
- `dispatch_branches_and_families_for_target`；
- 普通入口对候选与问题队列的消费。

必须落实：

1. **真实 batch 直接进入复用消费者。**一次真实取得后，第二次同协议请求复用原 ID；动作或 reset 不同则不能冒充复用。
2. **预算拒绝立即停止相关计算和知识更新。**
3. **固定实例失败、模板搜索未找到解、模板不可行分别记录。**
4. **路由后置候选在剩余预算允许时仍能实际尝试。**
5. **持久 pending 能恢复同一父研究，并核对父版本、实验上下文和累计预算。**
6. **子研究新确认后，再进入受影响分支的局部调和。**

验收重点应改成“生产者真实输出能否被消费者正确使用”。这能直接覆盖本轮多处“单独测试通过、普通路径未接通”的问题。

### 交付二：补真正公平的普通重建对照

主试验用这三组即可：

| 条件 | `R_lazy_noH` | `K_eager_noH` | `K_cold` |
|---|---|---|---|
| 合法研究 API、M0、目标预算 | 相同 | 相同 | 相同 |
| H* 前置调用 | 关闭 | 关闭 | 关闭或无该对象 |
| 完整来源及调和证据 | 实际可重放 | 同样证据 | 仅共同起点 |
| 派生知识构建 | 按需构建并缓存 | 源期提前构建并缓存 | 从目标研究 |
| 范围/路由算法 | 同一算法 | 同一算法 | 同一研究器 |
| 普通重新研究能力 | 保留 | 保留 | 保留 |

同时要求：

- 两组从相同 EvidenceBatch 构建的预测、guard 判定和反例关系应语义等价；
- 缓存键绑定内容和版本，不能只看记录数量；
- 普通组可以构建并记忆相同知识；
- H* 关闭时，不强迫普通组支付永不消费的 H* 构造费；
- 预组织了但后来没用到的工作，也计入 K 自己的维护成本。

先在已经曝光的 `1024000/T1` 补 R-noH，用它诊断，不当作新泛化成功。接口和配置稳定后，再冻结新来源链。

**上一轮我给的合同也有不够精确之处：**要求了三臂和同工具，却没有把普通组的 H* 调度开关逐字段冻结，也没有明确共享原始历史必须通过实际重建的行为比对。本轮合同已经补上这两项，避免再次出现字面满足、实际路径不同。

### 交付三：跟踪同一个修复假设的完整后继效用

可以继续使用现有 `focal_memory` 任务，不需要另开新领域。

完整实验应是：

1. 普通来源经历形成旧假设。
2. 一个真实新任务使旧假设失效。
3. 子研究形成具体新版本，例如引入历史状态的程序。
4. 新确认通过后，与历史分支重新比较。
5. 冻结这个具体对象。
6. 在实验室已支持的新 latent 参数、命名、布线等未见实例上复用。
7. 与具有相同 latent 搜索能力、相同原始证据和预算的普通方法比较。

还应加入一个对象移除干预：只移除该修复版本及由它独占支持的派生关系，保留原始经历和普通重建权限。观察系统是否需要重新发现它、成本增加多少，或者普通方法直接打平。

这能区分：

- 子问题触发已有工具是否有调度价值；
- 修复出的知识是否有后继复用价值；
- 提前保存对象是否只是省去一次重建；
- 是否真的存在普通方法在相同资源下尚未取得的研究改善。

不能再用“有预算的递归组”对比“预算不足的组”，来证明递归净收益。

---

## 七、你一直要求的分支修订、融合和消除，应落实成这些决策

| 情况 | 应执行的动作 |
|---|---|
| 旧参数实例失败，合法重新标定后可解释 | 修订实例，保留机制模板 |
| 两支适用范围不重叠 | 分支共存 |
| 重叠范围有实际冲突，只有一支失败 | 限制失败支在该范围的资格，并研究边界 |
| 双方都失败 | 限制两条具体声明，提出遗漏机制或表示问题 |
| 不同表达暂时给出相同预测 | 保留可能等价关系，必要时主动找区别 |
| 广义表达保留旧支持和已测试的拒绝 | 可以合并/反统一，保留来源与独立支持 |
| 合并丢失旧反例或损害某一来源 | 回退合并，恢复分支或缩小合并范围 |
| 一个来源撤回，另有独立支持 | 局部更新，不整体删除共同结构 |
| 路由特征不典型 | 降优先级，不能直接签科学反驳 |
| 无预算、协议不符、搜索未完成 | 保持未知或延期 |

尤其要坚持：**“参数变了”“机制变了”“搜索器没找到”“路由暂时不优先”是不同事件。**目前不少剩余问题来自这些事件在实现中被合并成同一种排除动作。

---

## 八、结合 AWM 和近期自改进研究，下一步该借鉴什么

本次补看了 AWM v2、AIDE² 和 PAST-Bench，最相关的是三项原则。

**AWM：维护已测试的区分，并把模型内部成立与外部世界正确分开。**对 Hegel 当前最直接的作用，是要求合并和范围修订保留反例语义、继续接受真实环境检验；暂时没有必要为了这一点扩充全面 Lean 化。[arxiv.org](https://arxiv.org/html/2610.40503v2?utm_source=chatgpt.com)

**AIDE²：研究能力的改善需要固定预算和独立评价。**Hegel 可以先证明知识层的改进，不必同时开展代码自改写；但比较双方必须拥有同等工具和合理策略，后继任务不能参与选择。[arxiv.org](https://arxiv.org/html/2609.26457v1?utm_source=chatgpt.com)

**PAST-Bench：后继成绩与实际存储、检索、更新路径要同时检查。**这与本轮缺口直接对应：保存字段、缓存命中或进入队列，仍需要进一步证明实际消费及其效果。[arxiv.org](https://arxiv.org/html/2608.04003v1?utm_source=chatgpt.com)

这些工作提供评价方法，不能替本项目自身证明收益。

---

## 九、下一轮应怎样判定完成，避免继续反复迭代却说不清贡献

建议把结论分成三个层次：

1. **机制接通。**  
   真实证据身份正确，预算不足不制造反驳，延期问题能恢复，修复后的对象能重新进入分支维护。

2. **个案因果贡献。**  
   能追踪同一个新假设，从来源到修订，再到具体未见目标；同能力普通对照与对象干预能说明它改变了什么。

3. **工作负载上的总体收益。**  
   在冻结新来源链上，同时报告严格完成、假确认、负迁移和全部成本，并按来源链配对分析。

如果公平的 R-noH 最终打平，应该接受：当前价值主要落在可审计知识组织、可解释修订或特定工作负载的时延改善。如果成本优势仍在，但假确认增加，就要明确质量与资源的取舍。如果某个修复对象在同策略普通对照下仍稳定减少研究成本，才有充分理由进一步强化 self-recursive assumption 的效用主张。

**我建议下一轮的验收句只保留这一句：**

> **系统从哪些真实经历形成了哪个具体新假设，因什么证据修订了哪些分支；这个同一对象怎样改变了哪个未见问题，而允许重建相同知识的强普通方法实际做得怎样。**

V13 已经具备完成这条证据链的若干关键部件。现在最值得投入的，是让它们在真实运行中连成同一条链，并把对照条件彻底对齐。