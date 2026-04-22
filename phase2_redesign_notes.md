# Phase 2 重做笔记：从 161 条到 Wisdom Library

写于 2026-04-22，在 phase2_v2 (6 aphorism triggers by GPT-5.4) 失败之后。目的是把**当前 Phase 2 的所有不足 + 新发现的 LLM 消化机制 + v3 设计**都沉淀下来。

---

## 一、原 Phase 2 (161 triggers) 的问题回顾

### 1.1 来源极其狭窄

161 条 triggers 的**唯一来源**是：我们 `orient_hybrid vs baseline` 实验中 55 条失败案例的反向归纳。也就是说，它们是"**从我们自己的具体失败症状里提炼的警觉**"，**不是从人类千年文献里沉淀下来的跨时代智慧**。

爱因斯坦、罗素、Popper、塔木德、孙子兵法、箴言等真正的人类 wisdom corpus，**一条都没有进入我们的 trigger library**。

**例证（用户提供）：**
- 罗素："一切悲剧来自该坚持时不坚持、不该坚持时坚持了" —— 覆盖所有"是否继续投入"类问题
- 爱因斯坦："凡事应尽可能简单，但不能过度简单" —— 覆盖所有"过度压缩 vs 过度复杂"的权衡

这些直接的、强 orientation 的警句，我们**从来没考虑过把它们作为 base corpus**。

### 1.2 粒度混乱：三类 trigger 质量差别巨大

手动扫完 161 条后：
- **~30 条真 transferable**（具体可操作、跨问题通用、有 insight）
- **~80 条模板化稀释**（"是否充分识别并重述..."之类的 meta-reflection，说一万遍等于没说）
- **~50 条过度具体**（绑死在某个 problem instance，无法迁移）

没有质量过滤，直接拿来用，但整体上确实 work（phase2_triggers 53% vs baseline）。

### 1.3 Category-keyed retrieval 本质上还是 rule library

最初目标：**attention priors**（改变 LLM 注意力朝向的先验）。
实际实现：**problem → (domain, difficulty) → retrieve top-4 triggers**。

**这就是 rule library 按 key 索引**，只是 key 从"策略 ID"换成了"领域+难度"。同一个 category 下 100 个不同问题都拿到相同 4 条 triggers，大部分并不贴合具体问题。

### 1.4 其他漏洞

- **post-hoc warning，不是 pre-hoc 早期信号**（违反"一叶知秋"本意）
- **挖一次冻结**，没有持续迭代
- **只 mine 了 55 losses**，如果扩到 500 会更好（但没测）
- **没人工筛选**就直接用了
- **judge dimension 可能 reward verbosity**，53% 或许是 artifact

---

## 二、phase2_v2 (6 aphorism triggers) 失败的启示

用 GPT-5.4 从 55 losses 聚类 + 提炼，产出 6 条极高质量警句：

1. 雾里岔路多，先立坐标再迈步（12 字）
2. 约束一旦点名，答案别绕着走（12 字）
3. 一团可能性时，顺序比深度值钱（13 字）
4. 久攻不下时，先重审前提与骨架（13 字）
5. 大数不吓人，占比才吓人（11 字）
6. 约束都在台面上，别急着重写问题（14 字）

**结果：vs baseline = 42%（Δ -0.44），vs phase2_v1 = 42%（Δ -0.51）。大幅退步。**

### 2.1 根本原因：警句需要 LLM 执行时 unpack，而 Flash 做不稳定

"雾里岔路多先立坐标"要 Flash：
1. **读这 11 个字** → 识别语义（雾 / 岔路 / 坐标）
2. **映射到当前问题**（哪里是"雾"？哪里是"岔路"？什么是"坐标"？）
3. **产出 concrete action**（先做什么才算"立坐标"？）

这三步的**任何一步失误**都让 trigger 失效。

而原 phase2_v1 的 "当感觉'没出路'时，问自己'哪条约束是真必要的'"：
- **已经 pre-unpacked**：scenario (当没出路时) + self-question (哪条约束真必要)
- Flash 只需直接填槽即可

**结论：** LLM 需要的不是 aphorism-level 压缩，是 scenario+self-question 展开形式。**这个 "unpack step" 必须由更强的 model 或人类预先做好，不能留给执行 LLM。**

### 2.2 我的 EXECUTE prompt 强制 explicit 引用警句

phase2_v2 的 EXECUTE 写着：

> "如果某条真的 fire 了（不只是'感觉沾边'），在答案里**明确提及它如何塑形了你的思考**（举例："'大数不吓人，占比才吓人' 让我先算出 X%..."）"

这**逼 Flash 把警句生硬塞进答案**。即使警句不真正适用，Flash 会强行引用以避免"违反指令"。结果答案质量被稀释。

**对比：** phase2_v1 的 EXECUTE 只说"如果某个警觉让你发现了盲点，在结尾点出即可"。**柔性、不强制。**

### 2.3 161 条混合 triggers 的"噪声 + 冗余"是功能特性

之前认为 161 条质量参差是弱点。v2 失败后意识到：**这是特性不是 bug**。

- 161 条**稀释出更高的 hit rate**：即使大部分模板化，每题总有几条 fire
- 6 条**高密度但低命中率**：警句太 specific 或太泛，大部分题目激活不了任何一条

**教训：** 对 LLM 消化而言，**冗余覆盖 > 高密度压缩**。这和人类记忆原则相反（人类偏好精炼），但和 LLM 的 statistical activation 特性一致。

---

## 三、其他 Phase 2 路线的不足与偏离（之前没做好的）

### 3.1 Phase 1 改造（orient_hybrid）可能根本没贡献

**没测过：** Phase 2 triggers 叠在 **ours_27 technique form** 上是什么效果。所有 phase2 变体都叠在 orient_hybrid 上。如果直接 ours_27 + phase2_triggers 也 53%，那 **Phase 1 改造等于零贡献**。

**未来必须做的 ablation。**

### 3.2 触发器没做 signal-level retrieval

原设想"一叶知秋" = 看到 signal 就激活对应警觉。
**实际只做了 category → trigger 键值查表。**

真正的设计应该是：每个 trigger 配 `signal_pattern`（什么字样/结构/embedding 匹配时激活），retrieve 时做 signal → trigger 绑定。

### 3.3 Judge 可能在奖励 verbosity，不是 quality

LLM-as-judge 四维度（问题理解/分析深度/结构化/实用性）天然偏好**更长、更结构化、更多分点**的答案。phase2_triggers 的 53% 胜因可能是**答案变长**，不是**真正有更好 insight**。

**未做：** 人工 blind judge 20 题验证，截短到 200 字重跑验证。

### 3.4 canonical vs paraphrase 的 "LLM-as-translator" 实验被误读

clean ablation 显示 canonical (Polya 原文翻译) 46.5% vs phase2_triggers 53%。我当时下结论 "人类翻译者有不可替代 insight compression"。

**但现在回头看：这个结论部分是由 model capacity 决定的**。Gemini Flash 压缩不到 aphorism 层，所以：
- 原典问句（"你以前见过这个问题吗？"）→ 需要 unpack → Flash 做不好 → 低分
- 我的 paraphrase（已经 pre-unpacked）→ Flash 直接能用 → 高分

如果换 Claude Opus 4 或 GPT-5.4 做**执行 LLM**，原典/aphorism 可能立即 work。我们的结论是**针对 Flash 消费者的结论**，不是普适真理。

### 3.5 Phase 3 archetypes 的失败也和 unpack 能力有关

Phase 3 的 archetype 库（沉没成本、路径依赖、观念根源等 20 条）注入后，**实用领域轻微改善，formal 领域大幅退步**。这和 v2 失败一个道理：archetypes 也需要 unpack，Flash 在 formal 领域用不上。

### 3.6 没做 held-out validation（过拟合风险）

所有实验用同一 100 题（seed=42 shuffled test split 的前 100 条）。Stage 1 的 seed examples 从 train 集抽取，但 **evaluation 都是同一批 test**。

**风险：** Prompt/trigger 选择、aphorism 筛选、pipeline 调优都在这 100 题上调的。我们在无意识中对这 100 题做了 meta-overfit。

**未做：** 留出独立的 held-out 100 题（不同 seed），在方案全部调完后一次性验收。当前的 53% 在 held-out 上可能只有 48-50%。

### 3.7 没测其他 base LLMs

所有 execution 都用 Gemini 2.5 Flash。对 Flash 有效的 trigger 形式（pre-unpacked scenario+self-question）不一定对 Claude Opus / GPT-5 / 国产模型有效。

**完整结论需要跨 model 验证。** 可能 Flash 需要 pre-unpacked，但 Opus 直接消化 aphorism 更好。结论需要加 "for Gemini 2.5 Flash" 限定词。

### 3.8 没测 larger loss pool mining

我们只 mine 了 55 losses (orient_hybrid vs baseline)。如果加上 ours_27 vs baseline 的 53 losses，以及 vanilla_39 vs baseline 的 50 losses，可达 150+ losses，triggers 质量可能更高。**没试过。**

### 3.9 100 题样本量小，6pp 差距可能不稳健

phase2_triggers 53% vs baseline 47% = 6pp 差距，n=100。Wilson 95% CI 约 ±10pp —— **6pp 差距在统计学上勉强显著**。

**未做：** 跑到 n=300 或 500 题验证差距是否稳定。

### 3.10 Self-Discover 架构的 50% 天花板没被挑战过

我们所有 Phase 2/3 变体都在 Self-Discover 骨架（SELECT / ADAPT / IMPLEMENT / EXECUTE）内改内容。**从来没测过"跳出 Self-Discover"** 的变体。

如果 Self-Discover 对这类开放 advisory 问题的 ceiling 就是 ~53-55%，那我们不管怎么改 triggers 都出不去。值得测：
- 去掉 Self-Discover 的 SELECT/ADAPT/IMPLEMENT，只保留 EXECUTE（直接给全部 triggers + 问题）
- 多步 reasoning（CoT）代替单次 EXECUTE
- Verifier-critic 模式（一次生成，一次自检）

### 3.11 我们在测 **prompt engineering**，不是测 **方法论本身**

本质上从头到尾都在**prompt 字符串层面**改东西。**从没改过 LLM 的 context 用法、工具使用、检索机制、多轮交互**。

如果 Phase 2 要真正突破，可能需要跳出 prompt layer 到 **系统 layer**：
- RAG：per-problem retrieve wisdom from external corpus at execute time
- Tool use：让 LLM 调用"反例 search"、"类比 search" 等工具
- Multi-turn：第一轮给初答案，第二轮给反驳，第三轮给最终答

---

## 3A. 完整 1-15 条清单（前面分散表述的归纳）

为避免遗漏，把整体回顾浓缩成 15 条 explicit 清单：

| # | 类别 | 问题 | 在本文位置 |
|---|---|---|---|
| 1 | 来源 | Triggers 只来自 55 losses，不含人类 wisdom corpus | §1.1 |
| 2 | 质量 | 161 条中只 ~30 条 transferable，其余模板化/过度具体 | §1.2 |
| 3 | 架构 | category-keyed retrieval 本质还是 rule library 按 key 索引 | §1.3 |
| 4 | 本意 | post-hoc warning，不是 pre-hoc 早期信号（违背"一叶知秋"） | §1.4 |
| 5 | 本意 | attention priors 被稀释为 prompt 文字块 | §1.4 |
| 6 | 流程 | 挖一次冻结，没持续迭代 | §1.4 |
| 7 | 工程 | 没人工筛选就直接用了 | §1.4 |
| 8 | 过程 | 没测 Phase 2 alone on ours_27（Phase 1 是否必要不明） | §3.1 |
| 9 | 架构 | 没做 signal-level retrieval（真正的"一叶知秋"） | §3.2 |
| 10 | 评估 | Judge 可能奖励 verbosity | §3.3 |
| 11 | 过程 | canonical vs paraphrase 的 translator 结论被 model capacity 混淆 | §3.4 |
| 12 | 评估 | 没做 held-out validation，可能 meta-overfit | §3.6 |
| 13 | 评估 | 没跨 base LLM 验证（所有结论只限 Gemini Flash） | §3.7 |
| 14 | 评估 | 100 题 n 偏小，6pp 差距统计显著性有限 | §3.9 |
| 15 | 架构 | 没跳出 Self-Discover 骨架，可能受其 ceiling 限制 | §3.10 |

**v2 失败后新增的 3 个发现（insights, not gaps）：**

- [I1] LLM 需要 **pre-unpacked scenario+self-question**，不是 aphorism；unpack 步骤必须由强 model 或人类做 | §2.1
- [I2] EXECUTE prompt 的 **强制引用指令** 会稀释答案质量；要用软性"可选引用" | §2.2
- [I3] 对 LLM 消费者而言，**冗余覆盖 > 高密度压缩**（161 条噪声胜 6 条警句） | §2.3

---

## 四、phase2_v3 设计（Wisdom Library 路线）

**同时解决：**
- 数量不够 → 50-100 条 entries
- 来源窄 → 跨文明人类非虚构智慧（不再只挖我们的 losses）
- Aphorism 难 execute → 每条配 pre-unpacked 版本
- 选择压力 → LLM 按问题特异性挑 3-5 条，不是全量喂入
- Execute prompt 强制引用 → 改为软性（"可引用，非必须"）

### Schema

```json
{
  "id": "W07",
  "aphorism": "一切悲剧来自该坚持时不坚持、不该坚持时坚持了",
  "source": "Bertrand Russell",
  "signal": "涉及权衡、进退两难、是否继续投入、放弃 vs 续投",
  "unpacked_for_llm": "当你考虑放弃或继续时，先问两个问题：(1) 我之前的坚持来源是真 commitment 还是 sunk cost？(2) 这次放弃是理智回撤还是因为短期不舒服？",
  "cross_domain_examples": [
    "A: 工程项目 —— 已投入 3 年的失败产品该不该继续",
    "B: 日常生活 —— 多年恋情是否该结束"
  ]
}
```

### Pipeline

1. **Library 生成**：用 GPT-5.4 从跨文明 wisdom corpus 生成 80-100 条 entries。
   - 来源：传道书/箴言/孙子/道德经/Polya/Popper/Hayek/Kahneman/Russell/Einstein/芒格 等非虚构
   - 每条含 aphorism + source + signal + unpacked + examples
   - 强制跨领域过滤（trigger 本身不含 domain 特异术语）

2. **Per-problem 选择（Flash）**：展示全部 80-100 aphorism（短列表），Flash 按问题选 3-5 条最相关。

3. **Execute（Flash）**：只把选中的 **unpacked 版本**作为 attention priors 喂入。Aphorism 仅放在最后 "可选引用" 区域 —— **不强制**。

4. **Judge**：vs phase2_triggers (v1), vs baseline。

### 预期成本

- Library 生成：10 × GPT-5.4 call（每次产 ~10 条）= 10 calls，~8 min
- 选择：100 × Flash call = 100 calls，~5 min
- Execute：100 × Flash call = 100 calls，~5 min
- Judges：2 × 100 = 200 calls，~6 min
- **总计 ~410 calls，~25 min**

### 决策规则

- vs phase2_triggers ≥ **+5pp** → KEEP，作为新 winning variant
- ±3pp → INCONCLUSIVE，停止，接受 phase2_triggers
- <-3pp → REVERT，phase2_triggers 为最终版本

---

## 五、如果 phase2_v3 也失败，后续方向

1. **换执行 LLM**：用 Claude Opus 4 或 GPT-5.4 本身做执行者。可能 aphorism/unpacked 对强 model 等价。
2. **Signal-level retrieval**：给每条 wisdom entry 配 embedding signal，做真正的"一叶知秋"激活。
3. **Hybrid library**：v3 Wisdom Library + v1 的 161 条失败 triggers 都保留，分别按 signal 激活。
4. **验证 Judge 偏差**：换 evaluator 或人工 blind judge 20 题。

---

## 六、本次学到的工程原则（沉淀给后续 phases）

1. **LLM 需要 pre-unpacked scenario + self-question，不是 aphorism。** Unpack step 必须由人或强 model 代做。
2. **EXECUTE prompt 的"强制引用"指令会稀释答案。** 用软性"可选引用"。
3. **Trigger 数量与质量是 trade-off：**低密度冗余 > 高密度压缩（对 LLM 消费者）。
4. **Source corpus 不能只是"自己的失败记录"。** 人类千年非虚构沉淀不能跳过。
5. **所有结论限于 Gemini Flash 这个 model**。强 model 可能反向。
