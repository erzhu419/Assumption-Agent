# Phase 1 和 Phase 3 "完美版" 重做计划

写于 Phase 2 已 KEPT（53% vs baseline）+ Phase 3 已 REVERTED 之后。诚实审视 Phase 1 和 Phase 3 的当前实现哪里**不够好**，以及真正接近原意应该怎么做。

---

## Phase 1 大改: 从"我的 paraphrase"到"原典的 heuristic questions"

### 当前实现的缺陷

1. **我写的 orientation 是凭记忆的 paraphrase，不是源头。** 比如 S03 我写"养成每遇新问题就反问'我见过类似的吗'的习惯"。这是我**猜** Polya 会这么说。Polya 的 *How to Solve It* 实际上有 20+ 条**原始启发式问句**（"Have you seen it before?"、"Can you think of a familiar problem having the same or a similar unknown?"、"Is there a related problem?"、"Look at the unknown!"），承载力比我的概括丰富。

2. **Self-Discover 的 SELECT/ADAPT/IMPLEMENT 框架和 orientation 冲突。** orientation 的本质是"始终保持觉知"，不是"在某一步选择并改写"。把 orientation 塞进 "pick-adapt-implement" 流程 = 用 KPI 表承载修行法门，形式吞掉本质。

3. **Hybrid 混合丑陋：** 12 orient + 15 technique 两种形式共存，LLM 看异构 module list 会混乱判断。

### 重做方案

#### 步骤 1: 提取原典 heuristic questions

让 Gemini 从预训练知识里**逐字还原** Polya *How to Solve It*（1945）和 Popper *Logic of Scientific Discovery*（1959）里的原始启发式问句，对我们 12 个策略每一个至少拿到 3-5 条**原话**（中英双语）。

每个策略的新 JSON schema：

```json
{
  "id": "S03",
  "name": {"zh": "类比推理", "en": "Analogical Reasoning"},
  "form": "canonical_orientation",
  "source": {
    "author": "Polya, George",
    "work": "How to Solve It",
    "year": 1945,
    "section": "Second Main Part: How to Solve It — A Dialogue"
  },
  "heuristic_questions_en": [
    "Have you seen it before?",
    "Or have you seen the same problem in a slightly different form?",
    "Do you know a related problem?",
    "Do you know a problem with the same unknown?",
    "Could you restate the problem?"
  ],
  "heuristic_questions_zh": [
    "你以前见过类似的问题吗？",
    "或者这个问题的变形你见过吗？",
    "你知道一个相关的问题吗？",
    "你是否见过一个有同样未知量的问题？",
    "你能否用自己的话重新表述这个问题？"
  ]
}
```

这 12 个策略写入新目录 `phase zero/kb/strategies_canonical/`。

#### 步骤 2: Abandon SELECT/ADAPT/IMPLEMENT for these 12; use persona prompting

不再走 Self-Discover 流程。Test problem 的 EXECUTE prompt 变成：

```
你作为一个**内化了以下方法论精神**的思考者来回答。
不是"应用步骤"，是"带着这些问句在脑中"直接思考。

【来自 Polya 的习惯问句 (How to Solve It, 1945)】
- 你以前见过类似的问题吗？
- 你知道一个相关的问题吗？
- 你能否用自己的话重新表述这个问题？

【来自 Popper 的警觉 (Logic of Scientific Discovery, 1959)】
- 什么观察能证明我错？
- 我在寻找支持证据还是反驳证据？

【从历史失败中积累的 category-specific 警觉 (Phase 2)】
- [triggers]

问题：[...]

要求：
- 不要用 "Step 1, Step 2" 格式
- 带着上面这些气质直接回答
- 让启发式问句在你读问题时"自动浮现"，而不是"被应用"
```

对哪些 canonical 策略激活？三种方式：
- **Option A（简单）**：对每题总是激活全部 12 个 canonical（prompt 会很长但 LLM 自己筛）
- **Option B（中等）**：per category 预选 3-5 个 canonical（和 Phase 2 triggers 同级别）
- **Option C（per-problem）**：用 embedding 相似度为每题选 3 个最相关的 canonical

推荐 Option B（成本可控 + cache 可复用）。

#### 步骤 3: 保留 15 个非 Polya/Popper 策略为 technique 形式

这 15 个（S01 控制变量法 by Mill、S05 奥卡姆剃刀 by Occam、S07 反向推理 by 张贤亮、S11 满意化 by Simon 等）保持原 operational_steps，理由：
- 它们本身就偏 procedural（控制变量就是 procedural method），orientation 形式反而不准
- 降低复杂度 —— 只测 Polya/Popper 这一轴

#### 步骤 4: Stack with Phase 2 triggers

EXECUTE 同时包含 canonical 原典问句 + phase2 失败警觉 triggers。这样就是"原典意识 + 执行经验"的融合。

### 预期 Variants

- `phase1_canonical`: 12 canonical orientations + 15 technique，无 phase2 triggers
- `phase1_canonical_plus_p2`: + phase2 triggers（主测）

### 评估对比

| 对比 | 要回答的问题 |
|---|---|
| phase1_canonical_plus_p2 vs baseline | 最终系统能否大幅过 baseline？|
| phase1_canonical_plus_p2 vs phase2_triggers | 原典 sourcing 是否在 phase2 之上 add value？|
| phase1_canonical vs orient_hybrid | **原典 heuristic questions** 是否比**我的 paraphrase** 更有效？（核心 ablation）|

### 实施成本估算

- 提取 canonical questions: 12 × 1 LLM call = 12（一次性，缓存）
- Stage 1 结构发现（复用或重建）: 18 × 3 = 54
- 生成 answers: 100
- 3 个 judges × 100 = 300
- **总计 ~466 calls, ~17 min**

---

## Phase 3 大改: Wisdom-RAG（真正的"跨时代智慧"）

### 当前实现的缺陷

1. **Archetype 被我压成 2 行总结，但真正的承载不在原典文本里，而在"多案例的一致性"里。** 用户关键纠正：

   > "苹果和铅球都会掉下来，都是万有引力定律。有力量的是**宏观规律的一致性完全和牛顿的理论吻合**，也许不是《自然哲学的数学原理》原著的论文。"

   这意味着**原典 RAG 是错的方向**。真正有穿透力的不是读传道书全文，而是看到 10-20 个**跨时代、跨领域、跨领域都印证"已有的后必再有"的具体案例**：罗马帝国税制崩溃 ↔ 现代福利国家问题 ↔ 某家公司扩张期腐败；1637 郁金香泡沫 ↔ 1929 大萧条 ↔ 2000 互联网泡沫 ↔ 2008 金融危机。

   LLM 不是靠读原文"理解"archetype，是靠**观察足够多的同型实例**"识别"archetype。这是人类学会"万有引力"的方式（摔下来的苹果不止一个，每个都印证），也应该是 LLM 学会 archetype 的方式。

   **"A17: 这一次不一样几乎总是错的"本身是一个命题**，它的**力量来自 20 个印证它的历史泡沫**，不是传道书那 8 个字的韵律。

2. **Flat list 没有本体论结构。** 沉没成本（认知偏误层）、看不见的手（涌现秩序层）、已有的后必再有（形而上时间层）在**不同本体论层次**，却被我并列成同一粒度的 20 条。

3. **Pre-selection 违背 emergent wisdom。** 真正的智慧不是事先选定的清单 —— 它在**读问题的过程中**自己浮现。LLM 按 category 预选 3 个 archetype 塞进 prompt = "给问题戴眼镜"，不是让问题自己说话。

4. **Layer 3 当信息注入，不是姿态浸染。** "Layer 3 原型警觉：沉没成本谬误"是**告知**，不是**浸染**。LLM 把它当"要点 3"应付，不会改变思维本身。

### 重做方案：Case-Instance Library (非原典 RAG)

**核心原则修正（来自用户 2026-04-22）：**
> "力量在于多案例的一致性，不在原典文本本身。苹果和铅球都掉下来 —— 引力规律在**实例的共振**里，不在《原理》论文的文字里。"

所以不再做"索引传道书全文"这种 wisdom-RAG。而是：**每个 archetype 配一个 10-20 个跨领域案例的库**，让 LLM 通过**看实例**识别模式。

#### 步骤 1: 构建 Archetype → Case Library（案例 + 隐喻双轨）

**用户二次澄清（2026-04-22）：** "隐喻也没错，但不止原文。"

所以每个 archetype 的 library 包含两类条目：

**(a) 具体案例 (cases)** —— 历史事件、企业故事、个人决策、科学发现的**真实记录**。每条 100-300 字。作用：让 LLM 看到"在若干真实情境下这个 archetype 长什么样"。

**(b) 跨源隐喻 (metaphors)** —— 承载该 archetype 的**比喻/寓言/直观类比**，**不限于原典**。可以是：
- 古典文本里的（传道书的"日头出来，日头落下，急归所出之地"）
- 民间谚语 / 成语（"温水煮青蛙"、"饮鸩止渴"、"刻舟求剑"）
- 科学类比（"熵增"作为一个用滥了的 metaphor）
- 流行文化（"第 22 条军规"、"洞穴寓言的现代版"）
- 甚至是用户自己造的新隐喻，只要结构上同型

**为什么两类都要：** Cases 给 LLM 识别 pattern 的**实证素材**（"我见过 15 次"），Metaphors 给 LLM **认知压缩工具**（"哦这是'温水煮青蛙'那种情况"）。两者是**互补的**，都不在原典里，但都能穿透到 archetype 的本质。

**设计：** Cases + Metaphors 都跨时代跨领域，以展示同一 archetype 的**时空穿透力**。

**例子 — A17 "已有的后必再有" 的 case library：**

- 1637 郁金香泡沫 → 1929 股市崩盘 → 2000 互联网泡沫 → 2008 金融危机（经济周期性）
- 罗马帝国税制扩张后的财政崩溃 → 明朝一条鞭法后的卫所崩溃 → 现代福利国家可持续性危机
- 亚历山大 → 蒙古 → 拿破仑 → 美国的帝国扩张与维系困境
- 工业革命前后的技术焦虑 → 电气时代 → 计算机时代 → AI 时代的"工作被替代"恐慌
- 等...(15-20 个)

**例子 — A01 "沉没成本谬误" 的 case library：**

- 协和飞机项目明知不经济仍续建
- 越战升级时美国"已经死了这么多人，不能白死"
- 某公司继续投入失败产品"为了之前投的 5 年"
- 读到一半的烂书硬读完
- 恋爱多年尽管知道不合适仍结婚
- 等...(10-15 个)

**规模：** 20 archetypes × 10-15 cases = 200-300 cases。约 **40K-80K 中文字**（3-5 天可以手工 + LLM 辅助产出）。

#### 步骤 2: Embedding 索引 (cases 级别)

对每个 case 生成 embedding。每个 case 的 record：

```json
{
  "archetype_id": "A17",
  "archetype_name": "已有的后必再有",
  "case_id": "A17_c03",
  "title": "罗马帝国财政崩溃",
  "description": "3 世纪罗马帝国为维系边境军团持续扩张税基，中央通过通胀货币掠夺市民、向行省施压，导致地方精英逃避、生产崩溃、税基反而缩小...",
  "keywords": ["帝国扩张", "财政透支", "通胀", "地方反抗"],
  "era": "3rd century CE",
  "domain": "政治经济"
}
```

#### 步骤 3: Per-problem Case Retrieval

每个 test problem 的处理：

```
Step A: LLM 读问题 → 生成"问题骨架分析"（包含：核心冲突、主要约束、潜在结构）
Step B: 用骨架分析 embedding 检索 top-5 最相关的 **cases**（跨 archetype）
Step C: 看检索到哪些 archetype 的 cases 聚集最多 → 识别出 1-2 个"可能适用的 archetype"
Step D: 把 top 5 cases（完整描述）+ 识别出的 archetype 放进 EXECUTE
Step E: LLM 读 cases 再回答
```

关键：
- **retrieved 的是 cases（具体情境），不是原典文本**
- **Archetype 是"多个 cases 聚合后涌现的标签"，不是事先塞给 LLM 的 label**
- LLM 通过看"多个跨领域同型 case"**归纳**出共同模式，这就是"看到苹果看到铅球看到行星 → 识别到引力"

#### 步骤 4: EXECUTE prompt

```
你正在读一个问题。根据问题骨架分析，系统从**跨时代跨领域的案例库**中检索到
以下几个在结构上相似的案例：

【案例 1 (archetype: 已有的后必再有 | era: 1637 荷兰)】
1637 年郁金香球茎价格飙升至一位球茎等同一栋阿姆斯特丹运河房。市场参与者
都相信"这一次不同——郁金香是稀缺资源"。崩盘后价格 6 周内跌 99%。

【案例 2 (archetype: 已有的后必再有 | era: 2000 互联网)】
1999 年 pets.com 市值 4 亿美元，靠风投烧钱做广告。投资者相信"新经济规则"
让现金流不重要。18 个月后破产。

【案例 3 (archetype: 沉没成本 | era: 1960s 协和)】
英法合作的协和超音速客机明知永远无法盈利，但"已投入 30 亿英镑不能白费"。
项目一直拖到 2003 年才终止，累计亏损远超初始预算。

共同的模式在这些案例里清晰吗？现在看问题：

问题：[...]

请不要用 Step 格式。读完案例带着这些 resonance 再读问题。在答案里显性
提及你从哪个案例识别到类似模式（如果有）。
```

### 预期 Variant

- `phase3_wisdom_rag`: Wisdom-RAG 架构 + Phase 2 triggers

### 评估对比

| 对比 | 要回答的问题 |
|---|---|
| phase3_wisdom_rag vs baseline | 最终系统过 baseline 多少？|
| phase3_wisdom_rag vs phase2_triggers | RAG 相比单纯 triggers 加值多少？|
| phase3_wisdom_rag vs phase1_canonical_plus_p2 | 原典 RAG vs 原典问句注入 哪个更有效？|

### 实施成本估算（**远高于 Phase 1 改造**）

- Corpus 准备（文本收集 + 切分 + 清洗）: **约 1-2 天人工**
- Embedding 索引: 1-2 小时算力（用现成 cache 的 sentence-transformer）
- 每题 3 次 LLM 调用（分析 + retrieve prompt 生成 + execute）: 100 × 3 = 300
- 3 个 judges × 100 = 300
- **总计 ~600 calls + 1-2 天工程**

---

## 优先级建议

用户选择先做 Phase 1 大改（带 Phase 2，不上 Phase 3）。Phase 3 Wisdom-RAG 暂缓。

### 顺序

1. **现在做：** Phase 1 大改（canonical orientations）
2. **然后：** 在 Phase 1 canonical base 上 stack Phase 2 triggers
3. **评估：** vs baseline, vs phase2_triggers, vs orient_hybrid
4. **决策：** 如果有效继续；如果无效回到 phase2_triggers 为最终版本
5. **Phase 3 Wisdom-RAG 作为独立 future work** 单独立项

### 决策规则保持

- +5pp vs phase2_triggers = KEEP canonical
- ±3pp = INCONCLUSIVE，不替换
- <-3pp = REVERT
