# 阶段零：哲学方法论知识库——完整开发文档 (v2)

## 0. 文档概述

### 0.1 本阶段在整体架构中的位置

本阶段是整个"递归假设-验证智能体"系统的地基。后续所有阶段——调度器（阶段一）、经验反馈（阶段二）、形式化对齐（阶段三）、新假设生成（阶段四）——都依赖本阶段输出的知识库作为已知可靠组件。

**关键设计原则：知识库从第一天起就必须为"持续演化"而设计。** 阶段二的经验反馈机制将持续修改策略的适用条件、调整置信度、添加新规则。如果阶段零输出的是一个冻结的静态结构，阶段二就必须重写整个 schema。因此本文档的所有数据结构设计都显式考虑了可演化性。

### 0.2 本阶段目标

从人类哲学、科学方法论、认知科学、数学问题解决文献中，提取并结构化 15-20 条核心元策略（meta-strategies），构建一个机器可读、人类可审计、**可持续演化**的方法论知识库。

### 0.3 交付物

1. 一个 JSON 格式的方法论知识库文件（`philosophical_kb.json`）
2. 一个独立的经验日志结构（`experience_log/`），供阶段二写入原始经验记录
3. 一个结构化的变更日志系统（`change_history/`），支持自动化更新和回滚
4. 一个标注指南文档，用于人类评审者标注"问题-策略"匹配关系
5. 一个包含 100-150 道多领域问题的标注数据集（`benchmark_problems.json`）
6. 一份人类标注一致性分析报告
7. 一份可更新性测试报告（证明知识库的更新流程能正常工作）
8. 一篇可投稿的数据集论文（可选，视质量而定）

### 0.4 时间预算

总计 2-3 个月。文献调研与策略提取：3-4 周。知识库结构设计与填充：2-3 周。标注任务设计与执行：3-4 周。分析与迭代：2 周。

---

## 1. 策略来源与提取

### 1.1 一级文献来源

以下文献构成策略提取的核心语料。每一条最终进入知识库的策略都必须能追溯到至少一个一级来源。

**科学方法论：**
- George Polya, "How to Solve It" (1945) —— 数学问题解决的启发式方法词典
- Karl Popper, "The Logic of Scientific Discovery" (1959) —— 证伪主义、可检验性
- Thomas Kuhn, "The Structure of Scientific Revolutions" (1962) —— 范式转换
- Imre Lakatos, "Proofs and Refutations" (1976) —— 数学发现的辩证法
- Paul Feyerabend, "Against Method" (1975) —— 方法论多元主义

**认知科学与决策理论：**
- Daniel Kahneman, "Thinking, Fast and Slow" (2011) —— 双系统理论、认知偏误
- Gerd Gigerenzer, "Simple Heuristics That Make Us Smart" (1999) —— 生态理性、快速节俭启发式
- Herbert Simon, "The Sciences of the Artificial" (1969) —— 有限理性、满意化

**哲学逻辑：**
- Aristotle, "Organon" —— 演绎法、三段论、归类
- Francis Bacon, "Novum Organum" (1620) —— 归纳法、排除法
- John Stuart Mill, "A System of Logic" (1843) —— Mill 五法（求同法、求异法、共变法等）

**工程方法论：**
- W. Edwards Deming, "Out of the Crisis" (1986) —— PDCA 循环、系统思维
- Gerald Weinberg, "The Psychology of Computer Programming" (1971) —— 无我编程、增量调试

**数学与计算机科学：**
- Donald Knuth, "The Art of Computer Programming" —— 算法设计策略
- Cormen et al., "Introduction to Algorithms" —— 分治法、贪心法、动态规划的形式化

### 1.2 策略提取规则

一条策略要进入知识库，必须满足以下全部条件：

1. **跨领域性：** 该策略在至少 3 个不同领域中有成功应用案例。仅在单一领域有效的技巧不算元策略。
2. **可操作性：** 该策略可以被表述为一系列具体的操作步骤，而不仅仅是一个模糊的原则。"要有创造性"不是可操作的；"尝试将当前问题与一个已解决的相似问题进行类比"是可操作的。
3. **可区分性：** 该策略和知识库中已有的其他策略有明确的区别——存在某些问题场景，使用此策略优于使用其他任何已有策略。
4. **可验证性：** 该策略的效果可以通过观察执行结果来评估——要么问题被解决了，要么至少问题的规模/复杂度降低了。

### 1.3 对比式适用条件精炼（参考 ExpeL）

**设计动机：** ExpeL 的 insight 提取 prompt（Figure 2）有一个关键设计——同时展示成功和失败的轨迹对，要求 LLM 通过对比来提取规则。这种对比式提取比单独分析成功案例更能揭示策略奏效的关键条件。

在从文献中初步提取策略并填写 `applicability_conditions` 后，对每条策略执行以下对比精炼流程：

**Step 1：** 为每条策略准备一个成功案例和一个**类似场景下的失败案例**。两个案例的问题表面特征应尽量相似（同领域、同复杂度），但策略在一个上成功、另一个上失败。

**Step 2：** 用 LLM 做对比分析：

```
你是一个方法论研究专家。

## 策略：{strategy_name}
{strategy_description}

## 成功案例
场景：{success_case_description}
结果：策略成功解决了问题
成功原因分析：{why_succeeded}

## 失败案例（类似场景）
场景：{failure_case_description}
结果：策略未能解决问题
失败原因分析：{why_failed}

## 分析任务
对比这两个案例，回答：
1. 成功案例具备而失败案例不具备的关键条件是什么？
2. 这个条件在知识库当前的 applicability_conditions 中是否已经被记录？
3. 如果未记录，请用一句话描述这个新条件，并说明它应该放在 favorable 还是 unfavorable 中。
```

**Step 3：** 用对比分析结果精炼 `applicability_conditions`——添加遗漏的条件，修改描述不够精确的条件。

**预期效果：** 初始的适用条件来自文献的定性描述（如"因素之间耦合度较低"），经过对比精炼后变成更具操作性的条件（如"组件之间不共享可变状态，且接口调用不产生副作用"）。

### 1.4 初始策略清单（待验证）

以下是基于文献调研的初始候选策略清单。每条策略都需要在后续步骤中被充实和验证。

| 编号 | 策略名称 | 一句话描述 | 主要来源 |
|------|---------|-----------|---------|
| S01 | 控制变量法 | 固定其他条件，每次只改变一个因素 | Mill, Bacon |
| S02 | 分而治之 | 将复杂问题分解为可独立解决的子问题 | Polya, Knuth |
| S03 | 类比推理 | 将当前问题映射到一个已解决的相似问题 | Polya, Aristotle |
| S04 | 反证法/归谬法 | 假设结论不成立，推导出矛盾 | Aristotle, Polya |
| S05 | 奥卡姆剃刀 | 在同等解释力下选择更简单的假设 | William of Ockham |
| S06 | 先特殊后一般 | 先解决最简单的特殊情况，再逐步推广 | Polya, Lakatos |
| S07 | 反向推理 | 从目标状态出发，反推需要什么前提条件 | Polya |
| S08 | 试错法/猜测-检验 | 提出一个猜测，测试，根据结果修正 | Popper, Polya |
| S09 | 降维/简化 | 去掉不影响核心的复杂因素，先解决简化版 | Polya, Simon |
| S10 | 对称性利用 | 识别问题中的对称结构以减少搜索空间 | Polya, 物理学传统 |
| S11 | 满意化 | 不追求最优解，找到第一个足够好的解即停止 | Simon |
| S12 | 贝叶斯更新 | 用新证据持续更新对假设的信念强度 | Bayes, Kahneman |
| S13 | 证伪优先 | 优先寻找能推翻当前假设的证据，而非确认它 | Popper |
| S14 | 边界条件分析 | 检查极端情况和边界值来测试假设的稳健性 | 工程传统, Polya |
| S15 | 增量构建 | 从最小可工作版本开始，逐步添加功能 | 敏捷开发, Deming |
| S16 | 求同法 | 找出所有成功案例的共同因素 | Mill |
| S17 | 求异法 | 找出成功和失败案例之间的关键差异 | Mill |
| S18 | 抽象化/泛化 | 去掉具体细节，提取问题的抽象结构 | Polya, 范畴论传统 |
| S19 | 约束松弛 | 暂时放宽某些约束条件，看问题是否变得可解 | 运筹学, Polya |
| S20 | 对偶/互补视角 | 从问题的对立面或互补角度重新审视 | 物理学传统, 哲学辩证法 |

---

## 2. 知识库数据结构

### 2.1 单条策略的完整 Schema（可演化设计）

**关键变化：** 每条适用条件不再是简单字符串，而是带元数据的对象。这使得阶段二的经验反馈可以独立地新增条件、调整置信度、甚至将条件从 favorable 移到 unfavorable。

```json
{
  "id": "S01",
  "name": {
    "zh": "控制变量法",
    "en": "Controlled Variable Method"
  },
  "aliases": ["单因素实验法", "隔离变量法", "ceteris paribus"],
  "category": "empirical_testing",
  "source_references": [
    {
      "author": "John Stuart Mill",
      "work": "A System of Logic",
      "year": 1843,
      "chapter": "Book III, Chapter VIII - Of the Four Methods of Experimental Inquiry",
      "relevance": "Mill 的求异法（Method of Difference）的操作化形式"
    },
    {
      "author": "Francis Bacon",
      "work": "Novum Organum",
      "year": 1620,
      "relevance": "排除法（Tables of Exclusion）的前身"
    }
  ],

  "description": {
    "one_sentence": "在测试某个因素的影响时，保持所有其他因素不变，每次只改变一个因素。",
    "detailed": "当面对一个结果受多个因素影响的系统时，逐一改变单个因素并观察结果变化，从而建立因果关系。核心假设是因素之间的交互效应相对于主效应而言较小，或者可以在后续步骤中单独处理。",
    "intuitive_analogy": "就像调音师调钢琴——每次只调一根弦，听效果，而不是同时拧所有的钉子。"
  },

  "operational_steps": [
    "1. 列出所有可能影响结果的因素",
    "2. 选定一个基准配置，确保在该配置下系统可以正常运行",
    "3. 选择一个因素进行改变，保持其余因素在基准值",
    "4. 观察并记录结果的变化",
    "5. 将该因素恢复到基准值",
    "6. 对下一个因素重复步骤 3-5",
    "7. 根据所有单因素实验的结果，形成对系统行为的初步理解",
    "8. 如果怀疑存在交互效应，设计针对性的多因素实验验证"
  ],

  "applicability_conditions": {
    "favorable": [
      {
        "condition_id": "S01_F_001",
        "condition": "系统的各组件可以被独立修改",
        "source": "literature",
        "source_ref": "Mill 1843",
        "confidence": 0.95,
        "supporting_cases": ["case_literature_001"],
        "contradicting_cases": [],
        "last_updated": "2025-XX-XX",
        "version": 1,
        "status": "active",
        "locked": false
      },
      {
        "condition_id": "S01_F_002",
        "condition": "存在一个已知可工作的基准配置",
        "source": "literature",
        "source_ref": "Deming, 敏捷开发实践",
        "confidence": 0.90,
        "supporting_cases": [],
        "contradicting_cases": [],
        "last_updated": "2025-XX-XX",
        "version": 1,
        "status": "active",
        "locked": false
      },
      {
        "condition_id": "S01_F_003",
        "condition": "因素之间的耦合度较低",
        "source": "literature",
        "source_ref": "Simon 1969",
        "confidence": 0.85,
        "supporting_cases": [],
        "contradicting_cases": [],
        "last_updated": "2025-XX-XX",
        "version": 1,
        "status": "active",
        "locked": false,
        "notes_for_future_refinement": "'耦合度较低' 目前是定性描述，阶段二可能需要给出定量阈值"
      }
    ],
    "unfavorable": [
      {
        "condition_id": "S01_U_001",
        "condition": "因素之间存在强耦合——改变一个因素必然导致另一个因素也变化",
        "source": "literature",
        "source_ref": "分布式系统文献",
        "confidence": 0.90,
        "supporting_cases": ["case_literature_002"],
        "contradicting_cases": [],
        "last_updated": "2025-XX-XX",
        "version": 1,
        "status": "active",
        "locked": false
      }
    ],
    "failure_modes": [
      {
        "mode_id": "S01_FM_001",
        "description": "遗漏了关键因素（未列入因素清单）",
        "source": "literature",
        "confidence": 0.85,
        "observed_cases": []
      }
    ]
  },

  "historical_cases": {
    "successes": [
      {
        "case_id": "S01_SUC_001",
        "domain": "医学",
        "case": "James Lind 的坏血病实验 (1747)",
        "description": "Lind 将12名坏血病水手分为6组，每组给予不同的膳食补充，其他条件相同。发现柑橘组康复最快。",
        "why_this_strategy_worked": "船上环境相对可控，其他生活条件（食物基础、劳动量等）基本一致，使得膳食补充成为唯一变量。",
        "demonstrates_conditions": ["S01_F_001", "S01_F_002"]
      },
      {
        "case_id": "S01_SUC_002",
        "domain": "软件工程",
        "case": "二分法调试（Git bisect）",
        "description": "当一个 bug 出现在大量代码提交之后，通过二分法逐步缩小引入 bug 的提交范围。",
        "why_this_strategy_worked": "每个 commit 是一个离散的变化单位，可以独立地检出和测试。",
        "demonstrates_conditions": ["S01_F_001"]
      },
      {
        "case_id": "S01_SUC_003",
        "domain": "物理学",
        "case": "密立根油滴实验 (1909)",
        "description": "通过精确控制电场和重力场，逐一测量单个油滴的电荷量，发现电荷量子化。",
        "why_this_strategy_worked": "实验设计使得每个油滴可以被独立观察，电场强度是唯一主动变化的参数。",
        "demonstrates_conditions": ["S01_F_001", "S01_F_003"]
      }
    ],
    "failures": [
      {
        "case_id": "S01_FAIL_001",
        "domain": "社会科学",
        "case": "早期营养学研究中的混淆变量问题",
        "description": "试图控制单一饮食因素（如脂肪摄入）来研究健康影响，但忽略了总热量摄入、运动量、遗传因素等的交互。",
        "why_this_strategy_failed": "人类生活方式的各因素高度耦合，真正的'控制'几乎不可能实现。",
        "demonstrates_conditions": ["S01_U_001"]
      },
      {
        "case_id": "S01_FAIL_002",
        "domain": "软件工程",
        "case": "分布式系统中的逐一组件测试",
        "description": "逐一测试微服务的各个组件均正常，但组合在一起时出现死锁。",
        "why_this_strategy_failed": "组件之间存在竞态条件（race condition），这种交互效应只在组件同时运行时才会显现。",
        "demonstrates_conditions": ["S01_U_001"]
      }
    ]
  },

  "relationships_to_other_strategies": [
    {
      "related_strategy": "S02",
      "relationship_type": "complementary",
      "description": "分而治之将问题拆分为子问题，控制变量法确保在测试每个子问题时其他部分不变。二者常组合使用。"
    },
    {
      "related_strategy": "S15",
      "relationship_type": "prerequisite",
      "description": "增量构建要求先有一个可工作的基准版本，然后逐一添加新模块——这本质上就是控制变量法在系统构建中的应用。"
    },
    {
      "related_strategy": "S06",
      "relationship_type": "complementary",
      "description": "先特殊后一般可以为控制变量法提供初始的基准配置——先在最简单的情况下让系统工作。"
    }
  ],

  "knowledge_triples": [
    {"subject": "测试因素", "relation": "逐一改变", "object": "观察结果变化"},
    {"subject": "其余因素", "relation": "保持不变", "object": "基准配置"},
    {"subject": "因素耦合度", "relation": "要求", "object": "低"},
    {"subject": "基准配置", "relation": "前提", "object": "已知可工作"}
  ],

  "formalization_hints": {
    "mathematical_structure": "因素空间 F = F_1 × F_2 × ... × F_n 中，控制变量法对应于沿坐标轴方向的逐一搜索（coordinate descent）。",
    "category_theory_analogue": "可以看作在因素范畴中，固定其他对象，只沿一个态射方向做变换。",
    "information_geometry_analogue": "在参数流形上，控制变量法对应于沿坐标测地线方向的移动，而非沿任意方向的测地线。",
    "connection_to_known_algorithms": ["坐标下降法 (Coordinate Descent)", "逐步回归 (Stepwise Regression)", "消融实验 (Ablation Study)"]
  },

  "metadata": {
    "version": "1.0",
    "created": "2025-XX-XX",
    "last_updated": "2025-XX-XX",
    "update_history_ref": "change_history/S01.jsonl",
    "confidence": "high",
    "completeness": "medium",
    "needs_review": ["unfavorable conditions 列表可能不完整", "failure_modes 需要更多案例"],
    "total_experience_records": 0,
    "successful_applications": 0,
    "failed_applications": 0,
    "effectiveness_score": 0.5
  }
}
```

### 2.2 策略类别体系

所有策略归入以下六个大类（一条策略可以属于多个类别）：

| 类别 ID | 类别名称 | 描述 | 典型策略 |
|--------|---------|------|---------|
| CAT_A | 问题分解与简化 | 将问题变小或变简单 | 分而治之、降维简化、先特殊后一般 |
| CAT_B | 搜索与探索 | 在解空间中寻找答案 | 试错法、反向推理、约束松弛 |
| CAT_C | 推理与证明 | 逻辑推导和论证 | 反证法、类比推理、演绎推理 |
| CAT_D | 实证与检验 | 通过实验或观察验证假设 | 控制变量法、证伪优先、边界条件分析 |
| CAT_E | 评估与选择 | 在多个候选方案中做选择 | 奥卡姆剃刀、满意化、贝叶斯更新 |
| CAT_F | 构建与迭代 | 逐步构造解 | 增量构建、抽象化泛化、对偶互补 |

### 2.3 策略关系图的结构

策略之间存在以下四种关系类型：

- **prerequisite**（前置关系）：策略 A 的执行是策略 B 的前提条件。例如"先特殊后一般"是"控制变量法"的前置——你需要先有一个可工作的简单版本作为基准。
- **complementary**（互补关系）：策略 A 和策略 B 经常组合使用，效果好于单独使用任一个。例如"分而治之"和"控制变量法"。
- **alternative**（替代关系）：策略 A 和策略 B 解决同一类问题，但方法不同。例如"试错法"和"反证法"都可以用来验证假设，但路径不同。
- **subsumption**（包含关系）：策略 A 是策略 B 的特殊情况。例如"增量构建"可以被看作"控制变量法"在系统构建场景下的特例。

### 2.4 知识三元组（Knowledge Triples）

**设计动机（参考 EvolveR）：** EvolveR 的每条 principle 由自然语言描述 + 结构化知识三元组两个组件组成。三元组提供了比自然语言更精确的检索锚点，使阶段一的调度器在匹配策略时能利用结构化的知识进行精确匹配，而不仅仅依赖文本嵌入的语义相似度。

每条策略的 `knowledge_triples` 字段存储 3-6 条 (subject, relation, object) 三元组，捕捉该策略的核心逻辑骨架。三元组的编写原则：

1. **因果性：** 至少一条三元组描述"为什么这个策略有效"的因果链
2. **前提性：** 至少一条三元组描述策略生效的必要前提
3. **操作性：** 至少一条三元组描述策略的核心操作

三元组在阶段一中的使用方式：调度器在匹配策略时，先将问题描述提取为三元组，再与各策略的 `knowledge_triples` 做结构化匹配（三元组的 subject/object 嵌入相似度），作为 `ProblemFeatures` 的补充特征。

### 2.5 动态评分机制

**设计动机（参考 EvolveR）：** EvolveR 用 $s(p) = \frac{c_{\text{succ}}(p) + 1}{c_{\text{use}}(p) + 2}$（Laplace 平滑的成功率）持续跟踪每条规则的实际效用。本知识库需要在阶段零就定义好评分机制，为阶段二的蒸馏器和整合器提供统一的"策略好坏"标准。

**评分公式：**

每条策略的 `metadata.effectiveness_score` 按以下公式计算：

$$\text{effectiveness\_score}(S) = \frac{\text{successful\_applications}(S) + 1}{\text{total\_experience\_records}(S) + 2}$$

- 初始值为 0.5（无数据时的无信息先验）
- 随经验积累自动更新
- 值域为 (0, 1)

**清理阈值：**

- `prune_threshold = 0.25`：低于此值的策略在阶段二的睡眠整合中被标记为 `status: "under_review"`
- `promotion_threshold = 0.70`：高于此值且 `total_experience_records ≥ 30` 的策略可以将 `confidence` 从 `"medium"` 升级为 `"high"`

**注意：** 评分仅反映策略在被选中使用时的成功率，不反映策略被选中的频率。一条很少被使用但每次都成功的策略可能得分很高——这是合理的。

### 2.6 策略组合模式（Strategy Compositions）

**设计动机（参考 Voyager）：** Voyager 的技能库展示了技能的可组合性——复杂技能通过组合简单技能来构建。方法论策略同样存在组合模式：很多复杂问题的最优解法不是单条策略，而是策略序列（如"先用 S06 简化问题，再用 S01 逐一排查"）。

阶段零的 `relationships_to_other_strategies` 只表达静态关系，不表达动态的组合编排。需要单独的组合模式存储。

**组合模式的 Schema：**

```json
{
  "composition_id": "COMP_001",
  "name": {
    "zh": "先简化再排查",
    "en": "Simplify-then-Isolate"
  },
  "sequence": ["S06", "S01"],
  "transition_condition": "S06 成功将问题简化为可控规模后，切换到 S01 逐一检查简化后的各因素",
  "applicable_when": "问题整体复杂度高（component_count > 5）但具有可简化的结构（decomposability > 0.4）",
  "source": "literature",
  "source_ref": "Polya, How to Solve It: 'If you cannot solve the proposed problem, try to solve first some related problem'",
  "historical_cases": [
    {
      "case_id": "COMP_001_CASE_001",
      "domain": "软件工程",
      "description": "调试大型系统时，先用最小可复现用例简化问题，再逐一排查简化场景中的组件"
    }
  ],
  "metadata": {
    "effectiveness_score": 0.5,
    "total_uses": 0,
    "successful_uses": 0
  }
}
```

组合模式存储在 `kb/compositions/` 目录下。阶段一的调度器动作空间 = 单策略列表 ∪ 组合模式列表。

**初始组合模式（待验证）：**

| 编号 | 名称 | 序列 | 适用场景 |
|------|------|------|---------|
| COMP_001 | 先简化再排查 | S06 → S01 | 高复杂度但可简化的问题 |
| COMP_002 | 分解后增量 | S02 → S15 | 可分解为独立子问题且需要逐步构建 |
| COMP_003 | 类比后验证 | S03 → S13 | 找到类似问题的解后用证伪法验证是否真正适用 |
| COMP_004 | 边界分析后反证 | S14 → S04 | 先找到边界条件再用反证法证明一般情况 |
| COMP_005 | 估计后精炼 | S12 → S01 | 先用贝叶斯估计缩小搜索范围，再逐一精确排查 |

### 2.7 经验记录 Schema（Episodic Memory）

知识库本身存储的是精炼后的规则（semantic memory）。阶段二会不断写入原始经验（episodic memory），这些经验经过蒸馏后才会影响知识库。两者必须分开存储。

每条经验记录的格式：

```json
{
  "execution_id": "exec_20260301_142301_task42",
  "timestamp": "2026-03-01T14:23:01Z",
  "task": {
    "task_id": "task_42",
    "description": "调试一个在某些输入下崩溃的分布式 Web 服务",
    "domain": "software_engineering",
    "difficulty": "medium",
    "complexity_features": {
      "component_count": 7,
      "coupling_estimate": "medium",
      "has_baseline": true,
      "randomness_level": "low"
    }
  },
  "strategy_selection": {
    "selected_strategy": "S01",
    "selection_reason": "任务特征匹配 S01_F_001 和 S01_F_002",
    "alternatives_considered": ["S17", "S02"],
    "selector_confidence": 0.75
  },
  "execution_trajectory": {
    "steps": [
      {"step_num": 1, "action": "列出所有组件", "result": "7 个组件"},
      {"step_num": 2, "action": "建立基准", "result": "简化输入下全部正常"}
    ],
    "total_steps": 15,
    "wall_clock_seconds": 1847
  },
  "outcome": {
    "success": false,
    "partial_success": true,
    "failure_reason": "发现组件 C 和组件 F 之间存在隐藏的共享状态，导致逐一测试无法复现 bug",
    "root_cause_type": "coupling_underestimated"
  },
  "attribution": {
    "matched_conditions": ["S01_F_001", "S01_F_002"],
    "violated_conditions": ["S01_F_003"],
    "newly_discovered_condition_candidates": [
      {
        "condition_text": "即使组件接口明确，也可能存在通过共享状态的隐性耦合",
        "evidence_strength": "strong",
        "suggested_placement": "S01_U_NEW"
      }
    ]
  },
  "metadata": {
    "evaluator_version": "v1.0",
    "human_reviewed": false,
    "distilled_into_kb": false,
    "distillation_ref": null
  }
}
```

这个 schema 的关键价值在于 `attribution` 字段——它是经验和知识库之间的桥梁。每条经验都显式地记录了哪些适用条件被满足、哪些被违反、是否发现了新的条件候选。阶段二的经验蒸馏器读取这些 attribution 字段来生成对知识库的更新建议。

### 2.5 变更历史 Schema

每条策略有一个对应的变更历史文件 `change_history/S01.jsonl`（JSON Lines 格式，便于追加）。每次对该策略的修改都是一个独立的 JSON 行：

```
{"change_id": "chg_001", "timestamp": "2025-XX-XX", "type": "initial_creation", "author": "human", "changes": "初始版本", "evidence_refs": ["Mill 1843"], "previous_version": null, "new_version": 1}
{"change_id": "chg_042", "timestamp": "2026-03-15T08:30:00Z", "type": "condition_added", "author": "phase2_auto", "changes": {"added": {"condition_id": "S01_U_002", "condition": "存在隐性共享状态的耦合", "confidence": 0.80}}, "evidence_refs": ["exec_20260301_142301_task42", "exec_20260303_091122_task57"], "previous_version": 5, "new_version": 6}
{"change_id": "chg_043", "timestamp": "2026-03-18T10:15:00Z", "type": "confidence_adjusted", "author": "phase2_auto", "changes": {"condition_id": "S01_F_003", "old_confidence": 0.85, "new_confidence": 0.70}, "evidence_refs": ["exec_20260315_..."], "previous_version": 6, "new_version": 7}
```

变更类型（type）的枚举值：
- `initial_creation`：初始创建
- `condition_added`：新增适用条件
- `condition_removed`：移除适用条件
- `condition_modified`：修改已有条件（如文本表述）
- `confidence_adjusted`：调整某条件的置信度
- `condition_moved`：条件在 favorable/unfavorable 之间迁移
- `case_added`：新增历史案例
- `relationship_added`：新增与其他策略的关系
- `relationship_modified`：修改已有关系
- `rollback`：回滚到某个历史版本

### 2.6 回滚机制

当阶段二的自动更新被发现引入了错误规则时，需要能快速回到已验证的稳定版本。回滚机制包括：

- **版本标签（tags）：** 知识库的重要稳定版本用 Git tag 标记，如 `v1.0-stable`、`v1.5-validated`。
- **自动检测可疑更新：** 如果某次自动更新后，该策略在后续任务中的成功率显著下降（比如下降超过 20%），触发警告。
- **回滚指令：** 一个命令行工具 `kb_rollback.py`，可以将某条策略回滚到指定版本，同时在变更历史中记录一条 `rollback` 类型的变更。
- **人工覆盖：** 某些关键适用条件可以被标记为 `locked: true`，防止被自动更新修改。

---

## 3. 标注任务设计

### 3.1 标注目标

验证知识库的质量，具体包括：
1. 策略定义是否足够清晰——不同标注者对同一策略的理解是否一致
2. 策略-问题匹配是否合理——面对同一问题，不同标注者是否倾向于选择相同的策略
3. 适用条件是否准确——标注者选择某策略的理由是否和知识库中描述的适用条件一致

### 3.2 标注者招募

招募 8-12 名标注者，要求覆盖以下背景（每个背景至少 2 人）：
- 计算机科学/软件工程研究者
- 数学/物理学研究者
- 社会科学/经济学研究者
- 工程实践者（有工业经验的工程师）
- 哲学/认知科学研究者（如果可得）

背景多样性是关键：如果只有 CS 背景的人标注，可能会系统性地偏好某些策略（如分而治之），而忽略其他策略在非 CS 领域的适用性。

### 3.3 标注任务格式

每个标注者需要完成以下任务：

**任务 A：策略-问题匹配（主任务）**

**防范 experience-following 偏差（参考 Memory Management 论文）：** 该论文发现 LLM agent 在输入与记忆高度相似时，会直接复制记忆中的输出模式。人类标注者也可能被策略描述中的关键词"锚定"，导致偏向表面匹配而非深层匹配。为此，标注者被随机分为两组：

- **A 组（开放标注，占标注者的 40%）：** 先看问题描述，先用自己的语言写下解决思路（50-100 字），**再**看策略列表并匹配。这确保标注者的判断不受策略描述措辞的锚定影响。
- **B 组（直接标注，占标注者的 60%）：** 直接看到问题描述和策略列表，从中选择。这是更高效的标注方式。

如果 A 组和 B 组在同一问题上的策略选择一致性（Cohen's Kappa）低于 0.5，说明 experience-following 偏差显著存在，需要增加 A 组比例并根据 A 组结果调整策略描述的措辞。

标注者需要完成以下任务：

呈现一个问题描述，要求标注者：
1. （仅 A 组）用自己的语言描述解决思路
2. 从 20 条策略中选择最适合的 1-3 条（排序）
3. 对每条选择写一句话的理由
4. 标注置信度（高/中/低）

示例问题：

```
问题：你正在开发一个机器人控制系统。系统由感知模块、规划模块和执行模块
三个部分组成。你已经分别测试了每个模块，单独运行时都工作正常，但三个模块
连接在一起后，机器人会在某些特定动作序列下突然停止响应。你需要找到并修复
这个问题。
```

期望标注：S01（控制变量法）—— 逐一连接模块，找到引起问题的组合；S17（求异法）—— 对比正常和异常动作序列的差异；S14（边界条件分析）—— 检查特定动作序列是否触发了边界情况。

**任务 B：适用条件验证（辅助任务）**

呈现一条策略和其适用条件描述，加上一个具体场景，要求标注者判断：
1. 该策略在此场景下是否适用（是/否/部分适用）
2. 知识库中列出的适用条件是否和标注者的判断一致
3. 是否有知识库未列出的重要适用/不适用条件

### 3.4 问题集设计

构建 100-150 道问题，覆盖以下 6 个领域（每领域 15-25 道）：

| 领域 | 问题类型示例 |
|------|------------|
| 软件工程 | 调试、系统设计、性能优化、架构选择 |
| 数学/逻辑 | 证明策略选择、问题建模、估算 |
| 科学实验 | 实验设计、假说检验、结果解释 |
| 商业决策 | 市场进入策略、资源分配、风险评估 |
| 日常问题解决 | 计划制定、冲突解决、学习策略 |
| 工程设计 | 原型迭代、故障排除、需求分析 |

每道问题需包含：
- 问题描述（150-300 字）
- 领域标签
- 难度等级（简单/中等/复杂）
- 参考答案（最优策略 + 理由，由出题者提供，不给标注者看）

### 3.5 难度控制

三种难度的定义：

- **简单：** 最优策略明显，几乎只有一个合理选择。例："你有一个函数返回错误结果，你知道它调用了 5 个子函数。如何定位 bug？" → 显然是控制变量法/二分法。
- **中等：** 有 2-3 个合理的策略选择，最优策略取决于具体权衡。例："你要在 3 个月内完成一个从未做过的复杂项目。" → 增量构建 vs 先特殊后一般 vs 类比推理（参考类似项目）。
- **复杂：** 需要策略组合，或者存在策略冲突的情况。例："你的实验数据同时支持两个互相矛盾的假设。" → 证伪优先 + 边界条件分析 + 贝叶斯更新的组合。

### 3.6 一致性分析方法

计算以下指标：

1. **Fleiss' Kappa（多标注者一致性）：** 对"最优策略"选择的一致性。目标值 > 0.6（substantial agreement）。
2. **前三名命中率：** 标注者选择的前 3 策略和参考答案的重叠比例。目标值 > 0.7。
3. **按策略的混淆矩阵：** 哪些策略对最容易被混淆？这将直接指导知识库中策略定义的改进——被频繁混淆的两条策略要么需要更清晰的区分，要么应该合并。
4. **按领域的一致性差异：** 某些领域的一致性是否系统性地低于其他领域？如果是，可能说明知识库中某些策略的适用条件描述对该领域不够具体。

---

## 4. 迭代与质量控制

### 4.1 迭代流程

整个阶段零采用两轮迭代：

**第一轮（探索轮）：**
1. 完成初始知识库 v0.1（20 条策略的基本信息）
2. 用 30 道问题对 3-4 名标注者做小规模预标注
3. 分析预标注结果，识别：
   - 定义不清的策略（一致性低）
   - 缺失的策略（标注者写出了不在列表中的策略）
   - 冗余的策略（两条策略总是被一起选择，可能应合并）
4. 修订知识库为 v0.2

**第二轮（验证轮）：**
1. 用完整的 100-150 道问题对全部 8-12 名标注者做正式标注
2. 计算一致性指标
3. 对一致性低于阈值的策略和问题做定向修改
4. 输出最终知识库 v1.0

### 4.2 策略合并/拆分规则

- 如果两条策略在 70% 以上的问题上被同时选择 → 考虑合并
- 如果一条策略被标注者系统性地忽略（在不到 5% 的问题上被选择）→ 考虑删除或重新定义
- 如果标注者频繁写出一条不在列表中的策略 → 考虑添加

### 4.3 可更新性测试

在阶段零结束前，必须验证知识库的更新流程能正常工作——这是交付给阶段二的最关键保证。

**测试步骤：**

1. 手工构造 10-20 条模拟经验记录（`experience_log/executions/` 下的 JSON 文件），覆盖以下场景：
   - 一条成功经验，完全符合某策略的已知 favorable 条件 → 预期：更新 `metadata.successful_applications` 计数
   - 一条失败经验，违反了某策略的已知 unfavorable 条件 → 预期：该条件的 confidence 上升
   - 一条失败经验，发现了知识库中未列出的新条件 → 预期：生成一条 `newly_discovered_condition_candidates`，进入人工审核队列
   - 一条成功经验，但在知识库标记为 unfavorable 的条件下成功 → 预期：降低该 unfavorable 条件的 confidence
   - 多条一致性经验共同支持同一个新条件 → 预期：新条件被自动加入 favorable/unfavorable 列表
   - 后续经验推翻之前自动加入的条件 → 预期：触发回滚

2. 编写一个 mock 版的阶段二更新流程（不需要真正的 LLM 调用，用规则函数模拟），跑一遍所有模拟经验。

3. 验证：
   - 所有更新都正确写入了变更历史日志
   - 知识库的 JSON Schema 在更新后仍然有效
   - 经验记录和知识库规则的关联指针（`supporting_cases`, `contradicting_cases`, `evidence_refs`）正确建立
   - 回滚命令能将任意策略恢复到指定历史版本
   - 被锁定（`locked: true`）的条件不会被自动修改

**通过标准：** 所有 6 种测试场景都按预期行为运行，变更历史完整可读，回滚机制可用。

这个测试的价值在于：如果阶段零交付时这个测试跑不通，说明 schema 设计有缺陷。现在修改的成本是几天；阶段二再发现问题就要重写几周的代码。

### 4.4 知识库版本管理

Git 管理整体仓库的版本变更，每次人工编辑都有 commit。但自动化更新产生的变更过于频繁，不应该每次都创建 Git commit。因此采用双层版本机制：

- **Git 层：** 记录人工修改和重要版本的发布（tag：`v1.0-stable`、`v1.5-validated`）
- **变更历史层：** 记录所有细粒度变更，包括自动更新，存储在 `change_history/*.jsonl` 中。这一层比 Git 更细致，包含了每次变更对应的证据指针

定期（每月或每重大版本）将变更历史层的内容整合进 Git commit，产生一个人类可读的变更摘要。

---

## 5. 技术实现

### 5.1 知识库文件结构

```
philosophical_kb/
├── README.md                      # 项目说明
├── kb/                            # Semantic Memory：精炼后的规则
│   ├── strategies/
│   │   ├── S01_controlled_variable.json
│   │   ├── S02_divide_and_conquer.json
│   │   ├── ...
│   │   └── S20_dual_perspective.json
│   ├── categories.json            # 策略类别定义
│   ├── relationships.json         # 策略间关系图
│   └── schema.json                # JSON Schema 定义
├── experience_log/                # Episodic Memory：原始经验记录
│   ├── executions/                # 每次任务执行的完整记录
│   │   ├── 2026/03/
│   │   │   ├── exec_20260301_142301_task42.json
│   │   │   └── ...
│   │   └── schema.json
│   └── distilled/                 # 从 executions 蒸馏出的规则更新候选
│       ├── pending_review/        # 等待人工审核
│       ├── approved/              # 已应用到 kb/
│       └── rejected/              # 审核不通过
├── change_history/                # 变更历史
│   ├── S01.jsonl
│   ├── S02.jsonl
│   ├── ...
│   └── global_log.jsonl           # 跨策略的全局变更
├── benchmark/
│   ├── problems/
│   │   ├── software_engineering/
│   │   ├── mathematics/
│   │   ├── science/
│   │   ├── business/
│   │   ├── daily_life/
│   │   └── engineering/
│   ├── annotation_guide.md
│   ├── annotations/
│   └── analysis/
├── scripts/
│   ├── validate_kb.py             # 知识库格式验证
│   ├── compute_agreement.py       # 标注一致性计算
│   ├── visualize_relations.py     # 策略关系图可视化
│   ├── export_for_agent.py        # 导出为阶段一调度器可用的格式
│   ├── apply_experience.py        # 阶段二将调用的更新脚本（mock 版）
│   ├── kb_rollback.py             # 回滚工具
│   └── test_updatability.py       # 可更新性测试
└── docs/
    ├── strategy_extraction_log.md
    ├── annotation_report.md
    ├── updatability_test_report.md
    └── design_decisions.md
```

### 5.2 JSON Schema 验证（更新后）

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["id", "name", "description", "operational_steps",
               "applicability_conditions", "historical_cases",
               "relationships_to_other_strategies", "metadata"],
  "properties": {
    "id": { "type": "string", "pattern": "^S[0-9]{2}$" },
    "name": {
      "type": "object",
      "required": ["zh", "en"]
    },
    "applicability_conditions": {
      "type": "object",
      "required": ["favorable", "unfavorable"],
      "properties": {
        "favorable": {
          "type": "array",
          "minItems": 2,
          "items": {
            "type": "object",
            "required": ["condition_id", "condition", "source", "confidence", "version", "status"],
            "properties": {
              "condition_id": { "type": "string" },
              "condition": { "type": "string" },
              "source": { "enum": ["literature", "experience", "human_review"] },
              "source_ref": { "type": "string" },
              "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
              "supporting_cases": { "type": "array", "items": { "type": "string" } },
              "contradicting_cases": { "type": "array", "items": { "type": "string" } },
              "last_updated": { "type": "string", "format": "date-time" },
              "version": { "type": "integer", "minimum": 1 },
              "status": { "enum": ["active", "deprecated", "under_review"] },
              "locked": { "type": "boolean", "default": false }
            }
          }
        },
        "unfavorable": {
          "type": "array",
          "minItems": 2,
          "items": { "$ref": "#/properties/applicability_conditions/properties/favorable/items" }
        },
        "failure_modes": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["mode_id", "description", "source", "confidence"]
          }
        }
      }
    },
    "historical_cases": {
      "type": "object",
      "required": ["successes", "failures"],
      "properties": {
        "successes": {
          "type": "array",
          "minItems": 2,
          "items": {
            "type": "object",
            "required": ["case_id", "domain", "case", "description", "why_this_strategy_worked"]
          }
        },
        "failures": {
          "type": "array",
          "minItems": 1
        }
      }
    },
    "metadata": {
      "type": "object",
      "required": ["version", "confidence", "update_history_ref"],
      "properties": {
        "confidence": { "enum": ["high", "medium", "low"] },
        "completeness": { "enum": ["high", "medium", "low"] },
        "update_history_ref": { "type": "string" },
        "total_experience_records": { "type": "integer", "minimum": 0 },
        "successful_applications": { "type": "integer", "minimum": 0 },
        "failed_applications": { "type": "integer", "minimum": 0 }
      }
    }
  }
}
```

### 5.3 与阶段一的接口约定

阶段一的调度器需要从知识库中获取以下信息：

1. **策略列表**：调度器的动作空间 = 知识库中所有策略的 ID 集合
2. **策略描述**：当调度器选择某策略后，将该策略的 `operational_steps` 传递给 LLM 执行器作为指令
3. **适用条件**：调度器在选择策略时，将当前问题的特征和各策略的 `applicability_conditions` 进行匹配（使用 `confidence` 字段做加权）
4. **策略关系图**：当某策略失败时，调度器可以通过 `relationships_to_other_strategies` 快速找到替代策略

导出格式（供调度器使用）：

```json
{
  "action_space": ["S01", "S02", "..."],
  "strategy_prompts": {
    "S01": "你现在需要使用控制变量法来解决这个问题。具体步骤如下：1. 列出所有可能影响结果的因素..."
  },
  "applicability_features": {
    "S01": {
      "favorable": [
        {"condition": "系统的各组件可以被独立修改", "confidence": 0.95, "condition_id": "S01_F_001"}
      ],
      "unfavorable": [
        {"condition": "因素之间存在强耦合", "confidence": 0.90, "condition_id": "S01_U_001"}
      ]
    }
  },
  "fallback_graph": {
    "S01": {
      "if_failed_try": ["S17", "S14", "S08"],
      "reason": ["求异法可作为控制变量法的补充", "检查边界条件", "退回到试错法"]
    }
  }
}
```

### 5.4 与阶段二的接口约定

阶段二需要能够：

1. **写入经验记录：** 按 `experience_log/executions/schema.json` 的格式写入每次执行的完整轨迹
2. **读取当前知识库状态：** 通过 `scripts/export_for_agent.py` 获得最新的规则快照
3. **提交更新候选：** 将蒸馏出的规则更新写入 `experience_log/distilled/pending_review/`
4. **自动应用经过置信度阈值的更新：** 置信度高于 0.85 的更新可以自动应用，低于该阈值的进入人工审核队列
5. **触发回滚：** 当检测到某策略的成功率显著下降时，调用 `kb_rollback.py`

---

## 6. 风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| 标注者一致性过低（Kappa < 0.4） | 中 | 高 | 说明策略定义不够清晰。回到迭代流程，重写定义，增加示例，做第三轮标注 |
| 20 条策略不够覆盖标注任务中的问题 | 中 | 中 | 预标注阶段就会暴露。根据标注者反馈添加新策略 |
| 跨领域标注者难以招募 | 低 | 中 | 降低标准为每个领域至少 1 人，或用领域专家顾问代替全量标注 |
| 知识库过于偏向西方哲学传统 | 低 | 低 | 在迭代中检查是否有重要的非西方方法论被遗漏（如中国哲学中的"格物致知"、印度哲学中的"anvikshiki"） |
| 阶段一发现知识库接口不够用 | 中 | 中 | 预留 `formalization_hints` 和 `metadata.needs_review` 字段作为扩展接口，阶段一可以请求添加新字段 |
| 阶段二的自动更新频率过高，变更历史不可读 | 中 | 中 | 双层版本机制（Git + 变更历史层）；定期整合变更历史为人类可读摘要 |
| 自动更新引入错误规则，污染知识库 | 高 | 高 | 置信度阈值门控（低置信度需人工审核）；重要条件可锁定；回滚机制；可更新性测试验证这些机制 |
| 经验日志数据量过大，存储/检索性能问题 | 低 | 中 | 按年/月分目录存储；定期归档旧经验为压缩格式；保留关键经验的索引，完整数据可外部存储 |

---

## 7. 完成标准（Definition of Done）

阶段零在以下所有条件同时满足时视为完成：

1. 知识库包含至少 15 条策略，每条策略的所有必填字段都已填写
2. 每条策略至少有 3 个成功案例和 1 个失败案例，跨越至少 2 个不同领域
3. 标注数据集包含至少 100 道问题，覆盖至少 5 个领域
4. 至少 6 名标注者完成了标注任务
5. 整体 Fleiss' Kappa ≥ 0.5（moderate agreement），前三名命中率 ≥ 0.65
6. 所有 JSON 文件通过 Schema 验证
7. 策略关系图已经过至少一轮人工审核
8. 导出的调度器接口文件可以被阶段一的代码直接读取
9. **可更新性测试全部 6 种场景通过，更新流程、变更历史、回滚机制均验证可用**
10. **经验日志目录结构已建立，schema 已定义，并通过至少一组 mock 数据的读写测试**
11. **change_history 系统已建立，每条策略都有对应的 JSONL 文件（即使目前只有 initial_creation 一条记录）**
